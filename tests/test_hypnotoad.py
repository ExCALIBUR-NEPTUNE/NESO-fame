from typing import Callable, cast

import numpy as np
import numpy.typing as npt
from hypnotoad import Equilibrium, Point2D  # type: ignore
from hypnotoad.cases.tokamak import TokamakEquilibrium  # type: ignore
from hypothesis import given, settings
from hypothesis.extra.numpy import array_shapes, arrays
from hypothesis.strategies import builds, floats, integers, lists, nothing, one_of
from pytest import fixture
from scipy.integrate import quad
from scipy.optimize import root_scalar
from scipy.special import ellipeinc

from neso_fame.hypnotoad import (
    equilibrium_trace,
    flux_surface_edge,
    perpendicular_edge,
    smallest_angle_between,
)
from neso_fame.mesh import CoordinateSystem, SliceCoord

from .conftest import whole_numbers


class FakeEquilibrium(Equilibrium):
    """Equilibrium with elliptical magnetic field. Magnetic flux
    follows the profile

    psi = -(R - R0)**2 / a - (Z - Z0)**2 / b

    The toroidal flux is set such that the safety factor is uniform
    along flux surfaces, making integration straightforward. It is set
    to be equal to constant q.

    Only functions used when tracing field lines in the toroidal
    direction are implemented.

    """

    def __init__(self, o_point=Point2D(1.0, 1.0), a=0.25, b=0.25, q=1.0):
        self.o_point = o_point
        self.a = a
        self.b = b
        self.q = q

    def psi_func(
        self, x: npt.ArrayLike, y: npt.ArrayLike, dx=0, dy=0, grid=True
    ) -> npt.NDArray:
        """Mimics the signature of RectBivariateSpline."""
        assert not grid
        x = np.asarray(x)
        y = np.asarray(y)
        if dx == 0 and dy == 0:
            return np.asarray(
                (x - self.o_point.R) ** 2 / self.a + (y - self.o_point.Z) ** 2 / self.b
            )
        elif dx == 1 and dy == 0:
            return np.asarray(2.0 * (x - self.o_point.R) / self.a)
        elif dx == 2 and dy == 0:
            return np.asarray(2.0 / self.a)
        elif dx == 0 and dy == 1:
            return np.asarray(2.0 * (y - self.o_point.Z) / self.b)
        elif dx == 0 and dy == 2:
            return np.asarray(2.0 / self.b)
        else:
            return np.asarray(0.0)

    def psi(self, R, Z):
        "Return the poloidal flux at the given (R,Z) location"
        return self.psi_func(R, Z, grid=False)

    def f_R(self, R: npt.ArrayLike, Z: npt.ArrayLike) -> npt.NDArray:
        """returns the R component of the vector Grad(psi)/|Grad(psi)|**2."""
        dpsidR = self.psi_func(R, Z, dx=1, grid=False)
        dpsidZ = self.psi_func(R, Z, dy=1, grid=False)
        return dpsidR / (dpsidR**2 + dpsidZ**2)

    def f_Z(self, R, Z):
        """returns the Z component of the vector Grad(psi)/|Grad(psi)|**2."""
        dpsidR = self.psi_func(R, Z, dx=1, grid=False)
        dpsidZ = self.psi_func(R, Z, dy=1, grid=False)
        return dpsidZ / (dpsidR**2 + dpsidZ**2)

    def Bp_R(self, R: float | npt.NDArray, Z: float | npt.NDArray) -> npt.NDArray:
        return np.asarray(2 * (Z - self.o_point.Z) / (R * self.b))

    def Bp_Z(
        self, R: float | npt.NDArray, Z: float | npt.NDArray
    ) -> float | npt.NDArray:
        return np.asarray(-2 * (R - self.o_point.R) / (R * self.a))

    def Bzeta(
        self, R: float | npt.NDArray, Z: float | npt.NDArray
    ) -> float | npt.NDArray:
        R_diff = R - self.o_point.R
        Z_diff = Z - self.o_point.Z
        r = np.sqrt(R_diff**2 + Z_diff**2)
        sin = Z_diff / r
        cos = R_diff / r
        return np.asarray(
            self.q * R * (-self.Bp_R(R, Z) * sin + self.Bp_Z(R, Z) * cos) / r
        )

    def to_RZ(
        self, psi: float | npt.NDArray, theta: float | npt.NDArray
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Convenience fucntion to calculate (R, Z) coordinates
        analytically from (psi, theta) ones."""
        r = np.sqrt(psi / (np.cos(theta) ** 2 / self.a + np.sin(theta) ** 2 / self.b))
        return self.o_point.R + r * np.cos(theta), self.o_point.Z + r * np.sin(theta)

    def perpendicular_curve(
        self, R: float, Z: float
    ) -> Callable[[npt.ArrayLike], tuple[npt.NDArray, npt.NDArray]]:
        """Returns a function of psi which is a curve that is
        perpendicular to the magnetic field, passing through the R, Z
        point specificed.

        """

        def x_of_s(s: npt.ArrayLike) -> npt.NDArray:
            return np.asarray(
                self.o_point.R
                + (R - self.o_point.R) * np.exp(2 * np.asarray(s) / self.a)
            )

        def y_of_s(s: npt.ArrayLike) -> npt.NDArray:
            return np.asarray(
                self.o_point.Z
                + (Z - self.o_point.Z) * np.exp(2 * np.asarray(s) / self.b)
            )

        def curve(psi: float) -> tuple[float, float]:
            s_solve = root_scalar(
                lambda s: self.psi_func(x_of_s(s), y_of_s(s), grid=False) - psi,
                bracket=[-11.0, 11.0],
                xtol=1e-10,
                rtol=1e-10,
            )
            if not s_solve.converged:
                raise RuntimeError("Solution did not converge.")
            s = s_solve.root
            return float(x_of_s(s)), float(y_of_s(s))

        return np.vectorize(curve)

    def perpendicular_distance(
        self, R: float, Z: float
    ) -> Callable[[npt.ArrayLike], tuple[npt.NDArray, npt.NDArray]]:
        """Returns a function of psi which returns the distance along
        a curve originating at R, Z and is perpendicular to the
        magnetic field.

        """
        x = R - self.o_point.R
        y = Z - self.o_point.Z
        asq = self.a * self.a
        bsq = self.b * self.b

        def x_of_s(s: npt.ArrayLike) -> npt.NDArray:
            return np.asarray(self.o_point.R + x * np.exp(2 * np.asarray(s) / self.a))

        def y_of_s(s: npt.ArrayLike) -> npt.NDArray:
            return np.asarray(self.o_point.Z + y * np.exp(2 * np.asarray(s) / self.b))

        def integrand(s: float) -> float:
            return float(
                2
                * np.sqrt(
                    x * x * np.exp(4 * s / self.a) / asq
                    + y * y * np.exp(4 * s / self.b) / bsq
                )
            )

        def distance(psi: float) -> tuple[float, float]:
            s_solve = root_scalar(
                lambda s: self.psi_func(x_of_s(s), y_of_s(s), grid=False) - psi,
                bracket=[-11.0, 11.0],
                xtol=1e-10,
                rtol=1e-10,
            )
            if not s_solve.converged:
                raise RuntimeError("Solution did not converge.")
            s = s_solve.root
            dist, err = quad(integrand, 0, s)
            return float(dist), float(err)

        return np.vectorize(distance)


equilibria = builds(
    FakeEquilibrium,
    builds(Point2D, floats(1.0, 10.0), whole_numbers),
    floats(0.1, 0.99),
    floats(0.1, 2.0),
    floats(0.1, 10.0),
)


@settings(deadline=None)
@given(
    equilibria,
    one_of(
        floats(-2 * np.pi, 2 * np.pi),
        arrays(
            np.float64,
            array_shapes(max_side=3, max_dims=2),
            elements=floats(-np.pi, np.pi),
            fill=nothing(),
        ),
    ),
    floats(0.01, 1.0),
    floats(-2 * np.pi, 2 * np.pi),
)
def test_field_trace(
    eq: FakeEquilibrium, phis: npt.NDArray, psi_start: float, theta_start: float
):
    trace = equilibrium_trace(cast(Equilibrium, eq))
    R_start, Z_start = eq.to_RZ(psi_start, theta_start)
    positions, distances = trace(
        SliceCoord(float(R_start), float(Z_start), CoordinateSystem.CYLINDRICAL),
        phis,
    )
    R_exp, Z_exp = eq.to_RZ(psi_start, theta_start + phis / eq.q)
    np.testing.assert_allclose(positions.x1, R_exp, 1e-8, 1e-8)
    np.testing.assert_allclose(positions.x2, Z_exp, 1e-8, 1e-8)
    assert np.all(distances * phis >= 0)
    sorted_distances = np.ravel(distances)[np.argsort(np.ravel(phis))]
    assert np.all(sorted_distances[:-1] <= sorted_distances[1:])


@fixture
def x_point_equilibrium():
    """Creates an equilibrium with an X-point."""
    nx = 65
    ny = 65

    r1d = np.linspace(1.0, 2.0, nx)
    z1d = np.linspace(-1.0, 1.0, ny)
    r2d, z2d = np.meshgrid(r1d, z1d, indexing="ij")

    r0 = 1.5
    z0 = -0.3

    # This has two O-points, and one x-point at (r0, z0)
    def psi_func(R, Z):
        return np.exp(-((R - r0) ** 2 + (Z - z0 - 0.3) ** 2) / 0.3**2) + np.exp(
            -((R - r0) ** 2 + (Z - z0 + 0.3) ** 2) / 0.3**2
        )

    return TokamakEquilibrium(
        r1d,
        z1d,
        psi_func(r2d, z2d),
        np.linspace(0.0, 1.0, nx),  # psi1d
        np.linspace(0.0, 1.0, nx),  # fpol1d
        make_regions=False,
    )


def test_trace_outside_separatrix(x_point_equilibrium):
    """Tests tracing the field when there is an X-point present, to
    ensure it stays on the correct side of the separatrix.

    """
    eq = x_point_equilibrium
    r0 = eq.x_points[0].R
    z0 = eq.x_points[0].Z

    # Find a starting point on the separatrix, below the X-point
    Z0 = z0 - 0.1
    psi_val = eq.psi(r0, z0)
    R0_sol = root_scalar(lambda R: eq.psi(R, Z0) - psi_val, bracket=[r0, 2 * r0])
    assert R0_sol.converged
    R0 = R0_sol.root + 1e-8
    positions, _ = equilibrium_trace(eq)(
        SliceCoord(R0, Z0, CoordinateSystem.CYLINDRICAL), np.linspace(0.0, 0.52, 9)
    )
    for R, Z, Z_prev in zip(positions.x1[1:], positions.x2[1:], positions.x2[:-1]):
        assert R > r0
        assert Z > Z_prev
    assert np.any(positions.x2 > z0), "Did not integrate past X-point"


def test_trace_private_flux_region(x_point_equilibrium):
    """Tests tracing the field when there is an X-point present, to
    ensure it stays on the correct side of the separatrix.

    """
    eq = x_point_equilibrium
    r0 = eq.x_points[0].R
    z0 = eq.x_points[0].Z

    # Find a starting point on the separatrix, below the X-point
    Z0 = z0 - 0.1
    psi_val = eq.psi(r0, z0)
    R0_sol = root_scalar(lambda R: eq.psi(R, Z0) - psi_val, bracket=[r0, 2 * r0])
    assert R0_sol.converged
    R0 = R0_sol.root - 1e-8
    positions, _ = equilibrium_trace(eq)(
        SliceCoord(R0, Z0, CoordinateSystem.CYLINDRICAL), np.linspace(0.0, 0.5, 9)
    )
    for R, R_prev, Z in zip(positions.x1[1:], positions.x1[:-1], positions.x2[1:]):
        assert R < R_prev
        assert Z < z0
    assert np.any(positions.x1 < r0), "Did not integrate past X-point"


def test_trace_solenoid(x_point_equilibrium):
    """Tests tracing the field when there is an X-point present, to
    ensure it stays on the correct side of the separatrix.

    """
    eq = x_point_equilibrium
    r0 = eq.x_points[0].R
    z0 = eq.x_points[0].Z
    psi_val = eq.psi(r0, z0)

    Z_sol = root_scalar(lambda Z: eq.psi(r0, Z) - psi_val, bracket=[eq.o_point.Z, 2])
    assert Z_sol.converged
    Z_max = Z_sol.root
    R_sol = root_scalar(
        lambda R: eq.psi(R, eq.o_point.Z) - psi_val, bracket=[0.0, eq.o_point.R]
    )
    assert R_sol.converged
    R_min = R_sol.root
    R_sol = root_scalar(
        lambda R: eq.psi(R, eq.o_point.Z) - psi_val,
        bracket=[eq.o_point.R, 2 * eq.o_point.R],
    )
    assert R_sol.converged
    R_max = R_sol.root

    # Find a starting point on the separatrix, below the X-point
    Z0 = z0 + 0.1
    R0_sol = root_scalar(lambda R: eq.psi(R, Z0) - psi_val, bracket=[r0, 2 * r0])
    assert R0_sol.converged
    R0 = R0_sol.root - 1e-7
    positions, _ = equilibrium_trace(eq)(
        SliceCoord(R0, Z0, CoordinateSystem.CYLINDRICAL), np.linspace(0.0, 2.0, 31)
    )
    R0_crossings = (
        -1
    )  # First iteration will show as a crossing, but don't want to count it
    for R, R_prev, Z, Z_prev in zip(
        positions.x1[1:], positions.x1[:-1], positions.x2[1:], positions.x2[:-1]
    ):
        assert Z > z0
        assert Z < Z_max
        assert R > R_min
        assert R < R_max
        if R_prev <= R0 and R > R0:
            assert Z_prev <= Z0
            assert Z > Z0
            R0_crossings += 1
    assert R0_crossings >= 2


def test_trace_x_point(x_point_equilibrium):
    """Test tracing from the X-point. There should be no
    movement. However, as it is an unstable equilibrium, this test is
    being done with very high tolerances."""
    eq = x_point_equilibrium
    r0 = eq.x_points[0].R
    z0 = eq.x_points[0].Z
    positions, _ = equilibrium_trace(eq)(
        SliceCoord(r0, z0, CoordinateSystem.CYLINDRICAL), np.linspace(0.0, 0.1, 9)
    )
    np.testing.assert_allclose(positions.x1, r0, 1e-3, 1e-3)
    np.testing.assert_allclose(positions.x2, z0, 1e-3, 1e-3)


def test_trace_o_point(x_point_equilibrium):
    """Test tracing from the O-point. There should be no
    movement. However, as it is not a stable equilibrium, this test is
    being done with higher tolerance than usual."""
    eq = x_point_equilibrium
    R0 = eq.o_point.R
    Z0 = eq.o_point.Z
    positions, _ = equilibrium_trace(eq)(
        SliceCoord(R0, Z0, CoordinateSystem.CYLINDRICAL), np.linspace(0.1, 9)
    )
    tol = 1e-6
    np.testing.assert_allclose(positions.x1, R0, tol, tol)
    np.testing.assert_allclose(positions.x2, Z0, tol, tol)


# TODO: Generate some fake EQDSK data so I can test the equilibrium reader

# We don't want psis that are too close together. psi=0 is a singular
# point in our equations, so will have bad accuracy; avoid it!
psis = integers(-99, 400).map(lambda x: (x + 100) / 100)


@settings(deadline=None)
@given(
    equilibria,
    floats(0.0, 2 * np.pi),
    lists(psis, min_size=2, max_size=2, unique=True),
    arrays(
        np.float64,
        array_shapes(max_side=3),
        # The scipy integration routines don't behave well if we have
        # numbers that differ by around machine-epsilon, so we enforce
        # larger differences between values
        elements=integers(-500, 500).map(lambda x: (x + 500) / 1000),
        fill=nothing(),
    ),
)
def test_perpendicular_edges(
    eq: FakeEquilibrium, angle: float, termini_psi: list[float], positions: npt.NDArray
) -> None:
    R_start, Z_start = map(float, eq.to_RZ(termini_psi[0], angle))
    curve_of_psi = eq.perpendicular_curve(R_start, Z_start)
    R_end, Z_end = map(float, curve_of_psi(termini_psi[1]))
    start = SliceCoord(R_start, Z_start, CoordinateSystem.CYLINDRICAL)
    end = SliceCoord(R_end, Z_end, CoordinateSystem.CYLINDRICAL)
    edge = perpendicular_edge(cast(Equilibrium, eq), start, end)
    # Check starts and ends of the curve are at the right locations
    assert edge(0.0).to_coord() == start
    assert edge(1.0).to_coord() == end
    actual = edge(positions)
    actual_psis = eq.psi_func(actual.x1, actual.x2, grid=False)
    # Check positions of points on line correspond to our
    # semi-analytic expression
    expected_R, expected_Z = curve_of_psi(actual_psis)
    np.testing.assert_allclose(actual.x1, expected_R, 1e-8, 1e-8)
    np.testing.assert_allclose(actual.x2, expected_Z, 1e-8, 1e-8)
    # Check distances along curve are proportional to the normalised
    # `position` parameter

    distance = eq.perpendicular_distance(R_start, Z_start)
    dist, err = distance(actual_psis)
    total_distance, terr = distance(termini_psi[1])
    # Standard uncertainty propagation
    total_err = (
        np.sqrt(total_distance**2 * err**2 + dist**2 * terr**2)
        / total_distance**2
    )
    for d, e, p in np.nditer([dist, total_err, positions]):
        np.testing.assert_allclose(d / total_distance, p, 1e-8, max(1e-8, float(e)))


@settings(deadline=None)
@given(
    equilibria,
    lists(
        integers(0, 999).map(lambda x: x / 1000 * 2 * np.pi),
        min_size=2,
        max_size=2,
        unique=True,
    ),
    psis,
    one_of(
        floats(-0.5, 1.0),
        arrays(
            np.float64,
            array_shapes(max_side=3),
            # The scipy integration routines don't behave well if we have
            # numbers that differ by around machine-epsilon, so we enforce
            # larger differences between values
            elements=integers(-100, 50).map(lambda x: (x + 50) / 100),
            fill=nothing(),
        ),
    ),
)
def test_flux_surface_edges(
    eq: FakeEquilibrium,
    start_end_points: list[float],
    psi: float,
    positions: npt.NDArray,
) -> None:
    R_start, Z_start = eq.to_RZ(psi, start_end_points[0])
    start = SliceCoord(float(R_start), float(Z_start), CoordinateSystem.CYLINDRICAL)
    R_end, Z_end = eq.to_RZ(psi, start_end_points[1])
    end = SliceCoord(float(R_end), float(Z_end), CoordinateSystem.CYLINDRICAL)
    curve = flux_surface_edge(cast(TokamakEquilibrium, eq), start, end)
    # Check termini of curve
    assert curve(0.0).to_coord() == start
    assert curve(1.0).to_coord() == end
    actual = curve(positions)
    # Check all points have the correct psi value
    np.testing.assert_allclose(eq.psi(actual.x1, actual.x2), psi, 1e-8, 1e-8)
    # Calculate distances of points along ellipse and check they are
    # proportional to the normalised `position` parameter
    a_prime = np.sqrt(eq.a * psi)
    b_prime = np.sqrt(eq.b * psi)
    semi_maj = max(a_prime, b_prime)
    semi_min = min(a_prime, b_prime)
    m = 1 - (semi_min / semi_maj) ** 2
    parameters = np.arctan2(
        a_prime * (actual.x2 - eq.o_point.Z), b_prime * (actual.x1 - eq.o_point.R)
    )
    start_parameter = np.arctan2(
        a_prime * (start.x2 - eq.o_point.Z), b_prime * (start.x1 - eq.o_point.R)
    )
    end_parameter = (
        smallest_angle_between(
            np.arctan2(
                a_prime * (end.x2 - eq.o_point.Z), b_prime * (end.x1 - eq.o_point.R)
            ),
            start_parameter,
        )
        + start_parameter
    )
    t = smallest_angle_between(parameters, start_parameter) + start_parameter
    offset = 0.5 * np.pi if eq.b < eq.a else 0.0
    start_distance = ellipeinc(start_parameter - offset, m)
    distances = ellipeinc(t - offset, m) - start_distance
    total_arc = ellipeinc(end_parameter - offset, m) - start_distance
    np.testing.assert_allclose(distances / total_arc, positions, 1e-8, 1e-8)
