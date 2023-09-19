from typing import cast

from hypnotoad import Equilibrium, Point2D  # type: ignore
from hypnotoad.cases.tokamak import TokamakEquilibrium  # type: ignore
from hypothesis import given
from hypothesis.strategies import builds, floats, nothing, one_of
from hypothesis.extra.numpy import arrays, array_shapes
import numpy as np
import numpy.typing as npt
from pytest import fixture
from scipy.optimize import root_scalar

from neso_fame.hypnotoad import equilibrium_trace
from neso_fame.mesh import CoordinateSystem, SliceCoord

from .conftest import whole_numbers


class FakeEquilibrium:
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

    def psi_func(self, x, y, dx=0, dy=0, grid=True):
        """Mimics the signature of RectBivariateSpline."""
        assert not grid
        if dx == 0 and dy == 0:
            return (
                -((x - self.o_point.R) ** 2) / self.a
                - (y - self.o_point.Z) ** 2 / self.b
            )
        elif dx == 1 and dy == 0:
            return -2.0 * (x - self.o_point.R) / self.a
        elif dx == 2 and dy == 0:
            return -2.0 / self.a
        elif dx == 0 and dy == 1:
            return -2.0 * (y - self.o_point.Z) / self.b
        elif dx == 0 and dy == 2:
            return -2.0 / self.b
        else:
            return 0.0

    def Bp_R(self, R, Z):
        return -2 * (Z - self.o_point.Z) / (R * self.b)

    def Bp_Z(self, R, Z):
        return 2 * (R - self.o_point.R) / (R * self.a)

    def Bzeta(self, R, Z):
        R_diff = R - self.o_point.R
        Z_diff = Z - self.o_point.Z
        r = np.sqrt(R_diff**2 + Z_diff**2)
        sin = Z_diff / r
        cos = R_diff / r
        return self.q * R * (-self.Bp_R(R, Z) * sin + self.Bp_Z(R, Z) * cos) / r

    def to_RZ(self, psi, theta):
        """Convenience fucntion to calculate (R, Z) coordinates
        analytically from (psi, theta) ones."""
        r = -np.sqrt(psi / (np.cos(theta) ** 2 / self.a + np.sin(theta) ** 2 / self.b))
        return self.o_point.R + r * np.cos(theta), self.o_point.Z + r * np.sin(theta)


@given(
    builds(
        FakeEquilibrium,
        builds(Point2D, floats(1., 10.), whole_numbers),
        floats(0.01, 0.99),
        floats(0.01, 10.),
        floats(0.1, 10.0),
    ),
    one_of(floats(-4*np.pi, 4*np.pi), arrays(
        np.float64,
        array_shapes(),
        elements=floats(-np.pi, np.pi),
        fill=nothing(),
        unique=True,
    )),
    floats(0.01, 1.0),
    floats(-2 * np.pi, 2 * np.pi),
)
def test_field_trace(
    eq: FakeEquilibrium, phis: npt.NDArray, psi_start: float, theta_start: float
):
    trace = equilibrium_trace(cast(Equilibrium, eq))
    R_start, Z_start = eq.to_RZ(psi_start, theta_start)
    positions, distances = trace(
        SliceCoord(R_start, Z_start, CoordinateSystem.CYLINDRICAL),
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
    R0_sol = root_scalar(lambda R: eq.psi(R, Z0) - psi_val, bracket=[r0, 2*r0])
    assert R0_sol.converged
    R0 = R0_sol.root + 1e-8
    positions, _ = equilibrium_trace(eq)(SliceCoord(R0, Z0, CoordinateSystem.CYLINDRICAL), np.linspace(0., 0.52, 9))
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
    R0_sol = root_scalar(lambda R: eq.psi(R, Z0) - psi_val, bracket=[r0, 2*r0])
    assert R0_sol.converged
    R0 = R0_sol.root - 1e-8
    positions, _ = equilibrium_trace(eq)(SliceCoord(R0, Z0, CoordinateSystem.CYLINDRICAL), np.linspace(0., 0.5, 9))
    for R, R_prev, Z in zip(positions.x1[1:], positions.x1[:-1], positions.x2[1:]):
        print(R, R_prev)
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
    R_sol = root_scalar(lambda R: eq.psi(R, eq.o_point.Z) - psi_val, bracket=[0.0, eq.o_point.R])
    assert R_sol.converged
    R_min = R_sol.root
    R_sol = root_scalar(lambda R: eq.psi(R, eq.o_point.Z) - psi_val, bracket=[eq.o_point.R, 2*eq.o_point.R])
    assert R_sol.converged
    R_max = R_sol.root

    # Find a starting point on the separatrix, below the X-point
    Z0 = z0 + 0.1
    R0_sol = root_scalar(lambda R: eq.psi(R, Z0) - psi_val, bracket=[r0, 2*r0])
    assert R0_sol.converged
    R0 = R0_sol.root - 1e-7
    positions, _ = equilibrium_trace(eq)(SliceCoord(R0, Z0, CoordinateSystem.CYLINDRICAL), np.linspace(0., 2.0, 31))
    R0_crossings = -1 # First iteration will show as a crossing, but don't want to count it
    for R, R_prev, Z, Z_prev in zip(positions.x1[1:], positions.x1[:-1], positions.x2[1:], positions.x2[:-1]):
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
    positions, _ = equilibrium_trace(eq)(SliceCoord(r0, z0, CoordinateSystem.CYLINDRICAL), np.linspace(0., 0.1, 9))
    np.testing.assert_allclose(positions.x1, r0, 1e-3, 1e-3)
    np.testing.assert_allclose(positions.x2, z0, 1e-3, 1e-3)


def test_trace_o_point(x_point_equilibrium):
    """Test tracing from the O-point. There should be no
    movement. However, as it is not a stable equilibrium, this test is
    being done with higher tolerance than usual."""
    eq = x_point_equilibrium
    R0 = eq.o_point.R
    Z0 = eq.o_point.Z
    positions, _ = equilibrium_trace(eq)(SliceCoord(R0, Z0, CoordinateSystem.CYLINDRICAL), np.linspace(0.1, 9))
    tol = 1e-6
    np.testing.assert_allclose(positions.x1, R0, tol, tol)
    np.testing.assert_allclose(positions.x2, Z0, tol, tol)
