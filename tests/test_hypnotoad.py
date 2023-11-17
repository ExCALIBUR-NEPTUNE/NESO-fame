from functools import cache
import itertools
import operator
import os.path
import warnings
from collections import Counter
from collections.abc import Iterable, Sequence
from tempfile import TemporaryDirectory
from typing import Any, Callable, NamedTuple, cast
from unittest.mock import patch

import numpy as np
import numpy.typing as npt
from hypnotoad import Equilibrium, Point2D  # type: ignore
from hypnotoad.cases.tokamak import TokamakEquilibrium  # type: ignore
from hypnotoad.core.mesh import Mesh, MeshRegion  # type: ignore
from hypnotoad.geqdsk._geqdsk import write as write_geqdsk  # type: ignore
from hypothesis import given, settings, HealthCheck
from hypothesis.extra.numpy import array_shapes, arrays
from hypothesis.strategies import (
    booleans,
    builds,
    composite,
    floats,
    integers,
    just,
    lists,
    nothing,
    one_of,
    sampled_from,
    shared,
    tuples,
)
from pytest import fixture, mark
from scipy.integrate import quad
from scipy.optimize import root_scalar
from scipy.special import ellipeinc

from neso_fame.hypnotoad_interface import (
    _get_region_boundaries,
    eqdsk_equilibrium,
    equilibrium_trace,
    flux_surface_edge,
    get_mesh_boundaries,
    perpendicular_edge,
)
from neso_fame.mesh import (
    CoordinateSystem,
    FieldTracer,
    Quad,
    SliceCoord,
    SliceCoords,
    StraightLineAcrossField,
)

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

    def __init__(
        self,
        o_point: Point2D = Point2D(1.0, 1.0),
        a: float = 0.25,
        b: float = 0.25,
        q: float = 1.0,
    ) -> None:
        self.o_point = o_point
        self.a = a
        self.b = b
        self.q = q
        self.x_points: list[Point2D] = []

    def psi_func(
        self,
        x: npt.ArrayLike,
        y: npt.ArrayLike,
        dx: int = 0,
        dy: int = 0,
        grid: bool = True,
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

    def psi(self, R: npt.ArrayLike, Z: npt.ArrayLike) -> npt.NDArray:
        "Return the poloidal flux at the given (R,Z) location"
        return self.psi_func(R, Z, grid=False)

    def f_R(self, R: npt.ArrayLike, Z: npt.ArrayLike) -> npt.NDArray:
        """returns the R component of the vector Grad(psi)/|Grad(psi)|**2."""
        dpsidR = self.psi_func(R, Z, dx=1, grid=False)
        dpsidZ = self.psi_func(R, Z, dy=1, grid=False)
        return dpsidR / (dpsidR**2 + dpsidZ**2)

    def f_Z(self, R: npt.ArrayLike, Z: npt.ArrayLike) -> npt.NDArray:
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

    def fpol(
        self, R: float | npt.NDArray, Z: float | npt.NDArray
    ) -> float | npt.NDArray:
        R_diff = R - self.o_point.R
        Z_diff = Z - self.o_point.Z
        r = np.sqrt(R_diff**2 + Z_diff**2)
        sin = Z_diff / r
        cos = R_diff / r
        return np.asarray(
            self.q * R * R * (-self.Bp_R(R, Z) * sin + self.Bp_Z(R, Z) * cos) / r
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


def fake_fpol(eq: FakeEquilibrium, R: npt.NDArray, Z: npt.NDArray) -> npt.NDArray:
    return np.asarray(eq.fpol(R, Z))


fake_equilibria = builds(
    FakeEquilibrium,
    builds(Point2D, floats(1.0, 10.0), whole_numbers),
    floats(0.25, 0.99),
    floats(0.25, 2.0),
    floats(0.1, 10.0),
)

class OPoint(NamedTuple):
    R: float
    Z: float
    aspect_ratio: float
    magnitude: float


def create_equilibrium_psi_func(
    o_points: tuple[OPoint, ...]
) -> Callable[[npt.NDArray, npt.NDArray], npt.NDArray]:
    def semi_axes_squared(point: OPoint) -> tuple[float, float]:
        aspect_sq = point.aspect_ratio * point.aspect_ratio
        asq = point.magnitude * point.magnitude * 2 / (1 + aspect_sq)
        bsq = aspect_sq * asq
        return asq, bsq

    def psi_func(R: npt.NDArray, Z: npt.NDArray) -> npt.NDArray:
        return np.asarray(
            sum(
                np.exp(-((R - r0) ** 2) / asq - (Z - z0) ** 2 / bsq)
                for (r0, z0), (asq, bsq) in zip(
                    map(operator.itemgetter(slice(2)), o_points),
                    map(semi_axes_squared, o_points),
                )
            )
        )

    return psi_func


def create_equilibrium_dpsi_dz(
    o_points: tuple[OPoint, ...]
) -> Callable[[npt.NDArray], npt.NDArray]:
    if len(frozenset(o.R for o in o_points)) > 1:
        raise ValueError("Can only produce accurate values if o-points vertically aligned.")
    
    def semi_axes_squared(point: OPoint) -> float:
        aspect_sq = point.aspect_ratio * point.aspect_ratio
        asq = point.magnitude * point.magnitude * 2 / (1 + aspect_sq)
        bsq = aspect_sq * asq
        return bsq

    def psi_func(Z: npt.NDArray) -> npt.NDArray:
        return np.asarray(
            sum(
                - 2 * (Z - z0) / bsq * np.exp(- (Z - z0) ** 2 / bsq)
                for z0, bsq in zip(
                    map(operator.itemgetter(1), o_points),
                    map(semi_axes_squared, o_points),
                )
            )
        )

    return psi_func


def create_equilibrium_data(
    nx: int,
    ny: int,
    r_lims: tuple[float, float],
    z_lims: tuple[float, float],
    o_points: tuple[OPoint, ...],
) -> dict[str, npt.NDArray]:
    psi_func = create_equilibrium_psi_func(o_points)

    r1d = np.linspace(r_lims[0], r_lims[1], nx)
    z1d = np.linspace(z_lims[0], z_lims[1], ny)
    r2d, z2d = np.meshgrid(r1d, z1d, indexing="ij")

    if len(o_points) == 2:
        psi1d = np.linspace(0.0, 1.0, nx)
    else:
        psi1d = psi_func(
            np.linspace(sum(o.R for o in o_points) / len(o_points), r_lims[1], nx),
            np.full(nx, 0.5 * sum(z_lims)),
        )

    return {
        "R1D": r1d,
        "Z1D": z1d,
        "psi2d": psi_func(r2d, z2d),
        "psi1d": psi1d,
        "fpol1d": np.linspace(0.0, 1.0, nx),
    }


# FIXME: Switch this out to bea normal function rather than a Hypothesis strategy
def eqdsk_data(
    nx: int,
    ny: int,
    r_lims: tuple[float, float] = (1.0, 2.0),
    z_lims: tuple[float, float] = (-1.0, 1.0),
    o_points: tuple[OPoint, ...] = (OPoint(1.5, -0.6, 1.0, 0.3), OPoint(1.5, 0.0, 1.0, 0.3)),
) -> dict[str, int | float | npt.NDArray]:
    # Note: assumes conetnts of `o_points` are vertically aligned.
    data = create_equilibrium_data(nx, ny, r_lims, z_lims, o_points)
    psi_func = create_equilibrium_psi_func(o_points)
    R = o_points[0].R
    dpsi_dz = create_equilibrium_dpsi_dz(o_points)

    def find_zero(lower: float, upper: float) -> float:
        sol = root_scalar(dpsi_dz, bracket=[lower, upper])
        if not sol.converged:
            raise RuntimeError("Could not converge on Z-value")
        return cast(float, sol.root)
        
    # Get vertical position of O-point
    main_o_Z = find_zero(-0.1, 0.1)
    # Get vertical position of X-point
    lower_points = [o.Z for o in o_points if o.Z < 0.]
    upper_points = [o.Z for o in o_points if o.Z > 0.]
    lower_x = find_zero(max(lower_points) + 0.1, -0.1) if len(lower_points) > 0 else -1e4
    upper_x = find_zero(min(upper_points) - 0.1, 0.1) if len(upper_points) > 0 else 1e4
    if abs(lower_x) < abs(upper_x):
        main_x_Z = lower_x
    else:
        main_x_Z = upper_x

    dR = r_lims[1] - r_lims[0]
    dZ = z_lims[1] - z_lims[0]

    return {
        "nx": nx,
        "ny": ny,
        "rdim": r_lims[1] - r_lims[0],
        "zdim": z_lims[1] - z_lims[0],
        "rcentr": sum(o.R for o in o_points) / len(o_points),
        "bcentr": 2.,
        "rleft": r_lims[0],
        "zmid": sum(z_lims) / 2,
        "rmagx": R,
        "zmagx": main_o_Z,
        "simagx": float(psi_func(np.asarray(R), np.asarray(main_o_Z))),
        "sibdry": float(psi_func(np.asarray(R), np.asarray(main_x_Z))),
        "cpasma": 1e8,
        "pres": np.ones(nx),
        "qpsi": np.ones(nx),
        "fpol": data["fpol1d"],
        "psi": data["psi2d"],
        "rlim": np.array([r_lims[0] + 0.05 * dR, r_lims[0] + 0.05 * dR, r_lims[1] - 0.05 * dR, r_lims[1] - 0.05 * dR, r_lims[0] + 0.05 * dR]),
        "zlim": np.array([z_lims[0] + 0.05 * dZ, z_lims[1] - 0.05 * dZ, z_lims[1] - 0.05 * dZ, z_lims[0] + 0.05 * dZ, z_lims[0] + 0.05 * dZ]),
    }


def create_equilibrium(
    nx: int = 65,
    ny: int = 65,
    r_lims: tuple[float, float] = (1.0, 2.0),
    z_lims: tuple[float, float] = (-1.0, 1.0),
    o_points: tuple[OPoint, ...] = (OPoint(1.5, -0.6, 1.0, 0.3), OPoint(1.5, 0.0, 1.0, 0.3)),
    make_regions: bool = False,
    options: tuple[tuple[str, Any], ...] = tuple(),
) -> TokamakEquilibrium:
    """Creates an equilibrium. Defaults to an equilibrium with one X-point."""
    data = create_equilibrium_data(nx, ny, r_lims, z_lims, o_points)
    dR = r_lims[1] - r_lims[0]
    dZ = z_lims[1] - z_lims[0]

    return TokamakEquilibrium(
        data["R1D"],
        data["Z1D"],
        data["psi2d"],
        data["psi1d"],
        data["fpol1d"],
        wall=[
            (r_lims[0] + 0.05 * dR, z_lims[0] + 0.05 * dZ),
            (r_lims[0] + 0.05 * dR, z_lims[1] - 0.05 * dZ),
            (r_lims[1] - 0.05 * dR, z_lims[1] - 0.05 * dZ),
            (r_lims[1] - 0.05 * dR, z_lims[0] + 0.05 * dZ),
            (r_lims[0] + 0.05 * dR, z_lims[0] + 0.05 * dZ),
        ],
        make_regions=make_regions,
        settings={"refine_atol": 1e-10} | dict(options),
    )


MESH_SIZE = (
    ("nx_core", 3),
    ("nx_sol", 3),
    ("ny_inner_divertor", 3),
    ("ny_outer_divertor", 3),
    ("ny_sol", 4),
)

MeshArgs = tuple[tuple[OPoint, ...], tuple[tuple[str, Any], ...]]

LOWER_SINGLE_NULL: MeshArgs = ((OPoint(1.5, -0.8, 1.0, 0.4), OPoint(1.5, 0.0, 1.0, 0.4)),
MESH_SIZE,
    )
UPPER_SINGLE_NULL: MeshArgs = (
        (OPoint(1.5, 0.8, 1.0, 0.4), OPoint(1.5, 0.0, 1.0, 0.4)),
        MESH_SIZE,
    )
# Very, very slightly offset one of the points to make hypnotoad
# more reliably choose which one should be treated as primary
CONNECTED_DOUBLE_NULL: MeshArgs = (
        (
            OPoint(1.5, 0.8000001, 1.0, 0.4),
            OPoint(1.5, -0.8, 1.0, 0.4),
            OPoint(1.5, 0.0, 1.0, 0.4),
        ),
        MESH_SIZE,
    )
LOWER_DOUBLE_NULL: MeshArgs = (
        (
            OPoint(1.5, 0.801, 1.0, 0.4),
            OPoint(1.5, -0.8, 1.0, 0.4),
            OPoint(1.5, 0.0, 1.0, 0.4),
        ),
        MESH_SIZE + (("nx_inter_sep", 3),),
    )
UPPER_DOUBLE_NULL: MeshArgs = (
        (
            OPoint(1.5, 0.8, 1.0, 0.4),
            OPoint(1.5, -0.801, 1.0, 0.4),
            OPoint(1.5, 0.0, 1.0, 0.4),
        ),
        MESH_SIZE + (("nx_inter_sep", 3),),
    )


@fixture
def x_point_equilibrium() -> TokamakEquilibrium:
    return create_equilibrium(make_regions=False)


@settings(deadline=None)
@given(
    fake_equilibria,
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
) -> None:
    trace = equilibrium_trace(cast(TokamakEquilibrium, eq))
    R_start, Z_start = eq.to_RZ(psi_start, theta_start)
    # The fake field we are using isn't amenable for calculating _fpol
    # in the same way as real fields, so we patch that calculation
    with patch("neso_fame.hypnotoad_interface._fpol", fake_fpol):
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
    np.testing.assert_allclose(
        eq.psi(positions.x1, positions.x2), psi_start, 1e-8, 1e-8
    )


def test_trace_outside_separatrix(x_point_equilibrium: TokamakEquilibrium) -> None:
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
    np.testing.assert_allclose(
        eq.psi(positions.x1, positions.x2), eq.psi(R0, Z0), 1e-8, 1e-8
    )


def test_trace_private_flux_region(x_point_equilibrium: TokamakEquilibrium) -> None:
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
    np.testing.assert_allclose(
        eq.psi(positions.x1, positions.x2), eq.psi(R0, Z0), 1e-8, 1e-8
    )


def test_trace_core(x_point_equilibrium: TokamakEquilibrium) -> None:
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

    # Find a starting point on the separatrix, above the X-point
    Z0 = z0 + 0.1
    R0_sol = root_scalar(lambda R: eq.psi(R, Z0) - psi_val, bracket=[r0, 2 * r0])
    assert R0_sol.converged
    R0 = R0_sol.root - 1e-6
    positions, _ = equilibrium_trace(eq)(
        SliceCoord(R0, Z0, CoordinateSystem.CYLINDRICAL), np.linspace(0.0, 2.0, 31)
    )
    R0_crossings = (
        -1  # First iteration will show as a crossing, but don't want to count it
    )
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
    np.testing.assert_allclose(
        eq.psi(positions.x1, positions.x2), eq.psi(R0, Z0), 1e-7, 1e-7
    )


def test_trace_x_point(x_point_equilibrium: TokamakEquilibrium) -> None:
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
    np.testing.assert_allclose(
        eq.psi(positions.x1, positions.x2), eq.psi(r0, z0), 1e-3, 1e-3
    )


def test_trace_o_point(x_point_equilibrium: TokamakEquilibrium) -> None:
    """Test tracing from the O-point. There should be no
    movement. However, as it is not a stable equilibrium, this test is
    being done with higher tolerance than usual."""
    eq = x_point_equilibrium
    R0 = eq.o_point.R
    Z0 = eq.o_point.Z
    positions, _ = equilibrium_trace(eq)(
        SliceCoord(R0, Z0, CoordinateSystem.CYLINDRICAL), np.linspace(0.0, 0.1, 9)
    )
    tol = 1e-6
    np.testing.assert_allclose(positions.x1, R0, tol, tol)
    np.testing.assert_allclose(positions.x2, Z0, tol, tol)
    np.testing.assert_allclose(
        eq.psi(positions.x1, positions.x2), eq.psi(R0, Z0), tol, tol
    )


# We don't want psis that are too close together. psi=0 is a singular
# point in our equations, so will have bad accuracy; avoid it!
psis = integers(-99, 400).map(lambda x: (x + 100) / 100)


@settings(deadline=None)
@given(
    fake_equilibria,
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
        np.sqrt(total_distance**2 * err**2 + dist**2 * terr**2) / total_distance**2
    )
    for d, e, p in np.nditer([dist, total_err, positions]):
        np.testing.assert_allclose(d / total_distance, p, 1e-7, max(1e-7, float(e)))


@np.vectorize
def _smallest_angle_between(end_angle: float, start_angle: float) -> float:
    delta_angle = end_angle - start_angle
    # Deal with cases where the angles stradle the atan2 discontinuity
    if delta_angle < -np.pi:
        return delta_angle + 2 * np.pi
    elif delta_angle > np.pi:
        return delta_angle - 2 * np.pi
    return delta_angle


@settings(deadline=None)
@given(
    fake_equilibria,
    floats(0.0, 2 * np.pi).flatmap(
        lambda x: tuples(
            just(x),
            one_of(floats(0.005, 0.25 * np.pi), floats(-0.25 * np.pi, -0.001)).map(
                lambda y: y + x
            ),
        )
    ),
    # lists(
    #     integers(0, 999).map(lambda x: x / 1000 * 2 * np.pi),
    #     min_size=2,
    #     max_size=2,
    #     unique=True,
    # ),
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
    start_end_points: tuple[float, float],
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
        _smallest_angle_between(
            np.arctan2(
                a_prime * (end.x2 - eq.o_point.Z), b_prime * (end.x1 - eq.o_point.R)
            ),
            start_parameter,
        )
        + start_parameter
    )
    t = _smallest_angle_between(parameters, start_parameter) + start_parameter
    offset = 0.5 * np.pi if eq.b < eq.a else 0.0
    start_distance = ellipeinc(start_parameter - offset, m)
    distances = ellipeinc(t - offset, m) - start_distance
    total_arc = ellipeinc(end_parameter - offset, m) - start_distance
    np.testing.assert_allclose(distances / total_arc, positions, 1e-8, 1e-8)


# Building meshes is expensive, so use caching to avoid having to do
# it more than once
@cache
def to_mesh(args: tuple[tuple[OPoint, ...], tuple[tuple[str, Any], ...]]) -> Mesh:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", "divide by zero", RuntimeWarning, "hypnotoad.core.equilibrium"
        )
        warnings.filterwarnings(
            "ignore",
            "invalid value encountered in divide",
            RuntimeWarning,
            "hypnotoad.core.equilibrium",
        )
        eq = create_equilibrium(o_points=args[0], make_regions=True, options=args[1])
        m = Mesh(
            eq,
            {"follow_perpendicular_rtol": 1e-10, "follow_perpendicular_atol": 1e-10}
            | dict(eq.user_options),
        )
        m.calculateRZ()
    return m

sample_mesh_configs: list[MeshArgs] = [
    LOWER_SINGLE_NULL,
    UPPER_SINGLE_NULL,
    CONNECTED_DOUBLE_NULL,
    LOWER_DOUBLE_NULL,
    UPPER_DOUBLE_NULL,
]
real_meshes = sampled_from(sample_mesh_configs).map(to_mesh)
mesh_regions = real_meshes.map(lambda x: list(x.regions.values())).flatmap(sampled_from)

shared_mesh_regions = shared(mesh_regions, key=0)


@composite
def flux_surface_termini(
    draw: Any, region: MeshRegion
) -> tuple[SliceCoord, SliceCoord]:
    n, m = region.Rxy.corners.shape
    i = draw(integers(0, n - 1))
    j = draw(integers(0, m - 2))
    start = SliceCoord(
        region.Rxy.corners[i, j], region.Zxy.corners[i, j], CoordinateSystem.CYLINDRICAL
    )
    end = SliceCoord(
        region.Rxy.corners[i, j + 1],
        region.Zxy.corners[i, j + 1],
        CoordinateSystem.CYLINDRICAL,
    )
    if draw(booleans()):
        # Reverse direction of points
        return end, start
    else:
        return start, end


@composite
def perpendicular_termini(
    draw: Any, region: MeshRegion
) -> tuple[SliceCoord, SliceCoord]:
    n, m = region.Rxy.corners.shape
    i = draw(integers(0, n - 2))
    j = draw(integers(0, m - 1))
    start = SliceCoord(
        region.Rxy.corners[i, j], region.Zxy.corners[i, j], CoordinateSystem.CYLINDRICAL
    )
    end = SliceCoord(
        region.Rxy.corners[i + 1, j],
        region.Zxy.corners[i + 1, j],
        CoordinateSystem.CYLINDRICAL,
    )
    if draw(booleans()):
        # Reverse direction of points
        return end, start
    else:
        return start, end


@settings(deadline=None)
@given(
    shared_mesh_regions,
    shared_mesh_regions.flatmap(flux_surface_termini),
    one_of(
        floats(0.0, 1.0),
        arrays(
            np.float64,
            array_shapes(max_side=3),
            # The scipy integration routines don't behave well if we have
            # numbers that differ by around machine-epsilon, so we enforce
            # larger differences between values
            elements=integers(-50, 50).map(lambda x: (x + 50) / 100),
            fill=nothing(),
        ),
    ),
)
def test_flux_surface_realistic_topology(
    region: MeshRegion,
    start_end_points: tuple[SliceCoord, SliceCoord],
    positions: npt.NDArray,
) -> None:
    eq: TokamakEquilibrium = region.meshParent.equilibrium
    start, end = start_end_points
    curve = flux_surface_edge(eq, start, end)
    psi = eq.psi(start.x1, start.x2)
    assert np.isclose(psi, eq.psi(end.x1, end.x2), 1e-5, 1e-5)
    # Check termini of curve
    assert curve(0.0).to_coord() == start
    assert curve(1.0).to_coord() == end
    actual = curve(positions)
    # Check all points have the correct psi value
    np.testing.assert_allclose(eq.psi(actual.x1, actual.x2), psi, 1e-5, 1e-5)


@settings(deadline=None)
@mark.filterwarnings("ignore:divide by zero encountered in double_scalars")
@given(
    shared_mesh_regions,
    shared_mesh_regions.flatmap(perpendicular_termini),
    one_of(
        floats(0.0, 1.0),
        arrays(
            np.float64,
            array_shapes(max_side=3),
            # The scipy integration routines don't behave well if we have
            # numbers that differ by around machine-epsilon, so we enforce
            # larger differences between values
            elements=integers(-50, 50).map(lambda x: (x + 50) / 100),
            fill=nothing(),
        ),
    ),
)
def test_perpendicular_edges_realistic_topology(
    region: MeshRegion,
    start_end_points: tuple[SliceCoord, SliceCoord],
    positions: npt.NDArray,
) -> None:
    eq: TokamakEquilibrium = region.meshParent.equilibrium
    start, end = start_end_points
    psi_start = eq.psi(start.x1, start.x2)
    psi_end = eq.psi(end.x1, end.x2)
    assert psi_start != psi_end
    curve = perpendicular_edge(eq, start, end)
    # Check termini of curve
    assert curve(0.0).to_coord() == start
    assert curve(1.0).to_coord() == end
    actual = curve(positions)
    # Check all have psi values between those of start and end points
    actual_psi = eq.psi(actual.x1, actual.x2)
    assert np.all(actual_psi <= max(psi_start, psi_end) + 1e-10)
    assert np.all(actual_psi >= min(psi_start, psi_end) - 1e-10)


def check_coordinate_pairs_connected(
    coord_pairs: Iterable[tuple[SliceCoord, SliceCoord]], periodic: bool
) -> None:
    """Check that the pairs of coordinates describe line segments that
    all connect together to form a larger curve."""
    location_counts = Counter(itertools.chain(*coord_pairs))
    expected_hanging = 0 if periodic else 2
    hanging_nodes = len([c for c in location_counts.values() if c == 1])
    assert hanging_nodes == expected_hanging
    assert (
        len([c for c in location_counts.values() if c == 2])
        == len(location_counts) - hanging_nodes
    )


def dummy_trace(start: SliceCoord, s: npt.ArrayLike) -> tuple[SliceCoords, npt.NDArray]:
    return SliceCoords(
        np.full_like(s, start.x1),
        np.full_like(s, start.x2),
        CoordinateSystem.CYLINDRICAL,
    ), np.asarray(s)


def check_flux_surface_bound(
    eq: Equilibrium, bound: frozenset[Quad], periodic: bool
) -> None:
    quad_nodes = [(q.shape(0.0).to_coord(), q.shape(1.0).to_coord()) for q in bound]
    # Check quads are all adjacent
    check_coordinate_pairs_connected(quad_nodes, periodic)
    # Check quads all have same psi values
    psis = np.array(
        [[eq.psi(p[0].x1, p[0].x2), eq.psi(p[1].x1, p[1].x2)] for p in quad_nodes]
    )
    np.testing.assert_allclose(psis, psis[0][0], 1e-8, 1e-8)


def check_perpendicular_bounds(eq: Equilibrium, bound: frozenset[Quad]) -> None:
    quad_nodes = [(q.shape(0.0).to_coord(), q.shape(1.0).to_coord()) for q in bound]
    # Check quads are all adjacent
    check_coordinate_pairs_connected(quad_nodes, False)
    # Check quads start at unique psi
    psis = frozenset(float(eq.psi(p[0].x1, p[0].x2)) for p in quad_nodes)
    assert len(psis) == len(bound)


@settings(deadline=None)
@given(mesh_regions, floats())
def test_flux_surface_bounds(region: MeshRegion, dx3: float) -> None:
    eq = region.meshParent.equilibrium

    def constructor(north: SliceCoord, south: SliceCoord) -> Quad:
        return Quad(
            StraightLineAcrossField(north, south), FieldTracer(dummy_trace, 2), dx3
        )

    for b in filter(bool, _get_region_boundaries(region, constructor, constructor)[:5]):
        check_flux_surface_bound(eq, b, region.name == "core(0)")


@settings(deadline=None)
@given(mesh_regions, floats())
def test_perpendicular_bounds(region: MeshRegion, dx3: float) -> None:
    eq = region.meshParent.equilibrium

    def constructor(north: SliceCoord, south: SliceCoord) -> Quad:
        return Quad(
            StraightLineAcrossField(north, south), FieldTracer(dummy_trace, 2), dx3
        )

    for b in filter(bool, _get_region_boundaries(region, constructor, constructor)[5:]):
        check_perpendicular_bounds(eq, b)


def get_region(mesh_args: MeshArgs, name: str) -> MeshRegion:
    mesh = to_mesh(mesh_args)
    name_map = {region.name: region for region in mesh.regions.values()}
    return name_map[name]


@mark.parametrize(
    "mesh_args, region_name, is_boundary",
    [
        (
            LOWER_SINGLE_NULL, "core(0)",
            [True, False, False, False, False, False, False, False, False],
        ),
        (
            LOWER_SINGLE_NULL, "core(1)",
            [False, True, False, False, False, False, False, False, False],
        ),
        (
            LOWER_SINGLE_NULL, "inner_lower_divertor(0)",
            [False, False, False, False, True, True, False, False, False],
        ),
        (
            LOWER_SINGLE_NULL, "inner_lower_divertor(1)",
            [False, True, False, False, False, True, False, False, False],
        ),
        (
            UPPER_DOUBLE_NULL, "outer_core(0)",
            [True, False, False, False, False, False, False, False, False],
        ),
        (
            UPPER_DOUBLE_NULL, "outer_core(1)",
            [False, False, False, False, False, False, False, False, False],
        ),
        (
            UPPER_DOUBLE_NULL, "outer_core(2)",
            [False, False, True, False, False, False, False, False, False],
        ),
        (
            UPPER_DOUBLE_NULL, "outer_lower_divertor(2)",
            [False, False, True, False, False, False, True, False, False],
        ),
        (
            UPPER_DOUBLE_NULL, "outer_upper_divertor(0)",
            [False, False, False, True, False, False, False, False, True],
        ),
        (
            UPPER_DOUBLE_NULL, "outer_upper_divertor(1)",
            [False, False, False, False, False, False, False, False, True],
        ),
        (
            UPPER_DOUBLE_NULL, "outer_upper_divertor(2)",
            [False, False, True, False, False, False, False, False, True],
        ),
    ],
)
def test_region_bounds(mesh_args: MeshArgs, region_name: str, is_boundary: list[bool]) -> None:
    def constructor(north: SliceCoord, south: SliceCoord) -> Quad:
        return Quad(
            StraightLineAcrossField(north, south), FieldTracer(dummy_trace, 2), 1.0
        )

    region = get_region(mesh_args, region_name)
    boundaries = _get_region_boundaries(region, constructor, constructor)
    assert [len(b) > 0 for b in boundaries] == is_boundary


@mark.parametrize(
    "mesh_args, is_boundary",
    [
        (
            UPPER_SINGLE_NULL,
            [True, True, False, True, False, False, False, True, True],
        ),
        (CONNECTED_DOUBLE_NULL, [True] * 9),
        (LOWER_DOUBLE_NULL, [True] * 9),
    ],
)
def test_mesh_bounds(mesh_args: Mesh, is_boundary: list[bool]) -> None:
    def constructor(north: SliceCoord, south: SliceCoord) -> Quad:
        return Quad(
            StraightLineAcrossField(north, south), FieldTracer(dummy_trace, 2), 1.0
        )

    mesh = to_mesh(mesh_args)
    eq = mesh.equilibrium
    bounds = get_mesh_boundaries(mesh, constructor, constructor)
    assert [len(b) > 0 for b in bounds] == is_boundary
    check_flux_surface_bound(eq, bounds[0], True)
    for b in itertools.compress(bounds[1:5], is_boundary[1:5]):
        check_flux_surface_bound(eq, b, False)
    for b in itertools.compress(bounds[5:], is_boundary[5:]):
        check_perpendicular_bounds(eq, b)


equilibrium_opoints_options = shared(sampled_from(sample_mesh_configs), key=1001)
equilibrium_options = equilibrium_opoints_options.map(lambda x: x[1])
offsets = floats(-0.1, 0.1)
equilibrium_args = shared(
    tuples(
        integers(30, 50),
        integers(30, 50),
        tuples(offsets.map(lambda x: 1. + x), offsets.map(lambda x: 2. - x)),
        tuples(offsets.map(lambda x: -0.9 + x), offsets.map(lambda x: 0.9 - x)),
        equilibrium_opoints_options.map(lambda x: x[0]),
        just(False),
        equilibrium_options,
    ),
    key=-100,
)
eqdsk_output_data = equilibrium_args.map(lambda x: eqdsk_data(*x[:-2]))


@settings(deadline=300)
@given(eqdsk_output_data, equilibrium_options.map(dict), equilibrium_args.map(lambda x: create_equilibrium(*x)))
def test_eqdsk_equilibrium(eqdsk_output: dict[str, int | float | npt.NDArray], options: dict[str, Any], eq: TokamakEquilibrium) -> None:
    with TemporaryDirectory() as path:
        eqdsk_file = os.path.join(path, "eqdsk.g")
        with open(eqdsk_file, 'w') as f:
            write_geqdsk(eqdsk_output, f)
        new_eq = eqdsk_equilibrium(eqdsk_file, options)
    assert len(eq.x_points) == len(new_eq.x_points)
    for x1, x2 in zip(eq.x_points, new_eq.x_points):
        np.testing.assert_allclose(x1.R, x2.R, 1e-6, 1e-6)
        np.testing.assert_allclose(x1.Z, x2.Z, 1e-6, 1e-6)
    np.testing.assert_allclose(eq.o_point.R, new_eq.o_point.R, 1e-6, 1e-6)
    np.testing.assert_allclose(eq.o_point.Z, new_eq.o_point.Z, 1e-6, 1e-6)
