"""Functions for representing magnetic fields using GEQDSK data read
in using hypnotoad.

"""

import itertools
from functools import reduce
from pathlib import Path
from typing import Any, Callable, Optional, cast

import numpy as np
import numpy.typing as npt
from hypnotoad import Equilibrium, Mesh, MeshRegion  # type: ignore
from hypnotoad.cases.tokamak import TokamakEquilibrium, read_geqdsk  # type: ignore
from scipy.integrate import solve_ivp

from .mesh import (
    AcrossFieldCurve,
    CoordinateSystem,
    FieldTrace,
    Quad,
    SliceCoord,
    SliceCoords,
)

IntegratedFunction = Callable[[npt.ArrayLike], tuple[npt.NDArray, ...]]
Integrand = Callable[[npt.ArrayLike, npt.NDArray], tuple[npt.ArrayLike, ...]]


def integrate_vectorized(
    start: npt.ArrayLike,
    scale: float = 1.0,
    fixed_points: dict[float, npt.ArrayLike] = {},
    rtol: float = 1e-12,
    atol: float = 1e-14,
    vectorize_integrand_calls: bool = True,
) -> Callable[[Integrand], IntegratedFunction]:
    """Decorator that will numerically integrate a function and return
    a callable that is vectorised. The integration will start from
    t=0. You must specify the y-value at that starting point. You may
    optionally specify additional "fixed points" which should fall
    along the curve at chosen t-values. If someone requests a result
    at this position, the fixed-point will be returned instead. This
    can be useful if you need to enforce the end-point of a curve to
    within a very high level of precision.

    """
    start = np.asarray(start)
    if start.ndim > 1:
        raise ValueError("`start` must be 0- or 1-dimensional")
    fixed_points = {k: np.asarray(v) for k, v in fixed_points.items()}
    bad_points = {
        k: v
        for k, v in fixed_points.items()
        if cast(npt.NDArray, v).size != start.size or cast(npt.NDArray, v).ndim > 1
    }
    if len(bad_points) > 0:
        raise ValueError(
            "Specified fixed points with sizes/shapes different from those of "
            f"`start`: {bad_points}"
        )
    pos_fixed = {k: cast(npt.NDArray, v) for k, v in fixed_points.items() if k > 0}
    neg_fixed = {k: cast(npt.NDArray, v) for k, v in fixed_points.items() if k < 0}
    if 0.0 in fixed_points and any(cast(npt.NDArray, fixed_points[0.0]) != start):
        raise ValueError(
            "Specified 0 value in `fixed-points` which is different from `start`"
        )

    def wrap(func: Integrand) -> IntegratedFunction:
        def wrapper(s: npt.ArrayLike) -> tuple[npt.NDArray, ...]:
            s = np.asarray(s) * scale
            # Set very small values to 0 to avoid underflow problems
            if cast(npt.NDArray, s).ndim > 0:
                cast(npt.NDArray, s)[np.abs(s) < 1e-50] = 0.0
            s_sorted, invert_s = np.unique(s, return_inverse=True)
            products = s_sorted[:-1] * s_sorted[1:]
            pos_fixed_positions: dict[int, npt.NDArray] = {}
            neg_fixed_positions: dict[int, npt.NDArray] = {}
            if len(products) > 0:
                pivot: int = cast(int, np.argmin(products))
                zero = (
                    pivot + 1
                    if s_sorted[pivot + 1] == 0.0
                    else pivot
                    if s_sorted[pivot] == 0.0
                    else None
                )
                sign_change = products[pivot] < 0.0
                if s_sorted[0] < 0.0:
                    if zero is not None or sign_change:
                        neg_slice: Optional[slice | int] = slice(pivot, None, -1)
                    else:
                        neg_slice = slice(None, None, -1)
                    neg = s_sorted[neg_slice]
                    neg_limit = neg[-1]
                    for k, v in neg_fixed.items():
                        loc = np.where(neg == k)[0]
                        if len(loc) > 0:
                            assert len(loc) == 1
                            neg_fixed_positions[loc[0]] = v
                else:
                    neg_slice = neg_limit = neg = None
                if s_sorted[-1] > 0.0:
                    if zero is not None and s_sorted[0] != 0.0:
                        pos_slice: Optional[slice | int] = slice(pivot + 2, None)
                    elif sign_change:
                        pos_slice = slice(pivot + 1, None)
                    else:
                        pos_slice = slice(None, None)
                    pos = s_sorted[pos_slice]
                    for k, v in pos_fixed.items():
                        loc = np.where(pos == k)[0]
                        if len(loc) > 0:
                            assert len(loc) == 1
                            pos_fixed_positions[loc[0]] = v
                    pos_limit = pos[-1]
                else:
                    pos_slice = pos_limit = pos = None
            else:
                if s_sorted > 0:
                    pos_limit = s_sorted[0]
                    pos = s_sorted
                    neg_limit = neg = None
                    zero = None
                    for k, v in pos_fixed.items():
                        if pos == k:
                            pos_fixed_positions[0] = v
                elif s_sorted < 0:
                    neg_limit = s_sorted[0]
                    neg = s_sorted
                    pos_limit = pos = None
                    zero = None
                    for k, v in neg_fixed.items():
                        if neg == k:
                            neg_fixed_positions[0] = v
                else:
                    pos_limit = pos = None
                    neg_limit = neg = None
                    zero = 0
                pos_slice = slice(None, None)
                neg_slice = slice(None, None)
            result_tmp = np.empty((cast(npt.NDArray, start).size, s_sorted.size))
            if neg is not None:
                result = solve_ivp(
                    func,
                    (0.0, neg_limit),
                    start,
                    method="DOP853",
                    t_eval=neg,
                    rtol=rtol,
                    atol=atol,
                    dense_output=True,
                    vectorized=vectorize_integrand_calls,
                )
                if not result.success:
                    raise RuntimeError("Failed to integrate along field line")
                for i, v in neg_fixed_positions.items():
                    result.y[:, i] = v
                result_tmp[:, neg_slice] = result.y
            if pos is not None:
                result = solve_ivp(
                    func,
                    (0.0, pos_limit),
                    start,
                    method="DOP853",
                    t_eval=pos,
                    rtol=rtol,
                    atol=atol,
                    dense_output=True,
                    vectorized=vectorize_integrand_calls,
                )
                if not result.success:
                    raise RuntimeError("Failed to integrate along field line")
                for i, v in pos_fixed_positions.items():
                    result.y[:, i] = v
                result_tmp[:, pos_slice] = result.y
            if zero is not None:
                result_tmp[:, zero] = start
            return tuple(
                result_tmp[i, invert_s] for i in range(cast(npt.NDArray, start).size)
            )

        return wrapper

    return wrap


def equilibrium_trace(equilibrium: Equilibrium) -> FieldTrace:
    """Return a field trace corresponding to the hypnotoad equilibrium
    object.

    """

    def trace(start: SliceCoord, phi: npt.ArrayLike) -> tuple[SliceCoords, npt.NDArray]:
        if start.system != CoordinateSystem.CYLINDRICAL:
            raise ValueError("`start` must use a cylindrical coordinate system")

        @integrate_vectorized([start.x1, start.x2, 0.0], rtol=1e-10, atol=1e-11)
        def integrated(_: npt.ArrayLike, y: npt.NDArray) -> tuple[npt.NDArray, ...]:
            R = y[0]
            Z = y[1]
            R_over_B_t = R / equilibrium.Bzeta(R, Z)
            dR_dphi = equilibrium.Bp_R(R, Z) * R_over_B_t
            dZ_dphi = equilibrium.Bp_Z(R, Z) * R_over_B_t
            return (
                dR_dphi,
                dZ_dphi,
                np.sqrt(dR_dphi * dR_dphi + dZ_dphi * dZ_dphi + 1),
            )

        phi = np.asarray(phi)
        R, Z, dist = integrated(phi)
        return SliceCoords(
            R.reshape(phi.shape), Z.reshape(phi.shape), CoordinateSystem.CYLINDRICAL
        ), dist.reshape(phi.shape)

    return trace


def eqdsk_equilibrium(
    eqdsk: str | Path, options: dict[str, Any] = {}
) -> TokamakEquilibrium:
    """Reads in GEQDSK file and creates a hypnotoad
    TokamakEquilibrium object.

    Parameters
    ----------
    eqdsk
        The filename or path of the EQDSK file to read in.
    options
        Configurations for hypnotoad, see `Tokamak Options
        <https://hypnotoad.readthedocs.io/en/latest/_temp/options.html>`_
        and `Nonorthogonal Options
        <https://hypnotoad.readthedocs.io/en/latest/_temp/nonorthogonal-options.html>`_.

    """
    possible_options = list(TokamakEquilibrium.user_options_factory.defaults) + list(
        TokamakEquilibrium.nonorthogonal_options_factory.defaults
    )
    unused_options = [opt for opt in options if opt not in possible_options]
    if unused_options != []:
        raise ValueError(f"There are options that are not used: {unused_options}")
    with open(eqdsk, "r") as f:
        return read_geqdsk(f, options, options)


def perpendicular_edge(
    eq: Equilibrium, north: SliceCoord, south: SliceCoord
) -> AcrossFieldCurve:
    """Returns a line traveling at right angles to the magnetic field
    connecting the two points on the poloidal plane. If this is not
    possible, raise an exception.

    Parameters
    ----------
    eq
        A hypnotaod Equilibrium object describing the magnetic field
    north
        The starting point of the line
    south
        The end point of the line

    """
    # FIXME: Should this space points evenly by distance rather than by psi?
    if north.system != CoordinateSystem.CYLINDRICAL:
        raise ValueError("Coordinate system of `north` must be CYLINDRICAL")
    if south.system != CoordinateSystem.CYLINDRICAL:
        raise ValueError("Coordinate system of `south` must be CYLINDRICAL")
    psi_start = float(eq.psi_func(north.x1, north.x2, grid=False))
    psi_end = float(eq.psi_func(south.x1, south.x2, grid=False))
    sign = np.sign(psi_end - psi_start)

    def f(t: npt.NDArray, x: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
        dpsidR = eq.psi_func(x[0], x[1], dx=1, grid=False)
        dpsidZ = eq.psi_func(x[0], x[1], dy=1, grid=False)
        norm = np.sqrt(dpsidR * dpsidR + dpsidZ * dpsidZ)
        return sign * dpsidR / norm, sign * dpsidZ / norm

    def terminus(t: float, x: npt.NDArray) -> float:
        return float(eq.psi(x[0], x[1])) - psi_end

    terminus.terminal = True  # type: ignore

    # Integrate until reaching the terminus, to work out the distance
    # along the contour to normalise with.
    result = solve_ivp(
        f,
        (0.0, 1e2),
        [north.x1, north.x2],
        method="DOP853",
        rtol=5e-14,
        atol=1e-14,
        events=terminus,
        vectorized=True,
    )
    if not result.success:
        raise RuntimeError("Failed to integrate along field line")
    if len(result.t_events[0]) == 0:
        raise RuntimeError("Integration did not cross over expected end-point")
    if not np.isclose(result.y[0, -1], south.x1, 1e-8, 1e-8) and not np.isclose(
        result.y[1, -1], south.x2, 1e-8, 1e-8
    ):
        raise RuntimeError("Integration did not converge on expected location")
    total_distance: float = result.t[-1]

    @integrate_vectorized(tuple(north), total_distance, {total_distance: tuple(south)})
    def solution(s: npt.ArrayLike, x: npt.ArrayLike) -> tuple[npt.NDArray, ...]:
        return f(np.asarray(s) * total_distance, np.asarray(x))

    def solution_coords(s: npt.ArrayLike) -> SliceCoords:
        s = np.asarray(s)
        R, Z = solution(s)
        return SliceCoords(
            R.reshape(s.shape), Z.reshape(s.shape), CoordinateSystem.CYLINDRICAL
        )

    return solution_coords


@np.vectorize
def smallest_angle_between(end_angle: float, start_angle: float) -> float:
    delta_angle = end_angle - start_angle
    # Deal with cases where the angles stradle the atan2 discontinuity
    if delta_angle < -np.pi:
        return delta_angle + 2 * np.pi
    elif delta_angle > np.pi:
        return delta_angle - 2 * np.pi
    return delta_angle


def flux_surface_edge(
    eq: TokamakEquilibrium, north: SliceCoord, south: SliceCoord
) -> AcrossFieldCurve:
    if north.system != CoordinateSystem.CYLINDRICAL:
        raise ValueError("Coordinate system of `north` must be CYLINDRICAL")
    if south.system != CoordinateSystem.CYLINDRICAL:
        raise ValueError("Coordinate system of `south` must be CYLINDRICAL")

    psi_north = cast(float, eq.psi(north.x1, north.x2))
    psi_south = cast(float, eq.psi(south.x1, south.x2))
    if not np.isclose(psi_north, psi_south, 1e-10, 1e-10):
        raise ValueError(
            f"Start and end points {north} and {south} have different psi values "
            f"{psi_north} and {psi_south}"
        )

    start_angle = np.arctan2(north.x2 - eq.o_point.Z, north.x1 - eq.o_point.R)
    end_angle = np.arctan2(south.x2 - eq.o_point.Z, south.x1 - eq.o_point.R)
    delta_angle = smallest_angle_between(end_angle, start_angle)

    # Work out whether we need to reverse the direction of travel
    # along the flux surfaces
    sign = np.sign(delta_angle * eq.psi(north.x1, north.x2))

    def f(t: npt.NDArray, x: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
        dpsidR = eq.psi_func(x[0], x[1], dx=1, grid=False)
        dpsidZ = eq.psi_func(x[0], x[1], dy=1, grid=False)
        norm = np.sqrt(dpsidR * dpsidR + dpsidZ * dpsidZ)
        return -sign * dpsidZ / norm, sign * dpsidR / norm

    def terminus(t: float, x: npt.NDArray) -> float:
        return float(
            smallest_angle_between(
                end_angle, np.arctan2(x[1] - eq.o_point.Z, x[0] - eq.o_point.R)
            )
        )

    terminus.terminal = True  # type: ignore

    # Integrate until reaching the terminus, to work out the distance
    # along the contour to normalise with.
    result = solve_ivp(
        f,
        (0.0, 1e2),
        [north.x1, north.x2],
        method="DOP853",
        rtol=5e-14,
        atol=1e-14,
        events=terminus,
        vectorized=True,
    )
    if not result.success:
        raise RuntimeError("Failed to integrate along field line")
    if len(result.t_events[0]) == 0:
        raise RuntimeError("Integration did not cross over expected end-point")
    if not np.isclose(result.y[0, -1], south.x1, 1e-8, 1e-8) and not np.isclose(
        result.y[1, -1], south.x2, 1e-8, 1e-8
    ):
        raise RuntimeError("Integration did not converge on expected location")
    total_distance: float = result.t[-1]

    @integrate_vectorized(tuple(north), total_distance, {total_distance: tuple(south)})
    def solution(s: npt.ArrayLike, x: npt.ArrayLike) -> tuple[npt.NDArray, ...]:
        return f(np.asarray(s) * total_distance, np.asarray(x))

    def solution_coords(s: npt.ArrayLike) -> SliceCoords:
        s = np.asarray(s)
        R, Z = solution(s)
        return SliceCoords(
            R.reshape(s.shape), Z.reshape(s.shape), CoordinateSystem.CYLINDRICAL
        )

    return solution_coords


QuadMaker = Callable[[SliceCoord, SliceCoord], Quad]


def _make_bound(
    constructor: QuadMaker,
    region: MeshRegion,
    index: tuple[int | slice, ...],
) -> frozenset[Quad]:
    """Constructs the quads composing the boundary of `region`
    described by `index`.

    """
    R = region.Rxy.corners[index]
    Z = region.Zxy.corners[index]
    x = SliceCoords(R[:-1], Z[:-1], CoordinateSystem.CYLINDRICAL)
    y = SliceCoords(R[1:], Z[1:], CoordinateSystem.CYLINDRICAL)
    return frozenset(
        constructor(n, s) for n, s in zip(x.iter_points(), y.iter_points())
    )


def _get_region_boundaries(
    region: MeshRegion,
    flux_surface_quad: QuadMaker,
    perpendicular_quad: QuadMaker,
) -> list[frozenset[Quad]]:
    """Get a list of the boundaries for the region.

    Parameters
    ----------
    region
        The portion of the mesh for which to return boundaries.
    flux_surface_quad
        A function to produce an appropriate :class:`neso_fame.mesh.Quad`
        object for quads which are aligned to flux surfaces.
    perpendicular_quad
        A function to produce an appropriate :class:`neso_fame.mesh.Quad`
        object for quads which are perpendicular to flux surfaces.

    Returns
    -------
    A list of boundaries. Each boundary is represented by a frozenset
    of Quad objects. If that boundary is not present on the given
    region object, then the set will be empty. The order of the
    boundaries in the list is:

    #. Centre of the core region
    #. Inner edge of the plasma (or entire edge, for single-null)
    #. Outer edge of the plasma
    #. Internal edge of the upper private flux region
    #. Internal edge of hte lower private flux region
    #. Inner lower divertor
    #. Outer lower divertor
    #. Inner upper divertor
    #. Outer upper divertor

    """
    name = region.equilibriumRegion.name
    empty: frozenset[Quad] = frozenset()
    single_null = len(region.meshParent.equilibrium.x_points) == 1
    centre = (
        _make_bound(flux_surface_quad, region, (0, slice(None)))
        if name.endswith("core") and region.connections["inner"] is None
        else empty
    )
    inner_edge = (
        _make_bound(flux_surface_quad, region, (-1, slice(None)))
        if (
            name
            in {
                "inner_core",
                "core",
                "inner_lower_divertor",
                "inner_upper_divertor",
            }
            or (
                name in {"outer_lower_divertor", "outer_upper_divertor"} and single_null
            )
        )
        and region.connections["outer"] is None
        else empty
    )
    outer_edge = (
        _make_bound(flux_surface_quad, region, (-1, slice(None)))
        if name in {"outer_core", "outer_lower_divertor", "outer_upper_divertor"}
        and not single_null
        and region.connections["outer"] is None
        else empty
    )
    upper_pfr = (
        _make_bound(flux_surface_quad, region, (0, slice(None)))
        if name in {"inner_upper_divertor", "outer_upper_divertor"}
        and region.connections["inner"] is None
        else empty
    )
    lower_pfr = (
        _make_bound(flux_surface_quad, region, (0, slice(None)))
        if name in {"inner_lower_divertor", "outer_lower_divertor"}
        and region.connections["inner"] is None
        else empty
    )
    inner_lower_divertor = (
        _make_bound(perpendicular_quad, region, (slice(None), 0))
        if name == "inner_lower_divertor"
        else empty
    )
    outer_lower_divertor = (
        _make_bound(perpendicular_quad, region, (slice(None), 0))
        if name == "outer_lower_divertor"
        else empty
    )
    inner_upper_divertor = (
        _make_bound(perpendicular_quad, region, (slice(None), -1))
        if name == "inner_upper_divertor"
        else empty
    )
    outer_upper_divertor = (
        _make_bound(perpendicular_quad, region, (slice(None), -1))
        if name == "outer_upper_divertor"
        else empty
    )
    return [
        centre,
        inner_edge,
        outer_edge,
        upper_pfr,
        lower_pfr,
        inner_lower_divertor,
        outer_lower_divertor,
        inner_upper_divertor,
        outer_upper_divertor,
    ]


def get_mesh_boundaries(
    mesh: Mesh,
    flux_surface_quad: QuadMaker,
    perpendicular_quad: QuadMaker,
) -> list[frozenset[Quad]]:
    """Get a list of the boundaries for the meshj.

    Parameters
    ----------
    region
        The hypnotoad mesh object for which to return boundaries.
    flux_surface_quad
        A function to produce an appropriate :class:`neso_fame.mesh.Quad`
        object for quads which are aligned to flux surfaces.
    perpendicular_quad
        A function to produce an appropriate :class:`neso_fame.mesh.Quad`
        object for quads which are perpendicular to flux surfaces.

    Returns
    -------
    A list of boundaries. Each boundary is represented by a frozenset
    of Quad objects. If that boundary is not present on the given
    mesh object, then the set will be empty. The order of the
    boundaries in the list is:

    #. Centre of the core region
    #. Inner edge of the plasma (or entire edge, for single-null)
    #. Outer edge of the plasma
    #. Internal edge of the upper private flux region
    #. Internal edge of hte lower private flux region
    #. Inner lower divertor
    #. Outer lower divertor
    #. Inner upper divertor
    #. Outer upper divertor

    """

    def merge_bounds(
        lhs: list[frozenset[Quad]], rhs: list[frozenset[Quad]]
    ) -> list[frozenset[Quad]]:
        return [
            left.union(right)
            for left, right in itertools.zip_longest(
                lhs, rhs, fillvalue=cast(frozenset[Quad], frozenset())
            )
        ]

    boundaries = reduce(
        merge_bounds,
        (
            _get_region_boundaries(region, flux_surface_quad, perpendicular_quad)
            for region in mesh.regions.values()
        ),
    )
    return boundaries
