"""Functions for working with GEQDSK data processed using hypnotoad."""

import itertools
from enum import Enum
from functools import partial, reduce
from pathlib import Path
from typing import Any, Callable, Optional, cast

import numpy as np
import numpy.typing as npt
from hypnotoad import Mesh, MeshRegion  # type: ignore
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


def _process_integrate_vectorize_array_inputs(
    products: npt.NDArray,
    s_sorted: npt.NDArray,
    pos_fixed: dict[float, npt.NDArray],
    neg_fixed: dict[float, npt.NDArray],
) -> tuple[
    int | None,
    Optional[npt.NDArray],
    Optional[npt.NDArray],
    Optional[int | slice],
    Optional[int | slice],
    Optional[float],
    Optional[float],
    dict[int, npt.NDArray],
    dict[int, npt.NDArray],
]:
    pos_fixed_positions: dict[int, npt.NDArray] = {}
    neg_fixed_positions: dict[int, npt.NDArray] = {}
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
    return (
        zero,
        pos,
        neg,
        pos_slice,
        neg_slice,
        pos_limit,
        neg_limit,
        pos_fixed_positions,
        neg_fixed_positions,
    )


def _process_integrate_vectorize_inputs(
    s_sorted: npt.NDArray,
    pos_fixed: dict[float, npt.NDArray],
    neg_fixed: dict[float, npt.NDArray],
) -> tuple[
    int | None,
    Optional[npt.NDArray],
    Optional[npt.NDArray],
    Optional[int | slice],
    Optional[int | slice],
    Optional[float],
    Optional[float],
    dict[int, npt.NDArray],
    dict[int, npt.NDArray],
]:
    products = s_sorted[:-1] * s_sorted[1:]
    if len(products) > 0:
        return _process_integrate_vectorize_array_inputs(
            products, s_sorted, pos_fixed, neg_fixed
        )
    pos_fixed_positions: dict[int, npt.NDArray] = {}
    neg_fixed_positions: dict[int, npt.NDArray] = {}
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
    return (
        zero,
        pos,
        neg,
        pos_slice,
        neg_slice,
        pos_limit,
        neg_limit,
        pos_fixed_positions,
        neg_fixed_positions,
    )


def _process_fixed_points(
    start: npt.NDArray, fixed_points: dict[float, npt.ArrayLike]
) -> tuple[dict[float, npt.NDArray], dict[float, npt.NDArray]]:
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
    return pos_fixed, neg_fixed


def _handle_integration(
    func: Integrand,
    start: npt.NDArray,
    positions: Optional[npt.NDArray],
    limit: Optional[float],
    rtol: float,
    atol: float,
    vectorize_integrand_calls: bool,
    fixed_positions: dict[int, npt.NDArray],
    result_slice: Optional[int | slice],
    output: npt.NDArray,
) -> None:
    if positions is not None:
        result = solve_ivp(
            func,
            (0.0, limit),
            start,
            method="DOP853",
            t_eval=positions,
            rtol=rtol,
            atol=atol,
            dense_output=True,
            vectorized=vectorize_integrand_calls,
        )
        if not result.success:
            raise RuntimeError("Failed to integrate along field line")
        for i, v in fixed_positions.items():
            result.y[:, i] = v
            output[:, result_slice] = result.y


def integrate_vectorized(
    start: npt.ArrayLike,
    scale: float = 1.0,
    fixed_points: dict[float, npt.ArrayLike] = {},
    rtol: float = 1e-12,
    atol: float = 1e-14,
    vectorize_integrand_calls: bool = True,
) -> Callable[[Integrand], IntegratedFunction]:
    """Return a vectorised numerically-integrated function.

    Decorator that will numerically integrate a function and return
    a callable that is vectorised. The integration will start from
    t=0. You must specify the y-value at that starting point. You may
    optionally specify additional "fixed points" which should fall
    along the curve at chosen t-values. If someone requests a result
    at this position, the fixed-point will be returned instead. This
    can be useful if you need to enforce the end-point of a curve to
    within a very high level of precision.

    Group
    -----
    hypnotoad

    """
    start = np.asarray(start)
    if start.ndim > 1:
        raise ValueError("`start` must be 0- or 1-dimensional")
    pos_fixed, neg_fixed = _process_fixed_points(start, fixed_points)

    def wrap(func: Integrand) -> IntegratedFunction:
        def wrapper(s: npt.ArrayLike) -> tuple[npt.NDArray, ...]:
            s = np.asarray(s) * scale
            # Set very small values to 0 to avoid underflow problems
            if cast(npt.NDArray, s).ndim > 0:
                cast(npt.NDArray, s)[np.abs(s) < 1e-50] = 0.0
            s_sorted, invert_s = np.unique(s, return_inverse=True)
            (
                zero,
                pos,
                neg,
                pos_slice,
                neg_slice,
                pos_limit,
                neg_limit,
                pos_fixed_positions,
                neg_fixed_positions,
            ) = _process_integrate_vectorize_inputs(s_sorted, pos_fixed, neg_fixed)
            result_tmp = np.empty((cast(npt.NDArray, start).size, s_sorted.size))
            _handle_integration(
                func,
                start,
                neg,
                neg_limit,
                rtol,
                atol,
                vectorize_integrand_calls,
                neg_fixed_positions,
                neg_slice,
                result_tmp,
            )
            _handle_integration(
                func,
                start,
                pos,
                pos_limit,
                rtol,
                atol,
                vectorize_integrand_calls,
                pos_fixed_positions,
                pos_slice,
                result_tmp,
            )
            if zero is not None:
                result_tmp[:, zero] = start
            return tuple(
                result_tmp[i, invert_s].reshape(s.shape)
                for i in range(cast(npt.NDArray, start).size)
            )

        return wrapper

    return wrap


def _fpol(eq: TokamakEquilibrium, R: npt.NDArray, Z: npt.NDArray) -> npt.NDArray:
    return np.asarray(eq.f_spl(eq.psi_func(R, Z, grid=False) * eq.f_psi_sign))


def equilibrium_trace(equilibrium: TokamakEquilibrium) -> FieldTrace:
    """Return a field trace from the hypnotoad equilibrium object.

    Group
    -----
    hypnotoad

    """

    def trace(start: SliceCoord, phi: npt.ArrayLike) -> tuple[SliceCoords, npt.NDArray]:
        if start.system != CoordinateSystem.CYLINDRICAL:
            raise ValueError("`start` must use a cylindrical coordinate system")

        @integrate_vectorized([start.x1, start.x2, 0.0], rtol=1e-11, atol=1e-12)
        def integrated(_: npt.ArrayLike, y: npt.NDArray) -> tuple[npt.NDArray, ...]:
            R = y[0]
            Z = y[1]
            # Use low-level functions on equilibrium in order to avoid overheads
            inverse_B_tor = R / _fpol(equilibrium, R, Z)
            dR_dphi = equilibrium.psi_func(R, Z, dy=1, grid=False) * inverse_B_tor
            dZ_dphi = -equilibrium.psi_func(R, Z, dx=1, grid=False) * inverse_B_tor
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
    """Read in GEQDSK file and creates a hypnotoad TokamakEquilibrium object.

    Parameters
    ----------
    eqdsk
        The filename or path of the EQDSK file to read in.
    options
        Configurations for hypnotoad, see `Tokamak Options
        <https://hypnotoad.readthedocs.io/en/latest/_temp/options.html>`_
        and `Nonorthogonal Options
        <https://hypnotoad.readthedocs.io/en/latest/_temp/nonorthogonal-options.html>`_.

    Group
    -----
    hypnotoad

    """
    possible_options = list(TokamakEquilibrium.user_options_factory.defaults) + list(
        Mesh.user_options_factory.defaults
    )
    unused_options = [opt for opt in options if opt not in possible_options]
    if unused_options != []:
        raise ValueError(f"There are options that are not used: {unused_options}")
    with open(eqdsk, "r") as f:
        return read_geqdsk(f, options, options)


class XPointLocation(Enum):
    """Indicates if either end of a perpendicular edge ias an X-point."""

    NONE = 0
    NORTH = 1
    SOUTH = 2


def _handle_x_points(
    eq: TokamakEquilibrium, north: SliceCoord, south: SliceCoord
) -> tuple[
    XPointLocation, SliceCoord, SliceCoord, Callable[[npt.ArrayLike], npt.NDArray]
]:
    # TODO: Refactor so that this can be passed in. As we know where the
    # x-points lie within the hypnotoad regions, it would be more
    # efficient identify them in advance rather than have to check
    # every point.
    if any(
        np.isclose(north.x1, x.R, 1e-8, 1e-8) and np.isclose(north.x2, x.Z, 1e-8, 1e-8)
        for x in eq.x_points
    ):
        x_point = XPointLocation.NORTH
    elif any(
        np.isclose(south.x1, x.R, 1e-8, 1e-8) and np.isclose(south.x2, x.Z, 1e-8, 1e-8)
        for x in eq.x_points
    ):
        x_point = XPointLocation.SOUTH
    else:
        x_point = XPointLocation.NONE
    if x_point == XPointLocation.NORTH:
        start = south
        end = north

        def parameter(s: npt.ArrayLike) -> npt.NDArray:
            return 1 - np.asarray(s)
    else:
        start = north
        end = south

        def parameter(s: npt.ArrayLike) -> npt.NDArray:
            return np.asarray(s)

    return x_point, start, end, parameter


def _get_integration_distance(
    f: Callable[[npt.NDArray, npt.NDArray], tuple[npt.NDArray, ...]],
    terminus: Callable[[float, npt.NDArray], float],
    start: SliceCoord,
    end: SliceCoord,
) -> float:
    terminus.terminal = True  # type: ignore
    # Integrate until reaching the terminus, to work out the distance
    # along the contour to normalise with.
    result = solve_ivp(
        f,
        (0.0, 1e2),
        [start.x1, start.x2],
        method="DOP853",
        rtol=5e-14,
        atol=1e-14,
        events=terminus,
        vectorized=True,
    )
    # FIXME: This is diverging from the flux surface in some situations, somehow.
    if not result.success:
        raise RuntimeError("Integration failed")
    if len(result.t_events[0]) == 0:
        raise RuntimeError("Integration did not cross over expected end-point")
    if not np.isclose(result.y[0, -1], end.x1, 1e-5, 1e-5) and not np.isclose(
        result.y[1, -1], end.x2, 1e-5, 1e-5
    ):
        raise RuntimeError("Integration did not converge on expected location")
    total_distance: float = result.t[-1]
    return total_distance


def perpendicular_edge(
    eq: TokamakEquilibrium, north: SliceCoord, south: SliceCoord
) -> AcrossFieldCurve:
    """Return a line traveling at right angles to the magnetic field.

    This line will connect the two points on the poloidal plane. If
    this is not possible, raise an exception.

    Parameters
    ----------
    eq
        A hypnotaod Equilibrium object describing the magnetic field
    north
        The starting point of the line
    south
        The end point of the line

    Group
    -----
    hypnotoad

    """
    if north.system != CoordinateSystem.CYLINDRICAL:
        raise ValueError("Coordinate system of `north` must be CYLINDRICAL")
    if south.system != CoordinateSystem.CYLINDRICAL:
        raise ValueError("Coordinate system of `south` must be CYLINDRICAL")

    x_point, start, end, parameter = _handle_x_points(eq, north, south)
    psi_start = float(eq.psi_func(start.x1, start.x2, grid=False))
    psi_end = float(eq.psi_func(end.x1, end.x2, grid=False))
    sign = np.sign(psi_end - psi_start)

    def f(t: npt.NDArray, x: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
        dpsidR = eq.psi_func(x[0], x[1], dx=1, grid=False)
        dpsidZ = eq.psi_func(x[0], x[1], dy=1, grid=False)
        norm = np.sqrt(dpsidR * dpsidR + dpsidZ * dpsidZ)
        return sign * dpsidR / norm, sign * dpsidZ / norm

    def terminus(t: float, x: npt.NDArray) -> float:
        return float(eq.psi(x[0], x[1])) - psi_end

    total_distance = _get_integration_distance(f, terminus, start, end)

    @integrate_vectorized(tuple(start), total_distance, {total_distance: tuple(end)})
    def solution(s: npt.ArrayLike, x: npt.ArrayLike) -> tuple[npt.NDArray, ...]:
        return f(np.asarray(s) * total_distance, np.asarray(x))

    def solution_coords(s: npt.ArrayLike) -> SliceCoords:
        s = parameter(s)
        # Hypnotoad's perpendicular surfaces near the x-point aren't
        # entirely accurate, due to numerically difficulty if you get
        # too close. As a result, if north or south are located at the
        # x-point, we won't actually manage to integrate to be
        # sufficiently close to them. To get around this, the solution
        # is approximated as a linear combination of the integration
        # and a straight line between north and south. The weight of
        # the linaer component increases as the x-point is approached.
        R_sol, Z_sol = solution(s)
        if x_point == XPointLocation.NONE:
            R = R_sol
            Z = Z_sol
        else:
            m = s * s
            R_lin = start.x1 * (1 - s) + end.x1 * s
            Z_lin = start.x2 * (1 - s) + end.x2 * s
            R = R_sol * (1 - m) + m * R_lin
            Z = Z_sol * (1 - m) + m * Z_lin
        return SliceCoords(
            R.reshape(s.shape), Z.reshape(s.shape), CoordinateSystem.CYLINDRICAL
        )

    return solution_coords


@np.vectorize
def _smallest_angle_between(end_angle: float, start_angle: float) -> float:
    delta_angle = end_angle - start_angle
    # Deal with cases where the angles stradle the atan2 discontinuity
    if delta_angle < -np.pi:
        return delta_angle + 2 * np.pi
    elif delta_angle > np.pi:
        return delta_angle - 2 * np.pi
    return delta_angle


def _dpsi(eq: TokamakEquilibrium, x: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    dpsidR = eq.psi_func(x[0], x[1], dx=1, grid=False)
    dpsidZ = eq.psi_func(x[0], x[1], dy=1, grid=False)
    return dpsidR, dpsidZ


def _determine_integration_direction(
    dpsi: Callable[[npt.NDArray], tuple[npt.NDArray, npt.NDArray]],
    north: SliceCoord,
    south: SliceCoord,
) -> tuple[SliceCoord, SliceCoord, Callable[[npt.ArrayLike], npt.NDArray]]:
    # Start from the location with the stronger poloidal magnetic
    # field (as this will be farther from an x-point and thus less
    # prone to numerical error)
    if sum(x * x for x in dpsi(np.array(list(north)))) > sum(
        x * x for x in dpsi(np.array(list(south)))
    ):
        start = north
        end = south

        def parameter(s: npt.ArrayLike) -> npt.NDArray:
            return np.asarray(s)
    else:
        start = south
        end = north

        def parameter(s: npt.ArrayLike) -> npt.NDArray:
            return 1 - np.asarray(s)

    return start, end, parameter


def flux_surface_edge(
    eq: TokamakEquilibrium, north: SliceCoord, south: SliceCoord
) -> AcrossFieldCurve:
    """Return a line following a flux surface.

    This line will connect the two points on the poloidal plane. If
    this is not possible, raise an exception.

    .. warning::
        If the points are seperated by a large angle along the flux
        surface, there is a chance that the integration may fail. This
        won't normally be a problem, as hypnotoad generates meshes
        with points that are relatively close together, but it can be
        if choosing arbitrary points.

    Parameters
    ----------
    eq
        A hypnotaod Equilibrium object describing the magnetic field
    north
        The starting point of the line
    south
        The end point of the line

    Group
    -----
    hypnotoad

    """
    if north.system != CoordinateSystem.CYLINDRICAL:
        raise ValueError("Coordinate system of `north` must be CYLINDRICAL")
    if south.system != CoordinateSystem.CYLINDRICAL:
        raise ValueError("Coordinate system of `south` must be CYLINDRICAL")

    # FIXME: I don't think this will be able to handle the inexact
    # seperatrix you will get when doing a realistic connected double
    # null
    psi_north = cast(float, eq.psi(north.x1, north.x2))
    psi_south = cast(float, eq.psi(south.x1, south.x2))
    if not np.isclose(psi_north, psi_south, 1e-10, 1e-10):
        raise ValueError(
            f"Start and end points {north} and {south} have different psi values "
            f"{psi_north} and {psi_south}"
        )

    dpsi = partial(_dpsi, eq)

    # Work out whether we need to reverse the direction of travel
    # along the flux surfaces
    start, end, parameter = _determine_integration_direction(dpsi, north, south)

    def surface(x: npt.NDArray) -> npt.NDArray:
        dpsidR, dpsidZ = dpsi(x)
        norm = np.sqrt(dpsidR * dpsidR + dpsidZ * dpsidZ)
        return np.array([-dpsidZ / norm, dpsidR / norm])

    direction = surface(np.array(list(start)))
    sign = float(
        np.sign(direction[0] * (end.x1 - start.x1) + direction[1] * (end.x2 - start.x2))
    )

    def f(t: npt.NDArray, x: npt.NDArray) -> tuple[npt.NDArray]:
        return (sign * surface(x),)

    end_orthogonal = (
        eq.psi_func(end.x1, end.x2, dx=1, grid=False),
        eq.psi_func(end.x1, end.x2, dy=1, grid=False),
    )

    def terminus(t: float, x: npt.NDArray) -> float:
        # Use the cross-product of the vector between the point and
        # the target and the vector orthogonal to the flux surfaces at
        # the target to determine which side of the target we are on
        sign = np.sign(
            end_orthogonal[0] * (x[1] - end.x2) - end_orthogonal[1] * (x[0] - end.x1)
        )
        return float(sign * np.sqrt((x[0] - end.x1) ** 2 + (x[1] - end.x2) ** 2))

    total_distance = _get_integration_distance(f, terminus, start, end)

    @integrate_vectorized(tuple(start), total_distance, {total_distance: tuple(end)})
    def solution(s: npt.ArrayLike, x: npt.ArrayLike) -> tuple[npt.NDArray, ...]:
        return f(np.asarray(s) * total_distance, np.asarray(x))

    def solution_coords(s: npt.ArrayLike) -> SliceCoords:
        s = parameter(s)
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
    """Construct quads for boundary `index` of `region`."""
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
    """Get a list of the boundaries for the mesh.

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

    Group
    -----
    hypnotoad

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
