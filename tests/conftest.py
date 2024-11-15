import itertools
import operator
from collections.abc import Sequence
from functools import cache
from typing import Any, Optional, TypeVar, cast

import numpy as np
import numpy.typing as npt
from hypothesis import Verbosity, settings
from hypothesis.extra.numpy import (
    BroadcastableShapes,
    array_shapes,
    arrays,
    broadcastable_shapes,
    floating_dtypes,
    mutually_broadcastable_shapes,
)
from hypothesis.strategies import (
    SearchStrategy,
    booleans,
    builds,
    composite,
    floats,
    from_type,
    frozensets,
    integers,
    just,
    lists,
    one_of,
    register_type_strategy,
    sampled_from,
    shared,
    tuples,
)
from scipy.special import ellipeinc

from neso_fame import coordinates, mesh
from neso_fame.offset import Offset

settings.register_profile("ci", max_examples=200, deadline=None)
settings.register_profile(
    "debug", max_examples=10, verbosity=Verbosity.verbose, report_multiple_bugs=False
)
settings.register_profile("dev", max_examples=10)


Pair = tuple[float, float]


def non_nans() -> SearchStrategy[float]:
    return floats(allow_nan=False, allow_infinity=False)


def arbitrary_arrays() -> SearchStrategy[npt.NDArray]:
    return arrays(floating_dtypes(), array_shapes())


WHOLE_NUM_MAX = 1000
whole_numbers = integers(-WHOLE_NUM_MAX, WHOLE_NUM_MAX).map(float)
nonnegative_numbers = integers(1, WHOLE_NUM_MAX).map(float)
non_zero = whole_numbers.filter(bool)
orders = integers(1, 5)

# Large triangles can confuse Nektar++ due to numerical error
# exceeding some (very small) tolerances. Use smaller numbers for
# them.
SMALL_WHOLE_NUM_MAX = 2
small_whole_numbers = integers(-SMALL_WHOLE_NUM_MAX * 50, SMALL_WHOLE_NUM_MAX * 50).map(
    lambda x: x / 50
)
small_nonnegative_numbers = integers(1, SMALL_WHOLE_NUM_MAX * 50).map(lambda x: x / 50)
small_non_zero = small_whole_numbers.filter(bool)


def mutually_broadcastable_arrays(
    num_arrays: int,
) -> SearchStrategy[tuple[npt.NDArray, ...]]:
    return mutually_broadcastable_from(
        mutually_broadcastable_shapes(num_shapes=num_arrays)
    )


def mutually_broadcastable_from(
    strategy: SearchStrategy[BroadcastableShapes],
) -> SearchStrategy[tuple[npt.NDArray]]:
    def shape_to_array(
        shapes: BroadcastableShapes,
    ) -> SearchStrategy[tuple[npt.NDArray]]:
        return tuples(
            *(
                cast(
                    SearchStrategy[npt.NDArray],
                    arrays(np.float64, just(s), elements=whole_numbers),
                )
                for s in shapes.input_shapes
            )
        )

    return strategy.flatmap(shape_to_array)


coordinate_systems = sampled_from(coordinates.CoordinateSystem)
coordinate_systems3d = coordinate_systems.filter(
    lambda x: x != coordinates.CoordinateSystem.CARTESIAN2D
)


def slice_coord_for_system(
    system: coordinates.CoordinateSystem,
) -> SearchStrategy[coordinates.SliceCoord]:
    x1 = (
        non_zero
        if system == coordinates.CoordinateSystem.CYLINDRICAL
        else whole_numbers
    )
    return builds(
        coordinates.SliceCoord,
        x1,
        whole_numbers,
        just(system),
    )


def coord_for_system(
    system: coordinates.CoordinateSystem,
) -> SearchStrategy[coordinates.Coord]:
    x1 = (
        non_zero
        if system == coordinates.CoordinateSystem.CYLINDRICAL
        else whole_numbers
    )
    return builds(
        coordinates.Coord,
        x1,
        whole_numbers,
        whole_numbers,
        just(system),
    )


register_type_strategy(
    coordinates.SliceCoords,
    builds(
        lambda xs, c: coordinates.SliceCoords(
            np.abs(xs[0]) if c == coordinates.CoordinateSystem.CYLINDRICAL else xs[0],
            xs[1],
            c,
        ),
        mutually_broadcastable_arrays(2),
        sampled_from(coordinates.CoordinateSystem),
    ),
)

register_type_strategy(
    coordinates.SliceCoord,
    sampled_from(coordinates.CoordinateSystem).flatmap(slice_coord_for_system),
)

register_type_strategy(
    coordinates.Coords,
    builds(
        lambda xs, c: coordinates.Coords(xs[0], xs[1], xs[2], c),
        mutually_broadcastable_arrays(3),
        sampled_from(coordinates.CoordinateSystem),
    ),
)


register_type_strategy(
    coordinates.Coord,
    builds(
        coordinates.Coord,
        whole_numbers,
        whole_numbers,
        whole_numbers,
        sampled_from(coordinates.CoordinateSystem),
    ),
)


def compatible_alignments(
    coords: coordinates.SliceCoords,
) -> SearchStrategy[npt.NDArray]:
    shape = np.broadcast(*coords).shape
    return arrays(
        float,
        broadcastable_shapes(shape, max_dims=len(shape)),
        elements=floats(-1.0, 0.0).map(lambda y: y + 1),
        fill=just(1.0),
    )


compatible_coords_and_alignments = from_type(coordinates.SliceCoords).flatmap(
    lambda x: tuples(just(x), compatible_alignments(x))
)


num_divs = shared(integers(1, 10), key=999)


def straight_line_for_system(
    system: coordinates.CoordinateSystem,
) -> SearchStrategy[mesh.FieldAlignedCurve]:
    coords = builds(
        coordinates.Coord, whole_numbers, whole_numbers, whole_numbers, just(system)
    )
    return builds(
        mesh.straight_line,
        coords,
        coords,
    )


CARTESIAN_SYSTEMS = {
    coordinates.CoordinateSystem.CARTESIAN,
    coordinates.CoordinateSystem.CARTESIAN2D,
    coordinates.CoordinateSystem.CARTESIAN_ROTATED,
}


def linear_field_trace(
    a1: float,
    a2: float,
    a3: float,
    c: coordinates.CoordinateSystem,
    skew: float,
    centre: Pair,
) -> mesh.FieldTrace:
    a1p = a1 / a3 if c in CARTESIAN_SYSTEMS else 0.0
    a2p = a2 / a3 if c != coordinates.CoordinateSystem.CARTESIAN2D else 0.0

    def cartesian_func(
        start: coordinates.SliceCoord, x3: npt.ArrayLike, start_weight: float = 0.0
    ) -> tuple[coordinates.SliceCoords, npt.NDArray]:
        offset = np.sqrt((start.x1 - centre[0]) ** 2 + (start.x2 - centre[1]) ** 2)
        t = 1 - start_weight
        t2 = t * t
        b1 = a1p * (1 + skew * offset)
        b2 = a2p * (1 + skew * offset)
        if c in CARTESIAN_SYSTEMS:
            s = np.sqrt(t2 * b1 * b1 + t2 * b2 * b2 + 1) * np.asarray(x3)
        else:
            s = np.sqrt(t2 * b1 * b1 + t2 * b2 * b2 + start.x1 * start.x1) * np.asarray(
                x3
            )
        return (
            coordinates.SliceCoords(
                t * b1 * np.asarray(x3) + start.x1,
                t * b2 * np.asarray(x3) + start.x2,
                start.system,
            ),
            s,
        )

    return cartesian_func


linear_traces = builds(
    linear_field_trace,
    whole_numbers,
    whole_numbers,
    non_zero,
    coordinate_systems,
    floats(-2.0, 2.0),
    tuples(whole_numbers, whole_numbers),
)


@composite
def unbroadcastable_shape(draw: Any, shape: Sequence[int]) -> tuple[int, ...]:
    if len(shape) == 0:
        raise ValueError("0-length arrays are broadcastable with all array shapes")
    idx = draw(integers(0, len(shape) - 1))
    initial = shape[idx]
    new_val = draw(integers(2, 10).filter(lambda x: x != initial))
    tmp = list(shape)
    tmp[idx] = new_val
    return tuple(tmp)


def linear_field_line(
    order: int,
    a1: float,
    a2: float,
    a3: float,
    b1: float,
    b2: float,
    b3: float,
    c: coordinates.CoordinateSystem,
    skew: float,
    centre: Pair,
) -> mesh.Curve:
    offset = np.sqrt((b1 - centre[0]) ** 2 + (b2 - centre[1]) ** 2)

    def linear_func(x: npt.ArrayLike) -> coordinates.Coords:
        a = a1 if c in CARTESIAN_SYSTEMS else 0.0
        return coordinates.Coords(
            a * (1 + skew * offset) * np.asarray(x) + b1 - 0.5 * a,
            a2 * (1 + skew * offset) * np.asarray(x) + b2 - 0.5 * a2,
            a3 * np.asarray(x) + b3 - 0.5 * a3,
            c,
        )

    return mesh.Curve(linear_func(np.linspace(0.0, 1.0, order + 1)))


def linear_field_aligned_curve(
    order: int,
    a1: float,
    a2: float,
    a3: float,
    start: Pair,
    c: coordinates.CoordinateSystem,
    skew: float,
    division: int,
    num_divisions: int,
    start_weight: float,
    offset: float,
) -> mesh.FieldAlignedCurve:
    return Offset(
        mesh.FieldAlignedCurve(
            mesh.field_aligned_positions(
                mesh.SliceCoords(np.array(start[0]), np.array(start[1]), c),
                a3,
                linear_field_trace(a1, a2, a3, c, skew, start),
                np.array(start_weight),
                order,
                division,
                num_divisions,
            )
        ),
        offset,
    )


def trapezoidal_quad(
    order: int,
    a1: float,
    a2: float,
    a3: float,
    starts: tuple[Pair, Pair],
    c: coordinates.CoordinateSystem,
    skew: float,
    division: int,
    num_divisions: int,
    north_start_weight: float,
    south_start_weight: float,
    offset: float,
) -> Optional[mesh.Quad]:
    centre = (
        starts[0][0] + (starts[0][0] - starts[1][0]) / 2,
        starts[0][1] + (starts[0][1] - starts[1][1]) / 2,
    )
    if c == coordinates.CoordinateSystem.CYLINDRICAL and (
        starts[0][0] * starts[1][0] <= 0.0
    ):
        return None
    shape = mesh.straight_line_across_field(
        coordinates.SliceCoord(starts[0][0], starts[0][1], c),
        coordinates.SliceCoord(starts[1][0], starts[1][1], c),
        order,
    )
    trace = linear_field_trace(a1, a2, a3, c, skew, centre)
    return Offset(
        mesh.Quad(
            mesh.field_aligned_positions(
                shape,
                a3,
                trace,
                np.linspace(north_start_weight, south_start_weight, order + 1),
                order,
                division,
                num_divisions,
            )
        ),
        offset,
    )


@cache
def quad_weights(
    order: int,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    s = np.linspace(0.0, 1.0, order + 1)
    s1, s2 = np.meshgrid(s, s, copy=False, sparse=True)
    return (1.0 - s1) * (1.0 - s2), (1.0 - s1) * s2, s1 * s2, s1 * (1.0 - s2)


def corners_to_poloidal_quad(
    order: int, corners: tuple[Pair, Pair, Pair, Pair], c: coordinates.CoordinateSystem
) -> coordinates.SliceCoords:
    sorted_corners = sorted(corners, key=operator.itemgetter(1))
    north, east, south, west = sorted(
        sorted_corners[0:2], key=operator.itemgetter(0)
    ) + sorted(sorted_corners[2:4], key=operator.itemgetter(0), reverse=True)
    ns, es, ss, ws = quad_weights(order)
    return coordinates.SliceCoords(
        north[0] * ns + east[0] * es + south[0] * ss + west[0] * ws,
        north[1] * ns + east[1] * es + south[1] * ss + west[1] * ws,
        c,
    )


@cache
def tri_weights(order: int) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    s = np.linspace(0.0, 1.0, order + 1)
    s1, s2 = np.meshgrid(s, s, copy=False, sparse=False)
    x1 = np.empty((order + 1, order + 1))
    # Need to avoid computing the singularity at the end
    x1[:-1, :] = s1[:-1, :] / (1 - s2[:-1, :])
    x1[-1, 0] = 1
    x1[-1, 1:] = 1.1
    return (1 - x1) * (1 - s2), x1 * (1 - s2), s2


def corners_to_poloidal_tri(
    order: int, corners: tuple[Pair, Pair, Pair], c: coordinates.CoordinateSystem
) -> coordinates.SliceCoords:
    north, east, south = corners
    ns, es, ss = tri_weights(order)
    return coordinates.SliceCoords(
        north[0] * ns + east[0] * es + south[0] * ss,
        north[1] * ns + east[1] * es + south[1] * ss,
        c,
    )


def end_quad(
    order: int,
    corners: tuple[Pair, Pair, Pair, Pair],
    c: coordinates.CoordinateSystem,
    x3: float,
) -> Optional[mesh.UnalignedShape]:
    if c == coordinates.CoordinateSystem.CYLINDRICAL and (
        0.0 in [c[0] for c in corners]
    ):
        return None
    return mesh.UnalignedShape(
        mesh.PrismTypes.RECTANGULAR,
        corners_to_poloidal_quad(
            order,
            corners,
            c,
        ).to_3d_coords(x3),
    )


def end_triangle(
    order: int,
    corners: tuple[Pair, Pair, Pair],
    c: coordinates.CoordinateSystem,
    x3: float,
) -> Optional[mesh.UnalignedShape]:
    if c == coordinates.CoordinateSystem.CYLINDRICAL and (
        0.0 in [c[0] for c in corners]
    ):
        return None
    return mesh.UnalignedShape(
        mesh.PrismTypes.TRIANGULAR,
        corners_to_poloidal_tri(order, corners, c).to_3d_coords(x3),
    )


def _get_start_weight(fixed: bool) -> float:
    if fixed:
        return 1
    return 0


def trapezohedronal_hex(
    order: int,
    a1: float,
    a2: float,
    a3: float,
    starts: tuple[Pair, Pair, Pair, Pair],
    c: coordinates.CoordinateSystem,
    skew: float,
    division: int,
    num_divisions: int,
    offset: float,
    fixed_edges: tuple[bool, bool, bool, bool],
) -> Optional[mesh.Prism]:
    centre = (
        sum(map(operator.itemgetter(0), starts)),
        sum(map(operator.itemgetter(1), starts)),
    )
    if c == coordinates.CoordinateSystem.CYLINDRICAL and any(
        a * b <= 0.0 for a, b in itertools.combinations((p[0] for p in starts), 2)
    ):
        return None
    trace = linear_field_trace(a1, a2, a3, c, skew, centre)
    ns, es, ss, ws = quad_weights(order)
    fn, fe, fs, fw = map(_get_start_weight, fixed_edges)

    return Offset(
        mesh.Prism(
            mesh.PrismTypes.RECTANGULAR,
            mesh.field_aligned_positions(
                corners_to_poloidal_quad(order, starts, c),
                a3,
                trace,
                0.25 * (fn * ns + fs * ss + fe * es + fw * ws),
                order,
                division,
                num_divisions,
            ),
        ),
        offset,
    )


def simple_prism(
    order: int,
    a1: float,
    a2: float,
    a3: float,
    starts: tuple[Pair, Pair, Pair],
    c: coordinates.CoordinateSystem,
    skew: float,
    division: int,
    num_divisions: int,
    offset: float,
    fixed_edges: tuple[bool, bool, bool],
) -> Optional[mesh.Prism]:
    centre = (
        sum(map(operator.itemgetter(0), starts)),
        sum(map(operator.itemgetter(1), starts)),
    )
    if c == coordinates.CoordinateSystem.CYLINDRICAL and any(
        a * b <= 0.0 for a, b in itertools.combinations((p[0] for p in starts), 2)
    ):
        return None
    trace = linear_field_trace(a1, a2, a3, c, skew, centre)
    ns, es, ss = tri_weights(order)
    fn, fe, fs = map(_get_start_weight, fixed_edges)

    return Offset(
        mesh.Prism(
            mesh.PrismTypes.TRIANGULAR,
            mesh.field_aligned_positions(
                corners_to_poloidal_tri(order, starts, c),
                a3,
                trace,
                (fn * ns + fs * ss + fe * es) / 3,
                order,
                division,
                num_divisions,
            ),
        ),
        offset,
    )


def cylindrical_field_trace(centre: float, x2_slope: float) -> mesh.FieldTrace:
    def cylindrical_func(
        start: coordinates.SliceCoord, x3: npt.ArrayLike, start_weight: float = 0.0
    ) -> tuple[coordinates.SliceCoords, npt.NDArray]:
        assert start.system in CARTESIAN_SYSTEMS
        b = 1 - start_weight
        rad = start.x1 - centre
        sign = np.sign(rad)
        x3 = np.asarray(x3)
        x1 = (
            b * (centre + sign * np.sqrt(rad * rad - x3 * x3)) + start_weight * start.x1
        )
        dtheta = np.arcsin(x3 / rad)
        x2 = b * (np.asarray(start.x2) + x2_slope * dtheta) + start_weight * start.x2
        alpha = 1 + b * b * x2_slope * x2_slope
        return (
            coordinates.SliceCoords(x1, x2, start.system),
            np.sqrt(alpha) * ellipeinc(dtheta, (b * b - 1) / alpha) * rad,
        )

    return cylindrical_func


def cylindrical_field_line(
    order: int,
    centre: float,
    x1_start: float,
    x2_centre: float,
    x2_slope: float,
    x3_limits: Pair,
    system: coordinates.CoordinateSystem,
) -> mesh.Curve:
    assert system in CARTESIAN_SYSTEMS
    rad = x1_start - centre
    half_x3_sep = 0.5 * (x3_limits[1] - x3_limits[0])
    x3_centre = x3_limits[0] + half_x3_sep
    theta_max = 2 * np.arcsin(half_x3_sep / rad)

    def cylindrical_func(s: npt.ArrayLike) -> coordinates.Coords:
        theta = theta_max * (np.asarray(s) - 0.5)
        x1 = centre + rad * np.cos(theta)
        x2 = x2_centre + x2_slope * theta
        x3 = x3_centre + rad * np.sin(theta)
        return coordinates.Coords(x1, x2, x3, system)

    return mesh.Curve(cylindrical_func(np.linspace(0.0, 1.0, order + 1)))


def curved_quad(
    order: int,
    centre: float,
    x1_start: Pair,
    x2_centre: float,
    x2_slope: float,
    dx3: float,
    system: coordinates.CoordinateSystem,
    division: int,
    num_divisions: int,
    north_start_weight: float,
    south_start_weight: float,
    offset: float,
) -> mesh.Quad:
    trace = cylindrical_field_trace(centre, x2_slope)
    shape = mesh.straight_line_across_field(
        coordinates.SliceCoord(x1_start[0], x2_centre, system),
        coordinates.SliceCoord(x1_start[1], x2_centre, system),
        order,
    )
    return Offset(
        mesh.Quad(
            mesh.field_aligned_positions(
                shape,
                dx3,
                trace,
                np.linspace(north_start_weight, south_start_weight, order + 1),
                order,
                division,
                num_divisions,
            )
        ),
        offset,
    )


def curved_hex(
    order: int,
    centre: float,
    starts: tuple[Pair, Pair, Pair, Pair],
    x2_slope: float,
    dx3: float,
    system: coordinates.CoordinateSystem,
    division: int,
    num_divisions: int,
    offset: float,
    fixed_edges: tuple[bool, bool, bool, bool],
) -> mesh.Prism:
    trace = cylindrical_field_trace(centre, x2_slope)
    ns, es, ss, ws = quad_weights(order)
    fn, fe, fs, fw = map(_get_start_weight, fixed_edges)

    return Offset(
        mesh.Prism(
            mesh.PrismTypes.RECTANGULAR,
            mesh.field_aligned_positions(
                corners_to_poloidal_quad(order, starts, system),
                dx3,
                trace,
                0.25 * (fn * ns + fs * ss + fe * es + fw * ws),
                order,
                division,
                num_divisions,
            ),
        ),
        offset,
    )


def curved_prism(
    order: int,
    centre: float,
    starts: tuple[Pair, Pair, Pair],
    x2_slope: float,
    dx3: float,
    system: coordinates.CoordinateSystem,
    division: int,
    num_divisions: int,
    offset: float,
    fixed_edges: tuple[bool, bool, bool],
) -> mesh.Prism:
    trace = cylindrical_field_trace(centre, x2_slope)
    ns, es, ss = tri_weights(order)
    fn, fe, fs = map(_get_start_weight, fixed_edges)

    return Offset(
        mesh.Prism(
            mesh.PrismTypes.TRIANGULAR,
            mesh.field_aligned_positions(
                corners_to_poloidal_tri(order, starts, system),
                dx3,
                trace,
                (fn * ns + fs * ss + fe * es) / 3,
                order,
                division,
                num_divisions,
            ),
        ),
        offset,
    )


def make_arc(
    order: int,
    north: coordinates.SliceCoord,
    south: coordinates.SliceCoord,
    angle: float,
) -> mesh.AcrossFieldCurve:
    x1_1 = north.x1
    x2_1 = north.x2
    x1_2 = south.x1
    x2_2 = south.x2
    dx1 = x1_1 - x1_2
    dx2 = x2_1 - x2_2
    denom = 2 * dx1 * np.cos(angle) + 2 * dx2 * np.sin(angle)
    if abs(denom) < 1e-2:
        raise ValueError("Radius of circle too large.")
    r = (dx1 * dx1 + dx2 * dx2) / denom
    x1_c = x1_1 - r * np.cos(angle)
    x2_c = x2_1 - r * np.sin(angle)
    a = np.arctan2(x2_2 - x2_c, x1_2 - x1_c) - angle

    def curve(s: npt.ArrayLike) -> coordinates.SliceCoords:
        s = np.asarray(s)
        return coordinates.SliceCoords(
            x1_c + r * np.cos(angle + a * s),
            x2_c + r * np.sin(angle + a * s),
            north.system,
        )

    return mesh.AcrossFieldCurve(curve(np.linspace(0.0, 1.0, order + 1)))


def offset_line(line: mesh.AcrossFieldCurve, magnitude: float) -> mesh.AcrossFieldCurve:
    if magnitude == 0:
        return line
    north = line[0]
    south = line[-1]
    dx1 = north.x1 - south.x1
    dx2 = north.x2 - south.x2
    norm = np.sqrt(dx1 * dx1 + dx2 * dx2)
    perp = [dx2 / norm, -dx1 / norm]
    s = np.linspace(0.0, 1.0, len(line))
    t = 4 * magnitude * s * (1 - s)
    return mesh.AcrossFieldCurve(
        coordinates.SliceCoords(
            line.x1 + t * perp[0], line.x2 + t * perp[1], line.system
        )
    )


def higher_dim_quad(order: int, q: mesh.Quad, angle: float) -> Optional[mesh.Quad]:
    # This assumes that dx3/ds is an even function about the starting
    # x3 point from which the bounding field lines were projected
    try:
        curve = make_arc(
            order, q.nodes.start_points[0], q.nodes.start_points[-1], angle
        )
    except ValueError:
        return None
    x3 = q.x3_offset
    q = q.get_underlying_object()
    return Offset(
        mesh.Quad(
            mesh.field_aligned_positions(
                curve,
                q.nodes.x3[-1] - q.nodes.x3[0],
                q.nodes.trace,
                q.nodes.alignments,
                order,
                q.nodes.subdivision,
                q.nodes.num_divisions,
            ),
        ),
        x3,
    )


def _offset(
    x1: npt.NDArray,
    x2: npt.NDArray,
    north: tuple[int, int],
    south: tuple[int, int],
    weights: npt.NDArray,
    magnitude: float,
) -> None:
    dx1 = x1[north] - x1[south]
    dx2 = x2[north] - x2[south]
    norm = np.sqrt(dx1 * dx1 + dx2 * dx2)
    perp = [dx2 / norm, -dx1 / norm]
    s = np.linspace(0.0, 1.0, len(weights)).reshape(weights.shape)
    t = 4 * magnitude * s * (1 - s)
    x1 += t * perp[0]
    x2 += t * perp[1]


def higher_dim_hex(h: mesh.Prism, magnitudes: list[float]) -> mesh.Prism:
    order = h.nodes.start_points.shape[0]
    s = np.linspace(0.0, 1.0, order + 1)
    s1, s2 = np.meshgrid(s, s, copy=False, sparse=True)
    x1, x2 = h.nodes.start_points
    if h.shape == mesh.PrismTypes.RECTANGULAR:
        _offset(x1, x2, (0, 0), (0, -1), 1 - s2, magnitudes[0])
        _offset(x1, x2, (-1, 0), (-1, -1), s2, magnitudes[1])
        _offset(x1, x2, (0, 0), (-1, 0), 1 - s1, magnitudes[2])
        _offset(x1, x2, (0, -1), (-1, -1), s1, magnitudes[3])
    else:
        # This doesn't seem to match what I used to do with
        # _poloidal_map_tri, but that looks like it was wrong.
        _offset(x1, x2, (0, 0), (-1, 0), 1 - s1, magnitudes[0])
        _offset(x1, x2, (0, -1), (0, -1), s1, magnitudes[1])
    return mesh.Prism(
        h.shape,
        mesh.field_aligned_positions(
            mesh.SliceCoords(x1, x2, h.nodes.start_points.system),
            h.nodes.x3[-1] - h.nodes.x3[0],
            h.nodes.trace,
            h.nodes.alignments,
            order,
            h.nodes.subdivision,
            h.nodes.num_divisions,
        ),
    )


def _quad_mesh_elements(
    order: int,
    a1: float,
    a2: float,
    a3: float,
    limits: tuple[Pair, Pair],
    num_quads: int,
    c: coordinates.CoordinateSystem,
    left_fixed: bool,
    right_fixed: bool,
    layers: int = 1,
) -> Optional[list[mesh.Quad]]:
    trace = linear_field_trace(a1, a2, a3, c, 0, (0, 0))
    if (
        c == coordinates.CoordinateSystem.CYLINDRICAL
        and limits[0][0] * limits[1][0] <= 0
    ):
        return None

    points = np.linspace(limits[0], limits[1], num_quads * order + 1)
    positions = mesh.subdividable_field_aligned_positions(
        coordinates.SliceCoords(points[:, 0], points[:, 1], c),
        a3,
        trace,
        np.array(
            [0.0 if left_fixed else 1.0]
            + [1.0] * (num_quads * order - 1)
            + [0.0 if right_fixed else 1.0]
        ),
        order,
        layers,
    )

    return [
        mesh.Quad(positions[i * order : (i + 1) * order + 1]) for i in range(num_quads)
    ]


def _hex_mesh_arguments(
    order: int,
    a1: float,
    a2: float,
    a3: float,
    limits: tuple[Pair, Pair, Pair, Pair],
    num_hexes_x1: int,
    num_hexes_x2: int,
    c: coordinates.CoordinateSystem,
    fixed_bounds: bool,
    layers: int = 1,
) -> Optional[tuple[list[mesh.Prism], list[frozenset[mesh.Quad]]]]:
    sorted_starts = sorted(limits)
    sorted_starts = sorted(
        sorted_starts[0:2], key=lambda x: tuple(reversed(x))
    ) + sorted(sorted_starts[2:4], key=lambda x: tuple(reversed(x)), reverse=True)
    # Check that the quadrilateral from by the limits is convex (i.e.,
    # that the Jacobian is nowhere 0)
    aa = (sorted_starts[3][0] - sorted_starts[2][0]) * (
        sorted_starts[1][1] - sorted_starts[0][1]
    ) - (sorted_starts[3][1] - sorted_starts[2][1]) * (
        sorted_starts[1][0] - sorted_starts[0][0]
    )
    bb = (sorted_starts[2][0] - sorted_starts[1][0]) * (
        sorted_starts[3][1] - sorted_starts[0][1]
    ) - (sorted_starts[2][1] - sorted_starts[1][1]) * (
        sorted_starts[3][0] - sorted_starts[0][0]
    )
    cc = (sorted_starts[2][0] - sorted_starts[0][0]) * (
        sorted_starts[3][1] - sorted_starts[1][1]
    ) - (sorted_starts[2][1] - sorted_starts[0][1]) * (
        sorted_starts[3][0] - sorted_starts[1][0]
    )
    J00 = np.sign(cc - aa - bb)
    J10 = np.sign(cc + aa - bb)
    J01 = np.sign(cc - aa + bb)
    J11 = np.sign(cc + aa + bb)
    if J00 != J10 or J00 != J01 or J00 != J11:
        return None

    trace = linear_field_trace(a1, a2, a3, c, 0, (0, 0))
    if c == coordinates.CoordinateSystem.CYLINDRICAL and any(
        a * b <= 0.0 for a, b in itertools.combinations((lim[0] for lim in limits), 2)
    ):
        return None

    lower_points = np.linspace(
        sorted_starts[0], sorted_starts[1], num_hexes_x2 * order + 1
    )
    upper_points = np.linspace(
        sorted_starts[3], sorted_starts[2], num_hexes_x2 * order + 1
    )
    points = np.linspace(lower_points, upper_points, num_hexes_x1 * order + 1)
    alignments = np.ones(points.shape[:-1])
    if fixed_bounds:
        alignments[0, :] = 0.0
        alignments[-1, :] = 0.0
        alignments[:, 0] = 0.0
        alignments[:, -1] = 0.0
    positions = mesh.subdividable_field_aligned_positions(
        coordinates.SliceCoords(points[..., 0], points[..., 1], c),
        a3,
        trace,
        alignments,
        order,
        layers,
    )
    elements = [
        mesh.Prism(
            mesh.PrismTypes.RECTANGULAR,
            positions[i * order : (i + 1) * order + 1, j * order : (j + 1) * order + 1],
        )
        for i, j in itertools.product(range(num_hexes_x1), range(num_hexes_x2))
    ]
    bounds = [
        frozenset(
            mesh.Quad(positions[i * order : (i + 1) * order + 1, 0])
            for i in range(num_hexes_x1)
        ),
        frozenset(
            mesh.Quad(positions[i * order : (i + 1) * order + 1, -1])
            for i in range(num_hexes_x1)
        ),
        frozenset(
            mesh.Quad(positions[0, i * order : (i + 1) * order + 1])
            for i in range(num_hexes_x2)
        ),
        frozenset(
            mesh.Quad(positions[-1, i * order : (i + 1) * order + 1])
            for i in range(num_hexes_x2)
        ),
    ]
    return elements, bounds


def maybe_divide_hex(hexa: mesh.Prism, divide: bool) -> tuple[mesh.Prism, ...]:
    if divide:
        assert hexa.shape == mesh.PrismTypes.RECTANGULAR
        return (
            mesh.Prism(mesh.PrismTypes.TRIANGULAR, hexa.nodes),
            mesh.Prism(mesh.PrismTypes.TRIANGULAR, hexa.nodes[::-1, ::-1]),
        )
    return (hexa,)


@composite
def _hex_and_tri_prism_arguments(
    draw: Any, args: Optional[tuple[list[mesh.Prism], list[frozenset[mesh.Quad]]]]
) -> Optional[tuple[list[mesh.Prism], list[frozenset[mesh.Quad]]]]:
    """Split some of the elements in a hex-mesh into triangular prisms."""
    if args is None:
        return None

    elements, bounds = args
    n = len(elements)
    split_elements = draw(lists(booleans(), min_size=n, max_size=n))

    return list(
        itertools.chain.from_iterable(
            itertools.starmap(
                maybe_divide_hex,
                zip(elements, split_elements),
            )
        )
    ), bounds


def get_quad_boundaries(
    mesh_sequence: list[mesh.Quad],
) -> list[frozenset[mesh.FieldAlignedCurve]]:
    return [frozenset({mesh_sequence[0].north}), frozenset({mesh_sequence[-1].south})]


def _get_end_point(
    start: tuple[float, float, float], distance: float, angle: float
) -> tuple[float, float, float]:
    return (
        start[0] + distance * np.cos(angle),
        start[1] + distance * np.sin(angle),
        start[2],
    )


_centre = shared(whole_numbers, key=1)
_rad = shared(non_zero, key=2)
_dx3 = builds(operator.mul, _rad, floats(0.01, 1.99))
_x1_start = tuples(_centre, _rad).map(lambda x: x[0] + x[1])

_small_centre = shared(small_whole_numbers, key=41)
_small_rad = shared(small_non_zero, key=42)
_small_dx3 = builds(operator.mul, _small_rad, floats(0.01, 1.99))
_small_x1_start = tuples(_small_centre, _small_rad).map(lambda x: x[0] + x[1])

shared_coordinate_systems = shared(sampled_from(coordinates.CoordinateSystem), key=22)

common_slice_coords = shared_coordinate_systems.flatmap(slice_coord_for_system)
slice_coord_pair = cast(
    SearchStrategy[tuple[coordinates.SliceCoord, coordinates.SliceCoord]],
    frozensets(common_slice_coords, min_size=2, max_size=2).map(tuple),
)
shared_coord_pair = shared(slice_coord_pair, key=204)

straight_lines_across_field = builds(
    mesh.straight_line_across_field,
    shared_coord_pair.map(lambda x: x[0]),
    shared_coord_pair.map(lambda x: x[1]),
    orders,
)
across_field_curves = builds(
    offset_line, straight_lines_across_field, floats(-0.5, 0.5)
)

common_coords = shared_coordinate_systems.flatmap(coord_for_system)


T = TypeVar("T")


@composite
def hex_starts(
    draw: Any,
    base_x1: float = 0.0,
    offset_sign: int = 0,
    absmax: int = WHOLE_NUM_MAX,
) -> tuple[Pair, Pair, Pair, Pair]:
    # Give option to use different max, as Nektar++ doesn't behave
    # well with large triangles
    whole_numbers = integers(-absmax, absmax).map(float)
    non_zero = whole_numbers.filter(lambda x: x != 0.0)
    if offset_sign == 0:
        if draw(booleans()):
            offset_sign = 1
        else:
            offset_sign = -1
    base_x2 = draw(whole_numbers)
    offset1 = draw(
        tuples(
            non_zero.map(abs).map(lambda x: offset_sign * x),  # type: ignore
            whole_numbers,
        )
    )
    existing = {offset1}
    offset2 = draw(
        tuples(
            whole_numbers.map(abs).map(lambda x: offset_sign * x),  # type: ignore
            non_zero,
        ).filter(lambda x: x not in existing)
    )
    existing.add(offset2)
    offset3 = draw(
        tuples(
            non_zero.map(abs).map(lambda x: offset_sign * x),  # type: ignore
            non_zero,
        ).filter(lambda x: x not in existing)
    )
    return (
        (base_x1, base_x2),
        (
            base_x1 + offset1[0],
            base_x2 + offset1[1],
        ),
        (
            base_x1 + offset2[0],
            base_x2 + offset2[1],
        ),
        (
            base_x1 + offset3[0],
            base_x2 + offset3[1],
        ),
    )


_num_divisions = shared(integers(1, 5), key=171)
_divisions = _num_divisions.flatmap(lambda x: integers(0, x - 1))
fixed_edges = tuples(booleans(), booleans(), booleans(), booleans())
simple_field_aligned_curve = builds(
    linear_field_aligned_curve,
    integers(1, 5),
    whole_numbers,
    whole_numbers,
    non_zero,
    tuples(non_zero, whole_numbers),
    coordinate_systems,
    integers(-2 * WHOLE_NUM_MAX, 2 * WHOLE_NUM_MAX).map(lambda x: x / WHOLE_NUM_MAX),
    _divisions,
    _num_divisions,
    floats(0.0, 1.0),
    whole_numbers,
)

linear_quad = cast(
    SearchStrategy[mesh.Quad],
    builds(
        trapezoidal_quad,
        integers(1, 5),
        whole_numbers,
        whole_numbers,
        non_zero,
        tuples(
            tuples(non_zero, whole_numbers),
            tuples(non_zero, whole_numbers),
        ).filter(lambda x: x[0] != x[1]),
        coordinate_systems,
        integers(-2 * WHOLE_NUM_MAX, 2 * WHOLE_NUM_MAX).map(
            lambda x: x / WHOLE_NUM_MAX
        ),
        _divisions,
        _num_divisions,
        floats(0.0, 1.0),
        floats(0.0, 1.0),
        whole_numbers,
    )
    .filter(lambda x: x is not None)
    .filter(
        lambda x: len(
            coordinates.FrozenCoordSet(
                c.to_cartesian() for c in cast(mesh.Quad, x).corners()
            )
        )
        == 4
    ),
)
linear_hex = cast(
    SearchStrategy[mesh.Prism],
    builds(
        trapezohedronal_hex,
        integers(1, 5),
        whole_numbers,
        whole_numbers,
        non_zero,
        whole_numbers.flatmap(hex_starts),
        coordinate_systems3d,
        floats(-2.0, 2.0),
        _divisions,
        _num_divisions,
        whole_numbers,
        fixed_edges,
    ).filter(lambda x: x is not None),
)
linear_prism = cast(
    SearchStrategy[mesh.Prism],
    builds(
        simple_prism,
        integers(1, 5),
        small_whole_numbers,
        small_whole_numbers,
        small_non_zero,
        small_whole_numbers.flatmap(
            lambda x: hex_starts(x, absmax=SMALL_WHOLE_NUM_MAX)
        ).map(lambda x: x[:3]),
        coordinate_systems3d,
        floats(-2.0, 2.0),
        _divisions,
        _num_divisions,
        whole_numbers,
        fixed_edges.map(lambda x: x[:3]),
    ).filter(lambda x: x is not None),
)

_x1_start_south = tuples(_x1_start, _rad, floats(0.01, 10.0)).map(
    lambda x: x[0] + x[1] * x[2]
)
# FIXME: Not controlling x3 limits adequately
nonlinear_quad = builds(
    curved_quad,
    integers(2, 8),
    _centre,
    tuples(_x1_start, _x1_start_south),
    whole_numbers,
    whole_numbers,
    _dx3,
    sampled_from(
        [
            coordinates.CoordinateSystem.CARTESIAN,
            coordinates.CoordinateSystem.CARTESIAN_ROTATED,
        ]
    ),
    _divisions,
    _num_divisions,
    floats(0.0, 1.0),
    floats(0.0, 1.0),
    whole_numbers,
)
nonlinear_hex = builds(
    curved_hex,
    integers(2, 8),
    _centre,
    tuples(_x1_start, _rad.map(lambda r: -1 if r < 0 else 1)).flatmap(
        lambda x: hex_starts(*x)
    ),
    whole_numbers,
    _dx3,
    sampled_from(
        [
            coordinates.CoordinateSystem.CARTESIAN,
            coordinates.CoordinateSystem.CARTESIAN_ROTATED,
        ]
    ),
    _divisions,
    _num_divisions,
    whole_numbers,
    fixed_edges,
)
nonlinear_prism = builds(
    curved_prism,
    integers(2, 10),
    _small_centre,
    tuples(_small_x1_start, _small_rad.map(lambda r: -1 if r < 0 else 1))
    .flatmap(lambda x: hex_starts(*x, absmax=SMALL_WHOLE_NUM_MAX))
    .map(lambda x: x[:3]),
    small_whole_numbers,
    _small_dx3,
    sampled_from(
        [
            coordinates.CoordinateSystem.CARTESIAN,
            coordinates.CoordinateSystem.CARTESIAN_ROTATED,
        ]
    ),
    _divisions,
    _num_divisions,
    whole_numbers,
    fixed_edges.map(lambda x: x[:3]),
)

flat_quad = one_of(linear_quad, nonlinear_quad)
quad_in_3d = builds(higher_dim_quad, linear_quad, floats(-np.pi, np.pi)).filter(
    lambda x: x is not None
)
register_type_strategy(mesh.Quad, one_of(flat_quad))  # , quad_in_3d))


def subdivideable_quad(n: int) -> SearchStrategy[mesh.Quad]:
    """Build quad which can be subdivided into n layers"""
    return cast(
        SearchStrategy[mesh.Quad],
        builds(
            trapezoidal_quad,
            integers(1, 5).map(lambda x: x * max(1, n)),
            whole_numbers,
            whole_numbers,
            non_zero,
            tuples(
                tuples(non_zero, whole_numbers),
                tuples(non_zero, whole_numbers),
            ).filter(lambda x: x[0] != x[1]),
            coordinate_systems,
            integers(-2 * WHOLE_NUM_MAX, 2 * WHOLE_NUM_MAX).map(
                lambda x: x / WHOLE_NUM_MAX
            ),
            _divisions,
            _num_divisions,
            floats(0.0, 1.0),
            floats(0.0, 1.0),
            whole_numbers,
        )
        .filter(lambda x: x is not None)
        .filter(
            lambda x: len(
                coordinates.FrozenCoordSet(
                    c.to_cartesian() for c in cast(mesh.Quad, x).corners()
                )
            )
            == 4
        ),
    )


def subdivideable_hex(n: int) -> SearchStrategy[mesh.Prism]:
    """Build hex which can be subdivided into n layers"""
    return cast(
        SearchStrategy[mesh.Prism],
        builds(
            trapezohedronal_hex,
            integers(1, 5).map(lambda x: x * max(1, n)),
            whole_numbers,
            whole_numbers,
            non_zero,
            whole_numbers.flatmap(hex_starts),
            coordinate_systems3d,
            floats(-2.0, 2.0),
            just(0),
            just(1),
            whole_numbers,
            fixed_edges,
        ).filter(lambda x: x is not None),
    )


register_type_strategy(
    mesh.UnalignedShape,
    one_of(
        cast(
            SearchStrategy[mesh.UnalignedShape],
            builds(
                end_quad,
                orders,
                whole_numbers.flatmap(hex_starts),
                sampled_from(
                    [
                        coordinates.CoordinateSystem.CARTESIAN,
                        coordinates.CoordinateSystem.CARTESIAN_ROTATED,
                        coordinates.CoordinateSystem.CYLINDRICAL,
                    ]
                ),
                whole_numbers,
            ).filter(lambda x: x is not None),
        ),
        cast(
            SearchStrategy[mesh.UnalignedShape],
            builds(
                end_triangle,
                orders,
                whole_numbers.flatmap(hex_starts).map(lambda x: x[:3]),
                sampled_from(
                    [
                        coordinates.CoordinateSystem.CARTESIAN,
                        coordinates.CoordinateSystem.CARTESIAN_ROTATED,
                        coordinates.CoordinateSystem.CYLINDRICAL,
                    ]
                ),
                whole_numbers,
            ).filter(lambda x: x is not None),
        ),
    ),
)

flat_sided_hex = one_of(linear_hex, nonlinear_hex)
curve_sided_hex = cast(
    SearchStrategy[mesh.Prism],
    builds(higher_dim_hex, linear_hex, lists(floats(-0.05, 0.05), max_size=4)).filter(
        lambda x: x is not None
    ),
)
flat_sided_prism = one_of(linear_prism, nonlinear_prism)
register_type_strategy(mesh.Prism, one_of(flat_sided_hex, flat_sided_prism))


starts_and_ends = tuples(
    tuples(whole_numbers, whole_numbers, whole_numbers),
    floats(1.0, 1e3),
    floats(0.0, 2 * np.pi),
).map(lambda s: (s[0], _get_end_point(*s)))


def quad_mesh_elements(n: int) -> SearchStrategy[list[mesh.Quad]]:
    """Build quad mesh elements subdivideable into n layers"""
    return cast(
        SearchStrategy[list[mesh.Quad]],
        builds(
            _quad_mesh_elements,
            integers(1, 5),
            whole_numbers,
            whole_numbers,
            non_zero,
            starts_and_ends,
            integers(1, 4),
            coordinate_systems,
            booleans(),
            booleans(),
            just(n),
        ).filter(lambda x: x is not None),
    )


def quad_mesh_arguments(
    n: int,
) -> SearchStrategy[tuple[list[mesh.Quad], list[frozenset[mesh.FieldAlignedCurve]]]]:
    """Build quad mesh arguments with n layers"""
    return quad_mesh_elements(n).map(lambda x: (x, get_quad_boundaries(x)))


quad_mesh_layer_no_divisions = quad_mesh_arguments(1).map(lambda x: mesh.MeshLayer(*x))


def quad_mesh_layer(n: int) -> SearchStrategy[mesh.QuadMeshLayer]:
    """Build quad mesh layer which can be subdivided into n layers"""
    shared_quads = shared(quad_mesh_elements(n))
    return builds(
        mesh.MeshLayer.QuadMeshLayer,
        shared_quads,
        shared_quads.map(get_quad_boundaries),
        just(n),
    )


def hex_mesh_arguments(
    n: int,
) -> SearchStrategy[tuple[list[mesh.Prism], list[frozenset[mesh.Quad]]]]:
    """Build hex mesh arguments which can be subdivided into n layers"""
    return cast(
        SearchStrategy[tuple[list[mesh.Prism], list[frozenset[mesh.Quad]]]],
        builds(
            _hex_mesh_arguments,
            integers(1, 5),
            whole_numbers,
            whole_numbers,
            non_zero,
            whole_numbers.flatmap(hex_starts),
            integers(1, 3),
            integers(1, 3),
            coordinate_systems3d,
            booleans(),
            just(n),
        ).filter(lambda x: x is not None),
    )


def tri_prism_mesh_arguments(
    n: int,
) -> SearchStrategy[tuple[list[mesh.Prism], list[frozenset[mesh.Quad]]]]:
    """Build triangular prism mesh layer arguments which can be subdivided into n layers"""
    return cast(
        SearchStrategy[tuple[list[mesh.Prism], list[frozenset[mesh.Quad]]]],
        builds(
            _hex_mesh_arguments,
            integers(1, 5),
            small_whole_numbers,
            small_whole_numbers,
            small_non_zero,
            small_whole_numbers.flatmap(
                lambda x: hex_starts(x, absmax=SMALL_WHOLE_NUM_MAX)
            ),
            integers(1, 3),
            integers(1, 3),
            coordinate_systems3d,
            booleans(),
            just(n),
        )
        .flatmap(_hex_and_tri_prism_arguments)
        .filter(lambda x: x is not None),
    )


def prism_mesh_arguments(
    n: int,
) -> SearchStrategy[tuple[list[mesh.Prism], list[frozenset[mesh.Quad]]]]:
    """Build prism mesh arguments which can be subdivided into n layers"""
    return one_of(hex_mesh_arguments(n), tri_prism_mesh_arguments(n))


prism_mesh_layer_no_divisions = prism_mesh_arguments(1).map(
    lambda x: mesh.MeshLayer(*x)
)


def prism_mesh_layer(n: int) -> SearchStrategy[mesh.PrismMeshLayer]:
    """Build prism mesh layer which can be subdivided into n layers"""
    shared_prism_mesh_args = shared(prism_mesh_arguments(n))
    return builds(
        mesh.MeshLayer.PrismMeshLayer,
        shared_prism_mesh_args.map(lambda x: x[0]),
        shared_prism_mesh_args.map(lambda x: x[1]),
        just(n),
    ).filter(  # Check that all prisms have unique corners
        lambda m: all(
            (
                (n := len(frozenset(e.corners()))) == 8
                and e.shape == mesh.PrismTypes.RECTANGULAR
            )
            or (n == 6 and e.shape == mesh.PrismTypes.TRIANGULAR)
            for e in m
        )
    )


def subdivideable_mesh_arguments(
    n: int,
) -> SearchStrategy[
    tuple[
        list[mesh.Prism | mesh.Quad],
        list[frozenset[mesh.Quad | mesh.FieldAlignedCurve]],
    ]
]:
    """Build mesh arguments which can be subdivided into n layers"""
    return one_of(quad_mesh_arguments(n), prism_mesh_arguments(n))


mesh_arguments = integers(1, 5).flatmap(subdivideable_mesh_arguments)

register_type_strategy(
    mesh.MeshLayer,
    one_of(
        integers(1, 5).flatmap(quad_mesh_layer),
        integers(1, 5).flatmap(prism_mesh_layer),
    ),
)

x3_offsets = builds(np.linspace, whole_numbers, non_zero, integers(2, 4))
quad_meshes: SearchStrategy[mesh.QuadMesh] = builds(
    mesh.GenericMesh, integers(1, 5).flatmap(quad_mesh_layer), x3_offsets
)
prism_meshes: SearchStrategy[mesh.PrismMesh] = builds(
    mesh.GenericMesh, integers(1, 5).flatmap(prism_mesh_layer), x3_offsets
)
register_type_strategy(
    mesh.GenericMesh, builds(mesh.GenericMesh, from_type(mesh.MeshLayer), x3_offsets)
)


def simple_trace(
    start: coordinates.SliceCoord, x3: npt.ArrayLike, start_weight: float = 0.0
) -> tuple[coordinates.SliceCoords, npt.NDArray]:
    """Return the start-position.

    This is a dummy-implementation of a :class:`~neso_fame.mesh.FieldTrace`.
    """
    return (
        coordinates.SliceCoords(
            np.full_like(x3, start.x1),
            np.full_like(x3, start.x2),
            coordinates.CoordinateSystem.CARTESIAN,
        ),
        np.asarray(x3),
    )
