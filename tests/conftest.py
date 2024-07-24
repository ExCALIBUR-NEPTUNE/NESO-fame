import itertools
import operator
from functools import reduce
from operator import attrgetter, methodcaller
from typing import Any, Optional, TypeVar, cast

import numpy as np
import numpy.typing as npt
from hypothesis import Verbosity, settings
from hypothesis.extra.numpy import (
    BroadcastableShapes,
    array_shapes,
    arrays,
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

from neso_fame import mesh
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


coordinate_systems = sampled_from(mesh.CoordinateSystem)


def slice_coord_for_system(
    system: mesh.CoordinateSystem,
) -> SearchStrategy[mesh.SliceCoord]:
    x1 = non_zero if system == mesh.CoordinateSystem.CYLINDRICAL else whole_numbers
    return builds(
        mesh.SliceCoord,
        x1,
        whole_numbers,
        just(system),
    )


register_type_strategy(
    mesh.SliceCoords,
    builds(
        lambda xs, c: mesh.SliceCoords(xs[0], xs[1], c),
        mutually_broadcastable_arrays(2),
        sampled_from(mesh.CoordinateSystem),
    ),
)

register_type_strategy(
    mesh.SliceCoord,
    sampled_from(mesh.CoordinateSystem).flatmap(slice_coord_for_system),
)

register_type_strategy(
    mesh.Coords,
    builds(
        lambda xs, c: mesh.Coords(xs[0], xs[1], xs[2], c),
        mutually_broadcastable_arrays(3),
        sampled_from(mesh.CoordinateSystem),
    ),
)


register_type_strategy(
    mesh.Coord,
    builds(
        mesh.Coord,
        whole_numbers,
        whole_numbers,
        whole_numbers,
        sampled_from(mesh.CoordinateSystem),
    ),
)


num_divs = shared(integers(1, 10), key=999)


def straight_line_for_system(
    system: mesh.CoordinateSystem,
) -> SearchStrategy[mesh.StraightLine]:
    coords = builds(
        mesh.Coord, whole_numbers, whole_numbers, whole_numbers, just(system)
    )
    return builds(
        mesh.StraightLine,
        coords,
        coords,
        num_divs.flatmap(lambda n: integers(0, n - 1)),
        num_divs,
    )


register_type_strategy(
    mesh.StraightLine,
    coordinate_systems.flatmap(straight_line_for_system),
)

CARTESIAN_SYSTEMS = {
    mesh.CoordinateSystem.CARTESIAN,
    mesh.CoordinateSystem.CARTESIAN2D,
    mesh.CoordinateSystem.CARTESIAN_ROTATED,
}


def linear_field_trace(
    a1: float,
    a2: float,
    a3: float,
    c: mesh.CoordinateSystem,
    skew: float,
    centre: Pair,
) -> mesh.FieldTrace:
    a1p = a1 / a3 if c in CARTESIAN_SYSTEMS else 0.0
    a2p = a2 / a3 if c != mesh.CoordinateSystem.CARTESIAN2D else 0.0

    def cartesian_func(
        start: mesh.SliceCoord, x3: npt.ArrayLike, start_weight: float = 0.0
    ) -> tuple[mesh.SliceCoords, npt.NDArray]:
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
            mesh.SliceCoords(
                t * b1 * np.asarray(x3) + start.x1,
                t * b2 * np.asarray(x3) + start.x2,
                start.system,
            ),
            s,
        )

    return cartesian_func


def linear_field_line(
    a1: float,
    a2: float,
    a3: float,
    b1: float,
    b2: float,
    b3: float,
    c: mesh.CoordinateSystem,
    skew: float,
    centre: Pair,
) -> mesh.NormalisedCurve:
    offset = np.sqrt((b1 - centre[0]) ** 2 + (b2 - centre[1]) ** 2)

    def linear_func(x: npt.ArrayLike) -> mesh.Coords:
        a = a1 if c in CARTESIAN_SYSTEMS else 0.0
        return mesh.Coords(
            a * (1 + skew * offset) * np.asarray(x) + b1 - 0.5 * a,
            a2 * (1 + skew * offset) * np.asarray(x) + b2 - 0.5 * a2,
            a3 * np.asarray(x) + b3 - 0.5 * a3,
            c,
        )

    return linear_func


def trapezoidal_quad(
    a1: float,
    a2: float,
    a3: float,
    starts: tuple[Pair, Pair],
    c: mesh.CoordinateSystem,
    skew: float,
    resolution: int,
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
    if c == mesh.CoordinateSystem.CYLINDRICAL and (starts[0][0] * starts[1][0] <= 0.0):
        return None
    shape = mesh.StraightLineAcrossField(
        mesh.SliceCoord(starts[0][0], starts[0][1], c),
        mesh.SliceCoord(starts[1][0], starts[1][1], c),
    )
    trace = linear_field_trace(a1, a2, a3, c, skew, centre)
    return Offset(
        mesh.Quad(
            shape,
            mesh.FieldTracer(trace, resolution),
            a3,
            division,
            num_divisions,
            north_start_weight,
            south_start_weight,
        ),
        offset,
    )


def end_quad(
    corners: tuple[Pair, Pair, Pair, Pair], c: mesh.CoordinateSystem, x3: float
) -> Optional[mesh.EndShape]:
    if c == mesh.CoordinateSystem.CYLINDRICAL and (0.0 in [c[0] for c in corners]):
        return None

    sorted_corners = sorted(corners, key=operator.itemgetter(1))
    sorted_corners = sorted(sorted_corners[0:2], key=operator.itemgetter(0)) + sorted(
        sorted_corners[2:4], key=operator.itemgetter(0), reverse=True
    )

    def make_line(point1: Pair, point2: Pair) -> mesh.StraightLine:
        return mesh.StraightLine(
            mesh.Coord(point1[0], point1[1], x3, c),
            mesh.Coord(point2[0], point2[1], x3, c),
        )

    return mesh.EndShape(
        (
            make_line(sorted_corners[0], sorted_corners[1]),
            make_line(sorted_corners[2], sorted_corners[3]),
            make_line(sorted_corners[1], sorted_corners[2]),
            make_line(sorted_corners[3], sorted_corners[0]),
        )
    )


def end_triangle(
    corners: tuple[Pair, Pair, Pair], c: mesh.CoordinateSystem, x3: float
) -> Optional[mesh.EndShape]:
    if c == mesh.CoordinateSystem.CYLINDRICAL and (0.0 in [c[0] for c in corners]):
        return None

    def make_line(point1: Pair, point2: Pair) -> mesh.StraightLine:
        return mesh.StraightLine(
            mesh.Coord(point1[0], point1[1], x3, c),
            mesh.Coord(point2[0], point2[1], x3, c),
        )

    return mesh.EndShape(
        (
            make_line(corners[0], corners[1]),
            make_line(corners[1], corners[2]),
            make_line(corners[2], corners[0]),
        )
    )


def _get_start_weight(fixed: bool) -> float:
    if fixed:
        return 1
    return 0


def trapezohedronal_hex(
    a1: float,
    a2: float,
    a3: float,
    starts: tuple[Pair, Pair, Pair, Pair],
    c: mesh.CoordinateSystem,
    skew: float,
    resolution: int,
    division: int,
    num_divisions: int,
    offset: float,
    fixed_edges: tuple[bool, bool, bool, bool],
) -> Optional[mesh.Prism]:
    centre = (
        sum(map(operator.itemgetter(0), starts)),
        sum(map(operator.itemgetter(1), starts)),
    )
    sorted_starts = sorted(starts, key=operator.itemgetter(1))
    sorted_starts = sorted(sorted_starts[0:2], key=operator.itemgetter(0)) + sorted(
        sorted_starts[2:4], key=operator.itemgetter(0), reverse=True
    )
    if c == mesh.CoordinateSystem.CYLINDRICAL and any(
        a * b <= 0.0 for a, b in itertools.combinations((p[0] for p in starts), 2)
    ):
        return None
    trace = linear_field_trace(a1, a2, a3, c, skew, centre)

    def make_quad(point1: Pair, point2: Pair, fixed1: bool, fixed2: bool) -> mesh.Quad:
        shape = mesh.StraightLineAcrossField(
            mesh.SliceCoord(point1[0], point1[1], c),
            mesh.SliceCoord(point2[0], point2[1], c),
        )
        return Offset(
            mesh.Quad(
                shape,
                mesh.FieldTracer(trace, resolution),
                a3,
                division,
                num_divisions,
                _get_start_weight(fixed1),
                _get_start_weight(fixed2),
            ),
            offset,
        )

    return mesh.Prism(
        (
            make_quad(
                sorted_starts[0], sorted_starts[1], fixed_edges[0], fixed_edges[1]
            ),
            make_quad(
                sorted_starts[2], sorted_starts[3], fixed_edges[2], fixed_edges[3]
            ),
            make_quad(
                sorted_starts[1], sorted_starts[2], fixed_edges[1], fixed_edges[2]
            ),
            make_quad(
                sorted_starts[3], sorted_starts[0], fixed_edges[3], fixed_edges[0]
            ),
        )
    )


def simple_prism(
    a1: float,
    a2: float,
    a3: float,
    starts: tuple[Pair, Pair, Pair],
    c: mesh.CoordinateSystem,
    skew: float,
    resolution: int,
    division: int,
    num_divisions: int,
    offset: float,
    fixed_edges: tuple[bool, bool, bool],
) -> Optional[mesh.Prism]:
    centre = (
        sum(map(operator.itemgetter(0), starts)),
        sum(map(operator.itemgetter(1), starts)),
    )
    if c == mesh.CoordinateSystem.CYLINDRICAL and any(
        a * b <= 0.0 for a, b in itertools.combinations((p[0] for p in starts), 2)
    ):
        return None
    trace = linear_field_trace(a1, a2, a3, c, skew, centre)

    def make_quad(point1: Pair, point2: Pair, fixed1: bool, fixed2: bool) -> mesh.Quad:
        shape = mesh.StraightLineAcrossField(
            mesh.SliceCoord(point1[0], point1[1], c),
            mesh.SliceCoord(point2[0], point2[1], c),
        )
        return Offset(
            mesh.Quad(
                shape,
                mesh.FieldTracer(trace, resolution),
                a3,
                division,
                num_divisions,
                _get_start_weight(fixed1),
                _get_start_weight(fixed2),
            ),
            offset,
        )

    return mesh.Prism(
        (
            make_quad(starts[0], starts[1], fixed_edges[0], fixed_edges[1]),
            make_quad(starts[2], starts[0], fixed_edges[2], fixed_edges[0]),
            make_quad(starts[1], starts[2], fixed_edges[1], fixed_edges[2]),
        )
    )


def cylindrical_field_trace(centre: float, x2_slope: float) -> mesh.FieldTrace:
    def cylindrical_func(
        start: mesh.SliceCoord, x3: npt.ArrayLike, start_weight: float = 0.0
    ) -> tuple[mesh.SliceCoords, npt.NDArray]:
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
            mesh.SliceCoords(x1, x2, start.system),
            np.sqrt(alpha) * ellipeinc(dtheta, (b * b - 1) / alpha) * rad,
        )

    return cylindrical_func


def cylindrical_field_line(
    centre: float,
    x1_start: float,
    x2_centre: float,
    x2_slope: float,
    x3_limits: Pair,
    system: mesh.CoordinateSystem,
) -> mesh.NormalisedCurve:
    assert system in CARTESIAN_SYSTEMS
    rad = x1_start - centre
    half_x3_sep = 0.5 * (x3_limits[1] - x3_limits[0])
    x3_centre = x3_limits[0] + half_x3_sep
    theta_max = 2 * np.arcsin(half_x3_sep / rad)

    def cylindrical_func(s: npt.ArrayLike) -> mesh.Coords:
        theta = theta_max * (np.asarray(s) - 0.5)
        x1 = centre + rad * np.cos(theta)
        x2 = x2_centre + x2_slope * theta
        x3 = x3_centre + rad * np.sin(theta)
        return mesh.Coords(x1, x2, x3, system)

    return cylindrical_func


def curved_quad(
    centre: float,
    x1_start: Pair,
    x2_centre: float,
    x2_slope: float,
    dx3: float,
    system: mesh.CoordinateSystem,
    resolution: int,
    division: int,
    num_divisions: int,
    north_start_weight: float,
    south_start_weight: float,
    offset: float,
) -> mesh.Quad:
    trace = cylindrical_field_trace(centre, x2_slope)
    shape = mesh.StraightLineAcrossField(
        mesh.SliceCoord(x1_start[0], x2_centre, system),
        mesh.SliceCoord(x1_start[1], x2_centre, system),
    )
    return Offset(
        mesh.Quad(
            shape,
            mesh.FieldTracer(trace, resolution),
            dx3,
            division,
            num_divisions,
            north_start_weight,
            south_start_weight,
        ),
        offset,
    )


def curved_hex(
    centre: float,
    starts: tuple[Pair, Pair, Pair, Pair],
    x2_slope: float,
    dx3: float,
    system: mesh.CoordinateSystem,
    resolution: int,
    division: int,
    num_divisions: int,
    offset: float,
    fixed_edges: tuple[bool, bool, bool, bool],
) -> mesh.Prism:
    sorted_starts = sorted(starts, key=operator.itemgetter(0))
    sorted_starts = sorted(sorted_starts[0:2], key=operator.itemgetter(1)) + sorted(
        sorted_starts[2:4], key=operator.itemgetter(1), reverse=True
    )
    trace = cylindrical_field_trace(centre, x2_slope)

    def make_quad(point1: Pair, point2: Pair, fixed1: bool, fixed2: bool) -> mesh.Quad:
        shape = mesh.StraightLineAcrossField(
            mesh.SliceCoord(point1[0], point1[1], system),
            mesh.SliceCoord(point2[0], point2[1], system),
        )
        return Offset(
            mesh.Quad(
                shape,
                mesh.FieldTracer(trace, resolution),
                dx3,
                division,
                num_divisions,
                _get_start_weight(fixed1),
                _get_start_weight(fixed2),
            ),
            offset,
        )

    return mesh.Prism(
        (
            make_quad(
                sorted_starts[0], sorted_starts[1], fixed_edges[0], fixed_edges[1]
            ),
            make_quad(
                sorted_starts[2], sorted_starts[3], fixed_edges[2], fixed_edges[3]
            ),
            make_quad(
                sorted_starts[1], sorted_starts[2], fixed_edges[1], fixed_edges[2]
            ),
            make_quad(
                sorted_starts[3], sorted_starts[0], fixed_edges[3], fixed_edges[0]
            ),
        )
    )


def curved_prism(
    centre: float,
    starts: tuple[Pair, Pair, Pair],
    x2_slope: float,
    dx3: float,
    system: mesh.CoordinateSystem,
    resolution: int,
    division: int,
    num_divisions: int,
    offset: float,
    fixed_edges: tuple[bool, bool, bool],
) -> mesh.Prism:
    trace = cylindrical_field_trace(centre, x2_slope)

    def make_quad(point1: Pair, point2: Pair, fixed1: bool, fixed2: bool) -> mesh.Quad:
        shape = mesh.StraightLineAcrossField(
            mesh.SliceCoord(point1[0], point1[1], system),
            mesh.SliceCoord(point2[0], point2[1], system),
        )
        return Offset(
            mesh.Quad(
                shape,
                mesh.FieldTracer(trace, resolution),
                dx3,
                division,
                num_divisions,
                _get_start_weight(fixed1),
                _get_start_weight(fixed2),
            ),
            offset,
        )

    return mesh.Prism(
        (
            make_quad(starts[0], starts[1], fixed_edges[0], fixed_edges[1]),
            make_quad(starts[2], starts[0], fixed_edges[2], fixed_edges[0]),
            make_quad(starts[1], starts[2], fixed_edges[1], fixed_edges[2]),
        )
    )


def make_arc(
    north: mesh.SliceCoords, south: mesh.SliceCoords, angle: float
) -> mesh.AcrossFieldCurve:
    x1_1 = float(north.x1)
    x2_1 = float(north.x2)
    x1_2 = float(south.x1)
    x2_2 = float(south.x2)
    dx1 = x1_1 - x1_2
    dx2 = x2_1 - x2_2
    denom = 2 * dx1 * np.cos(angle) + 2 * dx2 * np.sin(angle)
    if abs(denom) < 1e-2:
        raise ValueError("Radius of circle too large.")
    r = (dx1 * dx1 + dx2 * dx2) / denom
    x1_c = x1_1 - r * np.cos(angle)
    x2_c = x2_1 - r * np.sin(angle)
    a = np.arctan2(x2_2 - x2_c, x1_2 - x1_c) - angle

    def curve(s: npt.ArrayLike) -> mesh.SliceCoords:
        s = np.asarray(s)
        return mesh.SliceCoords(
            x1_c + r * np.cos(angle + a * s),
            x2_c + r * np.sin(angle + a * s),
            north.system,
        )

    return curve

def offset_straight_line(line: mesh.StraightLineAcrossField, magnitude: float) -> mesh.AcrossFieldCurve:
    if magnitude == 0:
        return line
    dx1 = line.north.x1 - line.south.x1
    dx2 = line.north.x2 - line.south.x2
    norm = np.sqrt(dx1 * dx1 + dx2 * dx2)
    perp = [dx2 / norm, -dx1/norm]

    def result(s: npt.ArrayLike) -> mesh.SliceCoords:
        s = np.asarray(s)
        linear = line(s)
        t = 4 * magnitude * s * (1-s)
        return mesh.SliceCoords(linear.x1 + t * perp[0], linear.x2 + t * perp[1], linear.system)

    return result

def higher_dim_quad(q: mesh.Quad, angle: float) -> Optional[mesh.Quad]:
    # This assumes that dx3/ds is an even function about the starting
    # x3 point from which the bounding field lines were projected
    try:
        curve = make_arc(q.shape(0.0), q.shape(1.0), angle)
    except ValueError:
        return None
    x3 = q.x3_offset
    q = q.get_underlying_object()
    return Offset(
        mesh.Quad(
            curve,
            q.field,
            q.dx3,
            q.subdivision,
            q.num_divisions,
            q.north_start_weight,
            q.south_start_weight,
        ),
        x3,
    )


def higher_dim_hex(h: mesh.Prism, magnitudes: list[float]) -> Optional[mesh.Prism]:
    try:
        new_quads = tuple(
            Offset(
                mesh.Quad(
                    offset_straight_line(q.shape, m),
                    q.field,
                    q.dx3,
                    q.subdivision,
                    q.num_divisions,
                ),
                x3,
            )
            for q, x3, m in zip(
                map(methodcaller("get_underlying_object"), h),
                map(attrgetter("x3_offset"), h),
                itertools.chain(magnitudes, itertools.repeat(0))
            )
        )
    except ValueError:
        return None
    return mesh.Prism(new_quads)


def _quad_mesh_elements(
    a1: float,
    a2: float,
    a3: float,
    limits: tuple[Pair, Pair],
    num_quads: int,
    c: mesh.CoordinateSystem,
    resolution: int,
    left_fixed: bool,
    right_fixed: bool,
) -> Optional[list[mesh.Quad]]:
    trace = mesh.FieldTracer(linear_field_trace(a1, a2, a3, c, 0, (0, 0)), resolution)
    if c == mesh.CoordinateSystem.CYLINDRICAL and limits[0][0] * limits[1][0] <= 0:
        return None

    def make_shape(starts: tuple[Pair, Pair]) -> mesh.AcrossFieldCurve:
        return mesh.StraightLineAcrossField(
            mesh.SliceCoord(starts[0][0], starts[0][1], c),
            mesh.SliceCoord(starts[1][0], starts[1][1], c),
        )

    points = np.linspace(limits[0], limits[1], num_quads + 1)
    fixed = [left_fixed] + [False] * (num_quads - 1) + [right_fixed]

    return [
        mesh.Quad(
            shape,
            trace,
            a3,
            north_start_weight=north_weight,
            south_start_weight=south_weight,
        )
        for shape, (north_weight, south_weight) in zip(
            map(make_shape, itertools.pairwise(points)),
            itertools.pairwise(map(_get_start_weight, fixed)),
        )
    ]


def _hex_mesh_arguments(
    a1: float,
    a2: float,
    a3: float,
    limits: tuple[Pair, Pair, Pair, Pair],
    num_hexes_x1: int,
    num_hexes_x2: int,
    c: mesh.CoordinateSystem,
    resolution: int,
    fixed_bounds: bool,
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

    trace = mesh.FieldTracer(linear_field_trace(a1, a2, a3, c, 0, (0, 0)), resolution)
    if c == mesh.CoordinateSystem.CYLINDRICAL and any(
        a * b <= 0.0 for a, b in itertools.combinations((lim[0] for lim in limits), 2)
    ):
        return None

    def make_line(start: Pair, end: Pair) -> mesh.AcrossFieldCurve:
        return mesh.StraightLineAcrossField(
            mesh.SliceCoord(start[0], start[1], c),
            mesh.SliceCoord(end[0], end[1], c),
        )

    def get_start_weight(
        quad_is_bound: bool,
        edge_is_bound: bool,
    ) -> float:
        if not fixed_bounds:
            return 0.0
        if quad_is_bound:
            return 1.0
        return _get_start_weight(edge_is_bound)

    def make_element_and_bounds(
        pairs: list[Pair], is_bound: list[bool]
    ) -> tuple[mesh.Prism, list[frozenset[mesh.Quad]]]:
        edges = (
            mesh.Quad(
                make_line(pairs[0], pairs[1]),
                trace,
                a3,
                north_start_weight=get_start_weight(is_bound[0], is_bound[2]),
                south_start_weight=get_start_weight(is_bound[0], is_bound[3]),
            ),
            mesh.Quad(
                make_line(pairs[2], pairs[3]),
                trace,
                a3,
                north_start_weight=get_start_weight(is_bound[1], is_bound[2]),
                south_start_weight=get_start_weight(is_bound[1], is_bound[3]),
            ),
            mesh.Quad(
                make_line(pairs[0], pairs[2]),
                trace,
                a3,
                north_start_weight=get_start_weight(is_bound[2], is_bound[0]),
                south_start_weight=get_start_weight(is_bound[2], is_bound[1]),
            ),
            mesh.Quad(
                make_line(pairs[1], pairs[3]),
                trace,
                a3,
                north_start_weight=get_start_weight(is_bound[3], is_bound[0]),
                south_start_weight=get_start_weight(is_bound[3], is_bound[1]),
            ),
        )
        return mesh.Prism(edges), [
            frozenset({e}) if b else frozenset() for e, b in zip(edges, is_bound)
        ]

    def fold(
        x: tuple[list[mesh.Prism], list[frozenset[mesh.Quad]]],
        y: tuple[mesh.Prism, list[frozenset[mesh.Quad]]],
    ) -> tuple[list[mesh.Prism], list[frozenset[mesh.Quad]]]:
        hexes, bounds = x
        hexes.append(y[0])
        new_bounds = [b | y_comp for b, y_comp in zip(bounds, y[1])]
        return hexes, new_bounds

    lower_points = np.linspace(sorted_starts[0], sorted_starts[1], num_hexes_x2 + 1)
    upper_points = np.linspace(sorted_starts[3], sorted_starts[2], num_hexes_x2 + 1)
    points = np.linspace(lower_points, upper_points, num_hexes_x1 + 1)
    initial: tuple[list[mesh.Prism], list[frozenset[mesh.Quad]]] = (
        [],
        [frozenset()] * 4,
    )
    result = reduce(
        fold,
        (
            make_element_and_bounds(
                [
                    points[i, j],
                    points[i + 1, j],
                    points[i, j + 1],
                    points[i + 1, j + 1],
                ],
                [j == 0, j == num_hexes_x2 - 1, i == 0, i == num_hexes_x1 - 1],
            )
            for i in range(num_hexes_x1)
            for j in range(num_hexes_x2)
        ),
        initial,
    )
    return result


def get_connecting_quad(q1: mesh.Quad, q2: mesh.Quad) -> mesh.Quad:
    v1 = {
        (q1.shape(0.0).to_coord(), q1.north_start_weight),
        (q1.shape(1.0).to_coord(), q1.south_start_weight),
    }
    v2 = {
        (q2.shape(0.0).to_coord(), q2.north_start_weight),
        (q2.shape(1.0).to_coord(), q2.south_start_weight),
    }
    point1 = next(iter(v1 - v2))
    point2 = next(iter(v2 - v1))
    return mesh.Quad(
        mesh.StraightLineAcrossField(point1[0], point2[0]),
        q1.field,
        q1.dx3,
        q1.subdivision,
        q1.num_divisions,
        point1[1],
        point2[1],
    )


def maybe_divide_hex(hexa: mesh.Prism, divide: bool) -> tuple[mesh.Prism, ...]:
    if divide:
        sides = hexa.sides
        assert len(sides) == 4
        new_quad = get_connecting_quad(sides[3], sides[0])
        return (
            mesh.Prism((sides[0], new_quad, sides[3])),
            mesh.Prism((sides[2], new_quad, sides[1])),
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
) -> list[frozenset[mesh.Segment]]:
    return [frozenset({mesh_sequence[0].north}), frozenset({mesh_sequence[-1].south})]


def _get_end_point(
    start: tuple[float, float, float], distance: float, angle: float
) -> tuple[float, float, float]:
    return (
        start[0] + distance * np.cos(angle),
        start[1] + distance * np.sin(angle),
        start[2],
    )


def straight_field_line_for_system(
    system: mesh.CoordinateSystem,
) -> SearchStrategy[mesh.FieldAlignedCurve]:
    a1 = shared(whole_numbers, key=541)
    a2 = shared(whole_numbers, key=542)
    a3 = shared(non_zero, key=543)

    trace = builds(
        linear_field_trace,
        a1,
        a2,
        a3,
        just(system),
        just(0.0),
        just((0.0, 0.0)),
    )
    return builds(
        mesh.FieldAlignedCurve,
        builds(mesh.FieldTracer, trace, integers(2, 10)),
        slice_coord_for_system(system),
        non_zero,
        _divisions,
        _num_divisions,
        floats(0.0, 1.0),
    )


_centre = shared(whole_numbers, key=1)
_rad = shared(non_zero, key=2)
_dx3 = builds(operator.mul, _rad, floats(0.01, 1.99))
_x1_start = tuples(_centre, _rad).map(lambda x: x[0] + x[1])

_small_centre = shared(small_whole_numbers, key=41)
_small_rad = shared(small_non_zero, key=42)
_small_dx3 = builds(operator.mul, _small_rad, floats(0.01, 1.99))
_small_x1_start = tuples(_small_centre, _small_rad).map(lambda x: x[0] + x[1])


def curved_field_line_for_system(
    system: mesh.CoordinateSystem,
) -> SearchStrategy[mesh.FieldAlignedCurve]:
    trace = builds(cylindrical_field_trace, _centre, whole_numbers)

    return builds(
        mesh.FieldAlignedCurve,
        builds(mesh.FieldTracer, trace, integers(100, 200)),
        builds(
            mesh.SliceCoord,
            builds(operator.add, _centre, _rad),
            whole_numbers,
            just(system),
        ),
        _rad.map(abs).flatmap(lambda r: floats(0.01 * r, 1.99 * r)),  # type: ignore
        _divisions,
        _num_divisions,
        floats(0.0, 1.0),
    )


def field_aligned_curve_for_system(
    system: mesh.CoordinateSystem,
) -> SearchStrategy[mesh.FieldAlignedCurve]:
    if system in CARTESIAN_SYSTEMS:
        return one_of(
            (
                straight_field_line_for_system(system),
                curved_field_line_for_system(system),
            )
        )
    else:
        return straight_field_line_for_system(system)


straight_field_line = coordinate_systems.flatmap(straight_field_line_for_system)
curved_field_line = sampled_from(list(CARTESIAN_SYSTEMS)).flatmap(
    curved_field_line_for_system
)
register_type_strategy(
    mesh.FieldAlignedCurve,
    one_of(straight_field_line, curved_field_line),
)

shared_coordinate_systems = shared(sampled_from(mesh.CoordinateSystem), key=22)
segments: SearchStrategy[mesh.Segment] = one_of(
    (
        shared_coordinate_systems.flatmap(straight_line_for_system),
        shared_coordinate_systems.flatmap(field_aligned_curve_for_system),
    )
)


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
linear_quad = cast(
    SearchStrategy[mesh.Quad],
    builds(
        trapezoidal_quad,
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
        integers(3, 5),
        _divisions,
        _num_divisions,
        floats(0.0, 1.0),
        floats(0.0, 1.0),
        whole_numbers,
    )
    .filter(lambda x: x is not None)
    .filter(
        lambda x: len(
            frozenset(cast(mesh.Quad, x).corners().to_cartesian().iter_points())
        )
        == 4
    ),
)
linear_hex = cast(
    SearchStrategy[mesh.Prism],
    builds(
        trapezohedronal_hex,
        whole_numbers,
        whole_numbers,
        non_zero,
        whole_numbers.flatmap(hex_starts),
        coordinate_systems,
        floats(-2.0, 2.0),
        integers(2, 5),
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
        small_whole_numbers,
        small_whole_numbers,
        small_non_zero,
        small_whole_numbers.flatmap(
            lambda x: hex_starts(x, absmax=SMALL_WHOLE_NUM_MAX)
        ).map(lambda x: x[:3]),
        coordinate_systems,
        floats(-2.0, 2.0),
        integers(2, 5),
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
    _centre,
    tuples(_x1_start, _x1_start_south),
    whole_numbers,
    whole_numbers,
    _dx3,
    sampled_from(
        [mesh.CoordinateSystem.CARTESIAN, mesh.CoordinateSystem.CARTESIAN_ROTATED]
    ),
    integers(100, 200),
    _divisions,
    _num_divisions,
    floats(0.0, 1.0),
    floats(0.0, 1.0),
    whole_numbers,
)
nonlinear_hex = builds(
    curved_hex,
    _centre,
    tuples(_x1_start, _rad.map(lambda r: -1 if r < 0 else 1)).flatmap(
        lambda x: hex_starts(*x)
    ),
    whole_numbers,
    _dx3,
    sampled_from(
        [mesh.CoordinateSystem.CARTESIAN, mesh.CoordinateSystem.CARTESIAN_ROTATED]
    ),
    integers(100, 200),
    _divisions,
    _num_divisions,
    whole_numbers,
    fixed_edges,
)
nonlinear_prism = builds(
    curved_prism,
    _small_centre,
    tuples(_small_x1_start, _small_rad.map(lambda r: -1 if r < 0 else 1))
    .flatmap(lambda x: hex_starts(*x, absmax=SMALL_WHOLE_NUM_MAX))
    .map(lambda x: x[:3]),
    small_whole_numbers,
    _small_dx3,
    sampled_from(
        [mesh.CoordinateSystem.CARTESIAN, mesh.CoordinateSystem.CARTESIAN_ROTATED]
    ),
    integers(100, 200),
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

register_type_strategy(
    mesh.EndShape,
    one_of(
        cast(
            SearchStrategy[mesh.EndShape],
            builds(
                end_quad,
                whole_numbers.flatmap(hex_starts),
                sampled_from(
                    [
                        mesh.CoordinateSystem.CARTESIAN,
                        mesh.CoordinateSystem.CARTESIAN_ROTATED,
                        mesh.CoordinateSystem.CYLINDRICAL,
                    ]
                ),
                whole_numbers,
            ).filter(lambda x: x is not None),
        ),
        cast(
            SearchStrategy[mesh.EndShape],
            builds(
                end_triangle,
                whole_numbers.flatmap(hex_starts).map(lambda x: x[:3]),
                sampled_from(
                    [
                        mesh.CoordinateSystem.CARTESIAN,
                        mesh.CoordinateSystem.CARTESIAN_ROTATED,
                        mesh.CoordinateSystem.CYLINDRICAL,
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
quad_mesh_elements = cast(
    SearchStrategy[list[mesh.Quad]],
    builds(
        _quad_mesh_elements,
        whole_numbers,
        whole_numbers,
        non_zero,
        starts_and_ends,
        integers(1, 4),
        coordinate_systems,
        integers(2, 10),
        booleans(),
        booleans(),
    ).filter(lambda x: x is not None),
)
quad_mesh_arguments = quad_mesh_elements.map(lambda x: (x, get_quad_boundaries(x)))
quad_mesh_layer_no_divisions = quad_mesh_arguments.map(lambda x: mesh.MeshLayer(*x))
shared_quads = shared(quad_mesh_elements)
quad_mesh_layer = builds(
    mesh.MeshLayer.QuadMeshLayer,
    shared_quads,
    shared_quads.map(get_quad_boundaries),
    integers(1, 3),
)

hex_mesh_arguments = cast(
    SearchStrategy[tuple[list[mesh.Prism], list[frozenset[mesh.Quad]]]],
    builds(
        _hex_mesh_arguments,
        whole_numbers,
        whole_numbers,
        non_zero,
        whole_numbers.flatmap(hex_starts),
        integers(1, 3),
        integers(1, 3),
        coordinate_systems,
        integers(2, 5),
        booleans(),
    ).filter(lambda x: x is not None),
)
tri_prism_mesh_arguments = cast(
    SearchStrategy[tuple[list[mesh.Prism], list[frozenset[mesh.Quad]]]],
    builds(
        _hex_mesh_arguments,
        small_whole_numbers,
        small_whole_numbers,
        small_non_zero,
        small_whole_numbers.flatmap(
            lambda x: hex_starts(x, absmax=SMALL_WHOLE_NUM_MAX)
        ),
        integers(1, 3),
        integers(1, 3),
        coordinate_systems,
        integers(2, 5),
        booleans(),
    )
    .flatmap(_hex_and_tri_prism_arguments)
    .filter(lambda x: x is not None),
)
prism_mesh_arguments = one_of(hex_mesh_arguments, tri_prism_mesh_arguments)

prism_mesh_layer_no_divisions = prism_mesh_arguments.map(lambda x: mesh.MeshLayer(*x))
shared_prism_mesh_args = shared(prism_mesh_arguments)
prism_mesh_layer = builds(
    mesh.MeshLayer.PrismMeshLayer,
    shared_prism_mesh_args.map(lambda x: x[0]),
    shared_prism_mesh_args.map(lambda x: x[1]),
    integers(1, 3),
).filter(  # Check that all prisms have unique corners
    lambda m: all(len(e.corners()) == 2 * len(e.sides) for e in m)
)

mesh_arguments = one_of(quad_mesh_arguments, prism_mesh_arguments)

register_type_strategy(mesh.MeshLayer, one_of(quad_mesh_layer, prism_mesh_layer))

x3_offsets = builds(np.linspace, whole_numbers, non_zero, integers(2, 4))
quad_meshes: SearchStrategy[mesh.QuadMesh] = builds(
    mesh.GenericMesh, quad_mesh_layer, x3_offsets
)
prism_meshes: SearchStrategy[mesh.PrismMesh] = builds(
    mesh.GenericMesh, prism_mesh_layer, x3_offsets
)
register_type_strategy(
    mesh.GenericMesh, builds(mesh.GenericMesh, from_type(mesh.MeshLayer), x3_offsets)
)


def simple_trace(
    start: mesh.SliceCoord, x3: npt.ArrayLike, start_weight: float = 0.0
) -> tuple[mesh.SliceCoords, npt.NDArray]:
    return (
        mesh.SliceCoords(
            np.full_like(x3, start.x1),
            np.full_like(x3, start.x2),
            mesh.CoordinateSystem.CARTESIAN,
        ),
        np.asarray(x3),
    )
