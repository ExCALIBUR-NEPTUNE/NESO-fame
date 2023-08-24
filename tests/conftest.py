import itertools
import operator
from functools import reduce
from typing import Optional, TypeVar, cast

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
    none,
    one_of,
    register_type_strategy,
    sampled_from,
    shared,
    tuples,
)

from neso_fame import mesh

settings.register_profile("ci", max_examples=400, deadline=None)
settings.register_profile(
    "debug", max_examples=10, verbosity=Verbosity.verbose, report_multiple_bugs=False
)
settings.register_profile("dev", max_examples=10)


Pair = tuple[float, float]


def non_nans():
    return floats(allow_nan=False)


def arbitrary_arrays():
    return arrays(floating_dtypes(), array_shapes())


WHOLE_NUM_MAX = 1000
whole_numbers = integers(-WHOLE_NUM_MAX, WHOLE_NUM_MAX).map(float)
nonnegative_numbers = integers(1, WHOLE_NUM_MAX).map(float)
non_zero = whole_numbers.filter(lambda x: x != 0.0)


def mutually_broadcastable_arrays(
    num_arrays: int,
) -> SearchStrategy[tuple[npt.NDArray, ...]]:
    return mutually_broadcastable_from(
        mutually_broadcastable_shapes(num_shapes=num_arrays)
    )


def mutually_broadcastable_from(
    strategy: SearchStrategy[BroadcastableShapes],
) -> SearchStrategy[tuple[npt.NDArray]]:
    def shape_to_array(shapes):
        return tuples(
            *(
                arrays(np.float64, just(s), elements=whole_numbers)
                for s in shapes.input_shapes
            )
        )

    return strategy.flatmap(shape_to_array)


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

CARTESIAN_SYSTEMS = {mesh.CoordinateSystem.CARTESIAN, mesh.CoordinateSystem.CARTESIAN2D}


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
        start: mesh.SliceCoord, x3: npt.ArrayLike
    ) -> tuple[mesh.SliceCoords, npt.NDArray]:
        offset = np.sqrt((start.x1 - centre[0]) ** 2 + (start.x2 - centre[1]) ** 2)
        b1 = a1p * (1 + skew * offset)
        b2 = a2p * (1 + skew * offset)
        if c in CARTESIAN_SYSTEMS:
            s = np.sqrt(b1 * b1 + b2 * b2 + 1) * np.asarray(x3)
        else:
            s = np.sqrt(b1 * b1 + b2 * b2 + start.x1 * start.x1) * np.asarray(x3)
        return (
            mesh.SliceCoords(
                b1 * np.asarray(x3) + start.x1,
                b2 * np.asarray(x3) + start.x2,
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
    offset: float,
) -> Optional[mesh.Quad]:
    centre = (
        starts[0][0] + (starts[0][0] - starts[1][0]) / 2,
        starts[0][1] + (starts[0][1] - starts[1][1]) / 2,
    )
    if c == mesh.CoordinateSystem.CYLINDRICAL and (
        starts[0][0] == 0.0 or starts[1][0] == 0.0
    ):
        return None
    shape = mesh.StraightLineAcrossField(
        mesh.SliceCoord(starts[0][0], starts[0][1], c),
        mesh.SliceCoord(starts[1][0], starts[1][1], c),
    )
    trace = linear_field_trace(a1, a2, a3, c, skew, centre)
    return mesh.Quad(
        shape,
        mesh.FieldTracer(trace, resolution),
        a3,
        division,
        num_divisions,
        offset,
    )


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
) -> Optional[mesh.Hex]:
    centre = (
        sum(map(operator.itemgetter(0), starts)),
        sum(map(operator.itemgetter(1), starts)),
    )
    sorted_starts = sorted(starts, key=operator.itemgetter(1))
    sorted_starts = sorted(sorted_starts[0:2], key=operator.itemgetter(0)) + sorted(
        sorted_starts[2:4], key=operator.itemgetter(0), reverse=True
    )
    if c == mesh.CoordinateSystem.CYLINDRICAL and (0.0 in [s[0] for s in starts]):
        return None
    trace = linear_field_trace(a1, a2, a3, c, skew, centre)

    def make_quad(point1: Pair, point2: Pair) -> mesh.Quad:
        shape = mesh.StraightLineAcrossField(
            mesh.SliceCoord(point1[0], point1[1], c),
            mesh.SliceCoord(point2[0], point2[1], c),
        )
        return mesh.Quad(
            shape,
            mesh.FieldTracer(trace, resolution),
            a3,
            division,
            num_divisions,
            offset,
        )

    return mesh.Hex(
        make_quad(sorted_starts[0], sorted_starts[1]),
        make_quad(sorted_starts[1], sorted_starts[2]),
        make_quad(sorted_starts[2], sorted_starts[3]),
        make_quad(sorted_starts[3], sorted_starts[0]),
    )


def cylindrical_field_trace(centre: float, x2_slope: float) -> mesh.FieldTrace:
    def cylindrical_func(
        start: mesh.SliceCoord, x3: npt.ArrayLike
    ) -> tuple[mesh.SliceCoords, npt.NDArray]:
        assert start.system in CARTESIAN_SYSTEMS
        rad = start.x1 - centre
        sign = np.sign(rad)
        x3 = np.asarray(x3)
        x1 = centre + sign * np.sqrt(rad * rad - x3 * x3)
        dtheta = np.arcsin(x3 / rad)
        x2 = np.asarray(start.x2) + x2_slope * dtheta
        return (
            mesh.SliceCoords(x1, x2, start.system),
            np.sqrt(1 + x2_slope * x2_slope) * dtheta * rad,
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
    offset: float,
) -> mesh.Quad:
    trace = cylindrical_field_trace(centre, x2_slope)
    shape = mesh.StraightLineAcrossField(
        mesh.SliceCoord(x1_start[0], x2_centre, system),
        mesh.SliceCoord(x1_start[1], x2_centre, system),
    )
    return mesh.Quad(
        shape,
        mesh.FieldTracer(trace, resolution),
        dx3,
        division,
        num_divisions,
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
) -> mesh.Hex:
    sorted_starts = sorted(starts, key=operator.itemgetter(0))
    sorted_starts = sorted(sorted_starts[0:2], key=operator.itemgetter(1)) + sorted(
        sorted_starts[2:4], key=operator.itemgetter(1), reverse=True
    )
    trace = cylindrical_field_trace(centre, x2_slope)

    def make_quad(point1: Pair, point2: Pair) -> mesh.Quad:
        shape = mesh.StraightLineAcrossField(
            mesh.SliceCoord(point1[0], point1[1], system),
            mesh.SliceCoord(point2[0], point2[1], system),
        )
        return mesh.Quad(
            shape,
            mesh.FieldTracer(trace, resolution),
            dx3,
            division,
            num_divisions,
            offset,
        )

    return mesh.Hex(
        make_quad(sorted_starts[0], sorted_starts[1]),
        make_quad(sorted_starts[1], sorted_starts[2]),
        make_quad(sorted_starts[2], sorted_starts[3]),
        make_quad(sorted_starts[3], sorted_starts[0]),
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


def higher_dim_quad(q: mesh.Quad, angle: float) -> Optional[mesh.Quad]:
    # This assumes that dx3/ds is an even function about the starting
    # x3 point from which the bounding field lines were projected
    try:
        curve = make_arc(q.shape(0.0), q.shape(1.0), angle)
    except ValueError:
        return None
    return mesh.Quad(
        curve,
        q.field,
        q.dx3,
        q.subdivision,
        q.num_divisions,
        q.x3_offset,
    )


def higher_dim_hex(h: mesh.Hex, angle: float) -> Optional[mesh.Hex]:
    # This assumes that dx3/ds is an even function about the starting
    # x3 point from which the bounding field lines were projected
    try:
        new_quads = [
            mesh.Quad(
                make_arc(q.shape(0.0), q.shape(1.0), angle),
                q.field,
                q.dx3,
                q.subdivision,
                q.num_divisions,
                q.x3_offset,
            )
            for q in h
        ]
    except ValueError:
        return None
    return mesh.Hex(*new_quads)


def _quad_mesh_elements(
    a1: float,
    a2: float,
    a3: float,
    limits: tuple[Pair, Pair],
    num_quads: int,
    c: mesh.CoordinateSystem,
    resolution: int,
) -> Optional[list[mesh.Quad]]:
    trace = mesh.FieldTracer(linear_field_trace(a1, a2, a3, c, 0, (0, 0)), resolution)
    if c == mesh.CoordinateSystem.CYLINDRICAL and (
        limits[0][0] == 0.0 or limits[1][0] == 0.0 or limits[0][0] * limits[1][0] < 0
    ):
        return None

    def make_shape(starts: tuple[Pair, Pair]) -> mesh.AcrossFieldCurve:
        return mesh.StraightLineAcrossField(
            mesh.SliceCoord(starts[0][0], starts[0][1], c),
            mesh.SliceCoord(starts[1][0], starts[1][1], c),
        )

    points = np.linspace(limits[0], limits[1], num_quads + 1)
    return [
        mesh.Quad(shape, trace, a3)
        for shape in map(make_shape, itertools.pairwise(points))
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
) -> Optional[tuple[list[mesh.Hex], list[frozenset[mesh.Quad]]]]:
    sorted_starts = sorted(limits, key=operator.itemgetter(0))
    sorted_starts = sorted(sorted_starts[0:2], key=operator.itemgetter(1)) + sorted(
        sorted_starts[2:4], key=operator.itemgetter(1), reverse=True
    )
    trace = mesh.FieldTracer(linear_field_trace(a1, a2, a3, c, 0, (0, 0)), resolution)
    if c == mesh.CoordinateSystem.CYLINDRICAL and (
        limits[0][0] == 0.0 or limits[1][0] == 0.0
    ):
        return None

    def make_line(start: Pair, end: Pair) -> mesh.AcrossFieldCurve:
        return mesh.StraightLineAcrossField(
            mesh.SliceCoord(start[0], start[1], c),
            mesh.SliceCoord(end[0], end[1], c),
        )

    def make_element_and_bounds(
        pairs: list[Pair], is_bound: list[bool]
    ) -> tuple[mesh.Hex, list[frozenset[mesh.Quad]]]:
        edges = [
            mesh.Quad(make_line(pairs[0], pairs[1]), trace, a3),
            mesh.Quad(make_line(pairs[2], pairs[3]), trace, a3),
            mesh.Quad(make_line(pairs[0], pairs[2]), trace, a3),
            mesh.Quad(make_line(pairs[1], pairs[3]), trace, a3),
        ]
        return mesh.Hex(*edges), [
            frozenset({e}) if b else frozenset() for e, b in zip(edges, is_bound)
        ]

    def fold(
        x: tuple[list[mesh.Hex], list[frozenset[mesh.Quad]]],
        y: tuple[mesh.Hex, list[frozenset[mesh.Quad]]],
    ) -> tuple[list[mesh.Hex], list[frozenset[mesh.Quad]]]:
        hexes, bounds = x
        hexes.append(y[0])
        new_bounds = [b | y_comp for b, y_comp in zip(bounds, y[1])]
        return hexes, new_bounds

    lower_points = np.linspace(sorted_starts[0], sorted_starts[1], num_hexes_x2 + 1)
    upper_points = np.linspace(sorted_starts[3], sorted_starts[2], num_hexes_x2 + 1)
    points = np.linspace(lower_points, upper_points, num_hexes_x1 + 1)
    initial: tuple[list[mesh.Hex], list[frozenset[mesh.Quad]]] = ([], [frozenset()] * 4)
    return reduce(
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
            # mesh.Hex(
            #     mesh.Quad(make_line(points[i, j], points[i+1, j]), trace, a3),
            #     mesh.Quad(make_line(points[i, j+1], points[i+1, j+1]), trace, a3),
            #     mesh.Quad(make_line(points[i, j], points[i, j+1]), trace, a3),
            #     mesh.Quad(make_line(points[i+1, j], points[i+1, j+1]), trace, a3),
            # )
            for i in range(num_hexes_x1)
            for j in range(num_hexes_x2)
        ),
        initial,
    )


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


def straight_line_for_system(
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
        whole_numbers,
    )


_centre = shared(whole_numbers, key=1)
_rad = shared(non_zero, key=2)
_dx3 = builds(operator.mul, _rad, floats(0.01, 1.99))
_x1_start = tuples(_centre, _rad).map(lambda x: x[0] + x[1])


def curved_line_for_system(
    system: mesh.CoordinateSystem,
) -> SearchStrategy[mesh.FieldAlignedCurve]:
    trace = builds(cylindrical_field_trace, _centre, whole_numbers)

    return builds(
        mesh.FieldAlignedCurve,
        builds(mesh.FieldTracer, trace, integers(2, 10)),
        builds(
            mesh.SliceCoord,
            builds(operator.add, _centre, _rad),
            whole_numbers,
            just(system),
        ),
        _rad.map(abs).flatmap(lambda r: floats(0.01 * r, 1.99 * r)),  # type: ignore
        _divisions,
        _num_divisions,
        whole_numbers,
    )


coordinate_systems = sampled_from(mesh.CoordinateSystem)
straight_line = coordinate_systems.flatmap(straight_line_for_system)
curved_line = sampled_from(list(CARTESIAN_SYSTEMS)).flatmap(curved_line_for_system)
register_type_strategy(
    mesh.FieldAlignedCurve,
    one_of(straight_line, curved_line),
)

T = TypeVar("T")


@composite
def hex_starts(
    draw,
    base_x1: float = 0.0,
    offset_sign: int = 0,
) -> tuple[Pair, Pair, Pair, Pair]:
    if offset_sign == 0:
        if draw(booleans()):
            offset_sign = 1
        else:
            offset_sign = -1
    base_x2 = draw(whole_numbers)
    return (
        (base_x1, base_x2),
        (
            base_x1 + offset_sign * draw(non_zero.map(abs)),  # type: ignore
            base_x2 + draw(whole_numbers),
        ),
        (
            base_x1 + offset_sign * draw(whole_numbers.map(abs)),  # type: ignore
            base_x2 + draw(non_zero),
        ),
        (
            base_x1 + offset_sign * draw(non_zero.map(abs)),  # type: ignore
            base_x2 + draw(non_zero),
        ),
    )


_num_divisions = shared(integers(1, 5), key=171)
_divisions = _num_divisions.flatmap(lambda x: integers(0, x - 1))
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
        integers(2, 5),
        _divisions,
        _num_divisions,
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
    SearchStrategy[mesh.Hex],
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
    sampled_from(list(CARTESIAN_SYSTEMS)),
    integers(100, 200),
    _divisions,
    _num_divisions,
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
    sampled_from(list(CARTESIAN_SYSTEMS)),
    integers(100, 200),
    _divisions,
    _num_divisions,
    whole_numbers,
)

flat_quad = one_of(linear_quad, nonlinear_quad)
quad_in_3d = builds(higher_dim_quad, linear_quad, floats(-np.pi, np.pi)).filter(
    lambda x: x is not None
)
register_type_strategy(mesh.Quad, one_of(flat_quad))  # , quad_in_3d))

flat_sided_hex = one_of(linear_hex, nonlinear_hex)
curve_sided_hex = builds(higher_dim_hex, linear_hex, floats(-np.pi, np.pi)).filter(
    lambda x: x is not None
)
register_type_strategy(mesh.Hex, flat_sided_hex)


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
    ).filter(lambda x: x is not None),
)
quad_mesh_arguments = quad_mesh_elements.map(lambda x: (x, get_quad_boundaries(x)))
quad_mesh_layer_no_divisions = quad_mesh_arguments.map(lambda x: mesh.MeshLayer(*x))
shared_quads = shared(quad_mesh_elements)
quad_mesh_layer = builds(
    mesh.MeshLayer,
    shared_quads,
    shared_quads.map(get_quad_boundaries),
    one_of(none(), whole_numbers),
    integers(1, 3),
)

hex_mesh_arguments = cast(
    SearchStrategy[tuple[list[mesh.Hex], list[frozenset[mesh.Quad]]]],
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
    ).filter(lambda x: x is not None),
)
hex_mesh_layer_no_divisions = hex_mesh_arguments.map(lambda x: mesh.MeshLayer(*x))
shared_hex_mesh_args = shared(hex_mesh_arguments)
hex_mesh_layer = builds(
    mesh.MeshLayer,
    shared_hex_mesh_args.map(lambda x: x[0]),
    shared_hex_mesh_args.map(lambda x: x[1]),
    one_of(none(), whole_numbers),
    integers(1, 3),
)

mesh_arguments = one_of(quad_mesh_arguments, hex_mesh_arguments)

register_type_strategy(mesh.MeshLayer, one_of(quad_mesh_layer, hex_mesh_layer))

x3_offsets = builds(np.linspace, whole_numbers, non_zero, integers(2, 4))
quad_meshes: SearchStrategy[mesh.QuadMesh] = builds(
    mesh.GenericMesh, quad_mesh_layer, x3_offsets
)
hex_meshes: SearchStrategy[mesh.HexMesh] = builds(
    mesh.GenericMesh, hex_mesh_layer, x3_offsets
)
register_type_strategy(
    mesh.GenericMesh, builds(mesh.GenericMesh, from_type(mesh.MeshLayer), x3_offsets)
)
