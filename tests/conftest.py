import itertools
import operator

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
    builds,
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
    return builds(
        mesh.SliceCoord,
        whole_numbers,
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
    centre: tuple[float, float],
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
    centre: tuple[float, float],
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
    starts: tuple[tuple[float, float], tuple[float, float]],
    c: mesh.CoordinateSystem,
    skew: float,
    resolution: int,
    division: int,
    num_divisions: int,
    offset: float,
) -> mesh.Quad:
    centre = (
        starts[0][0] + (starts[0][0] - starts[1][0]) / 2,
        starts[0][1] + (starts[0][1] - starts[1][1]) / 2,
    )
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
    x3_limits: tuple[float, float],
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
    x1_start: tuple[float, float],
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


def higher_dim_quad(q: mesh.Quad, angle: float) -> mesh.Quad:
    # This assumes that dx3/ds is an even function about the starting
    # x3 point from which the bounding field lines were projected.
    north = q.north(0.5)
    south = q.south(0.5)
    np.testing.assert_allclose(north.x3, south.x3, atol=1e-10, rtol=1e-10)
    x1_1 = float(north.x1)
    x2_1 = float(north.x2)
    x1_2 = float(south.x1)
    x2_2 = float(south.x2)
    dx1 = x1_1 - x1_2
    dx2 = x2_1 - x2_2
    r = (dx1 * dx1 + dx2 * dx2) / (2 * dx1 * np.cos(angle) + 2 * dx2 * np.sin(angle))
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

    return mesh.Quad(
        curve,
        q.field,
        q.dx3,
        q.subdivision,
        q.num_divisions,
        q.x3_offset,
    )


def _quad_mesh_elements(
    a1: float,
    a2: float,
    a3: float,
    limits: tuple[tuple[float, float], tuple[float, float]],
    num_quads: int,
    c: mesh.CoordinateSystem,
    resolution: int,
    division: int,
    num_divisions: int,
    offset: float,
) -> list[mesh.Quad]:
    trace = mesh.FieldTracer(linear_field_trace(a1, a2, a3, c, 0, (0, 0)), resolution)

    def make_shape(
        starts: tuple[tuple[float, float], tuple[float, float]]
    ) -> mesh.AcrossFieldCurve:
        return mesh.StraightLineAcrossField(
            mesh.SliceCoord(starts[0][0], starts[0][1], c),
            mesh.SliceCoord(starts[1][0], starts[1][1], c),
        )

    points = np.linspace(limits[0], limits[1], num_quads + 1)
    return [
        mesh.Quad(shape, trace, a3, division, num_divisions, offset)
        for shape in map(make_shape, itertools.pairwise(points))
    ]


def get_boundaries(
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
_dx3 = builds(operator.mul, _rad, floats(0.01, 2.0))
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
        _rad.map(abs).flatmap(lambda r: floats(0, 2 * r)),  # type: ignore
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


_num_divisions = shared(integers(1, 5), key=171)
_divisions = _num_divisions.flatmap(lambda x: integers(0, x - 1))
linear_quad = builds(
    trapezoidal_quad,
    whole_numbers,
    whole_numbers,
    non_zero,
    tuples(
        tuples(non_zero, whole_numbers),
        tuples(non_zero, whole_numbers),
    ).filter(lambda x: x[0] != x[1]),
    coordinate_systems,
    floats(-2.0, 2.0),
    integers(2, 5),
    _divisions,
    _num_divisions,
    whole_numbers,
).filter(lambda x: len(frozenset(x.corners().to_cartesian().iter_points())) == 4)
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
    integers(40, 80),
    _divisions,
    _num_divisions,
    whole_numbers,
)
flat_quad = one_of(linear_quad, nonlinear_quad)
quad_in_3d = builds(higher_dim_quad, flat_quad, floats(0.0, 2 * np.pi))
register_type_strategy(mesh.Quad, one_of(flat_quad, quad_in_3d))


starts_and_ends = tuples(
    tuples(whole_numbers, whole_numbers, whole_numbers),
    floats(1.0, 1e3),
    floats(0.0, 2 * np.pi),
).map(lambda s: (s[0], _get_end_point(*s)))
quad_mesh_elements = builds(
    _quad_mesh_elements,
    whole_numbers,
    whole_numbers,
    non_zero,
    starts_and_ends,
    integers(1, 4),
    coordinate_systems,
    integers(2, 10),
    _divisions,
    _num_divisions,
    whole_numbers,
)
quad_mesh_arguments = quad_mesh_elements.map(lambda x: (x, get_boundaries(x)))
mesh_arguments = one_of(quad_mesh_arguments)
quad_mesh_layer_no_divisions = quad_mesh_arguments.map(lambda x: mesh.MeshLayer(*x))
shared_quads = shared(quad_mesh_elements)
quad_mesh_layer = builds(
    mesh.MeshLayer,
    shared_quads,
    shared_quads.map(get_boundaries),
    one_of(none(), whole_numbers),
    integers(1, 3),
)

# TODO: Create strategy for Tet meshes and make them an option when generating meshes
register_type_strategy(mesh.MeshLayer, quad_mesh_layer)

x3_offsets = builds(np.linspace, whole_numbers, non_zero, integers(2, 4))
register_type_strategy(
    mesh.GenericMesh, builds(mesh.GenericMesh, from_type(mesh.MeshLayer), x3_offsets)
)
