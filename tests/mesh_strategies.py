import itertools

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
    register_type_strategy,
    sampled_from,
    just,
    one_of,
    tuples,
)
import numpy as np
import numpy.typing as npt

from neso_fame import mesh

non_nans = lambda: floats(allow_nan=False)
arbitrary_arrays = lambda: arrays(floating_dtypes(), array_shapes())


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
                arrays(np.float64, just(s), elements={"allow_nan": False})
                for s in shapes.input_shapes
            )
        )

    return strategy.flatmap(shape_to_array)


register_type_strategy(
    mesh.SliceCoords,
    builds(
        lambda xs, c: mesh.SliceCoords(xs[0], xs[1], c),
        mutually_broadcastable_arrays(2),
        sampled_from(mesh.CoordinateSystem),
    ),
)

register_type_strategy(
    mesh.Coords,
    builds(
        lambda xs, c: mesh.Coords(xs[0], xs[1], xs[2], c),
        mutually_broadcastable_arrays(3),
        sampled_from(mesh.CoordinateSystem),
    ),
)


def linear_field_trace(a1: float, a2: float, a3: float, c: mesh.C) -> mesh.FieldTrace:
    a1p = a1 / a3 if c == mesh.CoordinateSystem.Cartesian else 0.0
    a2p = a2 / a3

    def cartesian_func(
        start: mesh.SliceCoord[mesh.CartesianCoordinates], x3: npt.ArrayLike
    ) -> tuple[mesh.SliceCoords, npt.NDArray]:
        if c == mesh.CoordinateSystem.Cartesian:
            s = np.sqrt(a1p * a1p + a2p * a2p + 1) * np.asarray(x3)
        else:
            s = np.sqrt(a1p * a1p + a2p * a2p + start.x1 * start.x1) * np.asarray(x3)
        return (
            mesh.SliceCoords(
                a1p * np.asarray(x3) + start.x1,
                a2p * np.asarray(x3) + start.x2,
                start.system,
            ),
            s,
        )

    return cartesian_func


def linear_field_line(
    a1: float, a2: float, a3: float, b1: float, b2: float, b3: float, c: mesh.C
) -> mesh.NormalisedFieldLine:
    def linear_func(x: npt.ArrayLike) -> mesh.Coords:
        a = a1 if c == mesh.CoordinateSystem.Cartesian else 0.0
        return mesh.Coords(
            a * np.asarray(x) + b1 - 0.5 * a,
            a2 * np.asarray(x) + b2 - 0.5 * a2,
            a3 * np.asarray(x) + b3 - 0.5 * a3,
            c,
        )

    return linear_func


def flat_quad(
    a1: float,
    a2: float,
    a3: float,
    starts: tuple[tuple[float, float, float], tuple[float, float, float]],
    c: mesh.C,
) -> mesh.Quad:
    trace = linear_field_trace(a1, a2, a3, c)
    north = mesh.Curve(linear_field_line(a1, a2, a3, *starts[0], c))
    south = mesh.Curve(linear_field_line(a1, a2, a3, *starts[1], c))
    return mesh.Quad(north, south, None, trace)


def _quad_mesh_connections(
    a1: float,
    a2: float,
    a3: float,
    limits: tuple[tuple[float, float, float], tuple[float, float, float]],
    num_quads: int,
    c: mesh.C,
) -> dict[mesh.Quad, dict[mesh.Quad, bool]]:
    trace = linear_field_trace(a1, a2, a3, c)
    starts = np.linspace(limits[0], limits[1], num_quads + 1)
    quads = [
        mesh.Quad(c1, c2, None, trace)
        for c1, c2 in itertools.pairwise(
            mesh.Curve(linear_field_line(a1, a2, a3, s[0], s[1], s[2], c))
            for s in starts
        )
    ]
    return {
        q: {quads[i + 1]: False}
        if i == 0
        else {quads[i - 1]: False}
        if i == num_quads - 1
        else {quads[i + 1]: False, quads[i - 1]: False}
        for i, q in enumerate(quads)
    }


def _get_end_point(
    start: tuple[float, float, float], distance: float, angle: float
) -> tuple[float, float, float]:
    return (
        start[0] + distance * np.cos(angle),
        start[1] + distance * np.sin(angle),
        start[2],
    )


coordinate_systems = sampled_from(mesh.CoordinateSystem)
whole_numbers = integers(-1000, 1000).map(float)
nonnegative_numbers = integers(1, 1000).map(float)
non_zero = whole_numbers.filter(lambda x: x != 0.0)
register_type_strategy(
    mesh.Curve,
    builds(
        linear_field_line,
        whole_numbers,
        whole_numbers,
        non_zero,
        whole_numbers,
        whole_numbers,
        whole_numbers,
        coordinate_systems,
    ).map(mesh.Curve),
)
register_type_strategy(
    mesh.Quad,
    builds(
        flat_quad,
        whole_numbers,
        whole_numbers,
        non_zero,
        tuples(
            tuples(whole_numbers, whole_numbers, whole_numbers),
            tuples(whole_numbers, whole_numbers, whole_numbers),
        ).filter(lambda x: x[0][0:2] != x[1][0:2]),
        coordinate_systems,
    ).filter(lambda x: x is not None),
)

starts_and_ends = tuples(
    tuples(whole_numbers, whole_numbers, whole_numbers),
    floats(1.0, 1e3),
    floats(0.0, 2 * np.pi),
).map(lambda s: (s[0], _get_end_point(*s)))
quad_mesh_connections = builds(
    _quad_mesh_connections,
    whole_numbers,
    whole_numbers,
    non_zero,
    starts_and_ends,
    integers(2, 8),
    coordinate_systems,
)
mesh_connections = one_of(quad_mesh_connections)
quad_mesh_layer = quad_mesh_connections.map(mesh.MeshLayer[mesh.Quad])

# TODO: Create strategy for Tet meshes and make them an option when generating meshes
register_type_strategy(mesh.MeshLayer, quad_mesh_layer)

x3_offsets = builds(np.linspace, whole_numbers, non_zero, integers(2, 4))
register_type_strategy(
    mesh.Mesh, builds(mesh.Mesh, from_type(mesh.MeshLayer), x3_offsets)
)