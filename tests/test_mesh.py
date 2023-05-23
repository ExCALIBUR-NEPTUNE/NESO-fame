import itertools
from typing import Type, Optional
from unittest.mock import MagicMock

from hypothesis import given
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
    lists,
    register_type_strategy,
    sampled_from,
    just,
    one_of,
    tuples,
    shared,
)
import numpy as np
import numpy.typing as npt
import pytest

from neso_fame import mesh

non_nans = lambda: floats(allow_nan=False)
arbitrary_arrays = lambda: arrays(floating_dtypes(), array_shapes())


def mutually_broadcastable_arrays(
    num_arrays: int,
) -> SearchStrategy[tuple[npt.NDArray]]:
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
                a1p * np.asarray(x3) + start.x1, a2p * np.asarray(x3) + start.x2, c
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
            a * np.asarray(x) + b1 - 0.5 * a1,
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
        ),
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


@given(non_nans(), non_nans(), coordinate_systems)
def test_slice_coord(x1: float, x2: float, c: mesh.C) -> None:
    coord = mesh.SliceCoord(x1, x2, c)
    coord_iter = iter(coord)
    assert next(coord_iter) == x1
    assert next(coord_iter) == x2
    with pytest.raises(StopIteration):
        next(coord_iter)


@pytest.mark.parametrize(
    "x1,x2,expected",
    [
        (
            np.array([0.0, 1.0, 2.0]),
            np.array([1.0, 0.5, 0.0]),
            [
                mesh.SliceCoord(0.0, 1.0, mesh.CoordinateSystem.Cartesian),
                mesh.SliceCoord(1.0, 0.5, mesh.CoordinateSystem.Cartesian),
                mesh.SliceCoord(2.0, 0.0, mesh.CoordinateSystem.Cartesian),
            ],
        ),
        (
            np.array([5.0, 0.0]),
            np.array([[1.0, 1.5], [-10.0, -11.0]]),
            [
                mesh.SliceCoord(5.0, 1.0, mesh.CoordinateSystem.Cartesian),
                mesh.SliceCoord(0.0, 1.5, mesh.CoordinateSystem.Cartesian),
                mesh.SliceCoord(5.0, -10.0, mesh.CoordinateSystem.Cartesian),
                mesh.SliceCoord(0.0, -11.0, mesh.CoordinateSystem.Cartesian),
            ],
        ),
    ],
)
def test_slice_coords_iter_points(
    x1: npt.NDArray, x2: npt.NDArray, expected: list[mesh.SliceCoord]
) -> None:
    for c1, c2 in zip(
        mesh.SliceCoords(x1, x2, mesh.CoordinateSystem.Cartesian).iter_points(),
        expected,
    ):
        assert c1 == c2


@given(mutually_broadcastable_arrays(2), coordinate_systems)
def test_slice_coords_iter(x: tuple[npt.NDArray, npt.NDArray], c: mesh.C) -> None:
    coords = mesh.SliceCoords(*x, c)
    coords_iter = iter(coords)
    assert np.all(next(coords_iter) == x[0])
    assert np.all(next(coords_iter) == x[1])
    with pytest.raises(StopIteration):
        next(coords_iter)


@given(mutually_broadcastable_arrays(2), coordinate_systems)
def test_slice_coords_len(x: tuple[npt.NDArray, npt.NDArray], c: mesh.C) -> None:
    coords = mesh.SliceCoords(*x, c)
    coords_iter = coords.iter_points()
    for _ in range(len(coords)):
        next(coords_iter)
    with pytest.raises(StopIteration):
        next(coords_iter)


s = mutually_broadcastable_shapes(num_shapes=2, min_dims=1)


@pytest.mark.parametrize(
    "x1,x2,index,expected",
    [
        (
            np.array([0.0, 1.0, 2.0]),
            np.array([1.0, 0.5, 0.0]),
            1,
            mesh.SliceCoord(1.0, 0.5, mesh.CoordinateSystem.Cartesian),
        ),
        (
            np.array([5.0, 0.0]),
            np.array([[1.0, 1.5], [-10.0, -11.0]]),
            (1, 0),
            mesh.SliceCoord(5.0, -10.0, mesh.CoordinateSystem.Cartesian),
        ),
    ],
)
def test_slice_coords_getitem(
    x1: npt.NDArray,
    x2: npt.NDArray,
    index: tuple[int, ...] | int,
    expected: mesh.SliceCoord[mesh.CartesianCoordinates],
) -> None:
    coords = mesh.SliceCoords(x1, x2, mesh.CoordinateSystem.Cartesian)
    coord = coords[index]
    assert coord == expected


@pytest.mark.parametrize(
    "x1,x2,index,expected",
    [
        (np.array([0.0, 1.0, 2.0]), np.array([1.0, 0.5, 0.0]), 4, IndexError),
        (np.array([0.0, 1.0, 2.0]), np.array([1.0, 0.5, 0.0]), slice(0, 0), TypeError),
        (np.array([5.0, 0.0]), np.array([[1.0, 1.5], [-10.0, -11.0]]), 1, TypeError),
    ],
)
def test_slice_coords_bad_getitem(
    x1: npt.NDArray, x2: npt.NDArray, index: int | slice, expected: Type[Exception]
) -> None:
    coords = mesh.SliceCoords(x1, x2, mesh.CoordinateSystem.Cartesian)
    with pytest.raises(expected):
        _ = coords[index]


@given(non_nans(), non_nans(), non_nans(), coordinate_systems)
def test_coord(x1: float, x2: float, x3: float, c: mesh.C) -> None:
    coord = mesh.Coord(x1, x2, x3, c)
    coord_iter = iter(coord)
    assert next(coord_iter) == x1
    assert next(coord_iter) == x2
    assert next(coord_iter) == x3
    with pytest.raises(StopIteration):
        next(coord_iter)


@pytest.mark.parametrize(
    "x1,x2,x3,expected",
    [
        (
            np.array([0.0, 1.0, 2.0]),
            np.array([1.0, 0.5, 0.0]),
            np.array([10.0, -5.0, 1.0]),
            [
                mesh.Coord(0.0, 1.0, 10.0, mesh.CoordinateSystem.Cartesian),
                mesh.Coord(1.0, 0.5, -5.0, mesh.CoordinateSystem.Cartesian),
                mesh.Coord(2.0, 0.0, 1.0, mesh.CoordinateSystem.Cartesian),
            ],
        ),
        (
            np.array([5.0, 0.0]),
            np.array([[1.0, 1.5], [-10.0, -11.0]]),
            np.array([[2.0], [-2.0]]),
            [
                mesh.Coord(5.0, 1.0, 2.0, mesh.CoordinateSystem.Cartesian),
                mesh.Coord(0.0, 1.5, 2.0, mesh.CoordinateSystem.Cartesian),
                mesh.Coord(5.0, -10.0, -2.0, mesh.CoordinateSystem.Cartesian),
                mesh.Coord(0.0, -11.0, -2.0, mesh.CoordinateSystem.Cartesian),
            ],
        ),
    ],
)
def test_coords_iter_points(
    x1: npt.NDArray, x2: npt.NDArray, x3: npt.NDArray, expected: list[mesh.Coord]
) -> None:
    for c1, c2 in zip(
        mesh.Coords(x1, x2, x3, mesh.CoordinateSystem.Cartesian).iter_points(),
        expected,
    ):
        assert c1 == c2


@given(from_type(mesh.Coords))
def test_coords_iter(coords: mesh.Coords) -> None:
    coords_iter = iter(coords)
    assert np.all(next(coords_iter) == coords.x1)
    assert np.all(next(coords_iter) == coords.x2)
    assert np.all(next(coords_iter) == coords.x3)
    with pytest.raises(StopIteration):
        next(coords_iter)


@given(from_type(mesh.Coords))
def test_coords_len(coords: mesh.Coords) -> None:
    coords_iter = coords.iter_points()
    for _ in range(len(coords)):
        _ = next(coords_iter)
    with pytest.raises(StopIteration):
        next(coords_iter)


@pytest.mark.parametrize(
    "x1,x2,x3,index,expected",
    [
        (
            np.array([0.0, 1.0, 2.0]),
            np.array([1.0, 0.5, 0.0]),
            np.array([10.0, -5.0, 1.0]),
            1,
            mesh.Coord(1.0, 0.5, -5.0, mesh.CoordinateSystem.Cartesian),
        ),
        (
            np.array([5.0, 0.0]),
            np.array([[1.0, 1.5], [-10.0, -11.0]]),
            np.array([[2.0], [-2.0]]),
            (1, 0),
            mesh.Coord(5.0, -10.0, -2.0, mesh.CoordinateSystem.Cartesian),
        ),
    ],
)
def test_coords_getitem(
    x1: npt.NDArray,
    x2: npt.NDArray,
    x3: npt.NDArray,
    index: tuple[int, ...] | int,
    expected: mesh.Coord[mesh.CartesianCoordinates],
) -> None:
    coords = mesh.Coords(x1, x2, x3, mesh.CoordinateSystem.Cartesian)
    coord = coords[index]
    assert coord == expected


@pytest.mark.parametrize(
    "x1,x2,x3,index,expected",
    [
        (
            np.array([0.0, 1.0, 2.0]),
            np.array([1.0, 0.5, 0.0]),
            np.array([10.0, -5.0, 1.0]),
            4,
            IndexError,
        ),
        (
            np.array([0.0, 1.0, 2.0]),
            np.array([1.0, 0.5, 0.0]),
            np.array([10.0, -5.0, 1.0]),
            slice(0, 0),
            TypeError,
        ),
        (
            np.array([5.0, 0.0]),
            np.array([[1.0, 1.5], [-10.0, -11.0]]),
            np.array([[2.0], [-2.0]]),
            1,
            TypeError,
        ),
    ],
)
def test_coords_bad_getitem(
    x1: npt.NDArray,
    x2: npt.NDArray,
    x3: npt.NDArray,
    index: int | slice,
    expected: Type[Exception],
) -> None:
    coords = mesh.Coords(x1, x2, x3, mesh.CoordinateSystem.Cartesian)
    with pytest.raises(expected):
        _ = coords[index]


@pytest.mark.filterwarnings("ignore:invalid value:RuntimeWarning")
@given(from_type(mesh.Coords), non_nans())
def test_coords_offset_x1_x2_unchanged(coords: mesh.Coords, x: float) -> None:
    new_coords = coords.offset(x)
    assert new_coords.x1 is coords.x1
    assert new_coords.x2 is coords.x2
    assert new_coords.x3 is not coords.x3
    assert new_coords.system is coords.system


@pytest.mark.parametrize(
    "x1,x2,x3,offset,expected",
    [
        (
            np.array([0.0, 1.0, 2.0]),
            np.array([1.0, 0.5, 0.0]),
            np.array([10.0, -5.0, 1.0]),
            5.0,
            np.array([15.0, 0.0, 6.0]),
        ),
        (
            np.array([5.0, 0.0]),
            np.array([[1.0, 1.5], [-10.0, -11.0]]),
            np.array([[2.0], [-2.0]]),
            np.array([1.0, 2.0]),
            np.array([[3.0, 4.0], [-1.0, 0.0]]),
        ),
    ],
)
def test_coords_offset(
    x1: npt.NDArray,
    x2: npt.NDArray,
    x3: npt.NDArray,
    offset: tuple[int, ...] | int,
    expected: npt.NDArray,
) -> None:
    coords = mesh.Coords(x1, x2, x3, mesh.CoordinateSystem.Cartesian)
    new_coords = coords.offset(offset)
    np.testing.assert_allclose(new_coords.x3, expected)


@pytest.mark.filterwarnings("ignore:invalid value:RuntimeWarning")
@given(mutually_broadcastable_arrays(3))
def test_coords_cartesian_to_cartesian(
    xs: tuple[npt.NDArray, npt.NDArray, npt.NDArray]
) -> None:
    coords = mesh.Coords(*xs, mesh.CoordinateSystem.Cartesian)
    new_coords = coords.to_cartesian()
    assert coords.x1 is new_coords.x1
    assert coords.x2 is new_coords.x2
    assert coords.x3 is new_coords.x3


@pytest.mark.filterwarnings("ignore:invalid value:RuntimeWarning")
@given(from_type(mesh.Coords))
def test_coords_cartesian_correct_system(coords: mesh.Coords) -> None:
    assert coords.to_cartesian().system is mesh.CoordinateSystem.Cartesian


@pytest.mark.filterwarnings("ignore:invalid value:RuntimeWarning")
@given(mutually_broadcastable_arrays(3))
def test_coords_cylindrical_to_cartesian_z_unchanged(
    xs: tuple[npt.NDArray, npt.NDArray, npt.NDArray]
) -> None:
    coords = mesh.Coords(*xs, mesh.CoordinateSystem.Cylindrical)
    assert np.all(coords.to_cartesian().x3 == coords.x2)


def test_coords_to_cartesian() -> None:
    coords = mesh.Coords(
        np.array([1.0, 1.5, 2.0]),
        np.array([1.0, 0.5, 0.0]),
        np.array([0.0, np.pi / 2.0, np.pi]),
        mesh.CoordinateSystem.Cylindrical,
    ).to_cartesian()
    np.testing.assert_allclose(coords.x1, [1.0, 0.0, -2.0], atol=1e-12)
    np.testing.assert_allclose(coords.x2, [0.0, 1.5, 0.0], atol=1e-12)
    np.testing.assert_allclose(coords.x3, [1.0, 0.5, 0.0], atol=1e-12)


def test_curve_call() -> None:
    mock = MagicMock()
    curve = mesh.Curve(mock)
    result = curve(1.0)
    mock.assert_called_once_with(1.0)
    assert result is mock.return_value


@given(
    one_of((non_nans(), lists(non_nans(), min_size=1, max_size=10))),
    from_type(mesh.Coords),
)
def test_curve_offset_0(arg: float | list[float], result: mesh.Coords) -> None:
    mock = MagicMock(return_value=result)
    curve = mesh.Curve(mock).offset(0.0)
    result = curve(arg)
    mock.assert_called_once_with(arg)
    assert np.all(result.x1 == mock.return_value.x1)
    assert np.all(result.x2 == mock.return_value.x2)
    assert np.all(result.x3 == mock.return_value.x3)
    assert result.system == mock.return_value.system


@pytest.mark.parametrize(
    "arg,offset,expected",
    [
        (1.0, 1.0, 2.0),
        (np.array([0.0, 0.5, 1.0]), 10.0, np.array([10.0, 10.5, 11.0])),
        (np.array([0.3, 0.1, 0.8]), -0.3, np.array([0.0, -0.2, 0.5])),
    ],
)
def test_curve_offset(arg: float, offset: float, expected: float) -> None:
    result = mesh.Curve(
        lambda x: mesh.Coords(
            np.asarray(x), np.asarray(x), np.asarray(x), mesh.CoordinateSystem.Cartesian
        )
    ).offset(offset)(arg)
    assert np.all(result.x1 == arg)
    assert np.all(result.x2 == arg)
    np.testing.assert_allclose(result.x3, expected)
    assert result.system is mesh.CoordinateSystem.Cartesian


def test_curve_control_points_cached() -> None:
    curve = mesh.Curve(
        lambda x: mesh.Coords(
            np.asarray(x), np.asarray(x), np.asarray(x), mesh.CoordinateSystem.Cartesian
        )
    )
    p1 = curve.control_points(2)
    p2 = curve.control_points(2)
    assert p1 is p2


@given(from_type(mesh.Curve), integers(1, 10))
def test_curve_control_points_size(curve: mesh.Curve, n: int) -> None:
    assert len(curve.control_points(n)) == n + 1


def test_curve_control_points_values() -> None:
    a = 2.0
    b = 0.5
    c = -1.0
    curve = mesh.Curve(
        lambda x: mesh.Coords(
            np.asarray(x) * a,
            np.asarray(x) * b,
            np.asarray(x) * c,
            mesh.CoordinateSystem.Cartesian,
        )
    )
    x1, x2, x3 = curve.control_points(2)
    np.testing.assert_allclose(x1, [0.0, 1.0, 2.0], atol=1e-12)
    np.testing.assert_allclose(x2, [0.0, 0.25, 0.5], atol=1e-12)
    np.testing.assert_allclose(x3, [0.0, -0.5, -1.0], atol=1e-12)


@given(from_type(mesh.Quad))
def test_quad_from_unordered_curves(original: mesh.Quad) -> None:
    q1 = mesh.Quad.from_unordered_curves(
        original.north, original.south, None, original.field
    )
    q2 = mesh.Quad.from_unordered_curves(
        original.south, original.north, None, original.field
    )
    assert q1.north is original.north or q1.north is original.south
    assert q1.south is original.north or q1.south is original.south
    assert q1.in_plane is None
    assert q1.field is original.field
    assert q1 is q2


@given(from_type(mesh.Quad))
def test_quad_corners(q: mesh.Quad) -> None:
    corners = q.corners()
    assert corners[0] == next(q.north(0.0).iter_points())
    assert corners[1] == next(q.north(1.0).iter_points())
    assert corners[2] == next(q.south(0.0).iter_points())
    assert corners[3] == next(q.south(1.0).iter_points())


@given(from_type(mesh.Quad), integers(1, 5))
def test_quad_control_points_within_corners(q: mesh.Quad, n: int) -> None:
    corners = q.corners()
    x1_max, x2_max, x3_max = map(np.max, corners)
    x1_min, x2_min, x3_min = map(np.min, corners)
    cp = q.control_points(n)
    assert len(cp) == (n + 1) ** 2
    assert cp.x1.ndim == 2
    assert cp.x2.ndim == 2
    assert cp.x2.ndim == 2
    assert np.all(cp.x1 <= x1_max)
    assert np.all(cp.x2 <= x2_max)
    assert np.all(cp.x3 <= x3_max)
    assert np.all(cp.x1 >= x1_min)
    assert np.all(cp.x2 >= x2_min)
    assert np.all(cp.x3 >= x3_min)
    # TODO: Check these are equally spaced along trace (easy in x3 direction, harder in others)
    # TODO: Write a check for some known values
    pass


@given(from_type(mesh.Quad), whole_numbers, integers(1, 5))
def test_quad_offset(q: mesh.Quad, x: float, n: int) -> None:
    actual = q.offset(x).control_points(n)
    expected = q.control_points(n).offset(x)
    np.testing.assert_allclose(actual.x1, expected.x1, atol=1e-12)
    np.testing.assert_allclose(actual.x2, expected.x2, atol=1e-12)
    np.testing.assert_allclose(actual.x3, expected.x3, atol=1e-12)
    assert actual.system == expected.system


def test_tet_corners() -> None:
    pass


def test_tet_control_points_cached() -> None:
    pass


def test_tet_control_points_size() -> None:
    pass


def test_tet_control_points_values() -> None:
    pass


def test_tet_get_quads() -> None:
    pass


def test_tet_offset() -> None:
    pass


@given(mesh_connections)
def test_mesh_layer_elements_no_offset(
    connections: dict[mesh.E, dict[mesh.E, bool]]
) -> None:
    layer = mesh.MeshLayer(connections, None)
    for actual, expected in zip(layer.elements(), connections):
        assert actual is expected


@given(mesh_connections, non_nans())
def test_mesh_layer_elements_with_offset(
    connections: dict[mesh.E, dict[mesh.E, bool]], offset: float
) -> None:
    layer = mesh.MeshLayer(connections, offset)
    for actual, expected in zip(layer.elements(), connections):
        actual_corners = actual.corners()
        expected_corners = expected.offset(offset).corners()
        np.testing.assert_allclose(actual_corners.x1, expected_corners.x1, atol=1e-12)
        np.testing.assert_allclose(actual_corners.x2, expected_corners.x2, atol=1e-12)
        np.testing.assert_allclose(actual_corners.x3, expected_corners.x3, atol=1e-12)


@given(from_type(mesh.MeshLayer))
def test_mesh_layer_len(layer: mesh.MeshLayer) -> None:
    layer_iter = iter(layer.elements())
    for _ in range(len(layer)):
        next(layer_iter)
    with pytest.raises(StopIteration):
        next(layer_iter)


quad_mesh_connectivity = (
    _quad_mesh_connections(
        1.0,
        1.0,
        10.0,
        ((0.0, 0.0, 0.0), (1.0, 0.0, 10.0)),
        4,
        mesh.CoordinateSystem.Cartesian,
    ),
)


@pytest.mark.parametrize(
    "connections",
    [
        (quad_mesh_connectivity,),
        # FIXME: Check for Tet-mesh
    ],
)
def test_mesh_layer_element_type(connections: dict[mesh.E, dict[mesh.E, bool]]) -> None:
    layer = mesh.MeshLayer(connections)
    element = next(iter(connections))
    assert layer.element_type is type(element)


@given(quad_mesh_layer)
def test_mesh_layer_quads_for_quads(layer: mesh.MeshLayer[mesh.Quad]) -> None:
    assert all(q1 is q2 for q1, q2 in zip(layer.elements(), layer.quads()))


def test_mesh_layer_quads_for_tets() -> None:
    # FIXME: Need to implement this
    pass


@given(from_type(mesh.MeshLayer))
def test_mesh_layer_num_unique_corners(layer: mesh.MeshLayer) -> None:
    corners = frozenset(
        itertools.chain.from_iterable(
            elem.corners().iter_points() for elem in layer.elements()
        )
    )
    assert layer.num_unique_corners == len(corners)


@given(from_type(mesh.MeshLayer), integers(1, 5))
def test_mesh_layer_num_unique_control_points(
    layer: mesh.MeshLayer, order: int
) -> None:
    control_points = frozenset(
        itertools.chain.from_iterable(
            elem.control_points(order).iter_points() for elem in layer.elements()
        )
    )
    assert layer.num_unique_control_points(order) == len(control_points)


@given(from_type(mesh.Mesh))
def test_mesh_iter_layers(m: mesh.Mesh) -> None:
    for layer, offset in zip(m.layers(), m.offsets):
        assert layer.reference_elements is m.reference_layer.reference_elements
        assert layer.offset == offset


@given(from_type(mesh.Mesh))
def test_mesh_len(m: mesh.Mesh) -> None:
    mesh_iter = itertools.chain.from_iterable(layer.elements() for layer in m.layers())
    for _ in range(len(m)):
        next(mesh_iter)
    with pytest.raises(StopIteration):
        next(mesh_iter)


@given(from_type(mesh.Mesh))
def test_mesh_num_unique_corners(m: mesh.Mesh) -> None:
    # Create a set of tuples of layer number and control point. Using
    # these tuples ensures any coincident points on adjacent layers
    # won't be treated as the same.
    corners = frozenset(
        itertools.chain.from_iterable(
            itertools.chain.from_iterable(
                (
                    ((i, p) for p in elem.corners().iter_points())
                    for elem in layer.elements()
                )
                for i, layer in enumerate(m.layers())
            )
        )
    )
    assert m.num_unique_corners == len(corners)


@given(from_type(mesh.Mesh), integers(1, 5))
def test_mesh_num_unique_control_points(m: mesh.Mesh, order: int) -> None:
    # Create a set of tuples of layer number and control point. Using
    # these tuples ensures any coincident points on adjacent layers
    # won't be treated as the same.
    control_points = frozenset(
        itertools.chain.from_iterable(
            itertools.chain.from_iterable(
                (
                    ((i, p) for p in elem.control_points(order).iter_points())
                    for elem in layer.elements()
                )
                for i, layer in enumerate(m.layers())
            )
        )
    )
    assert m.num_unique_control_points(order) == len(control_points)


scoords = shared(coordinate_systems, key=0)


@given(
    builds(linear_field_trace, whole_numbers, whole_numbers, non_zero, scoords),
    builds(mesh.SliceCoord, non_zero, whole_numbers, scoords),
    whole_numbers,
    non_zero,
    integers(2, 8),
    integers(3, 12),
)
def test_normalise_field_line(
    trace: mesh.FieldTrace,
    start: mesh.SliceCoord,
    x3_start: float,
    dx3: float,
    resolution: int,
    n: int,
) -> None:
    print(trace(start, x3_start), trace(start, x3_start + dx3))
    normalised = mesh.normalise_field_line(
        trace, start, x3_start, x3_start + dx3, resolution
    )
    checkpoints = np.linspace(0.0, 1.0, n)
    coords_normed = normalised(checkpoints)
    coords_trace, distances = trace(start, coords_normed.x3)
    np.testing.assert_allclose(coords_trace.x1, coords_normed.x1, atol=1e-7)
    np.testing.assert_allclose(coords_trace.x2, coords_normed.x2, atol=1e-7)
    spacing = distances[1:] - distances[:-1]
    np.testing.assert_allclose(spacing, spacing[0], atol=1e-7)


@given(
    from_type(mesh.Curve), builds(np.linspace, just(0.0), just(1.0), integers(3, 10))
)
def test_make_lagrange_interpolation_linear(
    curve: mesh.Curve, samples: npt.NDArray
) -> None:
    start = curve(0.0).to_cartesian()
    end = curve(1.0).to_cartesian()
    lagrange = mesh.make_lagrange_interpolation(curve)
    actual = lagrange(samples)
    np.testing.assert_allclose(actual.x1, start.x1 + (end.x1 - start.x1) * samples)
    np.testing.assert_allclose(actual.x2, start.x2 + (end.x2 - start.x2) * samples)
    np.testing.assert_allclose(actual.x3, start.x3 + (end.x3 - start.x3) * samples)


@given(
    whole_numbers,
    whole_numbers,
    whole_numbers,
    whole_numbers,
    whole_numbers,
    whole_numbers,
    whole_numbers,
    non_zero,
    whole_numbers,
    integers(3, 10).map(lambda n: np.linspace(0.0, 1.0, n)),
)
def test_make_lagrange_interpolation_quadratic(
    a1: float,
    b1: float,
    c1: float,
    a2: float,
    b2: float,
    c2: float,
    a3: float,
    b3: float,
    c3: float,
    samples: npt.NDArray,
) -> None:
    def func(s: npt.ArrayLike) -> mesh.Coords[mesh.CartesianCoordinates]:
        s = np.asarray(s)
        return mesh.Coords(
            a1 * s**2 + b1 * s + c1,
            a2 * s**2 + b2 * s + c2,
            a3 * s**2 + b3 * s + c3,
            mesh.CoordinateSystem.Cartesian,
        )

    lagrange = mesh.make_lagrange_interpolation(mesh.Curve(func), 2)
    actual = lagrange(samples)
    expected = func(samples)
    np.testing.assert_allclose(actual.x1, expected.x1)
    np.testing.assert_allclose(actual.x2, expected.x2)
    np.testing.assert_allclose(actual.x3, expected.x3)


@given(from_type(mesh.Curve), integers(1, 10))
def test_make_lagrange_interpolation_knots(curve: mesh.Curve, order: int) -> None:
    lagrange = mesh.make_lagrange_interpolation(curve, order)
    samples = np.linspace(0.0, 1.0, order + 1)
    actual = lagrange(samples)
    expected = curve(samples).to_cartesian()
    np.testing.assert_allclose(actual.x1, expected.x1)
    np.testing.assert_allclose(actual.x2, expected.x2)
    np.testing.assert_allclose(actual.x3, expected.x3)
