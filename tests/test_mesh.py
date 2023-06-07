import itertools
from typing import Type, Optional
from unittest.mock import MagicMock

from hypothesis import given
from hypothesis.strategies import (
    builds,
    from_type,
    integers,
    lists,
    one_of,
    shared,
)
import numpy as np
import numpy.typing as npt
import pytest

from neso_fame import mesh

from .mesh_strategies import (
    non_nans,
    coordinate_systems,
    linear_field_trace,
    mesh_connections,
    mutually_broadcastable_arrays,
    non_zero,
    _quad_mesh_connections,
    quad_mesh_layer_no_divisions,
    whole_numbers,
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
    mock.assert_called_once()
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


@given(integers(-50, 100))
def test_curve_subdivision_len(divisions: int) -> None:
    curve = mesh.Curve(
        lambda x: mesh.Coords(
            np.asarray(x), np.asarray(x), np.asarray(x), mesh.CoordinateSystem.Cartesian
        )
    )
    expected = max(1, divisions)
    divisions_iter = curve.subdivide(divisions)
    for _ in range(expected):
        _ = next(divisions_iter)
    with pytest.raises(StopIteration):
        next(divisions_iter)


@given(integers(-5, 100))
def test_curve_subdivision(divisions: int) -> None:
    curve = mesh.Curve(
        lambda x: mesh.Coords(
            np.asarray(x), np.asarray(x), np.asarray(x), mesh.CoordinateSystem.Cartesian
        )
    )
    divisions_iter = curve.subdivide(divisions)
    first = next(divisions_iter)
    coord = first(0.0)
    for c in coord:
        np.testing.assert_allclose(c, 0.0)
    prev = first(1.0)
    for curve in divisions_iter:
        for c, p in zip(curve(0.0), prev):
            np.testing.assert_allclose(c, p)
        prev = curve(1.0)
    for p in prev:
        np.testing.assert_allclose(p, 1.0)


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


@given(from_type(mesh.Quad), integers(-50, 100))
def test_quad_subdivision_len(quad: mesh.Quad, divisions: int) -> None:
    expected = max(1, divisions)
    divisions_iter = quad.subdivide(divisions)
    for _ in range(expected):
        _ = next(divisions_iter)
    with pytest.raises(StopIteration):
        next(divisions_iter)


@given(from_type(mesh.Quad), integers(-5, 100))
def test_quad_subdivision(quad: mesh.Quad, divisions: int) -> None:
    divisions_iter = quad.subdivide(divisions)
    quad_corners = quad.corners()
    first = next(divisions_iter)
    corners = first.corners()
    for c, q in zip(corners, quad_corners):
        np.testing.assert_allclose(c[[0, 2]], q[[0, 2]])
    prev = corners
    for quad in divisions_iter:
        corners = quad.corners()
        for c, p in zip(corners, prev):
            np.testing.assert_allclose(c[[0, 2]], p[[1, 3]])
        prev = corners
    for p, q in zip(prev, quad_corners):
        np.testing.assert_allclose(p[[1, 3]], q[[1, 3]])


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


@given(from_type(mesh.Tet), integers(-50, 100))
def test_tet_subdivision_len(tet: mesh.Tet, divisions: int) -> None:
    expected = max(1, divisions)
    divisions_iter = tet.subdivide(divisions)
    for _ in range(expected):
        _ = next(divisions_iter)
    with pytest.raises(StopIteration):
        next(divisions_iter)


@given(from_type(mesh.Tet), integers(-5, 100))
def test_tet_subdivision(tet: mesh.Tet, divisions: int) -> None:
    divisions_iter = tet.subdivide(divisions)
    tet_corners = tet.corners()
    first = next(divisions_iter)
    corners = first.corners()
    for c, t in zip(corners, tet_corners):
        np.testing.assert_allclose(c[::2], t[::2])
    prev = corners
    for tet in divisions_iter:
        corners = tet.corners()
        for c, p in zip(corners, prev):
            np.testing.assert_allclose(c[::2], p[1::2])
        prev = corners
    for p, t in zip(prev, tet_corners):
        np.testing.assert_allclose(p[1::2], t[1::2])


@given(mesh_connections)
def test_mesh_layer_elements_no_offset(
    connections: dict[mesh.E, dict[mesh.E, bool]]
) -> None:
    layer = mesh.MeshLayer(connections, None)
    for actual, expected in zip(layer, connections):
        assert actual is expected


@given(mesh_connections, non_nans())
def test_mesh_layer_elements_with_offset(
    connections: dict[mesh.E, dict[mesh.E, bool]], offset: float
) -> None:
    layer = mesh.MeshLayer(connections, offset)
    for actual, expected in zip(layer, connections):
        actual_corners = actual.corners()
        expected_corners = expected.offset(offset).corners()
        np.testing.assert_allclose(actual_corners.x1, expected_corners.x1, atol=1e-12)
        np.testing.assert_allclose(actual_corners.x2, expected_corners.x2, atol=1e-12)
        np.testing.assert_allclose(actual_corners.x3, expected_corners.x3, atol=1e-12)


@given(mesh_connections, integers(1, 10))
def test_mesh_layer_elements_with_subdivisions(
    connections: dict[mesh.E, dict[mesh.E, bool]], subdivisions: int
) -> None:
    layer = mesh.MeshLayer(connections, subdivisions=subdivisions)
    expected = frozenset(
        itertools.chain.from_iterable(
            map(
                lambda x: x.corners().iter_points(),
                itertools.chain.from_iterable(
                    map(lambda x: x.subdivide(subdivisions), connections)
                ),
            )
        )
    )
    actual = frozenset(
        itertools.chain.from_iterable(map(lambda x: x.corners().iter_points(), layer))
    )
    assert expected == actual


@given(from_type(mesh.MeshLayer))
def test_mesh_layer_len(layer: mesh.MeshLayer) -> None:
    layer_iter = iter(layer)
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


@given(quad_mesh_layer_no_divisions)
def test_mesh_layer_quads_for_quads(layer: mesh.MeshLayer[mesh.Quad]) -> None:
    assert all(q1 is q2 for q1, q2 in zip(layer, layer.quads()))


def test_mesh_layer_quads_for_tets() -> None:
    # FIXME: Need to implement this
    pass


@given(from_type(mesh.MeshLayer))
def test_mesh_layer_num_unique_corners(layer: mesh.MeshLayer) -> None:
    corners = frozenset(
        itertools.chain.from_iterable(elem.corners().iter_points() for elem in layer)
    )
    assert layer.num_unique_corners == len(corners)


@given(from_type(mesh.MeshLayer), integers(1, 5))
def test_mesh_layer_num_unique_control_points(
    layer: mesh.MeshLayer, order: int
) -> None:
    control_points = frozenset(
        itertools.chain.from_iterable(
            elem.control_points(order).iter_points() for elem in layer
        )
    )
    assert layer.num_unique_control_points(order) == len(control_points)


@given(from_type(mesh.Mesh))
def test_mesh_iter_layers(m: mesh.Mesh) -> None:
    for layer, offset in zip(m.layers(), m.offsets):
        assert layer.reference_elements is m.reference_layer.reference_elements
        assert layer.offset == offset
        assert layer.subdivisions == m.reference_layer.subdivisions


@given(from_type(mesh.Mesh))
def test_mesh_len(m: mesh.Mesh) -> None:
    mesh_iter = itertools.chain.from_iterable(layer for layer in m.layers())
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
                (((i, p) for p in elem.corners().iter_points()) for elem in layer)
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
                    for elem in layer
                )
                for i, layer in enumerate(m.layers())
            )
        )
    )
    assert m.num_unique_control_points(order) == len(control_points)


shared_coords = shared(coordinate_systems, key=0)


@given(
    builds(linear_field_trace, whole_numbers, whole_numbers, non_zero, shared_coords),
    builds(mesh.SliceCoord, non_zero, whole_numbers, shared_coords),
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
