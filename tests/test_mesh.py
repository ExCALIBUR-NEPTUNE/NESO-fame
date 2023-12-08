import itertools
from typing import Iterable, Type, cast

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given, settings
from hypothesis.strategies import (
    builds,
    floats,
    from_type,
    integers,
    sampled_from,
    shared,
    tuples,
)
from scipy.optimize import minimize_scalar

from neso_fame import mesh
from neso_fame.offset import (
    Offset,
)

from .conftest import (
    CARTESIAN_SYSTEMS,
    _hex_mesh_arguments,
    _quad_mesh_elements,
    coordinate_systems,
    cylindrical_field_line,
    cylindrical_field_trace,
    flat_sided_hex,
    linear_field_trace,
    mesh_arguments,
    mutually_broadcastable_arrays,
    non_nans,
    non_zero,
    quad_mesh_layer_no_divisions,
    whole_numbers,
)


@given(non_nans(), non_nans(), coordinate_systems)
def test_slice_coord(x1: float, x2: float, c: mesh.CoordinateSystem) -> None:
    coord = mesh.SliceCoord(x1, x2, c)
    coord_iter = iter(coord)
    assert next(coord_iter) == x1
    assert next(coord_iter) == x2
    with pytest.raises(StopIteration):
        next(coord_iter)


@pytest.mark.parametrize(
    "coord1,coord2,places",
    [
        (
            mesh.SliceCoord(1.0, 1.0, mesh.CoordinateSystem.CARTESIAN),
            mesh.SliceCoord(1.0001, 0.99999, mesh.CoordinateSystem.CARTESIAN),
            4,
        ),
        (
            mesh.SliceCoord(3314.44949999, 0.0012121, mesh.CoordinateSystem.CARTESIAN),
            mesh.SliceCoord(
                3314.44950001, 0.0012121441, mesh.CoordinateSystem.CARTESIAN
            ),
            5,
        ),
        (
            mesh.SliceCoord(1.0, 1.0, mesh.CoordinateSystem.CYLINDRICAL),
            mesh.SliceCoord(1.0001, 0.99999, mesh.CoordinateSystem.CYLINDRICAL),
            4,
        ),
    ],
)
def test_slice_coord_round(
    coord1: mesh.SliceCoord, coord2: mesh.SliceCoord, places: int
) -> None:
    assert coord1 != coord2
    assert coord1.round_to(places) == coord2.round_to(places)
    assert coord1.round_to(places + 1) != coord2.round_to(places + 1)


@pytest.mark.parametrize(
    "x1,x2,expected",
    [
        (
            np.array([0.0, 1.0, 2.0]),
            np.array([1.0, 0.5, 0.0]),
            [
                mesh.SliceCoord(0.0, 1.0, mesh.CoordinateSystem.CARTESIAN),
                mesh.SliceCoord(1.0, 0.5, mesh.CoordinateSystem.CARTESIAN),
                mesh.SliceCoord(2.0, 0.0, mesh.CoordinateSystem.CARTESIAN),
            ],
        ),
        (
            np.array([5.0, 0.0]),
            np.array([[1.0, 1.5], [-10.0, -11.0]]),
            [
                mesh.SliceCoord(5.0, 1.0, mesh.CoordinateSystem.CARTESIAN),
                mesh.SliceCoord(0.0, 1.5, mesh.CoordinateSystem.CARTESIAN),
                mesh.SliceCoord(5.0, -10.0, mesh.CoordinateSystem.CARTESIAN),
                mesh.SliceCoord(0.0, -11.0, mesh.CoordinateSystem.CARTESIAN),
            ],
        ),
    ],
)
def test_slice_coords_iter_points(
    x1: npt.NDArray, x2: npt.NDArray, expected: list[mesh.SliceCoord]
) -> None:
    for c1, c2 in zip(
        mesh.SliceCoords(x1, x2, mesh.CoordinateSystem.CARTESIAN).iter_points(),
        expected,
    ):
        assert c1 == c2


@given(mutually_broadcastable_arrays(2), coordinate_systems)
def test_slice_coords_iter(
    x: tuple[npt.NDArray, npt.NDArray], c: mesh.CoordinateSystem
) -> None:
    coords = mesh.SliceCoords(*x, c)
    coords_iter = iter(coords)
    assert np.all(next(coords_iter) == x[0])
    assert np.all(next(coords_iter) == x[1])
    with pytest.raises(StopIteration):
        next(coords_iter)


@given(mutually_broadcastable_arrays(2), coordinate_systems)
def test_slice_coords_len(
    x: tuple[npt.NDArray, npt.NDArray], c: mesh.CoordinateSystem
) -> None:
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
            mesh.SliceCoord(1.0, 0.5, mesh.CoordinateSystem.CARTESIAN),
        ),
        (
            np.array([5.0, 0.0]),
            np.array([[1.0, 1.5], [-10.0, -11.0]]),
            (1, 0),
            mesh.SliceCoord(5.0, -10.0, mesh.CoordinateSystem.CARTESIAN),
        ),
    ],
)
def test_slice_coords_getitem(
    x1: npt.NDArray,
    x2: npt.NDArray,
    index: tuple[int, ...] | int,
    expected: mesh.SliceCoord,
) -> None:
    coords = mesh.SliceCoords(x1, x2, mesh.CoordinateSystem.CARTESIAN)
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
    coords = mesh.SliceCoords(x1, x2, mesh.CoordinateSystem.CARTESIAN)
    with pytest.raises(expected):
        _ = coords[index]  # type: ignore


@pytest.mark.parametrize(
    "coord1,coord2,places",
    [
        (
            mesh.SliceCoords(
                np.array([1.0, 5.0]),
                np.array([1.0, 1.0]),
                mesh.CoordinateSystem.CARTESIAN,
            ),
            mesh.SliceCoords(
                np.array([1.0001, 5.0]),
                np.array([0.99999, 1.0]),
                mesh.CoordinateSystem.CARTESIAN,
            ),
            4,
        ),
        (
            mesh.SliceCoords(
                np.array([3315.05, 1e-12]),
                np.array([0.0012121, 1e100]),
                mesh.CoordinateSystem.CARTESIAN,
            ),
            mesh.SliceCoords(
                np.array([3315.05, 0.0]),
                np.array([0.0012121441, 1.0000055e100]),
                mesh.CoordinateSystem.CARTESIAN,
            ),
            5,
        ),
        (
            mesh.SliceCoords(
                np.array(1.0), np.array(1.0), mesh.CoordinateSystem.CYLINDRICAL
            ),
            mesh.SliceCoords(
                np.array(1.0001), np.array(0.99999), mesh.CoordinateSystem.CYLINDRICAL
            ),
            4,
        ),
    ],
)
def test_slice_coords_round(
    coord1: mesh.SliceCoords, coord2: mesh.SliceCoords, places: int
) -> None:
    def coords_equal(lhs: mesh.SliceCoords, rhs: mesh.SliceCoords) -> bool:
        return (
            bool(np.all(lhs.x1 == rhs.x1) and np.all(lhs.x2 == rhs.x2))
            and lhs.system == rhs.system
        )

    assert not coords_equal(coord1, coord2)
    assert coords_equal(coord1.round_to(places), coord2.round_to(places))
    assert not coords_equal(coord1.round_to(places + 1), coord2.round_to(places + 1))


@given(non_nans(), non_nans(), non_nans(), coordinate_systems)
def test_coord(x1: float, x2: float, x3: float, c: mesh.CoordinateSystem) -> None:
    coord = mesh.Coord(x1, x2, x3, c)
    coord_iter = iter(coord)
    assert next(coord_iter) == x1
    assert next(coord_iter) == x2
    assert next(coord_iter) == x3
    with pytest.raises(StopIteration):
        next(coord_iter)


@pytest.mark.parametrize(
    "coord1,coord2,places",
    [
        (
            mesh.Coord(1.0, 1.0, -0.5, mesh.CoordinateSystem.CARTESIAN),
            mesh.Coord(1.0001, 0.99999, -0.500005, mesh.CoordinateSystem.CARTESIAN),
            4,
        ),
        (
            mesh.Coord(3315.05, 0.0012121, 1e-12, mesh.CoordinateSystem.CARTESIAN),
            mesh.Coord(3315.05, 0.0012121441, 2e-12, mesh.CoordinateSystem.CARTESIAN),
            5,
        ),
        (
            mesh.Coord(1.0, 1, -10.0, mesh.CoordinateSystem.CYLINDRICAL),
            mesh.Coord(1.0001, 0.99999, -9.9999, mesh.CoordinateSystem.CYLINDRICAL),
            4,
        ),
    ],
)
def test_coord_round(coord1: mesh.Coord, coord2: mesh.Coord, places: int) -> None:
    assert coord1 != coord2
    assert coord1.round_to(places) == coord2.round_to(places)
    assert coord1.round_to(places + 1) != coord2.round_to(places + 1)


@given(builds(mesh.Coord), builds(mesh.Coord))
def test_coord_hash(coord1: mesh.Coord, coord2: mesh.Coord) -> None:
    if hash(coord1) != hash(coord2):
        assert coord1 != coord2


@pytest.mark.parametrize(
    "x1,x2,x3,expected",
    [
        (
            np.array([0.0, 1.0, 2.0]),
            np.array([1.0, 0.5, 0.0]),
            np.array([10.0, -5.0, 1.0]),
            [
                mesh.Coord(0.0, 1.0, 10.0, mesh.CoordinateSystem.CARTESIAN),
                mesh.Coord(1.0, 0.5, -5.0, mesh.CoordinateSystem.CARTESIAN),
                mesh.Coord(2.0, 0.0, 1.0, mesh.CoordinateSystem.CARTESIAN),
            ],
        ),
        (
            np.array([5.0, 0.0]),
            np.array([[1.0, 1.5], [-10.0, -11.0]]),
            np.array([[2.0], [-2.0]]),
            [
                mesh.Coord(5.0, 1.0, 2.0, mesh.CoordinateSystem.CARTESIAN),
                mesh.Coord(0.0, 1.5, 2.0, mesh.CoordinateSystem.CARTESIAN),
                mesh.Coord(5.0, -10.0, -2.0, mesh.CoordinateSystem.CARTESIAN),
                mesh.Coord(0.0, -11.0, -2.0, mesh.CoordinateSystem.CARTESIAN),
            ],
        ),
    ],
)
def test_coords_iter_points(
    x1: npt.NDArray, x2: npt.NDArray, x3: npt.NDArray, expected: list[mesh.Coord]
) -> None:
    for c1, c2 in zip(
        mesh.Coords(x1, x2, x3, mesh.CoordinateSystem.CARTESIAN).iter_points(),
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
            mesh.Coord(1.0, 0.5, -5.0, mesh.CoordinateSystem.CARTESIAN),
        ),
        (
            np.array([5.0, 0.0]),
            np.array([[1.0, 1.5], [-10.0, -11.0]]),
            np.array([[2.0], [-2.0]]),
            (1, 0),
            mesh.Coord(5.0, -10.0, -2.0, mesh.CoordinateSystem.CARTESIAN),
        ),
    ],
)
def test_coords_getitem(
    x1: npt.NDArray,
    x2: npt.NDArray,
    x3: npt.NDArray,
    index: tuple[int, ...] | int,
    expected: mesh.Coord,
) -> None:
    coords = mesh.Coords(x1, x2, x3, mesh.CoordinateSystem.CARTESIAN)
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
    coords = mesh.Coords(x1, x2, x3, mesh.CoordinateSystem.CARTESIAN)
    with pytest.raises(expected):
        _ = coords[index]  # type: ignore


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
    coords = mesh.Coords(x1, x2, x3, mesh.CoordinateSystem.CARTESIAN)
    new_coords = coords.offset(offset)
    np.testing.assert_allclose(new_coords.x3, expected)


@pytest.mark.filterwarnings("ignore:invalid value:RuntimeWarning")
@given(mutually_broadcastable_arrays(3))
def test_coords_cartesian_to_cartesian(
    xs: tuple[npt.NDArray, npt.NDArray, npt.NDArray],
) -> None:
    coords = mesh.Coords(*xs, mesh.CoordinateSystem.CARTESIAN)
    new_coords = coords.to_cartesian()
    assert coords.x1 is new_coords.x1
    assert coords.x2 is new_coords.x2
    assert coords.x3 is new_coords.x3


@pytest.mark.filterwarnings("ignore:invalid value:RuntimeWarning")
@given(from_type(mesh.Coords))
def test_coords_cartesian_correct_system(coords: mesh.Coords) -> None:
    assert coords.to_cartesian().system is mesh.CoordinateSystem.CARTESIAN


@pytest.mark.filterwarnings("ignore:invalid value:RuntimeWarning")
@given(mutually_broadcastable_arrays(3))
def test_coords_cylindrical_to_cartesian_z_unchanged(
    xs: tuple[npt.NDArray, npt.NDArray, npt.NDArray],
) -> None:
    coords = mesh.Coords(*xs, mesh.CoordinateSystem.CYLINDRICAL)
    assert np.all(coords.to_cartesian().x3 == coords.x2)


def test_coords_to_cartesian() -> None:
    coords = mesh.Coords(
        np.array([1.0, 1.5, 2.0]),
        np.array([1.0, 0.5, 0.0]),
        np.array([0.0, np.pi / 2.0, np.pi]),
        mesh.CoordinateSystem.CYLINDRICAL,
    ).to_cartesian()
    np.testing.assert_allclose(coords.x1, [1.0, 0.0, -2.0], atol=1e-12)
    np.testing.assert_allclose(coords.x2, [0.0, 1.5, 0.0], atol=1e-12)
    np.testing.assert_allclose(coords.x3, [1.0, 0.5, 0.0], atol=1e-12)


@pytest.mark.parametrize(
    "coord1,coord2,places",
    [
        (
            mesh.Coords(
                np.array([1.0, 5.0]),
                np.array([1.0, 1.0]),
                np.array([-1e-3, -2e-5]),
                mesh.CoordinateSystem.CARTESIAN,
            ),
            mesh.Coords(
                np.array([1.0001, 5.0]),
                np.array([0.99999, 1.0]),
                np.array([-1.0001e-3, -1.99999e-5]),
                mesh.CoordinateSystem.CARTESIAN,
            ),
            4,
        ),
        (
            mesh.Coords(
                np.array([3315.05, 1e-12]),
                np.array([0.0012121, 1e100]),
                np.array([0.0, 22.0]),
                mesh.CoordinateSystem.CARTESIAN,
            ),
            mesh.Coords(
                np.array([3315.05, 0.0]),
                np.array([0.0012121441, 1.0000055e100]),
                np.array([-0.000000009, 22.0]),
                mesh.CoordinateSystem.CARTESIAN,
            ),
            5,
        ),
        (
            mesh.Coords(
                np.array(1.0),
                np.array(1.0),
                np.array(1.0),
                mesh.CoordinateSystem.CYLINDRICAL,
            ),
            mesh.Coords(
                np.array(1.0001),
                np.array(0.99999),
                np.array(1.0),
                mesh.CoordinateSystem.CYLINDRICAL,
            ),
            4,
        ),
    ],
)
def test_coords_round(coord1: mesh.Coords, coord2: mesh.Coords, places: int) -> None:
    def coords_equal(lhs: mesh.Coords, rhs: mesh.Coords) -> bool:
        return (
            bool(
                np.all(lhs.x1 == rhs.x1)
                and np.all(lhs.x2 == rhs.x2)
                and np.all(lhs.x3 == rhs.x3)
            )
            and lhs.system == rhs.system
        )

    assert not coords_equal(coord1, coord2)
    assert coords_equal(coord1.round_to(places), coord2.round_to(places))
    assert not coords_equal(coord1.round_to(places + 1), coord2.round_to(places + 1))


@given(from_type(mesh.FieldAlignedCurve), floats(0.0, 1.0))
def test_curve_start_weight_1(sample_curve: mesh.FieldAlignedCurve, arg: float) -> None:
    # Create a new curve with the start_weight set to 1
    line = mesh.FieldAlignedCurve(
        sample_curve.field,
        sample_curve.start,
        sample_curve.dx3,
        sample_curve.subdivision,
        sample_curve.num_divisions,
        1.0,
    )
    p = line(arg)
    # Use low-ish accuracy to reflect the fact that there is interpolation happening
    np.testing.assert_allclose(p.x1, line.start.x1, 1e-8, 1e-8)
    np.testing.assert_allclose(p.x2, line.start.x2, 1e-8, 1e-8)


@given(from_type(mesh.FieldAlignedCurve), floats(0.0, 1.0))
def test_curve_against_trace(curve: mesh.FieldAlignedCurve, arg: float) -> None:
    p1 = curve(arg)
    p2, _ = curve.field.trace(curve.start, p1.x3, curve.start_weight)
    np.testing.assert_allclose(p1.x1, p2.x1, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(p1.x2, p2.x2, atol=1e-6, rtol=1e-6)


@given(from_type(mesh.FieldAlignedCurve), whole_numbers, floats(0.0, 1.0))
def test_curve_offset(curve: mesh.FieldAlignedCurve, offset: float, arg: float) -> None:
    curve2 = Offset(curve, offset)
    p1 = curve(arg)
    p2 = curve2(arg)
    np.testing.assert_allclose(p1.x1, p2.x1, atol=1e-12)
    np.testing.assert_allclose(p1.x2, p2.x2, atol=1e-12)
    np.testing.assert_allclose(p1.x3, p2.x3 - offset, atol=1e-12)


@given(from_type(mesh.FieldAlignedCurve), integers(-50, 100))
def test_curve_subdivision_len(curve: mesh.FieldAlignedCurve, divisions: int) -> None:
    expected = max(1, divisions)
    divisions_iter = curve.subdivide(divisions)
    for _ in range(expected):
        _ = next(divisions_iter)
    with pytest.raises(StopIteration):
        next(divisions_iter)


@given(from_type(mesh.FieldAlignedCurve), integers(-5, 100))
def test_curve_subdivision(curve: mesh.FieldAlignedCurve, divisions: int) -> None:
    divisions_iter = curve.subdivide(divisions)
    first = next(divisions_iter)
    coord = first(0.0)
    for component, expected in zip(coord, curve(0.0)):
        np.testing.assert_allclose(component, expected, rtol=1e-7, atol=1e-10)
    prev = first(1.0)
    for curve in divisions_iter:
        for c, p in zip(curve(0.0), prev):
            np.testing.assert_allclose(c, p, rtol=1e-7, atol=1e-10)
        prev = curve(1.0)
    for component, expected in zip(prev, curve(1.0)):
        np.testing.assert_allclose(component, expected, rtol=1e-7, atol=1e-10)


@given(from_type(mesh.FieldAlignedCurve), integers(1, 10))
def test_curve_control_points_cached(curve: mesh.FieldAlignedCurve, order: int) -> None:
    p1 = mesh.control_points(curve, order)
    p2 = mesh.control_points(curve, order)
    assert p1 is p2


@given(from_type(mesh.FieldAlignedCurve), integers(1, 10))
def test_curve_control_points_size(curve: mesh.FieldAlignedCurve, n: int) -> None:
    assert len(mesh.control_points(curve, n)) == n + 1


def example_trace(
    start: mesh.SliceCoord, x3: npt.ArrayLike, start_weight: float = 0.0
) -> tuple[mesh.SliceCoords, npt.NDArray]:
    a = -2.0
    b = -0.5
    c = 1 - start_weight
    return (
        mesh.SliceCoords(
            c * np.asarray(x3) * a + start_weight * start.x1,
            c * np.asarray(x3) * b + start_weight * start.x2,
            mesh.CoordinateSystem.CARTESIAN,
        ),
        np.sqrt(1.0 + c * c * (a * a + b * b)) * np.asarray(x3),
    )


def test_curve_control_points_values() -> None:
    curve = Offset(
        mesh.FieldAlignedCurve(
            mesh.FieldTracer(
                example_trace,
                5,
            ),
            mesh.SliceCoord(0.0, 0.0, mesh.CoordinateSystem.CARTESIAN),
            -1.0,
        ),
        x3_offset=-0.5,
    )

    x1, x2, x3 = mesh.control_points(curve, 2)
    np.testing.assert_allclose(x1, [-1.0, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(x2, [-0.25, 0.0, 0.25], atol=1e-12)
    np.testing.assert_allclose(x3, [0.0, -0.5, -1.0], atol=1e-12)


@given(from_type(mesh.StraightLine), floats(0.0, 1.0))
def test_straight_line_between_termini(line: mesh.StraightLine, s: float) -> None:
    position = line(s).to_coord()
    for p, n, s in zip(position, line.north, line.south):
        xs = sorted([p, n, s])
        assert p == xs[1]


@given(from_type(mesh.StraightLine), integers(-50, 100))
def test_straight_line_subdivision_len(
    curve: mesh.StraightLine, divisions: int
) -> None:
    expected = max(1, divisions)
    divisions_iter = curve.subdivide(divisions)
    for _ in range(expected):
        _ = next(divisions_iter)
    with pytest.raises(StopIteration):
        next(divisions_iter)


@given(from_type(mesh.StraightLine), integers(-5, 100))
def test_striaght_line_subdivision(curve: mesh.StraightLine, divisions: int) -> None:
    divisions_iter = curve.subdivide(divisions)
    first = next(divisions_iter)
    coord = first(0.0)
    for component, expected in zip(coord, curve(0.0)):
        np.testing.assert_allclose(component, expected, rtol=1e-7, atol=1e-10)
    prev = first(1.0)
    for curve in divisions_iter:
        for c, p in zip(curve(0.0), prev):
            np.testing.assert_allclose(c, p, rtol=1e-7, atol=1e-10)
        prev = curve(1.0)
    for component, expected in zip(prev, curve(1.0)):
        np.testing.assert_allclose(component, expected, rtol=1e-7, atol=1e-10)


@given(from_type(mesh.Quad), floats(0.0, 1.0))
def test_quad_north(q: mesh.Quad, s: float) -> None:
    actual = q.north(s)
    if q.aligned_edges in {mesh.QuadAlignment.ALIGNED, mesh.QuadAlignment.NORTH}:
        x3_offset = q.x3_offset
        x1, x2 = q.field.trace(q.shape(0.0).to_coord(), actual.x3 - x3_offset)[0]
        np.testing.assert_allclose(actual.x1, x1, rtol=2e-4, atol=1e-5)
        np.testing.assert_allclose(actual.x2, x2, rtol=2e-4, atol=1e-5)
    else:
        x1, x2 = q.shape(0.0)
        np.testing.assert_allclose(actual.x1, x1, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(actual.x2, x2, rtol=1e-8, atol=1e-8)


@given(from_type(mesh.Quad), floats(0.0, 1.0))
def test_quad_south(q: mesh.Quad, s: float) -> None:
    actual = q.south(s)
    if q.aligned_edges in {mesh.QuadAlignment.ALIGNED, mesh.QuadAlignment.SOUTH}:
        x3_offset = q.x3_offset
        x1, x2 = q.field.trace(q.shape(1.0).to_coord(), actual.x3 - x3_offset)[0]
        np.testing.assert_allclose(actual.x1, x1, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(actual.x2, x2, rtol=1e-4, atol=1e-5)
    else:
        x1, x2 = q.shape(1.0)
        np.testing.assert_allclose(actual.x1, x1, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(actual.x2, x2, rtol=1e-8, atol=1e-8)


@given(from_type(mesh.Quad), floats(0.0, 1.0), floats(0.0, 1.0))
def test_quad_get_field_line(q: mesh.Quad, s: float, t: float) -> None:
    line = q.get_field_line(s)
    start = q.shape(s).to_coord()
    # Check the line passes through the correct location on `q.shape`
    if q.num_divisions == 1:
        expected_start_x1, expected_start_x2 = start

        def distance_from_start_point(t: float) -> npt.NDArray:
            coord = line(t)
            return np.sqrt(
                (coord.x1 - expected_start_x1) ** 2
                + (coord.x2 - expected_start_x2) ** 2
            )

        if np.isclose(distance_from_start_point(0.5), 0.0, 1e-8, 1e-8):
            # Handle case where line is constant (the minimiser doesn't like it)
            actual_start_x1, actual_start_x2, _ = line(0.5)
        else:
            res = minimize_scalar(
                distance_from_start_point, (0.0, 0.5, 1.0), (0.0, 1.0), tol=1e-12
            )
            assert res.success
            actual_start_x1, actual_start_x2, _ = line(res.x)
        np.testing.assert_allclose(
            actual_start_x1, expected_start_x1, rtol=1e-8, atol=1e-8
        )
        np.testing.assert_allclose(
            actual_start_x2, expected_start_x2, rtol=1e-8, atol=1e-8
        )
    # Check location along the line
    actual_x1, actual_x2, actual_x3 = line(t).to_coord()
    field_coord = q.field.trace(start, actual_x3 - q.x3_offset)[0].to_coord()
    if q.aligned_edges == mesh.QuadAlignment.ALIGNED:
        expected_x1, expected_x2 = field_coord
    elif q.aligned_edges == mesh.QuadAlignment.NONALIGNED:
        expected_x1, expected_x2 = start
    elif q.aligned_edges == mesh.QuadAlignment.NORTH:
        expected_x1 = s * start.x1 + (1 - s) * field_coord.x1
        expected_x2 = s * start.x2 + (1 - s) * field_coord.x2
    else:
        assert q.aligned_edges == mesh.QuadAlignment.SOUTH
        expected_x1 = (1 - s) * start.x1 + s * field_coord.x1
        expected_x2 = (1 - s) * start.x2 + s * field_coord.x2
    np.testing.assert_allclose(actual_x1, expected_x1, rtol=2e-4, atol=1e-5)
    np.testing.assert_allclose(actual_x2, expected_x2, rtol=2e-4, atol=1e-5)
    assert actual_x3 <= q.x3_offset + 0.5 * np.abs(q.dx3) + 1e12
    assert actual_x3 >= q.x3_offset - 0.5 * np.abs(q.dx3) - 1e-12


@settings(deadline=None)
@given(from_type(mesh.Quad), integers(2, 10))
def test_quad_near_edge(q: mesh.Quad, n: int) -> None:
    actual = frozenset(q.near(np.array([0.0, 1.0])).iter_points())
    expected = frozenset({q.north(0.0).to_coord(), q.south(0.0).to_coord()})
    assert expected == actual
    s = np.linspace(0.0, 1.0, n)
    cp = q.near(s)
    x1, x2, x3 = np.vectorize(lambda t: tuple(q.get_field_line(t)(0.0)))(
        s,
    )
    np.testing.assert_allclose(cp.x1, x1, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(cp.x2, x2, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(cp.x3, x3, rtol=1e-10, atol=1e-12)


@settings(deadline=None)
@given(from_type(mesh.Quad), integers(2, 10))
def test_quad_far_edge(q: mesh.Quad, n: int) -> None:
    actual = frozenset(q.far(np.array([0.0, 1.0])).iter_points())
    expected = frozenset({q.north(1.0).to_coord(), q.south(1.0).to_coord()})
    assert expected == actual
    s = np.linspace(0.0, 1.0, n)
    cp = q.far(s)
    x1, x2, x3 = np.vectorize(lambda t: tuple(q.get_field_line(t)(1.0)))(
        s,
    )
    np.testing.assert_allclose(cp.x1, x1, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(cp.x2, x2, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(cp.x3, x3, rtol=1e-10, atol=1e-12)


@given(from_type(mesh.Quad))
def test_quad_near_far_corners(q: mesh.Quad) -> None:
    expected = frozenset(q.corners().iter_points())
    actual = frozenset(q.far([0.0, 1.0]).iter_points()) | frozenset(
        q.near([0.0, 1.0]).iter_points()
    )
    assert expected == actual


@given(from_type(mesh.Quad))
def test_quad_corners(q: mesh.Quad) -> None:
    corners = q.corners()
    assert corners[0] == q.north(0.0).to_coord()
    assert corners[1] == q.north(1.0).to_coord()
    assert corners[2] == q.south(0.0).to_coord()
    assert corners[3] == q.south(1.0).to_coord()


@given(from_type(mesh.Quad), sampled_from(list(range(2, 10, 2))))
def test_quad_control_points_spacing(q: mesh.Quad, n: int) -> None:
    cp = mesh.control_points(q, n)
    samples = np.linspace(0.0, 1.0, n + 1)
    # Check spacing in the direction along the field lines (holds true
    # even for quads that aren't field-aligned)
    start_points = q.shape(samples)
    x1_starts = np.expand_dims(start_points.x1, 1)
    x2_starts = np.expand_dims(start_points.x2, 1)
    x3_offset = q.x3_offset
    if q.aligned_edges == mesh.QuadAlignment.ALIGNED:
        weights = np.zeros_like(x1_starts)
    elif q.aligned_edges == mesh.QuadAlignment.NONALIGNED:
        weights = np.ones_like(x1_starts)
    elif q.aligned_edges == mesh.QuadAlignment.NORTH:
        weights = np.expand_dims(samples, 1)
    else:
        assert q.aligned_edges == mesh.QuadAlignment.SOUTH
        weights = np.expand_dims(1 - samples, 1)
    distances = np.vectorize(
        lambda x1, x2, x3, start_weight: q.field.trace(
            mesh.SliceCoord(x1, x2, start_points.system), x3, start_weight
        )[1]
    )(
        x1_starts,
        x2_starts,
        cp.x3 - x3_offset,
        weights,
    )
    d_diff = distances[:, 1:] - distances[:, :-1]
    for i in range(n + 1):
        np.testing.assert_allclose(d_diff[i, 0], d_diff[i, :], rtol=1e-6, atol=1e-7)
    # Check points fall along field lines that are equally spaced at the start position
    x1, x2, x3 = np.vectorize(lambda s, t: tuple(q.get_field_line(s)(t)))(
        samples.reshape((-1, 1)),
        samples,
    )
    np.testing.assert_allclose(cp.x1, x1, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(cp.x2, x2, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(cp.x3, x3, rtol=1e-6, atol=1e-6)


@given(from_type(mesh.Quad), whole_numbers, integers(1, 5))
def test_quad_offset(q: mesh.Quad, x: float, n: int) -> None:
    actual = mesh.control_points(Offset(q, x), n)
    expected = mesh.control_points(q, n).offset(x)
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


@settings(deadline=None)
@given(from_type(mesh.Quad), integers(-5, 100))
def test_quad_subdivision(quad: mesh.Quad, divisions: int) -> None:
    divisions_iter = quad.subdivide(divisions)
    quad_corners = quad.corners()
    first = next(divisions_iter)
    corners = first.corners()
    for c, q in zip(corners, quad_corners):
        np.testing.assert_allclose(c[[0, 2]], q[[0, 2]], rtol=1e-8, atol=1e-10)
    prev = corners
    for quad in divisions_iter:
        corners = quad.corners()
        for c, p in zip(corners, prev):
            np.testing.assert_allclose(c[[0, 2]], p[[1, 3]], rtol=1e-8, atol=1e-10)
        prev = corners
    for p, q in zip(prev, quad_corners):
        np.testing.assert_allclose(p[[1, 3]], q[[1, 3]], rtol=1e-8, atol=1e-10)


@given(from_type(mesh.Prism))
def test_prism_near_edge(h: mesh.Prism) -> None:
    expected = frozenset(
        itertools.chain.from_iterable(
            (s.north(0.0).to_coord(), s.south(0.0).to_coord()) for s in h.sides
        )
    )
    actual = frozenset(h.near.corners().iter_points())
    assert expected == actual


@given(flat_sided_hex)
def test_prism_far_edge(h: mesh.Prism) -> None:
    expected = frozenset(
        itertools.chain.from_iterable(
            (
                s.north(1.0).to_coord(),
                s.south(1.0).to_coord(),
            )
            for s in h.sides
        )
    )
    actual = frozenset(h.far.corners().iter_points())
    assert expected == actual


@given(from_type(mesh.Prism))
def test_prism_near_far_corners(h: mesh.Prism) -> None:
    expected = frozenset(h.corners().iter_points())
    actual = frozenset(h.near.corners().iter_points()) | frozenset(
        h.far.corners().iter_points()
    )
    assert expected == actual


@given(flat_sided_hex)
def test_hex_corners(h: mesh.Prism) -> None:
    corners = h.corners()
    assert corners[0] == h.sides[0].north(0.0).to_coord()
    assert corners[1] == h.sides[0].south(0.0).to_coord()
    assert corners[2] == h.sides[1].north(0.0).to_coord()
    assert corners[3] == h.sides[1].south(0.0).to_coord()
    assert corners[4] == h.sides[0].north(1.0).to_coord()
    assert corners[5] == h.sides[0].south(1.0).to_coord()
    assert corners[6] == h.sides[1].north(1.0).to_coord()
    assert corners[7] == h.sides[1].south(1.0).to_coord()


@given(from_type(mesh.Prism))
def test_prism_get_quads(h: mesh.Prism) -> None:
    actual = frozenset(h)
    expected = frozenset(h.sides)
    assert actual == expected


@given(from_type(mesh.Prism), whole_numbers)
def test_prism_offset(h: mesh.Prism, x: float) -> None:
    actual = Offset(h, x).corners()
    expected = h.corners().offset(x)
    np.testing.assert_allclose(actual.x1, expected.x1, atol=1e-12)
    np.testing.assert_allclose(actual.x2, expected.x2, atol=1e-12)
    np.testing.assert_allclose(actual.x3, expected.x3, atol=1e-12)
    assert actual.system == expected.system


@given(from_type(mesh.Prism), integers(-15, 30))
def test_prism_subdivision_len(h: mesh.Prism, divisions: int) -> None:
    expected = max(1, divisions)
    divisions_iter = h.subdivide(divisions)
    for _ in range(expected):
        _ = next(divisions_iter)
    with pytest.raises(StopIteration):
        next(divisions_iter)


@given(from_type(mesh.Prism), integers(-5, 10))
def test_prism_subdivision(h: mesh.Prism, divisions: int) -> None:
    divisions_iter = h.subdivide(divisions)
    prism_corners = h.corners()
    first = next(divisions_iter)
    corners = first.corners()
    n = len(h.sides)
    for c, t in zip(corners, prism_corners):
        np.testing.assert_allclose(c[:n], t[:n], rtol=1e-8, atol=1e-8)
    prev = corners
    for h in divisions_iter:
        corners = h.corners()
        for c, p in zip(corners, prev):
            np.testing.assert_allclose(c[:n], p[n:], rtol=1e-8, atol=1e-8)
        prev = corners
    for p, t in zip(prev, prism_corners):
        np.testing.assert_allclose(p[n:], t[n:], rtol=1e-8, atol=1e-8)


@given(mesh_arguments)
def test_mesh_layer_elements_no_offset(
    args: tuple[list[mesh.E], list[frozenset[mesh.B]]],
) -> None:
    layer = mesh.MeshLayer(*args)
    for actual, expected in zip(layer, args[0]):
        assert actual is expected
    for actual_bound, expected_bound in zip(layer.boundaries(), args[1]):
        assert actual_bound == expected_bound


def get_corners(
    shape: mesh.Prism | mesh.EndShape | mesh.Quad | mesh.NormalisedCurve,
) -> mesh.Coords:
    if isinstance(shape, (mesh.Prism, mesh.Quad, mesh.EndShape)):
        return shape.corners()
    return shape([0.0, 1.0])


@settings(deadline=None)
@given(mesh_arguments, non_nans())
def test_mesh_layer_elements_with_offset(
    args: tuple[list[mesh.E], list[frozenset[mesh.B]]], offset: float
) -> None:
    layer = Offset(mesh.MeshLayer(*args), offset)
    for actual, expected in zip(layer, args[0]):
        actual_corners = actual.corners()
        expected_corners = Offset(expected, offset).corners()
        np.testing.assert_allclose(actual_corners.x1, expected_corners.x1, atol=1e-12)
        np.testing.assert_allclose(actual_corners.x2, expected_corners.x2, atol=1e-12)
        np.testing.assert_allclose(actual_corners.x3, expected_corners.x3, atol=1e-12)
    for actual_bound, expected_bound in zip(layer.boundaries(), args[1]):
        actual_elems = frozenset(
            frozenset(get_corners(elem).iter_points()) for elem in actual_bound
        )
        expected_elems = frozenset(
            frozenset(get_corners(elem).offset(offset).iter_points())
            for elem in expected_bound
        )
        assert actual_elems == expected_elems


@settings(deadline=None)
@given(mesh_arguments, integers(1, 10))
def test_mesh_layer_elements_with_subdivisions(
    args: tuple[list[mesh.E], list[frozenset[mesh.B]]], subdivisions: int
) -> None:
    layer = mesh.MeshLayer(*args, subdivisions=subdivisions)
    expected = frozenset(
        itertools.chain.from_iterable(
            (
                x.corners().iter_points()
                for x in itertools.chain.from_iterable(
                    (x.subdivide(subdivisions) for x in args[0])
                )
            )
        )
    )
    actual = frozenset(
        itertools.chain.from_iterable((x.corners().iter_points() for x in layer))
    )
    assert expected == actual
    for actual_bound, expected_bound in zip(layer.boundaries(), args[1]):
        expected_corners = frozenset(
            itertools.chain.from_iterable(
                (
                    get_corners(x).iter_points()
                    for x in itertools.chain.from_iterable(
                        (
                            cast(Iterable[mesh.B], x.subdivide(subdivisions))
                            for x in expected_bound
                        )
                    )
                )
            )
        )
        actual_corners = frozenset(
            itertools.chain.from_iterable(
                (get_corners(x).iter_points() for x in actual_bound)
            )
        )
        assert expected_corners == actual_corners


def evaluate_element(element: mesh.Quad | mesh.Prism, s: float) -> list[mesh.Coord]:
    if isinstance(element, mesh.Quad):
        return [element.north(s).to_coord(), element.south(s).to_coord()]
    # This works for hexahedrons and triangular prisms
    return evaluate_element(element.sides[0], s) + evaluate_element(element.sides[1], s)


@given(mesh_arguments, whole_numbers, integers(1, 5))
def test_mesh_layer_near_faces(
    args: tuple[list[mesh.E], list[frozenset[mesh.B]]],
    offset: float,
    subdivisions: int,
) -> None:
    layer = Offset(mesh.MeshLayer(*args, subdivisions), offset)
    expected = frozenset(
        itertools.chain.from_iterable(
            (evaluate_element(Offset(x, offset), 0.0) for x in args[0])
        ),
    )
    actual = frozenset(
        itertools.chain.from_iterable(
            (get_corners(x).iter_points() for x in layer.near_faces())
        ),
    )
    assert expected == actual


@given(mesh_arguments, whole_numbers, integers(1, 5))
def test_mesh_layer_far_faces(
    args: tuple[list[mesh.E], list[frozenset[mesh.B]]],
    offset: float,
    subdivisions: int,
) -> None:
    layer = Offset(mesh.MeshLayer(*args, subdivisions), offset)
    expected = frozenset(
        itertools.chain.from_iterable(
            (evaluate_element(Offset(x, offset), 1.0) for x in args[0])
        ),
    )
    actual = frozenset(
        itertools.chain.from_iterable(
            (get_corners(x).iter_points() for x in layer.far_faces())
        ),
    )
    assert expected == actual


@given(from_type(mesh.MeshLayer))
def test_mesh_layer_faces_in_elements(layer: mesh.MeshLayer) -> None:
    element_corners = frozenset(
        itertools.chain.from_iterable((get_corners(x).iter_points() for x in layer)),
    )
    near_face_corners = frozenset(
        itertools.chain.from_iterable(
            (get_corners(x).iter_points() for x in layer.near_faces())
        ),
    )
    far_face_corners = frozenset(
        itertools.chain.from_iterable(
            (get_corners(x).iter_points() for x in layer.far_faces())
        ),
    )
    assert near_face_corners < element_corners
    assert far_face_corners < element_corners


@given(from_type(mesh.MeshLayer))
def test_mesh_layer_len(layer: mesh.MeshLayer) -> None:
    layer_iter = iter(layer)
    for _ in range(len(layer)):
        next(layer_iter)
    with pytest.raises(StopIteration):
        next(layer_iter)


quad_mesh_elements = _quad_mesh_elements(
    1.0,
    1.0,
    10.0,
    ((0.0, 0.0), (1.0, 0.0)),
    4,
    mesh.CoordinateSystem.CARTESIAN,
    10,
    False,
    False,
)
hex_mesh_elements = cast(
    tuple[list[mesh.Prism], list[frozenset[mesh.Quad]]],
    _hex_mesh_arguments(
        1.0,
        1.0,
        10.0,
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        3,
        3,
        mesh.CoordinateSystem.CARTESIAN,
        10,
        False,
    ),
)[0]


@pytest.mark.parametrize(
    "elements",
    [(quad_mesh_elements,), (hex_mesh_elements,)],
)
def test_mesh_layer_element_type(elements: list[mesh.Quad] | list[mesh.Prism]) -> None:
    layer = mesh.MeshLayer(elements, [])  # type: ignore
    element = next(iter(elements))
    assert layer.element_type is type(element)


@given(quad_mesh_layer_no_divisions)
def test_mesh_layer_quads_for_quads(
    layer: mesh.MeshLayer[mesh.Quad, mesh.Segment, mesh.NormalisedCurve],
) -> None:
    assert all(q1 is q2 for q1, q2 in zip(layer, layer.quads()))


def test_mesh_layer_quads_for_prisms() -> None:
    # FIXME: Need to implement this
    pass


@given(from_type(mesh.GenericMesh))
def test_mesh_iter_layers(m: mesh.GenericMesh) -> None:
    for layer, offset in zip(m.layers(), m.offsets):
        assert (
            layer.get_underlying_object().reference_elements
            is m.reference_layer.reference_elements
        )
        assert layer.x3_offset == offset
        assert layer.subdivisions == m.reference_layer.subdivisions


@given(from_type(mesh.GenericMesh))
def test_mesh_len(m: mesh.GenericMesh) -> None:
    mesh_iter = itertools.chain.from_iterable(layer for layer in m.layers())
    for _ in range(len(m)):
        next(mesh_iter)
    with pytest.raises(StopIteration):
        next(mesh_iter)


shared_coords = shared(coordinate_systems, key=0)


@given(
    builds(
        linear_field_trace,
        whole_numbers,
        whole_numbers,
        non_zero,
        shared_coords,
        floats(-0.1, 0.1),
        tuples(whole_numbers, whole_numbers),
    ),
    builds(mesh.SliceCoord, non_zero, whole_numbers, shared_coords),
    whole_numbers,
    non_zero,
    integers(2, 8),
    integers(3, 12),
)
def test_normalise_straight_field_line(
    trace: mesh.FieldTrace,
    start: mesh.SliceCoord,
    x3_start: float,
    dx3: float,
    resolution: int,
    n: int,
) -> None:
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


centre = shared(whole_numbers, key=100)
rad = shared(non_zero, key=101)
x3_limit = shared(rad.flatmap(lambda r: floats(*sorted([0.01 * r, 0.99 * r]))), key=102)
x3_start = x3_limit.map(lambda x: -x)
slope = shared(whole_numbers, key=103)
x1_start = shared(tuples(centre, rad).map(lambda x: x[0] + x[1]), key=104)
x2_centre = shared(whole_numbers, key=105)
cartesian_coords = shared(sampled_from(list(CARTESIAN_SYSTEMS)), key=106)


@given(
    builds(cylindrical_field_trace, centre, slope),
    builds(
        cylindrical_field_line,
        centre,
        x1_start,
        x2_centre,
        slope,
        tuples(x3_start, x3_limit),
        cartesian_coords,
    ),
    builds(mesh.SliceCoord, x1_start, x2_centre, cartesian_coords),
    x3_start,
    x3_limit.map(lambda x: 2 * x),
    integers(100, 200),
    integers(3, 6),
)
def test_normalise_curved_field_line(
    trace: mesh.FieldTrace,
    line: mesh.NormalisedCurve,
    start: mesh.SliceCoord,
    x3_start: float,
    dx3: float,
    resolution: int,
    n: int,
) -> None:
    # Note: part of the motivation of this test is to ensure my
    # functions to generate curved fields are self-consistent.
    normalised = mesh.normalise_field_line(
        trace, start, x3_start, x3_start + dx3, resolution
    )
    checkpoints = np.linspace(0.0, 1.0, n)
    coords_normed = normalised(checkpoints)
    expected_coords = line(checkpoints)
    np.testing.assert_allclose(
        coords_normed.x1, expected_coords.x1, atol=1e-6, rtol=1e-6
    )
    np.testing.assert_allclose(
        coords_normed.x2, expected_coords.x2, atol=1e-6, rtol=1e-6
    )
    np.testing.assert_allclose(
        coords_normed.x3, expected_coords.x3, atol=1e-6, rtol=1e-6
    )
    _, distances = trace(start, expected_coords.x3)
    spacing = distances[1:] - distances[:-1]
    np.testing.assert_allclose(spacing, spacing[0])


# FIXME: Write unit tests for FieldTracer class
