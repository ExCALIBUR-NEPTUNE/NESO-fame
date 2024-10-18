from typing import Type

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given
from hypothesis.strategies import (
    builds,
    from_type,
    integers,
)

from neso_fame import coordinates

from .conftest import (
    coordinate_systems,
    mutually_broadcastable_arrays,
    non_nans,
)


@given(non_nans(), non_nans(), coordinate_systems)
def test_slice_coord(x1: float, x2: float, c: coordinates.CoordinateSystem) -> None:
    coord = coordinates.SliceCoord(x1, x2, c)
    coord_iter = iter(coord)
    assert next(coord_iter) == x1
    assert next(coord_iter) == x2
    with pytest.raises(StopIteration):
        next(coord_iter)


@given(non_nans(), non_nans(), non_nans(), coordinate_systems)
def test_slice_coord_to_3d(
    x1: float, x2: float, x3: float, c: coordinates.CoordinateSystem
) -> None:
    coord = coordinates.SliceCoord(x1, x2, c).to_3d_coord(x3)
    assert coord.x1 == x1
    assert coord.x2 == x2
    assert coord.x3 == x3
    assert coord.system == c


@given(non_nans(), non_nans(), coordinate_systems)
def test_slice_coord_equal_self(
    x1: float, x2: float, c: coordinates.CoordinateSystem
) -> None:
    coord = coordinates.SliceCoord(x1, x2, c)
    assert coord == coord


@given(non_nans(), non_nans(), coordinate_systems)
def test_slice_coord_equal_other_type(
    x1: float, x2: float, c: coordinates.CoordinateSystem
) -> None:
    coord = coordinates.SliceCoord(x1, x2, c)
    assert coord != x1
    assert coord != x2


@pytest.mark.parametrize(
    "coord1,coord2,places",
    [
        (
            coordinates.SliceCoord(1.0, 1.0, coordinates.CoordinateSystem.CARTESIAN),
            coordinates.SliceCoord(
                1.0001, 0.99999, coordinates.CoordinateSystem.CARTESIAN
            ),
            4,
        ),
        (
            coordinates.SliceCoord(
                3314.44949999, 0.0012121, coordinates.CoordinateSystem.CARTESIAN
            ),
            coordinates.SliceCoord(
                3314.44950001, 0.0012121441, coordinates.CoordinateSystem.CARTESIAN
            ),
            5,
        ),
        (
            coordinates.SliceCoord(1.0, 1.0, coordinates.CoordinateSystem.CYLINDRICAL),
            coordinates.SliceCoord(
                1.0001, 0.99999, coordinates.CoordinateSystem.CYLINDRICAL
            ),
            4,
        ),
    ],
)
def test_slice_coord_round(
    coord1: coordinates.SliceCoord, coord2: coordinates.SliceCoord, places: int
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
                coordinates.SliceCoord(
                    0.0, 1.0, coordinates.CoordinateSystem.CARTESIAN
                ),
                coordinates.SliceCoord(
                    1.0, 0.5, coordinates.CoordinateSystem.CARTESIAN
                ),
                coordinates.SliceCoord(
                    2.0, 0.0, coordinates.CoordinateSystem.CARTESIAN
                ),
            ],
        ),
        (
            np.array([5.0, 0.0]),
            np.array([[1.0, 1.5], [-10.0, -11.0]]),
            [
                coordinates.SliceCoord(
                    5.0, 1.0, coordinates.CoordinateSystem.CARTESIAN
                ),
                coordinates.SliceCoord(
                    0.0, 1.5, coordinates.CoordinateSystem.CARTESIAN
                ),
                coordinates.SliceCoord(
                    5.0, -10.0, coordinates.CoordinateSystem.CARTESIAN
                ),
                coordinates.SliceCoord(
                    0.0, -11.0, coordinates.CoordinateSystem.CARTESIAN
                ),
            ],
        ),
    ],
)
def test_slice_coords_iter_points(
    x1: npt.NDArray, x2: npt.NDArray, expected: list[coordinates.SliceCoord]
) -> None:
    for c1, c2 in zip(
        coordinates.SliceCoords(
            x1, x2, coordinates.CoordinateSystem.CARTESIAN
        ).iter_points(),
        expected,
    ):
        assert c1 == c2


@given(mutually_broadcastable_arrays(2), coordinate_systems)
def test_slice_coords_iter(
    x: tuple[npt.NDArray, npt.NDArray], c: coordinates.CoordinateSystem
) -> None:
    coords = coordinates.SliceCoords(*x, c)
    coords_iter = iter(coords)
    assert np.all(next(coords_iter) == x[0])
    assert np.all(next(coords_iter) == x[1])
    with pytest.raises(StopIteration):
        next(coords_iter)


def test_slice_coords_iter_points_empty() -> None:
    coords = coordinates.SliceCoords(
        np.array([]), np.array([]), coordinates.CoordinateSystem.CARTESIAN
    )
    assert len(list(coords.iter_points())) == 0


@given(mutually_broadcastable_arrays(2), coordinate_systems)
def test_slice_coords_len(
    x: tuple[npt.NDArray, npt.NDArray], c: coordinates.CoordinateSystem
) -> None:
    coords = coordinates.SliceCoords(*x, c)
    coords_iter = coords.iter_points()
    for _ in range(len(coords)):
        next(coords_iter)
    with pytest.raises(StopIteration):
        next(coords_iter)


@given(from_type(coordinates.SliceCoords))
def test_slice_coords_shape(x: coordinates.SliceCoords) -> None:
    if len(x.shape) > 0:
        assert np.prod(x.shape) == len(x)
    else:
        assert len(x) == 1


@pytest.mark.parametrize(
    "x1,x2,index,expected",
    [
        (
            np.array([0.0, 1.0, 2.0]),
            np.array([1.0, 0.5, 0.0]),
            1,
            coordinates.SliceCoord(1.0, 0.5, coordinates.CoordinateSystem.CARTESIAN),
        ),
        (
            np.array([5.0, 0.0]),
            np.array([[1.0, 1.5], [-10.0, -11.0]]),
            (1, 0),
            coordinates.SliceCoord(5.0, -10.0, coordinates.CoordinateSystem.CARTESIAN),
        ),
    ],
)
def test_slice_coords_getitem(
    x1: npt.NDArray,
    x2: npt.NDArray,
    index: tuple[int, ...] | int,
    expected: coordinates.SliceCoord,
) -> None:
    coords = coordinates.SliceCoords(x1, x2, coordinates.CoordinateSystem.CARTESIAN)
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
    coords = coordinates.SliceCoords(x1, x2, coordinates.CoordinateSystem.CARTESIAN)
    with pytest.raises(expected):
        _ = coords[index]  # type: ignore


@given(from_type(coordinates.SliceCoords).filter(lambda c: len(c) > 1))
def test_slice_coords_get_set(coords: coordinates.SliceCoords) -> None:
    coordset = coords.get_set(slice(None))
    assert len(coordset) <= len(coords)
    for coord in coords.iter_points():
        assert coord in coordset


def test_slice_coords_get_set_partial() -> None:
    coords = coordinates.SliceCoords(
        np.linspace(0.0, 1.0, 10),
        np.linspace(5.0, 11.0, 10),
        coordinates.CoordinateSystem.CARTESIAN,
    )
    coordset = coords.get_set(slice(0, 5))
    assert len(coordset) == 5
    for i in range(5):
        assert coords[i] in coordset
    for i in range(5, 10):
        assert coords[i] not in coordset


@pytest.mark.parametrize(
    "coord1,coord2,places",
    [
        (
            coordinates.SliceCoords(
                np.array([1.0, 5.0]),
                np.array([1.0, 1.0]),
                coordinates.CoordinateSystem.CARTESIAN,
            ),
            coordinates.SliceCoords(
                np.array([1.0001, 5.0]),
                np.array([0.99999, 1.0]),
                coordinates.CoordinateSystem.CARTESIAN,
            ),
            4,
        ),
        (
            coordinates.SliceCoords(
                np.array([3315.05, 1e-12]),
                np.array([0.0012121, 1e100]),
                coordinates.CoordinateSystem.CARTESIAN,
            ),
            coordinates.SliceCoords(
                np.array([3315.05, 0.0]),
                np.array([0.0012121441, 1.0000055e100]),
                coordinates.CoordinateSystem.CARTESIAN,
            ),
            5,
        ),
        (
            coordinates.SliceCoords(
                np.array(1.0), np.array(1.0), coordinates.CoordinateSystem.CYLINDRICAL
            ),
            coordinates.SliceCoords(
                np.array(1.0001),
                np.array(0.99999),
                coordinates.CoordinateSystem.CYLINDRICAL,
            ),
            4,
        ),
    ],
)
def test_slice_coords_round(
    coord1: coordinates.SliceCoords, coord2: coordinates.SliceCoords, places: int
) -> None:
    def coords_equal(
        lhs: coordinates.SliceCoords, rhs: coordinates.SliceCoords
    ) -> bool:
        return (
            bool(np.all(lhs.x1 == rhs.x1) and np.all(lhs.x2 == rhs.x2))
            and lhs.system == rhs.system
        )

    assert not coords_equal(coord1, coord2)
    assert coords_equal(coord1.round_to(places), coord2.round_to(places))
    assert not coords_equal(coord1.round_to(places + 1), coord2.round_to(places + 1))


@given(mutually_broadcastable_arrays(2), non_nans(), coordinate_systems)
def test_slice_coords_to_3d(
    x: tuple[npt.NDArray, npt.NDArray], x3: float, c: coordinates.CoordinateSystem
) -> None:
    coords = coordinates.SliceCoords(x[0], x[1], c).to_3d_coords(x3)
    assert np.all(coords.x1 == x[0])
    assert np.all(coords.x2 == x[1])
    assert coords.x3 == x3
    assert coords.system == c


@given(non_nans(), non_nans(), non_nans(), coordinate_systems)
def test_coord(
    x1: float, x2: float, x3: float, c: coordinates.CoordinateSystem
) -> None:
    coord = coordinates.Coord(x1, x2, x3, c)
    coord_iter = iter(coord)
    assert next(coord_iter) == x1
    assert next(coord_iter) == x2
    assert next(coord_iter) == x3
    with pytest.raises(StopIteration):
        next(coord_iter)


@given(non_nans(), non_nans(), non_nans(), coordinate_systems)
def test_coord_equal_self(
    x1: float, x2: float, x3: float, c: coordinates.CoordinateSystem
) -> None:
    coord = coordinates.Coord(x1, x2, x3, c)
    assert coord == coord


@given(non_nans(), non_nans(), non_nans(), coordinate_systems)
def test_coord_equal_other_type(
    x1: float, x2: float, x3: float, c: coordinates.CoordinateSystem
) -> None:
    coord = coordinates.Coord(x1, x2, x3, c)
    assert coord != x1
    assert coord != x2
    assert coord != x3


@given(non_nans(), non_nans(), non_nans(), coordinate_systems)
def test_coord_to_slice_coord(
    x1: float, x2: float, x3: float, c: coordinates.CoordinateSystem
) -> None:
    coord = coordinates.Coord(x1, x2, x3, c)
    slice_coord = coord.to_slice_coord()
    assert slice_coord.x1 == coord.x1
    assert slice_coord.x2 == coord.x2
    assert slice_coord.system == coord.system


@pytest.mark.parametrize(
    "coord1,coord2",
    [
        (
            coordinates.Coord(
                1.0, 2.0, np.pi / 2, coordinates.CoordinateSystem.CYLINDRICAL
            ),
            coordinates.Coord(0.0, 1.0, 2.0, coordinates.CoordinateSystem.CARTESIAN),
        ),
        (
            coordinates.Coord(1.0, 2.0, 3.0, coordinates.CoordinateSystem.CARTESIAN2D),
            coordinates.Coord(-3.0, 1.0, 0.0, coordinates.CoordinateSystem.CARTESIAN),
        ),
        (
            coordinates.Coord(
                1.0, 2.0, 3.0, coordinates.CoordinateSystem.CARTESIAN_ROTATED
            ),
            coordinates.Coord(-3.0, 1.0, 2.0, coordinates.CoordinateSystem.CARTESIAN),
        ),
    ],
)
def test_coord_to_cartesian(
    coord1: coordinates.Coord, coord2: coordinates.Coord
) -> None:
    assert coord1.to_cartesian().approx_eq(coord2)


@given(non_nans(), non_nans(), non_nans(), integers(1, 50), coordinate_systems)
def test_to_cartesian_idempotent(
    x1: float, x2: float, x3: float, n: int, c: coordinates.CoordinateSystem
) -> None:
    new_coords = cartesian_coords = coordinates.Coord(x1, x2, x3, c).to_cartesian()
    for _ in range(n):
        new_coords = new_coords.to_cartesian()
        assert new_coords.approx_eq(cartesian_coords)


@pytest.mark.parametrize(
    "coord1,coord2,places",
    [
        (
            coordinates.Coord(1.0, 1.0, -0.5, coordinates.CoordinateSystem.CARTESIAN),
            coordinates.Coord(
                1.0001, 0.99999, -0.500005, coordinates.CoordinateSystem.CARTESIAN
            ),
            4,
        ),
        (
            coordinates.Coord(
                3315.05, 0.0012121, 1e-12, coordinates.CoordinateSystem.CARTESIAN
            ),
            coordinates.Coord(
                3315.05, 0.0012121441, 2e-12, coordinates.CoordinateSystem.CARTESIAN
            ),
            5,
        ),
        (
            coordinates.Coord(1.0, 1, -10.0, coordinates.CoordinateSystem.CYLINDRICAL),
            coordinates.Coord(
                1.0001, 0.99999, -9.9999, coordinates.CoordinateSystem.CYLINDRICAL
            ),
            4,
        ),
    ],
)
def test_coord_round(
    coord1: coordinates.Coord, coord2: coordinates.Coord, places: int
) -> None:
    assert coord1 != coord2
    assert coord1.round_to(places) == coord2.round_to(places)
    assert coord1.round_to(places + 1) != coord2.round_to(places + 1)


@given(builds(coordinates.Coord), builds(coordinates.Coord))
def test_coord_hash(coord1: coordinates.Coord, coord2: coordinates.Coord) -> None:
    if hash(coord1) != hash(coord2):
        assert coord1 != coord2


@pytest.mark.filterwarnings("ignore:invalid value:RuntimeWarning")
@given(non_nans(), non_nans(), non_nans(), coordinate_systems, non_nans())
def test_coord_offset_x1_x2_unchanged(
    x1: float, x2: float, x3: float, c: coordinates.CoordinateSystem, x: float
) -> None:
    coord = coordinates.Coord(x1, x2, x3, c)
    new_coord = coord.offset(x)
    assert new_coord.x1 is coord.x1
    assert new_coord.x2 is coord.x2
    assert new_coord.x3 is not coord.x3
    assert new_coord.system is coord.system


@pytest.mark.parametrize(
    "x1,x2,x3,offset,expected",
    [
        (
            0.0,
            1.0,
            10.0,
            5.0,
            15.0,
        ),
        (
            5.0,
            1.0,
            2.0,
            1.0,
            3.0,
        ),
    ],
)
def test_coord_offset(
    x1: float,
    x2: float,
    x3: float,
    offset: float,
    expected: float,
) -> None:
    coord = coordinates.Coord(x1, x2, x3, coordinates.CoordinateSystem.CARTESIAN)
    new_coord = coord.offset(offset)
    assert coord.x1 == new_coord.x1
    assert coord.x2 == new_coord.x2
    assert coord.system == new_coord.system
    np.testing.assert_allclose(new_coord.x3, expected)


@pytest.mark.parametrize(
    "x1,x2,x3,expected",
    [
        (
            np.array([0.0, 1.0, 2.0]),
            np.array([1.0, 0.5, 0.0]),
            np.array([10.0, -5.0, 1.0]),
            [
                coordinates.Coord(
                    0.0, 1.0, 10.0, coordinates.CoordinateSystem.CARTESIAN
                ),
                coordinates.Coord(
                    1.0, 0.5, -5.0, coordinates.CoordinateSystem.CARTESIAN
                ),
                coordinates.Coord(
                    2.0, 0.0, 1.0, coordinates.CoordinateSystem.CARTESIAN
                ),
            ],
        ),
        (
            np.array([5.0, 0.0]),
            np.array([[1.0, 1.5], [-10.0, -11.0]]),
            np.array([[2.0], [-2.0]]),
            [
                coordinates.Coord(
                    5.0, 1.0, 2.0, coordinates.CoordinateSystem.CARTESIAN
                ),
                coordinates.Coord(
                    0.0, 1.5, 2.0, coordinates.CoordinateSystem.CARTESIAN
                ),
                coordinates.Coord(
                    5.0, -10.0, -2.0, coordinates.CoordinateSystem.CARTESIAN
                ),
                coordinates.Coord(
                    0.0, -11.0, -2.0, coordinates.CoordinateSystem.CARTESIAN
                ),
            ],
        ),
    ],
)
def test_coords_iter_points(
    x1: npt.NDArray, x2: npt.NDArray, x3: npt.NDArray, expected: list[coordinates.Coord]
) -> None:
    for c1, c2 in zip(
        coordinates.Coords(
            x1, x2, x3, coordinates.CoordinateSystem.CARTESIAN
        ).iter_points(),
        expected,
    ):
        assert c1 == c2


def test_coords_iter_points_empty() -> None:
    coords = coordinates.Coords(
        np.array([]), np.array([]), np.array([]), coordinates.CoordinateSystem.CARTESIAN
    )
    assert len(list(coords.iter_points())) == 0


@given(from_type(coordinates.Coords))
def test_coords_iter(coords: coordinates.Coords) -> None:
    coords_iter = iter(coords)
    assert np.all(next(coords_iter) == coords.x1)
    assert np.all(next(coords_iter) == coords.x2)
    assert np.all(next(coords_iter) == coords.x3)
    with pytest.raises(StopIteration):
        next(coords_iter)


@given(from_type(coordinates.Coords))
def test_coords_len(coords: coordinates.Coords) -> None:
    coords_iter = coords.iter_points()
    for _ in range(len(coords)):
        _ = next(coords_iter)
    with pytest.raises(StopIteration):
        next(coords_iter)


@given(from_type(coordinates.Coords))
def test_coords_shape(x: coordinates.Coords) -> None:
    if len(x.shape) > 0:
        assert np.prod(x.shape) == len(x)
    else:
        assert len(x) == 1


@pytest.mark.parametrize(
    "x1,x2,x3,index,expected",
    [
        (
            np.array([0.0, 1.0, 2.0]),
            np.array([1.0, 0.5, 0.0]),
            np.array([10.0, -5.0, 1.0]),
            1,
            coordinates.Coord(1.0, 0.5, -5.0, coordinates.CoordinateSystem.CARTESIAN),
        ),
        (
            np.array([5.0, 0.0]),
            np.array([[1.0, 1.5], [-10.0, -11.0]]),
            np.array([[2.0], [-2.0]]),
            (1, 0),
            coordinates.Coord(5.0, -10.0, -2.0, coordinates.CoordinateSystem.CARTESIAN),
        ),
    ],
)
def test_coords_getitem(
    x1: npt.NDArray,
    x2: npt.NDArray,
    x3: npt.NDArray,
    index: tuple[int, ...] | int,
    expected: coordinates.Coord,
) -> None:
    coords = coordinates.Coords(x1, x2, x3, coordinates.CoordinateSystem.CARTESIAN)
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
    coords = coordinates.Coords(x1, x2, x3, coordinates.CoordinateSystem.CARTESIAN)
    with pytest.raises(expected):
        _ = coords[index]  # type: ignore


@given(from_type(coordinates.Coords).filter(lambda c: len(c) > 1))
def test_coords_get_set(coords: coordinates.Coords) -> None:
    coordset = coords.get_set(slice(None))
    assert len(coordset) <= len(coords)
    for coord in coords.iter_points():
        assert coord in coordset


def test_coords_get_set_partial() -> None:
    coords = coordinates.Coords(
        np.linspace(0.0, 1.0, 10),
        np.linspace(-50.0, -68.0, 10),
        np.linspace(5.0, 11.0, 10),
        coordinates.CoordinateSystem.CARTESIAN,
    )
    coordset = coords.get_set(slice(0, 5))
    assert len(coordset) == 5
    for i in range(5):
        assert coords[i] in coordset
    for i in range(5, 10):
        assert coords[i] not in coordset


@pytest.mark.filterwarnings("ignore:invalid value:RuntimeWarning")
@given(from_type(coordinates.Coords), non_nans())
def test_coords_offset_x1_x2_unchanged(coords: coordinates.Coords, x: float) -> None:
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
    coords = coordinates.Coords(x1, x2, x3, coordinates.CoordinateSystem.CARTESIAN)
    new_coords = coords.offset(offset)
    np.testing.assert_allclose(new_coords.x3, expected)


@pytest.mark.filterwarnings("ignore:invalid value:RuntimeWarning")
@given(mutually_broadcastable_arrays(3))
def test_coords_cartesian_to_cartesian(
    xs: tuple[npt.NDArray, npt.NDArray, npt.NDArray],
) -> None:
    coords = coordinates.Coords(*xs, coordinates.CoordinateSystem.CARTESIAN)
    new_coords = coords.to_cartesian()
    assert coords.x1 is new_coords.x1
    assert coords.x2 is new_coords.x2
    assert coords.x3 is new_coords.x3


@pytest.mark.filterwarnings("ignore:invalid value:RuntimeWarning")
@given(from_type(coordinates.Coords))
def test_coords_cartesian_correct_system(coords: coordinates.Coords) -> None:
    assert coords.to_cartesian().system is coordinates.CoordinateSystem.CARTESIAN


@pytest.mark.filterwarnings("ignore:invalid value:RuntimeWarning")
@given(mutually_broadcastable_arrays(3))
def test_coords_cylindrical_to_cartesian_z_unchanged(
    xs: tuple[npt.NDArray, npt.NDArray, npt.NDArray],
) -> None:
    coords = coordinates.Coords(*xs, coordinates.CoordinateSystem.CYLINDRICAL)
    assert np.all(coords.to_cartesian().x3 == coords.x2)


def test_coords_to_cartesian() -> None:
    coords = coordinates.Coords(
        np.array([1.0, 1.5, 2.0]),
        np.array([1.0, 0.5, 0.0]),
        np.array([0.0, np.pi / 2.0, np.pi]),
        coordinates.CoordinateSystem.CYLINDRICAL,
    ).to_cartesian()
    np.testing.assert_allclose(coords.x1, [1.0, 0.0, -2.0], atol=1e-12)
    np.testing.assert_allclose(coords.x2, [0.0, 1.5, 0.0], atol=1e-12)
    np.testing.assert_allclose(coords.x3, [1.0, 0.5, 0.0], atol=1e-12)


@pytest.mark.parametrize(
    "coord1,coord2,places",
    [
        (
            coordinates.Coords(
                np.array([1.0, 5.0]),
                np.array([1.0, 1.0]),
                np.array([-1e-3, -2e-5]),
                coordinates.CoordinateSystem.CARTESIAN,
            ),
            coordinates.Coords(
                np.array([1.0001, 5.0]),
                np.array([0.99999, 1.0]),
                np.array([-1.0001e-3, -1.99999e-5]),
                coordinates.CoordinateSystem.CARTESIAN,
            ),
            4,
        ),
        (
            coordinates.Coords(
                np.array([3315.05, 1e-12]),
                np.array([0.0012121, 1e100]),
                np.array([0.0, 22.0]),
                coordinates.CoordinateSystem.CARTESIAN,
            ),
            coordinates.Coords(
                np.array([3315.05, 0.0]),
                np.array([0.0012121441, 1.0000055e100]),
                np.array([-0.000000009, 22.0]),
                coordinates.CoordinateSystem.CARTESIAN,
            ),
            5,
        ),
        (
            coordinates.Coords(
                np.array(1.0),
                np.array(1.0),
                np.array(1.0),
                coordinates.CoordinateSystem.CYLINDRICAL,
            ),
            coordinates.Coords(
                np.array(1.0001),
                np.array(0.99999),
                np.array(1.0),
                coordinates.CoordinateSystem.CYLINDRICAL,
            ),
            4,
        ),
    ],
)
def test_coords_round(
    coord1: coordinates.Coords, coord2: coordinates.Coords, places: int
) -> None:
    def coords_equal(lhs: coordinates.Coords, rhs: coordinates.Coords) -> bool:
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


@given(from_type(coordinates.Coords))
def test_coords_to_slice_coords(coords: coordinates.Coords) -> None:
    slice_coords = coords.to_slice_coords()
    assert np.all(slice_coords.x1 == coords.x1)
    assert np.all(slice_coords.x2 == coords.x2)
    assert slice_coords.system == coords.system
