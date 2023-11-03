import itertools

import hypnotoad  # type: ignore
import numpy as np

from neso_fame import generators
from neso_fame.fields import straight_field
from neso_fame.mesh import CoordinateSystem, SliceCoords, SliceCoord, control_points, Quad
from .test_hypnotoad import CONNECTED_DOUBLE_NULL, connected_double_null, to_mesh


# Test for simple grid
def test_simple_grid_2d() -> None:
    starts = SliceCoords(
        np.linspace(
            -1.0,
            1.0,
            5,
        ),
        np.zeros(5),
        CoordinateSystem.CARTESIAN,
    )
    field = straight_field()
    x3 = (0.0, 2.0)
    n = 4
    resolution = 2
    mesh = generators.field_aligned_2d(starts, field, x3, n, resolution)
    assert len(mesh) == 16
    x3_start = x3[0]
    dx3 = (x3[1] - x3[0]) / n
    # Check corners of quads are in correct locations
    for layer in mesh.layers():
        x3_end = x3_start + dx3
        for i, quad in enumerate(layer):
            corners = quad.corners()
            assert len(corners) == 4
            x1_0 = starts.x1[i]
            x1_2 = starts.x1[i + 1]
            np.testing.assert_allclose(corners.x1[[0, 1]], x1_0)
            np.testing.assert_allclose(corners.x1[[2, 3]], x1_2)
            assert np.all(corners.x2 == 0.0)
            np.testing.assert_allclose(corners.x3[[0, 2]], x3_start)
            np.testing.assert_allclose(corners.x3[[1, 3]], x3_end)
        x3_start = x3_end
        bounds = list(layer.boundaries())
        assert len(bounds) == 2
        expected_north = frozenset(
            (
                tuple(x.iter_points())
                for x in filter(
                    lambda x: np.all(x.x1 == starts.x1[0]),
                    (control_points(x.north, 1) for x in layer),
                )
            )
        )
        expected_south = frozenset(
            (
                tuple(x.iter_points())
                for x in filter(
                    lambda x: np.all(x.x1 == starts.x1[-1]),
                    (control_points(x.south, 1) for x in layer),
                )
            )
        )
        actual_north = frozenset(
            (tuple(control_points(x, 1).iter_points()) for x in bounds[0])
        )
        actual_south = frozenset(
            (tuple(control_points(x, 1).iter_points()) for x in bounds[1])
        )
        assert actual_north == expected_north
        assert actual_south == expected_south


def test_angled_grid_conforming_bounds_2d() -> None:
    angle = np.pi / 18
    m = 4
    starts = SliceCoords(
        np.linspace(
            1.0,
            4.0,
            m,
        ),
        np.zeros(m),
        CoordinateSystem.CARTESIAN,
    )
    field = straight_field(angle)
    x3 = (-2.0, 1.0)
    n = 5
    mesh = generators.field_aligned_2d(starts, field, x3, n, conform_to_bounds=True)
    assert len(mesh) == n * (m - 1)
    x3_start = x3[0]
    dx3 = (x3[1] - x3[0]) / n
    x1_offsets = np.array([-np.tan(angle) * dx3 / 2, 0.0, np.tan(angle) * dx3 / 2])
    # Check control points of quad curves are in right location
    for layer in mesh.layers():
        x3_end = x3_start + dx3
        x3_positions = np.array([x3_start, x3_start + dx3 / 2, x3_end])
        for i, quad in enumerate(layer):
            x1_south_mid = starts.x1[i + 1]
            x1_north_mid = starts.x1[i]
            south_points = control_points(quad.south, 2)
            if i == m - 2:
                np.testing.assert_allclose(south_points.x1, x1_south_mid)
            else:
                np.testing.assert_allclose(south_points.x1, x1_south_mid + x1_offsets)
            assert np.all(south_points.x2 == 0.0)
            np.testing.assert_allclose(south_points.x3, x3_positions)
            north_points = control_points(quad.north, 2)
            if i == 0:
                np.testing.assert_allclose(north_points.x1, x1_north_mid)
            else:
                np.testing.assert_allclose(north_points.x1, x1_north_mid + x1_offsets)
            assert np.all(north_points.x2 == 0.0)
            np.testing.assert_allclose(north_points.x3, x3_positions)
        x3_start = x3_end
        bounds = list(layer.boundaries())
        assert len(bounds) == 2
        expected_north = frozenset(
            (
                tuple(x.iter_points())
                for x in filter(
                    lambda x: np.all(x.x1 == starts.x1[0]),
                    (control_points(x.north, 1) for x in layer),
                )
            )
        )
        expected_south = frozenset(
            (
                tuple(x.iter_points())
                for x in filter(
                    lambda x: np.all(x.x1 == starts.x1[-1]),
                    (control_points(x.south, 1) for x in layer),
                )
            )
        )
        actual_north = frozenset(
            (tuple(control_points(x, 1).iter_points()) for x in bounds[0])
        )
        actual_south = frozenset(
            (tuple(control_points(x, 1).iter_points()) for x in bounds[1])
        )
        assert actual_north == expected_north
        assert actual_south == expected_south


# Test for angled grid
def test_angled_grid_jagged_bounds_2d() -> None:
    angle = np.pi / 18
    m = 4
    starts = SliceCoords(
        np.linspace(
            1.0,
            4.0,
            m,
        ),
        np.zeros(m),
        CoordinateSystem.CARTESIAN,
    )
    field = straight_field(angle)
    x3 = (-2.0, 1.0)
    n = 5
    mesh = generators.field_aligned_2d(starts, field, x3, n, conform_to_bounds=False)
    assert len(mesh) == n * (m - 1)
    x3_start = x3[0]
    dx3 = (x3[1] - x3[0]) / n
    x1_offsets = np.array([-np.tan(angle) * dx3 / 2, 0.0, np.tan(angle) * dx3 / 2])
    # Check control points of quad curves are in right location
    for layer in mesh.layers():
        x3_end = x3_start + dx3
        x3_positions = np.array([x3_start, x3_start + dx3 / 2, x3_end])
        for i, quad in enumerate(layer):
            x1_south_mid = starts.x1[i + 1]
            x1_north_mid = starts.x1[i]
            south_points = control_points(quad.south, 2)
            np.testing.assert_allclose(south_points.x1, x1_south_mid + x1_offsets)
            assert np.all(south_points.x2 == 0.0)
            np.testing.assert_allclose(south_points.x3, x3_positions)
            north_points = control_points(quad.north, 2)
            np.testing.assert_allclose(north_points.x1, x1_north_mid + x1_offsets)
            assert np.all(north_points.x2 == 0.0)
            np.testing.assert_allclose(north_points.x3, x3_positions)
        x3_start = x3_end
        bounds = list(layer.boundaries())
        assert len(bounds) == 2
        expected_north = frozenset(
            (
                tuple(x.iter_points())
                for x in filter(
                    lambda x: np.allclose(x.x1, starts.x1[0] + x1_offsets[[0, 2]]),
                    (control_points(x.north, 1) for x in layer),
                )
            )
        )
        expected_south = frozenset(
            (
                tuple(x.iter_points())
                for x in filter(
                    lambda x: np.allclose(x.x1, starts.x1[-1] + x1_offsets[[0, 2]]),
                    (control_points(x.south, 1) for x in layer),
                )
            )
        )
        actual_north = frozenset(
            (tuple(control_points(x, 1).iter_points()) for x in bounds[0])
        )
        actual_south = frozenset(
            (tuple(control_points(x, 1).iter_points()) for x in bounds[1])
        )
        assert actual_north == expected_north
        assert actual_south == expected_south


# Test for simple grid
def test_subdivided_grid_2d() -> None:
    starts = SliceCoords(
        np.linspace(
            -1.0,
            1.0,
            5,
        ),
        np.zeros(5),
        CoordinateSystem.CARTESIAN,
    )
    field = straight_field()
    x3 = (0.0, 2.0)
    n = 4
    resolution = 2
    mesh = generators.field_aligned_2d(starts, field, x3, 1, resolution, subdivisions=n)
    assert len(mesh) == 16
    dx3 = (x3[1] - x3[0]) / n
    # Check corners of quads are in correct locations
    for k, quad in enumerate(mesh):
        corners = quad.corners()
        assert len(corners) == 4
        i = k // n
        j = k % n
        x1_0 = starts.x1[i]
        x1_2 = starts.x1[i + 1]
        np.testing.assert_allclose(corners.x1[[0, 1]], x1_0)
        np.testing.assert_allclose(corners.x1[[2, 3]], x1_2)
        assert np.all(corners.x2 == 0.0)
        np.testing.assert_allclose(corners.x3[[0, 2]], x3[0] + j * dx3)
        np.testing.assert_allclose(corners.x3[[1, 3]], x3[0] + (j + 1) * dx3)
    for layer in mesh.layers():
        bounds = list(layer.boundaries())
        assert len(bounds) == 2
        expected_north = frozenset(
            (
                tuple(x.iter_points())
                for x in filter(
                    lambda x: np.all(x.x1 == starts.x1[0]),
                    (control_points(x.north, 1) for x in layer),
                )
            )
        )
        expected_south = frozenset(
            (
                tuple(x.iter_points())
                for x in filter(
                    lambda x: np.all(x.x1 == starts.x1[-1]),
                    (control_points(x.south, 1) for x in layer),
                )
            )
        )
        actual_north = frozenset(
            (tuple(control_points(x, 1).iter_points()) for x in bounds[0])
        )
        actual_south = frozenset(
            (tuple(control_points(x, 1).iter_points()) for x in bounds[1])
        )
        assert actual_north == expected_north
        assert actual_south == expected_south


# Test for simple grid
def test_simple_grid_3d() -> None:
    n1 = 5
    n2 = 4
    x1, x2 = np.meshgrid(
        np.linspace(
            -1.0,
            1.0,
            n1,
        ),
        np.linspace(0.0, 1.0, n2),
        copy=False,
        sparse=False,
    )
    starts = SliceCoords(
        x1,
        x2,
        CoordinateSystem.CARTESIAN,
    )
    elements = [
        ((i, j), (i, j + 1), (i + 1, j + 1), (i + 1, j))
        for i in range(n2 - 1)
        for j in range(n1 - 1)
    ]
    field = straight_field()
    x3 = (0.0, 2.0)
    n = 4
    resolution = 2
    mesh = generators.field_aligned_3d(starts, field, elements, x3, n, resolution)
    assert len(mesh) == 48
    x3_start = x3[0]
    dx3 = (x3[1] - x3[0]) / n
    # Check corners of quads are in correct locations
    for layer in mesh.layers():
        x3_end = x3_start + dx3
        for index, hexa in enumerate(layer):
            corners = hexa.corners()
            assert len(corners) == 8
            i = index // (n1 - 1)
            j = index % (n1 - 1)
            x1_0_0 = starts.x1[i, j]
            x1_1_0 = starts.x1[i + 1, j]
            x1_0_1 = starts.x1[i, j + 1]
            x1_1_1 = starts.x1[i + 1, j + 1]
            np.testing.assert_allclose(corners.x1[[0, 1]], x1_1_0)
            np.testing.assert_allclose(corners.x1[[2, 3]], x1_1_1)
            np.testing.assert_allclose(corners.x1[[4, 5]], x1_0_0)
            np.testing.assert_allclose(corners.x1[[6, 7]], x1_0_1)
            x2_0_0 = starts.x1[i, j]
            x2_1_0 = starts.x1[i + 1, j]
            x2_0_1 = starts.x1[i, j + 1]
            x2_1_1 = starts.x1[i + 1, j + 1]
            np.testing.assert_allclose(corners.x1[[0, 1]], x2_1_0)
            np.testing.assert_allclose(corners.x1[[2, 3]], x2_1_1)
            np.testing.assert_allclose(corners.x1[[4, 5]], x2_0_0)
            np.testing.assert_allclose(corners.x1[[6, 7]], x2_0_1)
            np.testing.assert_allclose(corners.x3[::2], x3_start)
            np.testing.assert_allclose(corners.x3[1::2], x3_end)
        x3_start = x3_end
        bounds = list(layer.boundaries())
        assert len(bounds) == 4
        expected_north = frozenset(
            (
                tuple(x.iter_points())
                for x in filter(
                    lambda x: np.all(x.x2 == starts.x2[-1, -1]),
                    (control_points(x.north, 1) for x in layer),
                )
            )
        )
        expected_south = frozenset(
            (
                tuple(x.iter_points())
                for x in filter(
                    lambda x: np.all(x.x2 == starts.x2[0, 0]),
                    (control_points(x.south, 1) for x in layer),
                )
            )
        )
        expected_east = frozenset(
            (
                tuple(x.iter_points())
                for x in filter(
                    lambda x: np.all(x.x1 == starts.x1[-1, -1]),
                    (control_points(x.east, 1) for x in layer),
                )
            )
        )
        expected_west = frozenset(
            (
                tuple(x.iter_points())
                for x in filter(
                    lambda x: np.all(x.x1 == starts.x1[0, 0]),
                    (control_points(x.west, 1) for x in layer),
                )
            )
        )
        actual_north = frozenset(
            (tuple(control_points(x, 1).iter_points()) for x in bounds[0])
        )
        actual_south = frozenset(
            (tuple(control_points(x, 1).iter_points()) for x in bounds[1])
        )
        actual_east = frozenset(
            (tuple(control_points(x, 1).iter_points()) for x in bounds[2])
        )
        actual_west = frozenset(
            (tuple(control_points(x, 1).iter_points()) for x in bounds[3])
        )
        assert actual_north == expected_north
        assert actual_south == expected_south
        assert actual_east == expected_east
        assert actual_west == expected_west


def test_angled_grid_conforming_bounds_3d() -> None:
    angle_x1 = np.pi / 18
    angle_x2 = np.pi / 36
    m1 = 6
    m2 = 7
    x1, x2 = np.meshgrid(
        np.linspace(
            -1.0,
            1.0,
            m1,
        ),
        np.linspace(0.0, 1.0, m2),
        copy=False,
        sparse=False,
    )
    starts = SliceCoords(
        x1,
        x2,
        CoordinateSystem.CARTESIAN,
    )
    elements = [
        ((i, j + 1), (i, j), (i + 1, j), (i + 1, j + 1))
        for i in range(m2 - 1)
        for j in range(m1 - 1)
    ]
    field = straight_field(angle_x1, angle_x2)
    x3 = (-2.0, 1.0)
    n = 5
    mesh = generators.field_aligned_3d(
        starts, field, elements, x3, n, conform_to_bounds=True
    )
    assert len(mesh) == n * (m1 - 1) * (m2 - 1)
    x3_start = x3[0]
    dx3 = (x3[1] - x3[0]) / n
    x1_offsets = np.array(
        [-np.tan(angle_x1) * dx3 / 2, 0.0, np.tan(angle_x1) * dx3 / 2]
    )
    x2_offsets = np.array(
        [-np.tan(angle_x2) * dx3 / 2, 0.0, np.tan(angle_x2) * dx3 / 2]
    )
    # Check control points of hex curves are in right location
    for layer in mesh.layers():
        x3_end = x3_start + dx3
        x3_positions = np.array([x3_start, x3_start + dx3 / 2, x3_end])
        for index, hexa in enumerate(layer):
            i = index // (m1 - 1)
            j = index % (m1 - 1)
            x1_0_0_mid = starts.x1[i, j]
            x1_1_0_mid = starts.x1[i + 1, j]
            x1_0_1_mid = starts.x1[i, j + 1]
            x1_1_1_mid = starts.x1[i + 1, j + 1]
            x2_0_0_mid = starts.x2[i, j]
            x2_1_0_mid = starts.x2[i + 1, j]
            x2_0_1_mid = starts.x2[i, j + 1]
            x2_1_1_mid = starts.x2[i + 1, j + 1]
            points_0_0 = control_points(hexa.south.north, 2)
            points_1_0 = control_points(hexa.north.north, 2)
            points_0_1 = control_points(hexa.south.south, 2)
            points_1_1 = control_points(hexa.north.south, 2)
            if i == 0:
                np.testing.assert_allclose(points_0_1.x2, x2_0_1_mid)
                np.testing.assert_allclose(points_0_0.x2, x2_0_0_mid)
            else:
                np.testing.assert_allclose(points_0_1.x2, x2_0_1_mid + x2_offsets)
                np.testing.assert_allclose(points_0_0.x2, x2_0_0_mid + x2_offsets)
            if i == m2 - 2:
                np.testing.assert_allclose(points_1_1.x2, x2_1_1_mid)
                np.testing.assert_allclose(points_1_0.x2, x2_1_0_mid)
            else:
                np.testing.assert_allclose(points_1_1.x2, x2_1_1_mid + x2_offsets)
                np.testing.assert_allclose(points_1_0.x2, x2_1_0_mid + x2_offsets)
            if j == 0:
                np.testing.assert_allclose(points_0_0.x1, x1_0_0_mid)
                np.testing.assert_allclose(points_1_0.x1, x1_1_0_mid)
            else:
                np.testing.assert_allclose(points_0_0.x1, x1_0_0_mid + x1_offsets)
                np.testing.assert_allclose(points_1_0.x1, x1_1_0_mid + x1_offsets)
            if j == m1 - 2:
                np.testing.assert_allclose(points_0_1.x1, x1_0_1_mid)
                np.testing.assert_allclose(points_1_1.x1, x1_1_1_mid)
            else:
                np.testing.assert_allclose(points_0_1.x1, x1_0_1_mid + x1_offsets)
                np.testing.assert_allclose(points_1_1.x1, x1_1_1_mid + x1_offsets)

            np.testing.assert_allclose(points_0_0.x3, x3_positions)
            np.testing.assert_allclose(points_0_1.x3, x3_positions)
            np.testing.assert_allclose(points_1_0.x3, x3_positions)
            np.testing.assert_allclose(points_1_1.x3, x3_positions)
        x3_start = x3_end
        bounds = list(layer.boundaries())
        assert len(bounds) == 4
        expected_north = frozenset(
            (
                tuple(x.iter_points())
                for x in filter(
                    lambda x: np.all(x.x2 == starts.x2[-1, -1]),
                    (control_points(x.north, 1) for x in layer),
                )
            )
        )
        expected_south = frozenset(
            (
                tuple(x.iter_points())
                for x in filter(
                    lambda x: np.all(x.x2 == starts.x2[0, 0]),
                    (control_points(x.south, 1) for x in layer),
                )
            )
        )
        expected_east = frozenset(
            (
                tuple(x.iter_points())
                for x in filter(
                    lambda x: np.all(x.x1 == starts.x1[-1, -1]),
                    (control_points(x.east, 1) for x in layer),
                )
            )
        )
        expected_west = frozenset(
            (
                tuple(x.iter_points())
                for x in filter(
                    lambda x: np.all(x.x1 == starts.x1[0, 0]),
                    (control_points(x.west, 1) for x in layer),
                )
            )
        )
        actual_north = frozenset(
            (tuple(control_points(x, 1).iter_points()) for x in bounds[0])
        )
        actual_south = frozenset(
            (tuple(control_points(x, 1).iter_points()) for x in bounds[1])
        )
        actual_east = frozenset(
            (tuple(control_points(x, 1).iter_points()) for x in bounds[2])
        )
        actual_west = frozenset(
            (tuple(control_points(x, 1).iter_points()) for x in bounds[3])
        )
        assert actual_north == expected_north
        assert actual_south == expected_south
        assert actual_east == expected_east
        assert actual_west == expected_west


# Test for angled grid
def test_angled_grid_jagged_bounds_3d() -> None:
    angle_x1 = np.pi / 18
    angle_x2 = np.pi / 36
    m1 = 4
    m2 = 5
    x1, x2 = np.meshgrid(
        np.linspace(
            -1.0,
            1.0,
            m1,
        ),
        np.linspace(0.0, 1.0, m2),
        copy=False,
        sparse=False,
    )
    starts = SliceCoords(
        x1,
        x2,
        CoordinateSystem.CARTESIAN,
    )
    elements = [
        ((i, j), (i, j + 1), (i + 1, j + 1), (i + 1, j))
        for i in range(m2 - 1)
        for j in range(m1 - 1)
    ]
    field = straight_field(angle_x1, angle_x2)
    x3 = (-2.0, 1.0)
    n = 5
    mesh = generators.field_aligned_3d(
        starts, field, elements, x3, n, conform_to_bounds=False
    )
    assert len(mesh) == n * (m1 - 1) * (m2 - 1)
    x3_start = x3[0]
    dx3 = (x3[1] - x3[0]) / n
    x1_offsets = np.array(
        [-np.tan(angle_x1) * dx3 / 2, 0.0, np.tan(angle_x1) * dx3 / 2]
    )
    x2_offsets = np.array(
        [-np.tan(angle_x2) * dx3 / 2, 0.0, np.tan(angle_x2) * dx3 / 2]
    )
    # Check control points of hex curves are in right location
    for layer in mesh.layers():
        x3_end = x3_start + dx3
        x3_positions = np.array([x3_start, x3_start + dx3 / 2, x3_end])
        for index, hexa in enumerate(layer):
            i = index // (m1 - 1)
            j = index % (m1 - 1)
            x1_0_0_mid = starts.x1[i, j]
            x1_1_0_mid = starts.x1[i + 1, j]
            x1_0_1_mid = starts.x1[i, j + 1]
            x1_1_1_mid = starts.x1[i + 1, j + 1]
            x2_0_0_mid = starts.x2[i, j]
            x2_1_0_mid = starts.x2[i + 1, j]
            x2_0_1_mid = starts.x2[i, j + 1]
            x2_1_1_mid = starts.x2[i + 1, j + 1]
            points_0_0 = control_points(hexa.south.north, 2)
            points_1_0 = control_points(hexa.north.north, 2)
            points_0_1 = control_points(hexa.south.south, 2)
            points_1_1 = control_points(hexa.north.south, 2)
            np.testing.assert_allclose(points_1_0.x1, x1_1_0_mid + x1_offsets)
            np.testing.assert_allclose(
                points_1_0.x2, x2_1_0_mid + x2_offsets, atol=1e-12
            )
            np.testing.assert_allclose(points_1_1.x1, x1_1_1_mid + x1_offsets)
            np.testing.assert_allclose(
                points_1_1.x2, x2_1_1_mid + x2_offsets, atol=1e-12
            )
            np.testing.assert_allclose(points_0_0.x1, x1_0_0_mid + x1_offsets)
            np.testing.assert_allclose(
                points_0_0.x2, x2_0_0_mid + x2_offsets, atol=1e-12
            )
            np.testing.assert_allclose(points_0_1.x1, x1_0_1_mid + x1_offsets)
            np.testing.assert_allclose(
                points_0_1.x2, x2_0_1_mid + x2_offsets, atol=1e-12
            )
            np.testing.assert_allclose(points_0_0.x3, x3_positions)
            np.testing.assert_allclose(points_0_1.x3, x3_positions)
            np.testing.assert_allclose(points_1_0.x3, x3_positions)
            np.testing.assert_allclose(points_1_1.x3, x3_positions)
        x3_start = x3_end
        bounds = list(layer.boundaries())
        assert len(bounds) == 4
        expected_north = frozenset(
            (
                tuple(x.iter_points())
                for x in filter(
                    lambda x: np.allclose(x.x2, starts.x2[-1, -1] + x2_offsets[[0, 2]]),
                    (control_points(x.north, 1) for x in layer),
                )
            )
        )
        expected_south = frozenset(
            (
                tuple(x.iter_points())
                for x in filter(
                    lambda x: np.allclose(x.x2, starts.x2[0, 0] + x2_offsets[[0, 2]]),
                    (control_points(x.south, 1) for x in layer),
                )
            )
        )
        expected_east = frozenset(
            (
                tuple(x.iter_points())
                for x in filter(
                    lambda x: np.allclose(x.x1, starts.x1[-1, -1] + x1_offsets[[0, 2]]),
                    (control_points(x.east, 1) for x in layer),
                )
            )
        )
        expected_west = frozenset(
            (
                tuple(x.iter_points())
                for x in filter(
                    lambda x: np.allclose(x.x1, starts.x1[0, 0] + x1_offsets[[0, 2]]),
                    (control_points(x.west, 1) for x in layer),
                )
            )
        )
        actual_north = frozenset(
            (tuple(control_points(x, 1).iter_points()) for x in bounds[0])
        )
        actual_south = frozenset(
            (tuple(control_points(x, 1).iter_points()) for x in bounds[1])
        )
        actual_east = frozenset(
            (tuple(control_points(x, 1).iter_points()) for x in bounds[2])
        )
        actual_west = frozenset(
            (tuple(control_points(x, 1).iter_points()) for x in bounds[3])
        )
        assert actual_north == expected_north
        assert actual_south == expected_south
        assert actual_east == expected_east
        assert actual_west == expected_west


# Test for simple grid
def test_subdivided_grid_3d() -> None:
    resolution = 2
    m1 = 4
    m2 = 5
    x1, x2 = np.meshgrid(
        np.linspace(
            -1.0,
            1.0,
            m1,
        ),
        np.linspace(0.0, 1.0, m2),
        copy=False,
        sparse=False,
    )
    starts = SliceCoords(
        x1,
        x2,
        CoordinateSystem.CARTESIAN,
    )
    elements = [
        ((i, j), (i, j + 1), (i + 1, j + 1), (i + 1, j))
        for i in range(m2 - 1)
        for j in range(m1 - 1)
    ]
    field = straight_field()
    x3 = (-2.0, 1.0)
    n = 5
    mesh = generators.field_aligned_3d(
        starts, field, elements, x3, 1, resolution, subdivisions=n
    )
    assert len(mesh) == n * (m1 - 1) * (m2 - 1)
    dx3 = (x3[1] - x3[0]) / n
    # Check corners of quads are in correct locations
    for index, hexa in enumerate(mesh):
        corners = hexa.corners()
        print(index, corners)
        assert len(corners) == 8
        i = index // n // (m1 - 1)
        j = index // n % (m1 - 1)
        k = index % n
        x1_0_0 = starts.x1[i, j]
        x1_1_0 = starts.x1[i + 1, j]
        x1_0_1 = starts.x1[i, j + 1]
        x1_1_1 = starts.x1[i + 1, j + 1]
        np.testing.assert_allclose(corners.x1[[0, 1]], x1_1_0)
        np.testing.assert_allclose(corners.x1[[2, 3]], x1_1_1)
        np.testing.assert_allclose(corners.x1[[4, 5]], x1_0_0)
        np.testing.assert_allclose(corners.x1[[6, 7]], x1_0_1)
        x2_0_0 = starts.x1[i, j]
        x2_1_0 = starts.x1[i + 1, j]
        x2_0_1 = starts.x1[i, j + 1]
        x2_1_1 = starts.x1[i + 1, j + 1]
        np.testing.assert_allclose(corners.x1[[0, 1]], x2_1_0)
        np.testing.assert_allclose(corners.x1[[2, 3]], x2_1_1)
        np.testing.assert_allclose(corners.x1[[4, 5]], x2_0_0)
        np.testing.assert_allclose(corners.x1[[6, 7]], x2_0_1)
        np.testing.assert_allclose(corners.x3[::2], x3[0] + k * dx3)
        np.testing.assert_allclose(corners.x3[1::2], x3[0] + (k + 1) * dx3)
    for layer in mesh.layers():
        bounds = list(layer.boundaries())
        assert len(bounds) == 4
        expected_north = frozenset(
            (
                tuple(x.iter_points())
                for x in filter(
                    lambda x: np.all(x.x2 == starts.x2[-1, -1]),
                    (control_points(x.north, 1) for x in layer),
                )
            )
        )
        expected_south = frozenset(
            (
                tuple(x.iter_points())
                for x in filter(
                    lambda x: np.all(x.x2 == starts.x2[0, 0]),
                    (control_points(x.south, 1) for x in layer),
                )
            )
        )
        expected_east = frozenset(
            (
                tuple(x.iter_points())
                for x in filter(
                    lambda x: np.all(x.x1 == starts.x1[-1, -1]),
                    (control_points(x.east, 1) for x in layer),
                )
            )
        )
        expected_west = frozenset(
            (
                tuple(x.iter_points())
                for x in filter(
                    lambda x: np.all(x.x1 == starts.x1[0, 0]),
                    (control_points(x.west, 1) for x in layer),
                )
            )
        )
        actual_north = frozenset(
            (tuple(control_points(x, 1).iter_points()) for x in bounds[0])
        )
        actual_south = frozenset(
            (tuple(control_points(x, 1).iter_points()) for x in bounds[1])
        )
        actual_east = frozenset(
            (tuple(control_points(x, 1).iter_points()) for x in bounds[2])
        )
        actual_west = frozenset(
            (tuple(control_points(x, 1).iter_points()) for x in bounds[3])
        )
        assert actual_north == expected_north
        assert actual_south == expected_south
        assert actual_east == expected_east
        assert actual_west == expected_west


def test_extruding_hypnotoad_mesh() -> None:
    hypno_mesh = CONNECTED_DOUBLE_NULL
    eq = hypno_mesh.equilibrium
    mesh = generators.hypnotoad_mesh(hypno_mesh, (0., 0.125*np.pi), 8, 21)
    actual_nodes = frozenset(itertools.chain.from_iterable((q.shape(0.).to_coord(), q.shape(1.).to_coord()) for q in itertools.chain.from_iterable(mesh)))
    expected_nodes = frozenset(itertools.chain.from_iterable(SliceCoords(r.Rxy.corners, r.Zxy.corners, CoordinateSystem.CYLINDRICAL).iter_points() for r in hypno_mesh.regions.values()))
    
    assert actual_nodes == expected_nodes
    lines = frozenset(itertools.chain.from_iterable((q.north, q.south) for q in itertools.chain.from_iterable(mesh)))
    for line in lines:
        print(line)
        R, Z, _ = control_points(line, 8)
        # Ignore any values that leave the domain, as these won't be accurate
        in_domain = np.logical_and(np.logical_and(R <= eq.Rmax, R >= eq.Rmin), np.logical_and(Z <= eq.Zmax, Z >= eq.Zmin))
        psis = np.asarray(eq.psi(R, Z))[in_domain]
        np.testing.assert_allclose(psis, psis[0], 1e-5, 1e-5)
    # Get a list all boundaries. Each entry in the list will be
    # another list containing the portions of that boundary in each
    # layer.
    bounds = list(zip(*(list(l.boundaries()) for l in mesh.layers())))
    for bound in bounds:
        sizes = {len(b) for b in bound}
        # Check that the boundary is the same size in each layer
        assert len(sizes) == 1
        # Check that the boundary is not empty
        assert all(s > 0 for s in sizes)
