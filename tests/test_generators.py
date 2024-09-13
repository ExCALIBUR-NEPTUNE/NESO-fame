import itertools
from collections.abc import Iterator
from functools import reduce
from typing import Optional
from unittest.mock import MagicMock

import numpy as np
import pytest

from neso_fame import generators
from neso_fame.element_builder import ElementBuilder
from neso_fame.fields import straight_field
from neso_fame.mesh import (
    Coord,
    CoordinateSystem,
    FieldAlignedCurve,
    FieldTracer,
    Prism,
    Quad,
    Segment,
    SliceCoord,
    SliceCoords,
    control_points,
)
from neso_fame.nektar_writer import nektar_3d_element
from tests.conftest import simple_trace

from .test_hypnotoad import CONNECTED_DOUBLE_NULL, to_mesh


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
            np.testing.assert_allclose(corners.x1[[0, 4]], x1_1_0)
            np.testing.assert_allclose(corners.x1[[1, 5]], x1_1_1)
            np.testing.assert_allclose(corners.x1[[2, 6]], x1_0_0)
            np.testing.assert_allclose(corners.x1[[3, 7]], x1_0_1)
            x2_0_0 = starts.x1[i, j]
            x2_1_0 = starts.x1[i + 1, j]
            x2_0_1 = starts.x1[i, j + 1]
            x2_1_1 = starts.x1[i + 1, j + 1]
            np.testing.assert_allclose(corners.x1[[0, 4]], x2_1_0)
            np.testing.assert_allclose(corners.x1[[1, 5]], x2_1_1)
            np.testing.assert_allclose(corners.x1[[2, 6]], x2_0_0)
            np.testing.assert_allclose(corners.x1[[3, 7]], x2_0_1)
            np.testing.assert_allclose(corners.x3[:4], x3_start)
            np.testing.assert_allclose(corners.x3[4:], x3_end)
        x3_start = x3_end
        bounds = list(layer.boundaries())
        assert len(bounds) == 4
        expected_north = frozenset(
            (
                tuple(x.iter_points())
                for x in filter(
                    lambda x: np.all(x.x2 == starts.x2[-1, -1]),
                    (control_points(x.sides[0], 1) for x in layer),
                )
            )
        )
        expected_south = frozenset(
            (
                tuple(x.iter_points())
                for x in filter(
                    lambda x: np.all(x.x2 == starts.x2[0, 0]),
                    (control_points(x.sides[1], 1) for x in layer),
                )
            )
        )
        expected_east = frozenset(
            (
                tuple(x.iter_points())
                for x in filter(
                    lambda x: np.all(x.x1 == starts.x1[-1, -1]),
                    (control_points(x.sides[2], 1) for x in layer),
                )
            )
        )
        expected_west = frozenset(
            (
                tuple(x.iter_points())
                for x in filter(
                    lambda x: np.all(x.x1 == starts.x1[0, 0]),
                    (control_points(x.sides[3], 1) for x in layer),
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
            points_0_0 = control_points(hexa.sides[1].north, 2)
            points_1_0 = control_points(hexa.sides[0].north, 2)
            points_0_1 = control_points(hexa.sides[1].south, 2)
            points_1_1 = control_points(hexa.sides[0].south, 2)
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
                    (control_points(x.sides[0], 1) for x in layer),
                )
            )
        )
        expected_south = frozenset(
            (
                tuple(x.iter_points())
                for x in filter(
                    lambda x: np.all(x.x2 == starts.x2[0, 0]),
                    (control_points(x.sides[1], 1) for x in layer),
                )
            )
        )
        expected_east = frozenset(
            (
                tuple(x.iter_points())
                for x in filter(
                    lambda x: np.all(x.x1 == starts.x1[-1, -1]),
                    (control_points(x.sides[2], 1) for x in layer),
                )
            )
        )
        expected_west = frozenset(
            (
                tuple(x.iter_points())
                for x in filter(
                    lambda x: np.all(x.x1 == starts.x1[0, 0]),
                    (control_points(x.sides[3], 1) for x in layer),
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
            points_0_0 = control_points(hexa.sides[1].north, 2)
            points_1_0 = control_points(hexa.sides[0].north, 2)
            points_0_1 = control_points(hexa.sides[1].south, 2)
            points_1_1 = control_points(hexa.sides[2].south, 2)
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
                    (control_points(x.sides[0], 1) for x in layer),
                )
            )
        )
        expected_south = frozenset(
            (
                tuple(x.iter_points())
                for x in filter(
                    lambda x: np.allclose(x.x2, starts.x2[0, 0] + x2_offsets[[0, 2]]),
                    (control_points(x.sides[1], 1) for x in layer),
                )
            )
        )
        expected_east = frozenset(
            (
                tuple(x.iter_points())
                for x in filter(
                    lambda x: np.allclose(x.x1, starts.x1[-1, -1] + x1_offsets[[0, 2]]),
                    (control_points(x.sides[2], 1) for x in layer),
                )
            )
        )
        expected_west = frozenset(
            (
                tuple(x.iter_points())
                for x in filter(
                    lambda x: np.allclose(x.x1, starts.x1[0, 0] + x1_offsets[[0, 2]]),
                    (control_points(x.sides[3], 1) for x in layer),
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
        assert len(corners) == 8
        i = index // n // (m1 - 1)
        j = index // n % (m1 - 1)
        k = index % n
        x1_0_0 = starts.x1[i, j]
        x1_1_0 = starts.x1[i + 1, j]
        x1_0_1 = starts.x1[i, j + 1]
        x1_1_1 = starts.x1[i + 1, j + 1]
        np.testing.assert_allclose(corners.x1[[0, 4]], x1_1_0)
        np.testing.assert_allclose(corners.x1[[1, 5]], x1_1_1)
        np.testing.assert_allclose(corners.x1[[2, 6]], x1_0_0)
        np.testing.assert_allclose(corners.x1[[3, 7]], x1_0_1)
        x2_0_0 = starts.x1[i, j]
        x2_1_0 = starts.x1[i + 1, j]
        x2_0_1 = starts.x1[i, j + 1]
        x2_1_1 = starts.x1[i + 1, j + 1]
        np.testing.assert_allclose(corners.x1[[0, 4]], x2_1_0)
        np.testing.assert_allclose(corners.x1[[1, 5]], x2_1_1)
        np.testing.assert_allclose(corners.x1[[2, 6]], x2_0_0)
        np.testing.assert_allclose(corners.x1[[3, 7]], x2_0_1)
        np.testing.assert_allclose(corners.x3[:4], x3[0] + k * dx3)
        np.testing.assert_allclose(corners.x3[4:], x3[0] + (k + 1) * dx3)
    for layer in mesh.layers():
        bounds = list(layer.boundaries())
        assert len(bounds) == 4
        expected_north = frozenset(
            (
                tuple(x.iter_points())
                for x in filter(
                    lambda x: np.all(x.x2 == starts.x2[-1, -1]),
                    (control_points(x.sides[0], 1) for x in layer),
                )
            )
        )
        expected_south = frozenset(
            (
                tuple(x.iter_points())
                for x in filter(
                    lambda x: np.all(x.x2 == starts.x2[0, 0]),
                    (control_points(x.sides[1], 1) for x in layer),
                )
            )
        )
        expected_east = frozenset(
            (
                tuple(x.iter_points())
                for x in filter(
                    lambda x: np.all(x.x1 == starts.x1[-1, -1]),
                    (control_points(x.sides[2], 1) for x in layer),
                )
            )
        )
        expected_west = frozenset(
            (
                tuple(x.iter_points())
                for x in filter(
                    lambda x: np.all(x.x1 == starts.x1[0, 0]),
                    (control_points(x.sides[3], 1) for x in layer),
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


def test_iterate_and_merge_elements() -> None:
    R = np.array(
        [
            [0.0, 0.05, 2.0, 3.95, 4.0],
            [0.0, 0.05, 2.0, 3.95, 4.0],
            [0.0, 1.0, 2.0, 3.0, 4.0],
        ]
    )
    Z = np.array(
        [
            [2.0, 2.0, 2.0, 2.0, 2.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    hypno_region = MagicMock()
    hypno_region.Rxy.corners = R
    hypno_region.Zxy.corners = Z
    hypno_region.equilibriumRegion.name = "core"
    hypno_region.connections = {"inner": None}
    elements = frozenset(
        generators._iter_element_corners(hypno_region, 10, CoordinateSystem.CYLINDRICAL)
    )

    def coord(R: float, Z: float) -> SliceCoord:
        return SliceCoord(R, Z, CoordinateSystem.CYLINDRICAL)

    assert elements == {
        (coord(0.0, 0.0), coord(1.0, 0.0), coord(0.0, 1.0), None),
        (coord(1.0, 0.0), coord(2.0, 0.0), coord(0.0, 1.0), coord(2.0, 1.0)),
        (coord(2.0, 0.0), coord(3.0, 0.0), coord(2.0, 1.0), coord(4.0, 1.0)),
        (coord(3.0, 0.0), coord(4.0, 0.0), coord(4.0, 1.0), None),
        (coord(0.0, 1.0), coord(2.0, 1.0), coord(0.0, 2.0), coord(2.0, 2.0)),
        (coord(2.0, 1.0), coord(4.0, 1.0), coord(2.0, 2.0), coord(4.0, 2.0)),
    }


def test_validate_wall_elements() -> None:
    c00 = SliceCoord(0.0, 0.0, CoordinateSystem.CARTESIAN)
    c03 = SliceCoord(0.0, 3.0, CoordinateSystem.CARTESIAN)
    c11 = SliceCoord(1.0, 1.0, CoordinateSystem.CARTESIAN)
    c20 = SliceCoord(2.0, 0.0, CoordinateSystem.CARTESIAN)
    c21 = SliceCoord(2.0, 1.0, CoordinateSystem.CARTESIAN)
    wall_vertices = frozenset({(c00, c20), (c20, c21)})
    builder = ElementBuilder(
        MagicMock(), FieldTracer(simple_trace, 2), 0.1, {}, CoordinateSystem.CARTESIAN
    )
    curved_quad = builder.make_wall_quad_for_prism(
        lambda s: SliceCoords(
            2 * np.asarray(s),
            np.interp(2 * np.asarray(s), [c00.x1, 0.1, c20.x1], [c00.x2, 0.5, c20.x2]),
            CoordinateSystem.CARTESIAN,
        )
    )
    p1, b1 = builder.make_outer_prism(c00, c20, c11, wall_vertices)
    p2, b2 = builder.make_outer_prism(c20, c21, c11, wall_vertices)
    p3, b3 = builder.make_outer_prism(c00, c11, c03, wall_vertices)
    prisms = [p1, p2, p3]
    bounds = b1 | b2 | b3
    assert curved_quad in bounds
    new_prisms, new_bounds = generators._validate_wall_elements(
        bounds,
        prisms,
        builder.get_element_for_quad,
        lambda x: next(iter(nektar_3d_element(x, 8, 3, -1)[0])).IsValid(),
    )
    p1_flat = p1.make_flat_faces()
    assert p1 not in new_prisms
    assert p1_flat in new_prisms
    assert p2 in new_prisms
    assert p3 in new_prisms
    assert curved_quad not in new_bounds
    assert curved_quad.make_flat_quad() in new_bounds
    assert len(bounds) == len(new_bounds)
    assert len(bounds & new_bounds) == 1
    assert len(frozenset(p1_flat.sides) & new_bounds) == 1


def test_extruding_hypnotoad_mesh() -> None:
    hypno_mesh = to_mesh(CONNECTED_DOUBLE_NULL)
    eq = hypno_mesh.equilibrium
    # Extrude only a very short distance to keep run-times quick
    mesh = generators.hypnotoad_mesh(hypno_mesh, (0.0, 0.001 * np.pi), 3, 21)
    actual_nodes = frozenset(
        itertools.chain.from_iterable(
            (q.shape(0.0).to_coord(), q.shape(1.0).to_coord())
            for q in itertools.chain.from_iterable(mesh)
        )
    )
    expected_nodes = frozenset(
        itertools.chain.from_iterable(
            SliceCoords(
                r.Rxy.corners, r.Zxy.corners, CoordinateSystem.CYLINDRICAL
            ).iter_points()
            for r in hypno_mesh.regions.values()
        )
    )
    assert actual_nodes == expected_nodes
    lines = frozenset(
        itertools.chain.from_iterable(
            (q.north, q.south) for q in itertools.chain.from_iterable(mesh)
        )
    )
    for line in lines:
        R, Z, _ = control_points(line, 8)
        # Ignore any values that leave the domain, as these won't be accurate
        in_domain = np.logical_and(
            np.logical_and(R <= eq.Rmax, R >= eq.Rmin),
            np.logical_and(Z <= eq.Zmax, Z >= eq.Zmin),
        )
        psis = np.asarray(eq.psi(R, Z))[in_domain]
        np.testing.assert_allclose(psis, psis[0], 1e-5, 1e-5)
    # Get a list all boundaries. Each entry in the list will be
    # another list containing the portions of that boundary in each
    # layer.
    bounds = list(zip(*(list(layer.boundaries()) for layer in mesh.layers())))
    for bound in bounds:
        sizes = {len(b) for b in bound}
        # Check that the boundary is the same size in each layer
        assert len(sizes) == 1
        # Check that the boundary is not empty
        assert all(s > 0 for s in sizes)


def test_extruding_hypnotoad_mesh_fill_core() -> None:
    hypno_mesh = to_mesh(CONNECTED_DOUBLE_NULL)
    eq = hypno_mesh.equilibrium
    # Extrude only a very short distance to keep run-times quick
    mesh = generators.hypnotoad_mesh(
        hypno_mesh, (0.0, 0.001 * np.pi / 3), 1, 11, mesh_to_core=True
    )
    tri_prisms = [p for p in mesh if len(p.sides) == 3]
    # Check triangles have been created at the centre of the mesh
    assert len(tri_prisms) > 0

    o_point = SliceCoord(eq.o_point.R, eq.o_point.Z, CoordinateSystem.CYLINDRICAL)

    def get_axis_edge(prism: Prism) -> Segment:
        curves = frozenset(q.north for q in prism.sides) | frozenset(
            q.south for q in prism.sides
        )
        assert len(curves) == 3
        axis_curve = [
            c
            for c in curves
            if (
                c.start
                if isinstance(c, FieldAlignedCurve)
                else c(0.0).to_coord().to_slice_coord()
            )
            == o_point
        ]
        assert len(axis_curve) == 1
        acurve = axis_curve[0]
        return acurve

    # Check all the triangles have one corner that is at the o-point
    axis_curves = frozenset(map(get_axis_edge, tri_prisms))
    assert len(axis_curves) == 1
    axis_curve = next(iter(axis_curves))
    assert axis_curve(0.5).to_coord().to_slice_coord() == o_point
    assert axis_curve(1.0).to_coord().to_slice_coord() == o_point


@pytest.mark.filterwarnings("ignore:Multiple vertex rings")
def test_extruding_hypnotoad_mesh_enforce_bounds() -> None:
    hypno_mesh = to_mesh(CONNECTED_DOUBLE_NULL)
    eq = hypno_mesh.equilibrium
    # Extrude only a very short distance to keep run-times quick
    mesh = generators.hypnotoad_mesh(
        hypno_mesh, (0.0, 0.001 * np.pi / 3), 1, 11, restrict_to_vessel=True
    )
    Rmin = min(p.R for p in eq.wall)
    Rmax = max(p.R for p in eq.wall)
    Zmin = min(p.Z for p in eq.wall)
    Zmax = max(p.Z for p in eq.wall)

    def in_domain(element: Prism) -> bool:
        corners = element.corners()
        return bool(
            np.all(corners.x1 <= Rmax)
            and np.all(corners.x1 >= Rmin)
            and np.all(corners.x2 <= Zmax)
            and np.all(corners.x2 >= Zmin)
        )

    assert all(map(in_domain, mesh))


def test_extruding_hypnotoad_mesh_to_wall() -> None:
    hypno_mesh = to_mesh(CONNECTED_DOUBLE_NULL)
    # Extrude only a very short distance to keep run-times quick
    mesh = generators.hypnotoad_mesh(
        hypno_mesh,
        (0.0, 0.001 * np.pi / 3),
        1,
        11,
        restrict_to_vessel=True,
        mesh_to_core=True,
        mesh_to_wall=True,
    )

    assert len(mesh.reference_layer.bounds) == 2
    assert len(mesh.reference_layer.bounds[0]) == 0

    bounds = mesh.reference_layer.bounds[1]
    wall = [
        SliceCoord(p.R, p.Z, CoordinateSystem.CYLINDRICAL)
        for p in hypno_mesh.equilibrium.wall
    ]

    def on_wall(q: Quad) -> bool:
        corners = q.corners()
        return any(
            all(point_on_surface(*segment, p) for p in corners.iter_points())
            for segment in itertools.pairwise(wall)
        )

    # Check the corners of all bounds fall on the tokamak walls
    assert all(map(on_wall, bounds))

    wall_remnants, remaining_quads = filter_quads_on_wall(
        itertools.pairwise(wall),
        frozenset(frozenset(q.shape([0.0, 1.0]).iter_points()) for q in bounds),
    )
    # Check the walls are entirely covered by the boundary quads
    assert len(wall_remnants) == 0
    # Check there are no boundary quads not covering the walls
    assert len(remaining_quads) == 0
    # FIXME: Writing CONNECTED_DOUBLE_NULL seems to be unbelievably
    # slow. I almost wonder if it's caught in an infinite loop?


def test_extruding_hypnotoad_mesh_to_wall_remesh() -> None:
    hypno_mesh = to_mesh(CONNECTED_DOUBLE_NULL)
    # Extrude only a very short distance to keep run-times quick
    mesh = generators.hypnotoad_mesh(
        hypno_mesh,
        (0.0, 0.0001 * np.pi / 3),
        1,
        11,
        restrict_to_vessel=True,
        mesh_to_core=True,
        mesh_to_wall=True,
        wall_resolution=1.0,
    )

    assert len(mesh.reference_layer.bounds) == 2
    assert len(mesh.reference_layer.bounds[0]) == 0

    def distance(start: SliceCoord, end: SliceCoord) -> float:
        return float(np.sqrt((start.x1 - end.x1) ** 2 + (start.x2 - end.x2) ** 2))

    wall_segment_lengths = [
        distance(*quad.shape([0.0, 1.0]).iter_points())
        for quad in mesh.reference_layer.bounds[1]
    ]
    initial_wall = hypno_mesh.equilibrium.wall
    initial_wall_length = min(
        np.sqrt((p1.R - p0.R) ** 2 + (p1.Z - p0.Z) ** 2)
        for p0, p1 in itertools.pairwise(initial_wall)
    )
    assert max(wall_segment_lengths) / min(wall_segment_lengths) <= 3
    assert all(length < initial_wall_length for length in wall_segment_lengths)


def point_on_surface(
    start: SliceCoord, end: SliceCoord, point: Coord | SliceCoord
) -> bool:
    if abs(start.x1 - end.x1) > abs(start.x2 - end.x2):
        x_start, y_start = start
        x_end, y_end = end
        x = point.x1
        y = point.x2
    else:
        y_start, x_start = start
        y_end, x_end = end
        y = point.x1
        x = point.x2
    xmin, xmax = sorted([x_start, x_end])
    if x < xmin - TOL or x > xmax + TOL:
        return False
    y_expected = (y_end - y_start) / (x_end - x_start) * (x - x_start) + y_start
    return abs(y - y_expected) <= TOL


TOL = 1e-8
QuadPoints = frozenset[frozenset[SliceCoord]]
WallSegment = tuple[SliceCoord, SliceCoord]


def filter_quads_on_wall_segment(
    start: SliceCoord, end: SliceCoord, quads: QuadPoints
) -> tuple[Optional[tuple[SliceCoord, SliceCoord]], QuadPoints]:
    possible_start_quads = [
        q
        for q in quads
        if start in q and point_on_surface(start, end, next(iter(q - {start})))
    ]
    # Corrupted mesh
    if len(possible_start_quads) > 1:
        raise ValueError("Overlapping quads")
    # Can't find any more quads on this wall, so return the remainder
    if len(possible_start_quads) == 0:
        return (start, end), quads
    new_start = next(iter(possible_start_quads[0] - {start}))
    new_quads = quads - {possible_start_quads[0]}
    # Have found sufficient quads to cover this wall
    if new_start == end:
        return None, new_quads
    return filter_quads_on_wall_segment(new_start, end, new_quads)


def filter_quads_on_wall(
    wall: Iterator[WallSegment], quads: QuadPoints
) -> tuple[list[WallSegment], QuadPoints]:
    def process_segment(
        result: tuple[list[WallSegment], QuadPoints], segment: WallSegment
    ) -> tuple[list[WallSegment], QuadPoints]:
        remnants, quads = result
        new_remnant, remaining_quads = filter_quads_on_wall_segment(*segment, quads)
        if new_remnant is not None:
            remnants.append(new_remnant)
        return remnants, remaining_quads

    leftovers: list[WallSegment] = []
    return reduce(process_segment, wall, (leftovers, quads))
