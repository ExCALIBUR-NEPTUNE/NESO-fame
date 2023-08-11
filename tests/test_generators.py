import numpy as np

from neso_fame import generators
from neso_fame.fields import straight_field
from neso_fame.mesh import control_points, CoordinateSystem, SliceCoords


# Test for simple grid
def test_simple_grid() -> None:
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
            map(
                lambda x: tuple(x.iter_points()),
                filter(
                    lambda x: np.all(x.x1 == starts.x1[0]),
                    map(lambda x: control_points(x.north, 1), layer),
                ),
            )
        )
        expected_south = frozenset(
            map(
                lambda x: tuple(x.iter_points()),
                filter(
                    lambda x: np.all(x.x1 == starts.x1[-1]),
                    map(lambda x: control_points(x.south, 1), layer),
                ),
            )
        )
        actual_north = frozenset(
            map(lambda x: tuple(control_points(x, 1).iter_points()), bounds[0])
        )
        actual_south = frozenset(
            map(lambda x: tuple(control_points(x, 1).iter_points()), bounds[1])
        )
        assert actual_north == expected_north
        assert actual_south == expected_south


def test_angled_grid_conforming_bounds() -> None:
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
            map(
                lambda x: tuple(x.iter_points()),
                filter(
                    lambda x: np.all(x.x1 == starts.x1[0]),
                    map(lambda x: control_points(x.north, 1), layer),
                ),
            )
        )
        expected_south = frozenset(
            map(
                lambda x: tuple(x.iter_points()),
                filter(
                    lambda x: np.all(x.x1 == starts.x1[-1]),
                    map(lambda x: control_points(x.south, 1), layer),
                ),
            )
        )
        actual_north = frozenset(
            map(lambda x: tuple(control_points(x, 1).iter_points()), bounds[0])
        )
        actual_south = frozenset(
            map(lambda x: tuple(control_points(x, 1).iter_points()), bounds[1])
        )
        assert actual_north == expected_north
        assert actual_south == expected_south


# Test for angled grid
def test_angled_grid_jagged_bounds() -> None:
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
            map(
                lambda x: tuple(x.iter_points()),
                filter(
                    lambda x: np.allclose(x.x1, starts.x1[0] + x1_offsets[[0, 2]]),
                    map(lambda x: control_points(x.north, 1), layer),
                ),
            )
        )
        expected_south = frozenset(
            map(
                lambda x: tuple(x.iter_points()),
                filter(
                    lambda x: np.allclose(x.x1, starts.x1[-1] + x1_offsets[[0, 2]]),
                    map(lambda x: control_points(x.south, 1), layer),
                ),
            )
        )
        actual_north = frozenset(
            map(lambda x: tuple(control_points(x, 1).iter_points()), bounds[0])
        )
        actual_south = frozenset(
            map(lambda x: tuple(control_points(x, 1).iter_points()), bounds[1])
        )
        assert actual_north == expected_north
        assert actual_south == expected_south


# Test for simple grid
def test_subdivided_grid() -> None:
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
            map(
                lambda x: tuple(x.iter_points()),
                filter(
                    lambda x: np.all(x.x1 == starts.x1[0]),
                    map(lambda x: control_points(x.north, 1), layer),
                ),
            )
        )
        expected_south = frozenset(
            map(
                lambda x: tuple(x.iter_points()),
                filter(
                    lambda x: np.all(x.x1 == starts.x1[-1]),
                    map(lambda x: control_points(x.south, 1), layer),
                ),
            )
        )
        actual_north = frozenset(
            map(lambda x: tuple(control_points(x, 1).iter_points()), bounds[0])
        )
        actual_south = frozenset(
            map(lambda x: tuple(control_points(x, 1).iter_points()), bounds[1])
        )
        assert actual_north == expected_north
        assert actual_south == expected_south
