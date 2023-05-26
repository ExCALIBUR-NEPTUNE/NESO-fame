import numpy as np

from neso_fame import generators
from neso_fame.fields import straight_field
from neso_fame.mesh import CoordinateSystem, SliceCoords


# Test for simple grid
def test_simple_grid() -> None:
    starts = SliceCoords(
        np.linspace(
            -1.0,
            1.0,
            5,
        ),
        np.empty(5),
        CoordinateSystem.Cartesian,
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
            # Bounding curves in quad are not ordered, so don't know
            # whether higher or lower will come first
            if corners.x1[2] > corners.x1[0]:
                x1_0 = starts.x1[i]
                x1_2 = starts.x1[i + 1]
            else:
                x1_0 = starts.x1[i + 1]
                x1_2 = starts.x1[i]
            np.testing.assert_allclose(corners.x1[[0, 1]], x1_0)
            np.testing.assert_allclose(corners.x1[[2, 3]], x1_2)
            assert np.all(corners.x2 == 0.0)
            np.testing.assert_allclose(corners.x3[[0, 2]], x3_start)
            np.testing.assert_allclose(corners.x3[[1, 3]], x3_end)
        x3_start = x3_end
    # Check connectivity of quads is correct
    ordered_quads = list(mesh.reference_layer.reference_elements.keys())
    for i, (quad, connections) in enumerate(
        mesh.reference_layer.reference_elements.items()
    ):
        assert not any(connections.values())
        if 0 < i < len(starts) - 2:
            assert len(connections) == 2
        else:
            assert len(connections) == 1
        if i != 0:
            assert any(c == ordered_quads[i - 1] for c in connections)
        if i != len(starts) - 2:
            assert any(c == ordered_quads[i + 1] for c in connections)


# Test for angled grid
def test_angled_grid() -> None:
    angle = np.pi / 18
    starts = SliceCoords(
        np.linspace(
            1.0,
            3.0,
            3,
        ),
        np.empty(3),
        CoordinateSystem.Cartesian,
    )
    field = straight_field(angle)
    x3 = (-2.0, 1.0)
    n = 5
    mesh = generators.field_aligned_2d(starts, field, x3, n)
    assert len(mesh) == 10
    x3_start = x3[0]
    dx3 = (x3[1] - x3[0]) / n
    x1_offsets = np.array([-np.tan(angle) * dx3 / 2, 0.0, np.tan(angle) * dx3 / 2])
    # Check control points of quad curves are in right location
    for layer in mesh.layers():
        x3_end = x3_start + dx3
        x3_positions = np.array([x3_start, x3_start + dx3 / 2, x3_end])
        for i, quad in enumerate(layer):
            # Bounding curves in quad are not ordered, so don't know
            # whether higher or lower will come first
            if quad.south(0.0).x1 < quad.north(0.0).x1:
                x1_south_mid = starts.x1[i]
                x1_north_mid = starts.x1[i + 1]
            else:
                x1_south_mid = starts.x1[i + 1]
                x1_north_mid = starts.x1[i]
            south_points = quad.south.control_points(2)
            np.testing.assert_allclose(south_points.x1, x1_south_mid + x1_offsets)
            assert np.all(south_points.x2 == 0.0)
            np.testing.assert_allclose(south_points.x3, x3_positions)
            north_points = quad.north.control_points(2)
            np.testing.assert_allclose(north_points.x1, x1_north_mid + x1_offsets)
            assert np.all(north_points.x2 == 0.0)
            np.testing.assert_allclose(north_points.x3, x3_positions)
        x3_start = x3_end