import itertools
import operator
from unittest.mock import MagicMock, call

import numpy as np
from hypnotoad import Point2D  # type: ignore
from hypothesis import given
from hypothesis.strategies import (
    booleans,
    builds,
    floats,
    integers,
    shared,
)

from neso_fame.coordinates import (
    CoordinateSystem,
    FrozenCoordSet,
    SliceCoord,
    SliceCoords,
)
from neso_fame.mesh import (
    straight_line_across_field,
)
from neso_fame.wall import (
    WallSegment,
    adjust_wall_resolution,
    find_external_points,
    get_all_rectangular_mesh_connections,
    get_immediate_rectangular_mesh_connections,
    periodic_pairwise,
    point_in_tokamak,
    wall_points_to_segments,
)

# For testing purposes, make regular polygons by choosing some number
# of points around a circle. It is then straightforward to generate
# points that are definitely inside or definitely outside the polygon.


def regular_polygon(radius: float, start_angle: float, n: int) -> list[Point2D]:
    angles = np.linspace(start_angle, start_angle + 2 * np.pi, n, False)
    x1 = radius * np.cos(angles)
    x2 = radius * np.sin(angles)
    return [Point2D(R, Z) for R, Z in np.nditer((x1, x2))]


def make_point(r: float, theta: float) -> SliceCoord:
    return SliceCoord(
        r * np.cos(theta), r * np.sin(theta), CoordinateSystem.CYLINDRICAL
    )


SAFETY = 1e-3
radii = shared(floats(0.1, 10.0), key=441)
num_points = shared(integers(3, 30), key=442)
polygons = builds(regular_polygon, radii, floats(0.0, 2 * np.pi), num_points)
angles = floats(0.0, 2 * np.pi)
_max_rad_inside = builds(
    lambda r, n: r * (1 - np.cos(0.5 * (n - 2) * np.pi / n) - SAFETY), radii, num_points
)
rad_of_points_inside = builds(operator.mul, floats(0.0, 1.0), _max_rad_inside)
rad_of_points_outside = builds(operator.mul, floats(1.01, 100.0), radii)
points_inside = builds(make_point, rad_of_points_inside, angles)
points_outside = builds(make_point, rad_of_points_outside, angles)
tolerances = radii.flatmap(lambda r: floats(1e-10 * r, 0.001 * r))
target_sizes = radii.flatmap(lambda x: floats(0.1 * x, 2 * x))
min_sizes = floats(0.02, 0.4)


@given(
    floats(-10.0, 10.0),
    floats(-10.0, 10),
    floats(-np.pi, np.pi),
    floats(0.1, 10.0),
    floats(-5.0, 5.0),
    floats(0.0, 1.0),
    floats(),
)
def test_minimum_distance_perpendicular(
    R0: float,
    Z0: float,
    angle: float,
    segment_length: float,
    perpendicular_offset: float,
    parallel_offset: float,
    tol: float,
) -> None:
    # Check the distance for points at some perpendicular offset from
    # a point along the segment.
    R1 = R0 + segment_length * np.cos(angle)
    Z1 = Z0 + segment_length * np.sin(angle)
    vec = [R1 - R0, Z1 - Z0]
    segment = WallSegment(R0, Z0, R1, Z1, tol)
    coord = SliceCoord(
        R0 + perpendicular_offset * vec[1] + parallel_offset * vec[0],
        Z0 - perpendicular_offset * vec[0] + parallel_offset * vec[1],
        CoordinateSystem.CYLINDRICAL,
    )
    np.testing.assert_allclose(
        segment.min_distance_squared(coord),
        (perpendicular_offset * segment_length) ** 2,
        1e-8,
        1e-8,
    )


@given(
    floats(-10.0, 10.0),
    floats(-10.0, 10),
    floats(-np.pi, np.pi),
    floats(0.1, 10.0),
    floats(0.0, np.pi),
    floats(0.0, 5.0),
    floats(),
    booleans(),
)
def test_minimum_distance_beyond_segment(
    R0: float,
    Z0: float,
    angle: float,
    segment_length: float,
    coord_angle: float,
    coord_distance: float,
    tol: float,
    switch: bool,
) -> None:
    # Check the distance for points for which the nearest point on the segment is one of the ends.
    R1 = R0 + segment_length * np.cos(angle)
    Z1 = Z0 + segment_length * np.sin(angle)
    if switch:
        segment = WallSegment(R1, Z1, R0, Z0, tol)
    else:
        segment = WallSegment(R0, Z0, R1, Z1, tol)
    ang = angle + np.pi / 2 + coord_angle
    coord = SliceCoord(
        R0 + coord_distance * np.cos(ang),
        Z0 + coord_distance * np.sin(ang),
        CoordinateSystem.CYLINDRICAL,
    )
    np.testing.assert_allclose(
        segment.min_distance_squared(coord), coord_distance**2, 1e-8, 1e-8
    )


@given(polygons, radii, min_sizes, floats(0.0, np.pi / 3))
def test_adjust_wall_resolution_distances(
    wall: list[Point2D], new_resolution: float, min_factor: float, angle: float
) -> None:
    new_wall = adjust_wall_resolution(wall, new_resolution, min_factor, angle)
    min_dist_sq = (min_factor * new_resolution) ** 2
    max_dist_sq = (1.5 * new_resolution) ** 2
    for start, end in periodic_pairwise(new_wall):
        dist_sq = (end.R - start.R) ** 2 + (end.Z - start.Z) ** 2
        assert dist_sq <= max_dist_sq
        assert dist_sq >= min_dist_sq


@given(polygons, radii, integers(5, 1000).map(lambda m: 2 * np.pi / (m + 0.5)))
def test_adjust_wall_resolution_angles(
    wall: list[Point2D], new_resolution: float, angle: float
) -> None:
    # Don't eliminate small edges here, as it makes it really
    # confusing when we can expect a sharp angle to be preserved.
    new_wall = adjust_wall_resolution(wall, new_resolution, 0, angle)

    def sharp_corner(start: Point2D, mid: Point2D, end: Point2D) -> bool:
        line1 = [mid.R - start.R, mid.Z - start.Z]
        line2 = [end.R - mid.R, end.Z - mid.Z]
        mag1 = np.sqrt(np.dot(line1, line1))
        mag2 = np.sqrt(np.dot(line2, line2))
        ang = np.arccos(np.clip(np.dot(line1, line2) / mag1 / mag2, -1.0, 1.0))
        return bool(ang >= angle)

    corners = [
        tuple(mid)
        for (start, mid), (_, end) in periodic_pairwise(periodic_pairwise(new_wall))
        if sharp_corner(start, mid, end)
    ]
    comparable_wall = [tuple(p) for p in wall]
    # This test doesn't work when we are splitting a "smooth" portion
    # into too few pieces, because can end up being sharper corners.
    n = len(wall)
    m = len(new_wall)
    if 2 * np.pi / n >= angle:
        # Case where angles in the original wall are preserved
        assert all((p in comparable_wall) for p in corners)
        assert len(corners) == n
        assert len(new_wall) % len(wall) == 0
    elif 2 * np.pi / m >= angle:
        # Case where original wall is smoothed out and interpolated into a
        # polygon with sharp angles
        # assert all((p not in comparable_wall) for p in corners)
        assert (
            len(corners) > 0
        )  # Angles won't all be equal due to inaccuracy in interpolation
    else:
        # Case where original wall is smoothed and interpolated into a
        # polygon with shallow angles
        assert (
            len(corners) < m
        )  # Angles won't all be equal due to inaccuracy in interpolation


def test_adjust_wall_resolution_register() -> None:
    register_func = MagicMock()
    wall = [
        Point2D(0.5, -0.5),
        Point2D(0.5, 0.5),
        Point2D(-0.5, 0.5),
        Point2D(-0.5, -0.5),
    ]
    _ = adjust_wall_resolution(wall, 1.0, register_segment=register_func)
    expected = [
        call(
            straight_line_across_field(
                SliceCoord(p1.R, p1.Z, CoordinateSystem.CYLINDRICAL),
                SliceCoord(p2.R, p2.Z, CoordinateSystem.CYLINDRICAL),
            )
        )
        for p1, p2 in periodic_pairwise(wall)
    ]
    register_func.assert_has_calls(expected, any_order=True)


def test_adjust_wall_resolution_register_higher_order() -> None:
    register_func = MagicMock()
    wall = [
        Point2D(0.5, -0.5),
        Point2D(0.51, 0.0),
        Point2D(0.5, 0.5),
        Point2D(0.0, 0.51),
        Point2D(-0.5, 0.5),
        Point2D(-0.51, 0.0),
        Point2D(-0.5, -0.5),
        Point2D(0.0, -0.51),
    ]
    _ = adjust_wall_resolution(wall, 1.0, register_segment=register_func)
    assert register_func.call_count == 4


@given(polygons, points_inside)
def test_point_inside_tokamak_wall(wall: list[Point2D], point: SliceCoord) -> None:
    assert point_in_tokamak(point, wall_points_to_segments(wall))


@given(polygons, points_outside)
def test_point_outside_tokamak_wall(wall: list[Point2D], point: SliceCoord) -> None:
    assert not point_in_tokamak(point, wall_points_to_segments(wall))


def displace_inwards(point: Point2D, distance: float) -> SliceCoord:
    rad = np.sqrt(point.R * point.R + point.Z * point.Z)
    # Apply both relative and absolute offsets
    a = 1 - distance / rad
    return SliceCoord(point.R * a, point.Z * a, CoordinateSystem.CYLINDRICAL)


@given(
    polygons,
    num_points.flatmap(lambda n: integers(0, n - 1)),
    tolerances,
    floats(0.0, 0.5),
)
def test_point_near_wall(
    wall: list[Point2D], vertex: int, tol: float, offset_factor: float
) -> None:
    point = displace_inwards(wall[vertex], tol * offset_factor)
    assert not point_in_tokamak(point, wall_points_to_segments(wall, tol))


@given(
    polygons,
    num_points.flatmap(lambda n: integers(0, n - 1)),
    tolerances,
    floats(
        5.0,
        100.0,
    ),
)
def test_point_just_inside(
    wall: list[Point2D], vertex: int, tol: float, offset_factor: float
) -> None:
    point = displace_inwards(wall[vertex], tol * offset_factor)
    assert point_in_tokamak(point, wall_points_to_segments(wall, tol))


grid_1d = builds(np.linspace, floats(-10.0, -0.05), floats(0.05, 10.0), integers(2, 10))
mesh_points = builds(np.meshgrid, grid_1d, grid_1d).map(
    lambda rz: SliceCoords(rz[0], rz[1], CoordinateSystem.CYLINDRICAL)
)


@given(mesh_points)
def test_all_mesh_connections_corners(points: SliceCoords) -> None:
    connections = get_all_rectangular_mesh_connections(points)
    assert len(connections[points[0, 0]]) == 3
    assert len(connections[points[0, -1]]) == 3
    assert len(connections[points[-1, 0]]) == 3
    assert len(connections[points[-1, -1]]) == 3


@given(mesh_points)
def test_all_mesh_connections_edges(points: SliceCoords) -> None:
    connections = get_all_rectangular_mesh_connections(points)
    shape = points.x1.shape
    assert all(len(connections[points[0, j]]) == 5 for j in range(1, shape[1] - 1))
    assert all(len(connections[points[-1, j]]) == 5 for j in range(1, shape[1] - 1))
    assert all(len(connections[points[i, 0]]) == 5 for i in range(1, shape[0] - 1))
    assert all(len(connections[points[i, -1]]) == 5 for i in range(1, shape[0] - 1))


@given(mesh_points)
def test_all_mesh_connections_interior(points: SliceCoords) -> None:
    connections = get_all_rectangular_mesh_connections(points)
    shape = points.x1.shape
    assert all(
        len(connections[points[i, j]]) == 8
        for i in range(1, shape[0] - 1)
        for j in range(1, shape[1] - 1)
    )


@given(mesh_points)
def test_immediate_mesh_connections_corners(points: SliceCoords) -> None:
    connections = get_immediate_rectangular_mesh_connections(points)
    assert len(connections[points[0, 0]]) == 2
    assert len(connections[points[0, -1]]) == 2
    assert len(connections[points[-1, 0]]) == 2
    assert len(connections[points[-1, -1]]) == 2


@given(mesh_points)
def test_immediate_mesh_connections_edges(points: SliceCoords) -> None:
    connections = get_immediate_rectangular_mesh_connections(points)
    shape = points.x1.shape
    assert all(len(connections[points[0, j]]) == 3 for j in range(1, shape[1] - 1))
    assert all(len(connections[points[-1, j]]) == 3 for j in range(1, shape[1] - 1))
    assert all(len(connections[points[i, 0]]) == 3 for i in range(1, shape[0] - 1))
    assert all(len(connections[points[i, -1]]) == 3 for i in range(1, shape[0] - 1))


@given(mesh_points)
def test_immediate_mesh_connections_interior(points: SliceCoords) -> None:
    connections = get_immediate_rectangular_mesh_connections(points)
    shape = points.x1.shape
    assert all(
        len(connections[points[i, j]]) == 4
        for i in range(1, shape[0] - 1)
        for j in range(1, shape[1] - 1)
    )


def get_bounds(points: SliceCoords) -> FrozenCoordSet[SliceCoord]:
    shape = points.x1.shape
    return FrozenCoordSet(
        itertools.chain(
            points.get_set((0, slice(0, shape[1]))),
            points.get_set((shape[0] - 1, slice(0, shape[1]))),
            points.get_set((slice(1, shape[0] - 1), 0)),
            points.get_set((slice(1, shape[0] - 1), shape[1] - 1)),
        )
    )


@given(mesh_points, polygons)
def test_find_external_points(points: SliceCoords, wall_points: list[Point2D]) -> None:
    wall = wall_points_to_segments(wall_points)
    connections = get_all_rectangular_mesh_connections(points)
    outside, skin = find_external_points(get_bounds(points), connections, wall)
    points_set = FrozenCoordSet(points.iter_points())
    inside = points_set - outside
    assert outside <= points_set
    assert skin <= inside
    assert all(not point_in_tokamak(p, wall) for p in outside)
    assert all(point_in_tokamak(p, wall) for p in inside)
    if len(inside) > 1:
        assert all(connections[p] & inside for p in skin)
    assert all(not (connections[p] & outside) for p in inside - skin)
