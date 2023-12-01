import operator

import numpy as np
from hypnotoad import Point2D  # type: ignore
from hypothesis import given
from hypothesis.strategies import (
    builds,
    floats,
    integers,
    shared,
)

from neso_fame.mesh import CoordinateSystem, SliceCoord
from neso_fame.wall import point_in_tokamak, wall_points_to_segments

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
