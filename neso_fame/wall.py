"""Functions for establishing which points are within the tokamak vessel."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from enum import Enum
from typing import NamedTuple, Optional

import numpy as np
from hypnotoad import Point2D

from neso_fame.mesh import SliceCoord


class Crossing(Enum):
    """Indicates whether a point crosses a line segment or is near it."""

    NOT_CROSSED = 0
    CROSSED = 1
    NEAR = 2


class WallSegment(NamedTuple):
    """Description of a line segment making up part of the tokamak wall.

    Group
    -----
    wall

    """

    inverse_slope: float
    R_offset: float
    min_Z: float
    max_Z: float
    tol: float

    def crossed_by_ray_from_point(self, point: SliceCoord) -> Crossing:
        """Test whether a ray extending rightward from `point` crosses the segment.

        Points falling within a distance of the segment that is less
        than the tolerance are indicated as being "near" it, rather
        than crossing.

        """
        Z = point.x2
        if self.min_Z <= Z < self.max_Z:
            # Tolerance is applied normal to the segment. Need to project
            # it into the x1 direction.
            tolerance = self.tol * np.sqrt(self.inverse_slope**2 + 1)
            R_intersect = self.inverse_slope * Z - self.R_offset
            print(R_intersect, point.x1, tolerance)
            if np.abs(R_intersect - point.x1) < tolerance:
                return Crossing.NEAR
            elif R_intersect > point.x1:
                return Crossing.CROSSED
        return Crossing.NOT_CROSSED


def wall_points_to_segments(
    points: Sequence[Point2D], tol: float = 1e-8
) -> list[WallSegment]:
    """Compute the edges connecting a series of vertices.

    This is useful for working with the list of points defining the
    wall of a tokamak. The wall is assumed to be closed and an edge
    will be inserted between the last and the first point.

    Group
    -----
    wall

    """
    inv_tol = 1.0 / tol

    def _points_to_segment(points: tuple[Point2D, Point2D]) -> Optional[WallSegment]:
        p1, p2 = points
        rise = p2.Z - p1.Z
        # Want to avoid divide-by-zero errors
        inv_slope = (p2.R - p1.R) / rise if abs(rise) > tol else inv_tol
        return WallSegment(
            inv_slope, inv_slope * p1.Z - p1.R, min(p2.Z, p1.Z), max(p2.Z, p1.Z), tol
        )

    return [
        seg
        for seg in map(
            _points_to_segment,
            itertools.pairwise(itertools.chain.from_iterable((points, [points[0]]))),
        )
        if seg is not None
    ]


def point_in_tokamak(point: SliceCoord, wall: Sequence[WallSegment]) -> bool:
    """Check if the point falls inside the wall of the tokamak.

    Points that are near the edge of the Tokamak, within a tolerance,
    are classified as falling outside. This avoids issue with
    numerical error.

    Group
    -----
    wall

    """
    crossings = 0
    for status in (seg.crossed_by_ray_from_point(point) for seg in wall):
        if status == Crossing.NEAR:
            return False
        elif status == Crossing.CROSSED:
            crossings += 1
    return crossings % 2 == 1
