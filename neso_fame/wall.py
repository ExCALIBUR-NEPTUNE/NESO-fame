"""Functions for establishing which points are within the tokamak vessel."""

from __future__ import annotations

import itertools
import operator
from collections.abc import Sequence
from enum import Enum
from functools import reduce
from typing import Callable, NamedTuple, Optional

import numpy as np
from hypnotoad import Point2D  # type: ignore

from neso_fame.mesh import SliceCoord, SliceCoords


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


Connections = dict[SliceCoord, frozenset[SliceCoord]]


def find_external_points(
    outermost: frozenset[SliceCoord],
    connections: Connections,
    wall: Sequence[WallSegment],
    in_tokamak_test: Callable[
        [SliceCoord, Sequence[WallSegment]], bool
    ] = point_in_tokamak,
) -> tuple[frozenset[SliceCoord], frozenset[SliceCoord]]:
    """Find the points in a mesh outside the wall of a tokamak.

    Parameters
    ----------
    outermost
        The set of points making up the external "skin" of the
        mesh, which might fall outside the wall.
    connections
        A mapping between all points in the mesh and the points
        they share an edge with.
    wall
        The line segments making up the wall of the tokamak
    in_tokamak_test
        Routine to determine if a node falls inside the
        tokamak. Custom routines can be passed to, e.g., check
        at multiple points along a field line.

    Returns
    -------
    A tuple of sets of points. The first element is all points falling
    outside the tokamak. The second is the outermost layer of points
    still inside the tokamak.

    Group
    -----
    wall

    """
    # outpoints, skinpoints, candidates, new_candidates
    return _find_external_points(
        outermost, frozenset(), frozenset(), connections, wall, in_tokamak_test
    )


def _find_external_points(
    candidates: frozenset[SliceCoord],
    outpoints: frozenset[SliceCoord],
    skinpoints: frozenset[SliceCoord],
    connections: Connections,
    wall: Sequence[WallSegment],
    in_tokamak_test: Callable[[SliceCoord, Sequence[WallSegment]], bool],
) -> tuple[frozenset[SliceCoord], frozenset[SliceCoord]]:
    if len(candidates) == 0:
        # If nothing to check, return previous results
        return outpoints, skinpoints
    # Check all candidates
    results = {point: in_tokamak_test(point, wall) for point in candidates}
    # Separate out candidates found to be outside the wall
    new_outpoints = frozenset(p for p, r in results.items() if not r)
    # Assemble a set of new candidates, consiting of neighbours of all
    # the points we found were outside the wall (that haven't already
    # been classified)
    neighbours: frozenset[SliceCoord] = reduce(
        operator.or_, (connections[p] for p in new_outpoints), frozenset()
    )
    # Repeat the process on the new set of candidates
    return _find_external_points(
        neighbours - candidates - outpoints - skinpoints,
        outpoints | new_outpoints,
        skinpoints | frozenset(p for p, r in results.items() if r),
        connections,
        wall,
        in_tokamak_test,
    )


def get_rectangular_mesh_connections(points: SliceCoords) -> Connections:
    """Return connectivity information for a logically-rectangular set of points."""
    shape = np.broadcast(points.x1, points.x2).shape

    def get_neighbours(i: int, j: int) -> frozenset[SliceCoord]:
        i_min = i - 1 if i > 0 else 0
        i_max = i + 2
        j_min = j - 1 if j > 0 else 0
        j_max = j + 2
        return (
            points.get_set((slice(i_min, i_max), j))
            | points.get_set((i, slice(j_min, j_max)))
        ) - points.get_set((i, j))

    return {
        points[i, j]: get_neighbours(i, j)
        for i in range(shape[0])
        for j in range(shape[1])
    }
