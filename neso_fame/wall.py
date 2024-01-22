"""Functions for establishing which points are within the tokamak vessel."""

from __future__ import annotations

import itertools
import operator
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from functools import cached_property, reduce
from typing import Callable, Iterable, Iterator, Optional, TypeVar

import numpy as np
import numpy.typing as npt
from hypnotoad import Point2D  # type: ignore
from scipy.interpolate import interp1d

from neso_fame.mesh import (
    AcrossFieldCurve,
    CoordinateSystem,
    Quad,
    SliceCoord,
    SliceCoords,
    StraightLineAcrossField,
)


class Crossing(Enum):
    """Indicates whether a point crosses a line segment or is near it."""

    NOT_CROSSED = 0
    CROSSED = 1
    NEAR = 2


@dataclass(frozen=True)
class WallSegment:
    """Description of a line segment making up part of the tokamak wall.

    Group
    -----
    wall

    """

    R1: float
    Z1: float
    R2: float
    Z2: float
    tol: float

    def crossed_by_ray_from_point(self, point: SliceCoord) -> Crossing:
        """Test whether a ray extending rightward from `point` crosses the segment.

        Points falling within a distance of the segment that is less
        than the tolerance are indicated as being "near" it, rather
        than crossing.

        """
        Z = point.x2
        if self._min_Z <= Z < self._max_Z:
            # Tolerance is applied normal to the segment. Need to project
            # it into the x1 direction.
            tolerance = self.tol * np.sqrt(self._inverse_slope**2 + 1)
            R_intersect = self._inverse_slope * Z - self._R_offset
            if np.abs(R_intersect - point.x1) < tolerance:
                return Crossing.NEAR
            elif R_intersect > point.x1:
                return Crossing.CROSSED
        return Crossing.NOT_CROSSED

    @cached_property
    def _inverse_slope(self) -> float:
        rise = self.Z2 - self.Z1
        # Want to avoid divide-by-zero errors
        return (self.R2 - self.R1) / rise if abs(rise) > self.tol else 1.0 / self.tol

    @cached_property
    def _R_offset(self) -> float:
        return self._inverse_slope * self.Z1 - self.R1

    @cached_property
    def _min_Z(self) -> float:
        return min(self.Z1, self.Z2)

    @cached_property
    def _max_Z(self) -> float:
        return max(self.Z1, self.Z2)

    @cached_property
    def _length_squared(self) -> float:
        dR = self.R2 - self.R1
        dZ = self.Z2 - self.Z1
        return dR * dR + dZ * dZ

    @cached_property
    def _m_R(self) -> float:
        return (self.R2 - self.R1) / self._length_squared

    @cached_property
    def _m_Z(self) -> float:
        return (self.Z2 - self.Z1) / self._length_squared

    def min_distance_squared(self, coord: SliceCoord) -> float:
        """Find the square of the minimum distance to a point on the line."""
        t = self._m_R * (coord.x1 - self.R1) + self._m_Z * (coord.x2 - self.Z1)
        if t <= 0:
            R = self.R1
            Z = self.Z1
        elif t >= 1:
            R = self.R2
            Z = self.Z2
        else:
            R = (1 - t) * self.R1 + t * self.R2
            Z = (1 - t) * self.Z1 + t * self.Z2
        dR = R - coord.x1
        dZ = Z - coord.x2
        return dR * dR + dZ * dZ


def _points_to_vector(points: tuple[Point2D, Point2D]) -> npt.NDArray:
    return np.array([points[1].R - points[0].R, points[1].Z - points[0].Z])


def _angle_between(v1: npt.NDArray, v2: npt.NDArray) -> float:
    return float(
        np.arccos(
            np.clip(
                np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0
            )
        )
    )


def _split_at_discontinuities(
    points: Sequence[Point2D], angle_threshold: float = np.pi / 8
) -> Iterator[Iterator[Point2D]]:
    angles = itertools.starmap(
        _angle_between,
        periodic_pairwise(map(_points_to_vector, periodic_pairwise(points))),
    )
    points_iter = iter(points[1:])
    points_with_angles = zip(itertools.chain(points_iter), angles)
    boundary_point: Point2D = points[0]
    finished = False

    def sub_iter(
        points_and_angles: Iterator[tuple[Point2D, float]],
    ) -> Iterator[Point2D]:
        nonlocal boundary_point, finished
        if boundary_point is not None:
            yield boundary_point
        a = 0.0
        while a < angle_threshold:
            try:
                p, a = next(points_and_angles)
            except StopIteration:
                finished = True
                return
            yield p
        boundary_point = p

    if finished:
        yield iter(points)
        yield points[0]
        return
    tail = list(sub_iter(points_with_angles))
    restarted_points_iter = zip(
        itertools.chain(points_iter, tail),
        itertools.chain(angles, itertools.repeat(0.0)),
    )
    while not finished:
        yield sub_iter(restarted_points_iter)


def _segment_length(points: Sequence[Point2D]) -> float:
    return sum(
        float(np.linalg.norm(_points_to_vector(ps)))
        for ps in itertools.pairwise(points)
    )


def _combine_small_portions(
    portions: Iterator[tuple[list[Point2D], float]],
) -> Iterator[tuple[list[Point2D], float]]:
    start_points, first_dist = next(portions)
    yield reduce(
        lambda x, y: (x[0] + y[0][1:], x[1] + y[1]),
        portions,
        (start_points, first_dist),
    )


Interpolator = Callable[[npt.ArrayLike], npt.NDArray]


def _make_element_shape(
    R_interp: Interpolator, Z_interp: Interpolator, start: float, end: float
) -> AcrossFieldCurve:
    def shape(s: npt.ArrayLike) -> SliceCoords:
        s_prime = start + (end - start) * np.asarray(s)
        return SliceCoords(
            R_interp(s_prime), Z_interp(s_prime), CoordinateSystem.CYLINDRICAL
        )

    return shape


def _interpolate_wall(
    points: Sequence[Point2D],
    n: int,
    register_segment: Optional[Callable[[AcrossFieldCurve], Quad]] = None,
) -> list[Point2D]:
    coords = np.array([tuple(p) for p in points])
    distances = np.cumsum(
        np.concatenate(([0.0], np.linalg.norm(coords[1:, :] - coords[:-1, :], axis=1)))
    )
    normed_distances = distances / distances[-1]
    m = len(points)
    kind = "linear" if m == 2 else "quadratic" if m == 3 else "cubic"
    R_interp = interp1d(normed_distances, coords[:, 0], kind)
    Z_interp = interp1d(normed_distances, coords[:, 1], kind)
    s = np.linspace(0.0, 1.0, n + 1)
    Rs = R_interp(s)
    Zs = Z_interp(s)
    if kind != "linear":
        shapes = (
            _make_element_shape(R_interp, Z_interp, segment[0], segment[1])
            for segment in itertools.pairwise(s)
        )
    else:
        shapes = (
            StraightLineAcrossField(
                SliceCoord(float(R0), float(Z0), CoordinateSystem.CYLINDRICAL),
                SliceCoord(float(R1), float(Z1), CoordinateSystem.CYLINDRICAL),
            )
            for (R0, Z0), (R1, Z1) in itertools.pairwise(np.nditer([Rs, Zs]))
        )
    if register_segment is not None:
        for shape in shapes:
            _ = register_segment(shape)
    return [Point2D(float(R), float(Z)) for R, Z in np.nditer([Rs[:-1], Zs[:-1]])]


def _reorder_portions(
    portions_lengths: Iterator[tuple[list[Point2D], float]],
    is_small: Callable[[tuple[list[Point2D], float]], bool],
) -> Iterator[tuple[list[Point2D], float]]:
    # FIXME: Don't like this. Is there a more functional way to express it?
    tail: list[tuple[list[Point2D], float]] = []
    initial = True
    for p_l in portions_lengths:
        if initial and is_small((p_l)):
            tail.append(p_l)
            continue
        initial = False
        yield p_l
    for p_l in tail:
        yield p_l


def _small_portion_centre_point(
    points: list[Point2D],
) -> list[Point2D]:
    n = len(points)
    return [Point2D(sum(p.R for p in points) / n, sum(p.Z for p in points) / n)]


def _left_terminus(
    portion: tuple[list[Point2D], float],
    is_small: Callable[[tuple[list[Point2D], float]], bool],
) -> list[Point2D]:
    """Return the left terminus of the portion to the right of this segment."""
    if is_small(portion):
        return _small_portion_centre_point(portion[0])
    return portion[0][-1:]


def _right_terminus(
    portion: tuple[list[Point2D], float],
    is_small: Callable[[tuple[list[Point2D], float]], bool],
) -> list[Point2D]:
    """Return the right terminus of the portion to the left of this segment."""
    if is_small(portion):
        return _small_portion_centre_point(portion[0])
    return portion[0][:1]


def _small_portion_length(
    portion: tuple[list[Point2D], float],
    is_small: Callable[[tuple[list[Point2D], float]], bool],
) -> float:
    """Return the contribution of this portion to the length of a neighbour.

    This will be half its length it is small and 0 otherwise.
    """
    if is_small(portion):
        return 0.5 * portion[1]
    return 0.0


def adjust_wall_resolution(
    points: Sequence[Point2D],
    target_size: float,
    min_size_factor: float = 1e-1,
    angle_threshold: float = np.pi / 8,
    register_segment: Optional[Callable[[AcrossFieldCurve], Quad]] = None,
) -> list[Point2D]:
    """Interpolate the points in the tokamak wall to the desired resolution.

    This will adjust the spacing of the points so that each segment
    connnecting them will have approximately the target length. Where
    the segments are fairly smooth, cubic interpolation will be
    used. However, sharp corners will be preserved.

    Parameters
    ----------
    points
        The initial set of points describing the wall.
    target_size
        The desired size of segments on the wall. Note that in the result
        the segments can be up to 1.5 times larger than this.
    min_size_factor
        The minimum size for a segment, as a fraction of the target size. If
        a segment is smaller than this it will be combined with an adjacent
        segment.
    angle_threshold
        The minimum angle between two segments for which to preserve the
        sharp corner.
    register_segment
        A function which can produce quad elements for a segment of the
        wall. These segments won't be used in this method, but it is assumed
        that the callback will register or cache them for use when constructing
        prisms in future. This is useful because it allows this method to pass
        in higher-order curvature information.

    Returns
    -------
    A more evely spaced set of points describing the wall.

    Group
    -----
    wall

    """
    # Split the wall up into portions that are smoothly-varying
    continuous_portions = list(
        map(list, _split_at_discontinuities(points, angle_threshold))
    )

    # Get the length of each of these portions
    portion_sizes = map(_segment_length, continuous_portions)

    def is_small(x: tuple[list[Point2D], float]) -> bool:
        return x[1] < target_size * min_size_factor

    # Combine adjacent small portions of the wall
    combined_portions = itertools.chain.from_iterable(
        _combine_small_portions(portions) if small else portions
        for small, portions in itertools.groupby(
            _reorder_portions(zip(continuous_portions, portion_sizes), is_small),
            is_small,
        )
    )

    # If there are any remaining small portions, merge them with the adjacent larger ones
    portions = (
        (
            _left_terminus(left, is_small)
            + portion[1:-1]
            + _right_terminus(right, is_small),
            max(
                1,
                round(
                    (
                        _small_portion_length(left, is_small)
                        + _small_portion_length(right, is_small)
                        + length
                    )
                    / target_size
                ),
            ),
        )
        for (left, (portion, length)), (_, right) in periodic_pairwise(
            periodic_pairwise(combined_portions)
        )
        if not is_small((portion, length))
    )

    # Interpolate each portion to divide it into segments as close to
    # the target size as possible
    return reduce(
        lambda x, y: x + y,
        (_interpolate_wall(p, n, register_segment) for p, n in portions),
    )


def wall_points_to_segments(
    points: Iterable[Point2D], tol: float = 1e-8
) -> list[WallSegment]:
    """Compute the edges connecting a series of vertices.

    This is useful for working with the list of points defining the
    wall of a tokamak. The wall is assumed to be closed and an edge
    will be inserted between the last and the first point.

    Group
    -----
    wall

    """

    def _points_to_segment(points: tuple[Point2D, Point2D]) -> Optional[WallSegment]:
        p1, p2 = points
        return WallSegment(p1.R, p1.Z, p2.R, p2.Z, tol)

    return [
        seg
        for seg in map(
            _points_to_segment,
            periodic_pairwise(points),
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
    """Return connectivity information for a logically-rectangular set of points.

    Some of the nodes returned here aren't actually connected in the
    sense of having an edge between them, but they are
    diagonally-opposite points on a common quad. If they are left out,
    then not all the outermost nodes will be identified in
    :func:`~neso_fame.wall.find_external_points`.

    """
    shape = np.broadcast(points.x1, points.x2).shape

    def get_neighbours(i: int, j: int) -> frozenset[SliceCoord]:
        i_min = i - 1 if i > 0 else 0
        i_max = i + 2
        j_min = j - 1 if j > 0 else 0
        j_max = j + 2
        return (
            points.get_set((slice(i_min, i_max), slice(j_min, j_max)))
        ) - points.get_set((i, j))

    return {
        points[i, j]: get_neighbours(i, j)
        for i in range(shape[0])
        for j in range(shape[1])
    }


T = TypeVar("T")


def periodic_pairwise(iterable: Iterable[T]) -> Iterable[tuple[T, T]]:
    """Return successive overlapping pairs taken from the input iterator.

    This is the same as :func:`itertools.pairwise`, except the last
    item in the returned iterator will be the pair of the last and
    first items in the original iterator.

    """
    iterator = iter(iterable)
    first = [next(iterator)]
    return itertools.pairwise(itertools.chain(first, iterator, first))
