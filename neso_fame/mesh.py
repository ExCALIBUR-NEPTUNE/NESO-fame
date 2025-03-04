"""Classes to represent meshes and their constituent elements."""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from functools import cache, cached_property
from typing import (
    Callable,
    Generic,
    Protocol,
    Type,
    TypeVar,
    cast,
    overload,
)

import numpy as np
import numpy.typing as npt
from scipy.interpolate import interp1d

from neso_fame.coordinates import (
    Coord,
    CoordinateSystem,
    Coords,
    FrozenCoordSet,
    SliceCoord,
    SliceCoords,
    coord_cache,
)
from neso_fame.offset import LazilyOffsetable, Offset


def field_aligned_positions(
    start_points: SliceCoords,
    dx3: float,
    field: FieldTrace,
    alignments: npt.NDArray,
    order: int,
    num_divisions: int = 1,
) -> FieldAlignedPositions:
    """Construct a :class:`~neso_fame.mesh.FieldAlignedPositions` object.

    This is the recommended way to instantiate
    ``FieldAlignedPositions`` objects, as it will ensure they end up
    in a valid

    Parameters
    ----------
    start_points
        The positions on the poloidal plane from which to project along field lines.
    dx3
        The width of the data in the x3-direction, before subdividing.
    field
        The underlying magnetic field along which to trace.
    alignments
        The extent to which projections from each of the starting
        points follows the field line versus stays fixed at the starting
        poloidal position. A value of 1 corresponds to it completely
        following the field line while 0 means it doesn't follow it at all
        (the x1 and x2 coordinates are unchanged as moving in the x3
        direction). Values in between result in linear interpolation
        between these two extremes. This array should be of the same or
        lower dimension as `start_points` and must be broadcast-compatible
        with it.
    order
        The order of accuracy with which to represent lines in the
        x3-direction. Determines the number of points that will be
        stored.
    num_divisions
        The number of segments this data is divided into along the
        x3-direction. Note that the result will not actually be divided
        in this way; it will just be able to be divided while still
        providing the desired order of accuracy.

    Group
    -----
    positions

    """
    start_array = np.broadcast(start_points.x1, start_points.x2)
    if start_array.ndim < alignments.ndim:
        raise ValueError(
            "`alignments` can not be of higher dimension than `start_points`"
        )
    try:
        np.broadcast(start_points.x1, start_points.x2, alignments)
    except ValueError:
        raise ValueError(
            "`alignments` must be broadcast-compatible with `start_points`"
        )
    if order < 1:
        raise ValueError("`order` must be strictly positive")
    if num_divisions < 1:
        raise ValueError("`num_divisions` must be strictly positive")
    n = num_divisions * order + 1
    x3 = np.linspace(-dx3 / 2, dx3 / 2, n)
    shape = start_array.shape + (n,)
    return FieldAlignedPositions(
        start_points,
        x3,
        field,
        alignments,
        0,
        1,
        np.empty(shape),
        np.empty(shape),
        np.full(start_array.shape + (1,), False),
    )


@dataclass(frozen=True, eq=False)
class FieldAlignedPositions:
    """Representation of positions of points spaced along field lines.

    .. warning::
        Objects of this class should not be constructed directly, as
        this can put them in an invalid state. Instead, use
        :func:`~neso_fame.mesh.field_aligned_positions`.

    This takes an array of starting positions on the poloidal plane
    and then creates 3D coordinates by following the magnetic field
    from those points. It can be used to hold data describing the
    shape of elements in a mesh.

    The object can be made to correspond to only a fraction of the
    data in the x3 direction by dividing it into a certain number of
    subdivisions and choosing one of these.

    Group
    -----
    positions

    """

    start_points: SliceCoords
    """The points on the poloidal surface from which field-line
    tracing begins. The points are logically arranged in an ND array
    (although some may be masked)."""
    x3: npt.NDArray
    """The x3 positions along the field lines at which to find the
    points."""
    trace: FieldTrace
    """The underlying magnetic field to which the geometry is aligned."""
    alignments: npt.NDArray
    """The extent to which projections from each of the starting
    points follows the field line versus stays fixed at the starting
    poloidal position. A value of 1 corresponds to it completely
    following the field line while 0 means it doesn't follow it at all
    (the x1 and x2 coordinates are unchanged as moving in the x3
    direction). Values in between result in linear interpolation
    between these two extremes. This array should be of the same or
    lower dimension as `start_points` and must be broadcast-compatible
    with it."""
    subdivision: int
    """The index of the segment of the data in the x3-direction that
    this particular object represents."""
    num_divisions: int
    """The number of segments this data is divided into along the x3-direction."""
    _x1: npt.NDArray
    """The x1 coordinate of the resulting points."""
    _x2: npt.NDArray
    """The x2 coordinate of the resulting points."""
    _computed: npt.NDArray
    """An array of booleans with the shape of `start_points`, plus an
    additional axis of length 1, indicating whether the positions
    along the field line originating at the corresponding start point
    have been calculated yet. The reason for the additional axis is to
    ensure that slicing always returns a view of the array, rather
    than a copy of the scalar."""

    def __getitem__(
        self, idx: int | slice | tuple[int | slice, ...]
    ) -> FieldAlignedPositions:
        """Slice the start-points of the data, returning the specificed subset.

        This returns a view of the data, not a copy. This means that
        as positions are calculated they will become available to
        other overlapping slices.

        """
        x1, x2, alignments = np.broadcast_arrays(
            self.start_points.x1, self.start_points.x2, self.alignments
        )
        return FieldAlignedPositions(
            SliceCoords(x1[idx], x2[idx], self.start_points.system),
            self.x3,
            self.trace,
            alignments[idx],
            self.subdivision,
            self.num_divisions,
            self._x1[idx],
            self._x2[idx],
            self._computed[idx],
        )

    # TODO: Add a hash and/or equality operator based on the locations, shape, and size of arrays in memory (plus subdivisions)
    @cached_property
    def order(self) -> int:
        """The order of accuracy for representing curves in the x3-direction."""
        return (len(self.x3) - 1) // self.num_divisions

    @property
    def _x3_start(self) -> int:
        """The index for the start of this subdivision."""
        return self.subdivision * self.order

    def subdivide(self, divisions: int) -> Iterator[FieldAlignedPositions]:
        """Split the data into the specified number of pieces in x3."""
        if divisions <= 1:
            yield self
            return
        if self.order % divisions != 0:
            raise ValueError(
                f"Can not subdivide {self.order} points in x3-direction into {divisions} equal parts"
            )
        for i in range(divisions):
            yield FieldAlignedPositions(
                self.start_points,
                self.x3,
                self.trace,
                self.alignments,
                self.subdivision * divisions + i,
                divisions * self.num_divisions,
                self._x1,
                self._x2,
                self._computed,
            )

    @cached_property
    def coords(self) -> Coords:
        """The 3D coordinates obtained by tracing the field lines."""
        alignments = np.broadcast_arrays(*self.start_points, self.alignments)[2]
        with np.nditer(self._computed, flags=["multi_index"]) as it:
            for x in it:
                if not x:
                    # self._computed was given an extra axis to ensure
                    # that when users index it an array will be
                    # returned. However, that will mess up indexing of
                    # other things, so need to drop it.
                    idx = it.multi_index[:-1]
                    positions, s = self.trace(
                        self.start_points[idx], self.x3, 1 - alignments[idx]
                    )
                    self._x1[idx] = positions.x1
                    self._x2[idx] = positions.x2
                    self._computed[idx] = True
        sl = slice(self._x3_start, self._x3_start + self.order + 1)
        return Coords(
            self._x1[..., sl],
            self._x2[..., sl],
            self.x3[sl].reshape((1,) * (self._x1.ndim - 1) + (self.order + 1,)),
            self.start_points.system,
        )


class FieldTrace(Protocol):
    """Representation of a magnetic field, used for tracing field lines.

    Group
    -----
    field line

    """

    def __call__(
        self, start: SliceCoord, x3: npt.ArrayLike, start_weight: float = 0.0
    ) -> tuple[SliceCoords, npt.NDArray]:
        """Calculate a position on a field line.

        Optionally, rather than strictly following the field line, you
        can weight so it stays closer to the starting position. This
        is done using the `start_weight` parameter. A value of 0 means
        that the function exactly follows the field line while a value
        of 1 means the x1 and x2 coordinates will be fixed to those of
        the `start` value. Values in between correspond to a weighted
        sum of these two options.

        The distance returned will correspond to the distance along
        the *weighted version* of the field line (rather than the
        original one). That is, it will always be the same as the
        distance you would get from integrating along the curve
        returned by this function.

        Parameters
        ----------
        start
            The position of the field-line in the x1-x2 plane at x3 = 0.
        x3
            x3 coordinates at which to calculate the position of the field line
        start_weight
            How much weight to apply to the start position versus the field line

        Returns
        -------
        The first element is the x1 and x2 coordinates of the field line at
        the provided x3 positions. The second is an array with the distance
        traveersed along the field line to those points.

        """
        ...


NormalisedCurve = Callable[[npt.ArrayLike], Coords]
"""A function describing a segment of a curve.

Often this curve represents a field line.

Parameters
----------
s : :obj:`numpy.typing.ArrayLike`
    An argument between 0 and 1, where 0 corresponds to the start of the
    curve and 1 to the end.

Returns
-------
Coords
    The locations on the curve. The distance of the point from the
    start of the field line is directly proportional to ``s``.

Group
-----
elements


.. rubric:: Alias
"""

AcrossFieldCurve = Callable[[npt.ArrayLike], SliceCoords]
"""A function describing a segment of a curve in the x1-x2 plane. This
curve is *not* aligned with the magnetic field.

Parameters
----------
s : :obj:`numpy.typing.ArrayLike`
    An argument between 0 and 1, where 0 corresponds to the start of the
    curve and 1 to the end.

Returns
-------
Coords
    The locations on the curve. The distance of the point from the
    start of the field line is directly proportional to ``s``.

Group
-----
elements


.. rubric:: Alias

"""


T = TypeVar("T")


class _ElementLike(Protocol):
    """Protocal defining the methods for manipulating mesh components.

    Exists for internal type-checking purposes.
    """

    def subdivide(self: T, num_divisions: int) -> Iterator[T]: ...


@dataclass(frozen=True)
class FieldAlignedCurve(LazilyOffsetable):
    """Represents a curve in 3D space which traces a field line.

    A curve is defined by a function which takes a single argument, 0
    <= s <= 1, and returns coordinates for the location on that curve
    in space. The distance along the curve from the start to the
    position represented by s is directly proportional to s.

    Group
    -----
    elements

    """

    field: FieldTracer
    """The underlying magnetic field to which the quadrilateral is aligned"""
    start: SliceCoord
    dx3: float
    """The span of the curve in the x3-direction."""
    subdivision: int = 0
    num_divisions: int = 1
    start_weight: float = 0.0

    @cached_property
    def function(self) -> NormalisedCurve:
        """Return the function representing this curve."""
        return self.field.get_normalised_subdivision(
            self.start,
            -0.5 * self.dx3,
            0.5 * self.dx3,
            self.subdivision,
            self.num_divisions,
            self.start_weight,
        )

    def __call__(self, s: npt.ArrayLike) -> Coords:
        """Calculate coordinates of position `s` on the curve.

        Convenience function so that a FieldAlignedCurve is itself a
        :obj:`~neso_fame.mesh.NormalisedCurve`.
        """
        return self.function(s)

    def subdivide(self, num_divisions: int) -> Iterator[FieldAlignedCurve]:
        """Split this curve into equal-length segments.

        Returns
        -------
        An iterator over each of the segments.
        """
        if num_divisions <= 1:
            yield self
        else:
            for i in range(num_divisions):
                yield FieldAlignedCurve(
                    self.field,
                    self.start,
                    self.dx3,
                    self.subdivision * num_divisions + i,
                    self.num_divisions * num_divisions,
                    self.start_weight,
                )


@overload
def control_points(element: NormalisedCurve | Quad, order: int) -> Coords: ...


@overload
def control_points(element: AcrossFieldCurve, order: int) -> SliceCoords: ...


@cache
def control_points(
    element: AcrossFieldCurve | NormalisedCurve | Quad, order
) -> SliceCoords | Coords:
    """Return locations to represent the shape to the specified order of accuracy.

    These points will be equally spaced. In the case of Quads, the order of
    the points in memory corresponds to that expected by Nektar++ when
    defining curved faces.

    Group
    -----
    elements

    """
    s = np.linspace(0.0, 1.0, order + 1)
    if isinstance(element, Quad):
        x1 = np.empty((order + 1, order + 1))
        x2 = np.empty((order + 1, order + 1))
        x3 = np.empty((order + 1, order + 1))
        for i, line in enumerate(map(element.get_field_line, s)):
            coords = line(s)
            x1[i, :] = coords.x1
            x2[i, :] = coords.x2
            x3[i, :] = coords.x3
        return Coords(x1, x2, x3, coords.system)
    return element(s)


@dataclass(frozen=True)
class StraightLineAcrossField(LazilyOffsetable):
    """A straight line that connects two points in the x1-x2 plane.

    It is an :obj:`~neso_fame.mesh.AcrossFieldCurve`.

    Group
    -----
    elements

    """

    north: SliceCoord
    south: SliceCoord

    def __call__(self, s: npt.ArrayLike) -> SliceCoords:
        """Calculate a position on the curve."""
        s = np.asarray(s)
        return SliceCoords(
            self.north.x1 + (self.south.x1 - self.north.x1) * s,
            self.north.x2 + (self.south.x2 - self.north.x2) * s,
            self.north.system,
        )


@dataclass(frozen=True)
class StraightLine(LazilyOffsetable):
    """A straight line that connects two points.

    It is a :obj:`~neso_fame.mesh.NormalisedCurve`.

    Group
    -----
    elements

    """

    north: Coord
    south: Coord
    subdivision: int = 0
    num_divisions: int = 1

    def __call__(self, s: npt.ArrayLike) -> Coords:
        """Calculate a position on the curve."""
        if self.north.system != self.south.system:
            raise ValueError(
                "Termini of line have different coordinate systems "
                f"'{self.north.system}' and '{self.south.system}'"
            )
        s = (self.subdivision + np.asarray(s)) / self.num_divisions
        return Coords(
            self.north.x1 + (self.south.x1 - self.north.x1) * s,
            self.north.x2 + (self.south.x2 - self.north.x2) * s,
            self.north.x3 + (self.south.x3 - self.north.x3) * s,
            self.north.system,
        )

    def subdivide(self, num_divisions: int) -> Iterator[StraightLine]:
        """Split this line into equal-length segments.

        Returns
        -------
        An iterator over each of the segments.
        """
        if num_divisions <= 1:
            yield self
        else:
            for i in range(num_divisions):
                yield StraightLine(
                    self.north,
                    self.south,
                    self.subdivision * num_divisions + i,
                    self.num_divisions * num_divisions,
                )


Segment = FieldAlignedCurve | StraightLine


@dataclass(frozen=True)
class Quad(LazilyOffsetable):
    """Representation of a four-sided polygon (quadrilateral).

    This is done using information about the shape of the magnetic
    field and the line where the quad intersects the x3=0 plane. You
    can choose to divide the quad into a number of conformal
    sub-quads, evenely spaced in x3, and pick one of them.

    Group
    -----
    elements

    """

    shape: AcrossFieldCurve
    """Desribes the shape of the quad in the poloidal plane from which it is
    projected."""
    field: FieldTracer
    """The underlying magnetic field to which the quadrilateral is aligned"""
    dx3: float
    """The width of the quad(s) in the x3-direction,
    before subdivideding."""
    subdivision: int = 0
    """The index for the quad being represented after it has been
    split into `num_divisions` in teh x3 direction."""
    num_divisions: int = 1
    """The number of conformal quads to split this into along the x3 direction."""
    north_start_weight: float = 0.0
    """The extent to which the south edge of the quad stays fixed at
    the starting poloidal position. A value of 0 corresponds to it
    completely following the field line while 1 means it doesn't
    follow it at all (the x1 and x2 coordinates are unchanged as
    moving in the x3 direction). Values in between result in linear
    interpolation between these two extremes."""
    south_start_weight: float = 0.0
    """The extent to which the south edge of the quad stays fixed at
    the starting poloidal position. A value of 0 corresponds to it
    completely following the field line while 1 means it doesn't
    follow it at all (the x1 and x2 coordinates are unchanged as
    moving in the x3 direction). Values in between result in linear
    interpolation between these two extremes.

    """

    def __iter__(self) -> Iterator[Segment]:
        """Iterate over the two curves defining the edges of the quadrilateral."""
        yield self.north
        yield self.south

    def get_field_line(self, s: float) -> Segment:
        """Get the field lign passing through location ``s`` of `Quad.shape`."""
        start = self.shape(s).to_coord()
        weight = (
            self.south_start_weight - self.north_start_weight
        ) * s + self.north_start_weight
        return FieldAlignedCurve(
            self.field,
            SliceCoord(start.x1, start.x2, start.system),
            self.dx3,
            self.subdivision,
            self.num_divisions,
            weight,
        )

    def _get_line_at_x3(self, x3: float) -> NormalisedCurve:
        """Return the 1-D shape the quad makes at the given x3 value."""
        x3_scaled = x3 / self.dx3 + 0.5
        x3 = (
            self.dx3 / self.num_divisions * (self.subdivision + x3_scaled)
            - 0.5 * self.dx3
        )
        s = np.linspace(0.0, 1.0, self.field.resolution)
        x1_coord = np.empty(self.field.resolution)
        x2_coord = np.empty(self.field.resolution)
        alignment_diff = self.south_start_weight - self.north_start_weight

        for i, (sval, start) in enumerate(zip(s, self.shape(s).iter_points())):
            # FIXME: This is duplicating work in the case of the actual edges. Also not caching anything.
            weight = alignment_diff * sval + self.north_start_weight
            x1_coord[i], x2_coord[i] = self.field.trace(start, x3, weight)[0].to_coord()
        coordinates = np.stack([x1_coord, x2_coord])
        order = (
            "cubic"
            if self.field.resolution > 3
            else "quadratic"
            if self.field.resolution > 2
            else "linear"
        )
        interp = interp1d(s, coordinates, order)
        coord_system = start.system

        def normalised_interpolator(s: npt.ArrayLike) -> Coords:
            locations = interp(s)
            return Coords(
                locations[0],
                locations[1],
                np.full_like(s, x3),
                coord_system,
            )

        return normalised_interpolator

    @cached_property
    def north(self) -> Segment:
        """Edge of the quadrilateral passing through ``self.shape(0.)``."""
        return self.get_field_line(0.0)

    @cached_property
    def south(self) -> Segment:
        """Edge of the quadrilateral passing through ``self.shape(1.)``."""
        return self.get_field_line(1.0)

    @cached_property
    def near(self) -> NormalisedCurve:
        """Cross-field edge of the quadrilateral with the smallest x3-value."""
        return self._get_line_at_x3(-0.5 * self.dx3)

    @cached_property
    def far(self) -> NormalisedCurve:
        """Cross-field edge of the quadrilateral with the largest x3-value."""
        return self._get_line_at_x3(0.5 * self.dx3)

    def corners(self) -> Coords:
        """Return the points corresponding to the corners of the quadrilateral."""
        north_corners = control_points(self.north, 1)
        south_corners = control_points(self.south, 1)
        return Coords(
            np.concatenate([north_corners.x1, south_corners.x1]),
            np.concatenate([north_corners.x2, south_corners.x2]),
            np.concatenate([north_corners.x3, south_corners.x3]),
            north_corners.system,
        )

    def subdivide(self, num_divisions: int) -> Iterator[Quad]:
        """Split the quad into the specified number of pieces.

        Returns an iterator of quad objects produced by splitting
        the bounding-line of this quad into the specified number of
        equally-sized segments. This has the effect of splitting the
        quad equally in the x3 direction.

        """
        if num_divisions <= 1:
            yield self
        else:
            for i in range(num_divisions):
                yield Quad(
                    self.shape,
                    self.field,
                    self.dx3,
                    self.subdivision * num_divisions + i,
                    self.num_divisions * num_divisions,
                    self.north_start_weight,
                    self.south_start_weight,
                )

    def make_flat_quad(self) -> Quad:
        """Create a new version of this Quad which is flat in the poloidal plane."""
        if isinstance(self.shape, StraightLineAcrossField):
            return self
        return Quad(
            StraightLineAcrossField(*self.shape([0.0, 1.0]).iter_points()),
            self.field,
            self.dx3,
            self.subdivision,
            self.num_divisions,
            self.north_start_weight,
            self.south_start_weight,
        )


@dataclass(frozen=True)
class EndShape(LazilyOffsetable):
    """Represents a polygon in an x3 plane.

    It is either the near or far end of a
    :class:`neso_fame.mesh.Prism`. It is in the x1-x2 plane and can have
    three or more curved edges. However, it is always flat.

    Group
    -----
    elements

    """

    edges: tuple[NormalisedCurve, ...]
    """Shape defining the edge of the polygon"""

    def __iter__(self) -> Iterator[NormalisedCurve]:
        """Iterate over the edges of the polygon."""
        return iter(self.edges)

    def corners(self) -> Coords:
        """Return the points corresponding to the vertices of the polygon."""
        all_points = list(
            FrozenCoordSet(
                itertools.chain.from_iterable(
                    control_points(e, 1).iter_points() for e in self.edges
                )
            )
        )
        return Coords(
            np.array([p.x1 for p in all_points]),
            np.array([p.x2 for p in all_points]),
            np.array([p.x3 for p in all_points]),
            next(iter(all_points)).system,
        )


def _ensure_function_start_direction(
    func: AcrossFieldCurve, start: SliceCoord
) -> AcrossFieldCurve:
    tmp1, tmp2 = func([0.0, 1.0]).iter_points()
    if tmp1.approx_eq(start):
        return func
    if not tmp2.approx_eq(start):
        raise RuntimeError("Neither terminus of `func` is at `start`")
    return _reverse(func)


def _reverse(func: AcrossFieldCurve) -> AcrossFieldCurve:
    return lambda x: func(1 - np.asarray(x))


@dataclass(frozen=True)
class Prism(LazilyOffsetable):
    """Representation of a triangular or rectangular (or other) prism.

    This is represented by a number quads making up its faces. The
    remaining two faces are made up of the edges of these quads at s=0
    and s=1 and are normal to the x3-direction. This will most often
    be used for hexahedrons and, occasionally, triangular prisms.

    .. note::
       When creating a 6-face prism (hexahedron), the first two sides
       must be opposite each other (meaning that the final two sides will
       also be opposite each other). I should probably change this so that
       they should be in order, instead.

    Group
    -----
    elements

    """

    sides: tuple[Quad, ...]
    """Shapes defining the edges of the hexahedron"""
    custom_poloidal_mapping: (
        None | Callable[[npt.ArrayLike, npt.ArrayLike], SliceCoords]
    ) = None
    """Custom mapping between reference coordinates and real
    coordinates on the poloidal plane. This is useful for triangular
    prisms where all three faces are curved or if you want more
    control over the placement of interior quadrature points."""

    def __iter__(self) -> Iterator[Quad]:
        """Iterate over the four quads defining the faces of the hexahedron."""
        return iter(self.sides)

    @cached_property
    def near(self) -> EndShape:
        """The face of the prism in the x3 plane with the smallest x3 value."""
        return EndShape(tuple(s.near for s in self.sides))

    @cached_property
    def far(self) -> EndShape:
        """The face of the prism in the x3 plane with the largest x3 value."""
        return EndShape(tuple(s.far for s in self.sides))

    def corners(self) -> Coords:
        """Return the points corresponding to the vertices of the hexahedron."""
        near_corners = self.near.corners()
        far_corners = self.far.corners()
        # TODO Check that east and west corners are the same as north and south
        return Coords(
            np.concatenate([near_corners.x1, far_corners.x1]),
            np.concatenate([near_corners.x2, far_corners.x2]),
            np.concatenate([near_corners.x3, far_corners.x3]),
            near_corners.system,
        )

    def subdivide(self, num_divisions: int) -> Iterator[Prism]:
        """Split the prism into the specified number of pieces.

        Returns an iterator of prism objects produced by splitting
        the bounding-quads of this prism into the specified number of
        equally-sized parts. This has the effect of splitting the
        prism equally in the x3 direction.

        """
        if num_divisions <= 1:
            yield self
        else:
            for sides in zip(*(s.subdivide(num_divisions) for s in self.sides)):
                yield Prism(sides)

    @staticmethod
    def _nonlinear_adjustment(
        func: AcrossFieldCurve,
        start: SliceCoord,
        end: SliceCoord,
        xref1: npt.NDArray,
        xref2: npt.NDArray,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Subtract a linear function from a nonlinear one.

        Values at the start and end will be zero 0.
        """
        x1, x2 = func(xref1)
        return (
            (x1 - start.x1 - (end.x1 - start.x1) * xref1) * xref2,
            (x2 - start.x2 - (end.x2 - start.x2) * xref1) * xref2,
        )

    def _poloidal_map_tri(self, s: npt.ArrayLike, t: npt.ArrayLike) -> SliceCoords:
        # Order sides so first one is linear, plus wrap functions so east and west start at south
        try:
            south = next(
                iter(
                    side
                    for side in self.sides
                    if isinstance(side.shape, StraightLineAcrossField)
                )
            )
        except StopIteration:
            raise NotImplementedError(
                "Can not currently compute poloidal map for triangular prism where all sides are curved."
            )
        sw, se = south.shape([0.0, 1.0]).iter_points()
        others = list(self.sides)
        others.remove(south)
        tmp1, tmp2 = others[0].shape([0.0, 1.0]).iter_points()

        if tmp1.approx_eq(sw):
            west = west_initial = others[0].shape
            north = tmp2
            east_initial = others[1].shape
            east = _ensure_function_start_direction(others[1].shape, se)
        elif tmp2.approx_eq(sw):
            west_initial = others[0].shape
            west = _reverse(others[0].shape)
            north = tmp1
            east_initial = others[1].shape
            east = _ensure_function_start_direction(others[1].shape, se)
        elif tmp1.approx_eq(se):
            east = east_initial = others[0].shape
            north = tmp2
            west_initial = others[1].shape
            west = _ensure_function_start_direction(others[1].shape, sw)
        elif tmp2.approx_eq(se):
            east_initial = others[0].shape
            east = _reverse(others[0].shape)
            north = tmp1
            west_initial = others[1].shape
            west = _ensure_function_start_direction(others[1].shape, sw)
        else:
            raise RuntimeError("Sides of triangular prism not conforming.")

        # Compute result for linear element
        s = cast(npt.NDArray, np.ma.asanyarray(s))
        t = cast(npt.NDArray, np.ma.asanyarray(t))
        real_x1 = sw.x1 * (1 - s) * (1 - t) + se.x1 * s * (1 - t) + north.x1 * t
        real_x2 = sw.x2 * (1 - s) * (1 - t) + se.x2 * s * (1 - t) + north.x2 * t

        # Adjust the result if east or west are not straight lines
        if not isinstance(east_initial, StraightLineAcrossField):
            x1_adj, x2_adj = self._nonlinear_adjustment(east, se, north, t, s)
            real_x1 += x1_adj
            real_x2 += x2_adj
        if not isinstance(west_initial, StraightLineAcrossField):
            x1_adj, x2_adj = self._nonlinear_adjustment(west, sw, north, t, 1 - s)
            real_x1 += x1_adj
            real_x2 += x2_adj
        return SliceCoords(real_x1, real_x2, sw.system)

    @staticmethod
    def _poloidal_quad_order_remaining_edges(
        nx: SliceCoord,
        x0: SliceCoord,
        x1: SliceCoord,
        south0: SliceCoord,
        south1: SliceCoord,
        xcurve_initial: AcrossFieldCurve,
        scurve_initial: AcrossFieldCurve,
        x_at_start: bool,
    ) -> tuple[SliceCoord, SliceCoord, AcrossFieldCurve, AcrossFieldCurve]:
        if nx.approx_eq(x0):
            sx = x1
            xcurve = _reverse(xcurve_initial)
        else:
            sx = x0
            xcurve = xcurve_initial
        if south0.approx_eq(sx):
            sy = south1
            if x_at_start:
                scurve = scurve_initial
            else:
                scurve = _reverse(scurve_initial)
        else:
            sy = south0
            if x_at_start:
                scurve = _reverse(scurve_initial)
            else:
                scurve = scurve_initial
        return sx, sy, xcurve, scurve

    def _poloidal_map_quad(self, s: npt.ArrayLike, t: npt.ArrayLike) -> SliceCoords:
        # Work out orientation of all sides
        # Is the order I'm choosing going to result in a negative Jacobian? Maybe I should treat north as 0 and south as 1? (and similar for east/west)
        north_initial, south_initial, east_initial, west_initial = iter(
            s.shape for s in self.sides
        )
        north0, north1 = north_initial([0.0, 1.0]).iter_points()
        east0, east1 = east_initial([0.0, 1.0]).iter_points()
        south0, south1 = south_initial([0.0, 1.0]).iter_points()
        west0, west1 = west_initial([0.0, 1.0]).iter_points()
        if north0.approx_eq(east0):
            ne = north0
            nw = north1
            north = _reverse(north_initial)
            east = _reverse(east_initial)
            sw, se, west, south = self._poloidal_quad_order_remaining_edges(
                nw, west0, west1, south0, south1, west_initial, south_initial, True
            )
        elif north0.approx_eq(east1):
            ne = north0
            nw = north1
            north = _reverse(north_initial)
            east = east_initial
            sw, se, west, south = self._poloidal_quad_order_remaining_edges(
                nw, west0, west1, south0, south1, west_initial, south_initial, True
            )
        elif north0.approx_eq(west0):
            ne = north1
            nw = north0
            north = north_initial
            west = _reverse(west_initial)
            se, sw, east, south = self._poloidal_quad_order_remaining_edges(
                ne, east0, east1, south0, south1, east_initial, south_initial, False
            )
        else:
            assert north0.approx_eq(west1)
            ne = north1
            nw = north0
            north = north_initial
            west = west_initial
            se, sw, east, south = self._poloidal_quad_order_remaining_edges(
                ne, east0, east1, south0, south1, east_initial, south_initial, False
            )
        # Compute result for linear element
        s = cast(npt.NDArray, np.ma.asanyarray(s))
        t = cast(npt.NDArray, np.ma.asanyarray(t))
        real_x1 = (
            sw.x1 * (1 - s) * (1 - t)
            + se.x1 * s * (1 - t)
            + nw.x1 * (1 - s) * t
            + ne.x1 * s * t
        )
        real_x2 = (
            sw.x2 * (1 - s) * (1 - t)
            + se.x2 * s * (1 - t)
            + nw.x2 * (1 - s) * t
            + ne.x2 * s * t
        )
        # Adjust the result for any edges that are not straight lines
        if not isinstance(north_initial, StraightLineAcrossField):
            x1_adj, x2_adj = self._nonlinear_adjustment(north, nw, ne, s, t)
            real_x1 += x1_adj
            real_x2 += x2_adj
        if not isinstance(south_initial, StraightLineAcrossField):
            x1_adj, x2_adj = self._nonlinear_adjustment(south, sw, se, s, 1 - t)
            real_x1 += x1_adj
            real_x2 += x2_adj
        if not isinstance(east_initial, StraightLineAcrossField):
            x1_adj, x2_adj = self._nonlinear_adjustment(east, se, ne, t, s)
            real_x1 += x1_adj
            real_x2 += x2_adj
        if not isinstance(west_initial, StraightLineAcrossField):
            x1_adj, x2_adj = self._nonlinear_adjustment(west, sw, nw, t, 1 - s)
            real_x1 += x1_adj
            real_x2 += x2_adj
        return SliceCoords(real_x1, real_x2, sw.system)

    def poloidal_map(self, s: npt.ArrayLike, t: npt.ArrayLike) -> SliceCoords:
        r"""Calculate positions on the poloidal (x3) cross-section of this prism.

        Converts between reference coordinates :math:`s, t
        \in [0, 1]` to coordinates in real space.

        """
        # Should I have a special case for the corners, to save computation?
        if self.custom_poloidal_mapping:
            return self.custom_poloidal_mapping(s, t)
        if len(self.sides) > 4:
            raise NotImplementedError(
                "Mapping only implemented for triangular and rectangular prisms"
            )
        elif len(self.sides) == 3:
            return self._poloidal_map_tri(s, t)
        else:
            return self._poloidal_map_quad(s, t)

    def make_flat_faces(self) -> Prism:
        """Create a new prism where sides don't curve in the poloidal plane."""
        return Prism(tuple(face.make_flat_quad() for face in self.sides))


E = TypeVar("E", Quad, Prism)
B = TypeVar("B", Segment, Quad)
C = TypeVar("C", NormalisedCurve, EndShape)


@dataclass(frozen=True)
class MeshLayer(Generic[E, B, C], LazilyOffsetable):
    """Representation of a single "layer" of the mesh.

    A layer is a region of the mesh where the elements are conformal
    and aligned with the magnetic field. A mesh may contain multiple
    layers, but there will be a non-conformal interface between each
    of them.

    Group
    -----
    mesh

    """

    reference_elements: Sequence[E]
    """A colelction of the :class:`~neso_fame.mesh.Quad` or
    :class:`~neso_fame.mesh.Prism` elements making up the layer (without
    any subdivision)."""
    bounds: Sequence[frozenset[B]]
    """An ordered collection of sets of :class:`~neso_fame.mesh.Curve`
    or :class:`~neso_fame.mesh.Quad` objects (faces or edges,
    repsectively). Each set describes a particular boundary regions of
    the layer. The near and far faces of the layer are not included in
    these."""
    subdivisions: int = 1
    """The number of elements deep the layer should be in the
    x3-direction."""

    @classmethod
    def QuadMeshLayer(
        cls,
        reference_elements: Sequence[Quad],
        bounds: Sequence[frozenset[Segment]],
        subdivisions: int = 1,
    ) -> QuadMeshLayer:
        """Construct a MeshLayer object made up of quads.

        This method isn't really necessary but can be useful to
        reassure the type-checker that it is a QuadMeshLayer that is being
        constructed.

        """
        return cls(reference_elements, bounds, subdivisions)  # type: ignore

    @classmethod
    def PrismMeshLayer(
        cls,
        reference_elements: Sequence[Prism],
        bounds: Sequence[frozenset[Quad]],
        subdivisions: int = 1,
    ) -> PrismMeshLayer:
        """Construct a MeshLayer object made up of prisms, in a type-safe way.

        This method isn't really necessary but can be useful to
        reassure the type-checker that it is a PrismMeshLayer that is being
        constructed.

        """
        return cls(reference_elements, bounds, subdivisions)  # type: ignore

    def __iter__(self) -> Iterator[E]:
        """Iterate over all of the elements making up this layer of the mesh."""
        return self._iterate_elements(self.reference_elements, self.subdivisions)

    def __len__(self) -> int:
        """Return the number of elements in this layer."""
        return len(self.reference_elements) * self.subdivisions

    @property
    def element_type(self) -> Type[E]:
        """Return the type of the elements of the mesh layer."""
        return type(next(iter(self.reference_elements)))

    def quads(self) -> Iterator[Quad]:
        """Iterate over the `Quad` objects in the layer.

        If the mesh is made up of quads then this is the same as
        iterating over the elements. Otherwise, it iterates over the
        quads defining the boundaries of the constituent `Prism`
        elements.

        """
        if len(self.reference_elements) > 0 and issubclass(self.element_type, Quad):
            return iter(self)
        else:
            return itertools.chain.from_iterable(
                map(iter, cast(MeshLayer[Prism, Quad, EndShape], self))
            )

    def boundaries(self) -> Iterator[frozenset[B]]:
        """Iterate over the boundary regions in this layer.

        This excludes boundaries normal to the x3-direction. There may
        be any number of boundary regions. If the mesh is made up of
        `Quad` elements then the boundaries are sets of `Curve`
        objects. If the mesh is made up of `Prism` elements, then the
        boundaries are sets of `Quad` objects.

        """
        return map(
            frozenset,
            (self._iterate_elements(b, self.subdivisions) for b in self.bounds),
        )

    def near_faces(self) -> Iterator[C]:
        """Iterate over the near faces of the elements in the layer.

        If the layer is subdivided (i.e., is more than one element
        deep in the x3-direction) then only the near faces of the
        first subdivision will be returned. This constitutes one of
        the boundaries normal to the x3-direction.

        """
        return (
            cast(C, e.near) for e in self._iterate_elements(self.reference_elements, 1)
        )

    def far_faces(self) -> Iterator[C]:
        """Iterate over the far faces of the elements in the layer.

        If the layer is subdivided (i.e., is more than one element
        deep in the x3-direction) then only the far faces of the last
        subdivision will be returned. This constitutes one of the
        boundaries normal to the x3-direction.

        Note
        ----
        This won't necessarily be bit-wise identical to the last
        subdivision. However, as coordinates are rounded to 8 decimal
        places when creating Nektar++ objects, it won't matter.

        """
        return (
            cast(C, e.far) for e in self._iterate_elements(self.reference_elements, 1)
        )

    @overload
    @staticmethod
    def _iterate_elements(elements: Iterable[E], subdivisions: int) -> Iterator[E]: ...

    @overload
    @staticmethod
    def _iterate_elements(elements: Iterable[B], subdivisions: int) -> Iterator[B]: ...

    @staticmethod
    def _iterate_elements(
        elements: Iterable[_ElementLike], subdivisions: int
    ) -> Iterator[_ElementLike]:
        """Iterate over elements of the layer.

        This is a convenience method used by other iteration
        methods. It handles subdivisions appropriately.

        """
        return itertools.chain.from_iterable(
            (e.subdivide(subdivisions) for e in elements)
        )


@dataclass(frozen=True)
class GenericMesh(Generic[E, B, C]):
    """Class representing a complete mesh.

    The mesh is defined by a representative layer and an array of
    offsets. Physically, these correspond to a mesh made up of a
    series of identical layers, with nonconformal interfaces, each
    offset by a certain ammount along the x3-direction.

    Note
    ----
    This class is generic in both the element and boundary types, but
    only certain combinations of these make sense in practice:
    :class:`~neso_fame.mesh.Quad` elements and
    :class:`~neso_fame.mesh.Curve` boundaries; or
    :class:`~neso_fame.mesh.Prism` elements and
    :class:`~neso_fame.mesh.Quad` boundaries. GenericMesh should not
    be used for type annotations; use :obj:`~neso_fame.mesh.QuadMesh`,
    :obj:`~neso_fame.mesh.PrismMesh`, or :obj:`~neso_fame.mesh.Mesh`
    instead, as these are constrained to the valid combinations.

    Group
    -----
    mesh

    """

    reference_layer: MeshLayer[E, B, C]
    """A layer from which all of the constituant layers of the mesh
    object will be produced."""
    offsets: npt.NDArray
    """The x3 offset for each layer of the mesh."""

    def layers(self) -> Iterable[MeshLayer[E, B, C]]:
        """Iterate over the layers of this mesh."""
        return (Offset(self.reference_layer, off) for off in self.offsets)

    def __iter__(self) -> Iterator[E]:
        """Iterate over all of the elements contained in this mesh."""
        return itertools.chain.from_iterable(map(iter, self.layers()))

    def __len__(self) -> int:
        """Return the number of elements in this mesh."""
        return len(self.reference_layer) * self.offsets.size


QuadMeshLayer = MeshLayer[Quad, Segment, NormalisedCurve]
PrismMeshLayer = MeshLayer[Prism, Quad, EndShape]


QuadMesh = GenericMesh[Quad, Segment, NormalisedCurve]
"""
Mesh made up of `Quad` elements.

Group
-----
mesh
"""
PrismMesh = GenericMesh[Prism, Quad, EndShape]
"""
Mesh made up of `Prism` elements.

Group
-----
mesh


.. rubric:: Alias

"""
Mesh = QuadMesh | PrismMesh
"""
Valid types of mesh, to be used for type annotations.

Group
-----
mesh


.. rubric:: Alias

"""


def normalise_field_line(
    trace: FieldTrace,
    start: SliceCoord,
    x3_min: float,
    x3_max: float,
    resolution: int = 10,
) -> NormalisedCurve:
    """Trace a magnetic field-line from a starting point.

    Takes a function defining a magnetic field and returns a new
    function tracing a field line within it.

    Parameters
    ----------
    trace
        A callable which takes a `SliceCoord` defining a position on
        the x3=0 plane and an array-like object with x3
        coordinates. It should return a 2-tuple. The first element is
        the locations found by tracing the magnetic field line
        beginning at the position of the first argument until reaching
        the x3 locations described in the second argument. The second
        element is the distance traversed along the field line.
    start
        The location on the x3=0 plane which the traced field line will
        pass through.
    x3_min
        The minimum x3 value to trace the field line from.
    x3_max
        The maximum x3 value to trace the field line to.
    resolution
        The number of locations used along the field line used to
        interpolate distances.

    Returns
    -------
    :obj:`~neso_fame.mesh.NormalisedCurve`
        A function taking an argument ``s`` between 0 and 1 and returning
        a coordinate along the field line. The distance of the point from
        the start of the field line is directly proportional to
        ``s``. This function is vectorised, so can be called with an
        array-like argument.

    Group
    -----
    field line

    """
    x3 = np.linspace(x3_min, x3_max, resolution)
    x1_x2_coords: SliceCoords
    s: npt.NDArray
    x1_x2_coords, s = trace(start, x3)
    coordinates = np.stack([*x1_x2_coords, x3])
    order = "cubic" if len(s) > 3 else "quadratic" if len(s) > 2 else "linear"
    distances = (s - s[0]) / (s[-1] - s[0])
    # Make sure limits are exact
    distances[0] = 0.0
    distances[-1] = 1.0
    interp = interp1d(distances, coordinates, order)
    coord_system = start.system

    def normalised_interpolator(s: npt.ArrayLike) -> Coords:
        locations = interp(s)
        return Coords(locations[0], locations[1], locations[2], coord_system)

    return normalised_interpolator


@dataclass(frozen=True)
class FieldTracer:
    """Manage the tracing of the field and production of normalised lines.

    This class exists assist with subdivided quads. To get information
    for the division at the end of a quad, it will be necessary to
    integrate along a field line through all of the previous
    divisions. This class handles caching of that information for
    later use. Without it, we would need to build the intermediate
    divisions all at once, which goes against the lazy approach to
    evaluation used by FAME.


    Group
    -----
    field line

    """

    trace: FieldTrace
    resolution: int = 10

    @staticmethod
    def _make_interpolator(
        s: npt.NDArray, coords: npt.NDArray, order: str, system: CoordinateSystem
    ) -> NormalisedCurve:
        interp = interp1d((s - s[0]) / (s[-1] - s[0]), coords, order)

        # FIXME: Does it really matter if I normalise this way? It significantly reduces accuracy of positions on the field line.
        def normalised_interpolator(s: npt.ArrayLike) -> Coords:
            locations = interp(s)
            return Coords(locations[0], locations[1], locations[2], system)

        return normalised_interpolator

    @coord_cache()
    def _normalise_and_subdivide(
        self,
        start: SliceCoord,
        x3_min: float,
        x3_max: float,
        divisions: int,
        start_weight: float = 0.0,
    ) -> list[NormalisedCurve]:
        x3 = np.linspace(x3_min, x3_max, (self.resolution - 1) * divisions + 1)
        x1_x2_coords: SliceCoords
        s: npt.NDArray
        x1_x2_coords, s = self.trace(start, x3, start_weight)
        coordinates = np.stack([*x1_x2_coords, x3])
        order = (
            "cubic"
            if self.resolution > 3
            else "quadratic"
            if self.resolution > 2
            else "linear"
        )

        slices = (
            slice(i * (self.resolution - 1), (i + 1) * (self.resolution - 1) + 1)
            for i in range(divisions)
        )
        return [
            self._make_interpolator(s[sl], coordinates[:, sl], order, start.system)
            for sl in slices
        ]

    def get_normalised_subdivision(
        self,
        start: SliceCoord,
        x3_min: float,
        x3_max: float,
        division: int,
        total_divisions: int,
        start_weight: float = 0.0,
    ) -> NormalisedCurve:
        """Normalises a field trace.

        Return a function describing the field line within a given
        division of a quad.

        Parameters
        ----------
        start
            The point on the x3=0 plane that the field line passes through.
        x3_min
            The lower x3 limit for the quad (before it has been subdivided)
        x3_max
            The upper x3 limit for the quad (before it has been subdivided)
        division
            The index of the subdivision of the quad to return a field line
            sigment for
        total_divisions
            The total number of subdivisions this quad is split into along
            the x3 direction
        start_weight
            How much (if at all) to shift the result so it stays close to
            `start`. A value of 0 means no shift will be be applied and the
            field will be traced exactly. A value of 1 means the function
            will always return the x1 and x2 values of the start position.
            Values in between correspond to a weighted sum of these two
            extremes.

        Returns
        -------
        :obj:`~neso_fame.mesh.NormalisedCurve`
            The segment of the field line passing through `start` which is
            within the specified subdivision of the quad.

        """
        assert division < total_divisions, (
            f"Can not request division {division} when only {total_divisions} available"
        )
        assert total_divisions > 0, "Number of divisions must be positive"
        assert division >= 0, "Division number must be non-negative"
        segments = self._normalise_and_subdivide(
            start, x3_min, x3_max, total_divisions, start_weight
        )
        return segments[division]
