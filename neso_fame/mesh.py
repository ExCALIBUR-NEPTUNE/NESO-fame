"""Classes to represent meshes and their constituent elements."""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from typing import (
    Generic,
    NewType,
    Protocol,
    Type,
    TypeVar,
    cast,
    overload,
)

import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from neso_fame.coordinates import (
    Coord,
    Coords,
    SliceCoord,
    SliceCoords,
)
from neso_fame.offset import LazilyOffsetable, Offset


class _ElementLike(Protocol):
    """Protocal defining the methods for manipulating mesh components.

    Exists for internal type-checking purposes.
    """

    def subdivide(self, num_divisions: int) -> Iterator[Self]: ...


def subdividable_field_aligned_positions(
    start_points: SliceCoords,
    dx3: float,
    field: FieldTrace,
    alignments: npt.NDArray,
    order: int,
    num_divisions: int = 1,
) -> FieldAlignedPositions:
    """Construct a :class:`~neso_fame.mesh.FieldAlignedPositions` object ready to be subdivided.

    This is one of the recommended ways to instantiate
    ``FieldAlignedPositions`` objects, as it will ensure they end up
    in a valid state.

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

    """
    return field_aligned_positions(
        start_points, dx3, field, alignments, num_divisions * order, 0, 1
    )


def field_aligned_positions(
    start_points: SliceCoords,
    dx3: float,
    field: FieldTrace,
    alignments: npt.NDArray,
    order: int,
    subdivision: int = 0,
    num_divisions: int = 1,
) -> FieldAlignedPositions:
    """Construct a :class:`~neso_fame.mesh.FieldAlignedPositions` object.

    This is one of the recommended way to instantiate
    ``FieldAlignedPositions`` objects, as it will ensure they end up
    in a valid state.

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
    subdivision
        The index of the segment of the data in the x3-direction that
        this particular object represents
    num_divisions
        The number of segments this data is divided into along the
        x3-direction.

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
        subdivision,
        num_divisions,
        np.empty(shape),
        np.empty(shape),
        np.full(start_array.shape + (1,), False),
    )


@dataclass(frozen=True, eq=False)
class FieldAlignedPositions(LazilyOffsetable):
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

        This returns a view of the data in the data, not a copy. This
        means that as positions are calculated they will become
        available to other overlapping slices.
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

    @property
    def poloidal_shape(self) -> tuple[int, ...]:
        """The logical shape of the array of starting points in the poloidal plane."""
        return np.broadcast(
            self.start_points.x1, self.start_points.x2, self.alignments
        ).shape

    @property
    def shape(self) -> tuple[int, ...]:
        """The logical shape of the 3D coordinates obtained by tracing field lines."""
        return self.poloidal_shape + self.x3.shape

    # TODO: Add a hash and/or equality operator based on the locations, shape, and size of arrays in memory (plus subdivisions)
    @cached_property
    def order(self) -> int:
        """The order of accuracy for representing curves in the x3-direction."""
        return (len(self.x3) - 1) // self.num_divisions

    @property
    def _x3_start(self) -> int:
        """The index for the start of this subdivision."""
        return self.subdivision * self.order

    @cached_property
    def diagonal(self) -> FieldAlignedPositions:
        """Get the diagonal of the start-points of the data.

        This is useful when the pointes represent a triangular prism,
        as the diagonal will correspond to one of the faces.

        """
        x1, x2, alignments = np.broadcast_arrays(
            self.start_points.x1, self.start_points.x2, self.alignments
        )
        return FieldAlignedPositions(
            SliceCoords(
                np.flipud(x1).diagonal(),
                np.flipud(x2).diagonal(),
                self.start_points.system,
            ),
            self.x3,
            self.trace,
            np.flipud(alignments).diagonal(),
            self.subdivision,
            self.num_divisions,
            # TODO: At some point NumPy will make the diagonals
            # writeable. At that point it becomes counterproductive to
            # make a copy.
            np.swapaxes(np.flipud(self._x1).diagonal(), -1, -2).copy(),
            np.swapaxes(np.flipud(self._x2).diagonal(), -1, -2).copy(),
            np.swapaxes(np.flipud(self._computed).diagonal(), -1, -2).copy(),
        )

    def subdivide(self, num_divisions: int) -> Iterator[FieldAlignedPositions]:
        """Split the data into the specified number of pieces in x3."""
        if num_divisions <= 1:
            yield self
            return
        if self.order % num_divisions != 0:
            raise ValueError(
                f"Can not subdivide {self.order} points in x3-direction into {num_divisions} equal parts"
            )
        for i in range(num_divisions):
            yield FieldAlignedPositions(
                self.start_points,
                self.x3,
                self.trace,
                self.alignments,
                self.subdivision * num_divisions + i,
                num_divisions * self.num_divisions,
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


Curve = NewType("Curve", Coords)
"""A 1D set of points tracing out a curve in 3D space.

Often this curve represents a field line.

Group
-----
elements


.. rubric:: Alias
"""

AcrossFieldCurve = NewType("AcrossFieldCurve", SliceCoords)
"""A 1D set of points tracing a curve in the poloidal plane.

This curve is *not* aligned with the magnetic field.

Group
-----
elements


.. rubric:: Alias

"""


FieldAlignedCurve = NewType("FieldAlignedCurve", FieldAlignedPositions)
"""Represents a curve in 3D space which traces a field line.

This is just a class:`~neso_fame.mesh.FieldAlignedPositions` object
with a scalar start-position. It may not be totally field-aligned,
depending on the value of the ``alignment`` attribute.

Group
-----
elements

"""


def order(element: AcrossFieldCurve | Curve | FieldAlignedCurve | Quad | Prism) -> int:
    """Get the order of accuracy used to represent the element."""
    if isinstance(element, (SliceCoords, Coords)):
        return len(element) - 1
    if isinstance(element, FieldAlignedPositions):
        return element.order
    return element.nodes.order


@overload
def control_points(element: Curve | FieldAlignedCurve | Quad | Prism) -> Coords: ...


@overload
def control_points(element: AcrossFieldCurve) -> SliceCoords: ...


def control_points(
    element: AcrossFieldCurve | Curve | FieldAlignedCurve | Quad | Prism,
) -> SliceCoords | Coords:
    """Return locations to represent the shape to the specified order of accuracy.

    These points will be equally spaced. In the case of Quads, the order of
    the points in memory corresponds to that expected by Nektar++ when
    defining curved faces.

    Group
    -----
    elements

    """
    if isinstance(element, (SliceCoords, Coords)):
        return element
    if isinstance(element, FieldAlignedPositions):
        return element.coords
    return element.nodes.coords


def straight_line_across_field(
    north: SliceCoord, south: SliceCoord, order: int
) -> AcrossFieldCurve:
    """Create a straight line that connects two points in the x1-x2 plane.

    It is an :obj:`~neso_fame.mesh.AcrossFieldCurve`.

    Group
    -----
    elements

    """
    if north.system != south.system:
        raise ValueError(
            f"Start and end points have different coordinate systems: {north.system} and {south.system}"
        )
    return AcrossFieldCurve(
        SliceCoords(
            np.linspace(north.x1, south.x1, order + 1),
            np.linspace(north.x2, south.x2, order + 1),
            north.system,
        )
    )


def _invalid_trace(
    start: SliceCoord, x3: npt.ArrayLike, start_weight: float = 0.0
) -> tuple[SliceCoords, npt.NDArray]:
    raise RuntimeError(
        "This trace should never be called; something has gone very wrong."
    )


def straight_line(
    north: Coord, south: Coord, order: int, subdivision: int = 0, num_divisions: int = 1
) -> FieldAlignedCurve:
    """Create a straight line that connects two points in 3D space.

    Group
    -----
    elements

    """
    if north.system != south.system:
        raise ValueError(
            f"Start and end points have different coordinate systems: {north.system} and {south.system}"
        )
    return FieldAlignedCurve(
        FieldAlignedPositions(
            SliceCoords(
                np.array(0.5 * (north.x1 + south.x1)),
                np.array(0.5 * (north.x2 + south.x2)),
                north.system,
            ),
            np.linspace(north.x3, south.x3, order + 1),
            _invalid_trace,  # This will never be used, so pass a dummy implementation.
            np.array(0.0),
            subdivision,
            num_divisions,
            np.linspace(north.x1, south.x1, order + 1),
            np.linspace(north.x2, south.x2, order + 1),
            np.array(True),
        )
    )


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

    nodes: FieldAlignedPositions
    """The control points for this prism. The start-positions for this
    should be arranged in a logically-1D array."""

    def __iter__(self) -> Iterator[FieldAlignedCurve]:
        """Iterate over the two curves defining the edges of the quadrilateral."""
        yield self.north
        yield self.south

    @property
    def north(self) -> FieldAlignedCurve:
        """Edge of the quadrilateral passing through ``self.shape(0.)``."""
        return FieldAlignedCurve(self.nodes[0])

    @property
    def south(self) -> FieldAlignedCurve:
        """Edge of the quadrilateral passing through ``self.shape(1.)``."""
        return FieldAlignedCurve(self.nodes[-1])

    @property
    def near(self) -> Curve:
        """Cross-field edge of the quadrilateral with the smallest x3-value."""
        return Curve(self.nodes.coords.get[:, 0])

    @property
    def far(self) -> Curve:
        """Cross-field edge of the quadrilateral with the largest x3-value."""
        return Curve(self.nodes.coords.get[:, -1])

    def corners(self) -> Iterator[Coord]:
        """Return the points corresponding to the corners of the quadrilateral."""
        yield self.nodes.coords[0, 0]
        yield self.nodes.coords[-1, 0]
        yield self.nodes.coords[0, -1]
        yield self.nodes.coords[-1, -1]

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
            for data in self.nodes.subdivide(num_divisions):
                yield Quad(data)

    def make_flat_quad(self) -> Quad:
        """Create a new version of this Quad which is flat in the poloidal plane."""
        north = self.nodes.start_points[0]
        south = self.nodes.start_points[-1]
        return Quad(
            field_aligned_positions(
                straight_line_across_field(
                    north, south, len(self.nodes.start_points) - 1
                ),
                self.nodes.x3[-1] - self.nodes.x3[0],
                self.nodes.trace,
                self.nodes.alignments,
                self.nodes.order,
                self.nodes.subdivision,
                self.nodes.num_divisions,
            )
        )


@dataclass(frozen=True)
class UnalignedShape(LazilyOffsetable):
    """Represents a polygon where none of the edges are field-aligned.

    It is typically either the near or far end of a
    :class:`neso_fame.mesh.Prism`. It that case it is in the x1-x2 plane and can have
    curved edges.

    Group
    -----
    elements

    """

    shape: PrismTypes
    """Indication of the number of edges this shape has."""
    nodes: Coords
    """Control points desribing the shape. They should be arranged in
    a logically 2D array."""

    def __iter__(self) -> Iterator[Curve]:
        """Iterate over the edges of the polygon."""
        yield Curve(self.nodes.get[0, :])
        yield Curve(self.nodes.get[-1, :])
        yield Curve(self.nodes.get[:, 0])
        yield Curve(self.nodes.get[:, -1])

    def corners(self) -> Iterator[Coord]:
        """Return the points corresponding to the vertices of the polygon."""
        yield self.nodes[0, 0]
        yield self.nodes[0, -1]
        yield self.nodes[-1, 0]
        if self.shape == PrismTypes.RECTANGULAR:
            yield self.nodes[-1, -1]


class PrismTypes(Enum):
    """The different cross-sections which a Prism can have."""

    TRIANGULAR = 3
    RECTANGULAR = 4


@dataclass(frozen=True)
class Prism(LazilyOffsetable):
    r"""Representation of a triangular or rectangular (or other) prism.

    This is represented by an array of field-aligned control
    points. The starting-points (on the poloidal plane) should be
    logically arranged into a 2D array. If the prism is triangular
    then only the starting-points :math:`S[i,j] | i \le N - j + 1, S \in
    \R^{N, N}` are used.

    Group
    -----
    elements

    """

    shape: PrismTypes
    """The type of prism this is."""
    nodes: FieldAlignedPositions
    """The control points for this prism. The start-positions for this
    should be arranged in a logically-2D array."""

    def __iter__(self) -> Iterator[Quad]:
        """Iterate over the quads defining the faces of the hexahedron."""
        if self.shape == PrismTypes.TRIANGULAR:
            yield Quad(self.nodes[0, :])
            yield Quad(self.nodes[:, 0])
            yield Quad(self.nodes.diagonal)
        elif self.shape == PrismTypes.RECTANGULAR:
            yield Quad(self.nodes[0, :])
            yield Quad(self.nodes[-1, :])
            yield Quad(self.nodes[:, 0])
            yield Quad(self.nodes[:, -1])
        else:
            raise NotImplementedError(
                f"Do not know how to get sides of prism of type {self.shape}"
            )

    @cached_property
    def near(self) -> UnalignedShape:
        """The face of the prism in the x3 plane with the smallest x3 value."""
        return UnalignedShape(self.shape, self.nodes.coords.get[..., 0])

    @cached_property
    def far(self) -> UnalignedShape:
        """The face of the prism in the x3 plane with the largest x3 value."""
        return UnalignedShape(self.shape, self.nodes.coords.get[..., -1])

    def corners(self) -> Iterator[Coord]:
        """Return the points corresponding to the vertices of the hexahedron."""
        yield self.nodes.coords[0, 0, 0]
        yield self.nodes.coords[0, -1, 0]
        yield self.nodes.coords[-1, 0, 0]
        if self.shape == PrismTypes.RECTANGULAR:
            yield self.nodes.coords[-1, -1, 0]
        yield self.nodes.coords[0, 0, -1]
        yield self.nodes.coords[0, -1, -1]
        yield self.nodes.coords[-1, 0, -1]
        if self.shape == PrismTypes.RECTANGULAR:
            yield self.nodes.coords[-1, -1, -1]

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
            for data in self.nodes.subdivide(num_divisions):
                yield Prism(self.shape, data)

    @property
    def poloidal_points(self) -> SliceCoords:
        r"""Get the control points for the poloidal (x3) cross-section of this prism."""
        return self.nodes.start_points

    def make_flat_faces(self) -> Prism:
        """Create a new prism where sides don't curve in the poloidal plane."""
        s = np.linspace(0.0, 1.0, len(self.nodes.start_points))
        s1, s2 = np.meshgrid(s, s, copy=False, sparse=True)
        if self.shape == PrismTypes.TRIANGULAR:
            north = self.nodes.start_points[0, 0]
            east = self.nodes.start_points[0, -1]
            south = self.nodes.start_points[-1, 0]
            ns = (1.0 - s1) * (1.0 - s2)
            es = (1.0 - s1) * s2
            ss = s2
            starts = SliceCoords(
                north.x1 * ns + east.x1 * es + south.x1 * ss,
                north.x2 * ns + east.x2 * es + south.x2 * ss,
                self.nodes.start_points.system,
            )
        elif self.shape == PrismTypes.RECTANGULAR:
            north = self.nodes.start_points[0, 0]
            east = self.nodes.start_points[0, -1]
            south = self.nodes.start_points[-1, -1]
            west = self.nodes.start_points[-1, 0]
            ns = (1.0 - s1) * (1.0 - s2)
            es = (1.0 - s1) * s2
            ss = s1 * s2
            ws = s1 * (1.0 - s2)
            starts = SliceCoords(
                north.x1 * ns + east.x1 * es + south.x1 * ss + west.x1 * ws,
                north.x2 * ns + east.x2 * es + south.x2 * ss + west.x2 * ws,
                self.nodes.start_points.system,
            )
        else:
            raise NotImplementedError(
                f"Do no know how to flatten prism of type {self.shape}"
            )
        return Prism(
            self.shape,
            field_aligned_positions(
                starts,
                self.nodes.x3[-1] - self.nodes.x3[0],
                self.nodes.trace,
                self.nodes.alignments,
                self.nodes.order,
                self.nodes.subdivision,
                self.nodes.num_divisions,
            ),
        )


E = TypeVar("E", Quad, Prism)
B = TypeVar("B", FieldAlignedCurve, Quad)
C = TypeVar("C", Curve, UnalignedShape)


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
        bounds: Sequence[frozenset[FieldAlignedCurve]],
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
            return iter(cast(QuadMeshLayer, self))
        else:
            return itertools.chain.from_iterable(map(iter, cast(PrismMeshLayer, self)))

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
        elements: Iterable[_ElementLike | FieldAlignedCurve], subdivisions: int
    ) -> Iterator[_ElementLike | FieldAlignedCurve]:
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


QuadMeshLayer = MeshLayer[Quad, FieldAlignedCurve, Curve]
PrismMeshLayer = MeshLayer[Prism, Quad, UnalignedShape]


QuadMesh = GenericMesh[Quad, FieldAlignedCurve, Curve]
"""
Mesh made up of `Quad` elements.

Group
-----
mesh
"""
PrismMesh = GenericMesh[Prism, Quad, UnalignedShape]
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
