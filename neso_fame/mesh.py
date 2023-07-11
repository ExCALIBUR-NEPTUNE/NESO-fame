"""Classes to represent meshes and their constituent elements.

"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from enum import Enum
from functools import cache, cached_property
import itertools
from typing import (
    Callable,
    cast,
    Generic,
    Optional,
    overload,
    Protocol,
    Type,
    TypeVar,
)

import numpy as np
import numpy.typing as npt
from scipy.interpolate import interp1d


class CoordinateSystem(Enum):
    """Represent the type of coordinate system being used.

    Group
    -----
    coordinates

    """

    CARTESIAN = 0
    CYLINDRICAL = 1
    CARTESIAN2D = 2


CartesianTransform = Callable[
    [npt.NDArray, npt.NDArray, npt.NDArray],
    tuple[npt.NDArray, npt.NDArray, npt.NDArray],
]

COORDINATE_TRANSFORMS: dict[CoordinateSystem, CartesianTransform] = {
    CoordinateSystem.CARTESIAN: lambda x1, x2, x3: (x1, x2, x3),
    CoordinateSystem.CYLINDRICAL: lambda x1, x2, x3: (
        x1 * np.cos(x3),
        x1 * np.sin(x3),
        x2,
    ),
    CoordinateSystem.CARTESIAN2D: lambda x1, _, x3: (x3, -x1, np.asarray(0.0)),
}


@dataclass(frozen=True)
class SliceCoord:
    """Representation of a point in a poloidal slice (or
    analogous).

    Group
    -----
    coordinates

    """

    x1: float
    """Coordinate in first dimension"""
    x2: float
    """Coordinate in second dimension"""
    system: CoordinateSystem
    """The type of coordinates being used"""

    def __iter__(self) -> Iterator[float]:
        """Iterate over the coordinates of the point."""
        yield self.x1
        yield self.x2


@dataclass
class SliceCoords:
    """Representation of a collection of points in a poloidal slice (or
    analogous).

    Group
    -----
    coordinates

    """

    x1: npt.NDArray
    """Coordinates in first dimension"""
    x2: npt.NDArray
    """Coordinates in second dimension"""
    system: CoordinateSystem
    """The type of coordinates being used"""

    def iter_points(self) -> Iterator[SliceCoord]:
        """Iterate over the points held in this object."""
        for x1, x2 in cast(
            Iterator[tuple[float, float]],
            zip(*map(np.nditer, np.broadcast_arrays(self.x1, self.x2))),
        ):
            yield SliceCoord(x1, x2, self.system)

    def __iter__(self) -> Iterator[npt.NDArray]:
        """Iterate over the coordinates of the points."""
        for array in np.broadcast_arrays(self.x1, self.x2):
            yield array

    def __len__(self) -> int:
        """Returns the number of points contained in the object."""
        return np.broadcast(self.x1, self.x2).size

    def __getitem__(self, idx) -> SliceCoord:
        """Return an individual point from the collection."""
        x1, x2 = np.broadcast_arrays(self.x1, self.x2)
        return SliceCoord(float(x1[idx]), float(x2[idx]), self.system)


@dataclass(frozen=True)
class Coord:
    """Represents a point in 3D space.

    Group
    -----
    coordinates

    """

    x1: float
    """Coordinate in first dimension"""
    x2: float
    """Coordinate in second dimension"""
    x3: float
    """Coordinate in third dimension"""
    system: CoordinateSystem
    """The type of coordinates being used"""

    def to_cartesian(self) -> "Coord":
        """Convert the point to Cartesian coordinates."""
        x1, x2, x3 = COORDINATE_TRANSFORMS[self.system](*self)
        return Coord(
            float(x1),
            float(x2),
            float(x3),
            CoordinateSystem.CARTESIAN,
        )

    def __iter__(self) -> Iterator[float]:
        """Iterate over the individual coordinates of the point."""
        yield self.x1
        yield self.x2
        yield self.x3


@dataclass
class Coords:
    """Represents a collection of points in 3D space.

    Group
    -----
    coordinates

    """

    x1: npt.NDArray
    """Coordinates in first dimension"""
    x2: npt.NDArray
    """Coordinates in second dimension"""
    x3: npt.NDArray
    """Coordinates in third dimension"""
    system: CoordinateSystem
    """The type of coordinates being used"""

    def iter_points(self) -> Iterator[Coord]:
        """Iterate over the points held in this object."""
        for x1, x2, x3 in cast(
            Iterator[tuple[float, float, float]],
            zip(*map(np.nditer, np.broadcast_arrays(self.x1, self.x2, self.x3))),
        ):
            yield Coord(float(x1), float(x2), float(x3), self.system)

    def offset(self, dx3: npt.ArrayLike) -> "Coords":
        """Changes the x3 coordinate by the specified ammount."""
        return Coords(self.x1, self.x2, self.x3 + dx3, self.system)

    def to_cartesian(self) -> "Coords":
        """Converts the points to be in Cartesian coordiantes."""
        x1, x2, x3 = COORDINATE_TRANSFORMS[self.system](self.x1, self.x2, self.x3)
        return Coords(
            x1,
            x2,
            x3,
            CoordinateSystem.CARTESIAN,
        )

    def __iter__(self) -> Iterator[npt.NDArray]:
        """Iterate over the individual coordinates of the points."""
        for array in np.broadcast_arrays(self.x1, self.x2, self.x3):
            yield array

    def __len__(self) -> int:
        """Returns the number of poitns present in the collection."""
        return np.broadcast(self.x1, self.x2, self.x3).size

    def __getitem__(self, idx) -> Coord:
        """Returns the coordinates of an individual point."""
        x1, x2, x3 = np.broadcast_arrays(self.x1, self.x2, self.x3)
        return Coord(float(x1[idx]), float(x2[idx]), float(x3[idx]), self.system)

    def to_coord(self) -> Coord:
        """Tries to convert the object to a `Coord` object. This will
        only work if the collection contains exactly one point."""
        return Coord(float(self.x1), float(self.x2), float(self.x3), self.system)


FieldTrace = Callable[[SliceCoord, npt.ArrayLike], tuple[SliceCoords, npt.NDArray]]
"""A function describing a field line.

Group
-----
field line

Parameters
----------
start : SliceCoord
    The position of the field-line in the x1-x2 plane at x3 = 0.
locations : :obj:`numpy.typing.ArrayLike`
    x3 coordinates at which to calculate the position of the field line

Returns
-------
tuple[:class:`~neso_fame.mesh.SliceCoord`, :obj:`numpy.typing.NDArray`]
    The first element is the x1 and x2 coordinates of the field line at
    the provided x3 positions. The second is an array with the distance
    traveersed along the field line to those points.


.. rubric:: Alias

"""

NormalisedFieldLine = Callable[[npt.ArrayLike], Coords]
"""A function describing a segment of a field line.

Parameters
----------
s : :obj:`numpy.typing.ArrayLike`
    An argument between 0 and 1, where 0 corresponds to the start of the
    field line and 1 to the end.

Returns
-------
Coords
    The locations on the field line. The distance of the point from the
    start of the field line is directly proportional to ``s``.

Group
-----
field line


.. rubric:: Alias
"""

T = TypeVar("T")


class _ElementLike(Protocol):
    """Protocal defining the methods that can be used to manipulate
    mesh components. Exists for internal type-checking.

    """

    def offset(self: T, offset: float) -> T:
        ...

    def subdivide(self: T, num_divisions: int) -> Iterator[T]:
        ...


@dataclass(frozen=True)
class Curve:
    """Represents a curve in 3D space.

    A curve is defined by a function which takes a single argument, 0
    <= s <= 1, and returns coordinates for the location on that curve
    in space. The distance along the curve from the start to the
    position represented by s is directly proportional to s.

    Group
    -----
    elements

    """

    function: NormalisedFieldLine
    """The function defining the shape of this curve"""

    def __call__(self, s: npt.ArrayLike) -> Coords:
        """Convenience function so that a Curve is itself a
        :obj:`~neso_fame.mesh.NormalisedFieldLine`"""
        return self.function(s)

    def offset(self, offset: float) -> "Curve":
        """Returns a new `Curve` object which is identical except
        offset by the specified ammount in the x3-direction.
        """
        return Curve(lambda s: self.function(s).offset(offset))

    def subdivide(self, num_divisions: int) -> Iterator[Curve]:
        """Returns an iterator of curves created by splitting this one
        up into equal-length segments.
        """

        def subdivision(
            func: NormalisedFieldLine, i: int, divs: int
        ) -> NormalisedFieldLine:
            return lambda s: func((i + np.asarray(s)) / divs)

        if num_divisions <= 1:
            yield self
        else:
            for i in range(num_divisions):
                yield Curve(subdivision(self.function, i, num_divisions))

    @cache
    def control_points(self, order) -> Coords:
        """Returns a set of locations on the line which can be used to
        represent it to the specified order of accuracy. These points
        will be equally spaced along the curve.
        """
        s = np.linspace(0.0, 1.0, order + 1)
        return self.function(s)


def _line_from_points(north: Coord, south: Coord) -> NormalisedFieldLine:
    """Creates a function representing a straight line between the two
    specified points.
    """

    def _line(s: npt.ArrayLike) -> Coords:
        s = np.asarray(s)
        return Coords(
            north.x1 + (south.x1 - north.x1) * s,
            north.x2 + (south.x2 - north.x2) * s,
            north.x3 + (south.x3 - north.x3) * s,
            north.system,
        )

    return _line


@dataclass(frozen=True)
class Quad:
    """Representation of a four-sided polygon (quadrilateral).

    This is represented by two curves representing opposite edges. The
    remaining edges are created by connecting the corresponding
    termini of the bounding lines. It also contains information on the
    magnetic field along which the curves defining the figure were
    traced.

    Note
    ----
    There is an optional attribute which is meant to define how the
    quadrilateral may curve into a third dimension, but this has not
    yet been used in the implementation. It probably will not be
    sufficiently general to be useful, in any case.

    Group
    -----
    elements

    """

    north: Curve
    """Curve defining one edge of the quadrilateral"""
    south: Curve
    """Curve defining the other edge of the quadrilateral"""
    in_plane: Optional[
        Curve
    ]  # FIXME: Don't think this is adequate to describe curved quads
    field: FieldTrace
    """The underlying magnetic field to which the quadrilateral is aligned"""

    def __iter__(self) -> Iterator[Curve]:
        """Iterate over the two curves defining the edges of the quadrilateral."""
        yield self.north
        yield self.south

    @cached_property
    def near(self) -> Curve:
        """Returns a curve connecting the starting points (s=0) of the
        curves defining the boundaries of the quadrilateral.

        """
        return Curve(
            _line_from_points(self.north(0.0).to_coord(), self.south(0.0).to_coord())
        )

    @cached_property
    def far(self) -> Curve:
        """Returns a curve connecting the end-points (s=1) of the
        curves defining the boundaries of the quadrilateral.

        """
        return Curve(
            _line_from_points(self.north(1.0).to_coord(), self.south(1.0).to_coord())
        )

    @classmethod
    @cache
    def _cached_quad(
        cls,
        north: Curve,
        south: Curve,
        in_plane: Optional[Curve],
        field: FieldTrace,
    ) -> Quad:
        return cls(north, south, in_plane, field)

    # FIXME: I don't think this is actually useful. Do I actually need
    # things to be cached? Think this is a holdover from when I was
    # filtering elements that fell outside the domain. Would be more
    # convenient to know something about the order of the curve though
    # (at least in the 2D case).
    @classmethod
    def from_unordered_curves(
        cls,
        curve1: Curve,
        curve2: Curve,
        in_plane: Optional[Curve],
        field: FieldTrace,
    ) -> Quad:
        """Returns the same quad object, regardless of the order the
        curve1 and curve2 arguments."""
        if hash(curve1) < hash(curve2):
            return cls._cached_quad(curve1, curve2, in_plane, field)
        return cls._cached_quad(curve2, curve1, in_plane, field)

    def corners(self) -> Coords:
        """Returns the points corresponding to the corners of the quadrilateral."""
        north_corners = self.north.control_points(1)
        south_corners = self.south.control_points(1)
        return Coords(
            np.concatenate([north_corners.x1, south_corners.x1]),
            np.concatenate([north_corners.x2, south_corners.x2]),
            np.concatenate([north_corners.x3, south_corners.x3]),
            north_corners.system,
        )

    def control_points(self, order) -> Coords:
        """Returns the coordinates of the control points for the
        surface, in an array of dimensions [3, order + 1, order + 1].

        """
        if self.in_plane is None:
            # FIXME: Is this strictly correct?
            # FIXME: can I be confident that the start and end will be exactly
            #        as specified?
            north_samples = self.north.control_points(order)
            south_samples = self.south.control_points(order)
            return Coords(
                np.linspace(north_samples.x1, south_samples.x1, order + 1),
                np.linspace(north_samples.x2, south_samples.x2, order + 1),
                np.linspace(north_samples.x3, south_samples.x3, order + 1),
                north_samples.system,
            )
        raise NotImplementedError(
            "Can not yet handle Quads where all four edges are curved"
        )

    def offset(self, offset: float) -> Quad:
        """Returns a quad which is identical except that it is shifted
        by the specified offset in the x3 direction.

        """
        return Quad(
            self.north.offset(offset),
            self.south.offset(offset),
            self.in_plane.offset(offset) if self.in_plane is not None else None,
            self.field,
        )

    def subdivide(self, num_divisions: int) -> Iterator[Quad]:
        """Returns an iterator of quad objects produced by splitting
        the bounding-linse of this quad into the specified number of
        equally-sized segments. This has the effect of splitting the
        quad equally in the x3 direction.

        """
        if num_divisions <= 1:
            yield self
        else:
            for n, s in zip(
                self.north.subdivide(num_divisions), self.south.subdivide(num_divisions)
            ):
                yield Quad(
                    n,
                    s,
                    self.in_plane.offset(cast(float, 0.5 * (n(0.0).x3 + s(0.0).x3)))
                    if self.in_plane is not None
                    else None,
                    self.field,
                )


@dataclass(frozen=True)
class Hex:
    """Representation of a six-sided solid (hexahedron).

    This is represented by four quads making up its faces. The
    remaining two faces are made up of the edges of these quads at s=0
    and s=1 and are normal to the x3-direction.

    Caution
    -------
    This requires more extensive testing.

    Group
    -----
    elements

    """

    north: Quad
    """A shape defining one edge of the hexahedron"""
    south: Quad
    """A shape defining one edge of the hexahedron"""
    east: Quad
    """A shape defining one edge of the hexahedron"""
    west: Quad
    """A shape defining one edge of the hexahedron"""

    def __iter__(self) -> Iterator[Quad]:
        """Iterate over the four quads defining the faces of the hexahedron."""
        yield self.north
        yield self.east
        yield self.south
        yield self.west

    @cached_property
    def near(self) -> Quad:
        """Returns a quad made from the near edges of the
        quadrilaterals defining this hexahedron. This corresponds to a
        face of the hexahedron normal to x3.

        """
        north = Curve(
            _line_from_points(
                self.north.north(0.0).to_coord(), self.north.south(0.0).to_coord()
            )
        )
        south = Curve(
            _line_from_points(
                self.south.north(0.0).to_coord(), self.south.south(0.0).to_coord()
            )
        )
        return Quad(north, south, None, self.north.field)

    @cached_property
    def far(self) -> Quad:
        """Returns a quad made from the far edges of the
        quadrilaterals defining this hexahedron. This corresponds to a
        face of the hexahedron normal to x3.

        """
        north = Curve(
            _line_from_points(
                self.north.north(1.0).to_coord(), self.north.south(1.0).to_coord()
            )
        )
        south = Curve(
            _line_from_points(
                self.south.north(1.0).to_coord(), self.south.south(1.0).to_coord()
            )
        )
        return Quad(north, south, None, self.north.field)

    def corners(self) -> Coords:
        """Returns the points corresponding to the vertices of the
        quadrilateral.

        """
        north_corners = self.north.corners()
        south_corners = self.south.corners()
        # TODO Check that east and west corners are the same as north and south
        return Coords(
            np.concatenate([north_corners.x1, south_corners.x1]),
            np.concatenate([north_corners.x2, south_corners.x2]),
            np.concatenate([north_corners.x3, south_corners.x3]),
            north_corners.system,
        )

    def offset(self, offset: float) -> Hex:
        """Returns a hex which is identical except that it is shifted
        by the specified offset in the x3 direction.

        """
        return Hex(
            self.north.offset(offset),
            self.south.offset(offset),
            self.east.offset(offset),
            self.west.offset(offset),
        )

    def subdivide(self, num_divisions: int) -> Iterator[Hex]:
        """Returns an iterator of hex objects produced by splitting
        the bounding-quads of this hex into the specified number of
        equally-sized parts. This has the effect of splitting the
        hex equally in the x3 direction.

        """
        if num_divisions <= 0:
            yield self
        else:
            for n, s, e, w in zip(
                self.north.subdivide(num_divisions),
                self.south.subdivide(num_divisions),
                self.east.subdivide(num_divisions),
                self.west.subdivide(num_divisions),
            ):
                yield Hex(n, s, e, w)


E = TypeVar("E", Quad, Hex)
B = TypeVar("B", Curve, Quad)


@dataclass(frozen=True)
class MeshLayer(Generic[E, B]):
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
    :class:`~neso_fame.mesh.Hex` elements making up the layer (without
    any offset or subdivision)."""
    bounds: Sequence[frozenset[B]]
    """An ordered collection of sets of :class:`~neso_fame.mesh.Curve`
    or :class:`~neso_fame.mesh.Quad` objects (faces or edges,
    repsectively). Each set describes a particular boundary regions of
    the layer. The near and far faces of the layer are not included in
    these."""
    offset: Optional[float] = None
    """The ammount by which the x3-coordinate of the elements and
    boundaries should be changed from those in
    :data:`~neso_fame.mesh.MeshLayer.reference_elements`."""
    subdivisions: int = 1
    """The number of elements deep the layer should be in the
    x3-direction."""

    def __iter__(self) -> Iterator[E]:
        """Iterate over all of hte elements (`Quad` or `Hex` objects)
        which make up this layer of the mesh.

        """
        return self._iterate_elements(
            self.reference_elements, self.offset, self.subdivisions
        )

    def __len__(self) -> int:
        """Returns the number of elements in this layer."""
        return len(self.reference_elements) * self.subdivisions

    @cached_property
    def element_type(self) -> Type[E]:
        """Returns the type object for the elements of the mesh layer."""
        return type(next(iter(self.reference_elements)))

    def quads(self) -> Iterator[Quad]:
        """Iterates over teh `Quad` objects in the mesh. If the mesh
        is made up of quads then this is the same as iterating over
        the elements. Otherwise, it iterates over the quads defining
        the boundaries of the constituent `Hex` elements.

        """
        if len(self.reference_elements) > 0 and issubclass(self.element_type, Quad):
            return iter(self)
        else:
            return itertools.chain.from_iterable(
                map(iter, cast(MeshLayer[Hex, Quad], self))
            )

    def boundaries(self) -> Iterator[frozenset[B]]:
        """Iterates over the boundary regions in this layer. This
        excludes boundaries normal to the x3-direction. There may be
        any number of boundary regions. If the mesh is made up of
        `Quad` elements then the boundaries are sets of `Curve`
        objects. If the mesh is made up of `Hex` elements, then the
        boundaries are sets of `Quad` objects.

        """
        return map(
            frozenset,
            map(
                lambda b: self._iterate_elements(b, self.offset, self.subdivisions),
                self.bounds,
            ),
        )

    def near_faces(self) -> Iterator[B]:
        """Iterates over the near faces of the elements in the
        layer. If the layer is subdivided (i.e., is more than one
        element deep in the x3-direction) then only the near faces of
        the first subdivision will be returned. This constitutes one
        of the boundaries normal to the x3-direction.

        """
        return map(
            lambda e: cast(B, e.near),
            self._iterate_elements(self.reference_elements, self.offset, 1),
        )

    def far_faces(self) -> Iterator[B]:
        """Iterates over the far faces of the elements in the
        layer. If the layer is subdivided (i.e., is more than one
        element deep in the x3-direction) then only the far faces of
        the last subdivision will be returned. This constitutes one
        of the boundaries normal to the x3-direction.

        Note
        ----
        This won't necessarily be bit-wise identical to the last
        subdivision. However, as round coordinates to 8 decimal places
        when creating Nektar++ objects, it won't matter.

        """
        return map(
            lambda e: cast(B, e.far),
            self._iterate_elements(self.reference_elements, self.offset, 1),
        )

    @overload
    @staticmethod
    def _iterate_elements(
        elements: Iterable[E], offset: Optional[float], subdivisions: int
    ) -> Iterator[E]:
        ...

    @overload
    @staticmethod
    def _iterate_elements(
        elements: Iterable[B], offset: Optional[float], subdivisions: int
    ) -> Iterator[B]:
        ...

    @staticmethod
    def _iterate_elements(
        elements: Iterable[_ElementLike], offset: Optional[float], subdivisions: int
    ) -> Iterator[_ElementLike]:
        """Convenience method used by other iteration methods. It
        handles offsets and subdivisions appropriately.

        """
        if isinstance(offset, float):
            x = offset
            return itertools.chain.from_iterable(
                map(
                    lambda e: e.offset(x).subdivide(subdivisions),  # typing: off
                    elements,
                )
            )
        else:
            return itertools.chain.from_iterable(
                map(lambda e: e.subdivide(subdivisions), elements)
            )


@dataclass(frozen=True)
class GenericMesh(Generic[E, B]):
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
    :class:`~neso_fame.mesh.Hex` elements and
    :class:`~neso_fame.mesh.Quad` boundaries. GenericMesh should not
    be used for type annotations; use :obj:`~neso_fame.mesh.QuadMesh`,
    :obj:`~neso_fame.mesh.HexMesh`, or :obj:`~neso_fame.mesh.Mesh`
    instead, as these are constrained to the valid combinations.

    Group
    -----
    mesh

    """

    reference_layer: MeshLayer[E, B]
    """A layer from which all of the constituant layers of the mesh
    object will be produced."""
    offsets: npt.NDArray
    """The x3 offset for each layer of the mesh."""

    def layers(self) -> Iterable[MeshLayer[E, B]]:
        """Iterate through the `MeshLayer` objects which make up this
        mesh.

        """
        return map(
            lambda off: MeshLayer(
                self.reference_layer.reference_elements,
                self.reference_layer.bounds,
                off,
                self.reference_layer.subdivisions,
            ),
            self.offsets,
        )

    def __iter__(self) -> Iterator[E]:
        """Iterate through all of the elements contained in this mesh."""
        return itertools.chain.from_iterable(map(iter, self.layers()))

    def __len__(self) -> int:
        """Returns the number of elements in this mesh."""
        return len(self.reference_layer) * self.offsets.size


QuadMesh = GenericMesh[Quad, Curve]
"""
Mesh made up of `Quad` elements.

Group
-----
mesh
"""
HexMesh = GenericMesh[Hex, Quad]
"""
Mesh made up of `Hex` elements.

Group
-----
mesh


.. rubric:: Alias

"""
Mesh = QuadMesh | HexMesh
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
    resolution=10,
) -> NormalisedFieldLine:
    """Takes a function defining a magnetic field and returns a new
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
    :obj:`~neso_fame.mesh.NormalisedFieldLine`
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
    interp = interp1d((s - s[0]) / (s[-1] - s[0]), coordinates, order)
    coord_system = start.system

    def normalised_interpolator(s: npt.ArrayLike) -> Coords:
        locations = interp(s)
        return Coords(locations[0], locations[1], locations[2], coord_system)

    return normalised_interpolator
