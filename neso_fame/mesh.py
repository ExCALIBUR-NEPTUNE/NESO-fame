"""Classes to represent meshes and their constituent elements."""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Context, Decimal
from enum import Enum
from functools import cache, cached_property
from typing import (
    Callable,
    ClassVar,
    Generic,
    Literal,
    Protocol,
    Type,
    TypeVar,
    cast,
    overload,
)

import numpy as np
import numpy.typing as npt
from scipy.interpolate import interp1d

from neso_fame.offset import LazilyOffsetable, Offset


class CoordinateSystem(Enum):
    """Represent the type of coordinate system being used.

    Group
    -----
    coordinates

    """

    CARTESIAN = 0
    CYLINDRICAL = 1
    CARTESIAN2D = 2
    CARTESIAN_ROTATED = 3


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
    CoordinateSystem.CARTESIAN2D: lambda x1, _, x3: (-x3, x1, np.asarray(0.0)),
    CoordinateSystem.CARTESIAN_ROTATED: lambda x1, x2, x3: (-x3, x1, x2),
}


@np.vectorize
def _round_to_sig_figs(x: float, figures: int) -> float:
    """Round floats to the given number of significant figures."""
    abs_x = np.abs(x)
    if abs_x < 10.0**-figures:
        return 0.0
    return float(Decimal(str(x)).normalize(Context(int(figures), ROUND_HALF_UP)))


@dataclass(frozen=True)
class SliceCoord:
    """Representation of a point in a poloidal slice (or analogous).

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
    TOLERANCE: ClassVar[float] = 1e-9
    """The absolute and relative tolerance to use when comparing Coord objects."""

    def __iter__(self) -> Iterator[float]:
        """Iterate over the coordinates of the point."""
        yield self.x1
        yield self.x2

    def round_to(self, figures: int = 8) -> SliceCoord:
        """Round coordinate values to the desired number of significant figures."""
        return SliceCoord(
            float(_round_to_sig_figs(self.x1, figures)),
            float(_round_to_sig_figs(self.x2, figures)),
            self.system,
        )

    def __hash__(self) -> int:
        """Hash the SliceCoord object.

        Hashing is done so that floats are the same to the number of
        significant figures set by TOLERANCE will have the same hash
        value.
        """
        decimal_places = -int(np.floor(np.log10(self.TOLERANCE))) - 1
        context = Context(decimal_places)

        # FIXME: This can still result in hashes being different for
        # two similar numbers, if rounding would affect multiple
        # decimal places (e.g., rounding .999995 up to 1.). The
        # likelihood is low but non-zero.
        def get_digits(
            x: float,
        ) -> tuple[int, tuple[int, ...], int | Literal["n", "N", "F"]]:
            y = Decimal(x).normalize(context).as_tuple()
            spare_places = decimal_places - len(y[1])
            if isinstance(y[2], int) and len(y[1]) + y[2] < -8:
                return 0, (), 0
            truncated = (y[1] + (0,) * spare_places)[:-3]
            if all(t == 0 for t in truncated):
                return 0, (), 0
            exponent = y[2]
            if isinstance(exponent, int):
                exponent -= spare_places - 3
            return y[0], truncated, exponent

        x1 = get_digits(self.x1)
        x2 = get_digits(self.x2)
        return hash((x1, x2, self.system))

    def to_3d_coord(self, x3: float) -> Coord:
        """Create a 3D coordinate object from this 2D one."""
        return Coord(self.x1, self.x2, x3, self.system)

    def __eq__(self, other: object) -> bool:
        """Check equality of coordinates within the the TOLERANCE."""
        if not isinstance(other, self.__class__):
            return False
        return self.system == other.system and cast(
            bool,
            np.isclose(self.x1, other.x1, self.TOLERANCE, self.TOLERANCE)
            and np.isclose(self.x2, other.x2, self.TOLERANCE, self.TOLERANCE),
        )


Index = int | tuple[int, ...]


@dataclass
class SliceCoords:
    """Representation of a collection of points in an x1-x2 plane.

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
        for x1, x2 in zip(*map(np.nditer, np.broadcast_arrays(self.x1, self.x2))):
            yield SliceCoord(float(x1), float(x2), self.system)

    def __iter__(self) -> Iterator[npt.NDArray]:
        """Iterate over the coordinates of the points."""
        for array in np.broadcast_arrays(self.x1, self.x2):
            yield array

    def __len__(self) -> int:
        """Return the number of points contained in the object."""
        return np.broadcast(self.x1, self.x2).size

    def __getitem__(self, idx: Index) -> SliceCoord:
        """Return an individual point from the collection."""
        x1, x2 = np.broadcast_arrays(self.x1, self.x2)
        return SliceCoord(float(x1[idx]), float(x2[idx]), self.system)

    def round_to(self, figures: int = 8) -> SliceCoords:
        """Round coordinate values to the desired number of significant figures."""
        return SliceCoords(
            _round_to_sig_figs(self.x1, figures),
            _round_to_sig_figs(self.x2, figures),
            self.system,
        )

    def to_coord(self) -> SliceCoord:
        """Convert the object to a `SliceCoord` object.

        This will only work if the collection contains exactly one
        point. Otherwise, an exception is raised.
        """
        return SliceCoord(float(self.x1), float(self.x2), self.system)

    def to_3d_coords(self, x3: float) -> Coords:
        """Create a 3D coordinates object from this 2D one."""
        return Coords(self.x1, self.x2, np.asarray(x3), self.system)


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
    TOLERANCE: ClassVar[float] = 1e-9
    """The absolute and relative tolerance to use when comparing Coord objects."""

    def to_cartesian(self) -> "Coord":
        """Convert the point to Cartesian coordinates."""
        x1, x2, x3 = COORDINATE_TRANSFORMS[self.system](*self)
        return Coord(
            float(x1),
            float(x2),
            float(x3),
            CoordinateSystem.CARTESIAN,
        )

    def offset(self, dx3: float) -> "Coord":
        """Change the x3 coordinate by the specified ammount."""
        return Coord(self.x1, self.x2, self.x3 + dx3, self.system)

    def __iter__(self) -> Iterator[float]:
        """Iterate over the individual coordinates of the point."""
        yield self.x1
        yield self.x2
        yield self.x3

    def round_to(self, figures: int = 8) -> Coord:
        """Round coordinate values to the desired number of significant figures."""
        return Coord(
            float(_round_to_sig_figs(self.x1, figures)),
            float(_round_to_sig_figs(self.x2, figures)),
            float(_round_to_sig_figs(self.x3, figures)),
            self.system,
        )

    def __hash__(self) -> int:
        """Hash the Coord object.

        Hashing is done so that floats are the same to the number of
        significant figures set by TOLERANCE will have the same hash
        value.
        """
        decimal_places = -int(np.floor(np.log10(self.TOLERANCE))) - 1
        context = Context(decimal_places)

        # FIXME: This can still result in hashes being different for
        # two similar numbers, if rounding would affect multiple
        # decimal places (e.g., rounding .999995 up to 1.). The
        # likelihood is low but non-zero.
        def get_digits(
            x: float,
        ) -> tuple[int, tuple[int, ...], int | Literal["n", "N", "F"]]:
            y = Decimal(x).normalize(context).as_tuple()
            spare_places = decimal_places - len(y[1])
            if isinstance(y[2], int) and len(y[1]) + y[2] < -8:
                return 0, (), 0
            truncated = (y[1] + (0,) * spare_places)[:-3]
            if all(t == 0 for t in truncated):
                return 0, (), 0
            exponent = y[2]
            if isinstance(exponent, int):
                exponent -= spare_places - 3
            return y[0], truncated, exponent

        x1 = get_digits(self.x1)
        x2 = get_digits(self.x2)
        x3 = get_digits(self.x3)

        return hash((x1, x2, x3, self.system))

    def __eq__(self, other: object) -> bool:
        """Check equality of coordinates within the the TOLERANCE."""
        if not isinstance(other, self.__class__):
            return False
        return self.system == other.system and cast(
            bool,
            np.isclose(self.x1, other.x1, self.TOLERANCE, self.TOLERANCE)
            and np.isclose(self.x2, other.x2, self.TOLERANCE, self.TOLERANCE)
            and np.isclose(self.x3, other.x3, self.TOLERANCE, self.TOLERANCE),
        )


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
        for x1, x2, x3 in zip(
            *map(np.nditer, np.broadcast_arrays(self.x1, self.x2, self.x3))
        ):
            yield Coord(float(x1), float(x2), float(x3), self.system)

    def offset(self, dx3: npt.ArrayLike) -> "Coords":
        """Change the x3 coordinate by the specified ammount."""
        return Coords(self.x1, self.x2, self.x3 + dx3, self.system)

    def to_cartesian(self) -> "Coords":
        """Convert the points to be in Cartesian coordiantes."""
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
        """Return the number of points present in the collection."""
        return np.broadcast(self.x1, self.x2, self.x3).size

    def __getitem__(self, idx: Index) -> Coord:
        """Return the coordinates of an individual point."""
        x1, x2, x3 = np.broadcast_arrays(self.x1, self.x2, self.x3)
        return Coord(float(x1[idx]), float(x2[idx]), float(x3[idx]), self.system)

    def to_coord(self) -> Coord:
        """Convert the object to a `Coord` object.

        This will only work if the collection contains exactly one
        point. Otherwise, an exception is raised.
        """
        return Coord(float(self.x1), float(self.x2), float(self.x3), self.system)

    def round_to(self, figures: int = 8) -> Coords:
        """Round coordinate values to the desired number of significant figures."""
        return Coords(
            _round_to_sig_figs(self.x1, figures),
            _round_to_sig_figs(self.x2, figures),
            _round_to_sig_figs(self.x3, figures),
            self.system,
        )


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

NormalisedCurve = Callable[[npt.ArrayLike], Coords]
"""A function describing a segment of a curve. Often this curve
represents a field line.

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

    def subdivide(self: T, num_divisions: int) -> Iterator[T]:
        ...


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

    @cached_property
    def function(self) -> NormalisedCurve:
        """Return the function representing this curve."""
        return self.field.get_normalised_subdivision(
            self.start,
            -0.5 * self.dx3,
            0.5 * self.dx3,
            self.subdivision,
            self.num_divisions,
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
                )


@overload
def control_points(element: NormalisedCurve | Quad | EndQuad, order: int) -> Coords:
    ...


@overload
def control_points(element: AcrossFieldCurve, order: int) -> SliceCoords:
    ...


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

    It is a :obj:`~neso_fame.mesh.AcrossFieldCurve`.

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

    def __call__(self, s: npt.ArrayLike) -> Coords:
        """Calculate a position on the curve."""
        s = np.asarray(s)
        return Coords(
            self.north.x1 + (self.south.x1 - self.north.x1) * s,
            self.north.x2 + (self.south.x2 - self.north.x2) * s,
            self.north.x3 + (self.south.x3 - self.north.x3) * s,
            self.north.system,
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

    def __iter__(self) -> Iterator[FieldAlignedCurve]:
        """Iterate over the two curves defining the edges of the quadrilateral."""
        yield self.north
        yield self.south

    def get_field_line(self, s: float) -> FieldAlignedCurve:
        """Get the field lign passing through location ``s`` of `Quad.shape`."""
        start = self.shape(s).to_coord()
        return FieldAlignedCurve(
            self.field,
            SliceCoord(start.x1, start.x2, start.system),
            self.dx3,
            self.subdivision,
            self.num_divisions,
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
        x3_coord = np.full(self.field.resolution, x3)
        for i, start in enumerate(self.shape(s).iter_points()):
            # FIXME: This is duplicating work in the case of the actual edges.
            coord = self.field.trace(start, x3)[0].to_coord()
            x1_coord[i] = coord.x1
            x2_coord[i] = coord.x2
        coordinates = np.stack([x1_coord, x2_coord, x3_coord])
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
                locations[2],
                coord_system,
            )

        return normalised_interpolator

    @cached_property
    def north(self) -> FieldAlignedCurve:
        """Edge of the quadrilateral passing through ``self.shape(0.)``."""
        return self.get_field_line(0.0)

    @cached_property
    def south(self) -> FieldAlignedCurve:
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
                )


@dataclass(frozen=True)
class EndQuad(LazilyOffsetable):
    """Represents a quad in an x3 plane.

    It is either the near or far end of a
    :class:`neso_fame.mesh.Hex`. It is in the x1-x2 plane and can have
    four curved edges. However, it is always flat.

    Group
    -----
    elements

    """

    north: NormalisedCurve
    """A shape defining one edge of the quad"""
    south: NormalisedCurve
    """A shape defining one edge of the quad"""
    east: NormalisedCurve
    """A shape defining one edge of the quad"""
    west: NormalisedCurve
    """A shape defining one edge of the quad"""

    def __iter__(self) -> Iterator[NormalisedCurve]:
        """Iterate over the four edges of the quad."""
        yield self.north
        yield self.east
        yield self.south
        yield self.west

    def corners(self) -> Coords:
        """Return the points corresponding to the vertices of the quadrilateral."""
        north_corners = control_points(self.north, 1)
        south_corners = control_points(self.south, 1)
        return Coords(
            np.concatenate([north_corners.x1, south_corners.x1]),
            np.concatenate([north_corners.x2, south_corners.x2]),
            np.concatenate([north_corners.x3, south_corners.x3]),
            north_corners.system,
        )


@dataclass(frozen=True)
class Hex(LazilyOffsetable):
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
    def near(self) -> EndQuad:
        """The face of the Hex in the x3 plane with the smallest x3 value."""
        return EndQuad(self.north.near, self.south.near, self.east.near, self.west.near)

    @cached_property
    def far(self) -> EndQuad:
        """The face of the Hex in the x3 plane with the largest x3 value."""
        return EndQuad(self.north.far, self.south.far, self.east.far, self.west.far)

    def corners(self) -> Coords:
        """Return the points corresponding to the vertices of the hexahedron."""
        north_corners = self.north.corners()
        south_corners = self.south.corners()
        # TODO Check that east and west corners are the same as north and south
        return Coords(
            np.concatenate([north_corners.x1, south_corners.x1]),
            np.concatenate([north_corners.x2, south_corners.x2]),
            np.concatenate([north_corners.x3, south_corners.x3]),
            north_corners.system,
        )

    def subdivide(self, num_divisions: int) -> Iterator[Hex]:
        """Split the hex into the specified number of pieces.

        Returns an iterator of hex objects produced by splitting
        the bounding-quads of this hex into the specified number of
        equally-sized parts. This has the effect of splitting the
        hex equally in the x3 direction.

        """
        if num_divisions <= 1:
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
B = TypeVar("B", FieldAlignedCurve, Quad)
C = TypeVar("C", NormalisedCurve, EndQuad)


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
    :class:`~neso_fame.mesh.Hex` elements making up the layer (without
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
        quads defining the boundaries of the constituent `Hex`
        elements.

        """
        if len(self.reference_elements) > 0 and issubclass(self.element_type, Quad):
            return iter(self)
        else:
            return itertools.chain.from_iterable(
                map(iter, cast(MeshLayer[Hex, Quad, EndQuad], self))
            )

    def boundaries(self) -> Iterator[frozenset[B]]:
        """Iterate over the boundary regions in this layer.

        This excludes boundaries normal to the x3-direction. There may
        be any number of boundary regions. If the mesh is made up of
        `Quad` elements then the boundaries are sets of `Curve`
        objects. If the mesh is made up of `Hex` elements, then the
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
    def _iterate_elements(elements: Iterable[E], subdivisions: int) -> Iterator[E]:
        ...

    @overload
    @staticmethod
    def _iterate_elements(elements: Iterable[B], subdivisions: int) -> Iterator[B]:
        ...

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
    :class:`~neso_fame.mesh.Hex` elements and
    :class:`~neso_fame.mesh.Quad` boundaries. GenericMesh should not
    be used for type annotations; use :obj:`~neso_fame.mesh.QuadMesh`,
    :obj:`~neso_fame.mesh.HexMesh`, or :obj:`~neso_fame.mesh.Mesh`
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


QuadMeshLayer = MeshLayer[Quad, FieldAlignedCurve, NormalisedCurve]
HexMeshLayer = MeshLayer[Hex, Quad, EndQuad]


QuadMesh = GenericMesh[Quad, FieldAlignedCurve, NormalisedCurve]
"""
Mesh made up of `Quad` elements.

Group
-----
mesh
"""
HexMesh = GenericMesh[Hex, Quad, EndQuad]
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

    @cache
    def _normalise_and_subdivide(
        self, start: SliceCoord, x3_min: float, x3_max: float, divisions: int
    ) -> list[NormalisedCurve]:
        x3 = np.linspace(x3_min, x3_max, (self.resolution - 1) * divisions + 1)
        x1_x2_coords: SliceCoords
        s: npt.NDArray
        x1_x2_coords, s = self.trace(start, x3)
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

        Returns
        -------
        :obj:`~neso_fame.mesh.NormalisedCurve`
            The segment of the field line passing through `start` which is
            within the specified subdivision of the quad.

        """
        assert (
            division < total_divisions
        ), f"Can not request division {division} when only {total_divisions} available"
        assert total_divisions > 0, "Number of divisions must be positive"
        assert division >= 0, "Division number must be non-negative"
        segments = self._normalise_and_subdivide(start, x3_min, x3_max, total_divisions)
        return segments[division]
