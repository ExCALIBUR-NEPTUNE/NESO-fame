from __future__ import annotations

from collections.abc import Iterator, Sequence, Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from functools import cache, cached_property
import itertools
from typing import (
    Any,
    cast,
    Callable,
    ClassVar,
    Generic,
    Literal,
    Optional,
    Type,
    TypeVar,
    overload,
    Protocol,
)

import numpy as np
import numpy.typing as npt
from scipy.interpolate import interp1d, lagrange


CoordTriple = tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]


def asarrays(coords: CoordTriple) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    return np.asarray(coords[0]), np.asarray(coords[1]), np.asarray(coords[2])


class CoordinateSystem(Enum):
    """Represent the type of coordinate system being used.
    """
    Cartesian = 0
    Cylindrical = 1
    Cartesian2D = 2


CartesianTransform = Callable[
    [npt.NDArray, npt.NDArray, npt.NDArray],
    tuple[npt.NDArray, npt.NDArray, npt.NDArray],
]

COORDINATE_TRANSFORMS: dict[CoordinateSystem, CartesianTransform] = {
    CoordinateSystem.Cartesian: lambda x1, x2, x3: (x1, x2, x3),
    CoordinateSystem.Cylindrical: lambda x1, x2, x3: (
        x1 * np.cos(x3),
        x1 * np.sin(x3),
        x2,
    ),
    CoordinateSystem.Cartesian2D: lambda x1, _, x3: (x3, -x1, 0.0),
}


@dataclass(frozen=True)
class SliceCoord:
    """Representation of a point in a poloidal slice (or
    analogous).

    """
    x1: float
    x2: float
    system: CoordinateSystem

    def __iter__(self) -> Iterator[float]:
        """Iterate over the coordinates of the point.

        """
        yield self.x1
        yield self.x2


@dataclass
class SliceCoords:
    """Representation of a collection of points in a poloidal slice (or
    analogous).

    """
    x1: npt.NDArray
    x2: npt.NDArray
    system: CoordinateSystem

    def iter_points(self) -> Iterator[SliceCoord]:
        """Iterate over the points held in this object.
        """
        for x1, x2 in cast(
            Iterator[tuple[float, float]],
            zip(*map(np.nditer, np.broadcast_arrays(self.x1, self.x2))),
        ):
            yield SliceCoord(x1, x2, self.system)

    def __iter__(self) -> Iterator[npt.NDArray]:
        """Iterate over the coordinates of the points.

        """
        for array in np.broadcast_arrays(self.x1, self.x2):
            yield array

    def __len__(self) -> int:
        """Returns the number of points contained in the object.

        """
        return np.broadcast(self.x1, self.x2).size

    def __getitem__(self, idx) -> SliceCoord:
        """Return an individual point from the collection.
        
        """
        x1, x2 = np.broadcast_arrays(self.x1, self.x2)
        return SliceCoord(float(x1[idx]), float(x2[idx]), self.system)


@dataclass(frozen=True)
class Coord:
    """Represents a point in 3D space.
    
    """
    x1: float
    x2: float
    x3: float
    system: CoordinateSystem

    def to_cartesian(self) -> "Coord":
        """Convert the point to Cartesian coordinates.

        """
        x1, x2, x3 = COORDINATE_TRANSFORMS[self.system](*self)
        return Coord(
            x1,
            x2,
            x3,
            CoordinateSystem.Cartesian,
        )

    def __iter__(self) -> Iterator[float]:
        """Iterate over the individual coordinates of the point.

        """
        yield self.x1
        yield self.x2
        yield self.x3


@dataclass
class Coords:
    """Represents a collection of points in 3D space.
    """
    x1: npt.NDArray
    x2: npt.NDArray
    x3: npt.NDArray
    system: CoordinateSystem

    def iter_points(self) -> Iterator[Coord]:
        """Iterate over the points held in this object.
        """
        for x1, x2, x3 in cast(
            Iterator[tuple[float, float, float]],
            zip(*map(np.nditer, np.broadcast_arrays(self.x1, self.x2, self.x3))),
        ):
            yield Coord(float(x1), float(x2), float(x3), self.system)

    def offset(self, dx3: npt.ArrayLike) -> "Coords":
        """Changes the x3 coordinate by the specified ammount.

        """
        return Coords(self.x1, self.x2, self.x3 + dx3, self.system)

    def to_cartesian(self) -> "Coords":
        """Converts the points to be in Cartesian coordiantes.
        
        """
        x1, x2, x3 = COORDINATE_TRANSFORMS[self.system](self.x1, self.x2, self.x3)
        return Coords(
            x1,
            x2,
            x3,
            CoordinateSystem.Cartesian,
        )

    def __iter__(self) -> Iterator[npt.NDArray]:
        """Iterate over the individual coordinates of the points.

        """
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
NormalisedFieldLine = Callable[[npt.ArrayLike], Coords]


T = TypeVar("T")


class ElementLike(Protocol):
    """Protocal defining the methods that can be used to manipulate
    mesh components. Exists for internal type-checking.

    """
    def offset(self: T, offset: float) -> T:
        ...

    def subdivide(self: T, num_divisions: int) -> Iterator[T]:
        ...


@dataclass(frozen=True)
class Curve:
    """Represents a curve in 3D space. A curve is defined by a
    function which takes a single argument, 0 <= s <= 1, and returns
    coordinates for the location on that curve in space. The distance
    along the curve from the start to the position represented by s is
    directly proportional to s.

    """
    function: NormalisedFieldLine

    def __call__(self, s: npt.ArrayLike) -> Coords:
        """Convenience function so that a Curve is itself a NormalisedFieldLine"""
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


def line_from_points(north: Coord, south: Coord) -> NormalisedFieldLine:
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
    """Representation of a four-sided polygon (quadrilateral). It is
    represented by two curves representing opposite edges. The
    remaining edges are created by connecting the corresponding
    termini of the bounding lines. It also contains information on the
    magnetic field along which the curves defining the figure were
    traced.

    ToDo
    ----
    There is an optional attribute which is meant to define how the
    quadrilateral may curve into a third dimension, but this has not
    yet been used in the implementation. It probably will not be
    sufficiently general to be useful, in any case.

    """
    north: Curve
    south: Curve
    in_plane: Optional[
        Curve
    ]  # FIXME: Don't think this is adequate to describe curved quads
    field: FieldTrace
    NUM_CORNERS: ClassVar[int] = 4

    def __iter__(self) -> Iterable[Curve]:
        """Iterate over the two curves definge the edges of the quadrilateral.
        """
        yield self.north
        yield self.south

    @cached_property
    def near(self) -> Curve:
        """Returns a curve connecting the starting points (s=0) of the
        curves defining the boundaries of the quadrilateral.

        """
        return Curve(
            line_from_points(self.north(0.0).to_coord(), self.south(0.0).to_coord())
        )

    @cached_property
    def far(self) -> Curve:
        """Returns a curve connecting the end-points (s=1) of the
        curves defining the boundaries of the quadrilateral.

        """
        return Curve(
            line_from_points(self.north(1.0).to_coord(), self.south(1.0).to_coord())
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
        else:
            return cls._cached_quad(curve2, curve1, in_plane, field)

    def corners(self) -> Coords:
        """Returns the points corresponding to the corners of the quadrilateral.
        
        """
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
            # FIXME: can I be confident that the start and end will be exactly as specified?
            north_samples = self.north.control_points(order)
            south_samples = self.south.control_points(order)
            return Coords(
                np.linspace(north_samples.x1, south_samples.x1, order + 1),
                np.linspace(north_samples.x2, south_samples.x2, order + 1),
                np.linspace(north_samples.x3, south_samples.x3, order + 1),
                north_samples.system,
            )
        else:
            raise NotImplementedError(
                "Can not yet handle Quads where all four edges are curved"
            )

    def offset(self, offset: float) -> Quad:
        """Returns a quad which is identical except that it is shifted
        by the specified offset in teh x3 direction.
        
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
        equally-sized segments.

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
class Tet:
    north: Quad
    south: Quad
    east: Quad
    west: Quad
    NUM_CORNERS: ClassVar[int] = 8

    def __iter__(self) -> Iterable[Quad]:
        yield self.north
        yield self.east
        yield self.south
        yield self.west

    @cached_property
    def near(self) -> Quad:
        north = Curve(
            line_from_points(
                self.north.north(0.0).to_coord(), self.north.south(0.0).to_coord()
            )
        )
        south = Curve(
            line_from_points(
                self.south.north(0.0).to_coord(), self.south.south(0.0).to_coord()
            )
        )
        return Quad(north, south, None, self.north.field)

    @cached_property
    def far(self) -> Quad:
        north = Curve(
            line_from_points(
                self.north.north(1.0).to_coord(), self.north.south(1.0).to_coord()
            )
        )
        south = Curve(
            line_from_points(
                self.south.north(1.0).to_coord(), self.south.south(1.0).to_coord()
            )
        )
        return Quad(north, south, None, self.north.field)

    def corners(self) -> Coords:
        north_corners = self.north.corners()
        south_corners = self.south.corners()
        # TODO Check that east and west corners are the same as north and south
        return Coords(
            np.concatenate([north_corners.x1, south_corners.x1]),
            np.concatenate([north_corners.x2, south_corners.x2]),
            np.concatenate([north_corners.x3, south_corners.x3]),
            north_corners.system,
        )

    @cached_property
    def control_points(self) -> npt.NDArray:
        """Returns the coordinates of the control points for the
        surface, in an array of dimensions [3, order + 1, order + 1, order + 1].

        """
        raise NotImplementedError("Not written yet")

    def quads(self) -> Iterable[Quad]:
        yield self.north
        yield self.east
        yield self.south
        yield self.west

    def offset(self, offset: float) -> Tet:
        return Tet(
            self.north.offset(offset),
            self.south.offset(offset),
            self.east.offset(offset),
            self.west.offset(offset),
        )

    def subdivide(self, num_divisions: int) -> Iterator[Tet]:
        if num_divisions <= 0:
            yield self
        else:
            for n, s, e, w in zip(
                self.north.subdivide(num_divisions),
                self.south.subdivide(num_divisions),
                self.east.subdivide(num_divisions),
                self.west.subdivide(num_divisions),
            ):
                yield Tet(n, s, e, w)


E = TypeVar("E", Quad, Tet)
B = TypeVar("B", Curve, Quad)


@dataclass(frozen=True)
class MeshLayer(Generic[E, B]):
    reference_elements: Sequence[E]
    # FIXME: This isn't a great solution, as we care about the nodes/curves which are on the boundaries, not elements
    bounds: Sequence[frozenset[B]]
    offset: Optional[float] = None
    subdivisions: int = 1

    def __iter__(self) -> Iterator[E]:
        return self._iterate_elements(
            self.reference_elements, self.offset, self.subdivisions
        )

    def __len__(self) -> int:
        return len(self.reference_elements) * self.subdivisions

    @cached_property
    def element_type(self) -> Type[E]:
        return type(next(iter(self.reference_elements)))

    def quads(self) -> Iterable[Quad]:
        if len(self.reference_elements) > 0 and issubclass(self.element_type, Quad):
            return cast(Iterable[Quad], self)
        else:
            return itertools.chain.from_iterable(
                map(lambda t: t.quads(), cast(Iterable[Tet], self))
            )

    def boundaries(self) -> Iterator[frozenset[B]]:
        return map(
            frozenset,
            map(
                lambda b: self._iterate_elements(b, self.offset, self.subdivisions),
                self.bounds,
            ),
        )

    def near_faces(self) -> Iterator[B]:
        return map(
            lambda e: cast(B, e.near),
            self._iterate_elements(self.reference_elements, self.offset, 1),
        )

    # FIXME: This won't be bit-wise identical to to the last subdivision. Can I access those objects instead, somehow?
    def far_faces(self) -> Iterator[B]:
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
        elements: Iterable[ElementLike], offset: Optional[float], subdivisions: int
    ) -> Iterator[ElementLike]:
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

    # @cached_property
    # def num_unique_corners(self) -> int:
    #     element_corners = self.element_type.NUM_CORNERS * (self.subdivisions + 1) // 2
    #     total_corners = element_corners * len(self.reference_elements)
    #     num_face_connections = sum(
    #         list(connections.values()).count(True)
    #         for connections in self.reference_elements.values()
    #     )
    #     num_edge_connections = sum(
    #         list(connections.values()).count(False)
    #         for connections in self.reference_elements.values()
    #     )
    #     assert num_face_connections % 2 == 0, "Ill-defined mesh connectivity"
    #     assert num_edge_connections % 2 == 0, "Ill-defined mesh connectivity"
    #     return (
    #         total_corners
    #         - (num_face_connections // 2) * (element_corners)
    #         - (num_edge_connections // 2) * (element_corners // 2)
    #     )

    # def num_unique_control_points(self, order: int) -> int:
    #     points_per_edge = order * self.subdivisions + 1
    #     points_per_face = (order + 1) * points_per_edge
    #     if issubclass(self.element_type, Quad):
    #         points_per_element = points_per_face
    #     else:
    #         points_per_element = (order + 1) * points_per_face
    #     total_control_points = points_per_element * len(self.reference_elements)
    #     num_shared_points = sum(
    #         sum(
    #             points_per_face if is_face else points_per_edge
    #             for is_face in connections.values()
    #         )
    #         for connections in self.reference_elements.values()
    #     )
    #     assert num_shared_points % 2 == 0, "Ill-defined mesh connectivity"
    #     return total_control_points - num_shared_points // 2


@dataclass(frozen=True)
class GenericMesh(Generic[E, B]):
    reference_layer: MeshLayer[E, B]
    offsets: npt.NDArray

    def layers(self) -> Iterable[MeshLayer[E, B]]:
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
        return itertools.chain.from_iterable(map(iter, self.layers()))

    def __len__(self) -> int:
        return len(self.reference_layer) * self.offsets.size

    # @property
    # def num_unique_corners(self) -> int:
    #     return self.offsets.size * self.reference_layer.num_unique_corners

    # def num_unique_control_points(self, order: int) -> int:
    #     return self.offsets.size * self.reference_layer.num_unique_control_points(order)


QuadMesh = GenericMesh[Quad, Curve]
TetMesh = GenericMesh[Tet, Quad]
Mesh = QuadMesh | TetMesh


def normalise_field_line(
    trace: FieldTrace,
    start: SliceCoord,
    x3_min: float,
    x3_max: float,
    resolution=10,
) -> NormalisedFieldLine:
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
