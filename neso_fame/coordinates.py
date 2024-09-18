"""Classes to represent coordinates."""

from __future__ import annotations

from collections.abc import (
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    MutableSet,
    Set,
)
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Context, Decimal
from enum import Enum
from typing import (
    Callable,
    ClassVar,
    Concatenate,
    Generic,
    ParamSpec,
    TypeVar,
    cast,
)

import numpy as np
import numpy.typing as npt
from rtree.index import Index, Property


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

    # def __hash__(self) -> int:
    #     """Hash the SliceCoord object.

    #     Hashing is done so that floats are the same to the number of
    #     significant figures set by TOLERANCE will have the same hash
    #     value.

    #     FIXME

    #     This isn't as important as it used to be, given that I now use
    #     an R-tree for caching the creation of Nektar++ coordinates,
    #     but it is still useful to be able to do approximate equality
    #     for sets (when testing if nothing else). Perhaps I can set up
    #     some sort of global or class-level R-tree for storing hash
    #     values and use that in this method?

    #     """
    #     decimal_places = -int(np.floor(np.log10(self.TOLERANCE))) - 1
    #     context = Context(decimal_places)

    #     # FIXME: This can still result in hashes being different for
    #     # two similar numbers, if rounding would affect multiple
    #     # decimal places (e.g., rounding .999995 up to 1.). The
    #     # likelihood is low but non-zero.
    #     def get_digits(
    #         x: float,
    #     ) -> tuple[int, tuple[int, ...], int | Literal["n", "N", "F"]]:
    #         y = Decimal(x).normalize(context).as_tuple()
    #         spare_places = decimal_places - len(y[1])
    #         if isinstance(y[2], int) and len(y[1]) + y[2] < -8:
    #             return 0, (), 0
    #         truncated = (y[1] + (0,) * spare_places)[:-3]
    #         if all(t == 0 for t in truncated):
    #             return 0, (), 0
    #         exponent = y[2]
    #         if isinstance(exponent, int):
    #             exponent -= spare_places - 3
    #         return y[0], truncated, exponent

    #     x1 = get_digits(self.x1)
    #     x2 = get_digits(self.x2)
    #     return hash((x1, x2, self.system))

    def to_3d_coord(self, x3: float) -> Coord:
        """Create a 3D coordinate object from this 2D one."""
        return Coord(self.x1, self.x2, x3, self.system)

    # def __eq__(self, other: object) -> bool:
    #     """Check equality of coordinates within the the TOLERANCE."""
    #     if not isinstance(other, self.__class__):
    #         return False
    #     return self.system == other.system and cast(
    #         bool,
    #         np.isclose(self.x1, other.x1, self.TOLERANCE, self.TOLERANCE)
    #         and np.isclose(self.x2, other.x2, self.TOLERANCE, self.TOLERANCE),
    #     )
    def approx_eq(
        self, other: SliceCoord, rtol: float = 1e-9, atol: float = 1e-9
    ) -> bool:
        """Check equality of coordinates within the the TOLERANCE."""
        return self.system == other.system and cast(
            bool,
            np.isclose(self.x1, other.x1, rtol, atol)
            and np.isclose(self.x2, other.x2, rtol, atol),
        )


CoordIndex = int | tuple[int, ...]
IndexSlice = int | slice | list[int | tuple[int, ...]] | tuple[int | slice, ...]


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
        if np.broadcast(self.x1, self.x2).size == 0:
            return
        for x1, x2 in zip(*map(np.nditer, np.broadcast_arrays(self.x1, self.x2))):
            yield SliceCoord(float(x1), float(x2), self.system)

    def __iter__(self) -> Iterator[npt.NDArray]:
        """Iterate over the coordinates of the points."""
        for array in np.broadcast_arrays(self.x1, self.x2):
            yield array

    def __len__(self) -> int:
        """Return the number of points contained in the object."""
        return np.broadcast(self.x1, self.x2).size

    def __getitem__(self, idx: CoordIndex) -> SliceCoord:
        """Return an individual point from the collection."""
        x1, x2 = np.broadcast_arrays(self.x1, self.x2)
        return SliceCoord(float(x1[idx]), float(x2[idx]), self.system)

    def get_set(self, index: IndexSlice) -> FrozenCoordSet[SliceCoord]:
        """Get a set of individual point objects from the collection."""
        x1, x2 = np.broadcast_arrays(self.x1, self.x2)
        return FrozenCoordSet(
            SliceCoords(x1[index], x2[index], self.system).iter_points()
        )

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

    # def __hash__(self) -> int:
    #     """Hash the Coord object.

    #     Hashing is done so that floats are the same to the number of
    #     significant figures set by TOLERANCE will have the same hash
    #     value.
    #     """
    #     decimal_places = -int(np.floor(np.log10(self.TOLERANCE))) - 1
    #     context = Context(decimal_places)

    #     # FIXME: This can still result in hashes being different for
    #     # two similar numbers, if rounding would affect multiple
    #     # decimal places (e.g., rounding .999995 up to 1.). The
    #     # likelihood is low but non-zero.
    #     def get_digits(
    #         x: float,
    #     ) -> tuple[int, tuple[int, ...], int | Literal["n", "N", "F"]]:
    #         y = Decimal(x).normalize(context).as_tuple()
    #         spare_places = decimal_places - len(y[1])
    #         if isinstance(y[2], int) and len(y[1]) + y[2] < -8:
    #             return 0, (), 0
    #         truncated = (y[1] + (0,) * spare_places)[:-3]
    #         if all(t == 0 for t in truncated):
    #             return 0, (), 0
    #         exponent = y[2]
    #         if isinstance(exponent, int):
    #             exponent -= spare_places - 3
    #         return y[0], truncated, exponent

    #     x1 = get_digits(self.x1)
    #     x2 = get_digits(self.x2)
    #     x3 = get_digits(self.x3)

    #     return hash((x1, x2, x3, self.system))

    # def __eq__(self, other: object) -> bool:
    #     """Check equality of coordinates within the the TOLERANCE."""
    #     if not isinstance(other, self.__class__):
    #         return False
    #     return self.system == other.system and cast(
    #         bool,
    #         np.isclose(self.x1, other.x1, self.TOLERANCE, self.TOLERANCE)
    #         and np.isclose(self.x2, other.x2, self.TOLERANCE, self.TOLERANCE)
    #         and np.isclose(self.x3, other.x3, self.TOLERANCE, self.TOLERANCE),
    #     )
    def approx_eq(self, other: Coord, rtol: float = 1e-9, atol: float = 1e-9) -> bool:
        """Check equality of coordinates within the the TOLERANCE."""
        return self.system == other.system and cast(
            bool,
            np.isclose(self.x1, other.x1, rtol, atol)
            and np.isclose(self.x2, other.x2, rtol, atol)
            and np.isclose(self.x3, other.x3, rtol, atol),
        )

    def to_slice_coord(self) -> SliceCoord:
        """Get the poloidal components of this coordinate."""
        return SliceCoord(self.x1, self.x2, self.system)


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
        if np.broadcast(self.x1, self.x2, self.x3).size == 0:
            return
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

    def __getitem__(self, idx: CoordIndex) -> Coord:
        """Return the coordinates of an individual point."""
        x1, x2, x3 = np.broadcast_arrays(self.x1, self.x2, self.x3)
        return Coord(float(x1[idx]), float(x2[idx]), float(x3[idx]), self.system)

    def get_set(self, index: IndexSlice) -> FrozenCoordSet[Coord]:
        """Get a set of individual point objects from the collection."""
        x1, x2, x3 = np.broadcast_arrays(self.x1, self.x2, self.x3)
        return FrozenCoordSet(
            Coords(x1[index], x2[index], x3[index], self.system).iter_points()
        )

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

    def to_slice_coords(self) -> SliceCoords:
        """Get the poloidal components of this coordinate."""
        return SliceCoords(self.x1, self.x2, self.system)


C = TypeVar("C", Coord, SliceCoord)
T = TypeVar("T")


class _CoordContainer(Generic[C]):
    """Base type for approximate coordinate comparison.

    This contains all the methods necessary for the Set ABC, but is
    kept seperate so it can be inherited from to implement the Mapping
    ABCs as well (without getting all the Set mixins).

    """

    _coords: list[C | None]
    _rtol: float
    _atol: float
    _rtree: Index
    _dim: int

    def __init__(
        self, coords: Iterable[C] = {}, rtol: float = 1e-9, atol: float = 1e-9
    ) -> None:
        self._atol = atol
        self._rtol = rtol
        self._coords = []
        coords_list = list(coords)
        if len(coords_list) > 0:
            self._dim = 3 if isinstance(coords_list[0], Coord) else 2
        else:
            self._dim = 3
        self._rtree = Index(interleaved=False, properties=Property(dimension=self._dim))
        for c in coords_list:
            self._check_coord_system(c)
            if c not in self:
                self._rtree.insert(len(self._coords), self._get_bound_box(c))
                self._coords.append(c)

    @property
    def system(self) -> CoordinateSystem | None:
        """The coordinate system use by the coordinates stored in this object."""
        try:
            return next(c.system for c in self._coords if c is not None)
        except StopIteration:
            return None

    def _check_coord_system(self, arg: C) -> None:
        if (system := self.system) is not None and system != arg.system:
            raise ValueError(
                f"{self.__class__.__name__} has coordinate system {system} but argument has {arg.system}"
            )

    def _get_bound_box(self, position: C) -> tuple[float, ...]:
        def offset(x: float) -> tuple[float, float]:
            dx = max(abs(self._atol), abs(x * self._rtol))
            return x - dx, x + dx

        if self._dim == 2:
            if isinstance(position, Coord):
                raise ValueError("Can not handle 3D coordinate in 2D R-tree.")
            return offset(position.x1) + offset(position.x2)
        assert self._dim == 3
        if isinstance(position, SliceCoord):
            return offset(position.x1) + offset(position.x2) + (0.0, 0.0)
        return offset(position.x1) + offset(position.x2) + offset(position.x3)

    def __contains__(self, c: object) -> bool:
        """Check if a coordinate is already stored in this object (within tolerance)."""
        try:
            expected_type = next(type(x) for x in self._coords if x is not None)
        except StopIteration:
            return False
        if not isinstance(c, expected_type):
            return False
        self._check_coord_system(c)
        try:
            _ = next(self._rtree.intersection(self._get_bound_box(c), objects=False))
            return True
        except StopIteration:
            return False

    def __iter__(self) -> Iterator[C]:
        """Iterate through the coordinates stored in this object."""
        return iter(c for c in self._coords if c is not None)

    def __len__(self) -> int:
        """Get the number of coordinates stored in this object."""
        return sum(c is not None for c in self._coords)


class FrozenCoordSet(_CoordContainer[C], Set[C]):
    """An immutable set of coordinates, evaluating equality to within a tolerance."""

    def __hash__(self) -> int:
        """Return a very dumb hash that will ensure things work logically (but inefficiently)."""
        return len(self)

    def __repr__(self) -> str:
        """Produce a string representation of this object."""
        return f"{self.__class__.__name__}({{{', '.join(repr(c) for c in self._coords if c is not None)}}})"


class CoordSet(FrozenCoordSet[C], MutableSet[C]):
    """A set of coordinates, evaluating equality to within a tolerance."""

    def add(self, position: C) -> None:
        """Add a new coordinate the set."""
        if position not in self:
            self._rtree.insert(len(self._coords), self._get_bound_box(position))
            self._coords.append(position)

    def discard(self, position: C) -> None:
        """Remove a coordinate from the set (if present within tolerance)."""
        self._check_coord_system(position)
        for item in self._rtree.intersection(
            self._get_bound_box(position), objects=True
        ):
            self._rtree.delete(item.id, item.bbox)
            self._coords[item.id] = None
            break


class CoordMap(_CoordContainer[C], Mapping[C, T]):
    """An immutable map taking coordinates as keys and comparing them within a tolerance."""

    _values: list[T | None]

    def __init__(
        self, data: Mapping[C, T] = {}, rtol: float = 1e-9, atol: float = 1e-9
    ) -> None:
        self._atol = atol
        self._rtol = rtol
        self._coords = []
        self._values = []
        try:
            c = next(iter(data))
            self._dim = 3 if isinstance(c, Coord) else 2
        except StopIteration:
            self._dim = 3
        self._rtree = Index(interleaved=False, properties=Property(dimension=self._dim))
        for i, (c, v) in enumerate(data.items()):
            self._check_coord_system(c)
            self._rtree.insert(i, self._get_bound_box(c))
            self._coords.append(c)
            self._values.append(v)

    def __getitem__(self, item: C) -> T:
        """Access the value associated with the coordinate."""
        self._check_coord_system(item)
        try:
            i = next(self._rtree.intersection(self._get_bound_box(item), objects=False))
            return cast(T, self._values[i])
        except StopIteration:
            raise KeyError(f"Coordinate {item} not present in mapping")

    def __repr__(self) -> str:
        """Produce a string representation of this object."""
        return f"{self.__class__.__name__}({{{', '.join(repr(c) + ': ' + repr(v) for c, v in zip(self._coords, self._values) if c is not None)}}})"

    # TODO: Ideally would override some of hte mixins for better
    # performance. Also not sure whether the MappingView will work
    # properly


class MutableCoordMap(CoordMap[C, T], MutableMapping[C, T]):
    """A map taking coordinates as keys and comparing them within a tolerance."""

    def __setitem__(self, key: C, value: T) -> None:
        """Assign a value to the coordinate in the mapping."""
        self._check_coord_system(key)
        for item in self._rtree.intersection(self._get_bound_box(key), objects=False):
            self._values[item] = value
            return
        self._rtree.insert(len(self._coords), self._get_bound_box(key))
        self._coords.append(key)
        self._values.append(value)

    def __delitem__(self, position: C) -> None:
        """Remove the coordinate and its associated value from the mapping."""
        self._check_coord_system(position)
        for item in self._rtree.intersection(
            self._get_bound_box(position), objects=True
        ):
            self._rtree.delete(item.id, item.bbox)
            self._coords[item.id] = None
            self._values[item.id] = None
            break


P = ParamSpec("P")
CoordParams = Concatenate[C, P]


def coord_cache(
    rtol: float = 1e-9, atol: float = 1e-9
) -> Callable[[Callable[Concatenate[C, P], T]], Callable[Concatenate[C, P], T]]:
    """Return a wrapped function that caches based on proximity of coordinates."""

    def decorator(
        func: Callable[Concatenate[C, P], T],
    ) -> Callable[Concatenate[C, P], T]:
        cache_data: dict[tuple, MutableCoordMap[C, T]] = {}

        def wrapper(position: C, *args: P.args, **kwargs: P.kwargs) -> T:
            idx = args + tuple(kwargs.items())
            if idx not in cache_data:
                obj = func(position, *args, **kwargs)
                cache_data[idx] = MutableCoordMap({position: obj}, rtol, atol)
                return obj
            cmap = cache_data[idx]
            try:
                return cmap[position]
            except KeyError:
                obj = func(position, *args, **kwargs)
                cmap[position] = obj
                return obj

        return wrapper

    return decorator
