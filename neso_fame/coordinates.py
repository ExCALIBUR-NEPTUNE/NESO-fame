"""Classes to represent coordinates."""

from __future__ import annotations

from collections import defaultdict
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
from functools import wraps
from types import EllipsisType
from typing import (
    Callable,
    Concatenate,
    Generic,
    ParamSpec,
    Type,
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

    def to_3d_coord(self, x3: float) -> Coord:
        """Create a 3D coordinate object from this 2D one."""
        return Coord(self.x1, self.x2, x3, self.system)

    def approx_eq(
        self, other: SliceCoord, rtol: float = 1e-9, atol: float = 1e-9
    ) -> bool:
        """Check equality of coordinates within the the tolerance."""
        return self.system == other.system and cast(
            bool,
            np.isclose(self.x1, other.x1, rtol, atol)
            and np.isclose(self.x2, other.x2, rtol, atol),
        )


CoordIndex = int | tuple[int, ...]
IndexSlice = (
    int
    | slice
    | EllipsisType
    | list[int | tuple[int, ...]]
    | tuple[int | slice | EllipsisType, ...]
)


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

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the logical shape of the coordinate array."""
        return np.broadcast(self.x1, self.x2).shape

    def __getitem__(self, idx: CoordIndex) -> SliceCoord:
        """Return an individual point from the collection."""
        x1, x2 = np.broadcast_arrays(self.x1, self.x2)
        return SliceCoord(float(x1[idx]), float(x2[idx]), self.system)

    @property
    def get(self) -> _CoordWrapper[SliceCoords]:
        """A view of this data which can be sliced.

        :method:`~neso_fame.coordinates.SliceCoords.__getitem__` and
        returns scalar :class:`~neso_fame.coordinates.SliceCoord`
        objects and will raise an error if the index does not
        correspond to a scalar value. Indexing the object returned by
        this property will return a
        :class:`~neso_fame.coordinates.SliceCoords` object instead,
        potentially containing an array of coordinates.

        """
        return _CoordWrapper(self)

    def _get(self, index: IndexSlice) -> SliceCoords:
        x1, x2 = np.broadcast_arrays(self.x1, self.x2)
        return SliceCoords(x1[index], x2[index], self.system)

    def get_set(self, index: IndexSlice) -> FrozenCoordSet[SliceCoord]:
        """Get a set of individual point objects from the collection."""
        return FrozenCoordSet(self._get(index).iter_points())

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
        if len(self) > 1:
            raise RuntimeError(
                "Can not convert array of coordinates to a single coordinate."
            )
        return SliceCoord(float(self.x1.flat[0]), float(self.x2.flat[0]), self.system)

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

    def approx_eq(self, other: Coord, rtol: float = 1e-9, atol: float = 1e-9) -> bool:
        """Check equality of coordinates within the the tolerance."""
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

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the logical shape of the coordinate array."""
        return np.broadcast(self.x1, self.x2, self.x3).shape

    def __getitem__(self, idx: CoordIndex) -> Coord:
        """Return the coordinates of an individual point."""
        x1, x2, x3 = np.broadcast_arrays(self.x1, self.x2, self.x3)
        return Coord(float(x1[idx]), float(x2[idx]), float(x3[idx]), self.system)

    @property
    def get(self) -> _CoordWrapper[Coords]:
        """A view of this data which can be sliced.

        :method:`~neso_fame.coordinates.Coords.__getitem__` and
        returns scalar :class:`~neso_fame.coordinates.Coord`
        objects and will raise an error if the index does not
        correspond to a scalar value. Indexing the object returned by
        this property will return a
        :class:`~neso_fame.coordinates.Coords` object instead,
        potentially containing an array of coordinates.

        """
        return _CoordWrapper(self)

    def _get(self, index: IndexSlice) -> Coords:
        x1, x2, x3 = np.broadcast_arrays(self.x1, self.x2, self.x3)
        return Coords(x1[index], x2[index], x3[index], self.system)

    def get_set(self, index: IndexSlice) -> FrozenCoordSet[Coord]:
        """Get a set of individual point objects from the collection."""
        return FrozenCoordSet(self._get(index).iter_points())

    def to_coord(self) -> Coord:
        """Convert the object to a `Coord` object.

        This will only work if the collection contains exactly one
        point. Otherwise, an exception is raised.
        """
        if len(self) > 1:
            raise RuntimeError(
                "Can not convert array of coordinates to a single coordinate."
            )
        return Coord(
            float(self.x1.flat[0]),
            float(self.x2.flat[0]),
            float(self.x3.flat[0]),
            self.system,
        )

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
Cs = TypeVar("Cs", Coords, SliceCoords)
T = TypeVar("T")


@dataclass(frozen=True)
class _CoordWrapper(Generic[Cs]):
    """Simple class to allow slicing coordinate objects.

    :method:`~neso_fame.coordinates.SliceCoords.__getitem__` and
    :method:`~neso_fame.coordinates.Coords.__getitem__` are designed
    to return scalar :class:`~neso_fame.coordinates.SliceCoord` and
    :class:`~neso_fame.coordinates.SliceCoord` objects,
    respectively. This wrapper class will return
    :class:`~neso_fame.coordinates.SliceCoords` and
    :class:`~neso_fame.coordinates.SliceCoords` instead.

    Group
    -----
    coordinates

    """

    data: Cs

    def __getitem__(self, index: IndexSlice) -> Cs:
        """Get a slice of the contained data."""
        return self.data._get(index)


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
        self,
        coords: Iterable[C] = {},
        rtol: float = 1e-9,
        atol: float = 1e-9,
        _dim: int | None = None,
    ) -> None:
        self._atol = atol
        self._rtol = rtol
        self._coords = []
        coords_list = list(coords)
        if _dim is not None:
            self._dim = _dim
        else:
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

    @property
    def contents_type(self) -> Type[C] | None:
        try:
            return type(next(iter(self)))
        except StopIteration:
            return None

    def __contains__(self, c: object) -> bool:
        """Check if a coordinate is already stored in this object (within tolerance)."""
        if (expected_type := self.contents_type) is None:
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
        return hash((len(self), self.contents_type, self._rtol, self._atol, self._dim))

    def __repr__(self) -> str:
        """Produce a string representation of this object."""
        return f"{self.__class__.__name__}({{{', '.join(repr(c) for c in self._coords if c is not None)}}})"

    def __len__(self) -> int:
        """Get the number of coordinates stored in this object.

        Optimised version for immutable case.
        """
        return len(self._coords)

    @classmethod
    def empty_slicecoord(
        cls, rtol: float = 1e-9, atol: float = 1e-9
    ) -> FrozenCoordSet[SliceCoord]:
        """Construct an empty set for SliceCoords."""
        return cast(FrozenCoordSet[SliceCoord], cls([], rtol, atol, 2))

    @classmethod
    def empty_coord(
        cls, rtol: float = 1e-9, atol: float = 1e-9
    ) -> FrozenCoordSet[Coord]:
        """Construct an empty set for Coords."""
        return cast(FrozenCoordSet[Coord], cls([], rtol, atol, 3))


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

    @classmethod
    def empty_slicecoord(
        cls, rtol: float = 1e-9, atol: float = 1e-9
    ) -> CoordSet[SliceCoord]:
        """Construct an empty set for SliceCoords."""
        return cast(CoordSet[SliceCoord], cls([], rtol, atol, 2))

    @classmethod
    def empty_coord(cls, rtol: float = 1e-9, atol: float = 1e-9) -> CoordSet[Coord]:
        """Construct an empty set for Coords."""
        return cast(CoordSet[Coord], cls([], rtol, atol, 3))


class CoordMap(_CoordContainer[C], Mapping[C, T]):
    """An immutable map taking coordinates as keys and comparing them within a tolerance."""

    _values: list[T | None]

    def __init__(
        self,
        data: Mapping[C, T] = {},
        rtol: float = 1e-9,
        atol: float = 1e-9,
        _dim: int | None = None,
    ) -> None:
        self._atol = atol
        self._rtol = rtol
        self._coords = []
        self._values = []
        if _dim is not None:
            self._dim = _dim
        else:
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

    @property
    def system(self) -> CoordinateSystem | None:
        """The coordinate system use by the coordinates stored in this object."""
        try:
            return next(c.system for c in self._coords if c is not None)
        except StopIteration:
            return None

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

    def __len__(self) -> int:
        """Get the number of coordinates stored in this object.

        Optimised version for immutable case.
        """
        return len(self._coords)

    @classmethod
    def empty_slicecoord(
        cls, t: Type[T], rtol: float = 1e-9, atol: float = 1e-9
    ) -> CoordMap[SliceCoord, T]:
        """Construct an empty mapping for SliceCoords."""
        return cast(CoordMap[SliceCoord, T], cls({}, rtol, atol, 2))

    @classmethod
    def empty_coord(
        cls, t: Type[T], rtol: float = 1e-9, atol: float = 1e-9
    ) -> CoordMap[Coord, T]:
        """Construct an empty mapping for Coords."""
        return cast(CoordMap[Coord, T], cls({}, rtol, atol, 3))

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

    @classmethod
    def empty_slicecoord(
        cls, t: Type[T], rtol: float = 1e-9, atol: float = 1e-9
    ) -> MutableCoordMap[SliceCoord, T]:
        """Construct an empty mapping for SliceCoords."""
        return cast(MutableCoordMap[SliceCoord, T], cls({}, rtol, atol, 2))

    @classmethod
    def empty_coord(
        cls, t: Type[T], rtol: float = 1e-9, atol: float = 1e-9
    ) -> MutableCoordMap[Coord, T]:
        """Construct an empty mapping for Coords."""
        return cast(MutableCoordMap[Coord, T], cls({}, rtol, atol, 3))


P = ParamSpec("P")
CoordParams = Concatenate[C, P]


@dataclass(frozen=True)
class _CoordRecord:
    ctype: Type[Coord] | Type[SliceCoord]
    system: CoordinateSystem
    index: int


def coord_cache(
    rtol: float = 1e-9, atol: float = 1e-9
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Wrap a function to cache results comparing coordinate arguments based on proximity."""

    def decorator(
        func: Callable[P, T],
    ) -> Callable[P, T]:
        coord_data: dict[CoordinateSystem, MutableCoordMap[Coord, int]] = defaultdict(
            lambda: MutableCoordMap.empty_coord(int, rtol, atol)
        )
        slicecoord_data: dict[CoordinateSystem, MutableCoordMap[SliceCoord, int]] = (
            defaultdict(lambda: MutableCoordMap.empty_slicecoord(int, rtol, atol))
        )
        cache_data: dict[tuple, T] = {}

        def process_arg(x: object) -> object:
            if isinstance(x, Coord):
                sys = x.system
                dat = coord_data[sys]
                return _CoordRecord(Coord, sys, dat.setdefault(x, len(dat)))
            elif isinstance(x, SliceCoord):
                sys = x.system
                sdat = slicecoord_data[sys]
                return _CoordRecord(
                    SliceCoord,
                    sys,
                    sdat.setdefault(x, len(sdat)),
                )
            return x

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            idx = tuple(map(process_arg, args + tuple(kwargs.items())))
            if idx not in cache_data:
                obj = func(*args, **kwargs)
                cache_data[idx] = obj
                return obj
            return cache_data[idx]

        return wrapper

    return decorator
