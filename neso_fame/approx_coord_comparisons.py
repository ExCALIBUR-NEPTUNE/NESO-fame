"""Tools for comparing coordinate positions within some tolerance."""

from collections.abc import Iterable, Iterator, Mapping, MutableMapping, MutableSet, Set
from typing import Callable, Concatenate, Generic, ParamSpec, TypeVar, cast

from rtree.index import Index, Property

from neso_fame.mesh import Coord, CoordinateSystem, SliceCoord

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

    def __init__(self, coords: Iterable[C], rtol: float = 1e-9, atol: float = 1e-9) -> None:
        self._atol = atol
        self._rtol = rtol
        self._coords = list(coords)
        if len(self._coords) > 0:
            self._dim = 3 if isinstance(self._coords[0], Coord) else 2
        else:
            self._dim = 3
        self._rtree = Index(interleaved=False, properties=Property(dimension=self._dim))
        for i, c in enumerate(cast(list[C], self._coords)):
            self._check_coord_system(c)
            self._rtree.insert(i, self._get_bound_box(c))

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

    pass


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
