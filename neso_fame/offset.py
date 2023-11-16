"""Provides a wrapper that offsets its contents by some ammount in the x3-direction."""

from collections.abc import Callable, Iterable, Sized
from dataclasses import dataclass
from typing import (
    Any,
    Generic,
    Iterator,
    NoReturn,
    Protocol,
    Sequence,
    TypeVar,
    cast,
    final,
    runtime_checkable,
)

from typing_extensions import Self


class OffsetableMeta(type):
    """Metaclass for offsetable clases, providing help with type-checking.

    Group
    -----
    offset

    """

    def __instancecheck__(self, instance: Any) -> bool:  # type: ignore
        """Check if ``instance`` is an instance of ``self``.

        This will treat ``EagerlyOffsetable[T]`` as an instance of ``T``.
        """
        if issubclass(type(instance), LazyOffset):
            return issubclass(type(instance.obj), self)
        else:
            return issubclass(type(instance), self)


class LazilyOffsetable(metaclass=OffsetableMeta):
    """Base class for types which can be wrapped in :class:`neso_fame.offset.LazyOffset`.

    Group
    -----
    offset

    """

    @final
    @property
    def x3_offset(self) -> float:
        """Return the offset in the x3 direction."""
        return 0.0

    @final
    @property
    def is_offset(self) -> bool:
        """Indicate whether the object has been offset."""
        return False

    @final
    def get_underlying_object(self) -> Self:
        """Return the object being offset."""
        return self


@runtime_checkable
class EagerlyOffsetable(Protocol):
    """Protocal for classes which should be eagerly offset.

    Group
    -----
    offset

    """

    def offset(self, offset: float) -> Self:
        """Return a version of the object offset in the x3 direction."""
        ...


T = TypeVar("T", bound=LazilyOffsetable | Callable)
BuiltinSequence = (list, frozenset, tuple)


@dataclass(frozen=True)
class LazyOffset(Generic[T]):
    """Wrap an object to represent it being offset in the x3 direction.

    All attribute lookup and function calls will be forwarded to the
    wrapped object. If the result has an offset method, then it will
    be called to eagerly evaluate the offset. Otherwise, the results
    will also be wrapped by a LazyOffset.

    Group
    -----
    offset

    """

    obj: T
    x3_offset: float

    def __init__(self, obj: T, x3_offset: float) -> None:
        """Wrap the object so that it is lazily offset.

        If the `obj` is already a LazyOffset, it is the contents of
        `obj` that will be wrapped and the offset will be increased
        (or decreased) accordingly.

        """
        if isinstance(obj, LazyOffset):
            self.__dict__["obj"] = obj.obj
            self.__dict__["x3_offset"] = obj.x3_offset + x3_offset
        else:
            self.__dict__["obj"] = obj
            self.__dict__["x3_offset"] = x3_offset

    @final
    @property
    def is_offset(self) -> bool:
        """Indicate whether the object has been offset."""
        return self.x3_offset != 0.0

    @final
    def get_underlying_object(self) -> Any:
        """Return the object being offset."""
        return self.obj

    def _wrap(self, result: Any) -> Any:
        """Wrap a result in a LazyOffset, if appropriate."""
        if isinstance(result, EagerlyOffsetable):
            return result.offset(self.x3_offset)
        elif isinstance(result, type):
            return result
        elif isinstance(result, LazilyOffsetable) or callable(result):
            return LazyOffset(result, self.x3_offset)
        # FIXME: Should I be operating on the Collection ABC instead?
        elif isinstance(result, BuiltinSequence):
            return result.__class__(map(self._wrap, result))
        elif isinstance(result, dict):
            return result.__class__(
                (self._wrap(k), self._wrap(v)) for k, v in result.items()
            )
        elif isinstance(result, Iterator):
            return map(self._wrap, result)
        # FIXME: Any other special cases I need to handle?
        else:
            return result

    def _raise_type_error(self, message: str) -> NoReturn:
        raise TypeError(f"{self.obj.__class__.__name__}' object is not {message}")

    def __getattr__(self, name: str) -> Any:
        """Get an attribute of the wrapped object and offset it."""
        return self._wrap(getattr(self.obj, name))

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Try to call the wrapped object and offset the result."""
        if not callable(self.obj):
            self._raise_type_error("callable")
        return self._wrap(self.obj(*args, *kwargs))

    def __iter__(self) -> Iterator[Any]:
        """Try to iterate the wrapped object and offset the iterator."""
        if not isinstance(self.obj, Iterable):
            self._raise_type_error("iterable")
        return self._wrap(self.obj.__iter__())

    def __len__(self) -> int:
        """Try to get the length of the wrapped object."""
        if not isinstance(self.obj, Sized):
            raise TypeError(
                f"object of type '{self.obj.__class__.__name__}' has no len()"
            )
        return len(self.obj)

    def __getitem__(self, index: Any) -> Any:
        """Try to index the wrapped object and offset the result."""
        if not isinstance(self.obj, Sequence):
            self._raise_type_error("subscriptable")
        return self._wrap(self.obj[index])

    def __next__(self) -> Any:
        """Try to get the next item of a wrapped iterator and offset the result."""
        if not isinstance(self.obj, Iterator):
            self._raise_type_error("an iterator")
        return self._wrap(next(self.obj))


def Offset(obj: T, x3_offset: float) -> T:
    """Offset the object in the x3-direction.

    This function wraps an object so that it has an associated offset
    in the x3-direction. The wrapper will then forward all attribute
    lookup to the wrapped object, but wrap the result with the offset
    again. The same applies to function calls. If the result has an
    `offset` method (e.g., if it is a Coord object or similar), then
    that is called instead.

    There is a bit of a hack here. Because the Python type-system
    doesn't have a good way to express all attribute lookup being
    forwarded to the wrapee, this function falsely claims to return
    the original type. This means mypy will think that it can perform
    all the usual operations on the wrapped type, but at the cost of
    losing some information about the real type of the object
    involved. This is considered a reasonable tradeoff, as the whole
    point of the offsetting is that the objects should behave
    indistinguishably.

    Group
    -----
    offset

    """
    return cast(T, LazyOffset(obj, x3_offset))
