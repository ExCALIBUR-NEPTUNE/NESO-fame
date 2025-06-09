import itertools
import operator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from hypnotoad import (
    Point2D,  # type: ignore
)
from hypothesis import given
from hypothesis.strategies import (
    sampled_from,
)

from neso_fame.coordinates import (
    CoordinateSystem,
    FrozenCoordSet,
    SliceCoord,
    SliceCoords,
)
from neso_fame.mesh import (
    straight_line_across_field,
)

from .conftest import simple_trace

R, Z = np.meshgrid(np.linspace(0.5, 1.5, 11), np.linspace(-1, 1, 11))
WEST_OUTER = SliceCoords(R[:, 0], Z[:, 0], CoordinateSystem.CYLINDRICAL)
EAST_OUTER = SliceCoords(R[:, -1], Z[:, -1], CoordinateSystem.CYLINDRICAL)
SOUTH_OUTER = SliceCoords(R[0, :], Z[0, :], CoordinateSystem.CYLINDRICAL)
NORTH_OUTER = SliceCoords(R[-1, :], Z[-1, :], CoordinateSystem.CYLINDRICAL)
WEST_INNER = SliceCoords(R[4:7, 4], Z[4:7, 4], CoordinateSystem.CYLINDRICAL)
EAST_INNER = SliceCoords(R[4:7:, 6], Z[4:7, 6], CoordinateSystem.CYLINDRICAL)
SOUTH_INNER = SliceCoords(R[4, 4:7], Z[4, 4:7], CoordinateSystem.CYLINDRICAL)
NORTH_INNER = SliceCoords(R[6, 4:7], Z[6, 4:7], CoordinateSystem.CYLINDRICAL)
OUTERMOST = (
    FrozenCoordSet(EAST_OUTER.iter_points())
    | FrozenCoordSet(WEST_OUTER.iter_points())
    | FrozenCoordSet(NORTH_OUTER.iter_points())
    | FrozenCoordSet(SOUTH_OUTER.iter_points())
)
INNERMOST = (
    FrozenCoordSet(EAST_INNER.iter_points())
    | FrozenCoordSet(WEST_INNER.iter_points())
    | FrozenCoordSet(NORTH_INNER.iter_points())
    | FrozenCoordSet(SOUTH_INNER.iter_points())
)


UNFINISHED_NORTH_EAST = SliceCoords(R[7:, -1], Z[7:, -1], CoordinateSystem.CYLINDRICAL)
UNFINISHED_SOUTH_EAST = SliceCoords(R[:4, -1], Z[:4, -1], CoordinateSystem.CYLINDRICAL)
UNFINISHED_NORTH_WEST = SliceCoords(R[7:, 0], Z[7:, 0], CoordinateSystem.CYLINDRICAL)
UNFINISHED_SOUTH_WEST = SliceCoords(R[:4, 0], Z[:4, 0], CoordinateSystem.CYLINDRICAL)
NOT_IN_UNFINISHED = SliceCoords(
    np.concatenate((R[4:7, 0], R[4:7, -1])),
    np.concatenate((Z[4:7, 0], Z[4:7, 0])),
    CoordinateSystem.CYLINDRICAL,
)

MOCK_MESH = MagicMock()
MOCK_MESH.equilibrium.o_point = Point2D(1.0, 0.0)

# BUILDER = ElementBuilder(MOCK_MESH, simple_trace, 0.1, EMPTY_MAP)
# BUILDER_UNFINISHED = ElementBuilder(MOCK_MESH, simple_trace, 0.1, EMPTY_MAP)
# with (
#     patch(
#         "neso_fame.element_builder.flux_surface_edge",
#         lambda _, north, south: straight_line_across_field(north, south, 2),
#     ),
# ):
#     # Create a square mesh with a hole in the middle
#     for corners in itertools.chain(
#         zip(
#             *map(
#                 operator.methodcaller("iter_points"),
#                 _element_corners(R[:5, :], Z[:5, :]),
#             )
#         ),
#         zip(
#             *map(
#                 operator.methodcaller("iter_points"),
#                 _element_corners(R[6:, :], Z[6:, :]),
#             )
#         ),
#         zip(
#             *map(
#                 operator.methodcaller("iter_points"),
#                 _element_corners(R[4:7, :5], Z[4:7, :5]),
#             )
#         ),
#         zip(
#             *map(
#                 operator.methodcaller("iter_points"),
#                 _element_corners(R[4:7, 6:], Z[4:7, 6:]),
#             )
#         ),
#     ):
#         _ = BUILDER.make_element(*corners)
#     # Don't build the middle elements
#     for corners in itertools.chain(
#         zip(
#             *map(
#                 operator.methodcaller("iter_points"),
#                 _element_corners(R[:4, :], Z[:4, :]),
#             )
#         ),
#         zip(
#             *map(
#                 operator.methodcaller("iter_points"),
#                 _element_corners(R[7:, :], Z[7:, :]),
#             )
#         ),
#     ):
#         _ = BUILDER_UNFINISHED.make_element(*corners)


outer_vertices = sampled_from(list(OUTERMOST))


def test_outermost_vertices_empty() -> None:
    builder = ElementBuilder(MOCK_MESH, simple_trace, 0.1, EMPTY_MAP)
    assert len(list(builder.outermost_vertices())) == 0


def test_outermost_vertices() -> None:
    ordered_outermost = list(BUILDER.outermost_vertices())
    assert len(ordered_outermost) == len(OUTERMOST)
    assert FrozenCoordSet(ordered_outermost) == OUTERMOST


def test_outermost_vertices_order() -> None:
    ordered_outermost = list(BUILDER.outermost_vertices())
    outermost_between = list(
        BUILDER.outermost_vertices_between(ordered_outermost[0], ordered_outermost[-1])
    )
    assert ordered_outermost == outermost_between


@given(outer_vertices, outer_vertices)
def test_outermost_between_termini(start: SliceCoord, end: SliceCoord) -> None:
    outermost_between = list(BUILDER.outermost_vertices_between(start, end))
    assert outermost_between[0] == start
    assert outermost_between[-1] == end


@given(outer_vertices)
def test_outermost_single_point(vertex: SliceCoord) -> None:
    outermost_between = list(BUILDER.outermost_vertices_between(vertex, vertex))
    assert len(outermost_between) == 1
    assert outermost_between[0] == vertex


def test_outermost_quads_empty() -> None:
    builder = ElementBuilder(MOCK_MESH, simple_trace, 0.1, EMPTY_MAP)
    assert len(list(builder.outermost_quads())) == 0


def test_outermost_quads_between() -> None:
    ordered_outermost = list(BUILDER.outermost_vertices())
    outermost_between = FrozenCoordSet(
        itertools.chain.from_iterable(
            q.shape([0.0, 1.0]).iter_points()
            for q in BUILDER.outermost_quads_between(
                ordered_outermost[0], ordered_outermost[-1]
            )
        )
    )
    assert FrozenCoordSet(ordered_outermost) == outermost_between


def test_outermost_quads() -> None:
    ordered_outermost = list(BUILDER.outermost_vertices())
    quads = list(BUILDER.outermost_quads())
    outermost = FrozenCoordSet(
        itertools.chain.from_iterable(q.shape([0.0, 1.0]).iter_points() for q in quads)
    )
    assert FrozenCoordSet(ordered_outermost) == outermost
    assert len(ordered_outermost) == len(quads)


@given(outer_vertices, outer_vertices)
def test_outermost_quads_between_termini(start: SliceCoord, end: SliceCoord) -> None:
    outermost_between = list(BUILDER.outermost_quads_between(start, end))
    assert start in FrozenCoordSet(outermost_between[0].shape([0.0, 1.0]).iter_points())
    assert end in FrozenCoordSet(outermost_between[-1].shape([0.0, 1.0]).iter_points())


def test_unfinished_outermost_vertices_between() -> None:
    start = UNFINISHED_SOUTH_WEST[2]
    end = SOUTH_OUTER[3]
    expected = list(
        SliceCoords(
            np.concatenate((UNFINISHED_SOUTH_WEST.x1[2::-1], SOUTH_OUTER.x1[1:4])),
            np.concatenate((UNFINISHED_SOUTH_WEST.x2[2::-1], SOUTH_OUTER.x2[1:4])),
            CoordinateSystem.CYLINDRICAL,
        ).iter_points()
    )
    actual = list(BUILDER_UNFINISHED.outermost_vertices_between(start, end))
    assert expected == actual


@given(
    sampled_from(list(NOT_IN_UNFINISHED.iter_points())),
    sampled_from(list(NORTH_OUTER.iter_points()) + list(SOUTH_OUTER.iter_points())),
)
def test_unifinished_vertices_between_no_start(
    start: SliceCoord, end: SliceCoord
) -> None:
    with pytest.raises(ValueError):
        _ = BUILDER_UNFINISHED.outermost_vertices_between(start, end)


@given(
    sampled_from(list(NORTH_OUTER.iter_points()) + list(SOUTH_OUTER.iter_points())),
    sampled_from(list(NOT_IN_UNFINISHED.iter_points())),
)
def test_unifinished_vertices_between_no_end(
    start: SliceCoord, end: SliceCoord
) -> None:
    with pytest.raises(ValueError):
        _ = BUILDER_UNFINISHED.outermost_vertices_between(start, end)


@given(
    sampled_from(list(NORTH_OUTER.iter_points())),
    sampled_from(list(SOUTH_OUTER.iter_points())),
)
def test_unfinished_vertices_between_different_fragments(
    start: SliceCoord, end: SliceCoord
) -> None:
    with pytest.raises(ValueError):
        _ = BUILDER_UNFINISHED.outermost_vertices_between(start, end)


@patch(
    "neso_fame.element_builder.flux_surface_edge",
    lambda _, north, south: straight_line_across_field(north, south),
)
def test_complex_outermost_vertices() -> None:
    # Leave out a few elements to test a more complex shape of the outermost edges
    unused_outer_point = WEST_OUTER[8]
    complex_outermost = (
        FrozenCoordSet(EAST_OUTER.iter_points())
        | FrozenCoordSet(p for p in WEST_OUTER.iter_points() if p != unused_outer_point)
        | FrozenCoordSet(NORTH_OUTER.iter_points())
        | FrozenCoordSet(SOUTH_OUTER.iter_points())
        | FrozenCoordSet(
            SliceCoords(
                R[7:10, 1], Z[7:10, 1], CoordinateSystem.CYLINDRICAL
            ).iter_points()
        )
    )
    builder = ElementBuilder(MOCK_MESH, simple_trace, 0.1, EMPTY_MAP)
    for corners in zip(
        *map(operator.methodcaller("iter_points"), _element_corners(R, Z))
    ):
        if unused_outer_point not in corners:
            _ = builder.make_element(*corners)

    ordered_outermost = list(builder.outermost_vertices())
    assert len(ordered_outermost) == len(complex_outermost)
    assert FrozenCoordSet(ordered_outermost) == complex_outermost


def test_innermost_vertices_empty() -> None:
    builder = ElementBuilder(MOCK_MESH, simple_trace, 0.1, EMPTY_MAP)
    assert len(list(builder.innermost_vertices())) == 0


def test_innermost_vertices() -> None:
    ordered_innermost = list(BUILDER.innermost_vertices())
    assert len(ordered_innermost) == len(INNERMOST)
    assert FrozenCoordSet(ordered_innermost) == INNERMOST


def test_innermost_quads_empty() -> None:
    builder = ElementBuilder(MOCK_MESH, simple_trace, 0.1, EMPTY_MAP)
    assert len(list(builder.innermost_quads())) == 0


def test_innermost_quads() -> None:
    ordered_innermost = list(BUILDER.innermost_vertices())
    quads = list(BUILDER.innermost_quads())
    innermost = FrozenCoordSet(
        itertools.chain.from_iterable(q.shape([0.0, 1.0]).iter_points() for q in quads)
    )
    assert FrozenCoordSet(ordered_innermost) == innermost
    assert len(ordered_innermost) == len(quads)
