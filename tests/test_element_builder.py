import itertools
import operator
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import numpy.typing as npt
import pytest
from hypnotoad import Mesh as HypnoMesh  # type: ignore
from hypnotoad import Point2D  # type: ignore
from hypothesis import given, settings
from hypothesis.strategies import (
    booleans,
    composite,
    floats,
    integers,
    sampled_from,
    shared,
    tuples,
)

from neso_fame.element_builder import ElementBuilder
from neso_fame.generators import _get_element_corners
from neso_fame.mesh import (
    CoordinateSystem,
    FieldTracer,
    SliceCoord,
    SliceCoords,
    StraightLineAcrossField,
)

from .conftest import simple_trace
from .test_hypnotoad import real_meshes

shared_meshes = shared(real_meshes, key=-45)


@composite
def perpendicular_points(draw: Any, mesh: HypnoMesh) -> tuple[SliceCoord, SliceCoord]:
    region = draw(sampled_from(list(mesh.regions.values())))
    shape = region.Rxy.corners.shape
    i = draw(integers(0, shape[0] - 2))
    j = draw(integers(0, shape[1] - 1))
    if draw(booleans()):
        i1 = i + 1
        i2 = i
    else:
        i1 = i
        i2 = i + 1
    return SliceCoord(
        region.Rxy.corners[i1, j],
        region.Zxy.corners[i1, j],
        CoordinateSystem.CYLINDRICAL,
    ), SliceCoord(
        region.Rxy.corners[i2, j],
        region.Zxy.corners[i2, j],
        CoordinateSystem.CYLINDRICAL,
    )


@composite
def flux_surface_points(draw: Any, mesh: HypnoMesh) -> tuple[SliceCoord, SliceCoord]:
    region = draw(sampled_from(list(mesh.regions.values())))
    shape = region.Rxy.corners.shape
    i = draw(integers(0, shape[0] - 1))
    j = draw(integers(0, shape[1] - 2))
    if draw(booleans()):
        j1 = j + 1
        j2 = j
    else:
        j1 = j
        j2 = j + 1
    return SliceCoord(
        region.Rxy.corners[i, j1],
        region.Zxy.corners[i, j1],
        CoordinateSystem.CYLINDRICAL,
    ), SliceCoord(
        region.Rxy.corners[i, j2],
        region.Zxy.corners[i, j2],
        CoordinateSystem.CYLINDRICAL,
    )


@composite
def connectable_to_o_points(draw: Any, mesh: HypnoMesh) -> tuple[SliceCoord, SliceCoord]:
    region = draw(sampled_from([r for r in mesh.regions.values() if r.name.endswith("core(0)")]))
    shape = region.Rxy.corners.shape
    i = draw(integers(0, shape[0] - 2))
    j = draw(integers(0, shape[1] - 2))
    if draw(booleans()):
        j1 = j + 1
        j2 = j
    else:
        j1 = j
        j2 = j + 1
    return SliceCoord(
        region.Rxy.corners[i, j1],
        region.Zxy.corners[i, j1],
        CoordinateSystem.CYLINDRICAL,
    ), SliceCoord(
        region.Rxy.corners[i, j2],
        region.Zxy.corners[i, j2],
        CoordinateSystem.CYLINDRICAL,
    )


@composite
def quad_points(
    draw: Any, mesh: HypnoMesh
) -> tuple[SliceCoord, SliceCoord, SliceCoord, SliceCoord]:
    region = draw(sampled_from(list(mesh.regions.values())))
    shape = region.Rxy.corners.shape
    i = draw(integers(0, shape[0] - 2))
    j = draw(integers(0, shape[1] - 2))
    if draw(booleans()):
        i1 = i + 1
        i2 = i
    else:
        i1 = i
        i2 = i + 1
    if draw(booleans()):
        j1 = j + 1
        j2 = j
    else:
        j1 = j
        j2 = j + 1
    return (
        SliceCoord(
            region.Rxy.corners[i1, j1],
            region.Zxy.corners[i1, j1],
            CoordinateSystem.CYLINDRICAL,
        ),
        SliceCoord(
            region.Rxy.corners[i1, j2],
            region.Zxy.corners[i1, j2],
            CoordinateSystem.CYLINDRICAL,
        ),
        SliceCoord(
            region.Rxy.corners[i2, j1],
            region.Zxy.corners[i2, j1],
            CoordinateSystem.CYLINDRICAL,
        ),
        SliceCoord(
            region.Rxy.corners[i2, j2],
            region.Zxy.corners[i2, j2],
            CoordinateSystem.CYLINDRICAL,
        ),
    )


@settings(deadline=None)
@given(
    shared_meshes,
    shared_meshes.flatmap(perpendicular_points),
    integers(2, 20),
    floats(1e-3, 1e3),
)
def test_perpendicular_quad(
    mesh: HypnoMesh,
    termini: tuple[SliceCoord, SliceCoord],
    interp_resolution: int,
    dx3: float,
) -> None:
    trace = FieldTracer(simple_trace, interp_resolution)
    builder = ElementBuilder(mesh, trace, dx3, frozenset())
    north, south = termini
    quad = builder.perpendicular_quad(north, south)
    assert quad.shape(0.0).to_coord() == north
    assert quad.shape(1.0).to_coord() == south
    assert quad.field == trace
    assert quad.dx3 == dx3


@settings(deadline=None)
@given(
    shared_meshes,
    shared_meshes.flatmap(flux_surface_points),
    integers(2, 20),
    floats(1e-3, 1e3),
)
def test_flux_surface_quad(
    mesh: HypnoMesh,
    termini: tuple[SliceCoord, SliceCoord],
    interp_resolution: int,
    dx3: float,
) -> None:
    trace = FieldTracer(simple_trace, interp_resolution)
    builder = ElementBuilder(mesh, trace, dx3, frozenset())
    north, south = termini
    quad = builder.flux_surface_quad(north, south)
    assert quad.shape(0.0).to_coord() == north
    assert quad.shape(1.0).to_coord() == south
    assert quad.field == trace
    assert quad.dx3 == dx3


@settings(deadline=None)
@given(
    shared_meshes,
    shared_meshes.flatmap(connectable_to_o_points).flatmap(sampled_from),
    integers(2, 20),
    floats(1e-3, 1e3),
)
def test_connecting_quad(
    mesh: HypnoMesh,
    point: SliceCoord,
    interp_resolution: int,
    dx3: float,
) -> None:
    trace = FieldTracer(simple_trace, interp_resolution)
    builder = ElementBuilder(mesh, trace, dx3, frozenset())
    quad = builder.make_connecting_quad(point)
    assert quad.shape(0.0).to_coord() == point
    assert quad.shape(1.0).to_coord() == SliceCoord(
        mesh.equilibrium.o_point.R,
        mesh.equilibrium.o_point.Z,
        CoordinateSystem.CYLINDRICAL,
    )
    assert quad.field == trace
    assert quad.dx3 == dx3


@settings(deadline=None)
@given(
    shared_meshes,
    shared_meshes.flatmap(quad_points),
    integers(2, 20),
    floats(1e-3, 1e3),
)
def test_make_hex(
    mesh: HypnoMesh,
    termini: tuple[SliceCoord, SliceCoord, SliceCoord, SliceCoord],
    interp_resolution: int,
    dx3: float,
) -> None:
    trace = FieldTracer(simple_trace, interp_resolution)
    builder = ElementBuilder(mesh, trace, dx3, frozenset())
    hexa = builder.make_hex(*termini)
    assert frozenset(hexa.corners().to_slice_coords().iter_points()) == frozenset(
        termini
    )
    for quad in hexa:
        assert quad.field == trace
        assert quad.dx3 == dx3


@settings(deadline=None)
@given(
    shared_meshes,
    shared_meshes.flatmap(connectable_to_o_points),
    integers(2, 20),
    floats(1e-3, 1e3),
)
def test_make_prism(
    mesh: HypnoMesh,
    termini: tuple[SliceCoord, SliceCoord],
    interp_resolution: int,
    dx3: float,
) -> None:
    trace = FieldTracer(simple_trace, interp_resolution)
    builder = ElementBuilder(mesh, trace, dx3, frozenset())
    north, south = termini
    o_point = SliceCoord(
        mesh.equilibrium.o_point.R,
        mesh.equilibrium.o_point.Z,
        CoordinateSystem.CYLINDRICAL,
    )
    prism = builder.make_prism(north, south)
    assert frozenset(prism.corners().to_slice_coords().iter_points()) == frozenset(
        termini + (o_point,)
    )
    for quad in prism:
        assert quad.field == trace
        assert quad.dx3 == dx3


def _element_corners(
    R: npt.NDArray, Z: npt.NDArray
) -> tuple[SliceCoords, SliceCoords, SliceCoords, SliceCoords]:
    Rs = _get_element_corners(R)
    Zs = _get_element_corners(Z)
    return (
        SliceCoords(Rs[0], Zs[0], CoordinateSystem.CYLINDRICAL),
        SliceCoords(Rs[1], Zs[1], CoordinateSystem.CYLINDRICAL),
        SliceCoords(Rs[2], Zs[2], CoordinateSystem.CYLINDRICAL),
        SliceCoords(Rs[3], Zs[3], CoordinateSystem.CYLINDRICAL),
    )


R, Z = np.meshgrid(np.linspace(0.5, 1.5, 11), np.linspace(-1, 1, 11))
WEST = SliceCoords(R[:, 0], Z[:, 0], CoordinateSystem.CYLINDRICAL)
EAST = SliceCoords(R[:, -1], Z[:, -1], CoordinateSystem.CYLINDRICAL)
SOUTH = SliceCoords(R[0, :], Z[0, :], CoordinateSystem.CYLINDRICAL)
NORTH = SliceCoords(R[-1, :], Z[-1, :], CoordinateSystem.CYLINDRICAL)
OUTERMOST = (
    frozenset(EAST.iter_points())
    | frozenset(WEST.iter_points())
    | frozenset(NORTH.iter_points())
    | frozenset(SOUTH.iter_points())
)
MOCK_MESH = MagicMock()
MOCK_MESH.equilibrium.o_point = Point2D(1.0, 0.0)
BUILDER = ElementBuilder(MOCK_MESH, FieldTracer(simple_trace, 10), 0.1, OUTERMOST)
BUILDER_UNFINISHED = ElementBuilder(
    MOCK_MESH, FieldTracer(simple_trace, 10), 0.1, OUTERMOST
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
with patch(
    "neso_fame.element_builder.flux_surface_edge",
    lambda _, north, south: StraightLineAcrossField(north, south),
), patch(
    "neso_fame.element_builder.perpendicular_edge",
    lambda _, north, south: StraightLineAcrossField(north, south),
):
    for corners in zip(
        *map(operator.methodcaller("iter_points"), _element_corners(R, Z))
    ):
        _ = BUILDER.make_hex(*corners)
    # Don't build the middle elements
    for corners in itertools.chain(
        zip(
            *map(
                operator.methodcaller("iter_points"),
                _element_corners(R[:4, :], Z[:4, :]),
            )
        ),
        zip(
            *map(
                operator.methodcaller("iter_points"),
                _element_corners(R[7:, :], Z[7:, :]),
            )
        ),
    ):
        _ = BUILDER_UNFINISHED.make_hex(*corners)

outer_vertices = sampled_from(list(OUTERMOST))


def test_outermost_vertices() -> None:
    ordered_outermost = list(BUILDER.outermost_vertices())
    assert len(ordered_outermost) == len(OUTERMOST)
    assert frozenset(ordered_outermost) == OUTERMOST


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


def test_unfinished_outermost_vertices_between() -> None:
    start = UNFINISHED_SOUTH_WEST[2]
    end = SOUTH[3]
    expected = list(
        SliceCoords(
            np.concatenate((UNFINISHED_SOUTH_WEST.x1[2::-1], SOUTH.x1[1:4])),
            np.concatenate((UNFINISHED_SOUTH_WEST.x2[2::-1], SOUTH.x2[1:4])),
            CoordinateSystem.CYLINDRICAL,
        ).iter_points()
    )
    actual = list(BUILDER_UNFINISHED.outermost_vertices_between(start, end))
    assert expected == actual


def test_unfinished_outermost_vertices() -> None:
    with pytest.raises(ValueError):
        _ = BUILDER_UNFINISHED.outermost_vertices()


@given(
    sampled_from(list(NOT_IN_UNFINISHED.iter_points())),
    sampled_from(list(NORTH.iter_points()) + list(SOUTH.iter_points())),
)
def test_unifinished_vertices_between_no_start(
    start: SliceCoord, end: SliceCoord
) -> None:
    with pytest.raises(ValueError):
        _ = BUILDER_UNFINISHED.outermost_vertices_between(start, end)


@given(
    sampled_from(list(NORTH.iter_points()) + list(SOUTH.iter_points())),
    sampled_from(list(NOT_IN_UNFINISHED.iter_points())),
)
def test_unifinished_vertices_between_no_end(
    start: SliceCoord, end: SliceCoord
) -> None:
    with pytest.raises(ValueError):
        _ = BUILDER_UNFINISHED.outermost_vertices_between(start, end)


@given(sampled_from(list(NORTH.iter_points())), sampled_from(list(SOUTH.iter_points())))
def test_unfinished_vertices_between_different_fragments(
    start: SliceCoord, end: SliceCoord
) -> None:
    with pytest.raises(ValueError):
        _ = BUILDER_UNFINISHED.outermost_vertices_between(start, end)


@given(
    sampled_from(list(UNFINISHED_NORTH_WEST.iter_points())),
    sampled_from(list(UNFINISHED_NORTH_EAST.iter_points())),
)
def test_unfinished_vertices_wrong_order1(start: SliceCoord, end: SliceCoord) -> None:
    with pytest.raises(ValueError):
        _ = BUILDER_UNFINISHED.outermost_vertices_between(start, end)


@given(
    sampled_from(list(UNFINISHED_SOUTH_EAST.iter_points())),
    sampled_from(list(UNFINISHED_SOUTH_WEST.iter_points())),
)
def test_unfinished_vertices_wrong_order2(start: SliceCoord, end: SliceCoord) -> None:
    with pytest.raises(ValueError):
        _ = BUILDER_UNFINISHED.outermost_vertices_between(start, end)
