import itertools
import operator
from functools import cache
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import numpy.typing as npt
import pytest
from hypnotoad import Mesh as HypnoMesh  # type: ignore
from hypnotoad import (
    MeshRegion,
    Point2D,  # type: ignore
)
from hypothesis import given, settings
from hypothesis.strategies import (
    SearchStrategy,
    booleans,
    composite,
    floats,
    integers,
    lists,
    sampled_from,
    shared,
    tuples,
)

from neso_fame.coordinates import (
    CoordinateSystem,
    CoordMap,
    FrozenCoordSet,
    SliceCoord,
    SliceCoords,
)
from neso_fame.element_builder import ElementBuilder
from neso_fame.generators import _get_element_corners
from neso_fame.mesh import (
    FieldTracer,
    straight_line_across_field,
)

from .conftest import simple_trace
from .test_hypnotoad import real_meshes

shared_meshes = shared(real_meshes, key=-45)
# Filter out core region for single-null meshes, as the periodicity
# causes some confusion when testing boundary-related stuff
shared_regions = shared(
    real_meshes.flatmap(lambda mesh: sampled_from(list(mesh.regions.values()))).filter(
        lambda x: x.equilibriumRegion.name != "core"
    ),
    key=-46,
)


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


EMPTY_MAP = CoordMap.empty_slicecoord(float)


@composite
def points_in_region(draw: Any, region: MeshRegion) -> SliceCoord:
    shape = region.Rxy.corners.shape
    i = draw(integers(0, shape[0] - 1))
    j = draw(integers(0, shape[1] - 1))
    return SliceCoord(
        region.Rxy.corners[i, j], region.Zxy.corners[i, j], CoordinateSystem.CYLINDRICAL
    )


def points_in_mesh(mesh: HypnoMesh) -> SearchStrategy[SliceCoord]:
    return sampled_from(list(mesh.regions.values())).flatmap(points_in_region)


def point_pairs(mesh: HypnoMesh) -> SearchStrategy[tuple[SliceCoord, SliceCoord]]:
    return tuples(points_in_mesh(mesh), points_in_mesh(mesh))


@cache
def _get_interiors(region: MeshRegion) -> list[tuple[npt.NDArray, ...]]:
    return list(
        np.nditer([region.Rxy.corners[1:-1, 1:-1], region.Zxy.corners[1:-1, 1:-1]])
    )


def region_interior_points(region: MeshRegion) -> SearchStrategy[SliceCoord]:
    return sampled_from(_get_interiors(region)).map(
        lambda x: SliceCoord(float(x[0]), float(x[1]), CoordinateSystem.CYLINDRICAL)
    )


@cache
def _get_boundaries(region: MeshRegion) -> list[tuple[npt.NDArray, ...]]:
    return (
        list(np.nditer([region.Rxy.corners[0, :], region.Zxy.corners[0, :]]))
        + list(np.nditer([region.Rxy.corners[-1, :], region.Zxy.corners[-1, :]]))
        + list(np.nditer([region.Rxy.corners[1:-1, 0], region.Zxy.corners[1:-1, 0]]))
        + list(np.nditer([region.Rxy.corners[1:-1, -1], region.Zxy.corners[1:-1, -1]]))
    )


@cache
def _get_boundary_weights(region: MeshRegion) -> CoordMap[SliceCoord, float]:
    return CoordMap(
        {
            SliceCoord(float(p[0]), float(p[1]), CoordinateSystem.CYLINDRICAL): 0.0
            for p in _get_boundaries(region)
        }
    )


def region_boundary_points(region: MeshRegion) -> SearchStrategy[SliceCoord]:
    return sampled_from(_get_boundaries(region)).map(
        lambda x: SliceCoord(float(x[0]), float(x[1]), CoordinateSystem.CYLINDRICAL)
    )


def builder_for_region(
    region: MeshRegion, interp_resolution: int = 5, dx3: float = 1e-1
) -> ElementBuilder:
    """Return an element builder with all elements already built for the region."""
    trace = FieldTracer(simple_trace, interp_resolution)
    builder = ElementBuilder(
        region.meshParent, trace, dx3, _get_boundary_weights(region)
    )
    for p1, p2, p3, p4 in zip(
        *map(
            operator.methodcaller("iter_points"),
            _element_corners(region.Rxy.corners, region.Zxy.corners),
        )
    ):
        _ = builder.make_element(p1, p2, p3, p4)
    return builder


@composite
def connectable_to_o_points(
    draw: Any, mesh: HypnoMesh
) -> tuple[SliceCoord, SliceCoord]:
    region = draw(
        sampled_from([r for r in mesh.regions.values() if r.name.endswith("core(0)")])
    )
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
    builder = ElementBuilder(mesh, trace, dx3, EMPTY_MAP)
    north, south = termini
    quad = builder._perpendicular_quad(north, south)
    assert quad.shape(0.0).to_coord() == north
    assert quad.shape(1.0).to_coord() == south
    assert quad.field == trace
    assert quad.dx3 == dx3
    assert quad.north_start_weight == 0.0
    assert quad.south_start_weight == 0.0


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
    builder = ElementBuilder(mesh, trace, dx3, EMPTY_MAP)
    north, south = termini
    quad = builder._flux_surface_quad(north, south)
    assert quad.shape(0.0).to_coord() == north
    assert quad.shape(1.0).to_coord() == south
    assert quad.field == trace
    assert quad.dx3 == dx3
    assert quad.north_start_weight == 0.0
    assert quad.south_start_weight == 0.0


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
    builder = ElementBuilder(mesh, trace, dx3, EMPTY_MAP)
    quad = builder.make_quad_to_o_point(point)
    assert quad.shape(0.0).to_coord() == point
    assert quad.shape(1.0).to_coord() == SliceCoord(
        mesh.equilibrium.o_point.R,
        mesh.equilibrium.o_point.Z,
        CoordinateSystem.CYLINDRICAL,
    )
    assert quad.field == trace
    assert quad.dx3 == dx3
    assert quad.north_start_weight == 0.0
    assert quad.south_start_weight == 0.0


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
    builder = ElementBuilder(mesh, trace, dx3, EMPTY_MAP)
    hexa = builder.make_element(*termini)
    assert FrozenCoordSet(
        hexa.corners().to_slice_coords().iter_points()
    ) == FrozenCoordSet(termini)
    for quad in hexa:
        assert quad.field == trace
        assert quad.dx3 == dx3
        assert quad.north_start_weight == 0.0
        assert quad.south_start_weight == 0.0


@settings(deadline=None)
@given(
    shared_meshes,
    shared_meshes.flatmap(quad_points),
    integers(2, 20),
    floats(1e-3, 1e3),
)
def test_make_prism(
    mesh: HypnoMesh,
    termini: tuple[SliceCoord, SliceCoord, SliceCoord, SliceCoord],
    interp_resolution: int,
    dx3: float,
) -> None:
    trace = FieldTracer(simple_trace, interp_resolution)
    builder = ElementBuilder(mesh, trace, dx3, EMPTY_MAP)
    prism = builder.make_element(*termini[:3], None)
    assert FrozenCoordSet(
        prism.corners().to_slice_coords().iter_points()
    ) == FrozenCoordSet(termini[:3])
    for quad in prism:
        assert quad.field == trace
        assert quad.dx3 == dx3
        assert quad.north_start_weight == 0.0
        assert quad.south_start_weight == 0.0


@settings(deadline=None)
@given(
    shared_meshes,
    shared_meshes.flatmap(connectable_to_o_points),
    integers(2, 20),
    floats(1e-3, 1e3),
)
def test_make_prism_to_centre(
    mesh: HypnoMesh,
    termini: tuple[SliceCoord, SliceCoord],
    interp_resolution: int,
    dx3: float,
) -> None:
    trace = FieldTracer(simple_trace, interp_resolution)
    builder = ElementBuilder(mesh, trace, dx3, EMPTY_MAP)
    north, south = termini
    o_point = SliceCoord(
        mesh.equilibrium.o_point.R,
        mesh.equilibrium.o_point.Z,
        CoordinateSystem.CYLINDRICAL,
    )
    prism = builder.make_prism_to_centre(north, south)
    assert FrozenCoordSet(
        prism.corners().to_slice_coords().iter_points()
    ) == FrozenCoordSet(termini + (o_point,))
    for quad in prism:
        assert quad.field == trace
        assert quad.dx3 == dx3


@settings(deadline=None)
@given(
    shared_meshes,
    shared_meshes.flatmap(point_pairs),
    integers(2, 20),
    floats(1e-3, 1e3),
)
def test_quad_for_prism(
    mesh: HypnoMesh,
    points: tuple[SliceCoord, SliceCoord],
    interp_resolution: int,
    dx3: float,
) -> None:
    trace = FieldTracer(simple_trace, interp_resolution)
    builder = ElementBuilder(mesh, trace, dx3, EMPTY_MAP)
    quad, boundary_quads = builder.make_quad_for_prism(*points, frozenset())
    assert quad.shape(0.0).to_coord().approx_eq(points[0])
    assert quad.shape(1.0).to_coord().approx_eq(points[1])
    assert quad.field == trace
    assert quad.dx3 == dx3
    assert quad.north_start_weight == 1
    assert quad.south_start_weight == 1
    assert len(boundary_quads) == 0


@settings(deadline=None)
@given(
    shared_meshes,
    shared_meshes.flatmap(point_pairs),
    integers(2, 20),
    floats(1e-3, 1e3),
)
def test_quad_for_prism_order_invariant(
    mesh: HypnoMesh,
    points: tuple[SliceCoord, SliceCoord],
    interp_resolution: int,
    dx3: float,
) -> None:
    trace = FieldTracer(simple_trace, interp_resolution)
    builder = ElementBuilder(mesh, trace, dx3, EMPTY_MAP)
    assert (
        builder.make_quad_for_prism(points[0], points[1], frozenset())[0]
        == builder.make_quad_for_prism(points[1], points[0], frozenset())[0]
    )


@settings(deadline=None)
@given(
    shared_regions.map(builder_for_region),
    shared_regions.flatmap(region_boundary_points),
    shared_regions.flatmap(region_boundary_points),
)
def test_quad_for_prism_aligned(
    builder: ElementBuilder,
    point1: SliceCoord,
    point2: SliceCoord,
) -> None:
    quad, _ = builder.make_quad_for_prism(point1, point2, frozenset())
    assert quad.north_start_weight == 0
    assert quad.south_start_weight == 0


@settings(deadline=None)
@given(
    shared_regions.map(builder_for_region),
    shared_regions.flatmap(region_boundary_points),
    shared_regions.flatmap(region_interior_points),
)
def test_quad_for_prism_aligned_north(
    builder: ElementBuilder,
    point1: SliceCoord,
    point2: SliceCoord,
) -> None:
    quad, _ = builder.make_quad_for_prism(point1, point2, frozenset())
    assert quad.north_start_weight == 0
    assert quad.south_start_weight == 1


@settings(deadline=None)
@given(
    shared_regions.map(builder_for_region),
    shared_regions.flatmap(region_interior_points),
    shared_regions.flatmap(region_boundary_points),
    shared_regions,
)
def test_quad_for_prism_aligned_south(
    builder: ElementBuilder,
    point1: SliceCoord,
    point2: SliceCoord,
    region: MeshRegion,
) -> None:
    quad, _ = builder.make_quad_for_prism(point1, point2, frozenset())
    assert quad.north_start_weight == 1
    assert quad.south_start_weight == 0


@settings(deadline=None)
@given(
    shared_meshes,
    shared_meshes.flatmap(point_pairs),
    integers(2, 20),
    floats(1e-3, 1e3),
)
def test_quad_on_wall_for_prism(
    mesh: HypnoMesh,
    points: tuple[SliceCoord, SliceCoord],
    interp_resolution: int,
    dx3: float,
) -> None:
    trace = FieldTracer(simple_trace, interp_resolution)
    builder = ElementBuilder(mesh, trace, dx3, EMPTY_MAP)
    q, b = builder.make_quad_for_prism(points[0], points[1], frozenset({points}))
    assert len(b) == 1
    assert q in b


@settings(deadline=None)
@given(
    shared_meshes,
    shared_meshes.flatmap(point_pairs),
    sampled_from([0, 1]),
    integers(2, 20),
    floats(1e-3, 1e3),
)
def test_quad_not_quite_on_wall_for_prism(
    mesh: HypnoMesh,
    points: tuple[SliceCoord, SliceCoord],
    on_wall: int,
    interp_resolution: int,
    dx3: float,
) -> None:
    trace = FieldTracer(simple_trace, interp_resolution)
    builder = ElementBuilder(mesh, trace, dx3, EMPTY_MAP)
    q, b = builder.make_quad_for_prism(points[0], points[1], frozenset())
    assert len(b) == 0


@settings(deadline=None)
@given(
    shared_meshes,
    shared_meshes.flatmap(point_pairs),
    integers(2, 20),
    floats(1e-3, 1e3),
)
def test_make_wall_quad(
    mesh: HypnoMesh,
    points: tuple[SliceCoord, SliceCoord],
    interp_resolution: int,
    dx3: float,
) -> None:
    trace = FieldTracer(simple_trace, interp_resolution)
    builder = ElementBuilder(mesh, trace, dx3, EMPTY_MAP)
    shape = straight_line_across_field(*points)
    q1 = builder.make_wall_quad_for_prism(shape)
    assert q1.shape is shape
    p1, p2 = shape([1.0, 0.0]).iter_points()
    q2, b = builder.make_quad_for_prism(p1, p2, frozenset())
    assert q2 is q1
    assert q2 in b


@settings(deadline=None)
@given(
    shared_meshes,
    shared_meshes.flatmap(point_pairs),
    integers(2, 20),
    floats(1e-3, 1e3),
)
def test_make_wall_quad_caching(
    mesh: HypnoMesh,
    points: tuple[SliceCoord, SliceCoord],
    interp_resolution: int,
    dx3: float,
) -> None:
    trace = FieldTracer(simple_trace, interp_resolution)
    builder = ElementBuilder(mesh, trace, dx3, EMPTY_MAP)
    shape = straight_line_across_field(*points)
    q1 = builder.make_wall_quad_for_prism(shape)
    q2 = builder.make_wall_quad_for_prism(shape)
    assert q1 is q2


@settings(deadline=None)
@given(
    shared_meshes,
    shared_meshes.flatmap(quad_points),
    integers(2, 20),
    floats(1e-3, 1e3),
)
def test_make_outer_prism(
    mesh: HypnoMesh,
    termini: tuple[SliceCoord, SliceCoord, SliceCoord, SliceCoord],
    interp_resolution: int,
    dx3: float,
) -> None:
    trace = FieldTracer(simple_trace, interp_resolution)
    prism_termini = termini[:3]
    builder = ElementBuilder(mesh, trace, dx3, EMPTY_MAP)
    prism, bounds = builder.make_outer_prism(*prism_termini, frozenset())
    assert FrozenCoordSet(
        prism.corners().to_slice_coords().iter_points()
    ) == FrozenCoordSet(prism_termini)
    for quad in prism:
        assert quad.field == trace
        assert quad.dx3 == dx3
    assert len(bounds) == 0


@settings(deadline=None)
@given(
    shared_meshes,
    shared_meshes.flatmap(quad_points),
    integers(0, 2),
    integers(2, 20),
    floats(1e-3, 1e3),
)
def test_make_outer_prism_no_bounds(
    mesh: HypnoMesh,
    termini: tuple[SliceCoord, SliceCoord, SliceCoord, SliceCoord],
    bound_point: int,
    interp_resolution: int,
    dx3: float,
) -> None:
    trace = FieldTracer(simple_trace, interp_resolution)
    prism_termini = termini[:3]
    builder = ElementBuilder(mesh, trace, dx3, EMPTY_MAP)
    prism, bounds = builder.make_outer_prism(*prism_termini, frozenset())
    assert FrozenCoordSet(
        prism.corners().to_slice_coords().iter_points()
    ) == FrozenCoordSet(prism_termini)
    assert len(bounds) == 0


@settings(deadline=None)
@given(
    shared_meshes,
    shared_meshes.flatmap(quad_points),
    lists(integers(0, 2), min_size=2, max_size=2, unique=True),
    integers(2, 20),
    floats(1e-3, 1e3),
)
def test_make_outer_prism_one_bound(
    mesh: HypnoMesh,
    termini: tuple[SliceCoord, SliceCoord, SliceCoord, SliceCoord],
    bound_points: list[int],
    interp_resolution: int,
    dx3: float,
) -> None:
    trace = FieldTracer(simple_trace, interp_resolution)
    prism_termini = termini[:3]
    bound_set = frozenset({(termini[bound_points[0]], termini[bound_points[1]])})
    builder = ElementBuilder(mesh, trace, dx3, EMPTY_MAP)
    prism, bounds = builder.make_outer_prism(*prism_termini, bound_set)
    assert FrozenCoordSet(
        prism.corners().to_slice_coords().iter_points()
    ) == FrozenCoordSet(prism_termini)
    assert bounds < frozenset(prism)
    assert len(bounds) == 1
    bound_quad = next(iter(bounds))
    assert FrozenCoordSet(
        bound_quad.corners().to_slice_coords().iter_points()
    ) == FrozenCoordSet(next(iter(bound_set)))


@settings(deadline=None)
@given(
    shared_meshes,
    shared_meshes.flatmap(quad_points),
    integers(2, 20),
    floats(1e-3, 1e3),
)
def test_get_element_for_quad(
    mesh: HypnoMesh,
    termini: tuple[SliceCoord, SliceCoord, SliceCoord, SliceCoord],
    interp_resolution: int,
    dx3: float,
) -> None:
    trace = FieldTracer(simple_trace, interp_resolution)
    builder = ElementBuilder(mesh, trace, dx3, EMPTY_MAP)
    prism1 = builder.make_element(*termini[:3], None)
    prism2, _ = builder.make_outer_prism(*termini[1:], frozenset())
    for q in prism1:
        if q not in prism2.sides:
            assert len(builder.get_element_for_quad(q)) == 0
        else:
            assert builder.get_element_for_quad(q) == [prism2]
    for q in prism2:
        assert builder.get_element_for_quad(q) == [prism2]


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

BUILDER = ElementBuilder(MOCK_MESH, FieldTracer(simple_trace, 10), 0.1, EMPTY_MAP)
BUILDER_UNFINISHED = ElementBuilder(
    MOCK_MESH, FieldTracer(simple_trace, 10), 0.1, EMPTY_MAP
)
with (
    patch(
        "neso_fame.element_builder.flux_surface_edge",
        lambda _, north, south: straight_line_across_field(north, south),
    ),
    patch(
        "neso_fame.element_builder.perpendicular_edge",
        lambda _, north, south: straight_line_across_field(north, south),
    ),
):
    # Create a square mesh with a hole in the middle
    for corners in itertools.chain(
        zip(
            *map(
                operator.methodcaller("iter_points"),
                _element_corners(R[:5, :], Z[:5, :]),
            )
        ),
        zip(
            *map(
                operator.methodcaller("iter_points"),
                _element_corners(R[6:, :], Z[6:, :]),
            )
        ),
        zip(
            *map(
                operator.methodcaller("iter_points"),
                _element_corners(R[4:7, :5], Z[4:7, :5]),
            )
        ),
        zip(
            *map(
                operator.methodcaller("iter_points"),
                _element_corners(R[4:7, 6:], Z[4:7, 6:]),
            )
        ),
    ):
        _ = BUILDER.make_element(*corners)
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
        _ = BUILDER_UNFINISHED.make_element(*corners)


outer_vertices = sampled_from(list(OUTERMOST))


def test_outermost_vertices_empty() -> None:
    builder = ElementBuilder(MOCK_MESH, FieldTracer(simple_trace, 10), 0.1, EMPTY_MAP)
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
    builder = ElementBuilder(MOCK_MESH, FieldTracer(simple_trace, 10), 0.1, EMPTY_MAP)
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
@patch(
    "neso_fame.element_builder.perpendicular_edge",
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
    builder = ElementBuilder(MOCK_MESH, FieldTracer(simple_trace, 10), 0.1, EMPTY_MAP)
    for corners in zip(
        *map(operator.methodcaller("iter_points"), _element_corners(R, Z))
    ):
        if unused_outer_point not in corners:
            _ = builder.make_element(*corners)

    ordered_outermost = list(builder.outermost_vertices())
    assert len(ordered_outermost) == len(complex_outermost)
    assert FrozenCoordSet(ordered_outermost) == complex_outermost


def test_innermost_vertices_empty() -> None:
    builder = ElementBuilder(MOCK_MESH, FieldTracer(simple_trace, 10), 0.1, EMPTY_MAP)
    assert len(list(builder.innermost_vertices())) == 0


def test_innermost_vertices() -> None:
    ordered_innermost = list(BUILDER.innermost_vertices())
    assert len(ordered_innermost) == len(INNERMOST)
    assert FrozenCoordSet(ordered_innermost) == INNERMOST


def test_innermost_quads_empty() -> None:
    builder = ElementBuilder(MOCK_MESH, FieldTracer(simple_trace, 10), 0.1, EMPTY_MAP)
    assert len(list(builder.innermost_quads())) == 0


def test_innermost_quads() -> None:
    ordered_innermost = list(BUILDER.innermost_vertices())
    quads = list(BUILDER.innermost_quads())
    innermost = FrozenCoordSet(
        itertools.chain.from_iterable(q.shape([0.0, 1.0]).iter_points() for q in quads)
    )
    assert FrozenCoordSet(ordered_innermost) == innermost
    assert len(ordered_innermost) == len(quads)
