from __future__ import annotations

import itertools
import operator
import os
import pathlib
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from functools import reduce
from tempfile import TemporaryDirectory
from typing import Callable, Iterator, Type, TypeGuard, TypeVar, Union, cast

import numpy as np
from hypothesis import given, settings
from hypothesis.strategies import (
    booleans,
    builds,
    from_type,
    integers,
    just,
    lists,
    sampled_from,
    shared,
)
from NekPy import LibUtilities as LU
from NekPy import SpatialDomains as SD
from pytest import approx, mark

from neso_fame import nektar_writer
from neso_fame.coordinates import (
    Coord,
    CoordinateSystem,
    FrozenCoordSet,
    SliceCoord,
    SliceCoords,
)
from neso_fame.mesh import (
    B,
    C,
    Curve,
    E,
    FieldAlignedCurve,
    GenericMesh,
    Mesh,
    MeshLayer,
    Prism,
    PrismMesh,
    PrismMeshLayer,
    PrismTypes,
    Quad,
    QuadMesh,
    QuadMeshLayer,
    UnalignedShape,
    control_points,
    field_aligned_positions,
    straight_line_across_field,
)
from neso_fame.mesh import (
    order as elem_order,
)
from neso_fame.offset import Offset

from .conftest import (
    flat_sided_hex,
    flat_sided_prism,
    linear_field_trace,
    non_nans,
    prism_meshes,
    quad_meshes,
    simple_field_aligned_curve,
    simple_trace,
)


def both_nan(a: float, b: float) -> bool:
    return cast(bool, np.isnan(a) and np.isnan(b))


def assert_nek_points_eq(actual: SD.PointGeom, expected: SD.PointGeom) -> None:
    for a, e in zip(actual.GetCoordinates(), expected.GetCoordinates()):
        assert a == approx(e) or both_nan(a, e)


def assert_points_eq(actual: SD.PointGeom, expected: Coord) -> None:
    for a, e in zip(actual.GetCoordinates(), expected.to_cartesian()):
        assert a == approx(e, 1e-8, 1e-8, True)


def poloidal_corners(element: Quad | Prism) -> Iterator[Coord]:
    coords = element.nodes.start_points
    if isinstance(element, Quad):
        yield coords[0].to_3d_coord(0.0)
        yield coords[-1].to_3d_coord(0.0)
        return
    yield coords[0, 0].to_3d_coord(0.0)
    yield coords[0, -1].to_3d_coord(0.0)
    yield coords[-1, 0].to_3d_coord(0.0)
    if element.shape == PrismTypes.RECTANGULAR:
        yield coords[-1, -1].to_3d_coord(0.0)


# FIXME: frozenset[FrozenCoordSet] won't compare properly
ComparableDimensionalGeometry = tuple[str, FrozenCoordSet[Coord]]
ComparableGeometry = Coord | ComparableDimensionalGeometry
ComparableComposite = frozenset[int]


def comparable_coord(c: Coord) -> Coord:
    return c.to_cartesian()


def comparable_nektar_point(geom: SD.PointGeom) -> Coord:
    coords = geom.GetCoordinates()
    return Coord(coords[0], coords[1], coords[2], CoordinateSystem.CARTESIAN)


def comparable_curve(curve: Curve) -> ComparableDimensionalGeometry:
    return SD.Curve.__name__, FrozenCoordSet(
        map(comparable_coord, control_points(curve).to_cartesian().iter_points())
    )


def comparable_edge(curve: Curve | FieldAlignedCurve) -> ComparableDimensionalGeometry:
    return SD.SegGeom.__name__, FrozenCoordSet(
        map(comparable_coord, control_points(curve).to_cartesian().iter_points())
    )


def comparable_quad(quad: Quad) -> ComparableDimensionalGeometry:
    return SD.QuadGeom.__name__, FrozenCoordSet(
        comparable_coord(c.to_cartesian()) for c in quad.corners()
    )


def comparable_poloidal_element(element: Prism | Quad) -> ComparableDimensionalGeometry:
    if isinstance(element, Prism):
        if element.shape == PrismTypes.TRIANGULAR:
            type_name = SD.TriGeom.__name__
        elif element.shape == PrismTypes.RECTANGULAR:
            type_name = SD.QuadGeom.__name__
        else:
            raise ValueError(f"Unrecognised prism type {element.shape}.")
    else:
        type_name = SD.SegGeom.__name__
    return type_name, FrozenCoordSet(map(comparable_coord, poloidal_corners(element)))


def comparable_prism(prism: Prism) -> ComparableDimensionalGeometry:
    if prism.shape == PrismTypes.TRIANGULAR:
        name = SD.PrismGeom.__name__
    elif prism.shape == PrismTypes.RECTANGULAR:
        name = SD.HexGeom.__name__
    else:
        raise ValueError(f"Unrecognised prism type {prism.shape}.")
    return name, FrozenCoordSet(
        comparable_coord(c.to_cartesian()) for c in prism.corners()
    )


def comparable_geometry(
    geom: SD.Geometry1D | SD.Geometry2D | SD.Geometry3D | SD.Curve,
) -> ComparableDimensionalGeometry:
    """Convert geometry elements into sets of points. This allows for
    convenient comparison.

    """
    if isinstance(geom, SD.SegGeom):
        return type(geom).__name__, FrozenCoordSet(
            comparable_nektar_point(geom.GetVertex(i)) for i in range(2)
        )
    elif isinstance(geom, SD.Geometry):
        return type(geom).__name__, FrozenCoordSet(
            comparable_nektar_point(geom.GetEdge(i).GetVertex(j))
            for j in range(2)
            for i in range(geom.GetNumEdges())
        )
    else:
        return type(geom).__name__, FrozenCoordSet(
            map(comparable_nektar_point, geom.points)
        )


def comparable_set(
    vals: Iterable[SD.Geometry1D | SD.Geometry2D | SD.Geometry3D | SD.Curve],
) -> frozenset[ComparableDimensionalGeometry]:
    return frozenset(map(comparable_geometry, vals))


def comparable_point_set(vals: Iterable[SD.PointGeom]) -> FrozenCoordSet[Coord]:
    return FrozenCoordSet(map(comparable_nektar_point, vals))


def comparable_composite(val: SD.Composite) -> frozenset[ComparableComposite]:
    return frozenset(map(operator.methodcaller("GetGlobalID"), val.geometries))


def comparable_composites(
    vals: Iterable[SD.Composite],
) -> frozenset[frozenset[ComparableComposite]]:
    return frozenset(map(comparable_composite, vals))


@mark.filterwarnings("ignore:invalid value:RuntimeWarning")
@given(from_type(Coord), integers())
def test_nektar_point(coord: Coord, i: int) -> None:
    nek_coord = nektar_writer.nektar_point(coord, 3, i)
    assert_points_eq(nek_coord, coord)


@mark.filterwarnings("ignore:invalid value:RuntimeWarning")
@given(
    lists(from_type(Coord), min_size=2, max_size=2, unique=True).filter(
        lambda x: x[0].to_cartesian() != x[1].to_cartesian()
    ),
    lists(integers(), min_size=2, max_size=2, unique=True),
)
def test_nektar_point_caching(coords: list[Coord], layers: list[int]) -> None:
    c1 = nektar_writer.nektar_point(coords[0], 3, layers[0])
    assert c1 is nektar_writer.nektar_point(coords[0], 3, layers[0])
    assert c1 is not nektar_writer.nektar_point(coords[1], 3, layers[0])
    assert c1 is not nektar_writer.nektar_point(coords[0], 3, layers[1])
    assert c1 is not nektar_writer.nektar_point(coords[0], 2, layers[0])


@given(simple_field_aligned_curve, integers())
def test_nektar_curve(curve: FieldAlignedCurve, layer: int) -> None:
    nek_curve, (start, end) = nektar_writer.nektar_curve(curve, 3, layer)
    assert nek_curve.ptype == LU.PointsType.PolyEvenlySpaced
    assert_nek_points_eq(nek_curve.points[0], start)
    assert_nek_points_eq(nek_curve.points[-1], end)
    assert len(nek_curve.points) == elem_order(curve) + 1


def test_circular_nektar_curve() -> None:
    curve = Offset(
        FieldAlignedCurve(
            field_aligned_positions(
                SliceCoords(np.array(1.0), np.array(0.0), CoordinateSystem.CYLINDRICAL),
                np.pi,
                linear_field_trace(
                    0.0, 0.2, np.pi, CoordinateSystem.CYLINDRICAL, 0, (0, 0)
                ),
                np.array(1.0),
                2,
            )
        ),
        x3_offset=np.pi / 2,
    )
    nek_curve, _ = nektar_writer.nektar_curve(curve, 3, 0)
    assert_points_eq(
        nek_curve.points[0], Coord(1.0, 0.0, -0.1, CoordinateSystem.CARTESIAN)
    )
    assert_points_eq(
        nek_curve.points[1], Coord(0.0, 1.0, 0.0, CoordinateSystem.CARTESIAN)
    )
    assert_points_eq(
        nek_curve.points[2], Coord(-1.0, 0.0, 0.1, CoordinateSystem.CARTESIAN)
    )


@given(simple_field_aligned_curve, integers())
def test_nektar_edge_higher_order(curve: FieldAlignedCurve, layer: int) -> None:
    nek_edge, (start, end) = nektar_writer.nektar_edge(curve, 3, layer)
    nek_curve = nek_edge.GetCurve()
    assert_nek_points_eq(nek_edge.GetVertex(0), start)
    assert_nek_points_eq(nek_curve.points[0], start)
    assert_nek_points_eq(nek_edge.GetVertex(1), end)
    assert_nek_points_eq(nek_curve.points[-1], end)
    n = elem_order(curve)
    if n > 1:
        assert nek_curve is not None
        assert len(nek_curve.points) == n + 1
        assert_points_eq(start, curve.coords[0])
        assert_points_eq(end, curve.coords[-1])
    else:
        assert nek_edge.GetCurve() is None


@given(from_type(Quad), integers(2, 12), integers(), sampled_from((2, 3)))
def test_nektar_quad(quad: Quad, order: int, layer: int, spatial_dim: int) -> None:
    quads, segments, points = nektar_writer.nektar_quad(quad, spatial_dim, layer)
    n = elem_order(quad)
    comparable_points = comparable_point_set(points)
    assert len(quads) == 1
    assert len(segments) == 4
    assert len(points) == (n + 1) * (n + 1)
    nek_quad = next(iter(quads))
    nek_quad_corners = comparable_point_set(
        nek_quad.GetEdge(i).GetVertex(j) for j in range(2) for i in range(4)
    )
    quad_corners = FrozenCoordSet(
        comparable_coord(c.to_cartesian()) for c in quad.corners()
    )
    assert nek_quad_corners == quad_corners
    assert quad_corners <= comparable_points
    nek_curve = nek_quad.GetCurve()
    if n > 1:
        assert nek_curve is not None
        assert len(nek_curve.points) == (n + 1) ** 2
        assert_nek_points_eq(nek_quad.GetEdge(0).GetVertex(0), nek_curve.points[0])
        assert_nek_points_eq(nek_quad.GetEdge(0).GetVertex(1), nek_curve.points[order])
        assert_nek_points_eq(
            nek_quad.GetEdge(2).GetVertex(0), nek_curve.points[-order - 1]
        )
        assert_nek_points_eq(nek_quad.GetEdge(2).GetVertex(1), nek_curve.points[-1])
    else:
        assert nek_curve is None


@given(from_type(Quad))
def test_poloidal_curve(q: Quad) -> None:
    curve = nektar_writer.poloidal_curve(q)
    x1, x2, x3 = curve
    expected = q.nodes.start_points
    np.testing.assert_allclose(x1, expected.x1, 1e-8, 1e-8)
    np.testing.assert_allclose(x2, expected.x2, 1e-8, 1e-8)
    np.testing.assert_allclose(x3, 0.0, 1e-8, 1e-8)


@given(from_type(Prism), integers())
def test_nektar_poloidal_face(solid: Prism, layer: int) -> None:
    shapes, segments, points = nektar_writer.nektar_poloidal_face(solid, 2, layer)
    corners = comparable_point_set(points)
    n = 3 if solid.shape == PrismTypes.TRIANGULAR else 4
    assert len(shapes) == 1
    assert len(segments) == n
    nek_shape = next(iter(shapes))
    assert corners == comparable_point_set(
        nek_shape.GetEdge(i).GetVertex(j) for j in range(2) for i in range(n)
    )
    assert corners == FrozenCoordSet(map(comparable_coord, poloidal_corners(solid)))


@given(from_type(UnalignedShape), integers())
def test_nektar_end_shape(shape: UnalignedShape, layer: int) -> None:
    shapes, segments, points = nektar_writer.nektar_end_shape(shape, 2, layer)
    corners = comparable_point_set(points)
    n = 3 if shape.shape == PrismTypes.TRIANGULAR else 4
    assert len(shapes) == 1
    assert len(segments) == n
    nek_shape = next(iter(shapes))
    assert corners == comparable_point_set(
        nek_shape.GetEdge(i).GetVertex(j) for j in range(2) for i in range(n)
    )
    assert corners == FrozenCoordSet(
        comparable_coord(c.to_cartesian()) for c in shape.corners()
    )


@given(flat_sided_hex, integers())
def test_nektar_hex(hexa: Prism, layer: int) -> None:
    hexes, _, segments, points = nektar_writer.nektar_3d_element(hexa, 3, layer)
    corners = comparable_point_set(points)
    assert len(hexes) == 1
    assert len(segments) == 12
    assert len(points) == 8
    assert corners == FrozenCoordSet(
        comparable_coord(c.to_cartesian()) for c in hexa.corners()
    )


@given(flat_sided_prism, integers())
def test_nektar_prism(prism: Prism, layer: int) -> None:
    prisms, _, segments, points = nektar_writer.nektar_3d_element(prism, 3, layer)
    corners = comparable_point_set(points)
    assert len(prisms) == 1
    assert len(segments) == 9
    assert len(points) == 6
    assert corners == FrozenCoordSet(
        comparable_coord(c.to_cartesian()) for c in prism.corners()
    )


def all_points(elements: Iterable[Quad | Prism]) -> Iterator[Coord]:
    return (
        c.to_cartesian()
        for c in itertools.chain.from_iterable(e.corners() for e in elements)
    )


def check_points(
    expected: Iterable[Quad | Prism],
    actual: Iterable[SD.PointGeom],
    iter_corners: Callable[[Iterable[Quad | Prism]], Iterator[Coord]] = all_points,
) -> None:
    expected_points = FrozenCoordSet(map(comparable_coord, iter_corners(expected)))
    actual_points = comparable_point_set(actual)
    assert expected_points == actual_points


MeshLike = MeshLayer[E, B, C] | GenericMesh[E, B, C]


def check_edges(
    mesh: MeshLike[Quad, FieldAlignedCurve, Curve]
    | MeshLike[Prism, Quad, UnalignedShape],
    elements: Iterable[SD.Geometry2D] | Iterable[SD.Geometry3D],
    edges: Iterable[SD.SegGeom],
) -> None:
    if issubclass(
        mesh.element_type
        if isinstance(mesh, MeshLayer)
        else cast(GenericMesh, mesh).reference_layer.element_type,
        Quad,
    ):
        mesh = cast(MeshLike[Quad, FieldAlignedCurve, Curve], mesh)
        expected_x3_aligned_edges = reduce(
            operator.or_,
            (
                frozenset({comparable_edge(q.north), comparable_edge(q.south)})
                for q in mesh
            ),
        )
        expected_near_faces = frozenset(comparable_edge(q.near) for q in mesh)
        expected_far_faces = frozenset(comparable_edge(q.far) for q in mesh)
    else:
        mesh = cast(MeshLike[Prism, Quad, UnalignedShape], mesh)
        expected_x3_aligned_edges = reduce(
            operator.or_,
            (
                frozenset(
                    itertools.chain.from_iterable(
                        [comparable_edge(side.north), comparable_edge(side.south)]
                        for side in q
                    )
                )
                for q in mesh
            ),
        )
        expected_near_faces = frozenset(
            itertools.chain.from_iterable(map(comparable_edge, q.near) for q in mesh)
        )
        expected_far_faces = frozenset(
            itertools.chain.from_iterable(map(comparable_edge, q.far) for q in mesh)
        )
    element_edges = frozenset(
        itertools.chain.from_iterable(
            map(comparable_geometry, map(e.GetEdge, range(e.GetNumEdges())))
            for e in elements
        )
    )
    actual_edges = comparable_set(edges)
    assert actual_edges == element_edges
    assert (
        expected_x3_aligned_edges | expected_near_faces | expected_far_faces
        == actual_edges
    )


def check_face_composites(
    expected: Iterable[Curve] | Iterable[UnalignedShape], actual: SD.Composite
) -> None:
    def comparable_item(
        item: Curve | UnalignedShape,
    ) -> tuple[str, FrozenCoordSet[Coord]]:
        if isinstance(item, UnalignedShape):
            if item.shape == PrismTypes.RECTANGULAR:
                name = SD.QuadGeom.__name__
            elif item.shape == PrismTypes.TRIANGULAR:
                name = SD.TriGeom.__name__
            else:
                raise ValueError(f"Unrecognised shape {item.shape}.")
            return name, FrozenCoordSet(
                comparable_coord(c.to_cartesian()) for c in item.corners()
            )
        else:
            return SD.SegGeom.__name__, FrozenCoordSet(
                {
                    comparable_coord(item[0].to_cartesian()),
                    comparable_coord(item[1].to_cartesian()),
                }
            )

    expected_faces = frozenset(map(comparable_item, expected))
    actual_faces = comparable_set(
        cast(Iterable[SD.Geometry1D | SD.Geometry2D], actual.geometries)
    )
    assert expected_faces == actual_faces


def comparable_element(x: Quad | Prism) -> ComparableGeometry:
    return (
        comparable_quad(x) if isinstance(x, Quad) else comparable_prism(cast(Prism, x))
    )


def check_elements(
    expected: Iterable[Quad] | Iterable[Prism],
    actual: Iterable[SD.Geometry2D] | Iterable[SD.Geometry3D],
    comparator: Callable[[Quad | Prism], ComparableGeometry] = comparable_element,
) -> None:
    actual_elements = comparable_set(actual)
    expected_elements = frozenset(map(comparator, expected))
    assert actual_elements == expected_elements


@settings(deadline=None)
@given(from_type(MeshLayer), integers(), sampled_from([2, 3]))
def test_nektar_layer_elements(
    mesh: MeshLayer[Quad, FieldAlignedCurve, Curve]
    | MeshLayer[Prism, Quad, UnalignedShape],
    layer: int,
    spatial_dim: int,
) -> None:
    nek_layer = nektar_writer.nektar_layer_elements(
        mesh, spatial_dim if issubclass(mesh.element_type, Quad) else 3, layer
    )
    if issubclass(mesh.element_type, Quad):
        assert isinstance(nek_layer, nektar_writer.NektarLayer2D)
        faces: frozenset[SD.SegGeom] | frozenset[SD.Geometry2D] = nek_layer.segments
    else:
        assert isinstance(nek_layer, nektar_writer.NektarLayer3D)
        faces = nek_layer.faces
    check_points(mesh, nek_layer.points)
    check_edges(mesh, nek_layer.elements, nek_layer.segments)
    assert nek_layer.near_face is not None
    assert nek_layer.far_face is not None
    check_face_composites(mesh.near_faces(), nek_layer.near_face)
    check_face_composites(mesh.far_faces(), nek_layer.far_face)
    assert frozenset(nek_layer.near_face.geometries) <= faces
    assert frozenset(nek_layer.far_face.geometries) <= faces
    check_elements(mesh, nek_layer.elements)


# Check all elements present when converting a mesh
@settings(deadline=None)
@given(from_type(GenericMesh), sampled_from([2, 3]))
def test_nektar_elements(mesh: QuadMesh | PrismMesh, spatial_dim: int) -> None:
    nek_mesh = nektar_writer.nektar_elements(
        mesh,
        spatial_dim if issubclass(mesh.reference_layer.element_type, Quad) else 3,
    )
    assert len(list(nek_mesh.layers())) == nek_mesh.num_layers()
    check_points(mesh, nek_mesh.points())
    check_edges(mesh, nek_mesh.elements(), nek_mesh.segments())
    for layer, near, far in zip(
        cast(Iterable[QuadMeshLayer | PrismMeshLayer], mesh.layers()),
        nek_mesh.near_faces(),
        nek_mesh.far_faces(),
    ):
        check_face_composites(layer.near_faces(), near)
        check_face_composites(layer.far_faces(), far)
    check_elements(mesh, nek_mesh.elements())
    # FIXME: should check bounds


def check_poloidal_edges(
    mesh: PrismMesh,
    elements: Iterable[SD.Geometry2D] | Iterable[SD.Geometry3D],
    edges: Iterable[SD.SegGeom],
) -> None:
    expected_edges = frozenset(
        comparable_edge(Curve(side.nodes.start_points.to_3d_coords(0.0)))
        for side in itertools.chain.from_iterable(mesh.reference_layer)
    )
    element_edges = frozenset(
        itertools.chain.from_iterable(
            map(comparable_geometry, map(e.GetEdge, range(e.GetNumEdges())))
            for e in elements
        )
    )
    actual_edges = comparable_set(edges)
    assert actual_edges == element_edges
    assert expected_edges == actual_edges


@settings(deadline=None)
@given(prism_meshes)
def test_nektar_poloidal_elements(
    mesh: PrismMesh,
) -> None:
    nek_mesh = nektar_writer.nektar_poloidal_elements(
        mesh,
    )
    assert len(list(nek_mesh.layers())) == 1
    check_points(
        mesh.reference_layer,
        nek_mesh.points(),
        lambda expected: itertools.chain.from_iterable(map(poloidal_corners, expected)),
    )
    check_poloidal_edges(mesh, nek_mesh.elements(), nek_mesh.segments())
    check_elements(mesh, nek_mesh.elements(), comparable_poloidal_element)
    # FIXME: should check bounds


@given(
    integers(-256, 256),
    lists(
        builds(
            SD.PointGeom,
            just(2),
            integers(-256, 256),
            non_nans(),
            non_nans(),
            non_nans(),
        ),
        max_size=5,
    ).map(lambda points: SD.Composite(cast(list[SD.Geometry], points))),
)
def test_nektar_composite_map(comp_id: int, composite: SD.Composite) -> None:
    comp_map = nektar_writer.nektar_composite_map({comp_id: composite})
    assert len(comp_map) == 1
    assert comparable_point_set(
        cast(list[SD.PointGeom], composite.geometries)
    ) == comparable_point_set(cast(list[SD.PointGeom], comp_map[comp_id].geometries))


order = shared(integers(1, 4))
NekType = Union[SD.Curve, SD.Geometry]
N = TypeVar("N", SD.Curve, SD.Geometry)


# TODO: Write some unit tests for the iterator methods on
# NektarElements. In particular, check that it doesn't return any
# boundary composites that are empty.

# TODO: Check that boundary elements already exist.


def extract_and_merge(nek_type: Type[N], *items: Iterator[NekType]) -> Iterable[N]:
    return filter(
        cast(Callable[[NekType], TypeGuard[N]], lambda y: isinstance(y, nek_type)),
        itertools.chain(*items),
    )


def find_item(i: int, geoms: frozenset[SD.Geometry]) -> SD.Geometry:
    for geom in geoms:
        if geom.GetGlobalID() == i:
            return geom
    raise IndexError(f"Item with ID {i} not found in set {geoms}")


def check_curved_edges(
    order: int,
    elements: nektar_writer.NektarElements,
    curved_edges: SD.CurveMap,
    actual_segments: SD.SegGeomMap,
) -> None:
    n_curve = len(curved_edges)
    if order == 1:
        assert n_curve == 0
    else:
        all_expected_segments = frozenset(elements.segments())
        n_curves = sum(seg.GetCurve() is not None for seg in all_expected_segments)
        assert n_curves > 0
        assert len(curved_edges) == n_curves
        for item in actual_segments:
            seg = item.data()
            curve = seg.GetCurve()
            if curve is not None:
                comparable_curve = comparable_geometry(curve)
                assert comparable_curve == comparable_geometry(
                    curved_edges[curve.curveID]
                )
                assert comparable_curve == comparable_geometry(
                    cast(
                        SD.SegGeom, find_item(seg.GetGlobalID(), all_expected_segments)
                    ).GetCurve()
                )
                assert comparable_nektar_point(seg.GetVertex(0)).approx_eq(
                    comparable_nektar_point(curve.points[0])
                )
                assert comparable_nektar_point(seg.GetVertex(1)).approx_eq(
                    comparable_nektar_point(curve.points[-1])
                )


def check_curved_faces(
    order: int,
    elements: nektar_writer.NektarElements,
    curved_faces: SD.CurveMap,
    actual_triangles: SD.TriGeomMap,
    actual_quads: SD.QuadGeomMap,
) -> None:
    if order == 1:
        assert len(curved_faces) == 0
    else:
        if isinstance(elements._layers[0], nektar_writer.NektarLayer3D):
            all_expected_faces: frozenset[SD.Geometry2D | SD.Geometry3D] = frozenset(
                elements.faces()
            )
        else:
            all_expected_faces = frozenset(elements.elements())
        n_curves = sum(
            face.GetCurve() is not None
            for face in cast(Iterator[SD.Geometry2D], all_expected_faces)
        )
        assert n_curves > 0
        assert len(curved_faces) == n_curves
        polygons: Iterator[SD._NekMapItem[SD.QuadGeom] | SD._NekMapItem[SD.TriGeom]] = (
            itertools.chain(actual_triangles, actual_quads)
        )
        for item in polygons:
            face = item.data()
            curve = face.GetCurve()
            if curve is not None:
                comparable_curve = comparable_geometry(curve)
                assert comparable_curve == comparable_geometry(
                    curved_faces[curve.curveID]
                )
                assert comparable_curve == comparable_geometry(
                    cast(
                        SD.Geometry2D, find_item(face.GetGlobalID(), all_expected_faces)
                    ).GetCurve()
                )


GeomMap = (
    SD.PointGeomMap
    | SD.SegGeomMap
    | SD.TriGeomMap
    | SD.QuadGeomMap
    | SD.TetGeomMap
    | SD.PrismGeomMap
    | SD.PyrGeomMap
    | SD.HexGeomMap
)


# TODO: Could I test this with some a NektarElements object produced
# directly using the constructor and without the constraints of those
# generated using the nektar_elements() method?
@settings(deadline=None)
@given(
    from_type(GenericMesh),
    sampled_from([2, 3]),
    order,
    booleans(),
    booleans(),
    booleans(),
)
def test_nektar_mesh(
    mesh: Mesh,
    spatial_dim: int,
    order: int,
    write_movement: bool,
    periodic: bool,
    compressed: bool,
) -> None:
    num_element_types = (
        len({p.shape for p in mesh.reference_layer})
        if issubclass(mesh.reference_layer.element_type, Prism)
        else 1
    )
    elements = nektar_writer.nektar_elements(
        mesh,
        spatial_dim if issubclass(mesh.reference_layer.element_type, Quad) else 3,
    )
    meshgraph = nektar_writer.nektar_mesh(
        elements, 2, 2, write_movement, periodic, compressed
    )
    actual_segments = meshgraph.GetAllSegGeoms()
    actual_triangles = meshgraph.GetAllTriGeoms()
    actual_quads = meshgraph.GetAllQuadGeoms()
    actual_geometries: list[GeomMap] = [
        actual_segments,
        actual_triangles,
        actual_quads,
        meshgraph.GetAllTetGeoms(),
        meshgraph.GetAllPyrGeoms(),
        meshgraph.GetAllPrismGeoms(),
        meshgraph.GetAllHexGeoms(),
    ]
    expected_geometries = list(
        map(
            list,
            [
                elements.segments(),
                extract_and_merge(SD.TriGeom, elements.faces(), elements.elements()),
                extract_and_merge(SD.QuadGeom, elements.faces(), elements.elements()),
                extract_and_merge(SD.TetGeom, elements.elements()),
                extract_and_merge(SD.PyrGeom, elements.elements()),
                extract_and_merge(SD.PrismGeom, elements.elements()),
                extract_and_merge(SD.HexGeom, elements.elements()),
            ],
        )
    )

    actual_points = meshgraph.GetAllPointGeoms()
    expected_points = list(elements.points())
    assert len(actual_points) == len(expected_points)
    assert all(item.key() == item.data().GetGlobalID() for item in actual_points)
    actual_comparable_points = comparable_point_set(
        cast(SD.PointGeom, item.data()) for item in actual_points
    )
    expected_comparable_points = comparable_point_set(expected_points)
    assert actual_comparable_points == expected_comparable_points

    for expected, actual in zip(expected_geometries, actual_geometries):
        assert len(actual) == len(expected)
        assert all(item.key() == item.data().GetGlobalID() for item in actual)
        actual_comparable = comparable_set(
            cast(SD.Geometry1D | SD.Geometry2D | SD.Geometry3D, item.data())
            for item in actual
        )
        expected_comparable = comparable_set(
            cast(list[SD.Geometry1D | SD.Geometry2D | SD.Geometry3D], expected)
        )
        assert actual_comparable == expected_comparable

    curved_edges = meshgraph.GetCurvedEdges()
    n = elem_order(cast(Quad | Prism, next(iter(mesh))))
    check_curved_edges(n, elements, curved_edges, actual_segments)
    curved_faces = meshgraph.GetCurvedFaces()
    check_curved_faces(n, elements, curved_faces, actual_triangles, actual_quads)

    actual_composites = meshgraph.GetComposites()
    n_layers = len(list(elements.layers()))
    n_comp = n_layers * (2 + num_element_types) + elements.num_bounds()
    assert len(actual_composites) == n_comp
    expected_layer_composites = comparable_composites(
        itertools.chain.from_iterable(elements.layers())
    )
    if periodic:
        expected_near_composites = comparable_composites(elements.near_faces())
        expected_far_composites = comparable_composites(elements.far_faces())
    else:
        expected_near_composites = comparable_composites(
            itertools.islice(elements.near_faces(), 1, None)
        )
        expected_far_composites = comparable_composites(
            itertools.islice(elements.far_faces(), n_layers - 1)
        )
    assert elements._layers[0].near_face is not None
    assert elements._layers[-1].far_face is not None
    expected_bound_composites = comparable_composites(elements.bounds()) | frozenset(
        {
            comparable_composite(elements._layers[0].near_face),
            comparable_composite(elements._layers[-1].far_face),
        }
    )
    assert (
        expected_layer_composites
        | expected_near_composites
        | expected_far_composites
        | expected_bound_composites
        == comparable_composites(actual_composites[i] for i in range(n_comp))
    )

    domains = meshgraph.GetDomain()
    assert len(domains) == n_layers
    assert all(len(domains[i]) == num_element_types for i in range(n_layers))
    actual_layers = comparable_composites(
        itertools.chain.from_iterable(
            (item.data() for item in domains[i]) for i in range(n_layers)
        )
    )
    assert len(actual_layers) // num_element_types == n_layers
    assert actual_layers == expected_layer_composites

    movement = meshgraph.GetMovement()
    zones = movement.GetZones()
    interfaces = movement.GetInterfaces()

    if write_movement:
        assert len(zones) == n_layers
        for i in range(n_layers):
            zone_domain = zones[i].GetDomain()
            assert len(zone_domain) == num_element_types
            assert comparable_composites(
                item.data() for item in zone_domain
            ) == comparable_composites(item.data() for item in domains[i])

        assert len(interfaces) == n_layers if periodic else n_layers - 1
        actual_near_composites = comparable_composites(
            actual_composites[
                next(iter(interface.GetRightInterface().GetCompositeIDs()))
            ]
            for interface in interfaces.values()
        )
        actual_far_composites = comparable_composites(
            actual_composites[
                next(iter(interface.GetLeftInterface().GetCompositeIDs()))
            ]
            for interface in interfaces.values()
        )
        assert len(actual_near_composites) == n_layers if periodic else n_layers - 1
        assert len(actual_far_composites) == n_layers if periodic else n_layers - 1
        assert actual_near_composites == expected_near_composites
        assert actual_far_composites == expected_far_composites
    else:
        assert len(zones) == 0
        assert len(interfaces) == 0


def find_element(parent: ET.Element, tag: str) -> ET.Element:
    elem = parent.find(tag)
    assert isinstance(elem, ET.Element)
    return elem


QUAD = Quad(
    field_aligned_positions(
        straight_line_across_field(
            SliceCoord(1.0, 0.0, CoordinateSystem.CARTESIAN),
            SliceCoord(0.0, 0.0, CoordinateSystem.CARTESIAN),
            1,
        ),
        1.0,
        simple_trace,
        np.array(1.0),
        1,
    )
)
SIMPLE_MESH = GenericMesh(
    MeshLayer([QUAD], [frozenset([QUAD.north]), frozenset([QUAD.south])]),
    np.array([0.5]),
)


def check_xml_vertices(vertices: ET.Element) -> tuple[int, int, int, int]:
    assert len(vertices) == 4
    north_east = north_west = south_east = south_west = -1
    for vertex in vertices:
        assert vertex.tag == "V" or vertex.tag == "VERTEX"
        i = int(cast(str, vertex.get("ID")))
        coord = tuple(map(float, cast(str, vertex.text).split()))
        if coord == (0.0, 0.0, 1.0):
            south_west = i
        elif coord == (0.0, 0.0, 0.0):
            south_east = i
        elif coord == (1.0, 0.0, 1.0):
            north_west = i
        elif coord == (1.0, 0.0, 0.0):
            north_east = i
        else:
            raise RuntimeError(f"Unexpected vertex location {coord}")
    assert north_east >= 0
    assert north_west >= 0
    assert south_east >= 0
    assert south_west >= 0
    return north_east, north_west, south_east, south_west


def check_xml_edges(
    edges: ET.Element, vertices: tuple[int, int, int, int]
) -> tuple[int, int, int, int]:
    north_east, north_west, south_east, south_west = vertices
    assert len(edges) == 4
    edge_vals: set[tuple[int, int]] = set()
    east = -1
    west = -1
    north = -1
    south = -1
    expected_east = tuple(sorted((north_east, south_east)))
    expected_west = tuple(sorted((north_west, south_west)))
    expected_north = tuple(sorted((north_east, north_west)))
    expected_south = tuple(sorted((south_east, south_west)))
    for edge in edges:
        assert edge.tag == "E" or edge.tag == "EDGE"
        i = int(cast(str, edge.get("ID")))
        termini = cast(
            tuple[int, int], tuple(sorted(map(int, cast(str, edge.text).split())))
        )
        if termini == expected_east:
            east = i
        elif termini == expected_west:
            west = i
        elif termini == expected_north:
            north = i
        elif termini == expected_south:
            south = i
        edge_vals.add(termini)
    assert edge_vals == {expected_east, expected_west, expected_north, expected_south}
    assert north >= 0
    assert south >= 0
    assert east >= 0
    assert west >= 0
    return north, east, south, west


def check_xml_composites(
    composites: ET.Element, edges: tuple[int, int, int, int]
) -> tuple[int, int, int]:
    assert len(composites) == 5
    north, east, south, west = edges
    domain_comp = -1
    east_comp = -1
    west_comp = -1
    north_comp = -1
    south_comp = -1
    for comp in composites:
        assert comp.tag == "C"
        i = int(cast(str, comp.get("ID")))
        content = cast(str, comp.text).strip()
        if content == "Q[0]":
            domain_comp = i
        elif content == f"E[{east}]":
            east_comp = i
        elif content == f"E[{west}]":
            west_comp = i
        elif content == f"E[{north}]":
            north_comp = i
        elif content == f"E[{south}]":
            south_comp = i
        else:
            raise RuntimeError(f"Unexpected composite {content}")
    assert domain_comp >= 0
    assert east_comp >= 0
    assert west_comp >= 0
    assert north_comp >= 0
    assert south_comp >= 0
    return domain_comp, east_comp, west_comp


# Integration test for very simple 1-element mesh
def test_write_nektar(tmp_path: pathlib.Path) -> None:
    nektar_writer.reset_id_counts()
    xml_file = tmp_path / "simple_mesh.xml"
    nektar_writer.write_nektar(SIMPLE_MESH, str(xml_file), 2, compressed=False)

    tree = ET.parse(xml_file)
    root = tree.getroot()
    assert isinstance(root, ET.Element)
    assert root.tag == "NEKTAR"
    assert root.attrib == {}

    geom = find_element(root, "GEOMETRY")
    assert geom.tag == "GEOMETRY"
    assert int(cast(str, geom.get("DIM"))) == 2
    vertices = check_xml_vertices(find_element(geom, "VERTEX"))
    edges = check_xml_edges(find_element(geom, "EDGE"), vertices)

    elements = find_element(geom, "ELEMENT")
    assert len(elements) == 1
    elem = elements[0]
    assert elem.tag == "Q" or elem.tag == "QUAD"
    assert elem.get("ID") == "0"
    assert tuple(sorted(map(int, cast(str, elem.text).split()))) == (0, 1, 2, 3)

    curves = find_element(geom, "CURVED")
    assert len(curves) == 0

    domain_comp, east_comp, west_comp = check_xml_composites(
        find_element(geom, "COMPOSITE"), edges
    )
    domains = find_element(geom, "DOMAIN")
    assert len(domains) == 1
    domain = domains[0]
    assert domain.tag == "D"
    assert domain.get("ID") == "0"
    assert cast(str, domain.text).strip() == f"C[{domain_comp}]"

    move = find_element(root, "MOVEMENT")
    zones = find_element(move, "ZONES")
    assert len(zones) == 1
    zone = zones[0]
    assert zone.tag == "F" or zone.tag == "FIXED"
    assert zone.get("ID") == "0"
    assert zone.get("DOMAIN") == "D[0]"

    interfaces = find_element(move, "INTERFACES")
    assert len(interfaces) == 1
    interface = interfaces[0]
    assert interface.tag == "INTERFACE"
    assert len(interface) == 2
    left = find_element(interface, "L")
    right = find_element(interface, "R")
    assert {int(cast(str, left.get("ID"))), int(cast(str, right.get("ID")))} == {0, 1}
    assert cast(str, left.get("BOUNDARY")).strip() == f"C[{west_comp}]"
    assert cast(str, right.get("BOUNDARY")).strip() == f"C[{east_comp}]"


@settings(deadline=None)
@given(quad_meshes)
def test_write_nektar_curves(mesh: QuadMesh) -> None:
    with TemporaryDirectory() as tmp_path:
        xml_file = pathlib.Path(tmp_path) / "simple_mesh.xml"
        nektar_writer.write_nektar(mesh, str(xml_file), 2, False, compressed=False)
        tree = ET.parse(xml_file)

    root = tree.getroot()
    assert isinstance(root, ET.Element)
    assert root.tag == "NEKTAR"

    vertices = find_element(root, "GEOMETRY/VERTEX")
    edges = find_element(root, "GEOMETRY/EDGE")
    curves = find_element(root, "GEOMETRY/CURVED")

    for curve in curves.findall("E"):
        edge_id = curve.get("EDGEID")
        data = list(map(float, cast(str, curve.text).split()))
        start = np.array(data[:3])
        end = np.array(data[-3:])
        edge = edges.find(f"E[@ID='{edge_id}']")
        assert edge is not None
        start_id, end_id = cast(str, edge.text).split()
        start_point = vertices.find(f"V[@ID='{start_id}']")
        end_point = vertices.find(f"V[@ID='{end_id}']")
        assert start_point is not None
        assert end_point is not None
        expected_start = np.array(list(map(float, cast(str, start_point.text).split())))
        expected_end = np.array(list(map(float, cast(str, end_point.text).split())))
        np.testing.assert_allclose(start, expected_start, 1e-8)
        np.testing.assert_allclose(end, expected_end, 1e-8)


def test_compressed_mesh(tmp_path: pathlib.Path) -> None:
    xml_file = tmp_path / "simple_mesh.xml"
    nektar_writer.write_nektar(SIMPLE_MESH, str(xml_file), 2, compressed=False)
    compressed_file = tmp_path / "compressed_mesh.xml"
    nektar_writer.write_nektar(SIMPLE_MESH, str(compressed_file), 2, compressed=True)
    assert os.stat(xml_file).st_size > os.stat(compressed_file).st_size


@settings(deadline=None)
@given(prism_meshes)
def test_write_nektar_poloidal(mesh: PrismMesh) -> None:
    with TemporaryDirectory() as tmp_path:
        xml_file = pathlib.Path(tmp_path) / "poloidal_mesh.xml"
        nektar_writer.write_poloidal_mesh(mesh, str(xml_file), False)
        tree = ET.parse(xml_file)

    root = tree.getroot()
    assert isinstance(root, ET.Element)
    assert root.tag == "NEKTAR"

    elements = find_element(root, "GEOMETRY/ELEMENT")
    assert elements.find("H") is None
    assert elements.find("P") is None
    assert elements.find("Q") is not None or elements.find("T") is not None
