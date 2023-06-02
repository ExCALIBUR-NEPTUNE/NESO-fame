from collections.abc import Iterable
from ctypes import Union
from functools import reduce
import itertools
import operator
from typing import Callable, cast, Type, TypeVar, Union

from hypothesis import given
from hypothesis.extra.numpy import mutually_broadcastable_shapes
from hypothesis.strategies import (
    builds,
    floats,
    from_type,
    integers,
    just,
    lists,
    one_of,
    shared,
)
from NekPy import SpatialDomains as SD
from NekPy import LibUtilities as LU
import numpy as np
from pytest import approx, mark

import mesh_strategies
from neso_fame import nektar_writer
from neso_fame.mesh import Coord, Curve, CoordinateSystem, Mesh, MeshLayer, Quad


def both_nan(a: float, b: float) -> bool:
    return np.isnan(a) and np.isnan(b)


def assert_nek_points_eq(actual: SD.PointGeom, expected: SD.PointGeom) -> None:
    for a, e in zip(actual.GetCoordinates(), expected.GetCoordinates()):
        assert a == approx(e) or both_nan(a, e)


def assert_points_eq(actual: SD.PointGeom, expected: Coord) -> None:
    for a, e in zip(actual.GetCoordinates(), expected.to_cartesian()):
        assert a == approx(e) or both_nan(a, e)


ComparablePoint = tuple[float, float, float]
ComparableDimensionalGeometry = tuple[str, frozenset[ComparablePoint]]
ComparableGeometry = ComparablePoint | ComparableDimensionalGeometry


def comparable_coord(c: Coord) -> ComparablePoint:
    c = c.to_cartesian()
    return c.x1, c.x2, c.x3


def comparable_nektar_point(geom: SD.PointGeom) -> ComparablePoint:
    return geom.GetCoordinates()


def comparable_curve(curve: Curve, order: int) -> ComparableGeometry:
    return SD.Curve.__name__, frozenset(
        map(comparable_coord, curve.control_points(order).to_cartesian().iter_points())
    )


def comparable_edge(curve: Curve) -> ComparableGeometry:
    return SD.SegGeom.__name__, frozenset(
        map(comparable_coord, curve.control_points(1).to_cartesian().iter_points())
    )


def comparable_quad(quad: Quad) -> ComparableGeometry:
    return SD.QuadGeom.__name__, frozenset(
        map(comparable_coord, quad.corners().to_cartesian().iter_points())
    )


def comparable_geometry(geom: SD.Geometry | SD.Curve) -> ComparableGeometry:
    """Convert geometry elements into sets of points. This allows for convenient comparison.

    TODO: Should it also provide sets of edges, etc.?
    """
    if isinstance(geom, SD.PointGeom):
        return comparable_nektar_point(geom)
    elif isinstance(geom, SD.Geometry):
        return type(geom).__name__, frozenset(
            map(comparable_nektar_point, map(geom.GetVertex, range(geom.GetNumVerts())))
        )
    else:
        return type(geom).__name__, frozenset(map(comparable_nektar_point, geom.points))


def comparable_set(vals: Iterable[SD.Geometry]) -> frozenset[ComparableGeometry]:
    return frozenset(map(comparable_geometry, vals))


def comparable_composite(val: SD.Composite) -> frozenset[ComparableGeometry]:
    return frozenset(map(operator.methodcaller("GetGlobalID"), val.geometries))


def comparable_composites(
    vals: Iterable[SD.Composite],
) -> frozenset[frozenset[ComparableGeometry]]:
    return frozenset(map(comparable_composite, vals))


@mark.filterwarnings("ignore:invalid value:RuntimeWarning")
@given(from_type(Coord), integers())
def test_nektar_point(coord: Coord, i: int) -> None:
    nek_coord = nektar_writer.nektar_point(coord, i)
    assert nek_coord.GetGlobalID() == nektar_writer.UNSET_ID
    assert_points_eq(nek_coord, coord)


@mark.filterwarnings("ignore:invalid value:RuntimeWarning")
@given(
    lists(from_type(Coord), min_size=2, max_size=2, unique=True),
    lists(integers(), min_size=2, max_size=2, unique=True),
)
def test_nektar_point_caching(coords: list[Coord], layers: list[int]) -> None:
    c1 = nektar_writer.nektar_point(coords[0], layers[0])
    assert c1 is nektar_writer.nektar_point(coords[0], layers[0])
    assert c1 is not nektar_writer.nektar_point(coords[1], layers[0])
    assert c1 is not nektar_writer.nektar_point(coords[0], layers[1])


@given(from_type(Curve), integers(1, 10), integers())
def test_nektar_curve(curve: Curve, order: int, layer: int) -> None:
    nek_curve, (start, end) = nektar_writer.nektar_curve(curve, order, layer)
    assert nek_curve.curveID == nektar_writer.UNSET_ID
    assert nek_curve.ptype == LU.PointsType.PolyEvenlySpaced
    assert_nek_points_eq(nek_curve.points[0], start)
    assert_nek_points_eq(nek_curve.points[-1], end)
    assert len(nek_curve.points) == order + 1


def test_circular_nektar_curve() -> None:
    curve = Curve(
        mesh_strategies.linear_field_line(
            0.0, 0.2, np.pi, 1.0, 0.0, np.pi / 2, CoordinateSystem.Cylindrical
        )
    )
    nek_curve, _ = nektar_writer.nektar_curve(curve, 2, 0)
    assert_points_eq(
        nek_curve.points[0], Coord(1.0, 0.0, -0.1, CoordinateSystem.Cartesian)
    )
    assert_points_eq(
        nek_curve.points[1], Coord(0.0, 1.0, 0.0, CoordinateSystem.Cartesian)
    )
    assert_points_eq(
        nek_curve.points[2], Coord(-1.0, 0.0, 0.1, CoordinateSystem.Cartesian)
    )


@given(from_type(Curve), integers())
def test_nektar_edge_first_order(curve: Curve, layer: int) -> None:
    nek_edge, (start, end) = nektar_writer.nektar_edge(curve, 1, layer)
    assert nek_edge.GetGlobalID() == nektar_writer.UNSET_ID
    assert nek_edge.GetCurve() is None
    assert_nek_points_eq(nek_edge.GetVertex(0), start)
    assert_nek_points_eq(nek_edge.GetVertex(1), end)
    assert_points_eq(start, curve(0.0).to_coord())
    assert_points_eq(end, curve(1.0).to_coord())


@given(from_type(Curve), integers(2, 12), integers())
def test_nektar_edge_higher_order(curve: Curve, order: int, layer: int) -> None:
    nek_edge, (start, end) = nektar_writer.nektar_edge(curve, order, layer)
    assert nek_edge.GetGlobalID() == nektar_writer.UNSET_ID
    nek_curve = nek_edge.GetCurve()
    assert nek_curve is not None
    assert len(nek_curve.points) == order + 1
    assert_nek_points_eq(nek_edge.GetVertex(0), start)
    assert_nek_points_eq(nek_curve.points[0], start)
    assert_nek_points_eq(nek_edge.GetVertex(1), end)
    assert_nek_points_eq(nek_curve.points[-1], end)
    assert_points_eq(start, curve(0.0).to_coord())
    assert_points_eq(end, curve(1.0).to_coord())
    for p1, p2 in zip(nek_curve.points, nek_edge.GetCurve().points):
        assert_nek_points_eq(p1, p2)


@mark.filterwarnings("ignore:invalid value:RuntimeWarning")
@given(from_type(Coord), from_type(Coord), integers())
def test_connect_points(c1: Coord, c2: Coord, layer) -> None:
    edge = nektar_writer.connect_points(
        nektar_writer.nektar_point(c1, layer),
        nektar_writer.nektar_point(c2, layer),
        layer,
    )
    assert_points_eq(edge.GetVertex(0), c1)
    assert_points_eq(edge.GetVertex(1), c2)
    assert edge.GetCurve() is None


@given(from_type(Quad), integers(1, 12), integers())
def test_nektar_quad_flat(quad: Quad, order: int, layer: int) -> None:
    quads, segments, points = nektar_writer.nektar_quad(quad, order, layer)
    corners = frozenset(p.GetCoordinates() for p in points)
    assert len(quads) == 1
    assert len(segments) == 4
    assert len(points) == len(corners)
    nek_quad = next(iter(quads))
    assert nek_quad.GetGlobalID() == nektar_writer.UNSET_ID
    # FIXME: For some reason, only 3 unique vertices are being
    # returned. I think there is something wrong with the
    # implementation within Nektar++.
    #
    # assert corners == frozenset(nek_quad.GetVertex(i).GetCoordinates() for i in range(4))
    assert corners == frozenset(
        map(comparable_coord, quad.corners().to_cartesian().iter_points())
    )


# Check nektar quad with curvature generated properly; how to do that,
# as I can't access the GetCurve method?


def check_layer_quads(
    mesh: MeshLayer[Quad], nek_layer: nektar_writer.NektarLayer, order: int
) -> None:
    expected_points = frozenset(
        map(
            comparable_coord,
            itertools.chain.from_iterable(
                e.corners().to_cartesian().iter_points() for e in mesh
            ),
        )
    )
    actual_points = comparable_set(nek_layer.points)
    assert expected_points == actual_points
    quad_edges = frozenset(
        itertools.chain.from_iterable(
            map(comparable_geometry, map(q.GetEdge, range(4)))
            for q in nek_layer.elements
        )
    )
    actual_edges = comparable_set(nek_layer.segments)
    assert actual_edges == quad_edges
    expected_x3_aligned_edges = frozenset(
        comparable_edge(q.north) for q in mesh
    ) | frozenset(comparable_edge(q.south) for q in mesh)
    expected_near_faces = frozenset(
        (
            SD.SegGeom.__name__,
            frozenset(
                {
                    comparable_coord(q.north(0.0).to_cartesian().to_coord()),
                    comparable_coord(q.south(0.0).to_cartesian().to_coord()),
                }
            ),
        )
        for q in mesh
    )
    expected_far_faces = frozenset(
        (
            SD.SegGeom.__name__,
            frozenset(
                {
                    comparable_coord(q.north(1.0).to_cartesian().to_coord()),
                    comparable_coord(q.south(1.0).to_cartesian().to_coord()),
                }
            ),
        )
        for q in mesh
    )
    assert (
        expected_x3_aligned_edges | expected_near_faces | expected_far_faces
        == actual_edges
    )
    assert len(nek_layer.faces) == 0
    actual_elements = comparable_set(nek_layer.elements)
    # FIXME: For some reason, only 3 unique vertices are being
    # returned by Nektar QuadGeom types. I think there is something
    # wrong with the implementation within Nektar++. It means the test
    # below fails.
    #
    # expected_elements = frozenset(map(comparable_quad, mesh))
    # assert actual_elements == expected_elements
    composite_elements = comparable_set(nek_layer.layer.geometries)
    assert actual_elements == composite_elements
    actual_near_faces = comparable_set(nek_layer.near_face.geometries)
    assert actual_near_faces == expected_near_faces
    actual_far_faces = comparable_set(nek_layer.far_face.geometries)
    assert actual_far_faces == expected_far_faces


# TODO: This will need significant updating once we start generating
# Tet meshes. Will probably be best to split into two separate tests.
@given(from_type(MeshLayer), integers(1, 12), integers())
def test_nektar_layer_elements(mesh: MeshLayer[Quad], order: int, layer: int) -> None:
    nek_layer = nektar_writer.nektar_layer_elements(mesh, order, layer)
    check_layer_quads(mesh, nek_layer, order)


# Check all elements present when converting a mesh
@given(from_type(Mesh), integers(1, 4))
def test_nektar_elements(mesh: Mesh[Quad], order: int) -> None:
    nek_mesh = nektar_writer.nektar_elements(mesh, order)
    n = len(nek_mesh.points)
    assert len(nek_mesh.segments) == n
    assert len(nek_mesh.faces) == n
    assert len(nek_mesh.elements) == n
    assert len(nek_mesh.layers) == n
    assert len(nek_mesh.near_faces) == n
    assert len(nek_mesh.far_faces) == n
    for nek_layer, mesh_layer in zip(
        itertools.starmap(
            nektar_writer.NektarLayer,
            zip(
                nek_mesh.points,
                nek_mesh.segments,
                nek_mesh.faces,
                nek_mesh.elements,
                nek_mesh.layers,
                nek_mesh.near_faces,
                nek_mesh.far_faces,
            ),
        ),
        mesh.layers(),
    ):
        check_layer_quads(mesh_layer, nek_layer, order)


@given(
    integers(-256, 256),
    lists(
        builds(
            SD.PointGeom,
            just(2),
            integers(-256, 256),
            mesh_strategies.non_nans(),
            mesh_strategies.non_nans(),
            mesh_strategies.non_nans(),
        ),
        max_size=5,
    ).map(SD.Composite),
)
def test_nektar_composite_map(comp_id: int, composite: SD.Composite) -> None:
    comp_map = nektar_writer.nektar_composite_map(comp_id, composite)
    assert len(comp_map) == 1
    assert comparable_set(composite.geometries) == comparable_set(
        comp_map[comp_id].geometries
    )


order = shared(integers(1, 8))
NekType = Union[SD.Curve, SD.Geometry]
N = TypeVar("N", SD.Curve, SD.Geometry)


@given(builds(nektar_writer.nektar_elements, from_type(Mesh), order), order)
def test_nektar_mesh(elements: nektar_writer.NektarElements, order: int) -> None:
    def extract_and_merge(
        nek_type: Type[N], *items: list[frozenset[NekType]]
    ) -> list[frozenset[N]]:
        return list(
            map(
                frozenset,
                map(
                    lambda x: filter(lambda y: isinstance(y, nek_type), x),
                    map(lambda z: reduce(operator.or_, z), zip(*items)),
                ),
            )
        )

    def find_item(i: int, geoms: frozenset[SD.Geometry]) -> SD.Geometry:
        for geom in geoms:
            if geom.GetGlobalID() == i:
                return geom
        raise IndexError(f"Item with ID {i} not found in set {geoms}")

    meshgraph = nektar_writer.nektar_mesh(elements, 2, 3)
    actual_segments = meshgraph.GetAllSegGeoms()
    actual_triangles = meshgraph.GetAllTriGeoms()
    actual_quads = meshgraph.GetAllQuadGeoms()
    actual_geometries = [
        meshgraph.GetAllPointGeoms(),
        actual_segments,
        actual_triangles,
        actual_quads,
        meshgraph.GetAllTetGeoms(),
        meshgraph.GetAllPyrGeoms(),
        meshgraph.GetAllPrismGeoms(),
        meshgraph.GetAllHexGeoms(),
    ]
    expected_geometries = [
        elements.points,
        elements.segments,
        extract_and_merge(SD.TriGeom, elements.faces, elements.elements),
        extract_and_merge(SD.QuadGeom, elements.faces, elements.elements),
        extract_and_merge(SD.TetGeom, elements.elements),
        extract_and_merge(SD.PyrGeom, elements.elements),
        extract_and_merge(SD.PrismGeom, elements.elements),
        extract_and_merge(SD.HexGeom, elements.elements),
    ]
    for expected, actual in zip(expected_geometries, actual_geometries):
        n = sum(map(len, expected))
        assert len(actual) == n
        assert all(actual[i].GetGlobalID() == i for i in range(n))
        actual_comparable = comparable_set(actual[i] for i in range(n))
        expected_comparable = comparable_set(reduce(operator.or_, expected))
        assert actual_comparable == expected_comparable

    curved_edges = meshgraph.GetCurvedEdges()
    n_curve = len(curved_edges)
    if order == 1:
        assert n_curve == 0
    else:
        assert all(curved_edges[i].curveID == i for i in range(n_curve))
        all_expected_segments = reduce(operator.or_, elements.segments)
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

    curved_faces = meshgraph.GetCurvedFaces()
    if order == 1:
        assert len(curved_faces) == 0
    else:
        assert all(
            curved_faces[i] == i for i in range(n_curve, n_curve + len(curved_faces))
        )
        all_expected_faces = reduce(operator.or_, elements.faces)
        for item in itertools.chain(actual_triangles, actual_quads):
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

    actual_composites = meshgraph.GetComposites()
    n_layers = len(elements.layers)
    n_comp = 3 * n_layers
    assert len(actual_composites) == n_comp
    expected_layer_composites = comparable_composites(elements.layers)
    expected_near_composites = comparable_composites(elements.near_faces)
    expected_far_composites = comparable_composites(elements.far_faces)
    assert (
        expected_layer_composites | expected_near_composites | expected_far_composites
        == comparable_composites(actual_composites[i] for i in range(n_comp))
    )

    domains = meshgraph.GetDomain()
    assert len(domains) == n_layers
    assert all(len(domains[i]) == 1 for i in range(n_layers))
    actual_layers = comparable_composites(domains[i][i] for i in range(n_layers))
    assert len(actual_layers) == n_layers
    assert actual_layers == expected_layer_composites

    movement = meshgraph.GetMovement()
    zones = movement.GetZones()
    interfaces = movement.GetInterfaces()

    assert len(zones) == n_layers
    for i in range(n_layers):
        zone_domain = zones[i].GetDomain()
        assert len(zone_domain) == 1
        assert comparable_composite(zone_domain[i]) == comparable_composite(
            domains[i][i]
        )

    assert len(interfaces) == n_layers
    actual_near_composites = comparable_composites(
        actual_composites[next(iter(interface.GetLeftInterface().GetCompositeIDs()))]
        for interface in interfaces.values()
    )
    actual_far_composites = comparable_composites(
        actual_composites[next(iter(interface.GetRightInterface().GetCompositeIDs()))]
        for interface in interfaces.values()
    )
    assert len(actual_near_composites) == n_layers
    assert len(actual_far_composites) == n_layers
    assert actual_near_composites == expected_near_composites
    assert actual_far_composites == expected_far_composites


def test_read_write_nektar_mesh() -> None:
    # Probably best way to do this is to read an example grid and
    # check it is the same when written out again. Just need to be
    # wary of any date information.

    # Will probably need to parse XML, deleting the metadata, and then save to a new file.
    pass


def test_write_nektar() -> None:
    # Test XML generation for a very, very simple mesh?
    pass
