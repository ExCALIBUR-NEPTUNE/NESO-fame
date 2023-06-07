from collections.abc import Iterable, Sequence
from functools import reduce
import itertools
import operator
import pathlib
from tempfile import TemporaryDirectory
from typing import Callable, cast, Type, TypeVar, Union
import xml.etree.ElementTree as ET

from hypothesis import given
from hypothesis.strategies import (
    builds,
    from_type,
    integers,
    just,
    lists,
    shared,
)
from NekPy import LibUtilities as LU
from NekPy import SpatialDomains as SD
from hypothesis.strategies._internal.core import booleans
from neso_fame.fields import straight_field
import numpy as np
from pytest import approx, mark

from . import mesh_strategies
from neso_fame import nektar_writer
from neso_fame.mesh import Coord, Coords, Curve, CoordinateSystem, Mesh, MeshLayer, Quad


def both_nan(a: float, b: float) -> bool:
    return cast(bool, np.isnan(a) and np.isnan(b))


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
    ).map(lambda points: SD.Composite(cast(list[SD.Geometry], points))),
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


# TODO: Could I test this with some a NektarElements object produced
# directly using the constructor and without the constraints of those
# generated using the nektar_elements() method?
@given(builds(nektar_writer.nektar_elements, from_type(Mesh), order), order, booleans())
def test_nektar_mesh(elements: nektar_writer.NektarElements, order: int, write_movement) -> None:
    def extract_and_merge(
        nek_type: Type[N], *items: Sequence[frozenset[NekType]]
    ) -> list[frozenset[N]]:
        # MyPy seemed to struggle with the typing here, so put in lots
        # of cast expressiosn to help it along
        return list(
            map(
                cast(Callable[[Iterable[N]], frozenset[N]], frozenset),
                cast(
                    Iterable[Iterable[N]],
                    map(
                        lambda x: filter(lambda y: isinstance(y, nek_type), x),
                        cast(
                            Iterable[frozenset[NekType]],
                            map(lambda z: reduce(operator.or_, z), zip(*items)),
                        ),
                    ),
                ),
            )
        )

    def find_item(i: int, geoms: frozenset[SD.Geometry]) -> SD.Geometry:
        for geom in geoms:
            if geom.GetGlobalID() == i:
                return geom
        raise IndexError(f"Item with ID {i} not found in set {geoms}")

    meshgraph = nektar_writer.nektar_mesh(elements, 2, 3, write_movement)
    actual_segments = meshgraph.GetAllSegGeoms()
    actual_triangles = meshgraph.GetAllTriGeoms()
    actual_quads = meshgraph.GetAllQuadGeoms()
    actual_geometries: list[SD.NekMap] = [
        meshgraph.GetAllPointGeoms(),
        actual_segments,
        actual_triangles,
        actual_quads,
        meshgraph.GetAllTetGeoms(),
        meshgraph.GetAllPyrGeoms(),
        meshgraph.GetAllPrismGeoms(),
        meshgraph.GetAllHexGeoms(),
    ]
    expected_geometries: list[Sequence[frozenset[SD.Geometry]]] = [
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
                assert comparable_geometry(seg.GetVertex(0)) == comparable_geometry(curve.points[0])
                assert comparable_geometry(seg.GetVertex(1)) == comparable_geometry(curve.points[-1])

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

    if write_movement:
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
    else:
        assert len(zones) == 0
        assert len(interfaces) == 0


def find_element(parent: ET.Element, tag: str) -> ET.Element:
        elem = parent.find(tag)
        assert isinstance(elem, ET.Element)
        return elem


# Integration test for very simple 1-element mesh
def test_write_nektar(tmp_path: pathlib.Path) -> None:
    simple_mesh = Mesh(
        MeshLayer(
            {
                Quad(
                    Curve(
                        lambda x: Coords(
                            np.array(1.0),
                            np.array(0.0),
                            np.asarray(x),
                            CoordinateSystem.Cartesian,
                        )
                    ),
                    Curve(
                        lambda x: Coords(
                            np.array(0.0),
                            np.array(0.0),
                            np.asarray(x),
                            CoordinateSystem.Cartesian,
                        )
                    ),
                    None,
                    straight_field(),
                ): {}
            }
        ),
        np.array([0.0]),
    )

    xml_file = tmp_path / "simple_mesh.xml"
    nektar_writer.write_nektar(simple_mesh, 1, str(xml_file))

    tree = ET.parse(xml_file)
    root = tree.getroot()
    assert isinstance(root, ET.Element)
    assert root.tag == "NEKTAR"
    assert root.attrib == {}

    geom = find_element(root, "GEOMETRY")
    assert geom.tag == "GEOMETRY"
    assert int(cast(str, geom.get("DIM"))) == 2

    vertices = find_element(geom, "VERTEX")
    assert len(vertices) == 4
    north_east = north_west = south_east = south_west = -1
    for i, vertex in enumerate(vertices):
        assert vertex.tag == "V" or vertex.tag == "VERTEX"
        assert int(cast(str, vertex.get("ID"))) == i
        coord = tuple(map(float, cast(str, vertex.text).split()))
        if coord == (0.0, 0.0, 0.0):
            south_west = i
        elif coord == (0.0, 0.0, 1.0):
            south_east = i
        elif coord == (1.0, 0.0, 0.0):
            north_west = i
        elif coord == (1.0, 0.0, 1.0):
            north_east = i
        else:
            raise RuntimeError(f"Unexpected vertex location {coord}")
    assert north_east >= 0
    assert north_west >= 0
    assert south_east >= 0
    assert south_west >= 0

    edges = find_element(geom, "EDGE")
    assert len(edges) == 4
    edge_vals: set[tuple[int, int]] = set()
    east = -1
    west = -1
    expected_east = tuple(sorted((north_east, south_east)))
    expected_west = tuple(sorted((north_west, south_west)))
    expected_north = tuple(sorted((north_east, north_west)))
    expected_south = tuple(sorted((south_east, south_west)))
    for i, edge in enumerate(edges):
        assert edge.tag == "E" or edge.tag == "EDGE"
        assert int(cast(str, edge.get("ID"))) == i
        termini = cast(
            tuple[int, int], tuple(sorted(map(int, cast(str, edge.text).split())))
        )
        if termini == expected_east:
            east = i
        elif termini == expected_west:
            west = i
        edge_vals.add(termini)
    assert edge_vals == {expected_east, expected_west, expected_north, expected_south}

    elements = find_element(geom, "ELEMENT")
    assert len(elements) == 1
    elem = elements[0]
    assert elem.tag == "Q" or elem.tag == "QUAD"
    assert elem.get("ID") == "0"
    assert tuple(sorted(map(int, cast(str, elem.text).split()))) == (0, 1, 2, 3)

    curves = find_element(geom, "CURVED")
    assert len(curves) == 0

    composites = find_element(geom, "COMPOSITE")
    assert len(composites) == 3
    domain_comp = -1
    east_comp = -1
    west_comp = -1
    for i, comp in enumerate(composites):
        assert comp.tag == "C"
        assert int(cast(str, comp.get("ID"))) == i
        content = cast(str, comp.text).strip()
        if content == "Q[0]":
            domain_comp = i
        elif content == f"E[{east}]":
            east_comp = i
        elif content == f"E[{west}]":
            west_comp = i
        else:
            raise RuntimeError(f"Unexpected composite {content}")
    assert domain_comp >= 0
    assert east_comp >= 0
    assert west_comp >= 0

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


@given(from_type(Mesh), integers(2, 10), booleans())
def test_write_nektar_curves(mesh: Mesh, order:int, write_movement: bool) -> None:
    with TemporaryDirectory() as tmp_path:
        xml_file = pathlib.Path(tmp_path) / "simple_mesh.xml"
        nektar_writer.write_nektar(mesh, order, str(xml_file), write_movement)
        nektar_writer.write_nektar(mesh, order, "last_mesh.xml", write_movement)
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
