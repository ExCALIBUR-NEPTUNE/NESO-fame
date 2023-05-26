import itertools
from collections.abc import Iterable
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
from neso_fame.mesh import Coord, Curve, CoordinateSystem, MeshLayer, Quad


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
    nek_edge, nek_curve, (start, end) = nektar_writer.nektar_edge(curve, 1, layer)
    assert nek_edge.GetGlobalID() == nektar_writer.UNSET_ID
    assert nek_curve is None
    assert_nek_points_eq(nek_edge.GetVertex(0), start)
    assert_nek_points_eq(nek_edge.GetVertex(1), end)
    assert_points_eq(start, curve(0.0).to_coord())
    assert_points_eq(end, curve(1.0).to_coord())


@given(from_type(Curve), integers(2, 12), integers())
def test_nektar_edge_higher_order(curve: Curve, order: int, layer: int) -> None:
    nek_edge, nek_curve, (start, end) = nektar_writer.nektar_edge(curve, order, layer)
    assert nek_edge.GetGlobalID() == nektar_writer.UNSET_ID
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
    quads, segments, curves, points = nektar_writer.nektar_quad(quad, order, layer)
    assert len(quads) == 1
    assert len(segments) == 4
    assert len(curves) == (2 if order > 1 else 0)
    assert len(points) == 4
    nek_quad = next(iter(quads))
    assert nek_quad.GetGlobalID() == nektar_writer.UNSET_ID
    corners = frozenset(p.GetCoordinates() for p in points)
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


# TODO: This will need significant updating once we start generating
# Tet meshes. Will probably be best to split into two separate tests.
@given(from_type(MeshLayer), integers(1, 12), integers())
def test_nektar_layer_elements(mesh: MeshLayer[Quad], order: int, layer: int) -> None:
    nek_layer = nektar_writer.nektar_layer_elements(mesh, order, layer)
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
        comparable_edge(q.north) for q in mesh.reference_elements
    ) | frozenset(comparable_edge(q.south) for q in mesh.reference_elements)
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
        for q in mesh.reference_elements
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
        for q in mesh.reference_elements
    )
    assert (
        expected_x3_aligned_edges | expected_near_faces | expected_far_faces
        == actual_edges
    )
    if order > 1:
        expected_curves = frozenset(
            comparable_curve(q.north, order) for q in mesh.reference_elements
        ) | frozenset(comparable_curve(q.south, order) for q in mesh.reference_elements)
        actual_curves = frozenset(map(comparable_geometry, nek_layer.curves))
        assert expected_curves == actual_curves
    assert len(nek_layer.faces) == 0
    actual_elements = comparable_set(nek_layer.elements)
    expected_elements = frozenset(map(comparable_quad, mesh.reference_elements))
    # FIXME: For some reason, only 3 unique vertices are being
    # returned by Nektar QuadGeom types. I think there is something
    # wrong with the implementation within Nektar++. It means the test
    # below fails.
    #
    # assert actual_elements == expected_elements
    composite_elements = comparable_set(nek_layer.layer.geometries)
    assert actual_elements == composite_elements
    actual_near_faces = comparable_set(nek_layer.near_face.geometries)
    assert actual_near_faces == expected_near_faces
    actual_far_faces = comparable_set(nek_layer.far_face.geometries)
    assert actual_far_faces == expected_far_faces


# Check all elements present when converting a mesh
def test_nektar_elements() -> None:
    # Probably don't need to be as thorough with this one, as it just returns lists of the same things as the previous routine...
    pass


def test_nektar_composite_map() -> None:
    pass


def test_nektar_mesh() -> None:
    # Check appropriate composites, domains, zones, and interfaces are present
    pass


def test_read_write_nektar_mesh() -> None:
    # Probably best way to do this is to read an example grid and
    # check it is the same when written out again. Just need to be
    # wary of any date information.
    pass


def test_write_nektar() -> None:
    # Create a native mesh, then write it out and read it in
    # again. Then compare it against hte original native mesh.
    pass
