from collections.abc import Iterable
from dataclasses import dataclass
from functools import cache, reduce
import itertools
from typing import Optional

import NekPy.SpatialDomains._SpatialDomains as SD
import NekPy.LibUtilities._LibUtilities as LU

from .generate_mesh import Mesh, MeshLayer, Coord, Curve, Quad

UNSET_ID = -1

NektarQuadGeomElements = tuple[
    frozenset[SD.QuadGeom],
    frozenset[SD.SegGeom],
    frozenset[SD.Curve],
    frozenset[SD.PointGeom],
]


@dataclass(frozen=True)
class NektarLayer:
    points: frozenset[SD.PointGeom]
    curves: frozenset[SD.Curve]
    segments: frozenset[SD.SegGeom]
    faces: frozenset[SD.Geometry2D]
    elements: list[SD.Geometry1D | SD.Geometry2D | SD.Geometry3D]
    layer: SD.Composite
    near_face: SD.Composite
    far_face: SD.Composite


@dataclass(frozen=True)
class NektarElements:
    points: list[frozenset[SD.PointGeom]]
    curves: list[frozenset[SD.Curve]]
    segments: list[frozenset[SD.SegGeom]]
    faces: list[frozenset[SD.Geometry2D]]
    elements: list[list[SD.Geometry1D | SD.Geometry2D | SD.Geometry3D]]
    layers: list[SD.Composite]
    near_faces: list[SD.Composite]
    far_faces: list[SD.Composite]


@cache
def nektar_point(position: Coord, layer_id: Optional[int] = None) -> SD.PointGeom:
    return SD.PointGeom(2, UNSET_ID, *position.to_cartesian())


@cache
def nektar_curve(
    curve: Curve, order: int, layer_id: Optional[int] = None
) -> tuple[SD.Curve, tuple[SD.PointGeom, SD.PointGeom]]:
    points = [nektar_point(coord, layer_id) for coord in curve.control_points(order)]
    nek_curve = SD.Curve(UNSET_ID, LU.PointsType.PolyEvenlySpaced)
    nek_curve.points = points
    return nek_curve, (points[0], points[1])


@cache
def nektar_edge(
    curve: Curve, order: int, layer_id: Optional[int] = None
) -> tuple[SD.SegGeom, Optional[SD.Curve], tuple[SD.PointGeom, SD.PointGeom]]:
    if order > 1:
        nek_curve, termini = nektar_curve(curve, order, layer_id)
    else:
        nek_curve = None
        end_points = curve.control_points(1)
        termini = (
            nektar_point(end_points[0], layer_id),
            nektar_point(end_points[1], layer_id),
        )
    return (
        SD.SegGeom(UNSET_ID, termini[0].GetCoordim(), list(termini), nek_curve),
        nek_curve,
        termini,
    )


@cache
def connect_points(
    start: SD.PointGeom, end: SD.PointGeom, layer_id: Optional[int] = None
) -> SD.SegGeom:
    return SD.SegGeom(UNSET_ID, start.GetCoordim(), [start, end], None)


@cache
def nektar_quad(
    quad: Quad, order: int, layer_id: Optional[int] = None
) -> NektarQuadGeomElements:
    if quad.in_plane is not None:
        raise NotImplementedError("Not yet dealing with Quads as faces.")
    north, north_curve, north_termini = nektar_edge(quad.north, order, layer_id)
    south, south_curve, south_termini = nektar_edge(quad.south, order, layer_id)
    if order > 1:
        assert north_curve is not None
        assert south_curve is not None
        curves = frozenset({north_curve, south_curve})
    else:
        curves = frozenset()
    edges = [
        north,
        connect_points(north_termini[0], south_termini[0], layer_id),
        south,
        connect_points(north_termini[1], south_termini[1], layer_id),
    ]
    return (
        frozenset({SD.QuadGeom(UNSET_ID, edges)}),
        frozenset(edges),
        curves,
        frozenset(north_termini + south_termini),
    )


def combine_quad_items(
    quad1: NektarQuadGeomElements, quad2: NektarQuadGeomElements
) -> NektarQuadGeomElements:
    return (
        quad1[0] | quad2[0],
        quad1[1] | quad2[1],
        quad1[2] | quad2[2],
        quad1[3] | quad2[3],
    )


def nektar_layer_elements(
    layer: MeshLayer, order: int, layer_id: Optional[int] = None
) -> NektarLayer:
    # FIXME: Currently inherantly 2D
    elements, edges, curves, points = reduce(
        combine_quad_items,
        (nektar_quad(elem, order, layer_id) for elem in layer.elements()),
    )
    layer_composite = SD.Composite(list(elements))
    near_face = SD.Composite([elem.GetEdge(1) for elem in elements])
    far_face = SD.Composite([elem.GetEdge(3) for elem in elements])
    return NektarLayer(
        points,
        curves,
        edges,
        frozenset(),
        list(elements),
        layer_composite,
        near_face,
        far_face,
    )


def combine_nektar_elements(left: NektarElements, right: NektarLayer) -> NektarElements:
    left.points.append(right.points)
    left.curves.append(right.curves)
    left.segments.append(right.segments)
    left.faces.append(right.faces)
    left.elements.append(right.elements)
    left.layers.append(right.layer)
    left.near_faces.append(right.near_face)
    left.far_faces.append(right.far_face)
    return left


def nektar_elements(mesh: Mesh, order: int) -> NektarElements:
    return reduce(
        combine_nektar_elements,
        (
            nektar_layer_elements(layer, order, i)
            for i, layer in enumerate(mesh.layers())
        ),
        NektarElements([], [], [], [], [], [], [], []),
    )


def nektar_composite_map(comp_id: int, composite: SD.Composite) -> SD.CompositeMap:
    comp_map = SD.CompositeMap()
    comp_map[comp_id] = composite
    return comp_map


def nektar_mesh(
    elements: NektarElements, mesh_dim: int, spatial_dim: int
) -> SD.MeshGraphXml:
    meshgraph = SD.MeshGraphXml(mesh_dim, spatial_dim)
    points = meshgraph.GetAllPointGeoms()
    segments = meshgraph.GetAllSegGeoms()
    curved_edges = meshgraph.GetCurvedEdges()
    tris = meshgraph.GetAllTriGeoms()
    quads = meshgraph.GetAllQuadGeoms()
    tets = meshgraph.GetAllTetGeoms()
    prisms = meshgraph.GetAllPrismGeoms()
    pyrs = meshgraph.GetAllPyrGeoms()
    hexes = meshgraph.GetAllHexGeoms()
    composites = meshgraph.GetComposites()
    domains = meshgraph.GetDomain()
    movement = meshgraph.GetMovement()

    for i, point in enumerate(itertools.chain.from_iterable(elements.points)):
        point.SetGlobalID(i)
        points[i] = point
    for i, seg in enumerate(itertools.chain.from_iterable(elements.segments)):
        seg.SetGlobalID(i)
        segments[i] = seg
    for i, curve in enumerate(itertools.chain.from_iterable(elements.curves)):
        curve.curveID = i
        curved_edges[i] = curve
    for i, face in enumerate(itertools.chain.from_iterable(elements.faces)):
        face.SetGlobalID(i)
        if isinstance(face, SD.TriGeom):
            tris[i] = face
        elif isinstance(face, SD.QuadGeom):
            quads[i] = face
        else:
            raise RuntimeError(f"Unexpected face geometry type {type(face)}.")
    for i, element in enumerate(itertools.chain.from_iterable(elements.elements)):
        element.SetGlobalID(i)
        if isinstance(element, SD.SegGeom):
            segments[i] = element
        elif isinstance(element, SD.TriGeom):
            tris[i] = element
        elif isinstance(element, SD.QuadGeom):
            quads[i] = element
        elif isinstance(element, SD.TetGeom):
            tets[i] = element
        elif isinstance(element, SD.PrismGeom):
            prisms[i] = element
        elif isinstance(element, SD.PyrGeom):
            pyrs[i] = element
        elif isinstance(element, SD.HexGeom):
            hexes[i] = element
        else:
            raise RuntimeError(f"Unexpected face geometry type {type(element)}.")

    # FIXME: The stuff related to Movement can probably be put in a
    # separate function, for tidiness, and/or use caching.
    for i, layer in enumerate(elements.layers):
        composites[i] = layer
        domain = nektar_composite_map(i, layer)
        domains[i] = domain
        movement.AddZone(SD.ZoneFixed(i, i, domain, 3))

    n = len(elements.layers)
    _near_faces = enumerate(elements.near_faces, n)
    far_faces = enumerate(elements.far_faces, n + len(elements.near_faces))
    first_near = next(_near_faces)
    near_faces = itertools.chain(_near_faces, [first_near])
    for i, ((j, near), (k, far)) in enumerate(zip(near_faces, far_faces)):
        composites[j] = near
        composites[k] = far
        near_interface = SD.Interface(2 * i, nektar_composite_map(j, near))
        far_interface = SD.Interface(2 * i + 1, nektar_composite_map(k, far))
        movement.AddInterface(f"Interface {i}", near_interface, far_interface)

    return meshgraph


def write_unstructured_grid(mesh: SD.MeshGraphXml, filename: str) -> None:
    mesh.Write(filename, False, SD.FieldMetaDataMap())


def write_nektar(mesh: Mesh, order: int, filename: str) -> None:
    nek_elements = nektar_elements(mesh, order)
    nek_mesh = nektar_mesh(nek_elements, 2, 3)
    write_unstructured_grid(nek_mesh, filename)
