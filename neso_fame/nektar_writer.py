from dataclasses import dataclass
from functools import cache, reduce
import itertools
from operator import attrgetter, or_
from typing import Iterator, Sequence

import NekPy.SpatialDomains as SD
import NekPy.LibUtilities as LU

from .mesh import QuadMesh, MeshLayer, Coord, Curve, Quad

UNSET_ID = -1

NektarQuadGeomElements = tuple[
    frozenset[SD.QuadGeom],
    frozenset[SD.SegGeom],
    frozenset[SD.PointGeom],
]


QuadLayerBoundItem = list[SD.SegGeom]
TetLayerBoundItem = list[SD.Geometry2D]
LayerBoundItem = QuadLayerBoundItem | TetLayerBoundItem
LayerBound = Sequence[LayerBoundItem]


@dataclass(frozen=True)
class NektarLayerCommon:
    points: frozenset[SD.PointGeom]
    segments: frozenset[SD.SegGeom]
    layer: SD.Composite
    near_face: SD.Composite
    far_face: SD.Composite


@dataclass(frozen=True)
class NektarLayer2D(NektarLayerCommon):
    elements: frozenset[SD.Geometry2D]
    layer: SD.Composite
    layer_bounds: Sequence[frozenset[SD.SegGeom]]


@dataclass(frozen=True)
class NektarLayer3D(NektarLayerCommon):
    faces: frozenset[SD.Geometry2D]
    elements: frozenset[SD.Geometry3D]
    bounds: Sequence[frozenset[SD.Geometry2D]]


NektarLayer = NektarLayer2D | NektarLayer3D


# FIXME: Do I really need this or could I just have a list of NektarLayer objects?
@dataclass
class NektarElements:
    _layers: list[NektarLayer]

    def points(self) -> Iterator[SD.PointGeom]:
        return itertools.chain.from_iterable(map(attrgetter("points"), self._layers))

    def segments(self) -> Iterator[SD.SegGeom]:
        return itertools.chain.from_iterable(map(attrgetter("segments"), self._layers))

    def faces(self) -> Iterator[SD.Geometry2D]:
        return itertools.chain.from_iterable(
            map(
                attrgetter("faces"),
                filter(lambda l: isinstance(l, NektarLayer3D), self._layers),
            )
        )

    def elements(self) -> Iterator[SD.Geometry2D] | Iterator[SD.Geometry3D]:
        return itertools.chain.from_iterable(map(attrgetter("elements"), self._layers))

    def layers(self) -> Iterator[SD.Composite]:
        return map(attrgetter("layer"), self._layers)

    def num_layers(self) -> int:
        return len(self._layers)

    def near_faces(self) -> Iterator[SD.Composite]:
        return map(attrgetter("near_face"), self._layers)

    def far_faces(self) -> Iterator[SD.Composite]:
        return map(attrgetter("far_face"), self._layers)

    def bounds(self) -> Iterator[SD.Composite]:
        zipped_bounds: Iterator[
            tuple[Sequence[SD.SegGeom | SD.Geometry2D]]
        ] = itertools.zip_longest(*map(attrgetter("layer_bounds"), self._layers), fillvalue=frozenset())
        return map(lambda ls: SD.Composite(list(reduce(or_, ls))), zipped_bounds)

    def num_bounds(self) -> int:
        return max(map(len, map(attrgetter("layer_bounds"), self._layers)))


@cache
def nektar_point(position: Coord, layer_id: int) -> SD.PointGeom:
    return SD.PointGeom(2, UNSET_ID, *position.to_cartesian())

# FIXME: There is a problem with caching these, as different calls to iterators produce different IDs for the functions contained in Curves. Working around it for now, but this is not robust.

@cache
def _nektar_curve(points: tuple[SD.PointGeom, ...], layer_id: int) -> tuple[SD.Curve, tuple[SD.PointGeom, SD.PointGeom]]:
    nek_curve = SD.Curve(UNSET_ID, LU.PointsType.PolyEvenlySpaced)
    nek_curve.points = list(points)
    return nek_curve, (points[0], points[-1])
    

def nektar_curve(
    curve: Curve, order: int, layer_id: int
) -> tuple[SD.Curve, tuple[SD.PointGeom, SD.PointGeom]]:
    points = tuple(
        nektar_point(coord, layer_id)
        for coord in curve.control_points(order).iter_points()
    )
    return _nektar_curve(points, layer_id)


@cache
def _nektar_edge(termini: tuple[SD.PointGeom, SD.PointGeom], nek_curve: SD.Curve) -> SD.SegGeom:
    return SD.SegGeom(UNSET_ID, termini[0].GetCoordim(), list(termini), nek_curve)


def nektar_edge(
    curve: Curve, order: int, layer_id: int
) -> tuple[SD.SegGeom, tuple[SD.PointGeom, SD.PointGeom]]:
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
        _nektar_edge(termini, nek_curve),
        termini,
    )


@cache
def nektar_quad(quad: Quad, order: int, layer_id: int) -> NektarQuadGeomElements:
    if quad.in_plane is not None:
        raise NotImplementedError("Not yet dealing with Quads as faces.")
    north, north_termini = nektar_edge(quad.north, order, layer_id)
    south, south_termini = nektar_edge(quad.south, order, layer_id)
    edges = [
        north,
        nektar_edge(quad.near, 1, layer_id)[0],
        south,
        nektar_edge(quad.far, 1, layer_id)[0],
    ]
    return (
        frozenset({SD.QuadGeom(UNSET_ID, edges)}),
        frozenset(edges),
        frozenset(north_termini + south_termini),
    )


def _combine_quad_items(
    quad1: NektarQuadGeomElements, quad2: NektarQuadGeomElements
) -> NektarQuadGeomElements:
    return (
        quad1[0] | quad2[0],
        quad1[1] | quad2[1],
        quad1[2] | quad2[2],
    )


def nektar_layer_elements(layer: MeshLayer, order: int, layer_id: int) -> NektarLayer:
    # FIXME: Currently inherantly 2D
    assert issubclass(layer.element_type, Quad)
    elems = list(layer)
    elements, edges, points = reduce(
        _combine_quad_items,
        (nektar_quad(elem, order, layer_id) for elem in elems),
    )
    layer_composite = SD.Composite(list(elements))
    # FIXME: This doesn't work for subdivided layers
    near_face = SD.Composite(list(layer.near_faces()))
    far_face = SD.Composite(list(layer.far_faces()))
    bounds = list(map(lambda x: frozenset(map(lambda y: nektar_edge(y, 1, layer_id)[0], x)), layer.boundaries()))
    return NektarLayer2D(
        points,
        edges,
        layer_composite,
        near_face,
        far_face,
        elements,
        bounds,
    )


def nektar_elements(mesh: QuadMesh, order: int) -> NektarElements:
    return NektarElements(
        [
            nektar_layer_elements(layer, order, i)
            for i, layer in enumerate(mesh.layers())
        ]
    )


def nektar_composite_map(comp_id: int, composite: SD.Composite) -> SD.CompositeMap:
    comp_map = SD.CompositeMap()
    comp_map[comp_id] = composite
    return comp_map


def nektar_mesh(
    elements: NektarElements, mesh_dim: int, spatial_dim: int, write_movement=True
) -> SD.MeshGraphXml:
    meshgraph = SD.MeshGraphXml(mesh_dim, spatial_dim)
    points = meshgraph.GetAllPointGeoms()
    segments = meshgraph.GetAllSegGeoms()
    curved_edges = meshgraph.GetCurvedEdges()
    tris = meshgraph.GetAllTriGeoms()
    quads = meshgraph.GetAllQuadGeoms()
    curved_faces = meshgraph.GetCurvedFaces()
    tets = meshgraph.GetAllTetGeoms()
    prisms = meshgraph.GetAllPrismGeoms()
    pyrs = meshgraph.GetAllPyrGeoms()
    hexes = meshgraph.GetAllHexGeoms()
    composites = meshgraph.GetComposites()
    domains = meshgraph.GetDomain()
    movement = meshgraph.GetMovement()

    for i, point in enumerate(elements.points()):
        point.SetGlobalID(i)
        points[i] = point
    for i, seg in enumerate(elements.segments()):
        seg.SetGlobalID(i)
        segments[i] = seg
        curve = seg.GetCurve()
        if curve is not None:
            curve.curveID = i
            curved_edges[i] = curve
    for i, face in enumerate(elements.faces()):
        face.SetGlobalID(i)
        if isinstance(face, SD.TriGeom):
            tris[i] = face
        elif isinstance(face, SD.QuadGeom):
            quads[i] = face
        else:
            raise RuntimeError(f"Unexpected face geometry type {type(face)}.")
        curve = face.GetCurve()
        if curve is not None:
            curve.curveID = i
            curved_faces[i] = curve
    for i, element in enumerate(elements.elements()):
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
    i = -1
    for i, layer in enumerate(elements.layers()):
        composites[i] = layer
        domain = nektar_composite_map(i, layer)
        domains[i] = domain
        if write_movement:
            movement.AddZone(SD.ZoneFixed(i, i, domain, 3))

    n = elements.num_layers()

    # FIXME: Make wrapping of interfaces optional
    _near_faces = enumerate(elements.near_faces(), n)
    far_faces = enumerate(elements.far_faces(), 2*n)
    first_near = next(_near_faces)
    near_faces = itertools.chain(_near_faces, [first_near])
    for i, ((j, near), (k, far)) in enumerate(zip(near_faces, far_faces)):
        composites[j] = near
        composites[k] = far
        if write_movement:
            near_interface = SD.Interface(2 * i, nektar_composite_map(j, near))
            far_interface = SD.Interface(2 * i + 1, nektar_composite_map(k, far))
            movement.AddInterface(f"Interface {i}", far_interface, near_interface)
    for i, bound in enumerate(elements.bounds(), 3*n):
        composites[i] = bound

    return meshgraph


def write_nektar(mesh: QuadMesh, order: int, filename: str, write_movement=True) -> None:
    nek_elements = nektar_elements(mesh, order)
    # FIXME: Need to be able to configure dimensiosn
    nek_mesh = nektar_mesh(nek_elements, 2, 2, write_movement)
    nek_mesh.Write(filename, True, SD.FieldMetaDataMap())
