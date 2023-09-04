"""Tools for creating Nektar++ objects for representing a mesh.

"""

import itertools
from ctypes import ArgumentError
from dataclasses import dataclass
from functools import cache, reduce
from operator import attrgetter, or_
from typing import Iterator, Optional, Sequence, cast

import NekPy.LibUtilities as LU
import NekPy.SpatialDomains as SD

from .mesh import (
    Coord,
    EndQuad,
    Hex,
    Mesh,
    MeshLayer,
    NormalisedCurve,
    Quad,
    QuadMesh,
    control_points,
)

UNSET_ID = -1


NektarQuadGeomElements = tuple[
    frozenset[SD.QuadGeom],
    frozenset[SD.SegGeom],
    frozenset[SD.PointGeom],
]

NektarOrderedQuadGeomElements = tuple[
    list[SD.QuadGeom], frozenset[SD.SegGeom], frozenset[SD.PointGeom]
]

NektarHexGeomElements = tuple[
    frozenset[SD.HexGeom],
    frozenset[SD.QuadGeom],
    frozenset[SD.SegGeom],
    frozenset[SD.PointGeom],
]


@dataclass(frozen=True)
class _NektarLayerCommon:
    """Base type for NektarLayer objects, containing attributes common
    to both 2D and 3D meshes.

    """

    points: frozenset[SD.PointGeom]
    """The Nektar++ point geometry objects contained in this layer."""
    segments: frozenset[SD.SegGeom]
    """The Nektar++ segment geometry (edge) objects contained in this layer."""
    layer: SD.Composite
    """The Nektar++ composite object representing this layer."""
    near_face: SD.Composite
    """The Nektar++ composite object representing the near face of this layer."""
    far_face: SD.Composite
    """The Nektar++ composite object representing the far face of this layer."""


@dataclass(frozen=True)
class NektarLayer2D(_NektarLayerCommon):
    """Represents the Nektar++ objects present in a single layer of
    a 2D mesh.

    Group
    -----
    collection

    """

    elements: frozenset[SD.Geometry2D]
    """The Nektar++ quad geometry elements for this layer."""
    bounds: Sequence[frozenset[SD.SegGeom]]
    """An ordered sequence of sets of edges, where set represents a
    particular boundary region."""


@dataclass(frozen=True)
class NektarLayer3D(_NektarLayerCommon):
    """Represents the Nektar++ objects present in a single layer of
    a 3D mesh.

    Group
    -----
    collection

    """

    faces: frozenset[SD.Geometry2D]
    """The Nektar++ quad geometry (face) objects contained in this layer."""
    elements: frozenset[SD.Geometry3D]
    """The Nektar++ hex geometry elements for this layer"""
    bounds: Sequence[frozenset[SD.Geometry2D]]
    """An ordered sequence of sets of faces, where set represents a
    particular boundary region."""


NektarLayer = NektarLayer2D | NektarLayer3D
"""Type representing a layer of either a 2D or 3D Nektar++ mesh.

Group
-----
collection


.. rubric:: Alias
"""


@dataclass
class NektarElements:
    """Represents all of the Nektar++ objects that make up a mesh, but
    not yet assembled into a MeshGraph objects.

    Group
    -----
    collection

    """

    _layers: list[NektarLayer]

    def points(self) -> Iterator[SD.PointGeom]:
        """Iterate over all of the PointGeom objects in the mesh."""
        return itertools.chain.from_iterable(map(attrgetter("points"), self._layers))

    def segments(self) -> Iterator[SD.SegGeom]:
        """Iterate over all of the SegGeom objects in the mesh."""
        return itertools.chain.from_iterable(map(attrgetter("segments"), self._layers))

    def faces(self) -> Iterator[SD.Geometry2D]:
        """Iterate over all of the 2D faces present in the mesh. Will
        be empty if your mesh is 2D.

        """
        return itertools.chain.from_iterable(
            map(
                attrgetter("faces"),
                filter(lambda layer: isinstance(layer, NektarLayer3D), self._layers),
            )
        )

    def elements(self) -> Iterator[SD.Geometry2D] | Iterator[SD.Geometry3D]:
        """Iterate over all of the elements in the mesh."""
        return itertools.chain.from_iterable(map(attrgetter("elements"), self._layers))

    def layers(self) -> Iterator[SD.Composite]:
        """Iterate ovre the Composite objects representing each of the
        layers of the mesh.

        """
        return map(attrgetter("layer"), self._layers)

    def num_layers(self) -> int:
        """Returns the number of layers present in the mesh."""
        return len(self._layers)

    def near_faces(self) -> Iterator[SD.Composite]:
        """Iterates over the Composite objects representing the near face of
        each layer of the mesh.

        """
        return map(attrgetter("near_face"), self._layers)

    def far_faces(self) -> Iterator[SD.Composite]:
        """Iterates over the Composite objects representing the far
        face of each layer of the mesh.

        """
        return map(attrgetter("far_face"), self._layers)

    def bounds(self) -> Iterator[SD.Composite]:
        """Iterates over the Composite objects representing each of
        the boundary regions of the mesh. This does not include
        boundaries which are normal to the x3-direction.

        """
        zipped_bounds: Iterator[
            tuple[Sequence[SD.SegGeom | SD.Geometry2D]]
        ] = itertools.zip_longest(
            *map(attrgetter("bounds"), self._layers), fillvalue=frozenset()
        )
        return map(lambda ls: SD.Composite(list(reduce(or_, ls))), zipped_bounds)

    def num_bounds(self) -> int:
        """Returns the number of boundary regions of teh mesh, not
        counting those which are perpendicular to the x3-direction.

        """
        return max(map(len, map(attrgetter("bounds"), self._layers)))


def _round_zero(x: float, tol: float) -> float:
    """Rounds the number to 0 if it is less than the tolerance."""
    if abs(x) < tol:
        return 0.0
    return x


@cache
def nektar_point(position: Coord, spatial_dim: int, layer_id: int) -> SD.PointGeom:
    """Returns a Nektar++ PointGeom object at the specified position
    in the given layer. Caching is used to ensure that, given the same
    location and layer, the object will always be the same.

    Group
    -----
    factory
    """
    pos = position.to_cartesian()
    tol = pos.TOLERANCE / 10
    return SD.PointGeom(
        spatial_dim,
        UNSET_ID,
        _round_zero(pos.x1, tol),
        _round_zero(pos.x2, tol),
        _round_zero(pos.x3, tol),
    )


@cache
def _nektar_curve(
    points: tuple[SD.PointGeom, ...], layer_id: int
) -> tuple[SD.Curve, tuple[SD.PointGeom, SD.PointGeom]]:
    nek_curve = SD.Curve(UNSET_ID, LU.PointsType.PolyEvenlySpaced)
    nek_curve.points = list(points)
    return nek_curve, (points[0], points[-1])


def nektar_curve(
    curve: NormalisedCurve | Quad, order: int, spatial_dim: int, layer_id: int
) -> tuple[SD.Curve, tuple[SD.PointGeom, SD.PointGeom]]:
    """Returns a Nektar++ Curve object and the PointGeom objects
    corresponding to the start and end of the given curve. The curve
    will be represented to the specified order. Caching is used to
    ensure that the same curve, in the same layer, represented to the
    same order, will always return the same objects. The caching is
    done based on the locations of the control points of the curve,
    rather than the identity of the function defining the curve.

    Group
    -----
    factory
    """
    points = tuple(
        nektar_point(coord, spatial_dim, layer_id)
        for coord in control_points(curve, order).iter_points()
    )
    return _nektar_curve(points, layer_id)


@cache
def _nektar_edge(
    termini: tuple[SD.PointGeom, SD.PointGeom], nek_curve: SD.Curve
) -> SD.SegGeom:
    return SD.SegGeom(UNSET_ID, termini[0].GetCoordim(), list(termini), nek_curve)


def nektar_edge(
    curve: NormalisedCurve, order: int, spatial_dim: int, layer_id: int
) -> tuple[SD.SegGeom, tuple[SD.PointGeom, SD.PointGeom]]:
    """Returns a Nektar++ SegGeom representing the curve in the
    specified layer, to the specified order. It also returns the
    PointGeom objects representing the start and end of the segment.
    Caching is used to ensure that the same curve, in the same layer,
    represented to the same order, will always return the same
    objects. The caching is done based on the locations of the control
    points of the curve, rather than the identity of the function
    defining the curve.

    Group
    -----
    factory
    """
    if order > 1:
        nek_curve, termini = nektar_curve(curve, order, spatial_dim, layer_id)
    else:
        nek_curve = None
        end_points = control_points(curve, 1)
        termini = (
            nektar_point(end_points[0], spatial_dim, layer_id),
            nektar_point(end_points[1], spatial_dim, layer_id),
        )
    return (
        _nektar_edge(termini, nek_curve),
        termini,
    )


@cache
def _nektar_quad(
    edges: tuple[SD.SegGeom, SD.SegGeom, SD.SegGeom, SD.SegGeom],
    nek_curve: Optional[SD.Curve],
) -> SD.QuadGeom:
    if nek_curve is not None:
        return SD.QuadGeom(UNSET_ID, list(edges), nek_curve)
    else:
        return SD.QuadGeom(UNSET_ID, list(edges))


def nektar_quad(
    quad: Quad | EndQuad, order: int, spatial_dim: int, layer_id: int
) -> NektarQuadGeomElements:
    """Returns a Nektar++ QuadGeom objects (along with the SegGeom and
    PointGeom objects that make it up) representing the given quad, to
    the given order. Caching is used to ensure the same quad, in the
    same layer, represented to the same order will always return the
    same objects. The caching is done based on the locations of the
    control points of the quad and its edgs, rather than the identity
    of the quad.

    Group
    -----
    factory

    """
    north, north_termini = nektar_edge(quad.north, order, spatial_dim, layer_id)
    south, south_termini = nektar_edge(quad.south, order, spatial_dim, layer_id)
    if isinstance(quad, EndQuad):
        east, east_termini = nektar_edge(quad.east, order, spatial_dim, layer_id)
        west, west_termini = nektar_edge(quad.west, order, spatial_dim, layer_id)
        points = frozenset(north_termini + south_termini + east_termini + west_termini)
        assert len(points) == 4, "Ill-formed quad; edges do not join into 4 corners"
        edges = (north, east, south, west)
    else:
        edges = (
            north,
            nektar_edge(quad.near, order, spatial_dim, layer_id)[0],
            south,
            nektar_edge(quad.far, order, spatial_dim, layer_id)[0],
        )
        points = frozenset(north_termini + south_termini)
    # FIXME: Pretty sure I should refactor so I can get curved surfaces for end quads
    if order > 1 and not isinstance(quad, EndQuad):
        curve, _ = nektar_curve(quad, order, spatial_dim, layer_id)
    else:
        curve = None
    nek_quad = _nektar_quad(edges, curve)
    return (frozenset({nek_quad}), frozenset(edges), points)


@cache
def nektar_hex(
    hexa: Hex, order: int, spatial_dim: int, layer_id: int
) -> NektarHexGeomElements:
    """Returns a Nektar++ HexGeom object (along with teh QuadGeom,
    SegGeom and PointGeom objects that make it up) representing a
    given hex, to the given order. Caching is used to ensure the same
    quad, in the same layer, represented to the same order will always
    return the same objects.

    """
    # Be careful with order of faces; needs to be bottom, vertical
    # faces, then top (although actual oreintation in space is
    # irrelevant)
    init: tuple[list[SD.QuadGeom], frozenset[SD.SegGeom], frozenset[SD.PointGeom]] = (
        [],
        frozenset(),
        frozenset(),
    )
    faces, segments, points = reduce(
        _combine_quad_items_ordered,
        (
            nektar_quad(quad, order, spatial_dim, layer_id)
            for quad in cast(
                Iterator[Quad | EndQuad], itertools.chain([hexa.near], hexa, [hexa.far])
            )
        ),
        init,
    )
    for i, edge in enumerate(segments):
        edge.SetGlobalID(i)
    for i, face in enumerate(faces):
        face.SetGlobalID(i)
    nek_hex = SD.HexGeom(UNSET_ID, faces)
    for edge in segments:
        edge.SetGlobalID(UNSET_ID)
    return frozenset({nek_hex}), frozenset(faces), segments, points


def _combine_quad_items(
    quad1: NektarQuadGeomElements, quad2: NektarQuadGeomElements
) -> NektarQuadGeomElements:
    return (
        quad1[0] | quad2[0],
        quad1[1] | quad2[1],
        quad1[2] | quad2[2],
    )


def _combine_quad_items_ordered(
    all_quads: NektarOrderedQuadGeomElements, new_quad: NektarQuadGeomElements
) -> NektarOrderedQuadGeomElements:
    quads = all_quads[0]
    quads.append(next(iter(new_quad[0])))
    return (quads, all_quads[1] | new_quad[1], all_quads[2] | new_quad[2])


def _combine_hex_items(
    hex1: NektarHexGeomElements, hex2: NektarHexGeomElements
) -> NektarHexGeomElements:
    return (
        hex1[0] | hex2[0],
        hex1[1] | hex2[1],
        hex1[2] | hex2[2],
        hex1[3] | hex2[3],
    )


def nektar_layer_elements(
    layer: MeshLayer, order: int, spatial_dim: int, layer_id: int
) -> NektarLayer:
    """Creates Nektar++ objects needed to represent the given mesh
    layer to the given order.

    Group
    -----
    factory

    """
    # FIXME: Currently inherantly 2D
    elems = list(layer)
    elements: frozenset[SD.QuadGeom] | frozenset[SD.HexGeom]
    if issubclass(layer.element_type, Quad):
        elements, edges, points = reduce(
            _combine_quad_items,
            (nektar_quad(elem, order, spatial_dim, layer_id) for elem in elems),
        )

        def make_face(
            item: NormalisedCurve | EndQuad | Quad,
            order: int,
            spatial_dim: int,
            layer_id,
        ) -> SD.SegGeom | SD.QuadGeom:
            assert not isinstance(item, (EndQuad, Quad))
            return nektar_edge(item, order, spatial_dim, layer_id)[0]

    else:
        elements, faces, edges, points = reduce(
            _combine_hex_items,
            (nektar_hex(elem, order, spatial_dim, layer_id) for elem in elems),
        )

        def make_face(
            item: NormalisedCurve | EndQuad | Quad,
            order: int,
            spatial_dim: int,
            layer_id,
        ) -> SD.SegGeom | SD.QuadGeom:
            assert isinstance(item, (EndQuad, Quad))
            return next(iter(nektar_quad(item, order, spatial_dim, layer_id)[0]))

    layer_composite = SD.Composite(list(elements))
    near_face = SD.Composite(
        [make_face(f, order, spatial_dim, layer_id) for f in layer.near_faces()]
    )
    far_face = SD.Composite(
        [make_face(f, order, spatial_dim, layer_id) for f in layer.far_faces()]
    )
    bounds = list(
        map(
            lambda x: frozenset(
                map(lambda y: make_face(y, 1, spatial_dim, layer_id), x)
            ),
            layer.boundaries(),
        )
    )
    if issubclass(layer.element_type, Quad):
        return NektarLayer2D(
            points,
            edges,
            layer_composite,
            near_face,
            far_face,
            cast(frozenset[SD.QuadGeom], elements),
            cast(list[frozenset[SD.SegGeom]], bounds),
        )
    else:
        return NektarLayer3D(
            points,
            edges,
            layer_composite,
            near_face,
            far_face,
            faces,
            cast(frozenset[SD.HexGeom], elements),
            cast(list[frozenset[SD.QuadGeom]], bounds),
        )


def nektar_elements(mesh: Mesh, order: int, spatial_dim: int) -> NektarElements:
    """Creates a collection of Nektar++ objects representing the given
    mesh.

    Group
    -----
    public nektar
    """
    return NektarElements(
        [
            nektar_layer_elements(cast(MeshLayer, layer), order, spatial_dim, i)
            for i, layer in enumerate(mesh.layers())
        ]
    )


def nektar_composite_map(comp_id: int, composite: SD.Composite) -> SD.CompositeMap:
    """Creates Nektar++ CompositeMap objects containing a single composite.

    Group
    -----
    factory

    """
    comp_map = SD.CompositeMap()
    comp_map[comp_id] = composite
    return comp_map


def nektar_mesh(
    elements: NektarElements,
    mesh_dim: int,
    spatial_dim: int,
    write_movement=True,
    periodic_interfaces=True,
) -> SD.MeshGraphXml:
    """Creates a Nektar++ MeshGraphXml object from a collection of
    Nektar++ geometry objects.

    Parameters
    ----------
    elements
        The collection of Nektar++ objects to be assembled into a
        MeshGraph object.
    mesh_dim
        The dimension of the elements of the mesh.
    spatial_dim
        The dimension of the space in which the mesh elements sit.
    write_movement
        Whether to write information on non-conformal zones and
        interfaces.
    periodic_interfaces
        If write_movement is True, whether the last layer joins
        back up with the first, requiring an interface to be
        defined between the two.

    Danger
    ------
    Be very careful if calling this function more than once in the
    same session. The IDs of the geometry objects are only set
    when creating the MeshGraph in this function and calling it a
    second time with some of the same elements could result in the
    ID being changed. You must take particular care because
    geometry elements are cached. Always write out (or otherwise
    finish with) a MeshGraph object before calling this method
    again. The safest way to do this is not to call this function
    direclty but instead use `write_nektar`.

    Group
    -----
    public nektar

    """
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
        if isinstance(element, SD.Geometry2D):
            curve = element.GetCurve()
            if curve is not None:
                curve.curveID = i
                curved_faces[i] = curve

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

    near_faces: Iterator[tuple[int, SD.Composite]] = enumerate(elements.near_faces(), n)
    first_near = next(near_faces)
    near_faces = itertools.chain(near_faces, [first_near])
    far_faces = enumerate(elements.far_faces(), 2 * n)
    for i, ((j, near), (k, far)) in enumerate(zip(near_faces, far_faces)):
        composites[j] = near
        composites[k] = far
        if write_movement and (i != n - 1 or periodic_interfaces):
            near_interface = SD.Interface(2 * i, nektar_composite_map(j, near))
            far_interface = SD.Interface(2 * i + 1, nektar_composite_map(k, far))
            movement.AddInterface(f"Interface {i}", far_interface, near_interface)
    for i, bound in enumerate(elements.bounds(), 3 * n):
        composites[i] = bound

    return meshgraph


def write_nektar(
    mesh: QuadMesh,
    order: int,
    filename: str,
    spatial_dim: int = 3,
    write_movement=True,
    periodic_interfaces=True,
) -> None:
    """Create a Nektar++ MeshGraph object from your mesh and write it
    to the disk.

    Parameters
    ----------
    mesh
        The mesh to be converted to Nektar++ format.
    order
        The order to use when representing the elements of the mesh.
    filename
        The name of the file to write the mesh to. Should use the
        ``.xml`` extension.
    spatial_dim
        The dimension of the space in which the mesh elements sit.
    write_movement
        Whether to write information on non-conformal zones and
        interfaces.
    periodic_interfaces
        If write_movement is True, whether the last layer joins
        back up with the first, requiring an interface to be
        defined between the two.

    Group
    -----
    public nektar

    """
    mesh_dim = 3 if isinstance(mesh.reference_layer.element_type, Hex) else 2
    if spatial_dim < mesh_dim:
        raise ArgumentError(
            f"Spatial dimension ({spatial_dim}) must be at least equal to mesh "
            f"dimension ({mesh_dim})"
        )
    nek_elements = nektar_elements(mesh, order, spatial_dim)
    # FIXME: Need to be able to configure dimensiosn
    nek_mesh = nektar_mesh(
        nek_elements, mesh_dim, spatial_dim, write_movement, periodic_interfaces
    )
    nek_mesh.Write(filename, True, SD.FieldMetaDataMap())
