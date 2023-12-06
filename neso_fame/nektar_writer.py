"""Tools for creating Nektar++ objects for representing a mesh."""

import itertools
from ctypes import ArgumentError
from dataclasses import dataclass
from functools import cache, reduce
from operator import attrgetter, or_
from typing import Iterable, Iterator, Optional, Sequence, cast

import NekPy.LibUtilities as LU
import NekPy.SpatialDomains as SD

from .mesh import (
    Coord,
    EndShape,
    Mesh,
    MeshLayer,
    NormalisedCurve,
    Prism,
    Quad,
    control_points,
)

UNSET_ID = -1


Nektar2dGeomElements = tuple[
    frozenset[SD.Geometry2D],
    frozenset[SD.SegGeom],
    frozenset[SD.PointGeom],
]

NektarOrdered2dGeomElements = tuple[
    list[SD.Geometry2D], frozenset[SD.SegGeom], frozenset[SD.PointGeom]
]

Nektar3dGeomElements = tuple[
    frozenset[SD.Geometry3D],
    frozenset[SD.Geometry2D],
    frozenset[SD.SegGeom],
    frozenset[SD.PointGeom],
]


@dataclass(frozen=True)
class _NektarLayerCommon:
    """Base type for NektarLayer objects.

    Contains attributes common to both 2D and 3D meshes.

    """

    points: frozenset[SD.PointGeom]
    """The Nektar++ point geometry objects contained in this layer."""
    segments: frozenset[SD.SegGeom]
    """The Nektar++ segment geometry (edge) objects contained in this layer."""
    layer: Sequence[SD.Composite]
    """The Nektar++ composite objects representing this layer."""
    near_face: SD.Composite
    """The Nektar++ composite object representing the near face of this layer."""
    far_face: SD.Composite
    """The Nektar++ composite object representing the far face of this layer."""


@dataclass(frozen=True)
class NektarLayer2D(_NektarLayerCommon):
    """Represents the Nektar++ objects present in a single layer of a 2D mesh.

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
    """Represents the Nektar++ objects present in a single layer of a 3D mesh.

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
    """Represents all of the Nektar++ objects that make up a mesh.

    These are real Nektar++ objects, but they are not yet assembled
    into a MeshGraph objects.

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
        """Iterate over all of the 2D faces present in the mesh.

        This will be empty if your mesh is 2D.

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

    def layers(self) -> Iterator[Sequence[SD.Composite]]:
        """Iterate over the Composite objects representing each layer of the mesh."""
        return map(attrgetter("layer"), self._layers)

    def num_layers(self) -> int:
        """Return the number of layers present in the mesh."""
        return len(self._layers)

    def near_faces(self) -> Iterator[SD.Composite]:
        """Iterate over Composite objects representing the near face of each layer."""
        return map(attrgetter("near_face"), self._layers)

    def far_faces(self) -> Iterator[SD.Composite]:
        """Iterate over Composite objects representing the far face of each layer."""
        return map(attrgetter("far_face"), self._layers)

    def bounds(self) -> Iterator[SD.Composite]:
        """Iterate over Composite objects representing boundaries of the mesh.

        This does not include boundaries which are normal to the x3-direction.

        """
        zipped_bounds = itertools.zip_longest(
            *(layer.bounds for layer in self._layers), fillvalue=frozenset()
        )

        def combine_bounds(
            bounds: Sequence[frozenset[SD.Geometry]]
        ) -> list[SD.Geometry]:
            return list(reduce(or_, bounds))

        return map(SD.Composite, filter(len, map(combine_bounds, zipped_bounds)))

    def num_bounds(self) -> int:
        """Return the number of boundary regions of the mesh.

        This is not those boundaries are perpendicular to the x3-direction.

        """
        return max(map(len, map(attrgetter("bounds"), self._layers)))


def _round_zero(x: float, tol: float) -> float:
    """Round the number to 0 if it is less than the tolerance."""
    if abs(x) < tol:
        return 0.0
    return x


@cache
def nektar_point(position: Coord, spatial_dim: int, layer_id: int) -> SD.PointGeom:
    """Return a Nektar++ PointGeom object at the specified position.

    Caching is used to ensure that, given the same location and layer,
    the object will always be the same.

    Group
    -----
    factory

    """
    pos = position.to_cartesian()
    tol = pos.TOLERANCE
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
    """Return a Nektar++ Curve object and its start and end points.

    The curve will be represented to the specified order. Caching is
    used to ensure that the same curve, in the same layer, represented
    to the same order, will always return the same objects. The
    caching is done based on the locations of the control points of
    the curve, rather than the identity of the function defining the
    curve.

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
    """Return a Nektar++ SegGeom representing the curve to the specified order.

    It also returns the PointGeom objects representing the start and
    end of the segment.  Caching is used to ensure that the same
    curve, in the same layer, represented to the same order, will
    always return the same objects. The caching is done based on the
    locations of the control points of the curve, rather than the
    identity of the function defining the curve.

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
def _nektar_triangle(
    edges: tuple[SD.SegGeom, SD.SegGeom, SD.SegGeom],
    nek_curve: Optional[SD.Curve],
) -> SD.TriGeom:
    if nek_curve is not None:
        return SD.TriGeom(UNSET_ID, list(edges), nek_curve)
    else:
        return SD.TriGeom(UNSET_ID, list(edges))


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
    quad: Quad, order: int, spatial_dim: int, layer_id: int
) -> Nektar2dGeomElements:
    """Return a Nektar++ QuadGeom object and its constituents.

    Curved quads will be represented to the given order of
    accuracy. Caching is used to ensure the same quad, in the same
    layer, represented to the same order will always return the same
    objects. The caching is done based on the locations of the control
    points of the quad and its edges, rather than the identity of the
    quad.

    Group
    -----
    factory

    """
    north, north_termini = nektar_edge(quad.north, order, spatial_dim, layer_id)
    south, south_termini = nektar_edge(quad.south, order, spatial_dim, layer_id)
    edges = (
        north,
        nektar_edge(quad.near, order, spatial_dim, layer_id)[0],
        south,
        nektar_edge(quad.far, order, spatial_dim, layer_id)[0],
    )
    points = frozenset(north_termini + south_termini)
    # FIXME: Pretty sure I should refactor so I can get curved surfaces for end quads
    if order > 1:
        curve, _ = nektar_curve(quad, order, spatial_dim, layer_id)
    else:
        curve = None
    nek_quad = _nektar_quad(edges, curve)
    return (frozenset({nek_quad}), frozenset(edges), points)


def nektar_end_shape(
    shape: EndShape, order: int, spatial_dim: int, layer_id: int
) -> Nektar2dGeomElements:
    """Return a Nektar++ QuadGeom or TriGeom object and its constituents.

    Curved quads/triangles will be represented to the given order of
    accuracy. Caching is used to ensure the same quad/triangle, in the same
    layer, represented to the same order will always return the same
    objects. The caching is done based on the locations of the control
    points of the edges, rather than the identity of the face.

    Group
    -----
    factory

    """
    if len(shape.edges) == 3:
        north, north_termini = nektar_edge(shape.edges[0], order, spatial_dim, layer_id)
        east, east_termini = nektar_edge(shape.edges[1], order, spatial_dim, layer_id)
        south, south_termini = nektar_edge(shape.edges[2], order, spatial_dim, layer_id)
        points = frozenset(north_termini + south_termini + east_termini)
        if len(points) != 3:
            raise RuntimeError("Ill-formed triangle; edges do not join into 3 corners")
        edges: tuple[SD.SegGeom, ...] = (north, east, south)
        nek_shape: SD.Geometry2D = _nektar_triangle(edges, None)
    elif len(shape.edges) == 4:
        north, north_termini = nektar_edge(shape.edges[0], order, spatial_dim, layer_id)
        south, south_termini = nektar_edge(shape.edges[1], order, spatial_dim, layer_id)
        east, east_termini = nektar_edge(shape.edges[2], order, spatial_dim, layer_id)
        west, west_termini = nektar_edge(shape.edges[3], order, spatial_dim, layer_id)
        points = frozenset(north_termini + south_termini + east_termini + west_termini)
        if len(points) != 4:
            raise RuntimeError("Ill-formed quad; edges do not join into 4 corners")
        edges = (north, east, south, west)
        nek_shape = _nektar_quad(edges, None)
    else:
        raise ValueError(f"Can not handle shape with {len(shape.edges)} sides.")
    return frozenset({nek_shape}), frozenset(edges), points


@cache
def nektar_3d_element(
    solid: Prism, order: int, spatial_dim: int, layer_id: int
) -> Nektar3dGeomElements:
    """Return a Nektar++ HexGeom or PrismGeom object and its components.

    Curved hexes and prisms will be represented to the given order of
    accuracy. Caching is used to ensure the same hex/prism, in the same
    layer, represented to the same order will always return the same
    objects.

    Group
    -----
    factory

    """
    # Be careful with order of faces; needs to be bottom, vertical
    # faces, then top (although actual oreintation in space is
    # irrelevant)
    init: tuple[list[SD.Geometry2D], frozenset[SD.SegGeom], frozenset[SD.PointGeom]] = (
        [],
        frozenset(),
        frozenset(),
    )
    if len(solid.sides) == 4:
        ordered_sides = [
            nektar_end_shape(solid.near, order, spatial_dim, layer_id),
            nektar_quad(solid.sides[0], order, spatial_dim, layer_id),
            nektar_quad(solid.sides[2], order, spatial_dim, layer_id),
            nektar_quad(solid.sides[1], order, spatial_dim, layer_id),
            nektar_quad(solid.sides[3], order, spatial_dim, layer_id),
            nektar_end_shape(solid.far, order, spatial_dim, layer_id),
        ]
    elif len(solid.sides) == 3:
        ordered_sides = [
            nektar_quad(solid.sides[0], order, spatial_dim, layer_id),
            nektar_end_shape(solid.near, order, spatial_dim, layer_id),
            nektar_quad(solid.sides[1], order, spatial_dim, layer_id),
            nektar_end_shape(solid.far, order, spatial_dim, layer_id),
            nektar_quad(solid.sides[2], order, spatial_dim, layer_id),
        ]
    else:
        raise ValueError(
            f"Element with {len(solid.sides)} quadrilateral faces is not recognized."
        )
    faces, segments, points = reduce(
        _combine_2d_items_ordered,
        ordered_sides,
        init,
    )
    for i, edge in enumerate(segments):
        edge.SetGlobalID(i)
    for i, face in enumerate(faces):
        face.SetGlobalID(i)
    if len(solid.sides) == 4:
        nek_solid: SD.Geometry3D = SD.HexGeom(UNSET_ID, cast(list[SD.QuadGeom], faces))
    elif len(solid.sides) == 3:
        nek_solid = SD.PrismGeom(UNSET_ID, faces)
    else:
        assert False
    for edge in segments:
        edge.SetGlobalID(UNSET_ID)
    return frozenset({nek_solid}), frozenset(faces), segments, points


def _combine_2d_items(
    geom1: Nektar2dGeomElements, geom2: Nektar2dGeomElements
) -> Nektar2dGeomElements:
    return (
        geom1[0] | geom2[0],
        geom1[1] | geom2[1],
        geom1[2] | geom2[2],
    )


def _combine_2d_items_ordered(
    all_geoms: NektarOrdered2dGeomElements, new_geom: Nektar2dGeomElements
) -> NektarOrdered2dGeomElements:
    geom = all_geoms[0]
    geom.append(next(iter(new_geom[0])))
    return (geom, all_geoms[1] | new_geom[1], all_geoms[2] | new_geom[2])


def _combine_3d_items(
    prism1: Nektar3dGeomElements, prism2: Nektar3dGeomElements
) -> Nektar3dGeomElements:
    return (
        prism1[0] | prism2[0],
        prism1[1] | prism2[1],
        prism1[2] | prism2[2],
        prism1[3] | prism2[3],
    )


def nektar_layer_elements(
    layer: MeshLayer, order: int, spatial_dim: int, layer_id: int
) -> NektarLayer:
    """Create all Nektar++ objects needed to represent a layer of a mesh.

    Group
    -----
    factory

    """
    elems = iter(layer)
    elements: frozenset[SD.Geometry2D] | frozenset[SD.Geometry3D]
    if issubclass(layer.element_type, Quad):
        elements, edges, points = reduce(
            _combine_2d_items,
            (nektar_quad(elem, order, spatial_dim, layer_id) for elem in elems),
        )

        def make_face(
            item: NormalisedCurve | EndShape | Quad,
        ) -> SD.SegGeom | SD.Geometry2D:
            assert not isinstance(item, (Quad, EndShape))
            return nektar_edge(item, order, spatial_dim, layer_id)[0]

    else:
        elements, faces, edges, points = reduce(
            _combine_3d_items,
            (nektar_3d_element(elem, order, spatial_dim, layer_id) for elem in elems),
        )

        def make_face(
            item: NormalisedCurve | EndShape | Quad,
        ) -> SD.SegGeom | SD.Geometry2D:
            assert isinstance(item, (Quad, EndShape))
            if isinstance(item, Quad):
                return next(iter(nektar_quad(item, order, spatial_dim, layer_id)[0]))
            else:
                return next(
                    iter(nektar_end_shape(item, order, spatial_dim, layer_id)[0])
                )

    def type_name(element: SD.Geometry) -> str:
        return element.__class__.__name__

    layer_composite = [SD.Composite(list(e)) for _, e in itertools.groupby(sorted(elements, key=type_name), type_name)]
    near_face = SD.Composite([make_face(f) for f in layer.near_faces()])
    far_face = SD.Composite([make_face(f) for f in layer.far_faces()])
    bounds = [frozenset(make_face(y) for y in x) for x in layer.boundaries()]
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


def _enumerate_layers(layers: Iterable[MeshLayer]) -> Iterator[tuple[int, MeshLayer]]:
    for i, ly in enumerate(layers):
        print(f"Processing layer centred at x3 = {ly.x3_offset}")
        yield i, ly


def nektar_elements(mesh: Mesh, order: int, spatial_dim: int) -> NektarElements:
    """Create a collection of Nektar++ objects representing the mesh.

    Group
    -----
    public nektar
    """
    print("Converting FAME mesh to NekPy objects")
    return NektarElements(
        [
            nektar_layer_elements(cast(MeshLayer, layer), order, spatial_dim, i)
            for i, layer in _enumerate_layers(mesh.layers())
        ]
    )


def nektar_composite_map(composite_map: dict[int, SD.Composite]) -> SD.CompositeMap:
    """Create Nektar++ CompositeMap objects containing a single composite.

    Group
    -----
    factory

    """
    comp_map = SD.CompositeMap()
    for i, composite in composite_map.items():
        comp_map[i] = composite
    return comp_map


def _assign_points(elements: NektarElements, meshgraph: SD.MeshGraphXml) -> None:
    points = meshgraph.GetAllPointGeoms()
    for i, point in enumerate(elements.points()):
        point.SetGlobalID(i)
        points[i] = point


def _assign_segments(elements: NektarElements, meshgraph: SD.MeshGraphXml) -> None:
    segments = meshgraph.GetAllSegGeoms()
    curved_edges = meshgraph.GetCurvedEdges()
    for i, seg in enumerate(elements.segments()):
        seg.SetGlobalID(i)
        segments[i] = seg
        curve = seg.GetCurve()
        if curve is not None:
            curve.curveID = i
            curved_edges[i] = curve


def _assign_faces(elements: NektarElements, meshgraph: SD.MeshGraphXml) -> None:
    """Assign faces to a MeshGraph object."""
    tris = meshgraph.GetAllTriGeoms()
    quads = meshgraph.GetAllQuadGeoms()
    curved_faces = meshgraph.GetCurvedFaces()
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


def _assign_face_curve(curved_faces: SD.CurveMap, element: SD.Geometry, i: int) -> None:
    if isinstance(element, SD.Geometry2D):
        curve = element.GetCurve()
        if curve is not None:
            curve.curveID = i
            curved_faces[i] = curve


def _assign_elements(elements: NektarElements, meshgraph: SD.MeshGraphXml) -> None:
    """Assign elements to a MeshGraph object."""
    segments = meshgraph.GetAllSegGeoms()
    tris = meshgraph.GetAllTriGeoms()
    quads = meshgraph.GetAllQuadGeoms()
    curved_faces = meshgraph.GetCurvedFaces()
    tets = meshgraph.GetAllTetGeoms()
    prisms = meshgraph.GetAllPrismGeoms()
    pyrs = meshgraph.GetAllPyrGeoms()
    hexes = meshgraph.GetAllHexGeoms()
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
        _assign_face_curve(curved_faces, element, i)


def nektar_mesh(
    elements: NektarElements,
    mesh_dim: int,
    spatial_dim: int,
    write_movement: bool = True,
    periodic_interfaces: bool = True,
    compressed: bool = True,
) -> SD.MeshGraphXml | SD.MeshGraphXmlCompressed:
    """Create a Nektar++ MeshGraphXml from Nektar++ geometry objects.

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
    compressed
        Whether to return a ``MeshGraphXmlCompressed`` object or
        a plain ``MeshGraphXml`` object

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
    print("Assembling Nektar++ MeshGraph")
    meshgraph = (
        SD.MeshGraphXmlCompressed(mesh_dim, spatial_dim)
        if compressed
        else SD.MeshGraphXml(mesh_dim, spatial_dim)
    )
    print("Assigning verticse")
    _assign_points(elements, meshgraph)
    print("Assigning segments")
    _assign_segments(elements, meshgraph)
    print("Assigning faces")
    _assign_faces(elements, meshgraph)
    print("Assigning elements")
    _assign_elements(elements, meshgraph)

    composites = meshgraph.GetComposites()
    domains = meshgraph.GetDomain()
    movement = meshgraph.GetMovement()
    print("Assigning domains")
    i = -1
    j = 0
    for i, layer in enumerate(elements.layers()):
        comp_map = {k: comp for k, comp in enumerate(layer, j)}
        j += len(comp_map)
        for n, composite in comp_map.items():
            composites[n] = composite
        domain = nektar_composite_map(comp_map)
        domains[i] = domain
        if write_movement:
            movement.AddZone(SD.ZoneFixed(i, i, domain, 3))
    n = elements.num_layers()
    m = len(composites)
    print("Assigning interfaces between layers")
    near_faces: Iterator[tuple[int, SD.Composite]] = enumerate(elements.near_faces(), m)
    first_near = next(near_faces)
    near_faces = itertools.chain(near_faces, [first_near])
    far_faces = enumerate(elements.far_faces(), m + n)
    for i, ((j, near), (k, far)) in enumerate(zip(near_faces, far_faces)):
        composites[j] = near
        composites[k] = far
        if write_movement and (i != n - 1 or periodic_interfaces):
            near_interface = SD.Interface(2 * i, nektar_composite_map({j: near}))
            far_interface = SD.Interface(2 * i + 1, nektar_composite_map({k: far}))
            movement.AddInterface(f"Interface {i}", far_interface, near_interface)
    print("Assigning boundary composites")
    for i, bound in enumerate(elements.bounds(), m + 2 * n):
        composites[i] = bound

    return meshgraph


def write_nektar(
    mesh: Mesh,
    order: int,
    filename: str,
    spatial_dim: int = 3,
    write_movement: bool = True,
    periodic_interfaces: bool = True,
    compressed: bool = True,
) -> None:
    """Create a Nektar++ MeshGraph object and write it to the disk.

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
    compressed
        Whether to use the Nektar++ compressed XML format

    Group
    -----
    public nektar

    """
    mesh_dim = 3 if issubclass(mesh.reference_layer.element_type, Prism) else 2
    if spatial_dim < mesh_dim:
        raise ArgumentError(
            f"Spatial dimension ({spatial_dim}) must be at least equal to mesh "
            f"dimension ({mesh_dim})"
        )
    nek_elements = nektar_elements(mesh, order, spatial_dim)
    nek_mesh = nektar_mesh(
        nek_elements,
        mesh_dim,
        spatial_dim,
        write_movement,
        periodic_interfaces,
        compressed,
    )
    print(f"Writing mesh to {filename}")
    nek_mesh.Write(filename, True, SD.FieldMetaDataMap())
