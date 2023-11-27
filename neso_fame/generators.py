"""Functions for generating full meshes from magnetic field data."""

from __future__ import annotations

import itertools
import operator
from collections import defaultdict
from collections.abc import Sequence
from functools import cache
from typing import Optional, TypeVar, cast

import numpy as np
import numpy.typing as npt
from hypnotoad import Mesh as HypnoMesh  # type: ignore
from hypnotoad import MeshRegion as HypnoMeshRegion  # type: ignore

from neso_fame.hypnotoad_interface import (
    equilibrium_trace,
    flux_surface_edge,
    get_mesh_boundaries,
    perpendicular_edge,
)

from .mesh import (
    CoordinateSystem,
    FieldTrace,
    FieldTracer,
    GenericMesh,
    PrismMesh,
    MeshLayer,
    Prism,
    Quad,
    QuadAlignment,
    QuadMesh,
    SliceCoord,
    SliceCoords,
    StraightLineAcrossField,
)

Connectivity = Sequence[tuple[int, int]]


def _ordered_connectivity(size: int) -> Connectivity:
    """Produce connectivity for a line of nodes.

    Produces connectivity information representing a sequence of
    nodes connected to each other one after the other.

    """
    return list(itertools.pairwise(range(size)))


BoundType = bool | tuple[float, float]


def _is_fixed(bound: BoundType) -> bool:
    if isinstance(bound, bool):
        return bound
    return False


def _is_planar(bound: BoundType) -> bool:
    return isinstance(bound, tuple)


def _is_bound(bound: BoundType) -> bool:
    if isinstance(bound, tuple):
        return True
    return bound


def _get_reference(
    shape: StraightLineAcrossField, north_fixed: bool, south_fixed: bool
) -> SliceCoord:
    if north_fixed:
        return shape(0.0).to_coord()
    elif south_fixed:
        return shape(1.0).to_coord()
    else:
        return SliceCoord(0.0, 0.0, shape.north.system)


def _get_vec(
    north_bound: BoundType,
    south_bound: BoundType,
    north_planar: bool,
    south_planar: bool,
) -> tuple[Optional[tuple[float, float]], Optional[float]]:
    if north_planar:
        vec = cast(tuple[float, float], north_bound)
    elif south_planar:
        vec = cast(tuple[float, float], south_bound)
    else:
        return None, None
    normed_vec = vec[0] * vec[0] + vec[1] * vec[1]
    return vec, normed_vec


def _get_factors(
    pos_on_shape: float,
    north_fixed: bool,
    south_fixed: bool,
    north_planar: bool,
    south_planar: bool,
) -> tuple[float, float]:
    if south_fixed:
        factor_fixed = 1 - pos_on_shape
    elif north_fixed:
        factor_fixed = pos_on_shape
    else:
        factor_fixed = 1.0
    if south_planar and north_planar:
        factor_planar = 0.0
    elif north_planar:
        factor_planar = pos_on_shape
    elif south_planar:
        factor_planar = 1 - pos_on_shape
    else:
        factor_planar = 1.0
    return factor_fixed * factor_fixed, factor_planar


def _constrain_to_plain(
    field: FieldTrace,
    shape: StraightLineAcrossField,
    north_bound: BoundType,
    south_bound: BoundType,
) -> FieldTrace:
    north_planar = _is_planar(north_bound)
    south_planar = _is_planar(south_bound)
    vec, normed_vec = _get_vec(north_bound, south_bound, north_planar, south_planar)

    assert vec is not None
    assert normed_vec is not None
    dx = (shape.south.x1 - shape.north.x1, shape.south.x2 - shape.north.x2)

    def get_position_on_shape(start: SliceCoord) -> float:
        if abs(dx[0]) > abs(dx[1]):
            return (start.x1 - shape.north.x1) / (dx[0])
        else:
            return (start.x2 - shape.north.x2) / (dx[1])

    def trace(
        start: SliceCoord, x3: npt.ArrayLike, start_weight: float = 0.0
    ) -> tuple[SliceCoords, npt.NDArray]:
        pos_on_shape = get_position_on_shape(start)
        factor_planar = (
            0.0
            if south_planar and north_planar
            else pos_on_shape
            if north_planar
            else 1 - pos_on_shape
            if south_planar
            else 1.0
        )
        x3_array = np.asarray(x3)
        initial, _ = field(start, x3, start_weight)
        projection_factor = (
            (initial.x1 - start.x1) * vec[0] + (initial.x2 - start.x2) * vec[1]
        ) / normed_vec
        position = SliceCoords(
            initial.x1 * factor_planar
            + (1 - factor_planar) * (start.x1 + projection_factor * vec[0]),
            initial.x2 * factor_planar
            + (1 - factor_planar) * (start.x2 + projection_factor * vec[1]),
            initial.system,
        )
        # FIXME: This assumes the trace is linear; how can I
        # estimate it for nonlinear ones?
        distance = np.sign(x3) * np.sqrt(
            (position.x1 - start.x1) ** 2
            + (position.x2 - start.x2) ** 2
            + x3_array * x3_array
        )
        return position, distance

    return trace


def field_aligned_2d(
    lower_dim_mesh: SliceCoords,
    field_line: FieldTrace,
    extrusion_limits: tuple[float, float] = (0.0, 1.0),
    n: int = 10,
    spatial_interp_resolution: int = 11,
    connectivity: Optional[Connectivity] = None,
    boundaries: tuple[int, int] = (0, -1),
    subdivisions: int = 1,
    conform_to_bounds: bool = True,
) -> QuadMesh:
    """Generate a 2D mesh.

    Element edges follow field lines. Start with a 1D mesh defined in
    the poloidal plane. Edges are then traced along the field lines
    both backwards and forwards in the toroidal direction to form a
    single layer of field-aligned elements. The field is assumed not
    to vary in the toroidal direction, meaning this layer can be
    repeated. However, each layer will be non-conformal with the next.

    Parameters
    ----------
    lower_dim_mesh
        Locations of nodes in the x1-x2 plane, from which to project
        along field-lines. Unless providing `connectivity`, must be
        ordered.
    field_line
        A callable which takes a `SliceCoord` defining a position on
        the x3=0 plane and an array-like object with x3
        coordinates. It should return a 2-tuple. The first element is
        the locations found by tracing the magnetic field line
        beginning at the position of the first argument until reaching
        the x3 locations described in the second argument. The second
        element is the distance traversed along the field line.
    extrusion_limits
        The lower and upper limits of the domain in the x3-direction.
    n
        Number of layers to generate in the x3 direction
    spatial_interp_resolution
        Number of points used to interpolate distances along the field
        line.
    connectivity
        Defines which points are connected to each other in the
        mesh. Consists of pairs of integers indicating the indices of
        two points which are connected by an edge. If not provided,
        assume points are connected in an ordered line.
    boundaries
        Indices of the quads (in the connectivity sequence) that make
        up the north and south boundary, respectively
    subdivisions
        Depth of cells in x3-direction in each layer.
    conform_to_bounds
        If True, make the first and last curves straight lines, so that
        there are regular edges to the domain.

    Returns
    -------
    :obj:`~neso_fame.mesh.QuadMesh`
        A 2D field-aligned, non-conformal grid

    Group
    -----
    generator

    """
    num_nodes = len(lower_dim_mesh)

    # Calculate x3 positions for nodes in final mesh
    dx3 = (extrusion_limits[1] - extrusion_limits[0]) / n
    x3_mid = np.linspace(
        extrusion_limits[0] + 0.5 * dx3, extrusion_limits[1] - 0.5 * dx3, n
    )
    tracer = FieldTracer(field_line, spatial_interp_resolution)

    if connectivity is None:
        connectivity = _ordered_connectivity(num_nodes)

    def make_quad(node1: int, node2: int) -> Quad:
        shape = StraightLineAcrossField(lower_dim_mesh[node1], lower_dim_mesh[node2])
        north_bound = node1 in (0, num_nodes - 1) and conform_to_bounds
        south_bound = node2 in (0, num_nodes - 1) and conform_to_bounds
        alignment = (
            QuadAlignment.NONALIGNED
            if north_bound and south_bound
            else QuadAlignment.NORTH
            if south_bound
            else QuadAlignment.SOUTH
            if north_bound
            else QuadAlignment.ALIGNED
        )
        return Quad(shape, tracer, dx3, aligned_edges=alignment)

    quads = list(itertools.starmap(make_quad, connectivity))

    return GenericMesh(
        MeshLayer(
            quads,
            [
                frozenset({quads[boundaries[0]].north}),
                frozenset({quads[boundaries[1]].south}),
            ],
            subdivisions=subdivisions,
        ),
        x3_mid,
    )


Index = TypeVar("Index", int, tuple[int, ...])
NodePair = tuple[Index, Index]


def _sort_node_pairs(
    lower_dim_mesh: SliceCoords, nodes: tuple[Index, Index, Index, Index]
) -> tuple[NodePair, NodePair, NodePair, NodePair]:
    """Return sorted pairs of nodes.

    Pairs are sorted so edges are in order north,
    south, east, west. Additionally, node indices will always be
    in ascending order within these pairs.

    """
    if isinstance(nodes[0], int):
        index = nodes
    else:
        index = tuple(zip(*nodes))
    x1: npt.NDArray = lower_dim_mesh.x1[index]
    x2: npt.NDArray = lower_dim_mesh.x2[index]
    order = np.argsort(x2)
    if x1[order[1]] < x1[order[0]]:
        tmp = order[1]
        order[1] = order[0]
        order[0] = tmp
    if x1[order[3]] > x1[order[2]]:
        tmp = order[3]
        order[3] = order[2]
        order[2] = tmp
    return (
        cast(tuple[Index, Index], tuple(sorted((nodes[order[2]], nodes[order[3]])))),
        cast(tuple[Index, Index], tuple(sorted((nodes[order[0]], nodes[order[1]])))),
        cast(tuple[Index, Index], tuple(sorted((nodes[order[1]], nodes[order[2]])))),
        cast(tuple[Index, Index], tuple(sorted((nodes[order[3]], nodes[order[0]])))),
    )


def field_aligned_3d(
    lower_dim_mesh: SliceCoords,
    field_line: FieldTrace,
    elements: Sequence[tuple[Index, Index, Index, Index]],
    extrusion_limits: tuple[float, float] = (0.0, 1.0),
    n: int = 10,
    spatial_interp_resolution: int = 11,
    subdivisions: int = 1,
    conform_to_bounds: bool = True,
) -> PrismMesh:
    """Generate a 3D mesh.

    Element edges follow field lines. Start with a 2D mesh defined in
    the poloidal plane. Edges are then traced along the field lines
    both backwards and forwards in the toroidal direction to form a
    single layer of field-aligned elements. The field is assumed not
    to vary in the toroidal direction, meaning this layer can be
    repeated. However, each layer will be non-conformal with the next.

    Parameters
    ----------
    lower_dim_mesh
        Locations of nodes in the x1-x2 plane, from which to project
        along field-lines. Unless providing `connectivity`, must be
        ordered.
    field_line
        A callable which takes a `SliceCoord` defining a position on
        the x3=0 plane and an array-like object with x3
        coordinates. It should return a 2-tuple. The first element is
        the locations found by tracing the magnetic field line
        beginning at the position of the first argument until reaching
        the x3 locations described in the second argument. The second
        element is the distance traversed along the field line.
    elements
        Defines groups of four points which together make up a quad in the
        2D mesh. Consists of four integers (or tuples of integers) indicating
        the indices of the points which make up the corners.
    extrusion_limits
        The lower and upper limits of the domain in the x3-direction.
    n
        Number of layers to generate in the x3 direction
    spatial_interp_resolution
        Number of points used to interpolate distances along the field
        line.
    subdivisions
        Depth of cells in x3-direction in each layer.
    conform_to_bounds
        If True, make the curves originating from boundary nodes
        straight lines, so that there are regular edges to the domain.

    Returns
    -------
    :obj:`~neso_fame.mesh.HexMesh`
        A 3D field-aligned, non-conformal grid

    Group
    -----
    generator

    """
    # Calculate x3 positions for nodes in final mesh
    dx3 = (extrusion_limits[1] - extrusion_limits[0]) / n
    x3_mid = np.linspace(
        extrusion_limits[0] + 0.5 * dx3, extrusion_limits[1] - 0.5 * dx3, n
    )
    tracer = FieldTracer(field_line, spatial_interp_resolution)

    element_node_pairs = [_sort_node_pairs(lower_dim_mesh, elem) for elem in elements]

    # Get the locations (north, south, east, west) of each quad in the hexes it builds
    face_locations: defaultdict[NodePair, list[int]] = defaultdict(list)
    for i, pair in itertools.chain.from_iterable(
        enumerate(x) for x in element_node_pairs
    ):
        face_locations[pair].append(i)
    # Find the quads that are on a boundary
    boundary_faces: dict[NodePair, int] = {
        pair: locs[0] for pair, locs in face_locations.items() if len(locs) == 1
    }
    faces_by_nodes: defaultdict[Index, list[NodePair]] = defaultdict(list)
    for j, k in boundary_faces:
        faces_by_nodes[j].append((j, k))
        faces_by_nodes[k].append((j, k))
    # Find the nodes that fall on a boundary

    def _bound_type(edges: list[NodePair]) -> BoundType:
        assert len(edges) == 2
        vecs = [
            (n1.x1 - n2.x1, n1.x2 - n2.x2)
            for n1, n2 in ((lower_dim_mesh[x[0]], lower_dim_mesh[x[1]]) for x in edges)
        ]
        if np.isclose(
            abs(vecs[0][0] * vecs[1][1]),
            abs(vecs[1][0] * vecs[0][1]),
            rtol=1e-8,
            atol=1e-8,
        ):
            return vecs[0]
        else:
            return True

    boundary_nodes = {i: _bound_type(edges) for i, edges in faces_by_nodes.items()}

    @cache
    def make_quad(node1: Index, node2: Index) -> Quad:
        shape = StraightLineAcrossField(lower_dim_mesh[node1], lower_dim_mesh[node2])
        if conform_to_bounds:
            north_bound = boundary_nodes.get(node1, False)
            south_bound = boundary_nodes.get(node2, False)
        else:
            north_bound = False
            south_bound = False
        alignment = (
            QuadAlignment.NONALIGNED
            if _is_fixed(north_bound) and _is_fixed(south_bound)
            else QuadAlignment.NORTH
            if _is_fixed(south_bound)
            else QuadAlignment.SOUTH
            if _is_fixed(north_bound)
            else QuadAlignment.ALIGNED
        )
        print(
            lower_dim_mesh[node1],
            lower_dim_mesh[node2],
            north_bound,
            south_bound,
            alignment,
        )
        local_tracer = (
            FieldTracer(
                _constrain_to_plain(field_line, shape, north_bound, south_bound),
                spatial_interp_resolution,
            )
            if _is_planar(north_bound) or _is_planar(south_bound)
            else tracer
        )
        return Quad(shape, local_tracer, dx3, aligned_edges=alignment)

    hexes = [
        Prism(tuple(itertools.starmap(make_quad, pairs)))
        for pairs in element_node_pairs
    ]

    return GenericMesh(
        MeshLayer(
            hexes,
            [
                frozenset(
                    itertools.starmap(
                        make_quad,
                        (pair for pair, loc in boundary_faces.items() if loc == i),
                    )
                )
                for i in range(4)
            ],
            subdivisions=subdivisions,
        ),
        x3_mid,
    )


def hypnotoad_mesh(
    hypnotoad_poloidal_mesh: HypnoMesh,
    extrusion_limits: tuple[float, float] = (0.0, 2 * np.pi),
    n: int = 10,
    spatial_interp_resolution: int = 11,
    subdivisions: int = 1,
) -> PrismMesh:
    """Generate a 3D mesh from hypnotoad-generage mesh.

    Edges are traced from the nodes making up the corners
    of elements. The tracing follows the magnetic field lines from the
    equilibrium backwards and forwards in the toroidal direction to
    form a single layer of field-aligned elements. The field is
    assumed not to vary in the toroidal direction, meaning this layer
    can be repeated. However, each layer will be non-conformal with
    the next.

    Parameters
    ----------
    hypnotoad_poloidal_mesh
        A mesh object created by hypnotoad from an equilibrium
        magnetic field.
    extrusion_limits
        The lower and upper limits of the domain in the toroidal
        direction (in radians).
    n
        Number of layers to generate in the x3 direction
    spatial_interp_resolution
        Number of points used to interpolate distances along the field
        line.
    subdivisions
        Depth of cells in x3-direction in each layer.


    Returns
    -------
    :obj:`~neso_fame.mesh.HexMesh`
        A 3D field-aligned, non-conformal grid

    Group
    -----
    generator

    """
    # FIXME: Perpendicular boundaries aren't going to work properly;
    # they need to be fixed in space. Think I will need to generate
    # them first, I guess with a different function. Then check if
    # already generated using a dict, I guess?
    #
    # Will freezing the edges be enough? Will I potentially need to
    # merge multiple elements to keep them from going down to 0?
    #
    # Or should I just fill in the spaces with additional elements
    # after the fact?
    if not hasattr(next(iter(hypnotoad_poloidal_mesh.regions.values())), "Rxy"):
        hypnotoad_poloidal_mesh.calculateRZ()
    dx3 = (extrusion_limits[1] - extrusion_limits[0]) / n
    x3_mid = np.linspace(
        extrusion_limits[0] + 0.5 * dx3, extrusion_limits[1] - 0.5 * dx3, n
    )
    tracer = FieldTracer(
        equilibrium_trace(hypnotoad_poloidal_mesh.equilibrium),
        spatial_interp_resolution,
    )

    @cache
    def perpendicular_quad(north: SliceCoord, south: SliceCoord) -> Quad:
        return Quad(
            perpendicular_edge(hypnotoad_poloidal_mesh.equilibrium, north, south),
            tracer,
            dx3,
        )

    @cache
    def flux_surface_quad(north: SliceCoord, south: SliceCoord) -> Quad:
        return Quad(
            flux_surface_edge(hypnotoad_poloidal_mesh.equilibrium, north, south),
            tracer,
            dx3,
        )

    def get_element_corners(
        x: npt.NDArray,
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
        return x[:-1, :-1], x[:-1, 1:], x[1:, :-1], x[1:, 1:]

    def element_corners(
        region: HypnoMeshRegion,
    ) -> tuple[SliceCoords, SliceCoords, SliceCoords, SliceCoords]:
        R = get_element_corners(region.Rxy.corners)
        Z = get_element_corners(region.Zxy.corners)
        return (
            SliceCoords(R[0], Z[0], CoordinateSystem.CYLINDRICAL),
            SliceCoords(R[1], Z[1], CoordinateSystem.CYLINDRICAL),
            SliceCoords(R[2], Z[2], CoordinateSystem.CYLINDRICAL),
            SliceCoords(R[3], Z[3], CoordinateSystem.CYLINDRICAL),
        )

    def make_hex(
        sw: SliceCoord, se: SliceCoord, nw: SliceCoord, ne: SliceCoord
    ) -> Prism:
        return Prism(
            (
                flux_surface_quad(nw, ne),
                flux_surface_quad(sw, se),
                perpendicular_quad(se, ne),
                perpendicular_quad(sw, nw),
            )
        )

    hexes = [
        make_hex(*corners)
        for corners in itertools.chain.from_iterable(
            zip(*map(operator.methodcaller("iter_points"), element_corners(region)))
            for region in hypnotoad_poloidal_mesh.regions.values()
        )
    ]

    boundaries = get_mesh_boundaries(
        hypnotoad_poloidal_mesh, flux_surface_quad, perpendicular_quad
    )

    return GenericMesh(
        MeshLayer(
            hexes,
            boundaries,
            subdivisions=subdivisions,
        ),
        x3_mid,
    )


# Properties to test: boundaries are on constant psi (that should be);
# all quads on a boundary are connected (?); all points from which
# hexes are extruded present in HypnoMesh (and vice versa); x-points
# in mesh
