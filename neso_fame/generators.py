"""Functions for generating full meshes from magnetic field data."""

from __future__ import annotations

import itertools
import operator
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from functools import reduce
from typing import Callable, Optional, TypeVar, cast
from warnings import warn

import numpy as np
import numpy.typing as npt
from hypnotoad import Mesh as HypnoMesh  # type: ignore
from hypnotoad import MeshRegion as HypnoMeshRegion  # type: ignore
from hypnotoad import Point2D

from neso_fame.coordinates import (
    CoordinateSystem,
    CoordMap,
    FrozenCoordSet,
    SliceCoord,
    SliceCoords,
)
from neso_fame.element_builder import ElementBuilder
from neso_fame.hypnotoad_interface import (
    equilibrium_trace,
    get_region_flux_surface_boundary_points,
    get_region_perpendicular_boundary_points,
)
from neso_fame.mesh import (
    FieldAlignedCurve,
    FieldAlignedPositions,
    FieldTrace,
    GenericMesh,
    MeshLayer,
    Prism,
    PrismMesh,
    PrismTypes,
    Quad,
    QuadMesh,
    control_points,
    straight_line_across_field,
    subdividable_field_aligned_positions,
)
from neso_fame.wall import (
    Connections,
    WallSegment,
    adjust_wall_resolution,
    find_external_points,
    get_all_rectangular_mesh_connections,
    get_immediate_rectangular_mesh_connections,
    periodic_pairwise,
    point_in_tokamak,
    wall_points_to_segments,
)

Connectivity = Sequence[tuple[int, int]]
FieldTracer = None  # Placeholder during refactoring


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


def field_aligned_2d(
    lower_dim_mesh: SliceCoords,
    field_line: FieldTrace,
    extrusion_limits: tuple[float, float] = (0.0, 1.0),
    n: int = 10,
    order: int = 3,
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
    order
        The order of accuracy with which to represent field-aligned
        (and other) edges.
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

    if connectivity is None:
        connectivity = _ordered_connectivity(num_nodes)

    def make_quad(node1: int, node2: int) -> Quad:
        shape = straight_line_across_field(
            lower_dim_mesh[node1], lower_dim_mesh[node2], order
        )
        north_weight = 0.0 if node1 in (0, num_nodes - 1) and conform_to_bounds else 1.0
        south_weight = 0.0 if node2 in (0, num_nodes - 1) and conform_to_bounds else 1.0
        return Quad(
            subdividable_field_aligned_positions(
                shape,
                dx3,
                field_line,
                np.linspace(north_weight, south_weight, order + 1),
                order,
                subdivisions,
            )
        )

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


def _constrain_to_plane(
    dx1: npt.NDArray,
    dx2: npt.NDArray,
    stencil: npt.NDArray,
    plain: tuple[float, float],
) -> tuple[npt.NDArray, npt.NDArray]:
    norm = plain[0] * plain[0] + plain[1] * plain[1]
    projection_factor = (dx1 * plain[0] + dx2 * plain[1]) / norm
    stencil2 = 1 - stencil
    return dx1 * stencil2 + stencil * projection_factor * plain[
        0
    ], dx2 * stencil2 + stencil * projection_factor * plain[1]


def _sort_nodes(
    lower_dim_mesh: SliceCoords, nodes: tuple[Index, Index, Index, Index]
) -> tuple[Index, Index, Index, Index]:
    """Return sorted nodes in the order top-left, top-right, bottom-left, bottom-right."""
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
    if x1[order[3]] < x1[order[2]]:
        tmp = order[3]
        order[3] = order[2]
        order[2] = tmp
    return nodes[order[0]], nodes[order[1]], nodes[order[2]], nodes[order[3]]


def field_aligned_3d(
    lower_dim_mesh: SliceCoords,
    field_line: FieldTrace,
    elements: Sequence[tuple[Index, Index, Index, Index]],
    extrusion_limits: tuple[float, float] = (0.0, 1.0),
    n: int = 10,
    order: int = 3,
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
    order
        The order of accuracy with which to represent field-aligned
        (and other) edges.
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

    element_nodes = [_sort_nodes(lower_dim_mesh, elem) for elem in elements]

    # Get the locations (north, south, east, west) of each quad in the hexes it builds
    face_locations: defaultdict[frozenset[Index], list[int]] = defaultdict(list)
    for node00, node01, node10, node11 in element_nodes:
        face_locations[frozenset({node00, node01})].append(0)
        face_locations[frozenset({node10, node11})].append(1)
        face_locations[frozenset({node01, node11})].append(2)
        face_locations[frozenset({node00, node10})].append(3)
    # Find the quads that are on a boundary
    boundary_faces: dict[frozenset[Index], int] = {
        pair: locs[0] for pair, locs in face_locations.items() if len(locs) == 1
    }
    print(boundary_faces)

    s = np.linspace(0.0, 1.0, order + 1)
    s1, s2 = np.meshgrid(s, s)
    w00 = (1 - s1) * (1 - s2)
    w10 = (1 - s1) * s2
    w01 = s1 * (1 - s2)
    w11 = s1 * s2

    def make_prism(
        node00: Index, node01: Index, node10: Index, node11: Index
    ) -> tuple[Prism, list[None | Quad]]:
        c00 = lower_dim_mesh[node00]
        c10 = lower_dim_mesh[node10]
        c01 = lower_dim_mesh[node01]
        c11 = lower_dim_mesh[node11]
        print(c00, c10, c01, c11)
        prism = Prism(
            PrismTypes.RECTANGULAR,
            subdividable_field_aligned_positions(
                SliceCoords(
                    c00.x1 * w00 + c10.x1 * w10 + c01.x1 * w01 + c11.x1 * w11,
                    c00.x2 * w00 + c10.x2 * w10 + c01.x2 * w01 + c11.x2 * w11,
                    c00.system,
                ),
                dx3,
                field_line,
                np.array(1.0),
                order,
                subdivisions,
            ),
        )
        # If any of the edges are on boundaries then constrain to them if necessary
        is_bound = [
            frozenset({node00, node01}) in boundary_faces,
            frozenset({node10, node11}) in boundary_faces,
            frozenset({node00, node10}) in boundary_faces,
            frozenset({node01, node11}) in boundary_faces,
        ]
        if conform_to_bounds and any(is_bound):
            x1s = np.expand_dims(prism.nodes.start_points.x1, -1)
            x2s = np.expand_dims(prism.nodes.start_points.x2, -1)
            # Make the FieldAlignedPositions object compute the
            # coordinates without accounting for the boundaries
            x1, x2, _ = prism.nodes.coords
            dx1 = x1 - x1s
            dx2 = x2 - x2s
            if is_bound[0]:
                dx1, dx2 = _constrain_to_plane(
                    dx1,
                    dx2,
                    np.expand_dims(1 - s2, -1),
                    (c01.x1 - c00.x1, c01.x2 - c00.x2),
                )
            if is_bound[1]:
                dx1, dx2 = _constrain_to_plane(
                    dx1, dx2, np.expand_dims(s2, -1), (c11.x1 - c10.x1, c11.x2 - c10.x2)
                )
            if is_bound[2]:
                dx1, dx2 = _constrain_to_plane(
                    dx1,
                    dx2,
                    np.expand_dims(1 - s1, -1),
                    (c10.x1 - c00.x1, c10.x2 - c00.x2),
                )
            if is_bound[3]:
                dx1, dx2 = _constrain_to_plane(
                    dx1, dx2, np.expand_dims(s1, -1), (c11.x1 - c01.x1, c11.x2 - c01.x2)
                )
            prism = Prism(
                PrismTypes.RECTANGULAR,
                FieldAlignedPositions(
                    prism.nodes.start_points,
                    prism.nodes.x3,
                    prism.nodes.trace,
                    prism.nodes.alignments,
                    prism.nodes.subdivision,
                    prism.nodes.num_divisions,
                    x1s + dx1,
                    x2s + dx2,
                    np.copy(prism.nodes._computed),
                ),
            )
        if any(is_bound):
            bounds = [side if bound else None for side, bound in zip(prism, is_bound)]
        else:
            bounds = [None] * 4
        return prism, bounds

    prisms = []
    boundaries: list[list[Quad]] = [[], [], [], []]
    for prism, bounds in itertools.starmap(make_prism, element_nodes):
        prisms.append(prism)
        for i, b in enumerate(bounds):
            if b is not None:
                boundaries[i].append(b)

    return GenericMesh(
        MeshLayer(
            prisms,
            list(map(frozenset, boundaries)),
            subdivisions=subdivisions,
        ),
        x3_mid,
    )


def _merge_connections(left: Connections, right: Connections) -> Connections:
    for k, v in right.items():
        if k in left:
            left[k] = cast(FrozenCoordSet, left[k] | v)
        else:
            left[k] = v
    return left


def _get_element_corners(
    x: npt.NDArray,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    return x[:-1, :-1], x[:-1, 1:], x[1:, :-1], x[1:, 1:]


def _element_corners(
    R_corners: npt.NDArray, Z_corners: npt.NDArray, system: CoordinateSystem
) -> Iterator[tuple[SliceCoord, SliceCoord, SliceCoord, SliceCoord]]:
    R = _get_element_corners(R_corners)
    Z = _get_element_corners(Z_corners)
    return zip(
        *map(
            operator.methodcaller("iter_points"),
            (SliceCoords(r, z, system) for r, z in zip(R, Z)),
        )
    )


def _element_aspect_ratio(
    R_left: npt.NDArray, R_right: npt.NDArray, Z_left: npt.NDArray, Z_right: npt.NDArray
) -> npt.NDArray:
    dx = np.sqrt((R_left - R_right) ** 2 + (Z_left - Z_right) ** 2)
    dy1 = np.sqrt((R_left[:-1] - R_left[1:]) ** 2 + (Z_left[:-1] - Z_left[1:]) ** 2)
    dy2 = np.sqrt((R_right[:-1] - R_right[1:]) ** 2 + (Z_right[:-1] - Z_right[1:]) ** 2)
    # return cast(npt.NDArray, np.abs((dx[:-1] + dx[1:]) / (dy1 + dy2)))
    return cast(npt.NDArray, np.abs((dy1 + dy2) / (dx[:-1] + dx[1:])))


Corners = tuple[SliceCoord, SliceCoord, SliceCoord, Optional[SliceCoord]]
CornersIterator = Iterator[Corners]


def _iter_merge_elements(
    R: npt.NDArray,
    Z: npt.NDArray,
    max_aspect_ratio: float,
    system: CoordinateSystem,
) -> tuple[int, CornersIterator]:
    """Iterate over elements, merging those that are too narrow.

    Always starts looking from index [0, 0]
    """

    def inner_func(
        count: int,
        prev_elements: CornersIterator,
        R_left: npt.NDArray,
        R_remainder: npt.NDArray,
        Z_left: npt.NDArray,
        Z_remainder: npt.NDArray,
    ) -> tuple[int, CornersIterator]:
        def iterate_column(merge_start: int | None) -> CornersIterator:
            # Handle elements that don't need to be merged
            for (R_sw, R_se, Z_sw, Z_se), (
                R_nw,
                R_ne,
                Z_nw,
                Z_ne,
            ) in itertools.pairwise(
                np.nditer(
                    (
                        R_left[:merge_start],
                        R_remainder[:merge_start, 0],
                        Z_left[:merge_start],
                        Z_remainder[:merge_start, 0],
                    )
                )
            ):
                yield (
                    SliceCoord(float(R_sw), float(Z_sw), system),
                    SliceCoord(float(R_se), float(Z_se), system),
                    SliceCoord(float(R_nw), float(Z_nw), system),
                    SliceCoord(float(R_ne), float(Z_ne), system),
                )
            # Return triangle
            if merge_start is not None:
                yield (
                    SliceCoord(
                        float(R_left[merge_start - 1]),
                        float(Z_left[merge_start - 1]),
                        system,
                    ),
                    SliceCoord(
                        float(R_remainder[merge_start - 1, 0]),
                        float(Z_remainder[merge_start - 1, 0]),
                        system,
                    ),
                    SliceCoord(
                        float(R_left[merge_start]),
                        float(Z_left[merge_start]),
                        system,
                    ),
                    None,
                )

        ratios = _element_aspect_ratio(
            R_left, R_remainder[:, 0], Z_left, Z_remainder[:, 0]
        )
        first_merging = int(np.argmax(ratios > max_aspect_ratio))
        # Deal with case where nothing needs to be merged or have reached the last column
        if (
            R_remainder.shape[1] == 1
            or first_merging == 0
            and not ratios[0] > max_aspect_ratio
        ):
            # TODO: Should I add further triangles just before the core?
            return count + 1, itertools.chain(prev_elements, iterate_column(None))
        # Never merge the first element, to make sure it stays
        # conformal across the boundary of the mesh region
        merge_start = max(1, first_merging)
        R_next = np.concatenate((R_remainder[:merge_start, 0], R_left[merge_start:]))
        Z_next = np.concatenate((Z_remainder[:merge_start, 0], Z_left[merge_start:]))
        return inner_func(
            count + 1,
            itertools.chain(prev_elements, iterate_column(merge_start)),
            R_next,
            R_remainder[:, 1:],
            Z_next,
            Z_remainder[:, 1:],
        )

    # Copy the first column to ensure we know exactly what its layout
    # will be in memory. Otherwise this can end up being inconsistent
    # for the first call compared to subsequent levels of recursion,
    # which construct this column from scratch.
    return inner_func(
        0, iter([]), np.copy(R[:, 0]), R[:, 1:], np.copy(Z[:, 0]), Z[:, 1:]
    )


def _flip_corners_horizontally(corners: Corners) -> Corners:
    if corners[3] is None:
        return corners[1], corners[0], corners[2], corners[3]
    return corners[1], corners[0], corners[3], corners[2]


def _flip_corners_vertically(corners: Corners) -> Corners:
    if corners[3] is None:
        return corners
    return corners[2], corners[3], corners[0], corners[1]


def _iter_element_corners(
    region: HypnoMeshRegion,
    max_aspect_ratio: float,
    system: CoordinateSystem,
) -> CornersIterator:
    """Iterate over the elements, returning the points at the corner for each one.

    This will merge elements radiating from the X-point if they are too oblong.
    """
    # Need to ensure the first coordinate is indexed away from the x-point
    R = region.Rxy.corners[::-1, :]
    Z = region.Zxy.corners[::-1, :]
    if (
        region.equilibriumRegion.name.endswith("core")
        and region.connections["inner"] is None
    ):
        half = R.shape[1] // 2
        start, left = _iter_merge_elements(
            R[:, : half + 1], Z[:, : half + 1], max_aspect_ratio, system
        )
        negative_end, right = _iter_merge_elements(
            np.flip(R[:, half:], 1),
            np.flip(Z[:, half:], 1),
            max_aspect_ratio,
            system,
        )
        end: int | None = None if negative_end == 0 else -negative_end
        return itertools.chain(
            left,
            _element_corners(R[:, start:end], Z[:, start:end], system),
            map(_flip_corners_horizontally, right),
        )
    else:
        return _element_corners(R, Z, system)


def _find_internal_neighbours(
    outermost: FrozenCoordSet[SliceCoord],
    external_points: FrozenCoordSet[SliceCoord],
    connections: Connections,
) -> Iterator[FrozenCoordSet[SliceCoord]]:
    yield outermost
    candidates = FrozenCoordSet(
        itertools.chain.from_iterable(connections[p] for p in outermost)
    )
    new_external = cast(FrozenCoordSet, external_points | outermost)
    new_outermost = cast(FrozenCoordSet, candidates - new_external)
    yield from _find_internal_neighbours(new_outermost, new_external, connections)


def _handle_edge_nodes(
    hypnotoad_poloidal_mesh: HypnoMesh,
    wall_points: Iterable[Point2D],
    restrict_to_vessel: bool,
    in_tokamak_test: Callable[[SliceCoord, Sequence[WallSegment]], bool],
    alignment_steps: int,
    system: CoordinateSystem,
) -> tuple[Callable[[Corners], bool], CoordMap[SliceCoord, float]]:
    """Work out which nodes fall outside the vessle and degree of field-alignment."""
    initial_outermost_nodes = FrozenCoordSet(
        itertools.chain.from_iterable(
            itertools.chain.from_iterable(
                get_region_flux_surface_boundary_points(region, system)[1:]
                + get_region_perpendicular_boundary_points(region, system)
                for region in hypnotoad_poloidal_mesh.regions.values()
            )
        )
    )
    wall = wall_points_to_segments(wall_points)
    if restrict_to_vessel:
        connections = reduce(
            _merge_connections,
            (
                get_all_rectangular_mesh_connections(
                    SliceCoords(
                        region.Rxy.corners,
                        region.Zxy.corners,
                        system,
                    )
                )
                for region in hypnotoad_poloidal_mesh.regions.values()
            ),
        )
        external_nodes, outermost_nodes = find_external_points(
            initial_outermost_nodes, connections, wall, in_tokamak_test
        )

        def corners_within_vessel(
            corners: tuple[SliceCoord, SliceCoord, SliceCoord, Optional[SliceCoord]],
        ) -> bool:
            return FrozenCoordSet(c for c in corners if c is not None).isdisjoint(
                external_nodes
            )

    else:

        def corners_within_vessel(
            corners: tuple[SliceCoord, SliceCoord, SliceCoord, Optional[SliceCoord]],
        ) -> bool:
            return True

        external_nodes = FrozenCoordSet()
        outermost_nodes = initial_outermost_nodes

    steps = np.flip(np.linspace(0.0, 1.0, alignment_steps + 1, endpoint=False))
    connections2 = reduce(
        _merge_connections,
        (
            get_immediate_rectangular_mesh_connections(
                SliceCoords(
                    region.Rxy.corners,
                    region.Zxy.corners,
                    system,
                )
            )
            for region in hypnotoad_poloidal_mesh.regions.values()
        ),
    )
    vertex_weights = CoordMap(
        dict(
            itertools.chain.from_iterable(
                zip(points, itertools.repeat(w))
                for w, points in zip(
                    steps,
                    _find_internal_neighbours(
                        outermost_nodes, external_nodes, connections2
                    ),
                )
            )
        )
    )
    return corners_within_vessel, vertex_weights


def _average_poloidal_spacing(hypnotoad_poloidal_mesh: HypnoMesh) -> float:
    def outermost_distances(region: HypnoMeshRegion) -> npt.NDArray:
        if region.connections["outer"] is None:
            R = region.Rxy.corners
            Z = region.Zxy.corners
            dR = R[-1, 1:] - R[-1, :-1]
            dZ = Z[-1, 1:] - Z[-1, :-1]
            return cast(npt.NDArray, np.sqrt(dR * dR + dZ * dZ))
        return np.array([])

    return float(
        np.mean(
            np.concatenate(
                [
                    np.ravel(outermost_distances(r))
                    for r in hypnotoad_poloidal_mesh.regions.values()
                ]
            )
        )
    )


def _merge_prisms(p1: Prism, p2: Prism) -> Prism:
    """Combine two triangular prisms into a hexahedron."""
    if len(p1.sides) != 3:
        raise ValueError("First element is not a triangular prism")
    if len(p2.sides) != 3:
        raise ValueError("Second element is not a triangular prism")
    common_face = set(p1.sides) & set(p2.sides)
    n = len(common_face)
    if n == 0:
        raise ValueError("Prisms do not share a face on which to join")
    if n > 1:
        raise ValueError("Prisms share more than one face; unclear how to join")
    join_on = next(iter(common_face))
    north, potential_east = (face for face in p1.sides if face != join_on)
    north_points = FrozenCoordSet(north.shape([0.0, 1.0]).iter_points())
    potential_east_points = FrozenCoordSet(
        potential_east.shape([0.0, 1.0]).iter_points()
    )
    q2_1, q2_2 = (face for face in p2.sides if face != join_on)
    if len(FrozenCoordSet(q2_1.shape([0.0, 1.0]).iter_points()) & north_points) == 0:
        south = q2_1
        potential_west = q2_2
    else:
        south = q2_2
        potential_west = q2_1
    # Choose east and west segments to ensure a positive Jacobian
    vertex0 = next(iter(north_points - potential_east_points))
    vertex1 = next(iter(potential_east_points - north_points))
    vertex3 = next(
        iter(
            FrozenCoordSet(potential_west.shape([0.0, 1.0]).iter_points())
            - north_points
        )
    )
    jacobian = (vertex1.x2 - vertex0.x2) * (vertex3.x1 - vertex0.x1) - (
        vertex1.x1 - vertex0.x1
    ) * (vertex3.x2 - vertex0.x2)
    if jacobian > 0:
        east = potential_east
        west = potential_west
    else:
        east = potential_west
        west = potential_east
    return Prism((north, south, east, west))


def _validate_wall_elements(
    boundary_faces: frozenset[Quad],
    elements: Sequence[Prism],
    quad_to_elements: Callable[[Quad], list[Prism]],
    validate: Callable[[Prism], bool],
) -> tuple[list[Prism], frozenset[Quad]]:
    """Return the elements with any self-intersections removed from boundaries.

    This is first attempted by combining an element an adjacent
    element containing the face being intersected. This only works if
    both elements are triangular prisms. If they are not, or if the
    new element also has a negative Jacobian, then the element is
    replaced with one with first-order faces. The routine assumes that

    Parameters
    ----------
    boundary_faces
        Quads in the external boundary of the mesh.
    elements
        All the elements being checked, plus adacent ones with which
        they may be combined.
    quad_to_elements
        A function that maps between a quad and thes elements which
        have it as a face.
    validate
        Function to check whether a given prism has a positive Jacobian
        (i.e., is not self-intersecting)

    Returns
    -------
    A list of prisms with any self-intersecting ones replaced. Also
    returns a new set of boundary quads with some of them potentially
    changed to prevent self-intersecting elements.

    """
    # FIXME: My refactor of how I represent meshes will mean I pretty much need to rewrite this.
    # TODO: Should I validate internal elements too? If I flatten one
    # then that could end up makign a further element invalid, which
    # sounds unpleasant to have to deal with...
    # FIXME: Hashing elements and faces like this can end up hashing
    # SliceCoord objects. This means there is no room for floating
    # point differences. So far that has not been a problem, but it
    # might become one.
    new_elements = set(elements)
    new_faces = set(boundary_faces)
    for prism in elements:
        # If element not in new_elements, it has already been
        # processed. If it is already valid there is no need to do
        # anything.
        if prism not in new_elements or validate(prism):
            continue
        # Try merging with adjacent triangles (which haven't already
        # been merged with another element, which would remove them
        # from new_elements)

        # FIXME: Will need to change how I map faces to elements; just use pairs of points?
        merge_candidates = frozenset(
            item
            for item in itertools.chain.from_iterable(
                (
                    (_merge_prisms(prism, p), p)
                    for p in quad_to_elements(q)
                    if len(p.sides) == 3 and p in new_elements and p != prism
                )
                for q in prism.sides
            )
            if validate(item[0])
        )
        # If that works, swap it for `prism` in `new_elements`
        if len(merge_candidates) != 0:
            # Will need to replace this element
            new_elements.remove(prism)
            assert len(merge_candidates) == 1  # Doesn't make sense otherwise
            new_hex, old_prism = next(iter(merge_candidates))
            new_elements.remove(old_prism)
            new_elements.add(new_hex)
        else:
            # Otherwise, convert the sides of the prism to be flat (in the poloidal plane)
            face_map = {f: f.make_flat_quad() for f in prism}
            # FIXME: This isn't detecting elements on the main plasma mesh...
            adjacent_elements = frozenset(
                itertools.chain.from_iterable(
                    (element for element in quad_to_elements(q) if element != prism)
                    for q in face_map
                )
            )
            # Don't fix elements that would require you to modify a
            # hex, as that would probably result in the hex becoming
            # self-intersecting.
            if any(len(element.sides) == 4 for element in adjacent_elements):
                warn("Can not fix negative Jacobian in prism without modifying a hex")
                continue
            # Will need to replace this element
            new_elements.remove(prism)
            new_elements.add(Prism(tuple(face_map.values())))
            # Replace the boundary faces for this prism with the flattened ones
            new_faces -= {old for old in face_map if old in boundary_faces}
            new_faces |= {new for old, new in face_map.items() if old in boundary_faces}
            # Update surrounding prisms to use the flattened faces
            for p in adjacent_elements:
                # FIXME: Could this result in element with curved
                # sides that were previously found to be OK not being
                # modified properly?
                if p in new_elements:
                    new_elements.remove(p)
                    new_elements.add(Prism(tuple(face_map.get(q, q) for q in p.sides)))
    return list(new_elements), frozenset(new_faces)


def hypnotoad_mesh(
    hypnotoad_poloidal_mesh: HypnoMesh,
    extrusion_limits: tuple[float, float] = (0.0, 2 * np.pi),
    n: int = 10,
    spatial_interp_resolution: int = 11,
    subdivisions: int = 1,
    max_aspect_ratio: float = 100,
    mesh_to_core: bool = False,
    restrict_to_vessel: bool = False,
    mesh_to_wall: bool = False,
    min_distance_to_wall: float = 0.025,
    wall_resolution: Optional[float] = None,
    wall_angle_threshold: float = np.pi / 12,
    alignment_steps: int = 0,
    validator: Optional[Callable[[Prism], bool]] = None,
    system: CoordinateSystem = CoordinateSystem.CYLINDRICAL,
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
    max_aspect_ratio
        The maximum ratio to allow between the length of the perpendicular
        and field-aligned edges of an element. If an element exceeds this
        ratio, it will be merged with an adjacent one. Note that this
        algorithm only checks elements radiating away from an X-point and
        may miss a few in order to maintain a conformal mesh.
    mesh_to_core
        Whether to add extra prism elements to fill in the core of
        the tokamak
    restrict_to_vessel
        Whether to remove the elements whose edges pass outside the
        tokamak wall
    mesh_to_wall
        Whether to add extra prism and hex elements to fill the space
        between the edge of the field-aligned mesh and the tokamak
        wall. Requires `restrict_to_vesel` to be true.
    min_distance_to_wall
        The minimum distance to leave between the hypnotoad mesh and
        the wall of the tokamak. Only used if `mesh_to_wall` is true.
    wall_resolution
        If present, indicates that the resolution of the tokamak wall
        should be adjusted so that the edges of elements on the wall
        are approximately the specified fraction of the size of those
        at the outer edge of the hypnotoad mesh. If `None` then use the
        wall elements specified in the eqdsk data, which may be of
        widely varying sizes.
    wall_angle_threshold
        If adjusting the resolution of the tokamak wall, any vertices
        with an angle above this threshold will be preserved as sharp
        corners. Angles below it will be smoothed out.
    alignment_steps
        The number of steps to take between aligned and unaligned elements
        near the wall. I.e., 0 indicates that the change happens immediately
        between the hypnotoad-generated elements and the traingular
        elements. 1 indicates that there will be nodes in-between which are
        averaged between aligned and unaligned. Higher values indicate
        additional nodes with the weight between aligned and unaligned
        changing more gradually.
    validator
        Function that checks whether the geometry of an element is
        valid. Default values means all elements will be assumed valid. This
        argument should change depending on the format you want to write
        your mesh to, the order of the basis for the element shapes, etc.
    system
        The coordinate system to use. This normally should not be
        changed. However, if you want to export the poloidal cross-section
        of the mesh then it can be useful to set this to be Cartesian.

    Returns
    -------
    :obj:`~neso_fame.mesh.PrismMesh`
        A 3D field-aligned, non-conformal grid

    Group
    -----
    generator

    """
    if mesh_to_wall and not restrict_to_vessel:
        raise ValueError(
            "If mesh_to_wall is true then restrict_to_vessel must be true as well."
        )
    if not hasattr(next(iter(hypnotoad_poloidal_mesh.regions.values())), "Rxy"):
        hypnotoad_poloidal_mesh.calculateRZ()
    dx3 = (extrusion_limits[1] - extrusion_limits[0]) / n
    x3_mid = np.linspace(
        extrusion_limits[0] + 0.5 * dx3, extrusion_limits[1] - 0.5 * dx3, n
    )
    tracer = FieldTracer(
        equilibrium_trace(hypnotoad_poloidal_mesh.equilibrium, system),
        spatial_interp_resolution,
    )
    min_dist_squared = min_distance_to_wall * min_distance_to_wall

    outermost_weight = alignment_steps / (alignment_steps + 1)

    def whole_line_in_tokamak(start: SliceCoord, wall: Sequence[WallSegment]) -> bool:
        if (
            point_in_tokamak(start, wall)
            and min(seg.min_distance_squared(start) for seg in wall) >= min_dist_squared
        ):
            line = FieldAlignedCurve(tracer, start, dx3, start_weight=outermost_weight)
            return all(
                point_in_tokamak(p.to_slice_coord(), wall)
                for p in control_points(line, spatial_interp_resolution).iter_points()
            )
        return False

    eqdsk_wall = hypnotoad_poloidal_mesh.equilibrium.wall[:-1]
    corners_within_vessel, vertex_weights = _handle_edge_nodes(
        hypnotoad_poloidal_mesh,
        eqdsk_wall,
        restrict_to_vessel,
        whole_line_in_tokamak,
        alignment_steps,
        system,
    )
    factory = ElementBuilder(
        hypnotoad_poloidal_mesh, tracer, dx3, vertex_weights, system
    )

    main_elements = [
        factory.make_element(*corners)
        for corners in itertools.chain.from_iterable(
            _iter_element_corners(region, max_aspect_ratio, system)
            for region in hypnotoad_poloidal_mesh.regions.values()
        )
        if corners_within_vessel(corners)
    ]
    # Probably more efficient just to iterate over inner regions
    if mesh_to_core:
        core_elements = list(
            itertools.starmap(
                factory.make_prism_to_centre,
                periodic_pairwise(factory.innermost_vertices()),
            )
        )
        inner_bounds: frozenset[Quad] = frozenset()
    else:
        core_elements = []
        inner_bounds = frozenset(factory.innermost_quads())
    if mesh_to_wall:
        # FIXME: Not capturing the curves of the outermost hypnotoad quads now, for some reason.

        # FIXME: Assemble coordinate pairs and mapping between these pairs and the list of Coords defining the curve
        plasma_points = [tuple(p) for p in factory.outermost_vertices()]
        # FIXME: Assemble list of Coords (one for each wall segment) and also coordinate pairs?
        if wall_resolution is not None:
            target = _average_poloidal_spacing(hypnotoad_poloidal_mesh)
            wall: list[Point2D] = adjust_wall_resolution(
                eqdsk_wall,
                target * wall_resolution,
                angle_threshold=wall_angle_threshold,
                register_segment=factory.make_wall_quad_for_prism,
            )
        else:
            wall = eqdsk_wall
        wall_points = [tuple(p) for p in wall]
        # Should be fine to require exact equality when comparing wall coordinates
        wall_coord_pairs = frozenset(
            periodic_pairwise(SliceCoord(p[0], p[1], system) for p in wall_points)
        )
        n = len(wall_points)
        import meshpy.triangle as triangle  # type: ignore

        info = triangle.MeshInfo()
        info.set_points(wall_points + plasma_points)
        info.set_facets(
            list(periodic_pairwise(iter(range(n))))
            + list(periodic_pairwise(iter(range(n, n + len(plasma_points)))))
        )
        info.set_holes([tuple(hypnotoad_poloidal_mesh.equilibrium.o_point)])
        wall_mesh = triangle.build(
            info, allow_volume_steiner=True, allow_boundary_steiner=False
        )
        wall_mesh_points = np.array(wall_mesh.points)
        triangles = np.array(wall_mesh.elements)
        wall_mesh_coords = SliceCoords(
            wall_mesh_points[:, 0], wall_mesh_points[:, 1], system
        )
        initial: tuple[list[Prism], frozenset[Quad]] = ([], frozenset())
        # FIXME: Take coordinate pairs, check if either of them correspond to curves and use those or else just use the pair. If any of the pairs are from the wall, create a boundary item as well.
        initial_wall_elements, initial_outer_bounds = reduce(
            lambda left, right: (left[0] + [right[0]], left[1] | right[1]),
            (
                factory.make_outer_prism(
                    wall_mesh_coords[i],
                    wall_mesh_coords[j],
                    wall_mesh_coords[k],
                    wall_coord_pairs,
                )
                for i, j, k in triangles
            ),
            initial,
        )
        if validator is not None:
            wall_elements, outer_bounds = _validate_wall_elements(
                initial_outer_bounds,
                initial_wall_elements,
                factory.get_element_for_quad,
                validator,
            )
        else:
            wall_elements = initial_wall_elements
            outer_bounds = initial_outer_bounds
    else:
        wall_elements = []
        outer_bounds = frozenset(factory.outermost_quads())

    return GenericMesh(
        MeshLayer(
            core_elements + main_elements + wall_elements,
            [inner_bounds, outer_bounds],
            subdivisions=subdivisions,
        ),
        x3_mid,
    )
