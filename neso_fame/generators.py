"""Functions for generating full meshes from magnetic field data."""

from __future__ import annotations

import itertools
import operator
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Sequence
from functools import cache, reduce
from typing import Callable, Optional, TypeVar, cast

import numpy as np
import numpy.typing as npt
from hypnotoad import Mesh as HypnoMesh  # type: ignore
from hypnotoad import MeshRegion, Point2D
from hypnotoad import MeshRegion as HypnoMeshRegion  # type: ignore
from hypnotoad.cases.tokamak import TokamakEquilibrium

from neso_fame.coordinates import (
    CoordinateSystem,
    CoordMap,
    CoordSet,
    FrozenCoordSet,
    SliceCoord,
    SliceCoords,
)
from neso_fame.hypnotoad_interface import (
    connect_to_o_point,
    equilibrium_trace,
    flux_surface_edge,
    get_region_flux_surface_boundary_indices,
    get_region_perpendicular_boundary_indices,
    iterate_points,
)
from neso_fame.mesh import (
    AcrossFieldCurve,
    FieldAlignedPositions,
    FieldTrace,
    GenericMesh,
    MeshLayer,
    Prism,
    PrismMesh,
    PrismTypes,
    Quad,
    QuadMesh,
    field_aligned_positions,
    straight_line_across_field,
    subdividable_field_aligned_positions,
)
from neso_fame.vertex_ring import VertexRing
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


def _make_3d_element_builder(
    lower_dim_mesh: SliceCoords,
    boundary_faces: dict[frozenset[Index], int],
    field_line: FieldTrace,
    dx3: float,
    order: int = 3,
    subdivisions: int = 1,
    conform_to_bounds: bool = True,
) -> Callable[[Index, Index, Index, Index], tuple[Prism, list[None | Quad]]]:
    """Make a function that can build prisms from indices on the lower-dim mesh."""
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

    return make_prism


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

    make_prism = _make_3d_element_builder(
        lower_dim_mesh,
        {pair: locs[0] for pair, locs in face_locations.items() if len(locs) == 1},
        field_line,
        dx3,
        order,
        subdivisions,
        conform_to_bounds,
    )

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


def _iterate_prisms(nodes: FieldAlignedPositions, order: int) -> Iterator[Prism]:
    n1, n2 = nodes.poloidal_shape
    return (
        Prism(
            PrismTypes.RECTANGULAR,
            nodes[i * order : (i + 1) * order + 1, j * order : (j + 1) * order + 1],
        )
        for i, j in itertools.product(
            range((n1 - 1) // order), range((n2 - 1) // order)
        )
    )


def _element_aspect_ratio(
    left_nodes: SliceCoords, right_nodes: SliceCoords
) -> npt.NDArray:
    dx = np.sqrt(
        (left_nodes.x1 - right_nodes.x1) ** 2 + (left_nodes.x2 - right_nodes.x2) ** 2
    )
    dy1 = np.sqrt(
        (left_nodes.x1[:-1] - left_nodes.x1[1:]) ** 2
        + (left_nodes.x2[:-1] - left_nodes.x2[1:]) ** 2
    )
    dy2 = np.sqrt(
        (right_nodes.x1[:-1] - right_nodes.x1[1:]) ** 2
        + (right_nodes.x2[:-1] - right_nodes.x2[1:]) ** 2
    )
    # return cast(npt.NDArray, np.abs((dx[:-1] + dx[1:]) / (dy1 + dy2)))
    return cast(npt.NDArray, np.abs((dy1 + dy2) / (dx[:-1] + dx[1:])))


Corners = tuple[SliceCoord, SliceCoord, SliceCoord, Optional[SliceCoord]]
CornersIterator = Iterator[Corners]


def _iter_merge_elements(
    nodes: FieldAlignedPositions,
    order: int,
    max_aspect_ratio: float,
) -> tuple[int, Iterator[Prism]]:
    """Iterate over elements, merging those that are too narrow.

    Always starts looking from index [0, 0]
    """
    # TODO check this is the right shape for broadcasting
    weights = np.linspace(0.0, 1.0, order + 1)
    one_minus_weights = 1 - weights

    def inner_func(
        count: int,
        prev_merge_start: int,
        prev_elements: Iterator[Prism],
    ) -> tuple[int, Iterator[Prism]]:
        def iterate_column(merge_start: int | None) -> Iterator[Prism]:
            # Handle elements that don't need to be merged
            for i in range(0, prev_merge_start - order, order):
                yield Prism(
                    PrismTypes.RECTANGULAR,
                    nodes[i : i + order + 1, count * order : (count + 1) * order + 1],
                )
            # FIXME: Can this handle the case where we merge elements at the very end?
            if prev_merge_start > 0 and prev_merge_start != merge_start:
                narrow_element_points = nodes[
                    prev_merge_start : prev_merge_start + count + 1,
                    count * order : (count + 1) * order + 1,
                ]
                wide_element_points = nodes[
                    prev_merge_start : prev_merge_start + count + 1,
                    : (count + 1) * order + 1 : count + 1,
                ]
                # FIXME: Linear interpolation is a bit of a hack, as
                # it won't ensure interior points stay on the same
                # flux surfaces.
                starts = SliceCoords(
                    narrow_element_points.start_points.x1 * one_minus_weights
                    + wide_element_points.start_points.x1 * weights,
                    narrow_element_points.start_points.x2 * one_minus_weights
                    + wide_element_points.start_points.x2 * weights,
                    nodes.start_points.system,
                )
                yield Prism(
                    PrismTypes.RECTANGULAR,
                    field_aligned_positions(
                        starts,
                        nodes.x3[-1] - nodes.x3[1],
                        nodes.trace,
                        narrow_element_points.alignments * one_minus_weights
                        + wide_element_points.alignments * weights,
                        len(nodes.x3) - 1,
                    ),
                )
            # Handle elements that are merged into adjacent ones
            for i in range(
                prev_merge_start + order,
                nodes.poloidal_shape[0] if merge_start is None else merge_start,
                order,
            ):
                yield Prism(
                    PrismTypes.RECTANGULAR,
                    nodes[i : i + order + 1, : (count + 1) * order + 1 : count + 1],
                )
            # If this column reach a point where its elements become
            # too narrow then it will be merged too. Return the
            # triangle that will start that merge.
            if merge_start == prev_merge_start:
                assert merge_start is not None
                yield Prism(
                    PrismTypes.TRIANGULAR,
                    nodes[
                        np.arange(merge_start, merge_start + order + 1).reshape(
                            (order + 1, 1)
                        ),
                        np.arange(count * order, 0, -count).reshape((order + 1, 1))
                        + np.arange(order + 1),
                    ],
                )
                pass
            elif merge_start is not None:
                # FIXME: Doesn't account for possibility of prev_merge_start == merge_start. In that case I think we'll need to create a new array.
                yield Prism(
                    PrismTypes.TRIANGULAR,
                    nodes[
                        merge_start : merge_start + order + 1,
                        : (count + 1) * order + 1 : count + 1,
                    ],
                )

        # FIXME: Not sure I'm quite getting the right number of elements from each here
        column_starts = nodes.start_points.get[:prev_merge_start:order, count * order]
        leftmost = nodes.start_points.get[prev_merge_start::order, 0]
        column_edge = SliceCoords(
            np.concatenate((column_starts.x1, leftmost.x1)),
            np.concatenate((column_starts.x2, leftmost.x2)),
            column_starts.system,
        )
        ratios = _element_aspect_ratio(
            column_edge, nodes.start_points.get[::order, (count + 1) * order]
        )
        first_merging = int(np.argmax(ratios > max_aspect_ratio)) * order
        # Deal with case where nothing needs to be merged or have reached the last column
        if (
            nodes.poloidal_shape[1] == 1
            or first_merging == 0
            and not ratios[0] > max_aspect_ratio
        ):
            # TODO: Should I add further triangles just before the core?
            return count + 1, itertools.chain(prev_elements, iterate_column(None))
        # Never merge the first element, to make sure it stays
        # conformal across the boundary of the mesh region
        merge_start = max(order, first_merging)
        return inner_func(
            count + 1,
            merge_start,
            itertools.chain(prev_elements, iterate_column(merge_start)),
        )

    return inner_func(0, 0, iter([]))


def _element_iterator_factory(
    dx3: float,
    vertex_weights: CoordMap[SliceCoord, float],
    order: int,
    subdivisions: int,
    system: CoordinateSystem,
    eq: TokamakEquilibrium,
    max_aspect_ratio: float,
    corners_within_vessel: Callable[[Prism], bool],
    mesh_to_core: bool,
) -> Callable[[MeshRegion], tuple[Iterator[Prism], Iterator[Quad]]]:
    """Produce a function for iterating over elements and inner boundaries of a region.

    This function will merge elements radiating from the X-point if they are
    too oblong, if the region is in the core.

    """
    tracer = equilibrium_trace(eq, system)
    # FIXME: This is inefficient. Would be better to construct the
    # arrays directly and avoid the vertex_weights dict
    make_weights = np.vectorize(
        lambda R, Z: vertex_weights.get(SliceCoord(R, Z, system), 1.0)
    )

    def _iter_elements(
        region: MeshRegion,
    ) -> tuple[Iterator[Prism], Iterator[Quad]]:
        nodes = subdividable_field_aligned_positions(
            # Need to ensure the first coordinate is indexed away from the x-point
            SliceCoords(
                region.Rxy.corners[::-1, :], region.Zxy.corners[::-1, :], system
            ),
            dx3,
            tracer,
            make_weights(region.Rxy.corners, region.Zxy.corners),
            order,
            subdivisions,
        )
        centre_core_bound = get_region_flux_surface_boundary_indices(region)[0]
        nodes = nodes[::-1, :]
        if centre_core_bound is None:
            return filter(corners_within_vessel, _iterate_prisms(nodes, order)), iter(
                []
            )
        else:
            half = nodes.poloidal_shape[1] // 2
            start_count, left = _iter_merge_elements(
                nodes[:, : half + 1], order, max_aspect_ratio
            )
            start = start_count * order
            end_count, right = _iter_merge_elements(
                nodes[:, half:].flip(1),
                order,
                max_aspect_ratio,
            )
            end: int | None = None if end_count == 0 else -end_count * order
            main_elements = filter(
                corners_within_vessel,
                itertools.chain(
                    left,
                    _iterate_prisms(nodes[:, start:end], order),
                    map(operator.methodcaller("flip", 1), right),
                ),
            )
            core_bound = nodes[centre_core_bound]
            first_bound = (
                core_bound[: start + 1 : start_count + 1] if start > 0 else None
            )
            last_bound = core_bound[end :: end_count + 1] if end is not None else None
            main_bound = core_bound[start:end]
            if mesh_to_core:
                return itertools.chain(
                    main_elements,
                    _iter_prisms_to_core(first_bound, order, eq),
                    _iter_prisms_to_core(main_bound, order, eq),
                    _iter_prisms_to_core(last_bound, order, eq),
                ), iter([])
            else:
                return main_elements, (
                    Quad(points)
                    for points in itertools.chain(
                        [first_bound],
                        (
                            main_bound[i : i + order + 1]
                            for i in range(main_bound.poloidal_shape[0] // order)
                        ),
                        [last_bound],
                    )
                    if points is not None
                )

    return _iter_elements


def _iter_prisms_to_core(
    nodes: FieldAlignedPositions | None,
    order: int,
    eq: TokamakEquilibrium,
) -> Iterator[Prism]:
    """Iterate over prisms connecting the inner boundary to the core."""
    if nodes is None:
        return
    assert np.all(nodes.alignments == 1.0)
    assert len(nodes.poloidal_shape) == 1
    n = nodes.poloidal_shape[0] // order
    alignments = np.ones((order + 1, order + 1))
    alignments[-1, 0] = 0.0
    # Get edges of triangles, connecting to the O-point
    connectors = [
        connect_to_o_point(eq, start, order)
        for start in nodes.start_points.get[::order].iter_points()
    ]
    # Create prisms in groups of two, so that they can share memory more efficiently
    for i in range(n // 2):
        edge1 = nodes[2 * i * order : (2 * i + 1) * order + 1]
        edge2 = nodes[(2 * i + 1) * order : 2 * (i + 1) * order + 1]
        c1 = connectors[2 * i]
        c2 = connectors[2 * i + 1]
        c3 = connectors[2 * (i + 1)]
        R_starts = np.empty((order + 1, order + 1))
        Z_starts = np.empty((order + 1, order + 1))
        R_starts[0, :] = edge1.start_points.x1
        Z_starts[0, :] = edge1.start_points.x2
        R_starts[:, -1] = edge2.start_points.x1
        Z_starts[:, -1] = edge2.start_points.x2
        R_starts[:, 0] = c1.x1
        Z_starts[:, 0] = c1.x2
        R_starts[-1, :] = c3.x1
        R_starts[-1, :] = c3.x2
        np.fill_diagonal(np.fliplr(R_starts), c2.x1)
        np.fill_diagonal(np.fliplr(Z_starts), c2.x2)
        # Fill in the interior points of the triangles
        for j in range(1, order - 1):
            points1 = flux_surface_edge(eq, c1[j], c2[j], order - j)
            points2 = flux_surface_edge(eq, c2[j], c3[j], order - j)
            R_starts[j, 1 : order - j] = points1.x1[:-1]
            Z_starts[j, 1 : order - j] = points1.x2[:-1]
            R_starts[j + 1 : -1, -j - 1] = points2.x1[:-1]
            Z_starts[j + 1 : -1, -j - 1] = points2.x2[:-1]
        triangle_nodes = field_aligned_positions(
            SliceCoords(R_starts, Z_starts, nodes.start_points.system),
            nodes.x3[-1] - nodes.x3[1],
            nodes.trace,
            alignments,
            len(nodes.x3) - 1,
        )
        # Precompute coordinates on the shared edge. These are stored
        # along a diagonal and Numpy doesn't currently return
        # writeable views of diagonals. This prevents caching of the
        # coordinates calculated for the quad emenating from that
        # edge. Instead we compute them now when accessing elements in
        # a way that will return writeable views.
        for i in range(order + 1):
            triangle_nodes[i, -i - 1].coords
        yield Prism(PrismTypes.TRIANGULAR, triangle_nodes)
        # For second prism, we need to transpose and then flip horizontally and vertically
        yield Prism(PrismTypes.TRIANGULAR, triangle_nodes.transpose().flip(0).flip(1))

    # If there are an odd number of triangles, create the last one
    if n % 2 == 1:
        edge1 = nodes[-order - 1 :]
        c1 = connectors[-2]
        c2 = connectors[-1]
        R_starts = np.empty((order + 1, order + 1))
        Z_starts = np.empty((order + 1, order + 1))
        R_starts[0, :] = edge1.start_points.x1
        Z_starts[0, :] = edge1.start_points.x2
        R_starts[:, 0] = c1.x1
        Z_starts[:, 0] = c1.x2
        np.fill_diagonal(np.fliplr(R_starts), c2.x1)
        np.fill_diagonal(np.fliplr(Z_starts), c2.x2)
        # Fill in the interior points of the triangles
        for j in range(1, order - 1):
            points1 = flux_surface_edge(eq, c1[j], c2[j], order - j)
            R_starts[j, 1 : order - j] = points1.x1[:-1]
            Z_starts[j, 1 : order - j] = points1.x2[:-1]
        triangle_nodes = field_aligned_positions(
            SliceCoords(R_starts, Z_starts, nodes.start_points.system),
            nodes.x3[-1] - nodes.x3[1],
            nodes.trace,
            alignments,
            len(nodes.x3) - 1,
        )
        yield Prism(PrismTypes.TRIANGULAR, triangle_nodes)


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


# FIXME: I think I can actually handle each region individually for
# this, which would make things _a lot_ simpler and faster. Only
# slightly tricky thing I'll need to worry about is if elements on one
# side of the region boundary are inside and the other are not. I'll
# need a bit of extra logic to make sure the relevant edges on the
# boundary are marked as outermost.
def _handle_edge_nodes(
    hypnotoad_poloidal_mesh: HypnoMesh,
    wall_points: Iterable[Point2D],
    order: int,
    restrict_to_vessel: bool,
    in_tokamak_test: Callable[[SliceCoord, Sequence[WallSegment]], bool],
    alignment_steps: int,
    system: CoordinateSystem,
) -> tuple[
    Callable[[Prism], bool], CoordMap[SliceCoord, float], FrozenCoordSet[SliceCoord]
]:
    """Work out which nodes fall outside the vessle.

    Returns a function to test for whether an element is inside the
    vessel, a map indicating the required degree of field-alignment
    for each node, and a set of all the corners making up the outer
    edge of the remaining mesh.

    """
    # FIXME: Would probably be better to rework this to have it return
    # FieldAlignedPositions objects, one for each region. Compute the
    # weights for each one and use masked arrays or weights of -1 to
    # mark the external nodes. Also return lists of indicies for
    # accessing the internal and external edges.

    # FIXME: Should only get the corners, somehow
    initial_outermost_points = FrozenCoordSet(
        itertools.chain.from_iterable(
            itertools.chain.from_iterable(
                iterate_points(region, index, CoordinateSystem.CYLINDRICAL)
                for index in get_region_flux_surface_boundary_indices(region)[1:]
                + get_region_perpendicular_boundary_indices(region)
                if index is not None
            )
            for region in hypnotoad_poloidal_mesh.regions.values()
        )
    )
    wall = wall_points_to_segments(wall_points)
    if restrict_to_vessel:
        connections = reduce(
            _merge_connections,
            (
                get_all_rectangular_mesh_connections(
                    # Only want to check corners of elements
                    SliceCoords(
                        region.Rxy.corners[::order, ::order],
                        region.Zxy.corners[::order, ::order],
                        system,
                    )
                )
                for region in hypnotoad_poloidal_mesh.regions.values()
            ),
        )
        # Only check those outermost points which are also corners
        #
        # FIXME: It would be more efficient to store these as region
        # names and coordinates rather than as SliceCoord
        # objects. However, difficult to work out how to handle the
        # points that are in more than one region... I guess I could
        # work out the name of the adjacent region and indices for it
        # and store both somehow.
        external_corners, outermost_corners = find_external_points(
            cast(
                FrozenCoordSet,
                initial_outermost_points & FrozenCoordSet(connections.keys()),
            ),
            connections,
            wall,
            in_tokamak_test,
        )

        if order > 1:
            outermost_nodes = CoordSet(outermost_corners)
            for region in hypnotoad_poloidal_mesh.regions.values():
                # Create a mask indicating which nodes on the mesh are on an outermost edge
                outermost_array = np.full_like(region.Rxy.corners, False)
                # Mark the outermost corners
                outermost_array[::order, ::order] = np.vectorize(
                    lambda x1, x2: SliceCoord(x1, x2, system) in outermost_corners
                )(
                    region.Rxy.corners[::order, ::order],
                    region.Zxy.corners[::order, ::order],
                )
                n1, n2 = region.Rxy.corners.shape
                i1 = np.arange(n1).reshape((n1, 1))
                i2 = np.arange(n2).reshape((1, n2))
                lower_corners1 = i1 // order * order
                upper_corners1 = np.maximum(lower_corners1 + 1, n1)
                lower_corners2 = i2 // order * order
                upper_corners2 = np.maximum(lower_corners2 + 1, n2)
                # Outermost edge nodes are ones for which both of the corresponding corners have been marked outermost
                mask = (
                    outermost_array[lower_corners1, i2]
                    & outermost_array[upper_corners1, i2]
                ) | (
                    outermost_array[i1, lower_corners2]
                    & outermost_array[i1, upper_corners2]
                )
                for R, Z in np.nditer(
                    region.Rxy.corners[mask], region.Zxy.corners[mask]
                ):
                    outermost_nodes.add(
                        SliceCoord(cast(float, R), cast(float, Z), system)
                    )
            else:
                outermost_nodes = outermost_corners

        def corners_within_vessel(element: Prism) -> bool:
            return not any(
                point in external_corners for point in element.poloidal_corners()
            )

    else:

        def corners_within_vessel(
            _: Prism,
        ) -> bool:
            return True

        external_corners = FrozenCoordSet()
        outermost_nodes = initial_outermost_points

    # Work out the weights for the internal nodes too
    steps = np.linspace(0.0, 1.0, alignment_steps * order, endpoint=False)
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
    # FIXME: It would be more efficient to do this as arrays of
    # weights (since that's what we'll need to produce eventually
    # anyway).
    vertex_weights = CoordMap(
        dict(
            itertools.chain.from_iterable(
                zip(points, itertools.repeat(w))
                for w, points in zip(
                    steps,
                    _find_internal_neighbours(
                        outermost_nodes, external_corners, connections2
                    ),
                )
            )
        )
    )
    return corners_within_vessel, vertex_weights, outermost_nodes


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


def _quad_interpolate(
    north: npt.NDArray, east: npt.NDArray, south: npt.NDArray, west: npt.NDArray
) -> npt.NDArray:
    s, t = _quad_control_points(len(north) - 1)
    return (
        (north - north[0] * (1 - s)) * t
        + (south - south[-1] * s) * (1 - t)
        + (east - east[-1] * t) * s
        + (west - west[0] * (1 - t)) * (1 - s)
    )


def _merge_prisms(p1: Prism, p2: Prism) -> Prism:
    """Combine two triangular prisms into a hexahedron."""
    if p1.shape != PrismTypes.TRIANGULAR:
        raise ValueError("First element is not a triangular prism")
    if p2.shape != PrismTypes.TRIANGULAR:
        raise ValueError("Second element is not a triangular prism")
    sides1 = {_quad_points(s): s for s in p1}
    sides2 = {_quad_points(s): s for s in p2}

    common_face = set(sides1) & set(sides2)
    n = len(common_face)
    if n == 0:
        raise ValueError("Prisms do not share a face on which to join")
    if n > 1:
        raise ValueError("Prisms share more than one face; unclear how to join")
    join_on = next(iter(common_face))
    (north_points, north), (potential_east_points, potential_east) = (
        item for item in sides1.items() if item[0] != join_on
    )
    (q2_1_points, q2_1), (q2_2_points, q2_2) = (
        item for item in sides2.items() if item[0] != join_on
    )
    if len(q2_1_points & north_points) == 0:
        south = q2_1
        potential_west = q2_2
        potential_west_points = q2_2_points
    else:
        south = q2_2
        potential_west = q2_1
        potential_west_points = q2_1_points
    # Choose east and west segments to ensure a positive Jacobian
    vertex0 = next(iter(north_points - potential_east_points))
    vertex1 = next(iter(potential_east_points - north_points))
    vertex3 = next(iter(potential_west_points - north_points))
    jacobian = (vertex1.x2 - vertex0.x2) * (vertex3.x1 - vertex0.x1) - (
        vertex1.x1 - vertex0.x1
    ) * (vertex3.x2 - vertex0.x2)
    if jacobian > 0:
        east = potential_east
        west = potential_west
    else:
        east = potential_west
        west = potential_east

    # Make sure all quads are going in proper directions in  the poloidal plane
    # TODO: See if I can combine this with checking the Jacobian, etc.
    north_nodes = north.nodes
    if north_nodes.start_points[-1].approx_eq(east.nodes.start_points[-1]):
        east_nodes = east.nodes
    else:
        east_nodes = east.nodes[::-1]
    if east_nodes.start_points[0].approx_eq(south.nodes.start_points[-1]):
        south_nodes = south.nodes
    else:
        south_nodes = south.nodes[::-1]
    if north_nodes.start_points[0].approx_eq(west.nodes.start_points[-1]):
        west_nodes = west.nodes
    else:
        west_nodes = west.nodes[::-1]

    R_starts = _quad_interpolate(
        north_nodes.start_points.x1,
        east_nodes.start_points.x1,
        south_nodes.start_points.x1,
        west_nodes.start_points.x1,
    )
    Z_starts = _quad_interpolate(
        north_nodes.start_points.x1,
        east_nodes.start_points.x1,
        south_nodes.start_points.x1,
        west_nodes.start_points.x1,
    )
    alignments = _quad_interpolate(
        north_nodes.alignments,
        east_nodes.alignments,
        south_nodes.alignments,
        west_nodes.alignments,
    )
    return Prism(
        PrismTypes.RECTANGULAR,
        field_aligned_positions(
            SliceCoords(R_starts, Z_starts, p1.nodes.start_points.system),
            p1.nodes.x3[-1] - p1.nodes.x3[1],
            p1.nodes.trace,
            alignments,
            len(p1.nodes.x3) - 1,
        ),
    )


def _quad_points(q: Quad) -> frozenset[SliceCoord]:
    return frozenset({q.nodes.start_points[0], q.nodes.start_points[-1]})


def _validate_wall_elements(
    boundary_faces: frozenset[Quad],
    elements: Sequence[Prism],
    sides_to_elements: dict[frozenset[SliceCoord], list[Prism]],
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
    sides_to_elements
        A dictionary mapping between a the pair of vertices defining an
        edge and the elements which have it as a face.
    validate
        Function to check whether a given prism has a positive Jacobian
        (i.e., is not self-intersecting)

    Returns
    -------
    A list of prisms with any self-intersecting ones replaced. Also
    returns a new set of boundary quads with some of them potentially
    changed to prevent self-intersecting elements.

    """
    # TODO: Should I refactor to validate internal elements too? If I
    # flatten one then that could end up making a further element
    # invalid, which sounds unpleasant to have to deal with...
    #
    # FIXME: Hashing elements and faces like this can end up hashing
    # SliceCoord objects. This means there is no room for floating
    # point differences. So far that has not been a problem, but it
    # might become one.
    new_elements = {frozenset(elem.poloidal_corners()): elem for elem in elements}
    new_faces = {_quad_points(face): face for face in boundary_faces}
    for prism in elements:
        # If element not in new_elements, it has already been
        # processed. If it is already valid there is no need to do
        # anything.
        corners = frozenset(prism.poloidal_corners())
        if corners not in new_elements or validate(prism):
            continue
        # Try merging with adjacent triangles (which haven't already
        # been merged with another element, which would remove them
        # from new_elements)
        merge_candidates = frozenset(
            item
            for item in itertools.chain.from_iterable(
                (
                    (_merge_prisms(prism, p), p)
                    for p in sides_to_elements[q]
                    if p in new_elements and p != prism
                )
                for q in map(_quad_points, prism)
            )
            if validate(item[0])
        )
        # If that works, swap it for `prism` in
        # `new_elements`. Otherwise, convert the sides of the prism to
        # be flat (in the poloidal plane)
        if len(merge_candidates) != 0:
            # Will need to replace this element
            assert len(merge_candidates) == 1  # Doesn't make sense otherwise
            del new_elements[corners]
            new_hex, old_prism = next(iter(merge_candidates))
            del new_elements[frozenset(old_prism.poloidal_corners())]
            new_elements[frozenset(new_hex.poloidal_corners())] = new_hex
        else:
            # Note: We don't need worry about faces between adjacent
            # elements no longer lining up. Curved faces will always
            # be on either the edge of the Tokamak vessel or the edge
            # of the plasma mesh. If the former, there will be no
            # adjecent element to worry about. If the latter, there is
            # nothing we can do about it right now without causing
            # further elements to become invalid, so they are ignored.
            # It is unlikely they'd actually be invalid anyway.
            flat_prism = prism.make_flat_faces()
            new_elements[frozenset(prism.poloidal_corners())] = flat_prism
            new_faces.update(
                {
                    points: flat_face
                    for face, flat_face in zip(prism, flat_prism)
                    if (points := _quad_points(face)) in boundary_faces
                }
            )
    return list(new_elements.values()), frozenset(new_faces.values())


@cache
def _quad_control_points(order: int) -> tuple[npt.NDArray, npt.NDArray]:
    x1, x2 = np.meshgrid(
        np.linspace(0.0, 1.0, order + 1),
        np.linspace(0.0, 1.0, order + 1),
        indexing="ij",
        sparse=True,
    )
    return x1, x2


@cache
def _triangle_control_points(order: int) -> tuple[npt.NDArray, npt.NDArray]:
    x1sq, x2 = _quad_control_points(order)
    x1 = np.empty(np.broadcast(x1sq, x2).shape)
    x1[:, :-1] = x1sq / (1 - x2[:, :-1])
    # Handle NaNs at top of triangle
    x1[0, -1] = 1
    x1[1:, -1] = 1.1
    x1_m = np.ma.masked_greater(x1, 1.0)
    return x1_m, np.ma.array(np.broadcast_to(x2, x1.shape), mask=x1_m.mask)


def _edges_to_prism(side1: Quad, side2: Quad) -> Prism:
    """Construct a prism from the 2 edges, with a straight line between the unconnected vertices."""
    # Order sides so first one is linear, plus ensure east and west sides start at south
    s1_1 = side1.nodes.start_points[0]
    s1_2 = side1.nodes.start_points[-1]
    s2_1 = side2.nodes.start_points[0]
    s2_2 = side2.nodes.start_points[-1]

    if s1_1.approx_eq(s2_1):
        west = side1.nodes[::-1]
        east = side2.nodes[::-1]
    elif s1_1.approx_eq(s2_2):
        west = side1.nodes
        east = side2.nodes[::-1]
    elif s1_2.approx_eq(s2_1):
        west = side1.nodes[::-1]
        east = side2.nodes
    elif s1_2.approx_eq(s2_2):
        west = side1.nodes
        east = side2.nodes
    else:
        raise RuntimeError("Sides of triangular prism do not share an edge.")

    n = east.order
    s, _ = _triangle_control_points(n)
    s2 = 1 - s
    real_x1 = east.start_points.x1 * s2 + west.start_points.x1 * s
    real_x2 = east.start_points.x2 * s2 + west.start_points.x2 * s
    alignments = east.alignments * s2 + west.alignments * s
    return Prism(
        PrismTypes.TRIANGULAR,
        field_aligned_positions(
            SliceCoords(real_x1, real_x2, side1.nodes.start_points.system),
            side1.nodes.x3[-1] - side1.nodes.x3[0],
            side1.nodes.trace,
            alignments,
            len(side1.nodes.x3) - 1,
            side1.nodes.subdivision,
            side1.nodes.num_divisions,
        ),
    )


def hypnotoad_mesh(
    hypnotoad_poloidal_mesh: HypnoMesh,
    extrusion_limits: tuple[float, float] = (0.0, 2 * np.pi),
    n: int = 10,
    order: int = 3,
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
    order
        The order of accuracy to use to describe curved elements. Element
        edges will be made up of `order + 1` points.
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
    # TODO: Probably can just check Jacobian of elements by
    # calculating it manually for each sub-quad/sub-hex now, right?
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
    min_dist_squared = min_distance_to_wall * min_distance_to_wall

    eqdsk_wall = hypnotoad_poloidal_mesh.equilibrium.wall[:-1]
    corners_within_vessel, vertex_weights, outermost_corners = _handle_edge_nodes(
        hypnotoad_poloidal_mesh,
        eqdsk_wall,
        order,
        restrict_to_vessel,
        lambda start, wall: point_in_tokamak(start, wall)
        and min(seg.min_distance_squared(start) for seg in wall) >= min_dist_squared,
        alignment_steps,
        system,
    )
    iter_elements = _element_iterator_factory(
        dx3,
        vertex_weights,
        order,
        subdivisions,
        system,
        hypnotoad_poloidal_mesh.equilibrium,
        max_aspect_ratio,
        corners_within_vessel,
        mesh_to_core,
    )

    main_elements_iter: Iterator[Prism]
    inner_bounds_iter: Iterator[Quad]
    main_elements_iter, inner_bounds_iter = map(
        itertools.chain.from_iterable,
        zip(*(iter_elements(r) for r in hypnotoad_poloidal_mesh.regions.values())),
    )
    main_elements = list(main_elements_iter)
    inner_bounds = frozenset(inner_bounds_iter)

    # Don't use a FrozenCoordSet, as these are very inefficient to
    # hash on. However, this might result in points not being exactly
    # the same. Ideally would use whatever value is stored in
    # outermost_corners.
    possible_plasma_edges = [
        (frozenset({p1, p2}), q)
        for q in itertools.chain.from_iterable(main_elements)
        if (p1 := q.nodes.start_points[0]) in outermost_corners
        and (p2 := q.nodes.start_points[-1]) in outermost_corners
    ]
    # Construct a mapping between end-points of outermost plasma mesh edges
    # and the corresponding quads.
    #
    # If there are any duplicate edges then that indicates they aren't
    # really outermost and should be dropped.
    edge_count = Counter(item[0] for item in possible_plasma_edges)
    plasma_edges = dict(
        edge for edge in possible_plasma_edges if edge_count[edge[0]] == 1
    )
    # Get the ordered list of vertices making up the outermost edge of the plasma mesh
    op = hypnotoad_poloidal_mesh.equilibrium.o_point
    o_point = SliceCoord(op.R, op.Z, system)
    ordered_outermost_vertices: VertexRing = reduce(
        lambda ring, item: ring.add_vertices(*item, o_point),
        plasma_edges,
        VertexRing([]),
    )
    # FIXME: Avoid creating a second of these
    tracer = equilibrium_trace(hypnotoad_poloidal_mesh.equilibrium, system)

    if mesh_to_wall:
        # FIXME: Not capturing the curves of the outermost hypnotoad quads now, for some reason.

        # FIXME: Assemble coordinate pairs and mapping between these pairs and the list of Coords defining the curve
        plasma_points = [tuple(p) for p in ordered_outermost_vertices]
        # FIXME: Assemble list of Coords (one for each wall segment) and also coordinate pairs?
        if wall_resolution is not None:
            target = _average_poloidal_spacing(hypnotoad_poloidal_mesh)
            wall_segments: list[AcrossFieldCurve] = list(
                adjust_wall_resolution(
                    eqdsk_wall,
                    target * wall_resolution,
                    order,
                    angle_threshold=wall_angle_threshold,
                    system=system,
                )
            )
            wall = [seg[0] for seg in wall_segments]
            wall_quads = {
                frozenset({seg[0], seg[-1]}): Quad(
                    subdividable_field_aligned_positions(
                        seg, dx3, tracer, np.array(0.0), order, subdivisions
                    )
                )
                for seg in wall_segments
            }
        else:
            wall = eqdsk_wall
            wall_quads = {
                frozenset({p1, p2}): Quad(
                    subdividable_field_aligned_positions(
                        straight_line_across_field(p1, p2, order),
                        dx3,
                        tracer,
                        np.array(0.0),
                        order,
                        subdivisions,
                    )
                )
                for p1, p2 in periodic_pairwise(wall)
            }
        wall_points = [tuple(p) for p in wall]
        # Should be fine to require exact equality when comparing wall coordinates
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

        def get_prism_edge(p1: SliceCoord, p2: SliceCoord) -> tuple[Quad, bool]:
            key = frozenset({p1, p2})
            if key in plasma_edges:
                return plasma_edges[key], False
            if key in wall_quads:
                return wall_quads[key], False
            return Quad(
                subdividable_field_aligned_positions(
                    straight_line_across_field(p1, p2, order),
                    dx3,
                    tracer,
                    np.array(0.0),
                    order,
                    subdivisions,
                )
            ), True

        def make_outer_prism(p1: SliceCoord, p2: SliceCoord, p3: SliceCoord) -> Prism:
            q1, q1_new = get_prism_edge(p1, p2)
            q2, q2_new = get_prism_edge(p2, p3)
            q3, q3_new = get_prism_edge(p3, p1)
            # Make sure any pre-existing quads representing the plasma
            # mesh or the wall are used, to preserve any curvature.
            if q1_new:
                return _edges_to_prism(q2, q3)
            elif q2_new:
                return _edges_to_prism(q1, q3)
            elif q3_new:
                return _edges_to_prism(q1, q2)
            else:
                raise RuntimeError(
                    "Can not construct prism when all sides are on the vessel wall or edge of the plasma mesh."
                )

        initial_wall_elements = {
            make_outer_prism(
                (p1 := wall_mesh_coords[i]),
                (p2 := wall_mesh_coords[j]),
                (p3 := wall_mesh_coords[k]),
            ): [frozenset({p1, p2}), frozenset({p2, p3}), frozenset({p3, p1})]
            for i, j, k in triangles
        }
        sides_to_elements: dict[frozenset[SliceCoord], list[Prism]] = {}
        for k, v in initial_wall_elements.items():
            for p in v:
                sides_to_elements.setdefault(p, []).append(k)

        initial_outer_bounds = frozenset(wall_quads.values())
        if validator is not None:
            wall_elements, outer_bounds = _validate_wall_elements(
                initial_outer_bounds,
                list(initial_wall_elements),
                sides_to_elements,
                validator,
            )
        else:
            wall_elements = list(initial_wall_elements)
            outer_bounds = initial_outer_bounds
    else:
        wall_elements = []
        outer_bounds = frozenset(plasma_edges.values())

    return GenericMesh(
        MeshLayer(
            main_elements + wall_elements,
            [inner_bounds, outer_bounds],
            subdivisions=subdivisions,
        ),
        x3_mid,
    )
