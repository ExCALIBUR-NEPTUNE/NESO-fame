from __future__ import annotations

from collections.abc import Iterator, Sequence
import itertools
from typing import Optional

import numpy as np

from .edge_filter import classify_node_position, Connectivity, NodeStatus
from .mesh import (
    C,
    Curve,
    FieldTrace,
    make_lagrange_interpolation,
    Mesh,
    MeshLayer,
    normalise_field_line,
    Quad,
    SliceCoords,
)


def ordered_connectivity(size: int) -> Connectivity:
    return [[1]] + [[i - 1, i + 1] for i in range(1, size - 1)] + [[size - 2]]


def all_connections(
    connectivity: Connectivity, node_status: Sequence[NodeStatus]
) -> Iterator[tuple[int, int]]:
    for i, (status, connections) in enumerate(zip(node_status, connectivity)):
        if status == NodeStatus.EXTERNAL:
            continue
        for j in connections:
            assert i != j
            if node_status[j] != NodeStatus.EXTERNAL:
                yield i, j


def field_aligned_2d(
    lower_dim_mesh: SliceCoords[C],
    field_line: FieldTrace[C],
    extrusion_limits: tuple[float, float] = (0.0, 1.0),
    bounds=(0.0, 1.0),
    n: int = 10,
    order: int = 1,
    connectivity: Optional[Connectivity] = None,
    skin_nodes: Optional[Sequence[bool]] = None,
) -> Mesh:
    """Generate a 2D mesh where element edges follow field
    lines. Start with a 1D mesh defined in the poloidal plane. Edges
    are then traced along the field lines both backwards and forwards
    in the toroidal direction to form a single layer of field-aligned
    elements. The field is assumed not to very in the toroidal
    direction, meaning this layer can be repeated. However, each layer
    will be non-conformal with the next.

    Parameters
    ----------
    lower_dim_mesh
        Locations of nodes in the x1-x2 plane, from which to project
        along field-lines. Unless providing `connectivity`, must be
        ordered.
    field_line
        Function returning the poloidal coordinate of a field line.
        The first argument is the starting poloidal position and the
        second is the toroidal offset.
    extrusion_limits
        The lower and upper limits of the domain in the toroidal
        direction.
    n
        Number of layers to generate in the x3 direction
    order
        Order of the elements created. Must be at least 1 (linear).
    connectivity
        Defines which points are connected to each other in the mesh.
        Item at index `n` is a sequence of the indices for all the
        other points connected to `n`. If not provided, assume points
        are connected in an ordered line.
    skin_nodes
        Sequence indicating whether the point at each index `n` is on
        the outer surface of the domain. If not provided, the first and
        last nodes will be treated as being on the outer surface.

    Returns
    -------
    MeshGraph
        A MeshGraph object containing the field-aligned, non-conformal
        grid
    """
    # Ensure poloidal mesh is actually 1-D (required to keep output 2-D)
    flattened_mesh = SliceCoords(
        lower_dim_mesh.x1, np.array(0.0), lower_dim_mesh.system
    )

    # Calculate x3 positions for nodes in final mesh
    dx3 = (extrusion_limits[1] - extrusion_limits[0]) / n
    x3_mid = np.linspace(
        extrusion_limits[0] + 0.5 * dx3, extrusion_limits[1] - 0.5 * dx3, n
    )
    curves = [
        Curve(
            normalise_field_line(
                field_line, coord, -0.5 * dx3, 0.5 * dx3, max(11, 2 * order + 1)
            )
        )
        for coord in flattened_mesh.iter_points()
    ]
    lagrange_curves = [make_lagrange_interpolation(line, order) for line in curves]

    num_nodes = len(flattened_mesh)
    if connectivity is None:
        connectivity = ordered_connectivity(num_nodes)
    if skin_nodes is None:
        skin_nodes = [True] + list(itertools.repeat(False, num_nodes - 2)) + [True]

    node_status, updated_skin_nodes = classify_node_position(
        lagrange_curves, bounds, connectivity, skin_nodes
    )

    # Need to filter out field lines that leave domain
    # Tag elements adjacent to boundaries
    # Work out max and min distances between near-boundary elements and the boundary
    # Add extra geometry elements to fill in boundaries

    quads_grouped_by_curve = itertools.groupby(
        sorted(
            (
                (
                    i,
                    Quad.from_unordered_curves(curves[i], curves[j], None, field_line),
                )
                for i, j in all_connections(connectivity, node_status)
            ),
            key=lambda q: q[0],
        ),
        lambda q: q[0],
    )
    adjacent_quads = itertools.chain.from_iterable(
        map(
            lambda g: itertools.permutations(g, 2),
            map(lambda g: g[1], quads_grouped_by_curve),
        )
    )
    # FIXME (minor): What would be the functional way to do this, without needing to mutate quad_map?
    quad_map: dict[Quad, dict[Quad, bool]] = {}
    for (_, q1), (_, q2) in adjacent_quads:
        if q1 in quad_map:
            quad_map[q1][q2] = False
        else:
            quad_map[q1] = {q2: False}

    return Mesh(MeshLayer(quad_map), x3_mid)
