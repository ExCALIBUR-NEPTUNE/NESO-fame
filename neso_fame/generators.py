from __future__ import annotations

from collections.abc import Iterator, Sequence
import itertools
from typing import Optional

import numpy as np

from .mesh import (
    C,
    Coord,
    Curve,
    FieldTrace,
    Mesh,
    MeshLayer,
    NormalisedFieldLine,
    normalise_field_line,
    Quad,
    SliceCoord,
    SliceCoords,
    Coords,
)

Connectivity = Sequence[Sequence[int]]


def _ordered_connectivity(size: int) -> Connectivity:
    return [[1]] + [[i - 1, i + 1] for i in range(1, size - 1)] + [[size - 2]]


def _boundary_curve(start: SliceCoord[C], dx3: float) -> Curve[C]:
    return Curve(lambda s: Coords(np.full_like(s, start.x1), np.full_like(s, start.x2), dx3 * (np.asarray(s) - 0.5), start.system)) 


def field_aligned_2d(
    lower_dim_mesh: SliceCoords[C],
    field_line: FieldTrace[C],
    extrusion_limits: tuple[float, float] = (0.0, 1.0),
    n: int = 10,
    spatial_interp_resolution: int = 11,
    connectivity: Optional[Connectivity] = None,
    subdivisions: int = 1,
    conform_to_bounds = True,
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
    spatial_interp_resolution
        Number of points used to interpolate distances along the field
        line.
    connectivity
        Defines which points are connected to each other in the mesh.
        Item at index `n` is a sequence of the indices for all the
        other points connected to `n`. If not provided, assume points
        are connected in an ordered line.
    subdivisions
        Depth of cells in x3-direction in each layer.
    conform_to_bounds
        If True, make the first and last curves straight lines, so that
        there are regular edges to the domain.

    Returns
    -------
    MeshGraph
        A MeshGraph object containing the field-aligned, non-conformal
        grid
    """
    num_nodes = len(lower_dim_mesh)

    # Calculate x3 positions for nodes in final mesh
    dx3 = (extrusion_limits[1] - extrusion_limits[0]) / n
    x3_mid = np.linspace(
        extrusion_limits[0] + 0.5 * dx3, extrusion_limits[1] - 0.5 * dx3, n
    )
    
    curves = [_boundary_curve(coord, dx3) if (i == 0 or i == num_nodes - 1) and conform_to_bounds else
        Curve(
            normalise_field_line(
                field_line, coord, -0.5 * dx3, 0.5 * dx3, spatial_interp_resolution * subdivisions
            )
        )
        for i, coord in enumerate(lower_dim_mesh.iter_points())
    ]

    # FIXME: Pretty sure I could represent this more efficiently as
    # pairs of node positions. Approach used here was when I expected
    # to be filtering edges and needed to be able to look up adjoining
    # nodes efficiently.
    if connectivity is None:
        connectivity = _ordered_connectivity(num_nodes)

    quads_grouped_by_curve = (
        (
            Quad.from_unordered_curves(curves[i], curves[j], None, field_line)
            for j in connections
        )
        for i, connections in enumerate(connectivity)
    )
    adjacent_quads = itertools.chain.from_iterable(
        itertools.permutations(g, 2) for g in quads_grouped_by_curve
    )
    # FIXME (minor): What would be the functional way to do this, without needing to mutate quad_map?
    quad_map: dict[Quad, dict[Quad, bool]] = {}
    for q1, q2 in adjacent_quads:
        if q1 in quad_map:
            quad_map[q1][q2] = False
        else:
            quad_map[q1] = {q2: False}

    return Mesh(MeshLayer(quad_map, subdivisions=subdivisions), x3_mid)
