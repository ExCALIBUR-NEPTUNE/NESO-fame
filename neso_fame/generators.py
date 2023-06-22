from __future__ import annotations

from collections.abc import Iterator, Sequence
import itertools
from typing import Optional

import numpy as np

from .mesh import (
    Curve,
    Quad,
    FieldTrace,
    GenericMesh,
    MeshLayer,
    QuadMesh,
    normalise_field_line,
    SliceCoord,
    SliceCoords,
    Coords,
)

Connectivity = Sequence[tuple[int, int]]


def _ordered_connectivity(size: int) -> Connectivity:
    return list(itertools.pairwise(range(size)))


def _boundary_curve(start: SliceCoord, dx3: float) -> Curve:
    return Curve(
        lambda s: Coords(
            np.full_like(s, start.x1),
            np.full_like(s, start.x2),
            dx3 * (np.asarray(s) - 0.5),
            start.system,
        )
    )


def field_aligned_2d(
    lower_dim_mesh: SliceCoords,
    field_line: FieldTrace,
    extrusion_limits: tuple[float, float] = (0.0, 1.0),
    n: int = 10,
    spatial_interp_resolution: int = 11,
    connectivity: Optional[Connectivity] = None,
    boundaries: tuple[int, int] = (0, -1),
    subdivisions: int = 1,
    conform_to_bounds=True,
) -> QuadMesh:
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
    QuadMesh
        A 2D field-aligned, non-conformal grid

    """
    num_nodes = len(lower_dim_mesh)

    # Calculate x3 positions for nodes in final mesh
    dx3 = (extrusion_limits[1] - extrusion_limits[0]) / n
    x3_mid = np.linspace(
        extrusion_limits[0] + 0.5 * dx3, extrusion_limits[1] - 0.5 * dx3, n
    )

    curves = [
        _boundary_curve(coord, dx3)
        if (i == 0 or i == num_nodes - 1) and conform_to_bounds
        else Curve(
            normalise_field_line(
                field_line,
                coord,
                -0.5 * dx3,
                0.5 * dx3,
                spatial_interp_resolution * subdivisions,
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

    quads = [Quad(curves[i], curves[j], None, field_line) for i, j in connectivity]
    
    return GenericMesh(MeshLayer(quads, [frozenset({quads[boundaries[0]].north}), frozenset({quads[boundaries[1]].south})], subdivisions=subdivisions), x3_mid)
