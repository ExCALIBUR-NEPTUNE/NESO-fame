"""Functions for generating full meshes from magnetic field data.

"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Optional

import numpy as np
import numpy.typing as npt

from .mesh import (
    Coords,
    Curve,
    FieldTrace,
    FieldTracer,
    GenericMesh,
    MeshLayer,
    NormalisedFieldLine,
    Quad,
    QuadMesh,
    SliceCoord,
    SliceCoords,
)

Connectivity = Sequence[tuple[int, int]]


def _ordered_connectivity(size: int) -> Connectivity:
    """Produces connectivity information representing a sequence of
    nodes connected to each other one after the other.

    """
    return list(itertools.pairwise(range(size)))


BOUNDARY_TRACER = FieldTracer(
    lambda start, x3: (SliceCoords(
        np.full_like(x3, start.x1),
        np.full_like(x3, start.x2),
        start.system), np.asarray(x3)), 2
)

def _boundary_curve(start: SliceCoord, dx3: float) -> Curve:
    """Produces a curve with constant x1 and x2 coordinates set by
    ``start`` and which goes from ``-x3/2`` to ``x3/2``.

    """
    return Curve(BOUNDARY_TRACER, start, (-0.5 * dx3, 0.5 * dx3))



def _line_from_points(north: SliceCoord, south: SliceCoord) -> NormalisedFieldLine:
    """Creates a function representing a straight line between the two
    specified points.
    """

    def _line(s: npt.ArrayLike) -> Coords:
        s = np.asarray(s)
        return Coords(
            north.x1 + (south.x1 - north.x1) * s,
            north.x2 + (south.x2 - north.x2) * s,
            np.asarray(0.),
            north.system,
        )

    return _line



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

    curves = [
        _boundary_curve(coord, dx3)
        if i in (0, num_nodes - 1) and conform_to_bounds
        else Curve(tracer, coord, (-0.5 * dx3, 0.5 * dx3),
        )
        for i, coord in enumerate(lower_dim_mesh.iter_points())
    ]

    if connectivity is None:
        connectivity = _ordered_connectivity(num_nodes)

    
    quads = [Quad(_line_from_points(lower_dim_mesh[i], lower_dim_mesh[j]), tracer, (-0.5*dx3, 0.5*dx3)) for i, j in connectivity]

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
