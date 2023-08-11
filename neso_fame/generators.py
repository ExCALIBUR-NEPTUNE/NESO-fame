"""Functions for generating full meshes from magnetic field data.

"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Optional

import numpy as np
import numpy.typing as npt

from .mesh import (
    CoordinateSystem,
    FieldAlignedCurve,
    FieldTrace,
    FieldTracer,
    GenericMesh,
    MeshLayer,
    Quad,
    QuadMesh,
    SliceCoord,
    SliceCoords,
    StraightLineAcrossField,
)

Connectivity = Sequence[tuple[int, int]]


def _ordered_connectivity(size: int) -> Connectivity:
    """Produces connectivity information representing a sequence of
    nodes connected to each other one after the other.

    """
    return list(itertools.pairwise(range(size)))


BOUNDARY_TRACER = FieldTracer(
    lambda start, x3: (
        SliceCoords(
            np.full_like(x3, start.x1), np.full_like(x3, start.x2), start.system
        ),
        np.asarray(x3),
    ),
    2,
)


def _boundary_curve(start: SliceCoord, dx3: float) -> FieldAlignedCurve:
    """Produces a curve with constant x1 and x2 coordinates set by
    ``start`` and which goes from ``-x3/2`` to ``x3/2``.

    """
    return FieldAlignedCurve(BOUNDARY_TRACER, start, dx3)


def _boundary_tracer(
    field: FieldTrace,
    shape: StraightLineAcrossField,
    north_bound: bool,
    south_bound: bool,
) -> FieldTrace:
    """Creates a field trace that describes a quad which may conform
    to its boundaries. Note that the distances will be approximate if
    the field is nonlinear.

    Parameters
    ----------
    field
        The field that is being traced
    shape
        The function describing the shape of the quad where it
        intersects with the x3=0 plane
    north_bound
        Whether the north edge of the quad conforms to the boundary
    south_bound
        Whether the south edge of the quad conforms to the boundary

    """
    if north_bound and south_bound:
        return lambda start, x3: (
            SliceCoords(np.asarray(start.x1), np.asarray(start.x2), start.system),
            np.asarray(x3),
        )

    if not north_bound and not south_bound:
        return field

    def get_position_on_shape(start: SliceCoord) -> float:
        if abs(shape.south.x1 - shape.north.x1) > abs(shape.south.x2 - shape.north.x2):
            return (start.x1 - shape.north.x1) / (shape.south.x1 - shape.north.x1)
        else:
            return (start.x2 - shape.north.x2) / (shape.south.x2 - shape.north.x2)

    def func(start: SliceCoord, x3: npt.ArrayLike) -> tuple[SliceCoords, npt.NDArray]:
        factor = get_position_on_shape(start)
        if north_bound:
            reference = shape(0.0).to_coord()
        else:
            factor = 1 - factor
            reference = shape(1.0).to_coord()
        factor *= factor
        position, distance = field(start, x3)
        x3_factor = start.x1 if start.system is CoordinateSystem.CYLINDRICAL else 1.0
        coord = SliceCoords(
            position.x1 * factor + reference.x1 * (1 - factor),
            position.x2 * factor + reference.x2 * (1 - factor),
            position.system,
        )
        dist = np.sign(distance) * np.sqrt(
            distance * distance * factor
            + (1 - factor) * x3_factor * x3_factor * np.asarray(x3) ** 2
        )
        return coord, dist

    return func


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
    elements. The field is assumed not to vary in the toroidal
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

    if connectivity is None:
        connectivity = _ordered_connectivity(num_nodes)

    def make_quad(node1: int, node2: int) -> Quad:
        shape = StraightLineAcrossField(lower_dim_mesh[node1], lower_dim_mesh[node2])
        north_bound = node1 in (0, num_nodes - 1) and conform_to_bounds
        south_bound = node2 in (0, num_nodes - 1) and conform_to_bounds
        local_tracer = (
            FieldTracer(
                _boundary_tracer(field_line, shape, north_bound, south_bound),
                spatial_interp_resolution,
            )
            if north_bound or south_bound
            else tracer
        )
        return Quad(shape, local_tracer, dx3)

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
