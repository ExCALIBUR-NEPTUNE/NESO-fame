# WARNING: This script is coverd by GPL, due to dependency on hypnotoad!

from collections import deque
from collections.abc import Iterable
from functools import partial
import itertools
from typing import Any, overload, Callable, Tuple, NamedTuple, TypeVar

import numpy as np
import numpy.typing as npt
from scipy.interpolate import interp1d, lagrange

import NekPy.SpatialDomains._SpatialDomains as SD
import NekPy.LibUtilities._LibUtilities as LU
from mesh_builder import MeshBuilder


class PoloidalCoord2D(NamedTuple):
    Z: float


class PoloidalCoord3D(NamedTuple):
    R: float
    Z: float


class PoloidalCoords2D(NamedTuple):
    Z: npt.NDArray


class PoloidalCoords3D(NamedTuple):
    R: npt.NDArray
    Z: npt.NDArray


class Coords2D(NamedTuple):
    Z: npt.NDArray
    phi: npt.NDArray

    def offset(self, dphi: npt.ArrayLike) -> "Coords2D":
        return Coords2D(self.Z, self.phi + dphi)


class Coords3D(NamedTuple):
    R: npt.NDArray
    Z: npt.NDArray
    phi: npt.NDArray

    def offset(self, dphi: npt.ArrayLike) -> "Coords3D":
        return Coords3D(self.R, self.Z, self.phi + dphi)


PoloidalCoord = PoloidalCoord2D | PoloidalCoord3D
PoloidalCoords = PoloidalCoord2D | PoloidalCoord3D
Coords = Coords2D | Coords3D

FieldTrace2D = Callable[
    [PoloidalCoord2D, npt.ArrayLike], tuple[PoloidalCoords2D, npt.NDArray]
]
FieldTrace3D = Callable[
    [PoloidalCoord3D, npt.ArrayLike], tuple[PoloidalCoords3D, npt.NDArray]
]
FieldTrace = FieldTrace2D | FieldTrace3D

NormalisedFieldLine2D = Callable[[npt.ArrayLike], Coords2D]
NormalisedFieldLine3D = Callable[[npt.ArrayLike], Coords3D]
NormalisedFieldLine = NormalisedFieldLine2D | NormalisedFieldLine3D


def normalise_field_line(
    trace: FieldTrace3D,
    start: PoloidalCoord3D,
    xtor_min: float,
    xtor_max: float,
    resolution=10,
) -> NormalisedFieldLine3D:
    xtors = np.linspace(xtor_min, xtor_max, resolution)
    xpol, s = trace(start, xtors)
    y = np.stack([*xpol, xtors])
    order = "cubic" if len(s) > 2 else "linear"
    interp = interp1d((s - s[0]) / (s[-1] - s[0]), y, order)
    return lambda s: Coords3D(*interp(s))


def make_lagrange_interpolation(
    norm_line: NormalisedFieldLine3D, order=1
) -> tuple[Coords3D, NormalisedFieldLine3D]:
    control_points = np.linspace(0.0, 1.0, order + 1)
    coords = norm_line(control_points)
    interpolators = [lagrange(control_points, coord) for coord in coords]
    CoordType = type(coords)
    return coords, lambda s: CoordType(*(interp(s) for interp in interpolators))


CurveAndPoints = tuple[SD.Curve, SD.PointGeom, SD.PointGeom]


def make_layer_composites(
    elements: Iterable[SD.QuadGeom],
) -> tuple[SD.Composite, SD.Composite, SD.Composite]:
    return tuple(
        SD.Composite(elems)
        for elems in map(
            list, zip(*((elem, elem.GetEdge(0), elem.GetEdge(2)) for elem in elements))
        )
    )


def element_key(element: SD.QuadGeom) -> float:
    return element.GetVertex(0).GetCoordinates()[1]


def field_aligned_2d(
    poloidal_mesh: npt.NDArray[np.float64],
    field_line: FieldTrace3D,
    limits: Tuple[float, float] = (0.0, 1.0),
    n: int = 10,
    order: int = 1,
):
    """Generate a 2D mesh where element edges follow field
    lines. Start with a 1D mesh defined in the poloidal plane. Edges
    are then traced along the field lines both backwards and forwards
    in the toroidal direction to form a single layer of field-aligned
    elements. The field is assumed not to very in the toroidal
    direction, meaning this layer can be repeated. However, each layer
    will be non-conformal with the next.

    Parameters
    ----------
    poloidal_mesh
        Locations of nodes in the poloidal plane from which to project
        along field-lines. Should be ordered.
    field_line
        Function returning the poloidal coordinate of a field line.
        The first argument is the starting poloidal position and the
        second is the toroidal offset.
    limits
        The lower and upper limits of the domain in the toroidal
        direction.
    n
        Number of layers to generate in the toroidal direction
    order
        Order of the elements created. Must be at least 1 (linear).

    Returns
    -------
    MeshGraph
        A MeshGraph object containing the field-aligned, non-conformal
        grid
    """
    # Calculate toroidal positions for nodes in final mesh
    dphi = (limits[1] - limits[0]) / n
    phi_mid = np.linspace(limits[0] + 0.5*dphi, limits[1] - 0.5*dphi, n)
    print(phi_mid, dphi)
    spline_field_lines = map(
        lambda x: normalise_field_line(
            field_line, PoloidalCoord3D(x, 0.), -0.5 * dphi, 0.5 * dphi, min(10, 2 * order)
        ),
        poloidal_mesh,
    )
    control_points_and_lagrange = map(
        lambda line: make_lagrange_interpolation(line, order), spline_field_lines
    )

    # Need to filter out field lines that leave domain
    # Tag elements adjacent to boundaries
    # Work out max and min distances between near-boundary elements and the boundary
    # Add extra geometry elements to fill in boundaries

    builder = MeshBuilder(2, 2)

    # Does Numpy have sin/cos functions that absorb multiples of pi, to reduce floating point error?
    curves_start_end = (
        builder.make_curves_and_points(*c[0].offset(phi))
        for phi, c in itertools.product(phi_mid, control_points_and_lagrange)
    )

    tmp1, tmp2, tmp3, tmp4 = itertools.tee(curves_start_end, 4)
    curves = (t[0] for t in tmp1)
    starts = (t[1] for t in tmp2)
    ends = (t[2] for t in tmp3)
    termini = ((t[1], t[2]) for t in tmp4)

    horizontal_edges = (
        builder.make_edge(start, end, curve)
        for curve, (start, end) in zip(curves, termini)
    )
    left = (
        builder.make_edge(s1, s2)
        for s1, s2 in itertools.pairwise(starts)
    )
    right = (
        builder.make_edge(e1, e2)
        for e1, e2 in itertools.pairwise(ends)
    )

    elements = itertools.starmap(
        lambda left, right, top_bottom: builder.make_quad_element(left, right, *top_bottom), zip(left, right, itertools.pairwise(horizontal_edges))
    )
    elements_left_right = ((elem, elem.GetEdge(0), elem.GetEdge(2)) for elem in elements)
    layers_left_right = (layer for _, layer in itertools.groupby(elements_left_right, lambda e: e[0].GetVertex(0).GetCoordinates()[1]))
   
    composites_left_right = map(lambda l: map(builder.make_composite, map(list, zip(*l))), layers_left_right)
    zones_left_right_interfaces = (
        (builder.make_zone(builder.make_domain([main]), 2), builder.make_interface([l]), builder.make_interface([r]))
        for main, l, r in composites_left_right
    )

    deque(
        (
            builder.add_interface_pair(far, near, f"Join {i}")
            for i, ((_, _, far), (near, _, _)) in enumerate(itertools.pairwise(zones_left_right_interfaces))
        ),
        maxlen=0,
    )
    return builder.meshgraph


m = field_aligned_2d(
    np.linspace(0, 1, 5), lambda y, x: (PoloidalCoords3D(*np.broadcast_arrays(x, *y)[1:]), x), (0.0, 1.0), 4, 2
)
m.Write("test_geometry.xml", False, SD.FieldMetaDataMap())
