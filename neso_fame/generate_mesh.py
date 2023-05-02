# WARNING: This script is coverd by GPL, due to dependency on hypnotoad!

from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
import itertools
from typing import (
    Callable,
    Tuple,
    TypeVar,
    Type,
    Generic,
)

import numpy as np
import numpy.typing as npt
from scipy.interpolate import interp1d, lagrange

import NekPy.SpatialDomains._SpatialDomains as SD
from mesh_builder import MeshBuilder


CoordTriple = tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]


def asarrays(coords: CoordTriple) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    return tuple(map(np.asarray, coords))


class CoordinateSystem(ABC):
    @abstractmethod
    @staticmethod
    def to_cartesian(
        x1: npt.ArrayLike, x2: npt.ArrayLike, x3: npt.ArrayLike
    ) -> CoordTriple:
        raise NotImplementedError("Must be overridden in subclass.")


class Cylindrical(CoordinateSystem):
    @staticmethod
    def to_cartesian(
        x1: npt.ArrayLike, x2: npt.ArrayLike, x3: npt.ArrayLike
    ) -> CoordTriple:
        return x1 * np.sin(x3), x1 * np.cos(x3), x2


class Cartesian(CoordinateSystem):
    @staticmethod
    def to_cartesian(
        x1: npt.ArrayLike, x2: npt.ArrayLike, x3: npt.ArrayLike
    ) -> CoordTriple:
        return x1, x2, x3


C = TypeVar("C", bound=CoordinateSystem)


@dataclass
class SliceCoord(Generic[C]):
    x1: float
    x2: float
    system: Type[C]

    def __iter__(self) -> Iterator[float]:
        yield self.x1
        yield self.x2


@dataclass
class SliceCoords(Generic[C]):
    x1: npt.NDArray
    x2: npt.NDArray
    system: Type[C]

    def iter_points(self) -> Iterator[SliceCoord]:
        for x1, x2 in zip(self.x1, self.x2):
            yield SliceCoord(x1, x2, self.system)

    def __iter__(self) -> Iterator[npt.NDArray]:
        yield self.x1
        yield self.x2


@dataclass
class Coords(Generic[C]):
    x1: npt.NDArray
    x2: npt.NDArray
    x3: npt.NDArray
    system: Type[C]

    def offset(self, dx3: npt.ArrayLike) -> "Coords[C]":
        return Coords(self.x1, self.x2, self.x3 + dx3, self.system)

    def to_cartesian(self) -> "Coords[Cartesian]":
        return Coords(
            *asarrays(self.system.to_cartesian(*self)),
            Cartesian,
        )

    def __iter__(self) -> Iterator[npt.NDArray]:
        yield self.x1
        yield self.x2
        yield self.x3


FieldTrace = Callable[
    [SliceCoord[C], npt.ArrayLike], tuple[SliceCoords[C], npt.NDArray]
]

NormalisedFieldLine = Callable[[npt.ArrayLike], Coords[C]]


def normalise_field_line(
    trace: FieldTrace[C],
    start: SliceCoord[C],
    xtor_min: float,
    xtor_max: float,
    resolution=10,
) -> NormalisedFieldLine[C]:
    x_extrusion = np.linspace(xtor_min, xtor_max, resolution)
    x_cross_section, s = trace(start, x_extrusion)
    coordinates = np.stack([*x_cross_section, x_extrusion])
    order = "cubic" if len(s) > 2 else "linear"
    interp = interp1d((s - s[0]) / (s[-1] - s[0]), coordinates, order)
    coord_system = start.system
    return lambda s: Coords(*asarrays(interp(s)), coord_system)


def make_lagrange_interpolation(
    norm_line: NormalisedFieldLine[C], order=1
) -> tuple[Coords[C], NormalisedFieldLine[C]]:
    control_points = np.linspace(0.0, 1.0, order + 1)
    coords = norm_line(control_points)
    interpolators = [lagrange(control_points, coord) for coord in coords]
    return coords, lambda s: Coords(*(interp(s) for interp in interpolators), coords.system)


CurveAndPoints = tuple[SD.Curve, SD.PointGeom, SD.PointGeom]


def field_aligned_2d(
    poloidal_mesh: SliceCoords[C],
    field_line: FieldTrace[C],
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
    phi_mid = np.linspace(limits[0] + 0.5 * dphi, limits[1] - 0.5 * dphi, n)
    print(phi_mid, dphi)
    spline_field_lines = map(
        lambda coord: normalise_field_line(
            field_line, coord, -0.5 * dphi, 0.5 * dphi, min(10, 2 * order)
        ),
        poloidal_mesh.iter_points(),
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
        builder.make_curves_and_points(*c[0].offset(phi).to_cartesian())
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
    left = (builder.make_edge(s1, s2) for s1, s2 in itertools.pairwise(starts))
    right = (builder.make_edge(e1, e2) for e1, e2 in itertools.pairwise(ends))

    elements = itertools.starmap(
        lambda left, right, top_bottom: builder.make_quad_element(
            left, right, *top_bottom
        ),
        zip(left, right, itertools.pairwise(horizontal_edges)),
    )
    elements_left_right = (
        (elem, elem.GetEdge(0), elem.GetEdge(2)) for elem in elements
    )
    layers_left_right = (
        layer
        for _, layer in itertools.groupby(
            elements_left_right, lambda e: e[0].GetVertex(0).GetCoordinates()[1]
        )
    )

    composites_left_right = map(
        lambda l: map(builder.make_composite, map(list, zip(*l))), layers_left_right
    )
    zones_left_right_interfaces = (
        (
            builder.make_zone(builder.make_domain([main]), 2),
            builder.make_interface([l]),
            builder.make_interface([r]),
        )
        for main, l, r in composites_left_right
    )

    deque(
        (
            builder.add_interface_pair(far, near, f"Join {i}")
            for i, ((_, _, far), (near, _, _)) in enumerate(
                itertools.pairwise(zones_left_right_interfaces)
            )
        ),
        maxlen=0,
    )
    return builder.meshgraph


def straight_field(angle=0.0) -> FieldTrace[C]:
    """Returns a field trace corresponding to straight field lines
    slanted at `angle` above the direction of extrusion into the first
    coordinate direction."""
    def trace(start: SliceCoord[C], perpendicular_coord: npt.ArrayLike) -> tuple[SliceCoords[C], npt.NDArray]:
        """Returns a trace for a straight field line."""
        x1 = start.x1 + perpendicular_coord * np.tan(angle)
        x2 = np.asarray(start.x2)
        x3 = np.asarray(perpendicular_coord)
        return SliceCoords(x1, x2, start.system), x3
    return trace


m = field_aligned_2d(
    SliceCoords(np.linspace(0, 1, 5), np.zeros(5), Cartesian),
    straight_field(),
    (0.0, 1.0),
    4,
    2,
)
m.Write("test_geometry.xml", False, SD.FieldMetaDataMap())
