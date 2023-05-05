# WARNING: This script is coverd by GPL, due to dependency on hypnotoad!

from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterator, Sequence, Iterable
from dataclasses import dataclass
from enum import Enum
import itertools
from typing import cast, Any, Callable, Optional, TypeVar, Type, Generic, Literal

import numpy as np
import numpy.typing as npt
from scipy.interpolate import interp1d, lagrange

import NekPy.SpatialDomains._SpatialDomains as SD
from mesh_builder import MeshBuilder


CoordTriple = tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]


def asarrays(coords: CoordTriple) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    return tuple(map(np.asarray, coords))


class CoordinateSystem(Enum):
    Cartesian = lambda x1, x2, x3: (x1, x2, x3)
    Cylindrical = lambda x1, x2, x3: (x1 * np.sin(x3), x1 * np.cos(x3), x2)


CartesianCoordinates = Literal[CoordinateSystem.Cartesian]
CylindricalCoordinates = Literal[CoordinateSystem.Cylindrical]
C = TypeVar("C", CartesianCoordinates, CylindricalCoordinates)


@dataclass
class SliceCoord(Generic[C]):
    x1: float
    x2: float
    system: C

    def __iter__(self) -> Iterator[float]:
        yield self.x1
        yield self.x2


@dataclass
class SliceCoords(Generic[C]):
    x1: npt.NDArray
    x2: npt.NDArray
    system: C

    def iter_points(self) -> Iterator[SliceCoord]:
        for x1, x2 in cast(
            Iterator[tuple[float, float]], zip(*np.broadcast_arrays(self.x1, self.x2))
        ):
            yield SliceCoord(x1, x2, self.system)

    def __iter__(self) -> Iterator[npt.NDArray]:
        for array in np.broadcast_arrays(self.x1, self.x2):
            yield array

    def __len__(self) -> int:
        return np.broadcast(self.x1, self.x2).size


@dataclass
class Coords(Generic[C]):
    x1: npt.NDArray
    x2: npt.NDArray
    x3: npt.NDArray
    system: C

    def offset(self, dx3: npt.ArrayLike) -> "Coords[C]":
        return Coords(self.x1, self.x2, self.x3 + dx3, self.system)

    def to_cartesian(self) -> "Coords[CartesianCoordinates]":
        return Coords(
            *asarrays(self.system(*self)),
            CoordinateSystem.Cartesian,
        )

    def __iter__(self) -> Iterator[npt.NDArray]:
        for array in np.broadcast_arrays(self.x1, self.x2, self.x3):
            yield array

    def __len__(self) -> int:
        return np.broadcast(self.x1, self.x2, self.x3).size


FieldTrace = Callable[
    [SliceCoord[C], npt.ArrayLike], tuple[SliceCoords[C], npt.NDArray]
]

NormalisedFieldLine = Callable[[npt.ArrayLike], Coords[C]]


@dataclass(frozen=True)
class Curve(Generic[C]):
    line: NormalisedFieldLine
    order: int
    offset: float
    system: Type[C]


@dataclass(frozen=True)
class Quad(Generic[C]):
    north: Curve[C]
    south: Curve[C]


@dataclass(frozen=True)
class Tet(Generic[C]):
    north: Curve[C]
    south: Curve[C]
    east: Curve[C]
    west: Curve[C]


Element = Quad | Tet


@dataclass(frozen=True)
class SavedElement:
    shape: Element
    element: Any
    near_face: Any
    far_face: Any


@dataclass(frozen=True)
class SavedLayer:
    elements: frozenset[SavedElement]
    domain: Any


def layer_coord(element: Element) -> float:
    return element.north.offset


Mesh = frozenset[Quad] | frozenset[Tet]
SavedCurve = tuple[int, Curve]


# Think the functional way to prevent duplicate points/curves being
# created when writing NekMesh file would be a monad which keeps track
# of what has been created already. Would be using a monad to
# represent the I/O anyway. Don't think there is a way to prevent
# duplication without having the whole list in memory.


def normalise_field_line(
    trace: "FieldTrace[C]",
    start: SliceCoord[C],
    x3_min: float,
    x3_max: float,
    resolution=10,
) -> NormalisedFieldLine[C]:
    x_extrusion = np.linspace(x3_min, x3_max, resolution)
    x_cross_section, s = trace(start, x_extrusion)
    coordinates = np.stack([*x_cross_section, x_extrusion])
    order = "cubic" if len(s) > 2 else "linear"
    interp = interp1d((s - s[0]) / (s[-1] - s[0]), coordinates, order)
    coord_system = start.system
    return lambda s: Coords(*asarrays(interp(s)), coord_system)


def make_lagrange_interpolation(
    norm_line: NormalisedFieldLine[C], order=13
) -> tuple[Coords[C], NormalisedFieldLine[C]]:
    control_points = np.linspace(0.0, 1.0, order + 1)
    coords = norm_line(control_points)
    interpolators = [lagrange(control_points, coord) for coord in coords]
    return coords, lambda s: Coords(
        *asarrays(tuple(interp(s) for interp in interpolators)), coords.system
    )


CurveAndPoints = tuple[SD.Curve, SD.PointGeom, SD.PointGeom]
Connectivity = Sequence[Sequence[int]]


def ordered_connectivity(size: int) -> Connectivity:
    return [[1]] + [[i - 1, i + 1] for i in range(1, size - 1)] + [[size - 2]]


class NodeStatus(Enum):
    UNKNOWN = 1
    EXTERNAL = 2
    INTERNAL = 3


def is_line_in_domain(field_line: NormalisedFieldLine, bounds) -> bool:
    # FIXME: This is just a dumb stub. It doesn't even work properly for 2-D, let alone 3-D.
    x1, _, _ = field_line(0.0)
    return x1 >= bounds[0] and x1 <= bounds[1]


def is_skin_node(
    node_status: NodeStatus, connections: Iterable[int], statuses: Sequence[NodeStatus]
) -> bool:
    return node_status != NodeStatus.EXTERNAL and any(
        statuses[i] == NodeStatus.EXTERNAL for i in connections
    )


def classify_node_position(
    lines: Sequence[NormalisedFieldLine],
    bounds,
    connectivity: Connectivity,
    skin_nodes: Sequence[bool],
    status: Optional[Sequence[NodeStatus]] = None,
) -> tuple[Sequence[NodeStatus], Sequence[bool]]:
    def check_is_in_domain(
        line: NormalisedFieldLine, is_skin: bool, status: NodeStatus
    ) -> tuple[NodeStatus, bool]:
        if is_skin and status == NodeStatus.UNKNOWN:
            if is_line_in_domain(line, bounds):
                return NodeStatus.INTERNAL, False
            else:
                return NodeStatus.EXTERNAL, True
        else:
            return status, False

    if status is None:
        status = [NodeStatus.UNKNOWN] * len(lines)

    updated_status, newly_external = zip(
        *itertools.starmap(check_is_in_domain, zip(lines, skin_nodes, status))
    )
    updated_skin = list(
        map(
            lambda x: is_skin_node(*x, updated_status),
            zip(updated_status, connectivity),
        )
    )
    if any(newly_external):
        return classify_node_position(
            lines, bounds, connectivity, updated_skin, updated_status
        )
    else:
        return updated_status, updated_skin


def field_aligned_2d(
    poloidal_mesh: SliceCoords[C],
    field_line: "FieldTrace[C]",
    extrusion_limits: tuple[float, float] = (0.0, 1.0),
    bounds=(0.0, 1.0),
    n: int = 10,
    order: int = 1,
    connectivity: Optional[Connectivity] = None,
    skin_nodes: Optional[Sequence[bool]] = None,
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
    # Ensure poloidal mesh is actually 1-D (required to keep output 2-D)
    flattened_mesh = SliceCoords(poloidal_mesh.x1, np.array(0.0), poloidal_mesh.system)

    # Calculate toroidal positions for nodes in final mesh
    dx3 = (extrusion_limits[1] - extrusion_limits[0]) / n
    x3_mid = np.linspace(
        extrusion_limits[0] + 0.5 * dx3, extrusion_limits[1] - 0.5 * dx3, n
    )
    spline_field_lines = map(
        lambda coord: normalise_field_line(
            field_line, coord, -0.5 * dx3, 0.5 * dx3, min(10, 2 * order)
        ),
        flattened_mesh.iter_points(),
    )
    control_points, lagrange_interp = zip(
        *map(lambda line: make_lagrange_interpolation(line, order), spline_field_lines)
    )

    num_nodes = len(flattened_mesh)
    if connectivity is None:
        connectivity = ordered_connectivity(num_nodes)
    if skin_nodes is None:
        skin_nodes = [True] + list(itertools.repeat(False, num_nodes - 2)) + [True]

    node_status, updated_skin_nodes = classify_node_position(
        lagrange_interp, bounds, connectivity, skin_nodes
    )

    # Need to filter out field lines that leave domain
    # Tag elements adjacent to boundaries
    # Work out max and min distances between near-boundary elements and the boundary
    # Add extra geometry elements to fill in boundaries

    # FIXME: how to get coordinate system type
    # FIXME: already evaluated at various points to get Lagrange interpolant, so maybe should just keep list of points
    # FIXME: Lagrange interpolant should only be returning cartesian coordinates, as that reflects how interpolation will be done in Nektar++
    mesh = frozenset(
        Quad(
            Curve(line1, order, x3, CoordinateSystem.Cartesian),
            Curve(line2, order, x3, CoordinateSystem.Cartesian),
        )
        for x3, (line1, line2) in itertools.product(
            x3_mid, itertools.pairwise(lagrange_interp)
        )
    )

    layers = (layer for _, layer in itertools.groupby(mesh, layer_coord))

    # mesh is now sufficient to represent the data in
    # Python. Everything else is specific to creating Nektar++ meshes
    # and can be placed elsewhere.

    # Use functools.cache to generate unique output objects
    
    builder = MeshBuilder(2, 2)

    curves_start_end = (
        builder.make_curves_and_points(*c.offset(phi).to_cartesian())
        for phi, c in itertools.product(
            x3_mid,
            (
                p
                for p, s in zip(control_points, node_status)
                if s != NodeStatus.EXTERNAL
            ),
        )
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
    # FIXME: Pretty sure this is making connections between nodes in adjacent layers
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

    # Evaluate all of the (lazy) iterators
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


def straight_field(angle=0.0) -> "FieldTrace[C]":
    """Returns a field trace corresponding to straight field lines
    slanted at `angle` above the direction of extrusion into the first
    coordinate direction."""

    def trace(
        start: SliceCoord[C], perpendicular_coord: npt.ArrayLike
    ) -> tuple[SliceCoords[C], npt.NDArray]:
        """Returns a trace for a straight field line."""
        x1 = start.x1 + perpendicular_coord * np.tan(angle)
        x2 = np.asarray(start.x2)
        x3 = np.asarray(perpendicular_coord)
        return SliceCoords(x1, x2, start.system), x3

    return trace


num_nodes = 5

m = field_aligned_2d(
    SliceCoords(
        np.linspace(0, 1, num_nodes), np.zeros(num_nodes), CoordinateSystem.Cartesian
    ),
    straight_field(),
    (0.0, 1.0),
    (0.0, 1.0),
    4,
    2,
)
m.Write("test_geometry.xml", False, SD.FieldMetaDataMap())
