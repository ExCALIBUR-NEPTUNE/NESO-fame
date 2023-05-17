# WARNING: This script is coverd by GPL, due to dependency on hypnotoad!
from __future__ import annotations

from collections.abc import Iterator, Sequence, Iterable
from dataclasses import dataclass
from enum import Enum
from functools import cache, cached_property
import itertools
from typing import (
    cast,
    Callable,
    ClassVar,
    Generic,
    Literal,
    Optional,
    Type,
    TypeVar,
)

import numpy as np
import numpy.typing as npt
from scipy.interpolate import interp1d, lagrange


CoordTriple = tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]


def asarrays(coords: CoordTriple) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    return np.asarray(coords[0]), np.asarray(coords[1]), np.asarray(coords[2])


class CoordinateSystem(Enum):
    Cartesian = 0
    Cylindrical = 1


COORDINATE_TRANSFORMS = {
    CoordinateSystem.Cartesian: lambda x1, x2, x3: (x1, x2, x3),
    CoordinateSystem.Cylindrical: lambda x1, x2, x3: (
        x1 * np.sin(x3),
        x1 * np.cos(x3),
        x2,
    ),
}

CartesianCoordinates = Literal[CoordinateSystem.Cartesian]
CylindricalCoordinates = Literal[CoordinateSystem.Cylindrical]
C = TypeVar("C", CartesianCoordinates, CylindricalCoordinates)


@dataclass(frozen=True)
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

    def iter_points(self) -> Iterator[SliceCoord[C]]:
        for x1, x2 in cast(
            Iterator[tuple[float, float]], zip(*np.broadcast_arrays(self.x1, self.x2))
        ):
            yield SliceCoord(x1, x2, self.system)

    def __iter__(self) -> Iterator[npt.NDArray]:
        for array in np.broadcast_arrays(self.x1, self.x2):
            yield array

    def __len__(self) -> int:
        return np.broadcast(self.x1, self.x2).size

    def __getitem__(self, idx) -> SliceCoord[C]:
        x1, x2 = np.broadcast_arrays(self.x1, self.x2)
        return SliceCoord(float(x1[idx]), float(x2[idx]), self.system)


@dataclass(frozen=True)
class Coord(Generic[C]):
    x1: float
    x2: float
    x3: float
    system: C

    def to_cartesian(self) -> "Coord[CartesianCoordinates]":
        return Coord(
            *COORDINATE_TRANSFORMS[self.system](*self),
            CoordinateSystem.Cartesian,
        )

    def __iter__(self) -> Iterator[float]:
        yield self.x1
        yield self.x2
        yield self.x3


@dataclass
class Coords(Generic[C]):
    x1: npt.NDArray
    x2: npt.NDArray
    x3: npt.NDArray
    system: C

    def iter_points(self) -> Iterator[Coord[C]]:
        for x1, x2, x3 in cast(
            Iterator[tuple[float, float, float]],
            zip(*np.broadcast_arrays(self.x1, self.x2, self.x3)),
        ):
            yield Coord(x1, x2, x3, self.system)

    def offset(self, dx3: npt.ArrayLike) -> "Coords[C]":
        return Coords(self.x1, self.x2, self.x3 + dx3, self.system)

    def to_cartesian(self) -> "Coords[CartesianCoordinates]":
        return Coords(
            *asarrays(COORDINATE_TRANSFORMS[self.system](*self)),
            CoordinateSystem.Cartesian,
        )

    def __iter__(self) -> Iterator[npt.NDArray]:
        for array in np.broadcast_arrays(self.x1, self.x2, self.x3):
            yield array

    def __len__(self) -> int:
        return np.broadcast(self.x1, self.x2, self.x3).size

    def __getitem__(self, idx) -> Coord[C]:
        x1, x2, x3 = np.broadcast_arrays(self.x1, self.x2, self.x3)
        return Coord(float(x1[idx]), float(x2[idx]), float(x3[idx]), self.system)


FieldTrace = Callable[
    [SliceCoord[C], npt.ArrayLike], tuple[SliceCoords[C], npt.NDArray]
]
NormalisedFieldLine = Callable[[npt.ArrayLike], Coords[C]]


@dataclass(frozen=True)
class Curve(Generic[C]):
    function: NormalisedFieldLine[C]
    # control_points: Coords[C]

    # @cached_property
    # def _interpolant(self) -> NormalisedFieldLine:
    #     s = np.linspace(0.0, 1.0, len(self.control_points))
    #     interpolators = [lagrange(s, coord) for coord in self.control_points]
    #     return lambda s: Coords(
    #         *asarrays(tuple(interp(s) for interp in interpolators)),
    #         self.control_points.system,
    #     )

    def __call__(self, s: npt.ArrayLike) -> Coords[C]:
        """Convenience function so that a Curve is itself a NormalisedFieldLine"""
        return self.function(s)

    def offset(self, offset: float) -> "Curve[C]":
        return Curve(lambda s: self.function(s).offset(offset))

    @cache
    def control_points(self, order) -> Coords[C]:
        s = np.linspace(0.0, 1.0, order + 1)
        return self.function(s)


class SharedBound(Enum):
    North = 1
    South = 2


# FIXME: Eventually we might want to distinguish between order of quads
# when defining the shape versus when representing data in them.
@dataclass(frozen=True)
class Quad(Generic[C]):
    north: Curve[C]
    south: Curve[C]
    in_plane: Optional[Curve[C]]
    field: FieldTrace[C]
    NUM_CORNERS: ClassVar[int] = 4

    def __iter__(self) -> Iterable[Curve[C]]:
        yield self.north
        yield self.south

    @classmethod
    @cache
    def _cached_quad(
        cls,
        north: Curve[C],
        south: Curve[C],
        in_plane: Optional[Curve[C]],
        field: FieldTrace[C],
    ) -> Quad[C]:
        return cls(north, south, in_plane, field)

    @classmethod
    def from_unordered_curves(
        cls,
        curve1: Curve[C],
        curve2: Curve[C],
        in_plane: Optional[Curve[C]],
        field: FieldTrace[C],
    ) -> Quad[C]:
        """Returns the same quad object, regardless of the order the
        curve1 and curve2 arguments."""
        if hash(curve1) < hash(curve2):
            return cls._cached_quad(curve1, curve2, in_plane, field)
        else:
            return cls._cached_quad(curve2, curve1, in_plane, field)

    def corners(self) -> Coords[C]:
        north_corners = self.north.control_points(1)
        south_corners = self.south.control_points(1)
        return Coords(
            np.concatenate([north_corners.x1, south_corners.x1]),
            np.concatenate([north_corners.x2, south_corners.x2]),
            np.concatenate([north_corners.x3, south_corners.x3]),
            north_corners.system,
        )

    def control_points(self, order) -> Coords[C]:
        """Returns the coordinates of the control points for the
        surface, in an array of dimensions [3, order + 1, order + 1].

        """
        if self.in_plane is None:
            # FIXME: Is this strictly correct?
            # FIXME: can I be confident that the start and end will be exactly as specified?
            north_samples = self.north.control_points(order)
            south_samples = self.south.control_points(order)
            return Coords(
                np.linspace(north_samples.x1, south_samples.x1, order + 1),
                np.linspace(north_samples.x2, south_samples.x2, order + 1),
                np.linspace(north_samples.x3, south_samples.x3, order + 1),
                north_samples.system,
            )
        else:
            raise NotImplementedError(
                "Can not yet handle Quads where all four edges are curved"
            )

    def offset(self, offset: float) -> Quad[C]:
        return Quad(
            self.north.offset(offset),
            self.south.offset(offset),
            self.in_plane.offset(offset) if self.in_plane is not None else None,
            self.field,
        )

    # def _normalised_map(self) -> RegularGridInterpolator:
    #     resolution = max(self.north.resolution, self.south.resolution)
    #     normed_coords = np.linspace(0., 1., resolution)
    #     start_points = SliceCoords(np.linspace(self.north.start.x1, self.south.start.x1, resolution), np.linspace(self.north.start.x2, self.south.start.x2, resolution), self.north.start.system)
    #     x3_min = cast(float, self.north(0.0).x3)
    #     x3_max = cast(float, self.north(1.0).x3)
    #     assert x3_min == self.south(0.0).x3, "Bounding curves must have same start and end in x3"
    #     assert x3_max == self.south(1.0).x3, "Bounding curves must have same start and end in x3"
    #     lines = itertools.chain([self.north], (Curve.normalise_field_line(self.field, start, x3_min, x3_max, resolution) for start in start_points.iter_points()), [self.south])
    #     positions = np.swapaxes(np.array([interp(normed_coords) for interp in lines]), 1, 2)
    #     # FIXME: spacing in the non-x3 direction isn't equal along curve of face
    #     order = "cubic" if resolution > 2 else "linear"
    #     return RegularGridInterpolator((normed_coords, normed_coords), positions, order)


@dataclass(frozen=True)
class Tet(Generic[C]):
    north: Quad[C]
    south: Quad[C]
    east: Quad[C]
    west: Quad[C]
    NUM_CORNERS: ClassVar[int] = 8

    def __iter__(self) -> Iterable[Quad[C]]:
        yield self.north
        yield self.east
        yield self.south
        yield self.west

    def corners(self) -> Coords[C]:
        north_corners = self.north.corners()
        south_corners = self.south.corners()
        # TODO Check that east and west corners are the same as north and south
        return Coords(
            np.concatenate([north_corners.x1, south_corners.x1]),
            np.concatenate([north_corners.x2, south_corners.x2]),
            np.concatenate([north_corners.x3, south_corners.x3]),
            north_corners.system,
        )

    @cached_property
    def control_points(self) -> npt.NDArray:
        """Returns the coordinates of the control points for the
        surface, in an array of dimensions [3, order + 1, order + 1, order + 1].

        """
        raise NotImplementedError("Not written yet")

    def quads(self) -> Iterable[Quad[C]]:
        yield self.north
        yield self.east
        yield self.south
        yield self.west

    def offset(self, offset: float) -> Tet[C]:
        return Tet(
            self.north.offset(offset),
            self.south.offset(offset),
            self.east.offset(offset),
            self.west.offset(offset),
        )


E = TypeVar("E", Quad, Tet)
ElementConnections = dict[E, bool]
Connectivity = Sequence[Sequence[int]]


@dataclass(frozen=True)
class MeshLayer(Generic[E]):
    reference_elements: dict[E, ElementConnections[E]]
    offset: Optional[float] = None

    def elements(self) -> Iterable[E]:
        if isinstance(self.offset, float):
            x = self.offset
            return map(lambda e: e.offset(x), self.reference_elements)
        else:
            return self.reference_elements

    def __len__(self) -> int:
        return len(self.reference_elements)

    @cached_property
    def element_type(self) -> Type[E]:
        return type(next(iter(self.reference_elements)))

    def quads(self) -> Iterable[Quad[C]]:
        if len(self.reference_elements) > 0 and issubclass(self.element_type, Quad):
            return cast(Iterable[Quad[C]], self.elements())
        else:
            return itertools.chain.from_iterable(
                map(lambda t: t.quads(), cast(Iterable[Tet[C]], self.elements()))
            )

    @cached_property
    def num_unique_corners(self) -> int:
        element_corners = self.element_type.NUM_CORNERS
        total_corners = element_corners * len(self.reference_elements)
        num_face_connections = sum(
            list(connections.values()).count(True)
            for connections in self.reference_elements.values()
        )
        num_edge_connections = sum(
            list(connections.values()).count(False)
            for connections in self.reference_elements.values()
        )
        assert num_face_connections % 2 == 0, "Ill-defined mesh connectivity"
        assert num_edge_connections % 2 == 0, "Ill-defined mesh connectivity"
        return (
            total_corners
            - (num_face_connections // 2) * (element_corners // 2)
            - (num_edge_connections // 2) * (element_corners // 4)
        )

    @cache
    def num_unique_control_points(self, order: int) -> int:
        points_per_edge = order + 1
        points_per_face = (order + 1) * points_per_edge
        points_per_element = (order + 1) * points_per_face
        total_control_points = points_per_element * len(self.reference_elements)
        num_shared_points = sum(
            sum(
                points_per_face if is_face else points_per_edge
                for is_face in connections.values()
            )
            for element, connections in self.reference_elements.items()
        )
        assert num_shared_points % 2 == 0, "Ill-defined mesh connectivity"
        return total_control_points - num_shared_points // 2


@dataclass(frozen=True)
class Mesh(Generic[E]):
    reference_layer: MeshLayer[E]
    offsets: npt.NDArray

    def layers(self) -> Iterable[MeshLayer[E]]:
        return map(
            lambda off: MeshLayer(self.reference_layer.reference_elements, off),
            self.offsets,
        )

    def __len__(self) -> int:
        return len(self.reference_layer) * self.offsets.size

    @property
    def num_unique_corners(self) -> int:
        return self.offsets.size * self.reference_layer.num_unique_corners

    def num_unique_control_points(self, order: int) -> int:
        return self.offsets.size * self.reference_layer.num_unique_control_points(order)


def normalise_field_line(
    trace: FieldTrace[C],
    start: SliceCoord[C],
    x3_min: float,
    x3_max: float,
    resolution=10,
) -> NormalisedFieldLine[C]:
    x3 = np.linspace(x3_min, x3_max, resolution)
    x1_x2_coords: SliceCoords[C]
    s: npt.NDArray
    x1_x2_coords, s = trace(start, x3)
    coordinates = np.stack([*x1_x2_coords, x3])
    order = "cubic" if len(s) > 2 else "linear"
    interp = interp1d((s - s[0]) / (s[-1] - s[0]), coordinates, order)
    coord_system = start.system

    def normalised_interpolator(s: npt.ArrayLike) -> Coords[C]:
        locations = interp(s)
        return Coords(locations[0], locations[1], locations[2], coord_system)

    return normalised_interpolator


def make_lagrange_interpolation(
    norm_line: Curve, order=1
) -> Curve[CartesianCoordinates]:
    s = np.linspace(0.0, 1.0, order + 1)
    coords = norm_line.control_points(order).to_cartesian()
    interpolators = [lagrange(s, coord) for coord in coords]
    return Curve(
        lambda s: Coords(
            interpolators[0](s),
            interpolators[1](s),
            interpolators[2](s),
            CoordinateSystem.Cartesian,
        )
    )


def ordered_connectivity(size: int) -> Connectivity:
    return [[1]] + [[i - 1, i + 1] for i in range(1, size - 1)] + [[size - 2]]


class NodeStatus(Enum):
    UNKNOWN = 1
    EXTERNAL = 2
    INTERNAL = 3


def is_line_in_domain(field_line: NormalisedFieldLine, bounds: Sequence[float]) -> bool:
    # FIXME: This is just a dumb stub. It doesn't even work properly for 2-D, let alone 3-D.
    location = field_line(0.0)
    x1 = float(location.x1)
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
            lambda x: is_skin_node(x[0], x[1], updated_status),
            zip(updated_status, connectivity),
        )
    )
    if any(newly_external):
        return classify_node_position(
            lines, bounds, connectivity, updated_skin, updated_status
        )
    else:
        return updated_status, updated_skin


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
