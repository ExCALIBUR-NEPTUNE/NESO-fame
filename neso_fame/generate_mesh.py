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


# def make_curve_2D(
#     control_points: Coords2D, curve_id: int, point_id_start=0
# ) -> tuple[CurveAndPoints, int]:
#     """Create a Nektar++ curve object from the control points. This
#     is constructed on a 2D cartesian plane, without any curvature in
#     the toroidal direction. Also returns the first and last point in the curve.

#     """
#     curve = SD.Curve(curve_id, LU.PointsType.PolyEvenlySpaced)
#     points = [
#         SD.PointGeom(2, point_id_start + i, *coord, 0.0)
#         for i, coord in enumerate(zip(*np.broadcast_arrays(*control_points)))
#     ]
#     curve.points = points
#     return (curve, points[0], points[-1]), point_id_start + len(points)


# def make_straight_edge(start: SD.PointGeom, end: SD.PointGeom, edge_id: int):
#     return SD.SegGeom(edge_id, start.GetCoordim(), [start, end])


# def make_curved_edge(
#     curve: SD.Curve, start: SD.PointGeom, end: SD.PointGeom, edge_id: int
# ) -> SD.SegGeom:
#     return SD.SegGeom(edge_id, start.GetCoordim(), [start, end], curve)


# def make_element_2D(
#     left: SD.SegGeom,
#     right: SD.SegGeom,
#     top: SD.SegGeom,
#     bottom: SD.SegGeom,
#     quad_id: int,
# ) -> SD.QuadGeom:
#     return SD.QuadGeom(quad_id, [left, top, right, bottom])


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


# def make_composite_map(comp: SD.Composite, composite_id: int) -> SD.CompositeMap:
#     comp_map = SD.CompositeMap()
#     comp_map[composite_id] = comp
#     return comp_map


# def make_domain(comp: SD.Composite, composite_id: int) -> SD.CompositeMap:
#     return make_composite_map(comp, composite_id)


# def make_interface(comp: SD.Composite, composite_id, interface_id: int) -> SD.Interface:
#     return SD.Interface(interface_id, make_composite_map(comp, composite_id))


# def make_zone(comp: SD.Composite, zone_id: int, coord_dim=3) -> SD.ZoneFixed:
#     return SD.ZoneFixed(zone_id, zone_id, make_domain(comp, zone_id), coord_dim)


# def make_interfaces(
#     layer: SD.Composite,
#     left: SD.Composite,
#     right: SD.Composite,
#     layer_id: int,
#     num_layers,
#     coord_dim=3,
# ) -> tuple[SD.Interface, SD.Interface]:
#     make_zone(layer, layer_id, coord_dim)
#     return make_interface(left, layer_id + num_layers, layer_id), make_interface(
#         right, layer_id + 2 * num_layers, layer_id + num_layers
#     )


# def make_interface_pairs(
#     movement: SD.Movement, left: SD.Interface, right: SD.Interface, pair_id: int
# ) -> None:
#     movement.AddInterface(f"Join {pair_id}", left, right)


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

    # poloidal_positions = field_line(PoloidalCoord2D(np.expand_dims(poloidal_mesh, axis=1)), control_points)

    # n_mesh = len(poloidal_mesh)
    # vertices_per_layer = 2 * n_mesh
    # edges_per_layer = n_mesh + 2 * (n_mesh - 1)
    # elements_per_layer = n_mesh - 1
    # composites_per_layer = 3
    # interfaces_per_layer = 2

    # mesh = SD.MeshGraphXml(2, 2)
    # vertices = mesh.GetAllPointGeoms()
    # edges = mesh.GetAllSegGeoms()
    # elements = mesh.GetAllQuadGeoms()
    # composites = mesh.GetComposites()
    # domains = mesh.GetDomain()

    # near_interfaces = []
    # far_interfaces = []

    # # FIXME: Need to handle truncation of elements at border
    # for i, x_tor in enumerate(toroidal_centres):
    #     # Create vertices for each cell
    #     start_points = [
    #         SD.PointGeom(
    #             2, j + i * vertices_per_layer, x_tor + control_points[0], x_pol, 0
    #         ) for j, x_pol in enumerate(poloidal_positions[:, 0])
    #     ]
    #     end_points = [
    #         SD.PointGeom(
    #             2, j + n_mesh + i * vertices_per_layer, x_tor + control_points[-1], x_pol, 0
    #         ) for j, x_pol in enumerate(poloidal_positions[:, -1])
    #     ]

    #     for vertex in itertools.chain(start_points, end_points):
    #         vertices[vertex.GetGlobalID()] = vertex

    #     # Create edges between nodes
    #     near_edges = [
    #         SD.SegGeom(j + i * edges_per_layer, 2, [v1, v2]) for
    #         j, (v1, v2) in enumerate(itertools.pairwise(start_points))
    #     ]
    #     far_edges = [
    #         SD.SegGeom(j + n_mesh - 1 + i * edges_per_layer, 2, [v1, v2]) for
    #         j, (v1, v2) in enumerate(itertools.pairwise(end_points))
    #     ]
    #     # FIXME: Add curve information
    #     cross_edges = [
    #         SD.SegGeom(j + 2*(n_mesh - 1) + i * edges_per_layer, 2, [v1, v2], None)
    #         for j, (v1, v2) in enumerate(zip(start_points, end_points))
    #     ]

    #     for e in itertools.chain(near_edges, far_edges, cross_edges):
    #         edges[e.GetGlobalID()] = e

    #     # Create quad elements from edges
    #     layer_elements = [
    #         SD.QuadGeom(j + i * elements_per_layer, list(edges))
    #         for j, edges in enumerate(
    #                 zip(near_edges, itertools.islice(cross_edges, 1, None), far_edges, cross_edges)
    #         )
    #     ]

    #     for element in layer_elements:
    #         elements[element.GetGlobalID()] = element

    #     # Create composites from elements and for poloidal faces
    #     # TODO: Add labels?
    #     layer_composite = SD.Composite(layer_elements)
    #     composites[i * composites_per_layer] = layer_composite
    #     near_face_composite = SD.Composite(near_edges)
    #     composites[i * composites_per_layer + 1] = near_face_composite
    #     near_comp_map = SD.CompositeMap()
    #     near_comp_map[i * composites_per_layer + 1] = near_face_composite
    #     far_face_composite = SD.Composite(far_edges)
    #     composites[i * composites_per_layer + 2] = far_face_composite
    #     far_comp_map = SD.CompositeMap()
    #     far_comp_map[i * composites_per_layer + 1] = far_face_composite

    #     # Create domain from element-composite
    #     # FIXME: Think this will need to use a constructor for the C++ type
    #     domain = SD.CompositeMap()
    #     domain[i * composites_per_layer] = layer_composite
    #     domains[i] = domain

    #     # Create zone from domain
    #     zone = SD.ZoneFixed(i, i, domain, 2)
    #     movement.AddZone(zone)

    #     # Create interfaces from face-composites
    #     near_interfaces.append(SD.Interface(i * interfaces_per_layer, near_comp_map))
    #     far_interfaces.append(SD.Interface(i * interfaces_per_layer + 1, far_comp_map))

    # # Create interface pairs for adjacent interfaces
    # for j, (near, far) in enumerate(zip(near_interfaces[1:], far_interfaces[:-1])):
    #     movement.AddInterface(f"Join {j}", near, far)

    # # FIXME: Add boundary composites

    # return mesh


m = field_aligned_2d(
    np.linspace(0, 1, 5), lambda y, x: (PoloidalCoords3D(*np.broadcast_arrays(x, *y)[1:]), x), (0.0, 1.0), 4, 2
)
m.Write("test_geometry.xml", False, SD.FieldMetaDataMap())
