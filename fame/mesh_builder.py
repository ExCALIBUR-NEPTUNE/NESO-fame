from collections import Counter
from collections.abc import Iterable
from enum import Enum
from typing import NamedTuple, Optional, Tuple

import NekPy.SpatialDomains._SpatialDomains as SD
import NekPy.LibUtilities._LibUtilities as LU
import numpy as np
from numpy import typing as npt

IDedComposite = Tuple[int, SD.Composite]
IDedDomain = Tuple[int, SD.CompositeMap]


class MeshBuilder:
    """Class to produce Nektar++ mesh elements."""

    class MeshComponentType(Enum):
        POINT = 1
        CURVE = 2
        SEGMENT = 3
        FACE = 4
        ELEMENT = 5
        COMPOSITE = 6
        DOMAIN = 7
        ZONE = 8
        INTERFACE = 9

    def __init__(self, mesh_dim: int, spatial_dim: int):
        self.meshgraph = SD.MeshGraphXml(mesh_dim, spatial_dim)
        self.element_counts: Counter[self.MeshComponentType] = Counter()
        self.points = self.meshgraph.GetAllPointGeoms()
        self.edges = self.meshgraph.GetAllSegGeoms()
        self.curved_edges = self.meshgraph.GetCurvedEdges()
        self.quads = self.meshgraph.GetAllQuadGeoms()
        self.composites = self.meshgraph.GetComposites()
        self.domains = self.meshgraph.GetDomain()
        self.movement = self.meshgraph.GetMovement()

    def _get_id(self, elem_type: MeshComponentType):
        val = self.element_counts[elem_type]
        self.element_counts[elem_type] += 1
        return val

    def make_curves_and_points(
        self, x: npt.ArrayLike, y: npt.ArrayLike, z: npt.ArrayLike
    ) -> Tuple[SD.Curve, SD.PointGeom, SD.PointGeom]:
        curve_id = self._get_id(self.MeshComponentType.CURVE)
        curve = SD.Curve(curve_id, LU.PointsType.PolyEvenlySpaced)
        points = [
            SD.PointGeom(2, -1, *coord)
            for coord in zip(*np.broadcast_arrays(x,y, z))
        ]
        start_id = self._get_id(self.MeshComponentType.POINT)
        points[0].SetGlobalID(start_id)
        end_id = self._get_id(self.MeshComponentType.POINT)
        points[-1].SetGlobalID(end_id)
        self.points[start_id] = points[0]
        self.points[end_id] = points[-1]
        curve.points = points
        self.curved_edges[curve_id] = curve
        return curve, points[0], points[-1]

    def make_edge(
        self, start: SD.PointGeom, end: SD.PointGeom, curve: Optional[SD.Curve] = None
    ) -> SD.SegGeom:
        edge_id = self._get_id(self.MeshComponentType.SEGMENT)
        edge = SD.SegGeom(edge_id, start.GetCoordim(), [start, end], curve)
        self.edges[edge_id] = edge
        return edge

    def make_quad_element(
        self, left: SD.SegGeom, right: SD.SegGeom, top: SD.SegGeom, bottom: SD.SegGeom
    ) -> SD.QuadGeom:
        element_id = self._get_id(self.MeshComponentType.ELEMENT)
        quad = SD.QuadGeom(element_id, [left, top, right, bottom])
        self.quads[element_id] = quad
        return quad

    def make_composite(self, components: list[SD.Geometry]) -> IDedComposite:
        comp_id = self._get_id(self.MeshComponentType.COMPOSITE)
        composite = SD.Composite(components)
        self.composites[comp_id] = composite
        return comp_id, composite

    @staticmethod
    def make_composite_map(composites: Iterable[IDedComposite]) -> SD.CompositeMap:
        comp_map = SD.CompositeMap()
        for comp_id, comp in composites:
            comp_map[comp_id] = comp
        return comp_map

    def make_domain(self, composites: Iterable[IDedComposite]) -> IDedDomain:
        domain_id = self._get_id(self.MeshComponentType.DOMAIN)
        domain = self.make_composite_map(composites)
        self.domains[domain_id] = domain
        return domain_id, domain

    def make_interface(self, composites: Iterable[IDedComposite]) -> SD.Interface:
        return SD.Interface(
            self._get_id(self.MeshComponentType.INTERFACE), self.make_composite_map(composites)
        )

    def make_zone(self, domain: IDedDomain, coord_dim=3) -> SD.ZoneFixed:
        zone = SD.ZoneFixed(
            self._get_id(self.MeshComponentType.ZONE), domain[0], domain[1], coord_dim
        )
        self.movement.AddZone(zone)
        return zone

    def add_interface_pair(
        self, left: SD.Interface, right: SD.Interface, description: str
    ) -> None:
        self.movement.AddInterface(description, left, right)
