"""Class to build mesh elements."""

from __future__ import annotations

from collections import defaultdict
from functools import cache

from hypnotoad import Mesh as HypnoMesh  # type: ignore

from neso_fame.hypnotoad_interface import (
    flux_surface_edge,
    perpendicular_edge,
)
from neso_fame.mesh import (
    CoordinateSystem,
    FieldTracer,
    Prism,
    Quad,
    QuadAlignment,
    SliceCoord,
    StraightLineAcrossField,
)


class ElementBuilder:
    """Provides methods for building mesh elements for a Hypnotoad mesh.

    Group
    -----
    builder

    """

    def __init__(
        self,
        hypnotoad_poloidal_mesh: HypnoMesh,
        tracer: FieldTracer,
        dx3: float,
        outermost_in_vessel: frozenset[SliceCoord],
    ) -> None:
        """Instantiate an object of this class.

        Parameters
        ----------
        hypnotoad_poloidal_mesh
            The 2D poloidal mesh from which a 3D mesh is being generated.
        tracer
            Object to follow along a field line.
        dx3
            Width of a layer of the mesh.
        outermost_in_vessel
            The outermost layer of nodes still inside the vessel of the
            tokamak.

        """
        self._equilibrium = hypnotoad_poloidal_mesh.equilibrium
        self._tracer = tracer
        self._dx3 = dx3
        self._outermost_in_vessel = outermost_in_vessel
        self.outermost_quads: defaultdict[SliceCoord, set[Quad]] = defaultdict(set)
        op = hypnotoad_poloidal_mesh.equilibrium.o_point
        self._o_point = SliceCoord(op.R, op.Z, CoordinateSystem.CYLINDRICAL)

    @cache
    def perpendicular_quad(self, north: SliceCoord, south: SliceCoord) -> Quad:
        """Create a quad between two points, perpendicular to flux surfaces."""
        q = Quad(
            perpendicular_edge(self._equilibrium, north, south),
            self._tracer,
            self._dx3,
        )
        if north in self._outermost_in_vessel and south in self._outermost_in_vessel:
            self.outermost_quads[north].add(q)
            self.outermost_quads[south].add(q)
        return q

    @cache
    def flux_surface_quad(self, north: SliceCoord, south: SliceCoord) -> Quad:
        """Create a quad between two points, following a flux surface."""
        q = Quad(
            flux_surface_edge(self._equilibrium, north, south),
            self._tracer,
            self._dx3,
        )
        if north in self._outermost_in_vessel and south in self._outermost_in_vessel:
            self.outermost_quads[north].add(q)
            self.outermost_quads[south].add(q)
        return q

    @cache
    def make_connecting_quad(self, coord: SliceCoord) -> Quad:
        """Create a quad along a straight line between the point and the magnetic axis."""
        return Quad(
            StraightLineAcrossField(coord, self._o_point),
            self._tracer,
            self._dx3,
            aligned_edges=QuadAlignment.NORTH,
        )

    def make_hex(
        self, sw: SliceCoord, se: SliceCoord, nw: SliceCoord, ne: SliceCoord
    ) -> Prism:
        """Createa a hexahedron from the four points on the poloidal plane."""
        return Prism(
            (
                self.flux_surface_quad(nw, ne),
                self.flux_surface_quad(sw, se),
                self.perpendicular_quad(se, ne),
                self.perpendicular_quad(sw, nw),
            )
        )

    def make_prism(self, north: SliceCoord, south: SliceCoord) -> Prism:
        """Createa a triangular prism from the two arguments and the O-point."""
        # Only half the permutations of edge ordering seem to work
        # with Nektar++, but this is not documented. I can't figure
        # out the rule, but this seems to work.
        return Prism(
            (
                self.make_connecting_quad(north),
                self.make_connecting_quad(south),
                self.flux_surface_quad(north, south),
            )
        )
