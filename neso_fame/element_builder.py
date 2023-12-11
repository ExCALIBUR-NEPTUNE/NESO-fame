"""Class to build mesh elements."""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cache
from typing import Optional, cast

from hypnotoad import Mesh as HypnoMesh  # type: ignore

from neso_fame.hypnotoad_interface import (
    connect_to_o_point,
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


@dataclass(frozen=True)
class _RingFragment:
    vertices: list[SliceCoord]
    quads: list[Quad]
    counterclockwiseness: int

    def __iter__(self) -> Iterator[SliceCoord]:
        if self.counterclockwiseness < 0:
            return iter(self.vertices[::-1])
        return iter(self.vertices)

    def vertex_list(self) -> list[SliceCoord]:
        if self.counterclockwiseness < 0:
            return self.vertices[::-1]
        return list(self.vertices)

    def iter_quads(self) -> Iterator[Quad]:
        if self.counterclockwiseness < 0:
            return iter(self.quads[::-1])
        return iter(self.quads)

    def quad_list(self) -> list[Quad]:
        if self.counterclockwiseness < 0:
            return self.quads[::-1]
        return list(self.quads)

    def coord_index(self, coord: SliceCoord) -> int:
        try:
            index = next(
                itertools.dropwhile(lambda x: x[1] != coord, enumerate(self.vertices))
            )[0]
        except StopIteration:
            raise ValueError(f"{coord} not present in fragment")
        return index

    def reverse(self) -> _RingFragment:
        return _RingFragment(
            self.vertices[::-1], self.quads[::-1], -self.counterclockwiseness
        )


RingFragments = list[_RingFragment]


@dataclass(frozen=True)
class _VertexRing:
    fragments: list[_RingFragment]
    linking_quad: Optional[Quad] = None

    def find_fragment_and_position(
        self, coord: SliceCoord
    ) -> tuple[_RingFragment, int]:
        for fragment in self.fragments:
            try:
                i = fragment.coord_index(coord)
                return fragment, i
            except ValueError:
                pass
        raise ValueError(f"{coord} not present in ring fragments.")

    @property
    def complete(self) -> bool:
        return self.linking_quad is not None

    @staticmethod
    def _fragment_with_coord_at_start(
        fragments: list[_RingFragment], coord: SliceCoord
    ) -> tuple[_RingFragment, RingFragments]:
        for i, fragment in enumerate(fragments):
            if fragment.vertices[0] == coord:
                return fragment, fragments[:i] + fragments[i + 1 :]
        # If can't find at the start of a fragment, maybe the fragment
        # is the wrong way round and we need to reverse it.
        for i, fragment in enumerate(fragments):
            if fragment.vertices[-1] == coord:
                return fragment.reverse(), fragments[:i] + fragments[i + 1 :]
        return _RingFragment([coord], [], 0), fragments

    @staticmethod
    def _fragment_with_coord_at_end(
        fragments: list[_RingFragment], coord: SliceCoord
    ) -> tuple[_RingFragment, RingFragments]:
        for i, fragment in enumerate(fragments):
            if fragment.vertices[-1] == coord:
                return fragment, fragments[:i] + fragments[i + 1 :]
        # If can't find at the end of a fragment, maybe the fragment
        # is the wrong way round and we need to reverse it.
        for i, fragment in enumerate(fragments):
            if fragment.vertices[0] == coord:
                return fragment.reverse(), fragments[:i] + fragments[i + 1 :]
        return _RingFragment([coord], [], 0), fragments

    def add_vertices(self, left: SliceCoord, right: SliceCoord, q: Quad) -> _VertexRing:
        # left and right assumed to be in counter-clockwise order
        if self.complete:
            raise ValueError("Can not add vertices to a complete ring.")
        start_fragment, remaining = self._fragment_with_coord_at_end(
            self.fragments, left
        )
        if right == start_fragment.vertices[0]:
            if len(remaining) > 0:
                raise ValueError(
                    "Fragment forms complete ring while leaving others disconnnected."
                )
            return _VertexRing([start_fragment], q)
        end_fragment, remaining = self._fragment_with_coord_at_start(remaining, right)
        return _VertexRing(
            [
                _RingFragment(
                    start_fragment.vertices + end_fragment.vertices,
                    start_fragment.quads + [q] + end_fragment.quads,
                    start_fragment.counterclockwiseness
                    + end_fragment.counterclockwiseness
                    + 1,
                )
            ]
            + remaining
        )

    def __iter__(self) -> Iterator[SliceCoord]:
        if not self.complete:
            raise ValueError("Can not iterate over incomplete ring.")
        return iter(self.fragments[0])

    def quads(self) -> Iterator[Quad]:
        if not self.complete:
            raise ValueError("Can not iterate over incomplete ring.")
        return itertools.chain(
            [cast(Quad, self.linking_quad)], self.fragments[0].iter_quads()
        )

    def vertices_between(
        self, start: SliceCoord, end: SliceCoord
    ) -> Iterator[SliceCoord]:
        fragment, start_index = self.find_fragment_and_position(start)
        end_index = fragment.coord_index(end)
        vertices = fragment.vertex_list()
        if end_index < start_index:
            if self.complete:
                return itertools.chain(
                    vertices[start_index:], vertices[: end_index + 1]
                )
            raise ValueError(
                f"{end} falls after {start} in a chain that hasn't been joined into a ring."
            )
        return iter(vertices[start_index : end_index + 1])

    def quads_between(self, start: SliceCoord, end: SliceCoord) -> Iterator[Quad]:
        fragment, start_index = self.find_fragment_and_position(start)
        end_index = fragment.coord_index(end)
        quads = fragment.quad_list()
        if end_index <= start_index:
            if self.complete:
                return itertools.chain(
                    quads[start_index:],
                    [cast(Quad, self.linking_quad)],
                    quads[:end_index],
                )
            raise ValueError(
                f"{end} falls after {start} in a chain that hasn't been joined into a ring."
            )
        return iter(quads[start_index:end_index])


class ElementBuilder:
    """Provides methods for building mesh elements for a Hypnotoad mesh.

    It is able to work out how the outermost layer of vertices in the
    mesh are connected to each other.

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
        self._outermost_vertex_ring: _VertexRing = _VertexRing([])
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
            self._add_vertices(north, south, q)
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
            self._add_vertices(north, south, q)
        return q

    @cache
    def make_quad_to_o_point(self, coord: SliceCoord) -> Quad:
        """Create a quad along a straight line between the point and the magnetic axis."""
        return Quad(
            connect_to_o_point(self._equilibrium, coord),
            self._tracer,
            self._dx3,
            aligned_edges=QuadAlignment.NORTH,
        )

    @cache
    def make_wall_quad(self, north: SliceCoord, south: SliceCoord) -> Quad:
        """Create a quad along the wall of the tokamak."""
        return Quad(
            StraightLineAcrossField(north, south),
            self._tracer,
            self._dx3,
            aligned_edges=QuadAlignment.NONALIGNED,
        )

    @cache
    def make_mesh_to_wall_quad(
        self, wall_coord: SliceCoord, mesh_coord: SliceCoord
    ) -> Quad:
        """Make a quad stretching from a point on the wall to the existing mesh."""
        return Quad(
            StraightLineAcrossField(wall_coord, mesh_coord),
            self._tracer,
            self._dx3,
            aligned_edges=QuadAlignment.SOUTH,
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
        """Create a a triangular prism from the two arguments and the O-point."""
        # Only half the permutations of edge ordering seem to work
        # with Nektar++, but this is not documented. I can't figure
        # out the rule, but this seems to work.
        return Prism(
            (
                self.make_quad_to_o_point(north),
                self.make_quad_to_o_point(south),
                self.flux_surface_quad(north, south),
            )
        )

    def make_wall_prism(
        self, mesh_vertex: SliceCoord, north: SliceCoord, south: SliceCoord
    ) -> Prism:
        """Create a triangular prism from two points on the wall and one on the mesh."""
        return Prism(
            (
                self.make_mesh_to_wall_quad(north, mesh_vertex),
                self.make_mesh_to_wall_quad(south, mesh_vertex),
                self.make_wall_quad(north, south),
            )
        )

    def make_wall_hex(
        self,
        mesh_quad: Quad,
        mesh_north: SliceCoord,
        mesh_south: SliceCoord,
        wall_north: SliceCoord,
        wall_south: SliceCoord,
    ) -> Prism:
        """Create a hexahedron with one quad on the mesh and another on the wall."""
        return Prism(
            (
                self.make_wall_quad(wall_north, wall_south),
                mesh_quad,
                self.make_mesh_to_wall_quad(wall_north, mesh_north),
                self.make_mesh_to_wall_quad(wall_south, mesh_south),
            )
        )

    def outermost_vertices_between(
        self, start: SliceCoord, end: SliceCoord
    ) -> Iterator[SliceCoord]:
        """Return the sequence of nodes on the outermost edge of the mesh.

        The iterator begins with `start` and then follows the
        outermost edge of the mesh counter-clockwise until it reaches
        `end`. These limits will both be included in the iterator.

        """
        return self._outermost_vertex_ring.vertices_between(start, end)

    def outermost_quads_between(
        self, start: SliceCoord, end: SliceCoord
    ) -> Iterator[Quad]:
        """Return the sequence of quads on the outermost edge of the mesh.

        The iterator begins with a quad where `start` is at one edge
        and takes adjacent quads, moving counter-clockwise, until it
        finally returns one containing `end`. These limits will both
        be included in the iterator.

        """
        return iter(self._outermost_vertex_ring.quads_between(start, end))

    def outermost_vertices(self) -> Iterator[SliceCoord]:
        """Iterate over the vertices at the outermost edge of the mesh still in the vessel.

        The order of the iteration is the order these vertices are
        connected into a ring, moving counter-clockwise. The
        start-point is arbitrary.

        """
        return iter(self._outermost_vertex_ring)

    def outermost_quads(self) -> Iterator[Quad]:
        """Iterate over the quads at the outermost edge of the mesh still in the vessel.

        The order of the iteration is the order these quads are
        connected into a ring, moving counter-clockwise. The
        start-point is arbitrary.

        """
        return self._outermost_vertex_ring.quads()

    def _order_vertices_counter_clockwise(
        self, left: SliceCoord, right: SliceCoord
    ) -> tuple[SliceCoord, SliceCoord]:
        if (
            left.system != CoordinateSystem.CYLINDRICAL
            or right.system != CoordinateSystem.CYLINDRICAL
        ):
            raise ValueError("Both vertices must use cylindrical coordinates")
        o_to_l = (left.x1 - self._o_point.x1, left.x2 - self._o_point.x2)
        l_to_r = (right.x1 - left.x1, right.x2 - left.x2)
        cross_prod = o_to_l[0] * l_to_r[1] - o_to_l[1] * l_to_r[0]
        if cross_prod == 0.0:
            raise ValueError("Ordering of coordinates unclear")
        if cross_prod > 0:
            return left, right
        return right, left

    def _add_vertices(self, left: SliceCoord, right: SliceCoord, q: Quad) -> None:
        self._outermost_vertex_ring = self._outermost_vertex_ring.add_vertices(
            *self._order_vertices_counter_clockwise(left, right),
            q,
        )
