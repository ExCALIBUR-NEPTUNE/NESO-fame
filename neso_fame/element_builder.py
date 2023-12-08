"""Class to build mesh elements."""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from functools import cache
from operator import itemgetter
from typing import Optional

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
)

# The quad is the one connecting this node to the one before it in the ring
RingLink = tuple[SliceCoord, Optional[Quad]]
RingFragment = tuple[RingLink, ...]
RingFragments = tuple[RingFragment, ...]
VertexRing = tuple[RingFragments, bool]


def _coord_index_in_fragment(fragment: RingFragment, coord: SliceCoord) -> int:
    try:
        index = next(
            itertools.dropwhile(lambda x: x[1][0] != coord, enumerate(fragment))
        )[0]
    except StopIteration:
        raise ValueError(f"{coord} not present in fragment")
    return index


def _fragment_and_position(
    fragments: RingFragments, coord: SliceCoord
) -> tuple[RingFragment, int]:
    for fragment in fragments:
        try:
            i = _coord_index_in_fragment(fragment, coord)
            return fragment, i
        except ValueError:
            pass
    raise ValueError(f"{coord} not present in ring fragments.")


def _fragment_with_coord_at_start(
    fragments: RingFragments, coord: SliceCoord, q: Quad
) -> tuple[RingFragment, RingFragments]:
    for i, fragment in enumerate(fragments):
        if fragment[0][0] == coord:
            assert fragment[0][1] is None
            return ((fragment[0][0], q),) + fragment[1:], fragments[:i] + fragments[
                i + 1 :
            ]
    return ((coord, q),), fragments


def _fragment_with_coord_at_end(
    fragments: RingFragments, coord: SliceCoord
) -> tuple[RingFragment, RingFragments]:
    for i, fragment in enumerate(fragments):
        if fragment[-1][0] == coord:
            assert fragment[-1][1] is not None
            return fragment, fragments[:i] + fragments[i + 1 :]
    return ((coord, None),), fragments


def _add_vertices(
    ring: VertexRing, left: SliceCoord, right: SliceCoord, q: Quad
) -> VertexRing:
    if ring[1]:
        raise ValueError("Can not add vertices to a complete ring.")
    start_fragment, remaining = _fragment_with_coord_at_end(ring[0], left)
    if right == start_fragment[0][0]:
        if len(remaining) > 0:
            raise ValueError(
                "Fragment forms complete ring while leaving others disconnnected."
            )
        return (((start_fragment[0][0], q),) + start_fragment[1:],), True
    end_fragment, remaining = _fragment_with_coord_at_start(remaining, right, q)
    return (start_fragment + end_fragment,) + remaining, False


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
        self._outermost_vertex_ring: VertexRing = ((), False)
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
    def make_connecting_quad(self, coord: SliceCoord) -> Quad:
        """Create a quad along a straight line between the point and the magnetic axis."""
        return Quad(
            connect_to_o_point(self._equilibrium, coord),
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

    def outermost_vertices_between(
        self, start: SliceCoord, end: SliceCoord
    ) -> Iterator[SliceCoord]:
        """Return the sequence of nodes on the outermost edge of the mesh.

        The iterator begins with `start` and then follows the
        outermost edge of the mesh counter-clockwise until it reaches
        `end`. These limits will both be included in the iterator.

        """
        fragments, complete = self._outermost_vertex_ring
        fragment, start_index = _fragment_and_position(fragments, start)
        end_index = _coord_index_in_fragment(fragment, end)
        if end_index < start_index:
            if complete:
                return map(
                    itemgetter(0),
                    itertools.chain(fragment[start_index:], fragment[: end_index + 1]),
                )
            raise ValueError(
                f"{end} falls after {start} in a chain that hasn't been joined into a ring."
            )
        return map(itemgetter(0), fragment[start_index : end_index + 1])

    def outermost_quads_between(
        self, start: SliceCoord, end: SliceCoord
    ) -> Iterator[Quad]:
        """Return the sequence of quads on the outermost edge of the mesh.

        The iterator begins with a quad where `start` is at one edge
        and takes adjacent quads until it finally returns one
        containing `end`. These limits will both be included in the
        iterator.

        """
        fragments, complete = self._outermost_vertex_ring
        fragment, start_index = _fragment_and_position(fragments, start)
        end_index = _coord_index_in_fragment(fragment, end)
        if end_index <= start_index:
            if complete:
                return map(
                    itemgetter(1),
                    itertools.chain(
                        fragment[start_index + 1 :], fragment[: end_index + 1]
                    ),
                )
            raise ValueError(
                f"{end} falls after {start} in a chain that hasn't been joined into a ring."
            )
        return map(itemgetter(1), fragment[start_index + 1 : end_index + 1])

    def outermost_vertices(self) -> Iterator[SliceCoord]:
        """Iterate over the vertices at the outermost edge of the mesh still in the vessel.

        The order of the iteration is the order tehse vertices are
        connected into a ring, moving counter-clockwise. The
        start-point is arbitrary.

        """
        if not self._outermost_vertex_ring[1]:
            raise ValueError("Outer vertices do not form a ring.")
        return map(itemgetter(0), self._outermost_vertex_ring[0][0])

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
        self._outermost_vertex_ring = _add_vertices(
            self._outermost_vertex_ring,
            *self._order_vertices_counter_clockwise(left, right),
            q,
        )
