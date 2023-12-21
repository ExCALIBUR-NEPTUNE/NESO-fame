"""Class to build mesh elements."""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cache, cached_property, reduce
from typing import Callable, Optional
from warnings import warn

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
    linking_quad: Optional[Quad] = None

    def __iter__(self) -> Iterator[SliceCoord]:
        return iter(self.vertices[self._slice()])

    def iter_quads(self) -> Iterator[Quad]:
        if self.linking_quad is None:
            return iter(self.quads[self._slice()])
        return itertools.chain(self.quads[self._slice()], [self.linking_quad])

    def _slice(self) -> slice:
        if self.counterclockwiseness < 0:
            return slice(None, None, -1)
        return slice(None)

    @property
    def complete(self) -> bool:
        return self.linking_quad is not None

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
            self.vertices[::-1],
            self.quads[::-1],
            -self.counterclockwiseness,
            self.linking_quad,
        )


RingFragments = list[_RingFragment]


@dataclass(frozen=True)
class _VertexRing:
    fragments: list[_RingFragment]

    @property
    def complete(self) -> bool:
        return len(self.fragments) == 1 and self.fragments[0].complete

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

    @staticmethod
    def _fragment_with_coord_at_start(
        fragments: list[_RingFragment], coord: SliceCoord
    ) -> tuple[_RingFragment, RingFragments]:
        for i, fragment in enumerate(fragments):
            if fragment.vertices[0] == coord and fragment.linking_quad is None:
                return fragment, fragments[:i] + fragments[i + 1 :]
        # If can't find at the start of a fragment, maybe the fragment
        # is the wrong way round and we need to reverse it.
        for i, fragment in enumerate(fragments):
            if fragment.vertices[-1] == coord and fragment.linking_quad is None:
                return fragment.reverse(), fragments[:i] + fragments[i + 1 :]
        return _RingFragment([coord], [], 0), fragments

    @staticmethod
    def _fragment_with_coord_at_end(
        fragments: list[_RingFragment], coord: SliceCoord
    ) -> tuple[_RingFragment, RingFragments]:
        for i, fragment in enumerate(fragments):
            if fragment.vertices[-1] == coord and fragment.linking_quad is None:
                return fragment, fragments[:i] + fragments[i + 1 :]
        # If can't find at the end of a fragment, maybe the fragment
        # is the wrong way round and we need to reverse it.
        for i, fragment in enumerate(fragments):
            if fragment.vertices[0] == coord and fragment.linking_quad is None:
                return fragment.reverse(), fragments[:i] + fragments[i + 1 :]
        return _RingFragment([coord], [], 0), fragments

    def add_vertices(self, left: SliceCoord, right: SliceCoord, q: Quad) -> _VertexRing:
        # left and right assumed to be in counter-clockwise order
        start_fragment, remaining = self._fragment_with_coord_at_end(
            self.fragments, left
        )
        if right == start_fragment.vertices[0]:
            return _VertexRing(
                [
                    _RingFragment(
                        start_fragment.vertices,
                        start_fragment.quads,
                        start_fragment.counterclockwiseness,
                        q,
                    )
                ]
                + remaining
            )
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
        if len(self.fragments) > 1:
            warn("Multiple vertex rings detected; iterating over largest.")
        fragment = sorted(self.fragments, key=lambda x: len(x.vertices))[0]
        if not fragment.complete:
            raise ValueError("Can not iterate over incomplete ring.")
        return iter(fragment)

    def quads(self) -> Iterator[Quad]:
        if len(self.fragments) > 1:
            warn("Multiple vertex rings detected; iterating over largest.")
        fragment = sorted(self.fragments, key=lambda x: len(x.vertices))[0]
        if not fragment.complete:
            raise ValueError("Can not iterate over incomplete ring.")
        return fragment.iter_quads()

    def vertices_between(
        self, start: SliceCoord, end: SliceCoord
    ) -> Iterator[SliceCoord]:
        fragment, start_index = self.find_fragment_and_position(start)
        end_index = fragment.coord_index(end)
        vertices = list(fragment)
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
        quads = list(fragment.iter_quads())
        if end_index <= start_index:
            if fragment.complete:
                return itertools.chain(
                    quads[start_index:],
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

        """
        self._equilibrium = hypnotoad_poloidal_mesh.equilibrium
        self._tracer = tracer
        self._dx3 = dx3
        self._edges: dict[frozenset[SliceCoord], Quad] = {}
        self._prism_quads: dict[
            frozenset[SliceCoord], tuple[Quad, frozenset[Quad]]
        ] = {}
        op = hypnotoad_poloidal_mesh.equilibrium.o_point
        self._o_point = SliceCoord(op.R, op.Z, CoordinateSystem.CYLINDRICAL)
        self._tracked_perpendicular_quad = self._track_edges(self.perpendicular_quad)
        self._tracked_flux_surface_quad = self._track_edges(self.flux_surface_quad)

    def _track_edges(
        self, func: Callable[[SliceCoord, SliceCoord], Quad]
    ) -> Callable[[SliceCoord, SliceCoord], Quad]:
        """Decorate a function to keep a record of quads it creates.

        When a quad is first created, a it will be added to
        ``self._edges`` with the a key made up of its end-points. If
        the function is called a second time with the same arguments,
        the quad will be removed from the dictionary. This means that,
        after the mesh is constructed, the dictionary will contain all
        quads that are faces of only one element. These are the
        external faces.

        """

        def check_edges(north: SliceCoord, south: SliceCoord) -> Quad:
            key = frozenset({north, south})
            try:
                return self._edges.pop(key)
            except KeyError:
                q = func(north, south)
                self._edges[key] = q
                return q

        return check_edges

    @cache
    def perpendicular_quad(self, north: SliceCoord, south: SliceCoord) -> Quad:
        """Create a quad between two points, perpendicular to flux surfaces."""
        return Quad(
            perpendicular_edge(self._equilibrium, north, south),
            self._tracer,
            self._dx3,
        )

    @cache
    def flux_surface_quad(self, north: SliceCoord, south: SliceCoord) -> Quad:
        """Create a quad between two points, following a flux surface."""
        return Quad(
            flux_surface_edge(self._equilibrium, north, south),
            self._tracer,
            self._dx3,
        )

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
                self._tracked_flux_surface_quad(nw, ne),
                self._tracked_flux_surface_quad(sw, se),
                self._tracked_perpendicular_quad(se, ne),
                self._tracked_perpendicular_quad(sw, nw),
            )
        )

    def make_prism_to_centre(self, north: SliceCoord, south: SliceCoord) -> Prism:
        """Create a a triangular prism from the two arguments and the O-point."""
        # Only half the permutations of edge ordering seem to work
        # with Nektar++, but this is not documented. I can't figure
        # out the rule, but this seems to work.
        return Prism(
            (
                self.make_quad_to_o_point(north),
                self.make_quad_to_o_point(south),
                self._tracked_flux_surface_quad(north, south),
            )
        )

    def make_quad_for_prism(
        self, north: SliceCoord, south: SliceCoord, wall_vertices: frozenset[SliceCoord]
    ) -> tuple[Quad, frozenset[Quad]]:
        """Make a quad for use in a triangular prism in the sapce by the wall.

        This will always return the same quad between two points,
        regardless of the order of the points. It will also return a
        frozenset that will contain the quad if it is on the wall and
        be empty otherwise.

        """
        key = frozenset({north, south})
        # Check if this quad will be on the outermost surface of the original hex-mesh
        if key in self._edges:
            return self._edges[key], frozenset()
        # Check if this quad has already been created
        if key in self._prism_quads:
            return self._prism_quads[key]
        outermost_north = north in self._outermost_vertex_set
        outermost_south = south in self._outermost_vertex_set
        alignment = (
            QuadAlignment.ALIGNED
            if outermost_north and outermost_south
            else QuadAlignment.NORTH
            if outermost_north
            else QuadAlignment.SOUTH
            if outermost_south
            else QuadAlignment.NONALIGNED
        )
        # PROBLEM: What if a triangle is formed connecting two
        # non-adjacent points on the plasma mesh? Termini have to be
        # aligned but then there would be no guarantee that the centre
        # would remain within the walls.
        q = Quad(
            StraightLineAcrossField(north, south),
            self._tracer,
            self._dx3,
            aligned_edges=alignment,
        )
        if north in wall_vertices and south in wall_vertices:
            b = frozenset({q})
        else:
            b = frozenset()
        self._prism_quads[key] = (q, b)
        return q, b

    def make_outer_prism(
        self,
        vertex1: SliceCoord,
        vertex2: SliceCoord,
        vertex3: SliceCoord,
        wall_vertices: frozenset[SliceCoord],
    ) -> tuple[Prism, frozenset[Quad]]:
        """Create a triangular prism between the hexahedral plasma-mesh and the wall.

        It will also return a (possibly empty) set of all the quads in
        that prism that lie on the wall.

        """
        # FIXME: None of the orders seem to work for all the prisms in
        # the mesh. For each prism I create in nektar writer, could I
        # just try rearanging them until one works?
        q1, b1 = self.make_quad_for_prism(vertex1, vertex2, wall_vertices)
        q2, b2 = self.make_quad_for_prism(vertex2, vertex3, wall_vertices)
        q3, b3 = self.make_quad_for_prism(vertex3, vertex1, wall_vertices)
        #        return Prism((q1, q2, q3)), b1 | b2 | b3
        #        return Prism((q1, q3, q2)), b1 | b2 | b3
        #        return Prism((q2, q1, q3)), b1 | b2 | b3
        #        return Prism((q2, q3, q1)), b1 | b2 | b3
        #        return Prism((q3, q1, q2)), b1 | b2 | b3
        return Prism((q3, q2, q1)), b1 | b2 | b3

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

    @cached_property
    def _outermost_vertex_ring(self) -> _VertexRing:
        return reduce(
            lambda ring, item: ring.add_vertices(
                *self._order_vertices_counter_clockwise(*item[0]), item[1]
            ),
            self._edges.items(),
            _VertexRing([]),
        )

    @cached_property
    def _outermost_vertex_set(self) -> frozenset[SliceCoord]:
        return frozenset(self.outermost_vertices())

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
