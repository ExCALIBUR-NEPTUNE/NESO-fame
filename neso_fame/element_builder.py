"""Class to build mesh elements."""

from __future__ import annotations

import itertools
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cached_property, reduce
from typing import Callable, Optional
from warnings import warn

import numpy as np
import numpy.typing as npt
from hypnotoad import Mesh as HypnoMesh  # type: ignore
from hypnotoad.cases.tokamak import TokamakEquilibrium  # type: ignore

from neso_fame.coordinates import (
    CoordinateSystem,
    CoordMap,
    FrozenCoordSet,
    SliceCoord,
    SliceCoords,
    coord_cache,
)
from neso_fame.hypnotoad_interface import (
    connect_to_o_point,
    flux_surface_edge,
    perpendicular_edge,
)
from neso_fame.mesh import (
    AcrossFieldCurve,
    FieldTracer,
    Prism,
    Quad,
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
        return all(f.complete for f in self.fragments)

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
        return self.largest_vertex_ring()

    def largest_vertex_ring(self) -> Iterator[SliceCoord]:
        if len(self.fragments) == 0:
            return iter(())
        fragment = sorted(self.fragments, key=lambda x: len(x.vertices))[-1]
        if not fragment.complete:
            raise ValueError("Can not iterate over incomplete ring.")
        return iter(fragment)

    def smallest_vertex_ring(self) -> Iterator[SliceCoord]:
        if len(self.fragments) == 0:
            return iter(())
        fragment = sorted(self.fragments, key=lambda x: len(x.vertices))[0]
        if not fragment.complete:
            raise ValueError("Can not iterate over incomplete ring.")
        return iter(fragment)

    def quads(self) -> Iterator[Quad]:
        if len(self.fragments) > 1:
            warn("Multiple vertex rings detected; iterating over largest.")
        return self.largest_quad_ring()

    def largest_quad_ring(self) -> Iterator[Quad]:
        if len(self.fragments) == 0:
            return iter(())
        fragment = sorted(self.fragments, key=lambda x: len(x.vertices))[-1]
        if not fragment.complete:
            raise ValueError("Can not iterate over incomplete ring.")
        return fragment.iter_quads()

    def smallest_quad_ring(self) -> Iterator[Quad]:
        if len(self.fragments) == 0:
            return iter(())
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


def _poloidal_map_between(
    eq: TokamakEquilibrium, north: AcrossFieldCurve, south: AcrossFieldCurve
) -> Callable[[npt.ArrayLike, npt.ArrayLike], SliceCoords]:
    """Return a poloidal map between two curves.

    These curves must share start and end flux surfaces.

    """

    def poloidal_map(s: npt.ArrayLike, t: npt.ArrayLike) -> SliceCoords:
        s_mask = s.mask if isinstance(s, np.ma.MaskedArray) else False
        t_mask = t.mask if isinstance(t, np.ma.MaskedArray) else False
        s, t = np.broadcast_arrays(s, t, subok=True)
        new_mask = np.logical_or(s_mask, t_mask)
        if isinstance(s, np.ma.MaskedArray):
            s.mask = new_mask
        if isinstance(t, np.ma.MaskedArray):
            t.mask = new_mask

        svals, s_invert = np.unique(s, return_inverse=True)
        tvals, t_invert = np.unique(t, return_inverse=True)
        souths = south(tvals)
        norths = north(tvals)
        shape = (len(tvals), len(svals))
        R_tmp = np.empty(shape)
        Z_tmp = np.empty(shape)
        for i, (sth, nth) in enumerate(zip(souths.iter_points(), norths.iter_points())):
            # FIXME: This might be more efficient if only
            # calculate for the s values needed for each t value
            coords = flux_surface_edge(eq, nth, sth)(svals)
            R_tmp[i, :] = coords.x1
            Z_tmp[i, :] = coords.x2
        return SliceCoords(
            np.ma.array(R_tmp[t_invert, s_invert].reshape(s.shape), mask=new_mask),
            np.ma.array(Z_tmp[t_invert, s_invert].reshape(s.shape), mask=new_mask),
            coords.system,
        )

    return poloidal_map


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
        vertex_start_weights: CoordMap[SliceCoord, float],
        system: CoordinateSystem = CoordinateSystem.CYLINDRICAL,
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
        vertex_start_weights
            A map from vertices to the `start_weight` to use for edges
            originating from them. If a vertex is not present in this,
            then the weight will be assumed to be 0 or 1, as appropriate
            for the context. It should always contain at least the
            "outermost" nodes of the plasma-aligned mesh.
        system
            The coordinate system to use for the constructed elements.

        """
        self._equilibrium = hypnotoad_poloidal_mesh.equilibrium
        self._tracer = tracer
        self._dx3 = dx3
        self._vertex_start_weights = vertex_start_weights
        self._edges: dict[FrozenCoordSet[SliceCoord], Quad] = {}
        self._prism_quads: dict[
            FrozenCoordSet[SliceCoord], tuple[Quad, frozenset[Quad]]
        ] = {}
        op = hypnotoad_poloidal_mesh.equilibrium.o_point
        self._o_point = SliceCoord(op.R, op.Z, system)
        self._tracked_perpendicular_quad = self._track_edges(self._perpendicular_quad)
        self._tracked_flux_surface_quad = self._track_edges(self._flux_surface_quad)
        self._quad_to_outer_prism_map: defaultdict[Quad, list[Prism]] = defaultdict(
            list
        )

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
            key = FrozenCoordSet([north, south])
            try:
                return self._edges.pop(key)
            except KeyError:
                q = func(north, south)
                self._edges[key] = q
                return q

        return check_edges

    @coord_cache()
    def _perpendicular_quad(self, north: SliceCoord, south: SliceCoord) -> Quad:
        """Create a quad between two points, perpendicular to flux surfaces."""
        try:
            shape = perpendicular_edge(self._equilibrium, north, south)
        except RuntimeError:
            # This is to account for when we're joining two narrow quads together
            shape = perpendicular_edge(self._equilibrium, north, south, True)
        return Quad(
            shape,
            self._tracer,
            self._dx3,
            north_start_weight=self._vertex_start_weights.get(north, 0.0),
            south_start_weight=self._vertex_start_weights.get(south, 0.0),
        )

    @coord_cache()
    def _flux_surface_quad(self, north: SliceCoord, south: SliceCoord) -> Quad:
        """Create a quad between two points, following a flux surface."""
        return Quad(
            flux_surface_edge(self._equilibrium, north, south),
            self._tracer,
            self._dx3,
            north_start_weight=self._vertex_start_weights.get(north, 0.0),
            south_start_weight=self._vertex_start_weights.get(south, 0.0),
        )

    @coord_cache()
    def make_quad_to_o_point(self, coord: SliceCoord) -> Quad:
        """Create a quad along a straight line between the point and the magnetic axis."""
        return Quad(
            connect_to_o_point(self._equilibrium, coord),
            self._tracer,
            self._dx3,
            north_start_weight=self._vertex_start_weights.get(coord, 0.0),
            south_start_weight=0,
        )

    def make_element(
        self, sw: SliceCoord, se: SliceCoord, nw: SliceCoord, ne: Optional[SliceCoord]
    ) -> Prism:
        """Create an element from the four points on the poloidal plane.

        If four coordinates are provided this will be a
        hexahedron. Otherwise this will be a prism.

        """
        sides: tuple[Quad, ...]
        if isinstance(ne, SliceCoord):
            sides = (
                self._tracked_flux_surface_quad(nw, ne),
                self._tracked_flux_surface_quad(sw, se),
                self._tracked_perpendicular_quad(se, ne),
                self._tracked_perpendicular_quad(sw, nw),
            )
            return Prism(
                sides,
                _poloidal_map_between(
                    self._equilibrium, sides[3].shape, sides[2].shape
                ),
            )
        sides = (
            self._tracked_flux_surface_quad(sw, se),
            self._tracked_perpendicular_quad(sw, nw),
            self._tracked_perpendicular_quad(se, nw),
        )
        return Prism(
            sides,
            _poloidal_map_between(self._equilibrium, sides[1].shape, sides[2].shape),
        )

    def make_prism_to_centre(self, north: SliceCoord, south: SliceCoord) -> Prism:
        """Create a a triangular prism from the two arguments and the O-point."""
        # Only half the permutations of edge ordering seem to work
        # with Nektar++, but this is not documented. I can't figure
        # out the rule, but this seems to work.
        sides = (
            self._tracked_flux_surface_quad(north, south),
            self.make_quad_to_o_point(south),
            self.make_quad_to_o_point(north),
        )

        return Prism(
            sides,
            _poloidal_map_between(self._equilibrium, sides[2].shape, sides[1].shape),
        )

    def make_quad_for_prism(
        self,
        north: SliceCoord,
        south: SliceCoord,
        wall_vertices: frozenset[tuple[SliceCoord, SliceCoord]],
    ) -> tuple[Quad, frozenset[Quad]]:
        """Make a quad for use in a triangular prism in the sapce by the wall.

        This will always return the same quad between two points,
        regardless of the order of the points. It will also return a
        frozenset that will contain the quad if it is on the wall and
        be empty otherwise.

        """
        key = FrozenCoordSet([north, south])
        # Check if this quad will be on the outermost surface of the original hex-mesh
        if key in self._edges:
            return self._edges[key], frozenset()
        # Check if this quad has already been created
        if key in self._prism_quads:
            return self._prism_quads[key]
        # FIXME: mixing aligned and unaligned edges in an element can
        # result in it being ill-formed. No fool-proof way to prevent
        # this, I don't think, but can reduce the probability by
        # transitioning from aligned to unaligned more
        # gradually. Probably even just one layer of edges would be
        # enough. Avoiding elements that are two small would help as well.
        #
        # Think I should build up a list of outermost nodes, followed
        # by those immediately adjacent to them, followed by those
        # immediately adjacent to those, etc. Can then use that to
        # assign "alignedness". Will require some refactoring of how I
        # handle alignment though.

        # PROBLEM: What if a triangle is formed connecting two
        # non-adjacent points on the plasma mesh? Termini have to be
        # aligned but then there would be no guarantee that the centre
        # would remain within the walls.
        q = Quad(
            StraightLineAcrossField(north, south),
            self._tracer,
            self._dx3,
            north_start_weight=self._vertex_start_weights.get(north, 1.0),
            south_start_weight=self._vertex_start_weights.get(south, 1.0),
        )
        if (north, south) in wall_vertices or (south, north) in wall_vertices:
            b = frozenset({q})
        else:
            b = frozenset()
        self._prism_quads[key] = (q, b)
        return q, b

    def make_wall_quad_for_prism(self, shape: AcrossFieldCurve) -> Quad:
        """Create a quad that is aligned to the vessel wall.

        These quads will be used for building triangular prisms by the
        tokamak wall. This method can be used for cases where you want
        the quad to be curved in the poloidal plane. Typically it
        would be called in advance of the construction of the
        prism. It will then keep a copy of the quad it builds which
        will be returned from future calls to
        :method:`~neso_fame.element_builder.ElementBuilder.make_quad_for_prism`.

        """
        key = FrozenCoordSet(shape([0.0, 1.0]).iter_points())
        if key in self._prism_quads:
            return self._prism_quads[key][0]
        q = Quad(
            shape, self._tracer, self._dx3, north_start_weight=1, south_start_weight=1
        )
        b = frozenset({q})
        self._prism_quads[key] = (q, b)
        return q

    def make_outer_prism(
        self,
        vertex1: SliceCoord,
        vertex2: SliceCoord,
        vertex3: SliceCoord,
        wall_vertex_pairs: frozenset[tuple[SliceCoord, SliceCoord]],
    ) -> tuple[Prism, frozenset[Quad]]:
        """Create a triangular prism between the hexahedral plasma-mesh and the wall.

        It will also return a (possibly empty) set of all the quads in
        that prism that lie on the wall.

        """
        q1, b1 = self.make_quad_for_prism(vertex1, vertex2, wall_vertex_pairs)
        q2, b2 = self.make_quad_for_prism(vertex2, vertex3, wall_vertex_pairs)
        q3, b3 = self.make_quad_for_prism(vertex3, vertex1, wall_vertex_pairs)
        p = Prism((q1, q2, q3))
        self._quad_to_outer_prism_map[q1].append(p)
        self._quad_to_outer_prism_map[q2].append(p)
        self._quad_to_outer_prism_map[q3].append(p)
        return p, b1 | b2 | b3

    def get_element_for_quad(self, q: Quad) -> list[Prism]:
        """Return the prisms with quad `q` as a face.

        Currently only prisms in the region near the wall will be
        returned, as they are the only ones this information is needed
        for.

        """
        return self._quad_to_outer_prism_map[q]

    @cached_property
    def _vertex_rings(self) -> _VertexRing:
        return reduce(
            lambda ring, item: ring.add_vertices(
                *self._order_vertices_counter_clockwise(*item[0]), item[1]
            ),
            self._edges.items(),
            _VertexRing([]),
        )

    def outermost_vertices_between(
        self, start: SliceCoord, end: SliceCoord
    ) -> Iterator[SliceCoord]:
        """Return the sequence of nodes on the outermost edge of the mesh.

        The iterator begins with `start` and then follows the
        outermost edge of the mesh counter-clockwise until it reaches
        `end`. These limits will both be included in the iterator.

        """
        return self._vertex_rings.vertices_between(start, end)

    def outermost_quads_between(
        self, start: SliceCoord, end: SliceCoord
    ) -> Iterator[Quad]:
        """Return the sequence of quads on the outermost edge of the mesh.

        The iterator begins with a quad where `start` is at one edge
        and takes adjacent quads, moving counter-clockwise, until it
        finally returns one containing `end`. These limits will both
        be included in the iterator.

        """
        return iter(self._vertex_rings.quads_between(start, end))

    def outermost_vertices(self) -> Iterator[SliceCoord]:
        """Iterate over the vertices at the outermost edge of the mesh still in the vessel.

        The order of the iteration is the order these vertices are
        connected into a ring, moving counter-clockwise. The
        start-point is arbitrary.

        """
        return self._vertex_rings.largest_vertex_ring()

    def innermost_vertices(self) -> Iterator[SliceCoord]:
        """Iterate over the vertices closest to the core.

        The order of the iteration is the order these vertices are
        connected into a ring, moving counter-clockwise. The
        start-point is arbitrary.

        """
        return self._vertex_rings.smallest_vertex_ring()

    def outermost_quads(self) -> Iterator[Quad]:
        """Iterate over the quads at the outermost edge of the mesh still in the vessel.

        The order of the iteration is the order these quads are
        connected into a ring, moving counter-clockwise. The
        start-point is arbitrary.

        """
        return self._vertex_rings.largest_quad_ring()

    def innermost_quads(self) -> Iterator[Quad]:
        """Iterate over the quads closest to the core.

        The order of the iteration is the order these quads are
        connected into a ring, moving counter-clockwise. The
        start-point is arbitrary.

        """
        return self._vertex_rings.smallest_quad_ring()

    def _order_vertices_counter_clockwise(
        self, left: SliceCoord, right: SliceCoord
    ) -> tuple[SliceCoord, SliceCoord]:
        if left.system != self._o_point.system or right.system != self._o_point.system:
            raise ValueError(
                f"Both vertices must use same {self._o_point.system} coordinates"
            )
        o_to_l = (left.x1 - self._o_point.x1, left.x2 - self._o_point.x2)
        l_to_r = (right.x1 - left.x1, right.x2 - left.x2)
        cross_prod = o_to_l[0] * l_to_r[1] - o_to_l[1] * l_to_r[0]
        if cross_prod == 0.0:
            raise ValueError("Ordering of coordinates unclear")
        if cross_prod > 0:
            return left, right
        return right, left
