"""Routines to output meshes using the meshio library."""

from collections.abc import Iterator, Sequence
from functools import cache
from typing import Callable, DefaultDict, TypedDict, TypeVar, overload

import meshio  # type: ignore
import numpy as np
import numpy.typing as npt

from .coordinates import Coord, Coords, SliceCoord, SliceCoords, coord_cache
from .mesh import (
    AcrossFieldCurve,
    EndShape,
    Mesh,
    NormalisedCurve,
    Prism,
    PrismMesh,
    Quad,
    control_points,
)


class _ElementType(TypedDict):
    line: str
    triangle: str
    quad: str
    prism: str
    hexahedron: str


C = TypeVar("C", Coord, SliceCoord)

# Each item in the list contains the names of the element types to use
# when the order is one greater than the corresponding list index
# (i.e., item 0 describes 1st order elements, item 1 describes 2nd
# order elements, etc.)
_ELEMENT_TYPES: list[_ElementType] = [
    {
        "line": "line",
        "triangle": "triangle",
        "quad": "quad",
        "prism": "wedge",
        "hexahedron": "hexahedron",
    },
    {
        "line": "line3",
        "triangle": "triangle6",
        "quad": "quad9",
        "prism": "wedge18",
        "hexahedron": "hexahedron27",
    },
    {
        "line": "line4",
        "triangle": "triangle10",
        "quad": "quad16",
        "prism": "wedge40",
        "hexahedron": "hexahedron64",
    },
    {
        "line": "line5",
        "triangle": "triangle15",
        "quad": "quad25",
        "prism": "wedge75",
        "hexahedron": "hexahedron125",
    },
    {
        "line": "line6",
        "triangle": "triangle21",
        "quad": "quad36",
        "prism": "wedge126",
        "hexahedron": "hexahedron216",
    },
    {
        "line": "line7",
        "triangle": "triangle28",
        "quad": "quad49",
        "prism": "wedge196",
        "hexahedron": "hexahedron343",
    },
    {
        "line": "line8",
        "triangle": "triangle36",
        "quad": "quad64",
        "prism": "wedge288",
        "hexahedron": "hexahedron512",
    },
    {
        "line": "line9",
        "triangle": "triangle45",
        "quad": "quad81",
        "prism": "wedge405",
        "hexahedron": "hexahedron729",
    },
    {
        "line": "line10",
        "triangle": "triangle55",
        "quad": "quad100",
        "prism": "wedge550",
        "hexahedron": "hexahedron1000",
    },
]

_ELEMENT_DIMS = {
    k: v
    for d in (
        {
            d["line"]: 1,
            d["triangle"]: 2,
            d["quad"]: 2,
            d["prism"]: 3,
            d["hexahedron"]: 3,
        }
        for d in _ELEMENT_TYPES
    )
    for k, v in d.items()
}


@cache
def _quad_control_points(order: int) -> tuple[npt.NDArray, npt.NDArray]:
    x1, x2 = np.meshgrid(
        np.linspace(0.0, 1.0, order + 1),
        np.linspace(0.0, 1.0, order + 1),
        indexing="ij",
        sparse=True,
    )
    return x1, x2


@cache
def _triangle_control_points(order: int) -> tuple[npt.NDArray, npt.NDArray]:
    x1sq, x2 = _quad_control_points(order)
    x1 = np.empty(np.broadcast(x1sq, x2).shape)
    x1[:, :-1] = x1sq / (1 - x2[:, :-1])
    # Handle NaNs at top of triangle
    x1[0, -1] = 1
    x1[1:, -1] = 1.1
    x1_m = np.ma.masked_greater(x1, 1.0)
    return x1_m, np.ma.array(np.broadcast_to(x2, x1.shape), mask=x1_m.mask)


@overload
def _meshio_line_point_order(points: SliceCoords) -> Iterator[SliceCoord]: ...
@overload
def _meshio_line_point_order(points: Coords) -> Iterator[Coord]: ...
def _meshio_line_point_order(
    points: SliceCoords | Coords,
) -> Iterator[SliceCoord | Coord]:
    """Iterate through points in order expected by meshio and gmsh.

    Input coords are expected to be a 1-D array going from the start
    to the end of the line.

    """
    shape = points.x1.shape
    if len(shape) != 1:
        raise RuntimeError("Points must be in a 1D sequence")
    yield points[0]
    yield points[-1]
    for i in range(1, shape[0] - 1):
        yield points[i]


@overload
def _meshio_quad_point_order(points: SliceCoords) -> Iterator[SliceCoord]: ...
@overload
def _meshio_quad_point_order(points: Coords) -> Iterator[Coord]: ...
def _meshio_quad_point_order(
    points: SliceCoords | Coords,
) -> Iterator[SliceCoord | Coord]:
    """Iterate through points in order expected by meshio and gmsh.

    Input coords are expected to be a 2-D array with indices
    corresponding to rows/columns of points on the quad.

    """
    shape = points.x1.shape
    if len(shape) != 2 or shape[0] != shape[1]:
        raise RuntimeError("Points must be in a square")
    n = shape[0]
    if len(points) == 1:
        yield points.to_coord()
        return
    yield points[0, 0]
    yield points[-1, 0]
    yield points[-1, -1]
    yield points[0, -1]
    if n > 2:
        for i in range(1, n - 1):
            yield points[i, 0]
        for i in range(1, n - 1):
            yield points[-1, i]
        for i in range(n - 2, 0, -1):
            yield points[i, -1]
        for i in range(1, n - 1):
            yield points[0, i]
        yield from _meshio_quad_point_order(
            type(points)(*(x[1:-1, 1:-1] for x in points), points.system)  # type: ignore
        )


@overload
def _meshio_triangle_point_order(points: SliceCoords) -> Iterator[SliceCoord]: ...
@overload
def _meshio_triangle_point_order(points: Coords) -> Iterator[Coord]: ...
def _meshio_triangle_point_order(
    points: SliceCoords | Coords,
) -> Iterator[SliceCoord | Coord]:
    """Iterate through points in order expected by meshio and gmsh.

    Input coords are expected to be a 2-D array with indices
    corresponding to rows/columns of points on the quad.

    """
    if len(points) == 0:
        return
    shape = points.x1.shape
    if len(shape) != 2 or shape[0] != shape[1]:
        raise RuntimeError("Points must be in a square")
    n = shape[0]
    if len(points) == 1:
        yield points.to_coord()
        return
    yield points[0, 0]
    yield points[-1, 0]
    yield points[0, -1]
    if n > 2:
        for i in range(1, n - 1):
            yield points[i, 0]
        for i in range(1, n - 1):
            yield points[n - i - 1, i]
        for i in range(1, n - 1):
            yield points[0, i]
        if n > 3:
            yield from _meshio_triangle_point_order(
                type(points)(*(x[1:-2, 1:-2] for x in points), points.system)  # type: ignore
            )


def _meshio_hex27_point_order(points: Coords) -> Iterator[Coord]:
    """Iterate through points of second-order hex in order expected by meshio.

    Meshio uses a different ordering for points (following that used
    by VTU) in a second-order hex than it does for higher-order ones
    (following that used by gmsh).

    Input coords are expected to be a 3-D array with indices
    corresponding to rows/columns/etc. of points in the hex.

    """
    shape = points.x1.shape
    if shape != (3, 3, 3):
        raise RuntimeError("Points must be in a 3 by 3 by 3 cube")
    # FIXME: Will need to check this
    yield points[0, 0, 0]
    yield points[2, 0, 0]
    yield points[2, 2, 0]
    yield points[0, 2, 0]
    yield points[0, 0, 2]
    yield points[2, 0, 2]
    yield points[2, 2, 2]
    yield points[0, 2, 2]
    yield points[1, 0, 0]
    yield points[2, 1, 0]
    yield points[1, 2, 0]
    yield points[0, 1, 0]
    yield points[1, 0, 2]
    yield points[2, 1, 2]
    yield points[1, 2, 2]
    yield points[0, 1, 2]
    yield points[0, 0, 1]
    yield points[2, 0, 1]
    yield points[2, 2, 1]
    yield points[0, 2, 1]
    yield points[0, 1, 1]
    yield points[2, 1, 1]
    yield points[1, 0, 0]
    yield points[1, 1, 2]
    yield points[1, 1, 0]
    yield points[1, 1, 2]
    yield points[1, 1, 1]


def _meshio_hex_point_order(points: Coords) -> Iterator[Coord]:
    """Iterate through points in order expected by meshio and gmsh.

    Input coords are expected to be a 3-D array with indices
    corresponding to rows/columns/etc. of points in the hex.

    """
    if len(points) == 0:
        return
    shape = points.x1.shape
    if len(shape) != 3 or shape[0] != shape[1] or shape[0] != shape[2]:
        raise RuntimeError("Points must be in a cube")
    n = shape[0]
    if len(points) == 1:
        yield points.to_coord()
        return
    if len(points) == 27:
        # Irritatingly, meshio uses a different order for hexes of 2nd order
        yield from _meshio_hex27_point_order(points)
        return
    yield points[0, 0, 0]
    yield points[-1, 0, 0]
    yield points[-1, -1, 0]
    yield points[0, -1, 0]
    yield points[0, 0, -1]
    yield points[-1, 0, -1]
    yield points[-1, -1, -1]
    yield points[0, -1, -1]
    if n <= 2:
        return
    for coord in [
        lambda i: (i, 0, 0),
        lambda i: (0, i, 0),
        lambda i: (0, 0, i),
        lambda i: (-1, i, 0),
        lambda i: (-1, 0, i),
        lambda i: (n - i - 1, -1, 0),
        lambda i: (-1, -1, i),
        lambda i: (0, -1, i),
        lambda i: (i, 0, -1),
        lambda i: (0, i, -1),
        lambda i: (-1, i, -1),
        lambda i: (n - i - 1, -1, -1),
    ]:
        for i in range(1, n - 1):
            yield points[coord(i)]

    # for i in range(1, n - 1):
    #     yield points[i, 0, 0]
    # for i in range(1, n - 1):
    #     yield points[0, i, 0]
    # for i in range(1, n - 1):
    #     yield points[0, 0, i]
    # for i in range(1, n - 1):
    #     yield points[-1, i, 0]
    # for i in range(1, n - 1):
    #     yield points[-1, 0, i]
    # for i in range(n - 2, 0, -1):
    #     yield points[i, -1, 0]
    # for i in range(1, n - 1):
    #     yield points[-1, -1, i]
    # for i in range(1, n - 1):
    #     yield points[0, -1, i]
    # for i in range(1, n - 1):
    #     yield points[i, 0, -1]
    # for i in range(1, n - 1):
    #     yield points[0, i, -1]
    # for i in range(1, n - 1):
    #     yield points[-1, i, -1]
    # for i in range(n - 2, 0, -1):
    #     yield points[i, -1, -1]

    # FIXME: Will the normals of these be in the right direction?
    yield from _meshio_quad_point_order(
        Coords(*(x[1:-1, 1:-1, 0] for x in points), points.system)  # type: ignore
    )
    yield from _meshio_quad_point_order(
        Coords(*(x[1:-1, 0, 1:-1] for x in points), points.system)  # type: ignore
    )
    yield from _meshio_quad_point_order(
        Coords(*(x[0, 1:-1, 1:-1] for x in points), points.system)  # type: ignore
    )
    yield from _meshio_quad_point_order(
        Coords(*(x[-1, 1:-1, 1:-1] for x in points), points.system)  # type: ignore
    )
    yield from _meshio_quad_point_order(
        Coords(*(x[1:-1, -1, 1:-1] for x in points), points.system)  # type: ignore
    )
    yield from _meshio_quad_point_order(
        Coords(*(x[1:-1, 1:-1, -1] for x in points), points.system)  # type: ignore
    )
    yield from _meshio_hex_point_order(
        Coords(*(x[1:-1, 1:-1, 1:-1] for x in points), points.system)  # type: ignore
    )


def _meshio_prism_point_order(points: Coords) -> Iterator[Coord]:
    """Iterate through points in order expected by meshio and gmsh.

    Input coords are expected to be a 3-D array with indices
    corresponding to rows/columns/etc. of points in the prism. The
    triangular cross section is in the first two coordinates.

    """
    if len(points) == 0:
        return
    shape = points.x1.shape
    if len(shape) != 3 or shape[0] != shape[1] or shape[0] != shape[2]:
        raise RuntimeError("Points must be in a cube")
    n = shape[0]
    if len(points) == 1:
        yield points.to_coord()
        return
    yield points[0, 0, 0]
    yield points[-1, 0, 0]
    yield points[0, -1, 0]
    yield points[0, 0, -1]
    yield points[-1, 0, -1]
    yield points[0, -1, -1]
    if n <= 2:
        return
    for coord in [
        lambda i: (i, 0, 0),
        lambda i: (0, i, 0),
        lambda i: (0, 0, i),
        lambda i: (n - i - 1, i, 0),
        lambda i: (-1, 0, i),
        lambda i: (0, -1, i),
        lambda i: (i, 0, -1),
        lambda i: (0, i, -1),
        lambda i: (n - i - 1, i, -1),
    ]:
        for i in range(1, n - 1):
            yield points[coord(i)]

    # for i in range(1, n - 1):
    #     yield points[i, 0, 0]
    # for i in range(1, n - 1):
    #     yield points[0, i, 0]
    # for i in range(1, n - 1):
    #     yield points[0, 0, i]
    # for i in range(1, n - 1):
    #     yield points[n - i - 1, i, 0]
    # for i in range(1, n - 1):
    #     yield points[-1, 0, i]
    # for i in range(1, n - 1):
    #     yield points[0, -1, i]
    # for i in range(1, n - 1):
    #     yield points[i, 0, -1]
    # for i in range(1, n - 1):
    #     yield points[0, i, -1]
    # for i in range(1, n - 1):
    #     yield points[n - i - 1, i, -1]

    # FIXME: Will the normals of these be in the right direction?
    # FIXME: I've taken my best guess at the ordering of the
    # triangular faces, based on analogy to hex, but not certain it's
    # right.
    yield from _meshio_triangle_point_order(
        Coords(*(x[1:-1, 1:-1, 0] for x in points), points.system)  # type: ignore
    )
    yield from _meshio_quad_point_order(
        Coords(*(x[1:-1, 0, 1:-1] for x in points), points.system)  # type: ignore
    )
    yield from _meshio_quad_point_order(
        Coords(*(x[0, 1:-1, 1:-1] for x in points), points.system)  # type: ignore
    )
    # FIXME: This is the complicated face that is going diagonally through data. Not 100% sure I got it right.
    yield from _meshio_quad_point_order(
        Coords(*(np.flipud(x).diagonal()[1:-1, 1:-1] for x in points), points.system)  # type: ignore
    )
    yield from _meshio_triangle_point_order(
        Coords(*(x[1:-1, 1:-1, -1] for x in points), points.system)  # type: ignore
    )
    yield from _meshio_prism_point_order(
        Coords(*(x[1:-2, 1:-2, 1:-1] for x in points), points.system)  # type: ignore
    )


def _sort_cellset(
    cellset: DefaultDict[str, list[int]], cells_order: Sequence[str]
) -> list[npt.ArrayLike]:
    return [cellset[cell_type] for cell_type in cells_order]


class MeshioData:
    """Manages data for assembling a mesh in the MeshIO format.

    It will store NESO-fame elements as an ordered collection of
    points and return integer IDs for these elements.

    Group
    -----
    public meshio

    """

    _points: list[npt.ArrayLike]
    _cells: DefaultDict[str, list[npt.ArrayLike]]
    _cell_sets: DefaultDict[str, DefaultDict[str, list[int]]]
    _dim_tags: list[tuple[int, int]]
    _next_dim_tag: list[int]

    def __init__(self) -> None:
        self._points = []
        self._cells = DefaultDict(list)
        self._cell_sets = DefaultDict(lambda: DefaultDict(list))
        self._dim_tags = []
        self._next_dim_tag = [1, 1, 1, 1]

    @cache
    def _element_dim_tag(self, element_type: str, layer_id: int) -> tuple[int, int]:
        dim = _ELEMENT_DIMS[element_type]
        tag = self._next_dim_tag[dim]
        # FIXME: This should probably just be += 1, right?
        # self._next_dim_tag[dim] += tag + 1
        self._next_dim_tag[dim] += 1
        return dim, tag

    @coord_cache()
    def point(
        self, coords: SliceCoord | Coord, element_type: str, layer_id: int
    ) -> int:
        """Add a point to the mesh data and return the integer ID for it."""
        pos = (
            coords.to_3d_coord(0.0) if isinstance(coords, SliceCoord) else coords
        ).to_cartesian()
        self._points.append(tuple(pos))
        self._dim_tags.append(self._element_dim_tag(element_type, layer_id))
        return len(self._points) - 1

    @cache
    def poloidal_face(
        self,
        solid: Prism,
        order: int,
        layer_id: int,
        cellsets: frozenset[str] = frozenset(),
    ) -> int:
        """Add a 2D element representing the poloidal cross-section of the prism.

        An integer ID for this element will be returned.

        Curved elements will be represented to the given order of
        accuracy. Caching is used to ensure the same quad, in the same
        layer, represented to the same order will always return the same
        objects. The caching is done based on the locations of the control
        points of the quad and its edges, rather than the identity of the
        quad.

        """
        # FIXME: Will need to have a different cellblock for each layer
        if len(solid.sides) == 3:
            s, t = _triangle_control_points(order)
            shape = _ELEMENT_TYPES[order - 1]["triangle"]
        elif len(solid.sides) == 4:
            s, t = _quad_control_points(order)
            shape = _ELEMENT_TYPES[order - 1]["quad"]
        else:
            raise NotImplementedError(
                "Currently only triangular and rectangular prisms are supported."
            )
        # FIXME: Avoid calling this if order == 1?
        coords = solid.poloidal_map(s, t)
        points = tuple(
            self.point(p, shape, layer_id)
            for p in (
                _meshio_quad_point_order(coords)
                if len(solid.sides) == 4
                else _meshio_triangle_point_order(coords)
            )
        )
        cell_list = self._cells[shape]
        cell_list.append(points)
        cell_id = len(cell_list) - 1
        for setname in cellsets:
            self._cell_sets[setname][shape].append(cell_id)
        return cell_id

    @cache
    def line(
        self,
        curve: NormalisedCurve | AcrossFieldCurve,
        order: int,
        layer_id: int,
        cellsets: frozenset[str] = frozenset(),
    ) -> int:
        """Add a 1D element to the mesh data and return the integer ID for it."""
        shape = _ELEMENT_TYPES[order - 1]["line"]
        points = tuple(
            self.point(p, shape, layer_id)
            for p in _meshio_line_point_order(control_points(curve, order))
        )
        cell_list = self._cells[shape]
        cell_list.append(points)
        cell_id = len(cell_list) - 1
        for setname in cellsets:
            self._cell_sets[setname][shape].append(cell_id)
        return cell_id

    @cache
    def end_shape(
        self,
        shape: EndShape,
        order: int,
        layer_id: int,
        cellsets: frozenset[str] = frozenset(),
    ) -> int:
        """Add a 2D element representing a poloidal face and return the integer ID for it."""
        return 0

    @cache
    def quad(
        self, quad: Quad, order: int, layer_id: int, cellsets: frozenset[str]
    ) -> int:
        """Add a 2D element representing a quad and return the integer ID for it."""
        return 0

    @cache
    def solid(
        self, solid: Prism, order: int, layer_id: int, cellsets: frozenset[str]
    ) -> int:
        """Add a 3D element representing to the mesh and return the integer ID for it."""
        return 0

    def meshio(self) -> meshio.Mesh:
        """Create a meshio mesh object from the stored data."""
        # Each cell block has a unique (per dimension) geometrical
        # tag. This matches the dim_tag of the nodes making up the
        # cells. (There may be dim_tag corresponding to nodes that do
        # not make up any cells.) Each cell block will also have a
        # physical entity ID, which need not be unique (i.e., physical
        # entities can be made up of more than one cell block).

        # In contrast, the meshio cell_sets allow elements of a cell
        # block to be int different cell_sets (and even more than one)
        type_order = list(self._cells)
        return meshio.Mesh(
            np.array(self._points),
            list(self._cells.items()),
            # FIXME: Can I construct some of this stuff from
            # cell_sets? Problem is needing separate nodes for each
            # dimension.
            point_data={"gmsh:dim_tags": self._dim_tags},
            # FIXME: Will need to have a different cellblock for each
            # layer, so repeat for different cell types
            cell_data={
                "gmsh:geometrical": [
                    np.full(len(v), self._element_dim_tag(k, 0)[1])
                    for k, v in self._cells.items()
                ],
                # FIXME: This needs to have different IDs for each layer, cell-type, and boundary/ Can numbers be shared between different types of elements? Appears not; each number must refer to only one cell-block (but is treated seperately for each dimension) Probably best way to handle this is to assemble it from cell_sets
                "gmsh:physical": [
                    np.full(len(v), self._element_dim_tag(k, 0)[1])
                    for k, v in self._cells.items()
                ],
            },
            # FIXME: Do I need gmsh:bounding_entities? Don't think it
            # would be very useful given there are random
            # quads/triangles floating around. Certainly, the way I
            # have stored the boundary data isn't particularly
            # amenable to knowing whether something belongs to a quad
            # or triangle. Not sure it's actually useful anyway.
            cell_sets={
                k: _sort_cellset(cells, type_order)
                for k, cells in self._cell_sets.items()
            },
        )


def meshio_poloidal_elements(mesh: PrismMesh, order: int) -> meshio.Mesh:
    """Create MeshIO mesh for intersection of the mesh with the poloidal plane.

    This can be useful for visualising the underlying mesh from whcih
    the 3D version is extruded.

    Group
    -----
    public meshio

    """
    print("Converting FAME mesh to MeshIO object.")
    layer = mesh.reference_layer
    if issubclass(layer.element_type, Quad):
        raise ValueError("Can not create poloidal mesh for 2D mesh")
    result = MeshioData()
    for element in layer:
        result.poloidal_face(element, order, 0)
    for i, bound in enumerate(layer.boundaries()):
        sets = frozenset({f"Boundary {i}"})
        for quad in bound:
            result.line(quad.shape, order, 0, sets)
    return result.meshio()


def meshio_elements(mesh: PrismMesh, order: int) -> meshio.Mesh:
    """Create a 3D MeshIO object representing the mesh.

    Group
    -----
    public meshio

    """
    print("Converting FAME mesh to MeshIO object.")
    make_element: Callable[[Quad | Prism, int, int, frozenset[str]], int]
    make_bound: Callable[[NormalisedCurve | Quad, int, int, frozenset[str]], int]
    make_interface: Callable[
        [NormalisedCurve | EndShape, int, int, frozenset[str]], int
    ]
    result = MeshioData()
    if issubclass(mesh.reference_layer.element_type, Quad):
        make_element = result.quad
        make_bound = result.line
        make_interface = result.line
    else:
        make_element = result.solid
        make_bound = result.quad
        make_interface = result.end_shape
    for i, layer in enumerate(mesh.layers()):
        sets = frozenset({f"Layer {i}"})
        for element in layer:
            make_element(element, order, i, sets)
        for j, bound in enumerate(layer.boundaries()):
            sets = frozenset({f"Boundary {j}"})
            for b in bound:
                make_bound(b, order, i, sets)
        sets = frozenset(f"Interface {2*i}")
        for f in layer.near_faces():
            make_interface(f, order, i, sets)
        sets = frozenset(f"Interface {2*i + 1}")
        for f in layer.far_faces():
            make_interface(f, order, i, sets)
    return result.meshio()


def write_meshio(
    mesh: Mesh,
    order: int,
    filename: str,
) -> None:
    """Create a Nektar++ MeshGraph object and write it to the disk.

    Parameters
    ----------
    mesh
        The mesh to be converted to Nektar++ format.
    order
        The order to use when representing the elements of the mesh.
    filename
        The name of the file to write the mesh to. The format is
        determined from the extension.

    Group
    -----
    public meshio

    """
    pass


def write_poloidal_mesh(
    mesh: PrismMesh,
    order: int,
    filename: str,
    file_format: str,
) -> None:
    """Create a Nektar++ MeshGraph object for the underlying poloidal mesh.

    This can be useful to visualise the mesh from which the 3D one is extruded.

    Parameters
    ----------
    mesh
        The mesh to be converted to Nektar++ format.
    order
        The order to use when representing the elements of the mesh.
    filename
        The name of the file to write the mesh to.
    file_format
        The meshio mesh format to use for the output.

    Group
    -----
    public meshio

    """
    output = meshio_poloidal_elements(mesh, order)
    output.write(filename, file_format)
