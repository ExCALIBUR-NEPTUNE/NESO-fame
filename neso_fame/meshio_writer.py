"""Routines to output meshes using the meshio library."""

import warnings
from collections.abc import Iterator, Sequence
from functools import cache
from typing import DefaultDict, TypedDict

import meshio  # type: ignore
import numpy as np
import numpy.typing as npt

from .mesh import (
    AcrossFieldCurve,
    Coord,
    Coords,
    Mesh,
    NormalisedCurve,
    Prism,
    PrismMesh,
    Quad,
    SliceCoord,
    SliceCoords,
    StraightLineAcrossField,
    control_points,
)


class _ElementType(TypedDict):
    line: str
    triangle: str
    quad: str
    prism: str
    hexahedron: str


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


@cache
def _quad_control_points(order: int) -> tuple[npt.NDArray, npt.NDArray]:
    x1, x2 = np.meshgrid(
        np.linspace(0.0, 1.0, order + 1), np.linspace(0.0, 1.0, order + 1), indexing="ij", sparse=True
    )
    return x1, x2


@cache
def _triangle_control_points(order: int) -> tuple[npt.NDArray, npt.NDArray]:
    x1sq, x2 = _quad_control_points(order)
    with warnings.catch_warnings():
        x1 = x1sq / (1 - x2)
    # Handle NaNs at top of triangle
    x1[0, -1] = 1
    x1[1:, -1] = 1.1
    x1_m = np.ma.masked_greater(x1, 1.0)
    return x1_m, np.ma.array(np.broadcast_to(x2, x1.shape), mask=x1_m.mask)


def _gmsh_line_point_order(
    points: SliceCoords | Coords,
) -> Iterator[SliceCoord | Coord]:
    """Iterate through points in order expected by meshio and gmsh."""
    shape = points.x1.shape
    if len(shape) != 1:
        raise RuntimeError("Points must be in a 1D sequence")
    yield points[0]
    yield points[-1]
    for i in range(1, shape[0] - 1):
        yield points[i]


def _gmsh_quad_point_order(
    points: SliceCoords | Coords,
) -> Iterator[SliceCoord | Coord]:
    """Iterate through points in order expected by meshio and gmsh."""
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
        for i in range(n - 2, 0, -1):
            yield points[0, i]
        yield from _gmsh_quad_point_order(
            type(points)(*(x[1:-1, 1:-1] for x in points), points.system)
        )


def _gmsh_triangle_point_order(
    points: SliceCoords | Coords,
) -> Iterator[SliceCoord | Coord]:
    """Iterate through points in order expected by meshio and gmsh."""
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
        for i in range(n - 2, 0, -1):
            yield points[0, i]
        if n > 3:
            yield from _gmsh_triangle_point_order(
                type(points)(*(x[1:-2, 1:-2] for x in points), points.system)
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

    def __init__(self) -> None:
        self._points = []
        self._cells = DefaultDict(list)
        self._cell_sets = DefaultDict(lambda: DefaultDict(list))

    @cache
    def point(self, coords: SliceCoord | Coord, layer_id: int) -> int:
        """Add a point to the mesh data and return the integer ID for it."""
        pos = (
            coords.to_3d_coord(0.0) if isinstance(coords, SliceCoord) else coords
        ).to_cartesian()
        self._points.append(tuple(pos))
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
        coords = solid.poloidal_map(s, t)
        points = tuple(
            self.point(p, layer_id)
            for p in (
                _gmsh_quad_point_order(coords)
                if len(solid.sides) == 4
                else _gmsh_triangle_point_order(coords)
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
        """Add a 1D element to the mesh data and returns the integer ID for it."""
        points = tuple(
            self.point(p, layer_id)
            for p in _gmsh_line_point_order(control_points(curve, order))
        )
        shape = _ELEMENT_TYPES[order - 1]["line"]
        cell_list = self._cells[shape]
        cell_list.append(points)
        cell_id = len(cell_list) - 1
        for setname in cellsets:
            self._cell_sets[setname][shape].append(cell_id)
        return cell_id

    def meshio(self) -> meshio.Mesh:
        """Create a meshio mesh object from the stored data."""
        type_order = list(self._cells)
        return meshio.Mesh(
            self._points,
            list(self._cells.items()),
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
