from functools import cache
import itertools

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk

from generate_mesh import Mesh, Coord, Quad


class Points:
    def __init__(self, size: int):
        self.points_locations = np.array((size, 3))
        self.counter = 0

    @cache
    def get_point_index(self, coord: Coord) -> int:
        cart = coord.to_cartesian()
        n = self.counter
        self.points_locations[n, 0] = cart.x1
        self.points_locations[n, 1] = cart.x2
        self.points_locations[n, 2] = cart.x3
        self.counter += 1
        return n

    def make_vtk_points(self) -> vtk.vtkPoints:
        points = vtk.vtkPoints()
        points.SetData(numpy_to_vtk(self.points_locations.flatten()))
        return points


def vtk_unstructured_grid(mesh: Mesh[Quad], order: int) -> vtk.vtkUnstructuredGrid:
    grid = vtk.vtkUnstructuredGrid()
    n = mesh.num_unique_control_points(order)
    points = Points(n)
    # Going to need:
    #  - xyz3d: (n, 3) array containing point coordinates
    #  - cells: array of sets of indices corresponding to points making up a cell
    #  - cell_locations: array of indices for start of each cell in previous array
    #  - cell_types: IDs for the type of cell being used (easy)
    #
    # Doesn't all have to be in that form. Can build up objects one by one.
    cells = vtk.vtkCellArray()
    cell_types = np.empty((n,), "B")
    cell_types[:] = vtk.VTK_LAGRANGE_QUADRILATERAL
    cell_locations = np.empty((n,))
    npoints = (order + 1) ** 2
    barycentric_index = [0, 0, 0, 0]
    for i, element in enumerate(
        itertools.chain.from_iterable(layer.elements() for layer in mesh.layers())
    ):
        quad = vtk.vtkLagrangeQuadrilateral()
        quad.GetPointIds().SetNumberOfIds(npoints)
        quad.GetPoints.SetNumberOfPoints(npoints)
        control_points = element.control_points(order)
        quad_ids = quad.GetPointIds()
        for j in range(npoints):
            # We compute the barycentric index of the point...
            quad.ToBarycentricIndex(j, barycentric_index)
            vtk_point_id = points.get_point_index(
                control_points[barycentric_index[0], barycentric_index[1]]
            )
            quad_ids.SetId(j, vtk_point_id)
        # Fixme: is this the right value to store in cell_locations?
        cell_locations[i] = cells.InsertNextCell(quad)

    grid.SetPoints(points.make_vtk_points())
    grid.SetCells(
        numpy_to_vtk(cell_types),
        numpy_to_vtk(cell_locations, deep=1, array_type=vtk.VTK_ID_TYPE),
        cells,
    )
    return grid


def write_unstructured_grid(grid: vtk.vtkUnstructuredGrid, filename: str) -> None:
    """Filename should use the .vtu extension."""
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(filename)
    writer.SetInput(grid)
    writer.Write()
