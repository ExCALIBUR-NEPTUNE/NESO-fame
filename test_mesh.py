import numpy as np
import numpy.typing as npt

from neso_fame.generate_mesh import (
    field_aligned_2d,
    FieldTrace,
    C,
    SliceCoord,
    SliceCoords,
    CoordinateSystem,
)
from neso_fame.nektar_writer import write_nektar


def straight_field(angle=0.0) -> "FieldTrace[C]":
    """Returns a field trace corresponding to straight field lines
    slanted at `angle` above the direction of extrusion into the first
    coordinate direction."""

    def trace(
        start: SliceCoord[C], perpendicular_coord: npt.ArrayLike
    ) -> tuple[SliceCoords[C], npt.NDArray]:
        """Returns a trace for a straight field line."""
        x1 = start.x1 + perpendicular_coord * np.tan(angle)
        x2 = np.asarray(start.x2)
        x3 = np.asarray(perpendicular_coord)
        return SliceCoords(x1, x2, start.system), x3

    return trace


# Have function to generate 2D mesh in Python form


num_nodes = 5

m = field_aligned_2d(
    SliceCoords(
        np.linspace(0, 1, num_nodes), np.zeros(num_nodes), CoordinateSystem.Cartesian
    ),
    straight_field(),
    (0.0, 1.0),
    (0.0, 1.0),
    4,
    2,
)
write_nektar(m, 1, "test_geometry.xml")
