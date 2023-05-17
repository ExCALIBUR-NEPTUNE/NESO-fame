import numpy as np
import numpy.typing as npt

from neso_fame.mesh import (
    FieldTrace,
    C,
    SliceCoord,
    SliceCoords,
)


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


def curved_field(weight=0.0) -> "FieldTrace[C]":
    """Returns a field trace corresponding to straight field lines
    slanted at `angle` above the direction of extrusion into the first
    coordinate direction."""

    def trace(
        start: SliceCoord[C], perpendicular_coord: npt.ArrayLike
    ) -> tuple[SliceCoords[C], npt.NDArray]:
        """Returns a trace for a straight field line."""
        x3 = np.asarray(perpendicular_coord)
        x1 = start.x1 + weight * x3**2 - weight * x3
        x2 = np.asarray(start.x2)
        return SliceCoords(x1, x2, start.system), np.sign(x3) * (
            np.sqrt(4 * weight**2 * x3**2 + 1) ** 3 / (12 * weight**2)
            - 1 / (12 * weight**2)
        )

    return trace
