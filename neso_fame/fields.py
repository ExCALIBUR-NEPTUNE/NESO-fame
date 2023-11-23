"""Some functions to produce callables defning magnetic fields."""

import numpy as np
import numpy.typing as npt

from neso_fame.mesh import CoordinateSystem, FieldTrace, SliceCoord, SliceCoords


def straight_field(angle_x1: float = 0.0, angle_x2: float = 0.0) -> "FieldTrace":
    """Return a field trace corresponding to straight field lines.

    The field lines will be slanted at `angle_x1` above the direction
    of extrusion into the first coordinate direction and `angle_x2`
    above the direction of extrusion into the second coordinate
    direction.

    Returns
    -------
    :obj:`~neso_fame.mesh.FieldTrace`

    Group
    -----
    field

    """

    def trace(
        start: SliceCoord, x3: npt.ArrayLike, start_weight: float = 0.0
    ) -> tuple[SliceCoords, npt.NDArray]:
        """Return a trace for a straight field line."""
        if start.system == CoordinateSystem.CYLINDRICAL:
            raise ValueError("Cylindrical coordinates not supported.")
        b = 1 - start_weight
        perp_coord = np.asarray(x3)
        x1 = start.x1 + b * perp_coord * np.tan(angle_x1)
        x2 = start.x2 + b * perp_coord * np.tan(angle_x2)
        return SliceCoords(x1, x2, start.system), perp_coord * np.sqrt(
            1 + b**2 / np.cos(angle_x1) ** 4 + b**2 / np.cos(angle_x2) ** 4
        )

    return trace
