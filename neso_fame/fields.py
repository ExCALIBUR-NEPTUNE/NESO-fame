"""Some functions to produce callables defning magnetic fields.

"""

import numpy as np
import numpy.typing as npt

from neso_fame.mesh import CoordinateSystem, FieldTrace, SliceCoord, SliceCoords


def _cylindrical_distance(x1_start, angle, x3):
    a = x1_start
    b = np.tan(angle)
    d = a + b * x3
    c = np.sqrt(d**2 + b**2)
    c0 = np.sqrt(a**2 + b**2)
    if np.abs(angle) < 1e-8:
        return a * x3
    return (
        c * d / (2 * b)
        + 0.5 * b * np.log(c + d)
        - (c0 * a / (2 * b) + 0.5 * b * np.log(c0 + a))
    )


def straight_field(angle_x1=0.0, angle_x2=0.0) -> "FieldTrace":
    """Returns a field trace corresponding to straight field lines
    slanted at `angle_x1` above the direction of extrusion into the
    first coordinate direction and `angle_x2` above the direction of
    extrusion into the second coordinate direction.

    Returns
    -------
    :obj:`~neso_fame.mesh.FieldTrace`

    Group
    -----
    field

    """

    def trace(
        start: SliceCoord, perpendicular_coord: npt.ArrayLike
    ) -> tuple[SliceCoords, npt.NDArray]:
        """Returns a trace for a straight field line."""
        x1 = start.x1 + perpendicular_coord * np.tan(angle_x1)
        x2 = start.x2 + perpendicular_coord * np.tan(angle_x2)
        return SliceCoords(x1, x2, start.system), _cylindrical_distance(
            start.x1, angle_x1, perpendicular_coord
        ) / np.cos(
            angle_x2
        ) if start.system == CoordinateSystem.CYLINDRICAL else np.asarray(
            perpendicular_coord
        ) / (
            np.cos(angle_x1) * np.cos(angle_x2)
        )

    return trace


def curved_field(weight=0.0) -> "FieldTrace":
    """Returns a field trace corresponding to straight field lines
    slanted at `angle` above the direction of extrusion into the first
    coordinate direction. The argument corresponds to the weight which
    should be given to the nonlinear component of the curve.

    Warning
    -------
    The distance calculation for this is wrong.

    Returns
    -------
    :obj:`~neso_fame.mesh.FieldTrace`

    Group
    -----
    field

    """

    def trace(
        start: SliceCoord, perpendicular_coord: npt.ArrayLike
    ) -> tuple[SliceCoords, npt.NDArray]:
        """Returns a trace for a straight field line."""
        x3 = np.asarray(perpendicular_coord)
        x1 = start.x1 + weight * x3**2  # - weight * x3
        x2 = np.asarray(start.x2)
        # FIXME: The expression for distance here is wrong
        return SliceCoords(x1, x2, start.system), np.sign(x3) * (
            np.sqrt(4 * weight**2 * x3**2 + 1) ** 3 / (12 * weight**2)
            - 1 / (12 * weight**2)
        )

    return trace
