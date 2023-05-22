import numpy as np
import numpy.typing as npt

from neso_fame import (
    field_aligned_2d,
    SliceCoords,
    CoordinateSystem,
    write_nektar,
)
from neso_fame.fields import curved_field


num_nodes = 5
m = field_aligned_2d(
    SliceCoords(
        np.linspace(0, 1, num_nodes), np.zeros(num_nodes), CoordinateSystem.Cartesian
    ),
    # straight_field(np.pi/6),
    curved_field(0.1),
    (0.0, 1.0),
    (0.0, 1.0),
    4,
    2,
)
write_nektar(m, 2, "test_geometry.xml")