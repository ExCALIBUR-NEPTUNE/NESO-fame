import numpy as np
import numpy.typing as npt

from neso_fame import (
    field_aligned_2d,
    SliceCoords,
    CoordinateSystem,
    write_nektar,
)
from neso_fame.fields import straight_field


num_nodes = 4
m = field_aligned_2d(
    SliceCoords(
        np.linspace(0, 1, num_nodes), np.zeros(num_nodes), CoordinateSystem.Cartesian
    ),
    straight_field(0.0),
    (0.0, 0.5 * np.pi),
    1,
    subdivisions=4,
)
write_nektar(m, 2, "test_conformal_geometry.xml", False)