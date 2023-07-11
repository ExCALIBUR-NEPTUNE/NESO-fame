import numpy as np

from neso_fame import (
    field_aligned_2d,
    SliceCoords,
    CoordinateSystem,
    write_nektar,
)
from neso_fame.fields import straight_field


num_nodes = 11
m = field_aligned_2d(
    SliceCoords(
        np.linspace(-100.0, 0.0, num_nodes),
        np.zeros(num_nodes),
        CoordinateSystem.CARTESIAN2D,
    ),
    straight_field(np.pi / 90),
    (0.0, 100.0),
    num_nodes - 1,
    subdivisions=1,
)
write_nektar(m, 2, "test_geometry.xml")
