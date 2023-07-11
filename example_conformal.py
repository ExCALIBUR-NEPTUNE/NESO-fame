import numpy as np

from neso_fame import CoordinateSystem, SliceCoords, field_aligned_2d, write_nektar
from neso_fame.fields import straight_field

num_nodes = 11
m = field_aligned_2d(
    SliceCoords(
        np.linspace(-1e2, 0, num_nodes),
        np.zeros(num_nodes),
        CoordinateSystem.CARTESIAN2D,
    ),
    straight_field(np.pi / 90),
    (0.0, 100.0),
    1,
    subdivisions=num_nodes - 1,
)
write_nektar(m, 1, "test_conformal_geometry.xml", False)
