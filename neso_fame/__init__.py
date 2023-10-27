"""Functions and scripts to generate field-aligned meshes.

This package produces meshes for finite-element plasma-physics
simulations. It comes with routines to convert them to the Nektar++
mesh format, but support can be added for others.

"""

from .generators import field_aligned_2d
from .mesh import CoordinateSystem, SliceCoords
from .nektar_writer import write_nektar

__all__ = ["CoordinateSystem", "field_aligned_2d", "SliceCoords", "write_nektar"]
