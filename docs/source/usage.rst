Usage
=====

For now this is just a library with some example scripts. Proper
scripts will be written at a later date.

You can try generating a simple 2D mesh by running::
  
  python example_mesh.py

.. note::
   The python executable you call must be one that can find NekPy on
   the PYTHONPATH.

This will create a nonconformal, field-aligned mesh. You can then
convert the Nektar++ mesh file to the VTK format to make it suitable for visualisation::

  FieldConvert test_geometry.xml test_geometry.vtu:vtu:highorder

Note that this requires Nektar++ to have been compiled with VTK
support. If it has not, approximate meshes can still be produced using::

  FieldConvert test_geometry.xml test_geometry.vtu

In that case, additional elements will be added to approximate the
higher-order shape of the element. This results in larger files and
uglier visualisations, though.

You can create a similar field-aligned but conformal mesh by running::

  python example_conformal.py
