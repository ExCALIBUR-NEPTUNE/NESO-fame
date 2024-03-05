Visualising Meshes
==================

You can then convert Nektar++ meshes to the VTK format to make them suitable for
visualisation. For example, if you have a mesh in the file ``test_geometry.xml``::

  FieldConvert test_geometry.xml test_geometry.vtu:vtu:highorder

Note that this requires Nektar++ to have been compiled with VTK
support. If it has not, approximate meshes can still be produced using::

  FieldConvert test_geometry.xml test_geometry.vtu

In that case, additional elements will be added to approximate the
higher-order shape of the element. This results in larger files and
uglier visualisations, though.

You can then open the result ``.vtu`` file with `ParaView
<https://www.paraview.org/>`_ in order to display it. To see
individual elements of the mesh, select to view it as either "Surface
With Edges" or "Wireframe". Examples of visualations using ParaView
are given throughout this tutorial.
