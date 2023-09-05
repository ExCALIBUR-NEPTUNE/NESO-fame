Usage
=====

You can generate simple meshes from the command-line using the
``fame-simple`` command. This generates either a 2D or a 3D mesh,
aligned with a straight, uniform magnetic field. By default, the field
is parallel to the x1 direction (for 2D meshes) or the x3 direction
(for 3D), but you can configure it to be offset by some angle.

Your mesh is made up of "layers" in the field-aligned
direction. Elements within a layer are all conformal with each other
and all follow the direction of the magnetic field. There will be a
nonconformal interface with adjacent layers. Each layer is identical,
apart from its position on the x1- or x3-axis. Each layer may be
further subdivided along the x1/x3 axis but, as stated above, these
elements will be conformal.

.. note::
   The python executable you call must be one that can find NekPy on
   the PYTHONPATH.

2D Meshes
---------

You can try generating a simple 10 by 10 2D mesh on the domain
:math:`x_1 \in [0, 1], x_2 \in [0, 1]`, aligned to a field shifted by
1 degree from the x1-direction, by running::
  
  fame-simple 2d --nx1 10 --nx2 10 --angle 1 simple_mesh.xml

This will save the mesh to ``simple_mesh.xml`` in the Nektar++
format. By default, the number of layers in the x1-direction is the
same as the number of elements. This can be adjusted by using the
``--layers`` option. For example, if you want your elements to be
conformal you can set the number of layers to 1::
  
  fame-simple 2d --nx1 10 --nx2 10 --angle 1 --layers 1 simple_mesh.xml

Run ``fame-simple 2d --help`` to find out about additional options for
configuring your mesh.

3D Meshes
---------
The process of generating a 3D mesh is very similar. For example, to
produce a mesh with the following properties
- 10 by 20 by 8 elements
- domain :math:`x_1 \in [0, 100], x_2 \in [0, 200], x_3 \in [0, 80]`
- aligned to a field rotate by 3 degrees away from the x3-axis and towards the x1-axis
- with four nonconformal layers

you should run::

  fame-simple 3d --nx1 10 --nx2 20 --nx3 8 --x1-extent 0 100 \
              --x2-extent 0 200--x3-extent 0 80 --angle1 3 \
              --layers 4 3d_mesh.xml

Again, you can get more information about the options available by
running ``fame-simple 3d --help``.
              
Visualising your Meshes
-----------------------
You can then convert the
Nektar++ mesh file to the VTK format to make it suitable for
visualisation::

  FieldConvert test_geometry.xml test_geometry.vtu:vtu:highorder

Note that this requires Nektar++ to have been compiled with VTK
support. If it has not, approximate meshes can still be produced using::

  FieldConvert test_geometry.xml test_geometry.vtu

In that case, additional elements will be added to approximate the
higher-order shape of the element. This results in larger files and
uglier visualisations, though.
