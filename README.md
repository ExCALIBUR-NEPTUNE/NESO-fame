# NESO-fame

The Field-Aligned Mesh Extrusion (fame) tool generates meshes for
Tokamaks with elements aligned with magnetic field lines. Currently
this work is highly experimental.

**Note:** Unlike other repositories in the NEPTUNE project, this one
is licensed under GPLv3, as it depends on
[hypnotoad](https://github.com/boutproject/hypnotoad), which is
similarly licensed.

**Note:** This code uses Nektar++ Python bindings which have not been
merged into the master branch. They are accessible from
https://gitlab.nektar.info/cmacmackin/nektar on branch
`cmacmackin/generate-nonconformal-meshes`.

## Running

You can try generating a simple 2D mesh by running `python test_mesh.py`
(ensuring that NekPy is in the PYTHONPATH). You can then convert this
to a file suitable for visualisation by using

`FieldConvert input_nektar_mesh.xml output_paraview_mesh.vtu:vtu:highorder`

Note that this requires Nektar++ to have been compiled with VTK
support. If it has not, approximate meshes can still be produced using

`FieldConvert input_nektar_mesh.xml output_paraview_mesh.vtu`

In that case, additional elements will be added to approximate the
higher-order shape of the element. This results in larger files and
uglier visualisations, though.
