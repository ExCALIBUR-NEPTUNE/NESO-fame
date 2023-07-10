# NESO-fame
![tests](https://github.com/ExCALIBUR-NEPTUNE/NESO-fame/actions/workflows/test.yml/badge.svg?branch=main)
![linting](https://github.com/ExCALIBUR-NEPTUNE/NESO-fame/actions/workflows/linting.yml/badge.svg?branch=main)

The Field-Aligned Mesh Extrusion (fame) tool generates meshes for
Tokamaks with elements aligned with magnetic field lines. Currently
this work is highly experimental. [Full
documentation](https://excalibur-neptune.github.io/NESO-fame/) is
hosted on GitHub Pages.

**Note:** This code uses Nektar++ Python bindings which have not been
merged into the master branch. They are accessible from
https://gitlab.nektar.info/cmacmackin/nektar on branch
`cmacmackin/generate-nonconformal-meshes`.

## Running
You can try generating a simple 2D mesh by running `python example_mesh.py`
(ensuring that NekPy is in the PYTHONPATH). You can then convert this
to a file suitable for visualisation by using

`FieldConvert test_geometry.xml test_geometry.vtu:vtu:highorder`

Note that this requires Nektar++ to have been compiled with VTK
support. If it has not, approximate meshes can still be produced using

`FieldConvert test_geometry.xml test_geometry.vtu`

In that case, additional elements will be added to approximate the
higher-order shape of the element. This results in larger files and
uglier visualisations, though.

## License
NESO-fame is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

Unlike other NEPTUNE/NESO projects, NESO-fame does not use a
permissive license, because it depends on
[hypnotoad](https://github.com/boutproject/hypnotoad) which has strong
copyleft.
