# NESO-fame
![tests](https://github.com/ExCALIBUR-NEPTUNE/NESO-fame/actions/workflows/test.yml/badge.svg?branch=main)
![linting](https://github.com/ExCALIBUR-NEPTUNE/NESO-fame/actions/workflows/linting.yml/badge.svg?branch=main)
[![codecov](https://codecov.io/gh/ExCALIBUR-NEPTUNE/NESO-fame/graph/badge.svg?token=74F5EHALWB)](https://codecov.io/gh/ExCALIBUR-NEPTUNE/NESO-fame)

The Field-Aligned Mesh Extrusion (fame) tool generates meshes for
Tokamaks with elements aligned with magnetic field lines. Currently
this work is highly experimental. [Full
documentation](https://excalibur-neptune.github.io/NESO-fame/) is
hosted on GitHub Pages.


## Running
You can generate simple 2D and 3D meshes with the `fame-simple`
command. Run `fame-simple --help`, `fame-simple 2d --help`, and
`fame-simple 3d --help` for more information.

In order to produce a mesh for a tokamak you will need data on its
equilibrium magnetic field in the G-EQDSK format. Start by opening
that file with the
[hypnotoad](https://github.com/boutproject/hypnotoad) GUI and playing
with various settings until you have a poloidal mesh you are happy
with. *Make sure that you do **not** tick the "Non-Orthogonal"
option.* Save your settings to a YAML file called something like
`mesh_settings.yml`. Then use `fame-hypnotoad` to generate the 3D
mesh. Use the `--help` option for information on the configurations
available.


If you would like to
visualise a Nektar++ mesh in file `mesh.xml`, run

```
FieldConvert test_geometry.xml test_geometry.vtu:vtu:highorder
```

Note that this requires Nektar++ to have been compiled with VTK
support. If it has not been, approximate meshes can be produced using

```
FieldConvert test_geometry.xml test_geometry.vtu
```

## License
NESO-fame is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

Unlike other NEPTUNE/NESO projects, NESO-fame does not use a
permissive license, because it depends on
[hypnotoad](https://github.com/boutproject/hypnotoad) which has strong
copyleft.
