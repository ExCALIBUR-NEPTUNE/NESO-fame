# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------

project = "NESO-fame"
copyright = "2023, Crown Copyright"
author = "Chris MacMackin"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_immaterial",
    "sphinx.ext.intersphinx",
    "sphinx_immaterial.apidoc.python.apigen",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_immaterial"
html_theme_options = {
    "site_url": "https://ExCALIBUR-NEPTUNE.github.io/NESO-fame/",
    "repo_url": "https://github.com/ExCALIBUR-NEPTUNE/NESO-fame",
    "edit_uri": "blob/main/docs/source/",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_css_files = ["custom.css"]
html_static_path = ["_static"]

html_title = "NESO-fame"

# Sphinx Immaterial theme options
html_theme_options = {
    "icon": {
        "repo": "fontawesome/brands/github",
    },
    "site_url": "https://ExCALIBUR-NEPTUNE.github.io/NESO-fame",
    "repo_url": "https://github.com/ExCALIBUR-NEPTUNE/NESO-fame",
    "repo_name": "ExCALIBUR-NEPTUNE/NESO-fame",
    "edit_uri": "blob/main/docs/source",
    "globaltoc_collapse": False,
    "features": [
        # "navigation.expand",
        "navigation.tabs",
        # "toc.integrate",
        # "navigation.sections",
        # "navigation.instant",
        # "header.autohide",
        "navigation.top",
        "navigation.tracking",
        "toc.follow",
        "toc.sticky",
        "content.tabs.link",
        "announce.dismiss",
    ],
    "palette": [
        {
            "media": "(prefers-color-scheme: light)",
            "scheme": "default",
            "toggle": {
                "icon": "material/weather-night",
                "name": "Switch to dark mode",
            },
        },
        {
            "media": "(prefers-color-scheme: dark)",
            "scheme": "slate",
            "toggle": {
                "icon": "material/weather-sunny",
                "name": "Switch to light mode",
            },
        },
    ],
}

html_last_updated_fmt = ""
html_use_index = True
html_domain_indices = True


# -- Extension configuration -------------------------------------------------

# Create hyperlinks to other documentation
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "hypnotoad": ("https://hypnotoad.readthedocs.io/en/latest/", None),
}

autodoc_typehints = "signature"
# autodoc_typehints_description_target = "documented"
# autodoc_typehints_format = "short"
autodoc_type_aliases = {
    "np.typing.NDArray": "numpy.typing.NDArray",
    "npt.NDArray": "numpy.typing.NDArray",
    "npt.ArrayLike": "numpy.typing.ArrayLike",
    "QuadMesh": "neso_fame.mesh.QuadMesh",
    "HexMesh": "neso_fame.mesh.HexMesh",
    "Mesh": "neso_fame.mesh.Mesh",
    "FieldTrace": "neso_fame.mesh.FieldTrace",
    "NormalisedCurve": "neso_fame.mesh.NormalisedCurve",
    "AcrossFieldCurve": "neso_fame.mesh.AcrossFieldCurve",
    "NektarLayer": "neso_fame.nektar_writer.NektarLayer",
    "HypnoMesh": "hypnotoad.core.mesh.Mesh",
    "HypnoMeshRegion": "hypnotoad.core.mesh.MeshRegion",
    "Integrand": "neso_fame.hypnotoad_interface.Integrand",
    "IntegratedFunction": "neso_fame.hypnotoad_interface.IntegratedFunction",
}

# FIXME: Need to improve display of ArrayLike

# -- Sphinx Immaterial configs -------------------------------------------------

# Python apigen configuration
python_apigen_modules = {
    "neso_fame.mesh": "api/autogen/mesh/",
    "neso_fame.fields": "api/autogen/fields/",
    "neso_fame.generators": "api/autogen/generators/",
    "neso_fame.nektar_writer": "api/autogen/nektar_writer/",
    "neso_fame.hypnotoad_interface": "api/autogen/hypnotoad/",
    "neso_fame.offset": "api/autogen/offset",
}
python_apigen_default_groups = [
    ("class:.*", "Classes"),
    ("data:.*", "Variables"),
    ("function:.*", "Functions"),
    ("classmethod:.*", "Class methods"),
    ("method:.*", "Methods"),
    (r"method:.*\.[A-Z][A-Za-z,_]*", "Constructors"),
    (r"method:.*\.__[A-Za-z,_]*__", "Special methods"),
    (r"method:.*\.__(init|new)__", "Constructors"),
    ("property:.*", "Properties"),
]
python_apigen_default_order = [
    ("class:.*", 10),
    ("data:.*", 11),
    ("function:.*", 12),
    ("classmethod:.*", 40),
    ("method:.*", 50),
    (r"method:.*\.[A-Z][A-Za-z,_]*", 20),
    (r"method:.*\.__[A-Za-z,_]*__", 28),
    (r"method:.*\.__(init|new)__", 20),
    ("property:.*", 60),
]
python_apigen_case_insensitive_filesystem = False
python_apigen_show_base_classes = True
python_transform_type_annotations_pep604 = True

# Python domain directive configuration
python_type_aliases = autodoc_type_aliases
python_module_names_to_strip_from_xrefs = [
    "collections.abc",
    "NekPy.SpatialDomains._SpatialDomains",
] + list(python_apigen_modules)

# General API configuration
object_description_options = [
    ("py:.*", {"include_rubrics_in_toc": True}),
]


current_module = None
filtered_docs = {
    "__setattr__",
    "__delattr__",
    "Connectivity",
    "NektarQuadGeomElements",
    "NodePair",
    "NektarOrderedQuadGeomElements",
    "NektarHexGeomElements",
    "HypnoMesh",
    "HypnoMeshRegion",
}


def autodoc_skip_member(app, what, name, obj, skip, options):
    """
    Instruct autodoc to skip imported members
    """
    global current_module
    if name == "__name__":
        current_module = obj

    if skip:
        # Continue skipping things Sphinx already wants to skip
        return skip

    if name in filtered_docs:
        return True

    if getattr(obj, "__name__", name) != name:
        # Heuristic for type aliases
        return False

    if hasattr(obj, "__module__"):
        return obj.__module__ != current_module

    return skip


def setup(app):
    app.connect("autodoc-skip-member", autodoc_skip_member)
