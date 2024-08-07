# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Opvious SDK"
copyright = "2023, Opvious"
author = "Opvious"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autodoc_type_aliases = {
    "ConstraintBody": "ConstraintBody",
    "DimensionArgument": "DimensionArgument",
    "ExpressionLike": "ExpressionLike",
    "KeyItem": "KeyItem",
    "ObjectiveBody": "ObjectiveBody",
    "Projection": "Projection",
    "Quantifiable": "Quantifiable",
    "Quantification": "Quantification",
    "Quantified": "Quantified",
    "SolveOutcome": "SolveOutcome",
    "Specification": "Specification",
    "Target": "Target",
    "TensorArgument": "TensorArgument",
    "TensorLike": "TensorLike",
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
