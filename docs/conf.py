# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
import pathlib
import sys
from datetime import datetime

import l3m

sys.path.insert(0, str(pathlib.Path(__file__).parent / "_ext"))


needs_sphinx = "5.0"

autodoc_mock_imports = ["megablocks"]

project = "L3M"
author = "Apple"
copyright = f"2024-{datetime.now():%Y}, {author}"
release = l3m.__version__
version = l3m.__version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "typed_returns",
]
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
    "fvcore": ("https://detectron2.readthedocs.io/en/latest/", None),
    "omegaconf": ("https://omegaconf.readthedocs.io/en/latest/", None),
}

master_doc = "index"

# autodoc
autosummary_generate = True
autodoc_typehints = "description"
always_document_param_types = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_book_theme"
