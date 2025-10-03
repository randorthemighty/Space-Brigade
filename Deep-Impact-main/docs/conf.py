import sys
import os

# Add the project root directory to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.curdir, "..")))

# Project information
project = "Deep Impact"
author = "The Kuiper Team"

# Sphinx extensions
extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon"]

# Source file suffix and master document
source_suffix = ".rst"
master_doc = "index"

# Patterns to exclude
exclude_patterns = ["_build"]

# Auto-documentation settings
autoclass_content = "both"
