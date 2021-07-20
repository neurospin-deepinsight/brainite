# -*- coding: utf-8 -*-
##########################################################################
# pysphinxdoc - Copyright (C) AGrigis, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# Imports
import sys
import os
import datetime
import subprocess
from distutils.version import LooseVersion
import sphinx
import pysphinxdoc
from unittest.mock import MagicMock


class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        print(cls, name)
        mock = MagicMock()
        mock().get_window_data.return_value = "{}"
        return mock
MOCK_MODULES = ["visdom"]
sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)


installdir = os.path.abspath("../..")
env = os.environ
if "PYTHONPATH" in env:
    env["PYTHONPATH"] = env["PYTHONPATH"] + ":" + installdir
else:
    env["PYTHONPATH"] = installdir
cmd = ["sphinxdoc", "-v 2", "-p",  installdir, "-n", "brainite", "-o", "..",
       "-i", "brainite"]
subprocess.check_call(cmd, env=env)
sys.path.insert(0, installdir)


if LooseVersion(sphinx.__version__) < LooseVersion("1"):
    raise RuntimeError("Need sphinx >= 1 for autodoc to work correctly.")
if LooseVersion(sphinx.__version__) < LooseVersion("1.8"):
    sphinx_math = "sphinx.ext.pngmath"
else:
    sphinx_math = "sphinx.ext.imgmath"

def skip(app, what, name, obj, would_skip, options):
    if name == "__init__":
        return False
    return would_skip

def setup(app):
    app.connect("autodoc-skip-member", skip)

# -- General configuration --------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
sphinx_dirname = os.path.dirname(pysphinxdoc.__file__)
sys.path.insert(0, os.path.join(sphinx_dirname, "sphinxext"))

# Add any Sphinx extension module names here, as strings.
# They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    sphinx_math,
    "sphinx.ext.ifconfig",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "numpy_ext.numpydoc",
    "custom_ext.hidden_code_block",
    "custom_ext.hidden_technical_block",
    "custom_ext.link_to_block",
    "sphinx_gallery.gen_gallery"]

# Configure gallery
sphinx_gallery_conf = {
    "doc_module": "brainite",
    "backreferences_dir": os.path.join("brainite", "gallery"),
    "examples_dirs": os.path.join(os.pardir, "examples"),
    "gallery_dirs": "auto_gallery"}

# Remove some numpy-linked warnings
numpydoc_show_class_members = False

# generate autosummary even if no references
autosummary_generate = True

# Add any paths that contain templates here, relative to this directory.
templates_path = [
    os.path.join(sphinx_dirname, "templates"),
    os.path.join("generated", "_templates")]

# The suffix of source filenames.
source_suffix = ".rst"

# The encoding of source files.
# source_encoding = 'utf-8'

# The master toctree document.
master_doc = "index"

# General information about the project.
project = u"brainite"
copyright = u"""{0}, brainite developers <antoine.grigis@cea.fr>""".format(
    datetime.date.today().year)

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = "0.0.0"
# The full version, including alpha/beta/rc tags.
release = version

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
# language = None

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
# today = ''
# Else, today_fmt is used as the format for a strftime call.
# today_fmt = ''

# List of documents that shouldn't be included in the build.
# unused_docs = []

# List of directories, relative to source directory, that shouldn't be searched
# for source files.
exclude_patterns = [
    "examples",
    templates_path[0],
    templates_path[1],
    os.path.join("scikit-learn", "static", "ML_MAPS_README.rst")]

# The reST default role (used for this markup: `text`) to use for all
# documents.
# default_role = None

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
# add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
# show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# A list of ignored prefixes for module index sorting.
# modindex_common_prefix = []


# -- Options for HTML output ------------------------------------------------


# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# The theme to use for HTML and HTML Help pages.  Major themes that come with
# Sphinx are currently 'default' and 'sphinxdoc'.
html_theme = "azmind"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    "oldversion": False,
    "collapsiblesidebar": True,
    "google_analytics": True,
    "surveybanner": False,
    "sprintbanner": True}

# Add any paths that contain custom themes here, relative to this directory.
html_theme_path = [
    os.path.join(sphinx_dirname, "themes")]

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = "brainite"

# A shorter title for the navigation bar.  Default is the same as html_title.
html_short_title = "brainite"

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
# html_logo = ""

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
# html_favicon = None

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = [
    "_static"
]

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
# html_last_updated_fmt = ''

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
# html_use_smartypants = True

# Custom sidebar templates, maps document names to template names.
# html_sidebars = {}

# Additional templates that should be rendered to pages, maps page names to
# template names.
# html_additional_pages = {}

# If false, no module index is generated.
html_use_modindex = False

# If false, no index is generated.
html_use_index = False

# If true, the index is split into individual pages for each letter.
# html_split_index = False

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = False

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
# html_use_opensearch = ''

# If nonempty, this is the file name suffix for HTML files (e.g. ".xhtml").
# html_file_suffix = ''

# Output file base name for HTML help builder.
htmlhelp_basename = "brainite"


# -- Options for LaTeX output -----------------------------------------------

# The paper size ('letter' or 'a4').
# latex_paper_size = 'letter'

# The font size ('10pt', '11pt' or '12pt').
# latex_font_size = '10pt'

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass
# [howto/manual]).
latex_documents = [
    ("index", "brainite.tex", "brainite Documentation", """brainite developers""",
     "manual"),
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
# latex_logo = None

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
# latex_use_parts = False

# Additional stuff for the LaTeX preamble.
# latex_preamble = ''

# Documents to append as an appendix to all manuals.
# latex_appendices = []

# If false, no module index is generated.
# latex_use_modindex = True


# Example configuration for intersphinx: refer to the Python standard library.
# intersphinx_mapping = {'http://docs.python.org/': None}


# -- Options for Texinfo output ---------------------------------------------

autodoc_default_flags = ["members", "undoc-members"]
