import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from jsmin import jsmin

# -- Global variables ------------------------------------------------
DOCS_DIR = Path(__file__).parent / ".."
PROJECT_ROOT_DIR = DOCS_DIR / ".."
EXAMPLES_GALLERY_REPOSITORY = "https://github.com/pySTEPS/pysteps_tutorials.git"

sys.path.insert(1, os.path.abspath(str(PROJECT_ROOT_DIR)))

# -- General configuration ------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "numpydoc",
    "sphinxcontrib.bibtex",
    "sphinx_gallery.gen_gallery",
]

bibtex_bibfiles = ["references.bib"]
templates_path = ["_templates"]
source_suffix = ".rst"
master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]
todo_include_todos = False
project = "pysteps"
copyright = "2020, PySteps developers"
author = "PySteps developers"


def get_version():
    """Returns project version as string from 'git describe' command."""
    from subprocess import check_output

    _version = check_output(["git", "describe", "--tags", "--always"])
    if _version:
        _version = _version.decode("utf-8")
    else:
        _version = "X.Y"

    return _version.lstrip("v").rstrip().split("-")[0]


release = get_version()

# Sphinx-Gallery options
sphinx_gallery_conf = {
    "examples_dirs": "../../examples",  # path to your example scripts
    "gallery_dirs": "user_guide/examples_gallery",  # path where to save gallery generated examples
    "filename_pattern": r"/*\.py",  # Include all the files in the examples dir
    'plot_gallery': False,  # Do not execute the examples
}


# -- Read the Docs build --------------------------------------------------
def set_root():
    fn = os.path.abspath(os.path.join("..", "..", "pysteps", "pystepsrc"))
    with open(fn, "r") as f:
        rcparams = json.loads(jsmin(f.read()))

    for key, value in rcparams["data_sources"].items():
        new_path = os.path.join("..", "..", "pysteps-data", value["root_path"])
        new_path = os.path.abspath(new_path)

        value["root_path"] = new_path

    fn = os.path.abspath(os.path.join("..", "..", "pystepsrc.rtd"))
    with open(fn, "w") as f:
        json.dump(rcparams, f, indent=4)


def pull_example_gallery_from_external_repo():
    global EXAMPLES_GALLERY_REPOSITORY

    # Default to the "latest" branch, containing the latest rendered notebooks.
    rtd_version = os.environ.get("READTHEDOCS_VERSION", "latest")

    if rtd_version == "stable":
        try:
            rtd_version = subprocess.check_output(
                ["git", "describe", "--abbrev=0", "--tags"], universal_newlines=True
            ).strip()
        except subprocess.CalledProcessError:
            rtd_version = "latest"

    print(f"\nRTD Version: {rtd_version}\n")

    with tempfile.TemporaryDirectory() as work_dir:
        cmd = (
            f"git clone {EXAMPLES_GALLERY_REPOSITORY} --depth 1 --branch {rtd_version} {work_dir}"
        )
        subprocess.check_output(cmd.split(" "))

        examples_gallery_dir = DOCS_DIR / "source/examples_gallery"
        shutil.rmtree(examples_gallery_dir, ignore_errors=True)

        examples_dir = PROJECT_ROOT_DIR / "examples"
        shutil.rmtree(examples_dir, ignore_errors=True)

        work_dir = Path(work_dir)
        example_gallery_dir_in_remote = work_dir / "docs/examples_gallery"
        shutil.copytree(example_gallery_dir_in_remote, examples_gallery_dir)

        examples_dir_in_remote = work_dir / "examples"
        shutil.copytree(examples_dir_in_remote, examples_dir)


pull_example_gallery_from_external_repo()

# -- Options for HTML output ----------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["../_static"]
html_css_files = ["../_static/pysteps.css"]
html_domain_indices = True
autosummary_generate = True

# -- Options for HTMLHelp output ------------------------------------------
htmlhelp_basename = "pystepsdoc"

# -- Options for LaTeX output ---------------------------------------------

# This hack is taken from numpy (https://github.com/numpy/numpy/blob/master/doc/source/conf.py).
latex_preamble = r"""
    \usepackage{amsmath}
    \DeclareUnicodeCharacter{00A0}{\nobreakspace}
    
    % In the parameters section, place a newline after the Parameters
    % header
    \usepackage{expdlist}
    \let\latexdescription=\description
    \def\description{\latexdescription{}{} \breaklabel}
    
    % Make Examples/etc section headers smaller and more compact
    \makeatletter
    \titleformat{\paragraph}{\normalsize\py@HeaderFamily}%
                {\py@TitleColor}{0em}{\py@TitleColor}{\py@NormalColor}
    \titlespacing*{\paragraph}{0pt}{1ex}{0pt}
    \makeatother
    
    % Fix footer/header
    \renewcommand{\chaptermark}[1]{\markboth{\MakeUppercase{\thechapter.\ #1}}{}}
    \renewcommand{\sectionmark}[1]{\markright{\MakeUppercase{\thesection.\ #1}}}
"""

latex_elements = {
    "papersize": "a4paper",
    "pointsize": "10pt",
    "preamble": latex_preamble
}

latex_domain_indices = False

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, "pysteps.tex", "pysteps Reference", author, "manual"),
]

# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "pysteps", "pysteps Reference", [author], 1)]

# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "pysteps",
        "pysteps Reference",
        author,
        "pysteps",
        "One line description of project.",
        "Miscellaneous",
    ),
]
