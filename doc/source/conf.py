# Pysteps-docs sphinx configuration
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import doc_utils

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
copyright = f"2018-{datetime.now():%Y}, pysteps developers"
author = "PySteps developers"

release = doc_utils.current_release()
version = release.lstrip("v").rstrip().split("-")[0]  # The short X.Y version.

is_run_in_read_the_docs = "READTHEDOCS" in os.environ
sphinx_gallery_conf = {
    "examples_dirs": "../../examples",
    "gallery_dirs": "auto_examples",
    "filename_pattern": r"/*\.py",
    "plot_gallery": not is_run_in_read_the_docs,
}

# -- Options for HTML output ----------------------------------------------
html_theme = "sphinx_book_theme"
html_title = ""
html_context = {
    "github_user": "pySTEPS",
    "github_repo": "pysteps",
    "github_version": "master",
    "doc_path": "doc",
}

html_theme_options = {
    "repository_url": "https://github.com/pySTEPS/pysteps",
    "repository_branch": "master",
    "path_to_docs": "doc/source",
    "use_edit_page_button": True,
    "use_repository_button": True,
    "use_issues_button": True,
}

html_logo = "../_static/pysteps_logo.png"
html_static_path = ["../_static"]
html_css_files = ["../_static/pysteps.css"]
html_domain_indices = True
autosummary_generate = True
htmlhelp_basename = "pystepsdoc"

# -- Download artifacts with rendered examples in RTD ----------------------------------
if is_run_in_read_the_docs:
    my_artifact_downloader = doc_utils.ArtifactDownloader(
        "pySTEPS/pysteps", prefix="auto_examples-for-"
    )

    DOC_ROOT_DIR = Path(__file__).parent / "../"
    token = os.environ.get("RTD_GITHUB_TOKEN")  # never print this!
    my_artifact_downloader.download_artifact(
        token,
        DOC_ROOT_DIR / "source/auto_examples",
        retry=3,
        tar_filename="auto_examples.tar",
    )
