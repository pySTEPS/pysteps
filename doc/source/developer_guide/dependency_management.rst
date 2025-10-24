Dependency Management
=====================

Overview
--------

As of version 1.18.1, pysteps uses **pyproject.toml** as the single source of truth 
for all project dependencies following `PEP 621 <https://peps.python.org/pep-0621/>`_ 
standards. This ensures reproducible builds and prevents unpredictable failures due to 
dependency upgrades.

All version constraints are explicitly defined to balance stability and compatibility:

- **Minimum versions** ensure required features are available
- **Maximum versions** prevent breaking changes from future releases
- All dependencies are pinned with version ranges (e.g., ``>=1.24.0,<3.0``)

Source of Truth: pyproject.toml
--------------------------------

The ``pyproject.toml`` file contains:

1. **Core dependencies** (``[project.dependencies]``)
   
   Required packages for basic pysteps functionality:
   
   - numpy
   - opencv-python
   - pillow
   - pyproj
   - scipy
   - matplotlib
   - jsmin
   - jsonschema
   - netCDF4

2. **Optional dependencies** (``[project.optional-dependencies]``)
   
   Organized by feature group:
   
   - ``performance``: dask, pyfftw
   - ``geo``: cartopy, rasterio
   - ``io``: h5py, pygrib
   - ``analysis``: scikit-image, scikit-learn, pandas, PyWavelets
   - ``all``: All optional dependencies combined
   - ``dev``: Development tools and all optional dependencies
   - ``docs``: Documentation building requirements

Generated Files
---------------

The following files are **auto-generated** from ``pyproject.toml`` and should 
**NOT be edited manually**:

- ``requirements.txt`` - pip requirements for core dependencies
- ``requirements_dev.txt`` - pip requirements including dev dependencies
- ``environment.yml`` - conda environment for core dependencies
- ``environment_dev.yml`` - conda environment including dev dependencies

These files are maintained for backwards compatibility and convenience.

Updating Dependencies
---------------------

To update dependencies in the project:

1. Edit the appropriate section in ``pyproject.toml``
2. Run the generation script::

    python scripts/generate_requirements.py

3. The script will regenerate all requirements files automatically

Example: Adding a New Dependency
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To add a new core dependency:

1. Edit ``pyproject.toml``::

    [project]
    dependencies = [
        # ... existing dependencies ...
        "new-package>=1.0.0,<2.0",
    ]

2. Regenerate files::

    python scripts/generate_requirements.py

3. Commit both ``pyproject.toml`` and the generated files

Installing pysteps
------------------

With pip
~~~~~~~~

Install core dependencies::

    pip install .

Install with optional features::

    pip install .[all]          # All optional dependencies
    pip install .[geo]          # Only geospatial features
    pip install .[performance]  # Only performance enhancements

For development::

    pip install -e .[dev]

With conda
~~~~~~~~~~

Using the generated environment file::

    conda env create -f environment.yml
    conda activate pysteps

For development::

    conda env create -f environment_dev.yml
    conda activate pysteps_dev

Dependency Version Policy
--------------------------

Version constraints follow these principles:

1. **Minimum version**: Set to a version known to work, typically from when 
   the dependency was added or last tested
   
2. **Maximum version**: Set to the next major version to prevent breaking changes
   Example: If minimum is 1.24.0, maximum is typically <3.0 or <2.0
   
3. **Python compatibility**: Tested on Python 3.11, 3.12, and 3.13

4. **Regular updates**: Dependencies should be reviewed and updated periodically
   to benefit from bug fixes and new features while maintaining stability

Automated Dependency Updates
-----------------------------

Pysteps uses GitHub's Dependabot to automatically monitor dependencies for:

- Security vulnerabilities
- Available updates
- Compatibility with newer versions

Dependabot is configured in ``.github/dependabot.yml`` to:

- Check for updates monthly
- Group minor and patch updates together
- Automatically create pull requests for updates
- Monitor both Python dependencies and GitHub Actions

When Dependabot creates a pull request:

1. Review the changes and check the changelog of updated packages
2. Ensure CI tests pass
3. Merge if everything looks good
4. The generated requirements files will be automatically updated via CI

To manually enable or configure dependabot, edit ``.github/dependabot.yml``.

Troubleshooting
---------------

If you encounter dependency conflicts:

1. Verify you're using a supported Python version (3.11-3.13)
2. Try creating a fresh virtual environment
3. Check if mixing pip and conda installations caused conflicts
4. Review the dependency versions in ``pyproject.toml``

Common Issues
~~~~~~~~~~~~~

**"No matching distribution found"**
  One of the dependencies may not be available for your platform.
  Check the package's PyPI page for platform availability.

**Version conflicts**
  If you see version conflicts, ensure you're not mixing packages from
  different sources (pip vs conda). Use one package manager consistently.

**Build failures**
  Some packages like pygrib may require system libraries. Install
  required system dependencies using your system package manager.

Migration from Old System
--------------------------

Prior to version 1.18.1, dependencies were scattered across multiple files.
The migration involved:

1. Consolidating all dependencies into ``pyproject.toml``
2. Adding explicit version constraints
3. Creating generation scripts for backwards compatibility
4. Updating ``setup.py`` to use ``pyproject.toml``

Old dependency files are now auto-generated and should not be edited manually.
