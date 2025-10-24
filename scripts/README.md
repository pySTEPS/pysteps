# Scripts

This directory contains utility scripts for the pysteps project.

## generate_requirements.py

Generates requirements.txt, requirements_dev.txt, environment.yml, and environment_dev.yml 
from pyproject.toml (the single source of truth for dependencies).

**Usage:**

```bash
python scripts/generate_requirements.py
```

**When to run:**

- After modifying dependencies in `pyproject.toml`
- Before committing dependency changes
- As part of the release preparation process

**Output:**

- `requirements.txt` - pip requirements for core dependencies
- `requirements_dev.txt` - pip requirements including dev dependencies  
- `environment.yml` - conda environment for core dependencies
- `environment_dev.yml` - conda environment including dev dependencies

**Note:** The generated files should be committed to the repository for backwards 
compatibility and convenience, but should never be edited manually.
