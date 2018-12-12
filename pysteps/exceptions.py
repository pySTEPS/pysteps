# -*- coding: utf-8 -*-

# Custom pySteps exceptions


class MissingOptionalDependency(Exception):
    """Raised when an optional dependency is needed but not found."""
    pass
