# -*- coding: utf-8 -*-

# Custom pySteps exceptions


class MissingOptionalDependency(Exception):
    """Raised when an optional dependency is needed but not found."""
    pass


class DataModelError(Exception):
    """Raised when a file is not compilant with the Data Information Model."""
    pass

class UnsupportedSomercProjection(Exception):
    """
    Raised when the Swiss Oblique Mercator (somerc) projection is passed to cartopy.
    Necessary since cartopy doesn't support the Swiss projection.
    TODO: remove once the somerc projection is supported in cartopy.
    """
    pass