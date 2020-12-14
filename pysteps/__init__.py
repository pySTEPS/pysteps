import json
import os
import stat
import sys
import warnings

from jsmin import jsmin
from jsonschema import Draft4Validator

# import subpackages
from . import cascade
from . import datasets
from . import decorators
from . import downscaling
from . import exceptions
from . import extrapolation
from . import io
from . import motion
from . import noise
from . import nowcasts
from . import postprocessing
from . import timeseries
from . import utils
from . import verification as vf
from . import visualization as plt


def _get_config_file_schema():
    """
    Return the path to the parameters file json schema.
    """
    module_file = _decode_filesystem_path(__file__)
    return os.path.join(os.path.dirname(module_file), "pystepsrc_schema.json")


def _fconfig_candidates_generator():
    """
    Configuration files candidates generator.

    See :py:func:~config_fname for more details.
    """

    yield os.path.join(os.getcwd(), "pystepsrc")

    try:
        pystepsrc = os.environ["PYSTEPSRC"]
    except KeyError:
        pass
    else:
        yield pystepsrc
        yield os.path.join(pystepsrc, "pystepsrc")

    if os.name == "nt":
        # Windows environment
        env_variable = "USERPROFILE"
        subdir = "pysteps"
    else:
        # UNIX like
        env_variable = "HOME"
        subdir = ".pysteps"

    try:
        pystepsrc = os.environ[env_variable]
    except KeyError:
        pass
    else:
        yield os.path.join(pystepsrc, subdir, "pystepsrc")

    module_file = _decode_filesystem_path(__file__)
    yield os.path.join(os.path.dirname(module_file), "pystepsrc")
    yield None


# Function adapted from matplotlib's *matplotlib_fname* function.
def config_fname():
    """
    Get the location of the config file.

    Looks for pystepsrc file in the following order:
    - $PWD/pystepsrc: Looks for the file in the current directory
    - $PYSTEPSRC: If the system variable $PYSTEPSRC is defined and it points
    to a file, it is used..
    - $PYSTEPSRC/pystepsrc: If $PYSTEPSRC points to a directory, it looks for
    the pystepsrc file inside that directory.
    - $HOME/.pysteps/pystepsrc (unix and Mac OS X) :
    If the system variable $HOME is defined, it looks
    for the configuration file in this path.
    - $USERPROFILE/pysteps/pystepsrc (windows only): It looks for the
    configuration file in the pysteps directory located user's home directory.
    - Lastly, it looks inside the library in pysteps/pystepsrc for a
    system-defined copy.
    """

    file_name = None
    for file_name in _fconfig_candidates_generator():

        if file_name is not None:
            if os.path.exists(file_name):
                st_mode = os.stat(file_name).st_mode
                if stat.S_ISREG(st_mode) or stat.S_ISFIFO(st_mode):
                    return file_name

            # Return first candidate that is a file,
            # or last candidate if none is valid
            # (in that case, a warning is raised at startup by `rc_params`).

    return file_name


def _decode_filesystem_path(path):
    if not isinstance(path, str):
        return path.decode(sys.getfilesystemencoding())
    else:
        return path


class _DotDictify(dict):
    """
    Class used to recursively access dict via attributes as well
    as index access.
    This is introduced to maintain backward compatibility with older pysteps
    configuration parameters implementations.

    Code adapted from:
    https://stackoverflow.com/questions/3031219/recursively-access-dict-via-attributes-as-well-as-index-access

    Credits: `Curt Hagenlocher`_

    .. _`Curt Hagenlocher`: https://stackoverflow.com/users/533/curt-hagenlocher
    """

    def __setitem__(self, key, value):
        if isinstance(value, dict) and not isinstance(value, _DotDictify):
            value = _DotDictify(value)
        super().__setitem__(key, value)

    def __getitem__(self, key):
        value = super().__getitem__(key)
        if isinstance(value, dict) and not isinstance(value, _DotDictify):
            value = _DotDictify(value)
            super().__setitem__(key, value)
        return value

    __setattr__, __getattr__ = __setitem__, __getitem__


rcparams = dict()


def load_config_file(params_file=None, verbose=False, dryrun=False):
    """
    Load the pysteps configuration file. The configuration parameters are available
    as a DotDictify instance in the `pysteps.rcparams` variable.

    Parameters
    ----------
    params_file: str
        Path to the parameters file to load. If `params_file=None`, it looks
        for a configuration file in the default locations.
    verbose: bool
        Print debugging information. False by default.
        This flag is overwritten by the silent_import=False in the
        pysteps configuration file.
    dryrun: bool
        If False, perform a dry run that does not update the `pysteps.rcparams`
        attribute.

    Returns
    -------
    rcparams: _DotDictify
        Configuration parameters loaded from file.
    """

    global rcparams

    if params_file is None:
        # Load default configuration
        params_file = config_fname()

        if params_file is None:
            warnings.warn(
                "pystepsrc file not found." + "The defaults parameters are left empty",
                category=ImportWarning,
            )

            _rcparams = dict()
            return

    with open(params_file, "r") as f:
        _rcparams = json.loads(jsmin(f.read()))

    if (not _rcparams.get("silent_import", False)) or verbose:
        print("Pysteps configuration file found at: " + params_file + "\n")

    with open(_get_config_file_schema(), "r") as f:
        schema = json.loads(jsmin(f.read()))
        validator = Draft4Validator(schema)

        error_msg = "Error reading pystepsrc file."
        error_count = 0
        for error in validator.iter_errors(_rcparams):
            error_msg += "\nError in " + "/".join(list(error.path))
            error_msg += ": " + error.message
            error_count += 1
        if error_count > 0:
            raise RuntimeError(error_msg)

    _rcparams = _DotDictify(_rcparams)

    if not dryrun:
        rcparams = _rcparams

    return _rcparams


# Load default configuration
load_config_file()

# After the sub-modules are loaded, register the discovered importers plugin.
io.interface.discover_importers()
