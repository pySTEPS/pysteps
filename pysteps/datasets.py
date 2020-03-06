"""
pysteps.datasets
================

Utilities to download the pysteps data and to create a default pysteps configuration
file pointing to that data.

.. autosummary::
    :toctree: ../generated/

    download_pysteps_data
    create_default_pystepsrc
    info
    load_dataset
"""

import json
import os
import sys
from distutils.dir_util import copy_tree
from logging.handlers import RotatingFileHandler
from tempfile import NamedTemporaryFile, TemporaryDirectory
from urllib import request
from zipfile import ZipFile

import time
from datetime import datetime
from jsmin import jsmin

import pysteps
from pysteps import io
from pysteps.exceptions import DirectoryNotEmpty
from pysteps.utils import conversion

_precip_events = {
    "fmi": "201609281445",
    "fmi2": "201705091045",
    "mch": "201505151545",
    "mch2": "201607112045",
    "mch3": "201701310945",
    "opera": "201808241800",
    "knmi": "201008260000",
    "bom": "201806161000",
}

_data_sources = {
    "fmi": "Finish Meteorological Institute",
    "mch": "MeteoSwiss",
    "bom": "Australian Bureau of Meteorology",
    "knmi": "Royal Netherlands Meteorological Institute",
    "opera": "OPERA",
}


# Include this function here to avoid a dependency on pysteps.__init__.py
def _decode_filesystem_path(path):
    if not isinstance(path, str):
        return path.decode(sys.getfilesystemencoding())
    else:
        return path


def info():
    """
    Describe the available datasets in the pysteps example data.

    >>> from pysteps import datasets
    >>> datasets.info()
    """
    print("\nAvailable datasets:\n")

    print(f"{'Case':<8} {'Event date':<22} {'Source':<45}\n")

    for case_name, case_date in _precip_events.items():
        _source = "".join([i for i in case_name if not i.isdigit()])
        _source = _data_sources[_source]

        _case_date = datetime.strptime(_precip_events[case_name], "%Y%m%d%H%M")
        _case_date = datetime.strftime(_case_date, "%Y-%m-%d %H:%M UTC")

        print(f"{case_name:<8} {_case_date:<22} {_source:<45}")


class ShowProgress(object):
    """
    Class used to report the download progress.

    Usage::

    >>> from urllib import request
    >>> pbar = ShowProgress()
    >>> request.urlretrieve("http://python.org/", "/tmp/index.html", pbar)
    >>> pbar.end()

    """

    def __init__(self, bar_length=20):
        self.prev_msg_width = 0
        self.init_time = None
        self.total_size = None
        self._progress_bar_length = bar_length

    def _clear_line(self):
        sys.stdout.write("\b" * self.prev_msg_width)
        sys.stdout.write("\r")

    def _print(self, msg):
        self.prev_msg_width = len(msg)
        sys.stdout.write(msg)

    def __call__(self, count, block_size, total_size):

        self._clear_line()

        downloaded_size = count * block_size / (1024 ** 2)

        if self.total_size is None and total_size > 0:
            self.total_size = total_size / (1024 ** 2)

        if count == 0:
            self.init_time = time.time()
            progress_msg = ""
        else:
            if self.total_size is not None:
                progress = count * block_size / total_size
                block = int(round(self._progress_bar_length * progress))

                elapsed_time = time.time() - self.init_time
                eta = (elapsed_time / progress - elapsed_time) / 60

                bar_str = "#" * block + "-" * (
                    self._progress_bar_length - block
                )

                progress_msg = (
                    f"Progress: [{bar_str}]"
                    f"({downloaded_size:.1f} Mb)"
                    f" - Time left: {int(eta):d}:{int(eta * 60)} [m:s]"
                )

            else:
                progress_msg = (f"Progress: ({downloaded_size:.1f} Mb)"
                                f" - Time left: unknown")

        self._print(progress_msg)

    @staticmethod
    def end(message="Download complete"):
        sys.stdout.write("\n" + message + "\n")


def download_pysteps_data(dir_path, force=True):
    """
    Download pysteps data from github.

    Parameters
    ----------
    dir_path: str
        Path to directory where the psyteps data will be placed.

    force : bool
        If the destination directory exits and force=False, the DirectoryNotEmpty
        exception if raised.
        If force=True, the data will we downloaded in the destination directory and may
        override existing files.

    """
    tmp_file = NamedTemporaryFile()

    # Check if directory exists but is not empty
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        if os.listdir(dir_path) and not force:
            raise DirectoryNotEmpty(
                dir_path + "is not empty.\n"
                           "Set force=True force the extraction of the files."
            )
    else:
        os.makedirs(dir_path)

    # NOTE:
    # The http response from github can either contain Content-Length (size of the file)
    # or use chunked Transfer-Encoding.
    # If Transfer-Encoding is chunked, then the Content-Length is not available since
    # the content is dynamically generated and we can't know the length a pr_iori easily.
    pbar = ShowProgress()
    request.urlretrieve(
        "https://github.com/pySTEPS/pysteps-data/archive/master.zip",
        tmp_file.name,
        pbar,
    )
    pbar.end()

    with ZipFile(tmp_file.name, "r") as zip_obj:
        tmp_dir = TemporaryDirectory()

        # Extract all the contents of zip file in the temp directory
        common_path = os.path.commonprefix(zip_obj.namelist())

        zip_obj.extractall(tmp_dir.name)

        copy_tree(os.path.join(tmp_dir.name, common_path), dir_path)


def create_default_pystepsrc(pysteps_data_dir, config_dir=None, file_name="pystepsrc"):
    """
    Create a default configuration file pointing to the pysteps data directory.

    If the configuration file already exists, it backup the existing file by appending
    to the filename the extensions '.1', '.2', up to '.5.'.
    A maximum of 5 files are kept. .2, up to app.log.5.

    File rotation is implemented for the backup files.
    For example, if the default configuration filename is 'pystepsrc' and the files
    pystepsrc, pystepsrc.1, pystepsrc.2, etc. exist, they are renamed to pystepsrc.1,
    pystepsrc.2, pystepsrc.2, etc. respectively. Finally, after the existing files are
    backed up, the new configuration file is written.
    
    Parameters
    ----------

    pysteps_data_dir : str
        Path to the directory with the pysteps data.

    config_dir : str
        Destination directory for the configuration file.
        Default values: $HOME/.pysteps (unix and Mac OS X)
        or $USERPROFILE/pysteps (windows).
        The directory is created if it does not exists.

    file_name : str
        Configuration file name. `pystepsrc` by default.

    Returns
    -------

    dest_path : str
        Configuration file path.
    """

    pysteps_lib_root = os.path.dirname(
        _decode_filesystem_path(pysteps.__file__)
    )

    # Load the library built-in configuration file
    with open(os.path.join(pysteps_lib_root, "pystepsrc"), "r") as f:
        rcparams_json = json.loads(jsmin(f.read()))

    for key, value in rcparams_json["data_sources"].items():
        value["root_path"] = os.path.abspath(
            os.path.join(pysteps_data_dir, value["root_path"])
        )

    if config_dir is None:
        home_dir = os.path.expanduser("~")
        if os.name == "nt":
            subdir = "pysteps"
        else:
            subdir = ".pysteps"
        config_dir = os.path.join(home_dir, subdir, file_name)

    if not os.path.isdir(config_dir):
        os.makedirs(config_dir)

    dest_path = os.path.join(config_dir, file_name)

    # Backup existing configuration files if it exists and rotate previous backups
    if os.path.isfile(dest_path):
        RotatingFileHandler(dest_path, backupCount=6).doRollover()

    with open(dest_path, "w") as f:
        json.dump(rcparams_json, f, indent=4)

    return dest_path


def load_dataset(case="fmi", frames=14):
    """
    Load a sequence of radar composites from the pysteps example data.

    To print the available datasets run

    >>> from pysteps import datasets
    >>> datasets.info()

    This function load by default 14 composites, corresponding to a 1h and 10min
    time window.
    For example, the first two composites can be used to obtain the motion field of
    the precipitation pattern, while the remaining twelve composites can be used to
    evaluate the quality of our forecast.

    Calling this function requires the pysteps-data installed, otherwise an exception
    is raised. To install the pysteps example data check the `example_data` section.

    Parameters
    ----------

    case : str
        Case to load.

    frames : int
        Number composites (radar images).
        Max allowed value: 24
        Default: 14

    Returns
    -------

    rainrate : array-like
        Precipitation data in mm/h. Dimensions: [time, lat, lon]

    metadata : dict
        The metadata observations attributes.

    timestep : number
        Time interval between composites in minutes.
    """

    if frames > 24:
        raise ValueError("The number of frames should be smaller than 25")

    case = case.lower()

    case_date = datetime.strptime(_precip_events[case], "%Y%m%d%H%M")

    source = "".join([i for i in case if not i.isdigit()])
    data_source = pysteps.rcparams.data_sources[source]

    # Find the input files from the archive
    file_names = io.archive.find_by_date(
        case_date,
        data_source["root_path"],
        data_source["path_fmt"],
        data_source["fn_pattern"],
        data_source["fn_ext"],
        data_source["timestep"],
        num_prev_files=0,
        num_next_files=frames - 1,
    )

    if None in file_names[0]:
        raise FileNotFoundError(f"Error loading {case} case. Some files are missing.")

    # Read the radar composites
    importer = io.get_method(data_source["importer"], "importer")
    importer_kwargs = data_source["importer_kwargs"]
    reflectivity, _, metadata = io.read_timeseries(
        file_names, importer, **importer_kwargs
    )

    # Convert to rain rate
    precip, metadata = conversion.to_rainrate(reflectivity, metadata)

    return precip, metadata, data_source["timestep"]
