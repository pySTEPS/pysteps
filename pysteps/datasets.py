# -*- coding: utf-8 -*-
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
import gzip
import json
import os
import shutil
import sys
import time
from datetime import datetime, timedelta
from distutils.dir_util import copy_tree
from logging.handlers import RotatingFileHandler
from tempfile import NamedTemporaryFile, TemporaryDirectory
from urllib import request
from urllib.error import HTTPError
from zipfile import ZipFile

from jsmin import jsmin

import pysteps
from pysteps import io
from pysteps.exceptions import DirectoryNotEmpty
from pysteps.utils import conversion

# "event name" , "%Y%m%d%H%M"
_precip_events = {
    "fmi": "201609281445",
    "fmi2": "201705091045",
    "mch": "201505151545",
    "mch2": "201607112045",
    "mch3": "201701310945",
    "opera": "201808241800",
    "knmi": "201008260000",
    "bom": "201806161000",
    "mrms": "201906100000",
}

_data_sources = {
    "fmi": "Finish Meteorological Institute",
    "mch": "MeteoSwiss",
    "bom": "Australian Bureau of Meteorology",
    "knmi": "Royal Netherlands Meteorological Institute",
    "opera": "OPERA",
    "mrms": "NSSL's Multi-Radar/Multi-Sensor System",
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

    def __call__(self, count, block_size, total_size, exact=True):

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

                bar_str = "#" * block + "-" * (self._progress_bar_length - block)

                if exact:
                    downloaded_msg = (
                        f"({downloaded_size:.1f} Mb / {self.total_size:.1f} Mb)"
                    )
                else:
                    downloaded_msg = (
                        f"(~{downloaded_size:.0f} Mb/ {self.total_size:.0f} Mb)"
                    )

                progress_msg = (
                    f"Progress: [{bar_str}]"
                    + downloaded_msg
                    + f" - Time left: {int(eta):d}:{int(eta * 60)} [m:s]"
                )

            else:
                progress_msg = (
                    f"Progress: ({downloaded_size:.1f} Mb)" f" - Time left: unknown"
                )

        self._print(progress_msg)

    @staticmethod
    def end(message="Download complete"):
        sys.stdout.write("\n" + message + "\n")


def download_mrms_data(dir_path, initial_date, final_date, timestep=2, nodelay=False):
    """
    Download a small dataset with 6 hours of the NSSL's Multi-Radar/Multi-Sensor
    System ([MRMS](https://www.nssl.noaa.gov/projects/mrms/)) precipitation
    product (grib format).

    All the available files in the archive in the indicated time period
    (`initial_date` to `final_date`) are downloaded.
    By default, the timestep between files downloaded is 2 min.
    If the `timestep` is exactly divisible by 2 min, the immediately lower
    multiple is used. For example, if  `timestep=5min`, the value is lowered to
    4 min.

    Note
    ----
    To reduce the load on the archive's server, an internal delay of 5 seconds
    every 30 files downloaded is implemented.
    This delay can be disabled by setting `nodelay=True`.


    Parameters
    ----------
    dir_path: str
        Path to directory where the MRMS data is be placed.
        If None, the default location defined in the pystepsrc file is used.
        The files are archived following the folder structure defined in
        the pystepsrc file.
        If the directory exists existing MRMS files may be overwritten.
    initial_date: datetime
        Beginning of the date period.
    final_date: datetime
        End of the date period.
    timestep: int or timedelta
        Timestep between downloaded files in minutes.
    nodelay: bool
        Do not implement a 5-seconds delay every 30 files downloaded.
    """

    if dir_path is None:
        data_source = pysteps.rcparams.data_sources["mrms"]
        dir_path = data_source["root_path"]

    if not isinstance(timestep, (int, timedelta)):
        raise TypeError(
            "'timestep' must be an integer or a timedelta object."
            f"Received: {type(timestep)}"
        )

    if isinstance(timestep, int):
        timestep = timedelta(seconds=timestep * 60)

    if timestep.total_seconds() < 120:
        raise ValueError(
            "The time step should be greater than 2 minutes."
            f"Received: {timestep.total_seconds()}"
        )

    _remainder = timestep % timedelta(seconds=120)
    timestep -= _remainder

    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

    if nodelay:

        def delay(_counter):
            return 0

    else:

        def delay(_counter):
            if _counter >= 30:
                _counter = 0
                time.sleep(5)
            return _counter

    archive_url = "https://mtarchive.geol.iastate.edu"
    print(f"Downloading MRMS data from {archive_url}")

    current_date = initial_date

    counter = 0
    while current_date <= final_date:

        counter = delay(counter)

        sub_dir = os.path.join(dir_path, datetime.strftime(current_date, "%Y/%m/%d"))

        if not os.path.isdir(sub_dir):
            os.makedirs(sub_dir)

        # Generate files URL from https://mtarchive.geol.iastate.edu
        dest_file_name = datetime.strftime(
            current_date, "PrecipRate_00.00_%Y%m%d-%H%M%S.grib2"
        )

        rel_url_fmt = (
            "/%Y/%m/%d"
            "/mrms/ncep/PrecipRate"
            "/PrecipRate_00.00_%Y%m%d-%H%M%S.grib2.gz"
        )

        file_url = archive_url + datetime.strftime(current_date, rel_url_fmt)

        try:
            print(f"Downloading {file_url} ", end="")
            tmp_file_name, _ = request.urlretrieve(file_url)
            print("DONE")

            dest_file_path = os.path.join(sub_dir, dest_file_name)

            # Uncompress the data
            with gzip.open(tmp_file_name, "rb") as f_in:
                with open(dest_file_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

            current_date = current_date + timedelta(seconds=60 * 2)
            counter += 1

        except HTTPError as err:
            print(err)


def download_pysteps_data(dir_path, force=True):
    """
    Download pysteps data from github.

    Parameters
    ----------
    dir_path: str
        Path to directory where the psyteps data will be placed.
    force: bool
        If the destination directory exits and force=False, a DirectoryNotEmpty
        exception if raised.
        If force=True, the data will we downloaded in the destination directory and may
        override existing files.
    """

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
    # the content is dynamically generated and we can't know the length a priori easily.
    pbar = ShowProgress()
    print("Downloading pysteps-data from github.")
    tmp_file_name, _ = request.urlretrieve(
        "https://github.com/pySTEPS/pysteps-data/archive/master.zip",
        reporthook=pbar,
    )
    pbar.end(message="Download complete\n")

    with ZipFile(tmp_file_name, "r") as zip_obj:
        tmp_dir = TemporaryDirectory()

        # Extract all the contents of zip file in the temp directory
        common_path = os.path.commonprefix(zip_obj.namelist())

        zip_obj.extractall(tmp_dir.name)

        copy_tree(os.path.join(tmp_dir.name, common_path), dir_path)


def create_default_pystepsrc(
    pysteps_data_dir, config_dir=None, file_name="pystepsrc", dryrun=False
):
    """
    Create a default configuration file pointing to the pysteps data directory.

    If the configuration file already exists, it will backup the existing file by
    appending the extensions '.1', '.2', up to '.5.' to the filename.
    A maximum of 5 files are kept. .2, up to app.log.5.

    File rotation is implemented for the backup files.
    For example, if the default configuration filename is 'pystepsrc' and the files
    pystepsrc, pystepsrc.1, pystepsrc.2, etc. exist, they are renamed to respectively
    pystepsrc.1, pystepsrc.2, pystepsrc.2, etc. Finally, after the existing files are
    backed up, the new configuration file is written.

    Parameters
    ----------
    pysteps_data_dir: str
        Path to the directory with the pysteps data.
    config_dir: str
        Destination directory for the configuration file.
        Default values: $HOME/.pysteps (unix and Mac OS X)
        or $USERPROFILE/pysteps (windows).
        The directory is created if it does not exists.
    file_name: str
        Configuration file name. `pystepsrc` by default.
    dryrun: bool
        Do not create the parameter file, nor create backups of existing files.
        No changes are made in the file system. It just returns the file path.

    Returns
    -------
    dest_path: str
        Configuration file path.
    """

    pysteps_lib_root = os.path.dirname(_decode_filesystem_path(pysteps.__file__))

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
        config_dir = os.path.join(home_dir, subdir)

    dest_path = os.path.join(config_dir, file_name)

    if not dryrun:

        if not os.path.isdir(config_dir):
            os.makedirs(config_dir)

        # Backup existing configuration files if it exists and rotate previous backups
        if os.path.isfile(dest_path):
            RotatingFileHandler(dest_path, backupCount=6).doRollover()

        with open(dest_path, "w") as f:
            json.dump(rcparams_json, f, indent=4)

    return os.path.normpath(dest_path)


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
    case: str
        Case to load.
    frames: int
        Number composites (radar images).
        Max allowed value: 24 (35 for MRMS product)
        Default: 14

    Returns
    -------
    rainrate: array-like
        Precipitation data in mm/h. Dimensions: [time, lat, lon]
    metadata: dict
        The metadata observations attributes.
    timestep: number
        Time interval between composites in minutes.
    """

    case = case.lower()

    if case == "mrms":
        max_frames = 36
    else:
        max_frames = 24
    if frames > max_frames:
        raise ValueError(
            f"The number of frames should be smaller than {max_frames + 1}"
        )

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
