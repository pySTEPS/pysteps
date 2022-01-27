.. _example_data:

Installing the example data
===========================

The examples scripts in the user guide, as well as the build-in tests,
use the example radar data available in a separate repository:
`pysteps-data <https://github.com/pySTEPS/pysteps-data>`_.

The easiest way to install the example data is by using the
:func:`~pysteps.datasets.download_pysteps_data` and
:func:`~pysteps.datasets.create_default_pystepsrc` functions from
the :mod:`pysteps.datasets` module.

Installation using the datasets module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Below is a snippet code that can be used to install can configure `pystepsrc` file to
point to that example data.


In the example below, the example data is placed in the user's home folder under the
**pysteps_data** directory. It also creates a default configuration file that points to
the downloaded data and places it in the $HOME/.pysteps (Unix and Mac OS X) or
$USERPROFILE/pysteps (Windows). This is one of the default locations where pysteps
looks for the configuration file (see :ref:`pysteps_lookup` for
more information).

.. code-block:: python

    import os

    # Import the helper functions
    from pysteps.datasets import download_pysteps_data, create_default_pystepsrc

    # In this example we will place it in the user's home folder on the
    # `pysteps_data` folder.
    home_dir = os.path.expanduser("~")
    pysteps_data_dir_path = os.path.join(home_dir, "pysteps_data")

    # Download the pysteps data.
    download_pysteps_data(pysteps_data_dir_path, force=True)

    # Create a default configuration file that points to the downloaded data.
    # By default it will place the configuration file in the
    # $HOME/.pysteps (unix and Mac OS X) or $USERPROFILE/pysteps (windows).
    config_file_path = create_default_pystepsrc(pysteps_data_dir_path)

Note that for these changes to take effect you need to restart the python interpreter or
use the :func:`pysteps.load_config_file` function as follows::

    # Load the new configuration file and replace the default configuration
    import pysteps
    pysteps.load_config_file(config_file_path, verbose=True)


To customize the default configuration file see the :ref:`pystepsrc` section.


Manual installation
~~~~~~~~~~~~~~~~~~~

Another alternative is to download the data manually into your computer and configure the
:ref:`pystepsrc <pystepsrc>` file to point to that example data.

First, download the data from the repository by
`clicking here <https://github.com/pySTEPS/pysteps-data/archive/master.zip>`_.

Unzip the data into a folder of your preference. Once the data is unzipped, the
directory structure looks like this::


    pysteps-data
    |
    ├── radar
          ├── KNMI
          ├── OPERA
          ├── bom
          ├── fmi
          ├── mch

The next step is updating the *pystepsrc* file to point to these directories,
as described in the :ref:`pystepsrc` section.




