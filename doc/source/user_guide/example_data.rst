.. _example_data:

Installing the Example data
===========================

The examples scripts in the user guide as well as the pySTEPS build-in tests
use the example radar data available in a separate repository:
`pysteps-data <https://github.com/pySTEPS/pysteps-data>`_.

The data must be downloaded manually into your computer and the ref:pystepsrc"
file need to configured to point to that example data.

First, download the data from the repository by
`clicking here <https://github.com/pySTEPS/pysteps-data/archive/master.zip>`_.

Unzip the data into a folder of your preference. Once the data is unzip, the
directory structure looks like this::


    pysteps-data
    |
    ├── radar
          ├── KNMI
          ├── OPERA
          ├── bom
          ├── fmi
          ├── mch

Now we need to update the pystepsrc file for the each of the data_sources
to point to these directories, as described in :ref:`pystepsrc`.




