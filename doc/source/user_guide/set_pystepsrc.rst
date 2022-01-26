.. _pystepsrc:

The pysteps configuration file (pystepsrc)
==========================================

.. _JSON: https://en.wikipedia.org/wiki/JSON

The pysteps package allows the users to customize the default settings
and configuration.
The configuration parameters used by default are loaded from a user-defined
JSON_ file and then stored in `pysteps.rcparams`, a dictionary-like object
that can be accessed as attributes or as items.
For example, the default parameters can be obtained using any of the following ways::

    import pysteps

    # Retrieve the colorscale for plots
    colorscale = pysteps.rcparams['plot']['colorscale']
    colorscale = pysteps.rcparams.plot.colorscale

    # Retrieve the the root directory of the fmi data
    pysteps.rcparams['data_sources']['fmi']['root_path']
    pysteps.rcparams.data_sources.fmi.root_path

A less wordy alternative::

    from pysteps import rcparams
    colorscale = rcparams['plot']['colorscale']
    colorscale = rcparams.plot.colorscale

    fmi_root_path = rcparams['data_sources']['fmi']['root_path']
    fmi_root_path = rcparams.data_sources.fmi.root_path

.. _pysteps_lookup:

Configuration file lookup
~~~~~~~~~~~~~~~~~~~~~~~~~

When the pysteps package imported, it looks for **pystepsrc** file in the
following order:

- **$PWD/pystepsrc** : Looks for the file in the current directory
- **$PYSTEPSRC** : If the system variable $PYSTEPSRC is defined and it
  points to a file, it is used.
- **$PYSTEPSRC/pystepsrc** : If $PYSTEPSRC points to a directory, it looks for the
  pystepsrc file inside that directory.
- **$HOME/.pysteps/pystepsrc** (Unix and Mac OS X) : If the system variable $HOME is defined, it looks
  for the configuration file in this path.
- **%USERPROFILE%\\pysteps\\pystepsrc** (Windows only): It looks for the configuration file
  in the pysteps directory located user's home directory (indicated by the %USERPROFILE%
  system variable).
- Lastly, it looks inside the library in *pysteps\\pystepsrc* for a
  system-defined copy.

The recommended method to setup the configuration files is to edit a copy
of the default **pystepsrc** file that is distributed with the package
and place that copy inside the user home folder.
See the instructions below.


Setting up the user-defined configuration file
----------------------------------------------


Linux and OSX users
~~~~~~~~~~~~~~~~~~~

For Linux and OSX users, the recommended way to customize the pysteps
configuration is placing the pystepsrc parameters file in the users home folder
${HOME} in the following path: **${HOME}/.pysteps/pystepsrc**

To steps to setup up the configuration file in the home directory first, we
need to create the directory if it does not exist. In a terminal, run::

    $ mkdir -p ${HOME}/.pysteps

The next step is to find the location of the library's default pystepsrc file.
When we import pysteps in a python interpreter, the configuration file loaded
is shown::

    import pysteps
    "Pysteps configuration file found at: /path/to/pysteps/library/pystepsrc"

Then we copy the library's default configuration file to that directory::

    $ cp /path/to/pysteps/library/pystepsrc ${HOME}/.pysteps/pystepsrc

Edit the file with the text editor of your preference and change the default
configurations with your preferences.

Finally, check that the correct configuration file is loaded by the library::

     import pysteps
     "Pysteps configuration file found at: /home/user_name/.pysteps/pystepsrc"


Windows
~~~~~~~

For windows users, the recommended way to customize the pysteps
configuration is placing the pystepsrc parameters file in the users' folder
(defined in the %USERPROFILE% environment variable) in the following path:
**%USERPROFILE%\\pysteps\\pystepsrc**

To setup up the configuration file in the home directory first, we
need to create the directory if it does not exist. In a **windows terminal**, run::

    $ mkdir %USERPROFILE%\pysteps

**Important**

It was reported that the %USERPROFILE% variable may be interpreted as an string
literal when the anaconda terminal is used.
This will result in a '%USERPROFILE%' folder being created in the current working directory
instead of the desired pysteps folder in the user's home.
If that is the case, use the explicit path to your home folder instead of `%USERPROFILE%`.
For example::

    $ mkdir C:\Users\your_username\pysteps

The next step is to find the location of the library's default pystepsrc file.
When we import pysteps in a python interpreter, the configuration file loaded
is shown::

    import pysteps
    "Pysteps configuration file found at: C:\path\to\pysteps\library\pystepsrc"

Then we copy the library's default configuration file to that directory::

    $ copy C:\path\to\pysteps\library\pystepsrc %USERPROFILE%\pysteps\pystepsrc

Edit the file with the text editor of your preference and change the default
configurations with your preferences.

Finally, check that the correct configuration file is loaded by the library::

     import pysteps
     "Pysteps configuration file found at: C:\User\Profile\.pysteps\pystepsrc"



More
----

.. toctree::
    :maxdepth: 1

    Example pystepsrc file <pystepsrc_example>
