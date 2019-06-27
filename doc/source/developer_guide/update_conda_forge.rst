.. _update_conda_feedstock:

==========================================
Updating the conda-forge pysteps-feedstock
==========================================


.. _pysteps-feedstock: https://github.com/conda-forge/pysteps-feedstock
.. _`conda-forge/pysteps-feedstock`: https://github.com/conda-forge/pysteps-feedstock

Here we will describe the steps to update the pysteps conda-forge feedstock.
This tutorial is intended for the core developers listed as maintainers of the
conda recipe in the `conda-forge/pysteps-feedstock`_.

Examples for needing to update the pysteps-feedstock are:

* New release
* Fix errors pysteps package errors

**The following tutorial was adapted from the official conda-forge.org documentation, released
under CC4.0 license**

What is a “conda-forge”
=======================

Conda-forge is a community effort that provides conda packages for a wide range of software.
The conda team from Anaconda packages a multitude of packages and provides them to all users
free of charge in their default channel.

**conda-forge** is a community-led conda channel of installable packages that allows users to share software
that is not included in the official Anaconda repository. The main advantages of **conda-forge** are:

- all packages are shared in a single channel named conda-forge
- care is taken that all packages are up-to-date
- common standards ensure that all packages have compatible versions
- by default, packages are built for macOS, linux amd64 and windows amd64

In order to provide high-quality builds, the process has been automated into the conda-forge GitHub organization.
The conda-forge organization contains one repository for each of the installable packages.
Such a repository is known as a **feedstock**.

The actual pysteps feedstock is https://github.com/conda-forge/pysteps-feedstock

A feedstock is made up of a conda recipe (the instructions on what and how to build the package) and the
necessary configurations for automatic building using freely available continuous integration services.

See the official `conda-forge documentation <http://conda-forge.org/docs/user/00_intro.html>`_ for more details.


Maintain pysteps conda-forge package
====================================

Pysteps Core developers that are maintainers of the pysteps feedstock.

All pysteps developers listed as maintainers of the pysteps feedstock are given push access to the feedstock repository.
This means that a maintainer can create branches in the main repository.

Every time that a new commit is pushed/merged in the feedstock repository, conda-forge runs Continuous Integration (CI)
system that run quality checks, builds the pysteps recipe on Windows, OSX, and Linux, and publish the built recipes in
the conda-forge channel.

Important
---------

For updates, using a branch in the main repo and a subsequent Pull Request (PR) to the master branch is discouraged because:
- CI is run on both the branch and on the Pull Request (if any) associated with that branch. This wastes CI resources.
- Branches are automatically published by the CI system. This mean that a for every push, the packages will be published
before the PR is actually merged.

For these reasons, to update the feedstock, the maintainers need to fork the feedstock, create a new branch in that
fork, push to that branch in the fork, and then open a PR to the conda-forge repo.


Workflow for updating a pysteps-feedstock
-----------------------------------------


The mandatory steps to update the pysteps-feedstock_ are:

1. Forking the pysteps-feedstock_.

    * Clone the forked repository in your computer::

        git clone https://github.com/<your-github-id>/pysteps-feedstock

#. Syncing your fork with the pysteps feedstock. This step is only needed if your local repository is not up to date
   the pysteps-feedstock_. If you just cloned the forked pysteps-feedstock_, you can ignore this step.

    * Make sure you are on the master branch::

        git checkout master

    * Register conda-forge’s feedstock with::

        git remote add upstream https://github.com/conda-forge/pysteps-feedstock

    * Fetch the latest updates with git fetch upstream::

        git fetch upstream

    * Pull in the latest changes into your master branch::

        git rebase upstream/master

#. Create a new branch::

    git checkout -b <branch-name>

#. Update the recipe and push changes in this new branch

    * See next section "Updating recipes" for more details
    * Push changes::

        git commit -m <commit message>

#. Pushing your changes to GitHub::

    git push origin <branch-name>

#. Propose a Pull Request

    * Create a pull request via the web interface


Updating pysteps recipe
=======================

The pysteps-feedstock_ should be updated when:

* We release a new pysteps version
* Need to fix errors in the pysteps package

New release
-----------

When a new pysteps version is released, before update the pysteps feedstock, the new version needs to be uploaded
to the Python Package Index (PyPI) (see :ref:`pypi_relase` for more details).
This step is needed because the conda recipe uses the PyPI to build the pysteps conda package.

Once the new version is available in the PyPI, the conda recipe in pysteps-feedstock/recipe/meta.yaml
needs to be updated by:

1. Updating version and hash

#. Checking the dependencies

#. When the package version changes, reset the build number back to 0.

The build number is increased when the source code for the package has not changed but you need to make a
new build. As a rule of thumb, the build number is increased whenever a new package with the same version needs to
be uploaded to the conda-forge channel.

Recipe fixing
-------------

In case that the recipe must be updated but the source code for the package has not changed the **build_number** in
the conda recipe in
`pysteps-feedstock/recipe/meta.yaml <https://github.com/conda-forge/pysteps-feedstock/blob/master/recipe/meta.yaml>`_
needs to be increased by 1.

Some examples for needing to increase the build number are:

    * updating the pinned dependencies
    * Fixing wrong dependencies
