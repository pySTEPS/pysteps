Contributing to Pysteps
=======================

Welcome! pySTEPS is a community-driven initiative for developing and
maintaining an easy to use, modular, free and open-source Python
framework for short-term ensemble prediction systems.

There are many ways to contribute to pysteps:

* contributing bug reports and feature requests
* contributing documentation
* code contributions, new features, or bug fixes
* contribute with usage examples

Workflow for code contributions
-------------------------------

We welcome all kinds of contributions, like documentation updates, bug fixes, or new features.
The workflow for the contibutions uses the usual
`GitHub pull-request flow <https://help.github.com/en/articles/github-flow>`_.

If you have ideas for new contributions to the project, feel free to get in touch with the pysteps community on our
`pysteps slack <https://pysteps.slack.com/>`__.
To get access to it, you need to ask for an invitation or you can use the automatic invitation page
`here <https://pysteps-slackin.herokuapp.com/>`__.
Our slack channel is a great place for preliminary discussions about new features or functionalities.
Another place where you can report bugs and suggest new enhancements is the
`project's issue tracker <https://github.com/pySTEPS/pysteps/issues>`_.


First Time Contributors
-----------------------

If you are interested in helping to improve pysteps,
the best way to get started is by looking for "Good First Issue" in the
`issue tracker <https://github.com/pySTEPS/pysteps/issues>`_.

In a nutshell, the main steps to follow for contributing to pysteps are:

* Setting up the development environment
* Fork the repository
* Install pre-commit hooks
* Create a new branch for each contribution
* Read the Code Style guide
* Work on your changes
* Test your changes
* Push to your fork repository and create a new PR in GitHub.


Setting up the Development environment
--------------------------------------

The recommended way to setup up the developer environment is the Anaconda
(commonly referred to as Conda).
Conda quickly installs, runs, and updates packages and their dependencies.
It also allows you to create, save, load, and switch between different environments on your local computer.

Before continuing, Mac OSX users also need to install a more recent compiler.
See instructions `here <https://pysteps.readthedocs.io/en/latest/user_guide/install_pysteps.html#install-osx-users>`__.

The developer environment can be created from the file
`environment_dev.yml <https://github.com/pySTEPS/pysteps/blob/master/environment_dev.yml>`_
in the project's root directory by running the command::

    conda env create -f environment_dev.yml

This will create the **pysteps_dev** environment that can be activated using::

    conda activate pysteps_dev

Once the environment is activated, the latest version of pysteps can be installed
in development mode, in such a way that the project appears to be installed,
but yet is still editable from the source tree::

    pip install -e <path to local pysteps repo>

To test if the installation went fine, you can try importing pysteps from the python interpreter by running::

    python -c "import pysteps"


Fork the repository
~~~~~~~~~~~~~~~~~~~

Once you have set the development environment, the next step is creating your local copy of the repository, where you will commit your modifications.
The steps to follow are:

#. Set up Git on your computer.
#. Create a GitHub account (if you don't have one).
#. Fork the repository in your GitHub.
#. Clone a local copy of your fork. For example::

    git clone https://github.com/<your-account>/pysteps.git

Done!, now you have a local copy of pysteps git repository.
If you are new to GitHub, below you can find a list of helpful tutorials:

- http://rogerdudler.github.io/git-guide/index.html
- https://www.atlassian.com/git/tutorials

Install pre-commit hooks
~~~~~~~~~~~~~~~~~~~~~~~~

After setting up your development environment, install the git pre-commit hook by executing the following command in the repository's
root::

    pre-commit install

The pre-commit hooks are scripts executed automatically in every commit to identify simple issues with the code.
When an issue is identified (the pre-commit script exits with non-zero status), the hook aborts the commit and prints the error.
Currently, Pysteps only tests that the code to be committed complies with black's format style.
In case that the commit is aborted, you only need to run black in the entire source code.
This can be done by running :code:`black .` or :code:`pre-commit run --all-files`.
The latter is recommended since it indicates if the commit contained any formatting errors (that are automatically corrected).
Black's configuration is stored in the `pyproject.toml` file to ensure that the same configuration is used in every development environment.
This configuration is automatically loaded when black is run from any directory in the
pysteps project.

IMPORTANT: Periodically update the black version used in the pre-commit hook by running::

    pre-commit autoupdate

For more information about git hooks and the pre-commit package, see:

- https://git-scm.com/book/en/v2/Customizing-Git-Git-Hooks
- https://pre-commit.com/


Create a new branch
~~~~~~~~~~~~~~~~~~~

As a collaborator, all the new contributions you want should be made in a new branch under your forked repository.
Working on the master branch is reserved for Core Contributors only.
Core Contributors are developers that actively work and maintain the repository.
They are the only ones who accept pull requests and push commits directly to the pysteps repository.

For more information on how to create and work with branches, see
`"Branches in a Nutshell" <https://git-scm.com/book/en/v2/Git-Branching-Branches-in-a-Nutshell>`__ in the Git documentation


Code Style
----------

We strongly suggest following the
`PEP8 coding standards <https://www.python.org/dev/peps/pep-0008/>`__.
Since PEP8 is a set of recommendations, these are the most important good coding practices for the pysteps project:

* Always use four spaces for indentation (don’t use tabs).
* Max line-length: 88 characters (note that we don't use the PEP8's 79 value). Enforced by `black`.
* Always indent wrapped code for readability. Enforced by `black`.
* Avoid extraneous whitespace. Enforced by `black`.
* Don’t use whitespace to line up assignment operators (=, :). Enforced by `black`.
* Avoid writing multiple statements in the same line.
* Naming conventions should follow the recomendations from
  the `Google's python style guide <http://google.github.io/styleguide/pyguide.html>`__, summarized as follows:

   .. raw:: html

        <table rules="all" border="1" cellspacing="2" cellpadding="2">

          <tr>
            <th>Type</th>
            <th>Public</th>
            <th>Internal</th>
          </tr>

          <tr>
            <td>Packages</td>
            <td><code>lower_with_under</code></td>
            <td></td>
          </tr>

          <tr>
            <td>Modules</td>
            <td><code>lower_with_under</code></td>
            <td><code>_lower_with_under</code></td>
          </tr>

          <tr>
            <td>Classes</td>
            <td><code>CapWords</code></td>
            <td><code>_CapWords</code></td>
          </tr>

          <tr>
            <td>Exceptions</td>
            <td><code>CapWords</code></td>
            <td></td>
          </tr>

          <tr>
            <td>Functions</td>
            <td><code>lower_with_under()</code></td>
            <td><code>_lower_with_under()</code></td>
          </tr>

          <tr>
            <td>Global/Class Constants</td>
            <td><code>CAPS_WITH_UNDER</code></td>
            <td><code>_CAPS_WITH_UNDER</code></td>
          </tr>

          <tr>
            <td>Global/Class Variables</td>
            <td><code>lower_with_under</code></td>
            <td><code>_lower_with_under</code></td>
          </tr>

          <tr>
            <td>Instance Variables</td>
            <td><code>lower_with_under</code></td>
            <td><code>_lower_with_under</code> (protected)</td>
          </tr>

          <tr>
            <td>Method Names</td>
            <td><code>lower_with_under()</code></td>
            <td><code>_lower_with_under()</code> (protected)</td>
          </tr>

          <tr>
            <td>Function/Method Parameters</td>
            <td><code>lower_with_under</code></td>
            <td></td>
          </tr>

          <tr>
            <td>Local Variables</td>
            <td><code>lower_with_under</code></td>
            <td></td>
          </tr>

        </table>
   (source: `Section 3.16.4, Google's python style guide <http://google.github.io/styleguide/pyguide.html>`__)

- If you need to ignore part of the variables returned by a function,
  use "_" (single underscore) or __ (double underscore)::

    precip, __, metadata = import_bom_rf3('example_file.bom')
    precip2, _, metadata2 = import_bom_rf3('example_file2.bom')


- Zen of Python (`PEP 20 <https://www.python.org/dev/peps/pep-0020/>`__), the guiding principles for Python’s
  design::

    >>> import this
    The Zen of Python, by Tim Peters

    Beautiful is better than ugly.
    Explicit is better than implicit.
    Simple is better than complex.
    Complex is better than complicated.
    Flat is better than nested.
    Sparse is better than dense.
    Readability counts.
    Special cases aren't special enough to break the rules.
    Although practicality beats purity.
    Errors should never pass silently.
    Unless explicitly silenced.
    In the face of ambiguity, refuse the temptation to guess.
    There should be one-- and preferably only one --obvious way to do it.
    Although that way may not be obvious at first unless you're Dutch.
    Now is better than never.
    Although never is often better than *right* now.
    If the implementation is hard to explain, it's a bad idea.
    If the implementation is easy to explain, it may be a good idea.
    Namespaces are one honking great idea -- let's do more of those!

For more suggestions on good coding practices for python, check these guidelines:

- `The Hitchhiker's Guide to Python <https://docs.python-guide.org/writing/style/>`__
- `Google's python style guide <http://google.github.io/styleguide/pyguide.html>`__
- `PEP8 <https://www.python.org/dev/peps/pep-0008/>`__


**Using Black auto-formatter**

To ensure a minimal style consistency, we use
`black <https://black.readthedocs.io/en/stable/>`__ to auto-format to the source code.
The black configuration used in the pysteps project is defined in the pyproject.toml,
and it is automatically detected by black.

Black can be installed using any of the following::

    conda install black

    #For the latest version:
    conda install -c conda-forge black

    pip install black

Check the `official documentation <https://black.readthedocs.io/en/stable/the_black_code_style.html>`__
for more information.

**Docstrings**

Every module, function, or class must have a docstring that describe its
purpose and how to use it. The docstrings follows the conventions described in the
`PEP 257 <https://www.python.org/dev/peps/pep-0257/#multi-line-docstrings>`__
and the
`Numpy's docstrings format <https://numpydoc.readthedocs.io/en/latest/format.html>`__.

Here is a summary of the most important rules:

- Always use triple quotes for doctrings, even if it fits a single line.
- For one-line docstring, end the phrase with a period.
- Use imperative mood for all docstrings ("""Return some value.""") rather than descriptive mood
  ("""Returns some value.""").

Here is an example of a docstring::

    def adjust_lag2_corrcoef1(gamma_1, gamma_2):
        """A simple adjustment of lag-2 temporal autocorrelation coefficient to
        ensure that the resulting AR(2) process is stationary when the parameters
        are estimated from the Yule-Walker equations.

        Parameters
        ----------
        gamma_1 : float
          Lag-1 temporal autocorrelation coeffient.
        gamma_2 : float
          Lag-2 temporal autocorrelation coeffient.

        Returns
        -------
        out : float
          The adjusted lag-2 correlation coefficient.
        """


Contributions guidelines
------------------------

The collaborator guidelines used in pysteps were largely inspired by those of the
`MyPy project <https://github.com/python/mypy>`__.

Collaborators guidelines
~~~~~~~~~~~~~~~~~~~~~~~~

As a collaborator, all your new contributions should be made in a new branch under your forked repository.
Working on the master branch is reserved for Core Contributors only to submit small changes only.
Core Contributors are developers that actively work and maintain the repository.
They are the only ones who accept pull requests and push commits directly to
the **pysteps** repository.

**IMPORTANT**
However, for contribution requires a significant amount of work, we strongly suggest opening a new issue with
the **enhancement** or **discussion** tag to encourage discussions.
The discussions will help clarify the best way to approach the suggested changes or raise potential concerns.

For code contributions, collaboratos can use the usual
`GitHub pull-request flow <https://help.github.com/en/articles/github-flow>`__.
Once your proposed changes are ready, you need to create a pull request (PR) from your fork in your GitHub account.
Afterward, core contributors will review your proposed changes, provide feedback in the PR discussion, and maybe,
request changes to the code. Once the PR is ready, a Core Developer will merge the changes into the main branch.

**Important:**
It is strongly suggested that each PR only addresses a single objective (e.g., fix a bug, improve documentation, etc.).
This will help to reduce the time needed to process the PR. For changes outside the PR's objectives, we highly
recommend opening a new PR.


Testing your changes
~~~~~~~~~~~~~~~~~~~~

Before committing changes or creating pull requests, check that all the tests in the pysteps suite pass.
See the `Testing pySTEPS <https://pysteps.readthedocs.io/en/latest/developer_guide/test_pysteps.html#testing-pysteps>`__
for detailed instruction to run the tests.

Although it is not strictly needed, we suggest creating minimal tests for new contributions to ensure that it achieves
the desired behavior. Pysteps uses the pytest framework that it is easy to use and also supports complex functional
testing for applications and libraries.
Check the `pytests official documentation <https://docs.pytest.org/en/latest/index.html>`_ for more information.

The tests should be placed under the
`pysteps.tests <https://github.com/pySTEPS/pysteps/tree/master/pysteps/tests>`_
module.
The file should follow the **test_*.py** naming convention and have a
descriptive name.

A quick way to get familiar with the pytest syntax and the testing procedures
is checking the python scripts present in the pysteps test module.

Core developer guidelines
~~~~~~~~~~~~~~~~~~~~~~~~

Working directly on the master branch is discouraged and is reserved only
for small changes and updates that do not compromise the stability of the code.
The *master* branch is a production branch that is ready to be deployed
(cloned, installed, and ready to use).
In consequence, this master branch is meant to be stable.

The pysteps repository uses the GitHub Actions service to run tests every time you commit to GitHub.
In that way, your modifications along with the entire library are tested.

Pushing untested or work-in-progress changes to the master branch can potentially introduce bugs or break the stability of the package.
Since the tests triggered by a commit to the master branch take around 20 minutes, any errors introduced there
will be noticed after the stablility of the master branch was compromised.
In addition, other developers start working on a new feature from master from a potentially broken state.

Instead, it is recommended to work on each new feature in its own branch, which can be pushed to the central repository
for backup/collaboration. When you’re done with the feature's development work, you can merge the feature branch into the
master or submit a Pull Request. This approach has two main advantages:

- Every commit on the feature branch is tested via GitHub Actions.
  If the tests fail, they do not affect the **master** branch.

- Once the changes are finished and the tests passed, the commits history can be squashed into a single commit and
  then merged into the master branch. Squashing the commits helps to keep a clean commit history in the main branch.


Processing pull requests
~~~~~~~~~~~~~~~~~~~~~~~~

.. _`Squash and merge`: https://github.com/blog/2141-squash-your-commits

To process the pull request, we follow similar rules to those used in the
`mypy <https://github.com/python/mypy/blob/master/CONTRIBUTING.md#core-developer-guidelines>`_
project:

* Always wait for tests to pass before merging PRs.
* Always use "`Squash and merge`_"  to merge PRs.
* Make sure that the subject of the commit message summarizes the objective of the PR and does not finish with a dot.
* Write a new commit message before merging that provides a detailed description of the changes introduced by the PR.
  Try to keep the maximum line length under 80 characters, spplitting lines if necessary.
  **IMPORTANT:** Make sure that the commit message doesn't contain the branch's commit history!
  Also, if the PR fixes an issue, mention this explicitly.
* Use the imperative mood in the subject line (e.g. "Fix typo in README").

After the PR is merged, the merged branch can be safely deleted.

Preparing a new release
~~~~~~~~~~~~~~~~~~~~~~~

Core developers should follow the steps to prepare a new release (version):

1. Before creating the actual release in GitHub, be sure that every item in the following checklist was followed:

    * In the file setup.py, update the **version="X.X.X"** keyword in the setup
      function.
    * Update the version in PKG-INFO file.
    * If new dependencies were added to pysteps since the last release, add
      them to the **environment.yml, requirements.txt**, and
      **requirements_dev.txt** files.
#. Create a new release in GitHub following
   `these guidelines <https://help.github.com/en/articles/creating-releases>`_.
   Include a detailed changelog in the release.
#. Generating the source distribution for new pysteps version and upload it to
   the `Python Package Index <https://pypi.org/>`_ (PyPI).
   See `Packaging the pysteps project <https://pysteps.readthedocs.io/en/latest/developer_guide/pypi.html#pypi-relase>`__
   for a detailed description of this process.
#. Update the conda-forge pysteps-feedstock following this guidelines:
   `Updating the conda-forge pysteps-feedstock <https://pysteps.readthedocs.io/en/latest/developer_guide/update_conda_forge.html#update-conda-feedstock>`__


Credits
-------

This document was based on contributors guides of two Python
open-source projects:

* Py-Art_: Copyright (c) 2013, UChicago Argonne, LLC.
  `License <https://github.com/ARM-DOE/pyart/blob/master/LICENSE.txt>`_.
* mypy_: Copyright (c) 2015-2016 Jukka Lehtosalo and contributors.
  `MIT License <https://github.com/python/mypy/blob/master/LICENSE>`_.
* Official github documentation (https://help.github.com)

.. _Py-Art: https://github.com/ARM-DOE/pyart
.. _mypy: https://github.com/python/mypy
