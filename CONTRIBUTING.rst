Contributing to Pysteps
=======================

Welcome! pySTEPS is a community-driven initiative for developing and
maintaining an easy to use, modular, free and open source Python
framework for short-term ensemble prediction systems.


If you haven't already, take a look at the project's
`README.rst file <README.rst>`_ and the
`pysteps documentation <https://pysteps.github.io/>`_.

There are many ways to contribute to pysteps:

- contributing bug reports and feature requests
- contributing documentation
- code contributions new features or bug fixes
- contribute with usage examples

Our main forum for discussion is the project's
`GitHub issue tracker <https://github.com/python/mypy/issues>`_.
This is the right place to start a discussion, report a bug, or request a new
feature.


Workflow for code contributions
-------------------------------

We welcome all kind of contributions, from documentation updates, a bug fix,
or a new feature.
If your new feature will take a lot of work,
we recommend creating an issue with the **enhancement** tag to encourage
discussions.

We use the usual
`GitHub pull-request flow <https://help.github.com/en/articles/github-flow>`_,
which may be familiar to you if you've contributed to other projects on GitHub.

**First Time Contributors**

If you are interested in helping to improve pysteps,
the best way to get started is by looking for "Good First Issue" in the
`issue tracker <https://github.com/pySTEPS/pysteps/issues>`_.

In a nutshell, the main steps to follow for contributing to pysteps are:

- Setting up the development environment
- Fork the repository
- Install pre-commit hooks
- Create a new branch for each contribution
- Read the Code Style guide
- Work on your changes
- Test your changes
- Push to your fork repository and create a new PR in GitHub.


Setting up the Development environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The recommended way to setup up the developer environment is the Anaconda
(commonly referred as Conda).
Conda quickly installs, runs and updates packages and their dependencies.
It also allows to easily create, save, load and switch between different
environments on your local computer.

Before continuing, Mac OSX users also need to install a more recent compiler.
See instructions here: :ref:`OSX users <install_osx_users>`.

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

To test if the installation went fine, you can try launching Python and importing
pysteps :ref:`Import pysteps <_import_pysteps>`.

**Note**: In case of installation or import errors, you can remove the pysteps_dev environment
and try starting from an empty environment (adapt the Python version as needed)::

    conda create -n pysteps_dev python=3.8

You can then activate it and install pysteps with pip as exaplained above.

Fork the repository
~~~~~~~~~~~~~~~~~~~

Once you have set the development environment, the next step is creating your
local copy of the repository where you will commit your modifications.
The steps to follow are:

1. Set up Git in your computer.
2. Create a GitHub account (if you don't have one).
3. Fork the repository in your GitHub.
4. Clone local copy of your fork. For example::

    git clone https://github.com/<your-account>/pysteps.git

Done!, now you have a local copy of pysteps git repository.
If you are new to GitHub, below you can find a list of useful tutorials:

- http://rogerdudler.github.io/git-guide/index.html
- https://www.atlassian.com/git/tutorials

Install pre-commit hooks
~~~~~~~~~~~~~~~~~~~~~~~~

Once you cloned the repository and the development environment is installed (and active),
install the git pre-commit hook by executing the following command in the repository's
root::

    pre-commit install

The pre-commit hooks are scripts that are executed in every commit automatically to
identify simple issues with the code. If an issue is identified (the pre-commit script
exits with non-zero status), the hook aborts the commit and print the error.
Currently, Pysteps only test the code to be committed complies with black's autoformat.
In case that the commit is aborted, you only need to run black in the entire source code.
This can be done by running :code:`black .` or :code:`pre-commit run --all-files`.
The latter is recommended since it indicates if the commit contained any formatting errors
(that are automatically corrected).
To ensure that the same configuration is used in every development environment, black's
configuration is stored in the `pyproject.toml` file.
This configuration is automatically load when black is run from any directory in the
pysteps project.

IMPORTANT: Periodically update the black version used in the pre-commit hook by running::

    pre-commit autoupdate

For more information about git hooks and the pre-commit package see:

- https://git-scm.com/book/en/v2/Customizing-Git-Git-Hooks
- https://pre-commit.com/


Create a new branch
~~~~~~~~~~~~~~~~~~~

As a collaborator, all the new contributions that you want should be done in a
new branch under your forked repository.
Working on the master branch is reserved for Core Contributors only.
Core Contributors are developers that actively work and maintain the repository.
They are the only ones who accept pull requests and push commits directly to
the pysteps repository.

For more information on how to create and work with branches, see
`"Branches in a Nutshell" <https://git-scm.com/book/en/v2/Git-Branching-Branches-in-a-Nutshell>`_ in the Git documentation


Code Style
~~~~~~~~~~

We strongly suggest to follow the
`PEP8 coding standards <https://www.python.org/dev/peps/pep-0008/>`_.
Note that this is not strictly enforced yet, since many source files in pysteps
are not PEP8 compliant.
However, we encourage new contributions to be compliant at least with the coding style
summarized next.

Coding style summary
^^^^^^^^^^^^^^^^^^^^

For quick reference, these are the most important good coding practices
to follow:

- Always use 4 spaces for indentation (don’t use tabs).
- Write UTF-8 (add **# -*- coding: utf-8 -*-** at the top of each file).
- Max line-length: 88 characters (note that we don't use the PEP8's 79 value).
- Always indent wrapped code for readability.
- Avoid extraneous whitespace.
- Don’t use whitespace to line up assignment operators (=, :).
- Spaces around = for assignment.
- No spaces around = for default parameter values (keywords).
- Spaces around mathematical operators, but group them sensibly.
- No multiple statements on the same line.
- Naming conventions:

   Function names, variable names, and filenames should be descriptive and self
   explanatory. Avoid using abbreviations that are ambiguous or unfamiliar to
   readers outside your project, and do not abbreviate by deleting letters
   within a word.
   Avoid single letter variables if possible and use more verbose names for
   clarity. An exception for this are indexes in loops (*i, j, k, etc*).

   The following table summarizes the conventions:

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

    Source: `Google's python style guide
    <http://google.github.io/styleguide/pyguide.html>`_

- Ignore returned variables:

  If you need to ignore part to the variables returned by a function,
  use "_" (single underscore) or __ (double underscore)::

    precip, __, metadata = import_bom_rf3('example_file.bom')
    precip2, _, metadata2 = import_bom_rf3('example_file2.bom')


- Zen of Python (PEP 20), the guiding principles for Python’s
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

For a detailed description of a pythonic code style check these guidelines:

- `The Hitchhiker's Guide to Python <https://docs.python-guide.org/writing/style/>`_
- `Google's python style guide <http://google.github.io/styleguide/pyguide.html>`_
- `PEP8 <https://www.python.org/dev/peps/pep-0008/>`_


**Auto-formatters**

Hand-formatting code according the PEP8 style guide can be a tedious process if it is done
manually. To make our lives easy, in pysteps, we use
`black <https://black.readthedocs.io/en/stable/>`_ to auto-format the code using its
default configuration. Black can be installed using any of the following::

    conda install black

    #For the latest version:
    conda install -c conda-forge black

    pip install black

Check the `official documentation <https://black.readthedocs.io/en/stable/the_black_code_style.html>`_
for more information.


**Docstrings**

Every module, function, or class must have a docstring that describe its
purpose and how to use it, following the conventions described in the
`PEP 257 <https://www.python.org/dev/peps/pep-0257/#multi-line-docstrings>`_
and the
`Numpy's docstrings format <https://numpydoc.readthedocs.io/en/latest/format.html>`_.

Here is a summary of the most important rules:

- One-line docstrings Triple quotes are used even though the string fits on one line.
  This makes it easy to later expand it.
- A one-line docstring is a phrase ending in a period.
- All docstrings should be written in imperative ("""Return some value.""")
  mood rather than descriptive mood ("""Returns some value.""").

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


Working on changes
~~~~~~~~~~~~~~~~~~

**IMPORTANT**

If your changes will take a significant amount of work,
we highly recommend opening an issue first, explaining what you want
to do and why. It is better to start the discussions early, in case other
contributors disagree with what you would like to do or have ideas
that will help you do it.


**Collaborators guidelines**

As a collaborator, all the new contributions that you want should be done in a
new branch under your forked repository.
Working on the master branch is reserved for Core Contributors only.
Core Contributors are developers that actively work and maintain the repository.
They are the only ones who accept pull requests and push commits directly to
the **pysteps** repository.

To include the contributions for collaborators, we use the usual
`GitHub pull-request flow <https://help.github.com/en/articles/github-flow>`_.
In their simplest form, pull requests are a mechanism for
a collaborator to notify to the pysteps project about a completed feature".

Once your proposed changes are ready, you need to create a pull request via
your GitHub account. Afterward, the core developers review the code and merge
it into the master branch.
Be aware that pull requests are more than just a notification, they are also
an excellent place for discussing the proposed feature. If there is any problem
with the changes, the other project collaborators can post feedback and the
author of the commit can even fix the problems by pushing follow-up commits to
feature branch.

Do not squash your commits after you have submitted a pull request, as this
erases context during the review.
The commits will be squashed when the pull request is merged.

To keep you forked repository clean, we suggest deleting branches for
once the Pull Requests (PRs) are accepted and merged.

Once you've created a pull request, you can push commits from your topic branch
to add them to your existing pull request.
These commits will appear in chronological order within your pull request and
the changes will be visible in the "Files changed" tab.

Other contributors can review your proposed changes, add review comments,
contribute to the pull request discussion, and even add commits to the pull
request.

**Important:** each PR should should only address a single objective
(e.g. fix a bug, improve documentation, etc).
Pushing changes to an open PR that are outside its objective are highly
discouraged.
Under this circumstances, the recommended way to proceed is creating a new PR
for changes, clearly explaining their goal.


Testing your changes
~~~~~~~~~~~~~~~~~~~~

Before committing changes or creating pull requests, check that the all the
tests in the pysteps suite pass.
See the :ref:`testing_pysteps` for the instruction to run the tests.


Although it is not strictly needed, we suggest creating minimal tests for
new contributions to ensure that it achieves the desired behavior.
Pysteps uses the pytest framework, that it is easy to use and also
supports complex functional testing for applications and libraries.
Check the
`pytests official documentation <https://docs.pytest.org/en/latest/index.html>`_
for more information.

The tests should be placed under the
`pysteps.tests <https://github.com/pySTEPS/pysteps/tree/master/pysteps/tests>`_
module.
The file should follow the **test_*.py** naming convention and have a
descriptive name.

A quick way to get familiar with the pytest syntax and the testing procedures
is checking the python scripts present in the pysteps test module.


Core developer guidelines
-------------------------

Working directly on the master branch is discouraged and is reserved only
for small changes and updates that do not compromise the stability of the code.
The *master* branch is a production branch that is ready to be deployed
(cloned, installed, and ready to use).
In consequence, this master branch is meant to be stable.

The pysteps repository uses a Travis CI, a Continuous Integration service that
automatically runs a series of tests every time you commit to GitHub.
In that way, your modifications along with the entire package is tested.

Pushing untested or work-in-progress changes to the master branch can potentially
introduce bugs or brake the stability of the package.
Since the tests takes around 10 minutes and the are run after the commit was
pushed, any errors introduced in that commit will be noticed after the stable
in the master branch was compromised.
In addition, other developers start working on a new feature from master,
they may start a potentially broken state.

Instead, it is recommended to work on each new feature in its own branch,
which can be pushed to the central repository for backup/collaboration.
When you’re done with the development work on the feature, then you can merge
the feature branch into the master or submit a Pull Request.
This approach has two main advantages:

- Every commit on the feature branch is tested using Travis CI.
  If the tests fail, they do not affect the **master** branch.

- Once the new feature, improvement, or bug correction is finished and the all
  tests passed, the commits history can be squashed into a single commit and
  then merged into the master branch.

This helps approach helps to keep the commits history clean and allows
experimentation in the branch without compromising the stability of the package.


Processing pull requests
~~~~~~~~~~~~~~~~~~~~~~~~

.. _`Squash and merge`: https://github.com/blog/2141-squash-your-commits

Core developers should follow these rules when processing pull requests:

* Always wait for tests to pass before merging PRs.
* Use "`Squash and merge`_"
  to merge PRs.
* Delete branches for merged PRs (by core devs pushing to the main repo).
* Edit the final commit message before merging to conform to the following
  style to help having a clean `git log` output:

    * When merging a multi-commit PR make sure that the commit message doesn't
      contain the local history from the committer and the review history from
      the PR. Edit the message to only describe the end state of the PR.

    * Make sure there is a *single* newline at the end of the commit message.
      This way there is a single empty line between commits in `git log`
      output.

    * Split lines as needed so that the maximum line length of the commit
      message is under 80 characters, including the subject line.

    * Capitalize the subject and each paragraph.

    * Make sure that the subject of the commit message has no trailing dot.

    * Use the imperative mood in the subject line (e.g. "Fix typo in README").

    * If the PR fixes an issue, make sure something like "Fixes #xxx." occurs
      in the body of the message (not in the subject).


Preparing a new release
~~~~~~~~~~~~~~~~~~~~~~~

Core developers should follow the steps to prepare a new release (version):

1. Before creating the actual release in GitHub, be sure that every item in the
   following checklist was followed:

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
   See :ref:`pypi_relase` for a detailed description of this process.

#. Update the conda-forge pysteps-feedstock following this guidelines:
   :ref:`update_conda_feedstock`


Credits
-------

This documents was based in contributors guides of two Python
open source projects:

* Py-Art_: Copyright (c) 2013, UChicago Argonne, LLC.
  `License <https://github.com/ARM-DOE/pyart/blob/master/LICENSE.txt>`_.
* mypy_: Copyright (c) 2015-2016 Jukka Lehtosalo and contributors.
  `MIT License <https://github.com/python/mypy/blob/master/LICENSE>`_.
* Official github documentation (https://help.github.com)

.. _Py-Art: https://github.com/ARM-DOE/pyart
.. _mypy: https://github.com/python/mypy
