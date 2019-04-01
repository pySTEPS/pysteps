Contributing to Pysteps
=======================

Welcome! pySTEPS is a community-driven initiative for developing and
maintaining an easy to use, modular, free and open source Python
framework for short-term ensemble prediction systems.


Getting started, building, and testing
--------------------------------------

If you haven't already, take a look at the project's
`README.rst file <README.rst>`_.
and the `pysteps documentation <https://pysteps.github.io/>`_.
There you will find all the necessary information to install pysteps.



Code Style
----------

Although it is not strictly enforced yet, we strongly suggest to follow the
`pep8 coding standards <https://www.python.org/dev/peps/pep-0008/>`_.
Two popular modules used to check pep8 are
`pycodestyle <https://pypi.org/project/pycodestyle/>`_ and
`pylint <https://pypi.org/project/pylint/>`_.

You can install them using pip::

    pip install pylint
    pip install pycodestyle

or using anaconda::

    conda install pylint
    conda install pycodestyle

For further instructions please refer to their official documentation.

- https://pycodestyle.readthedocs.io/en/latest/
- https://www.pylint.org/


Discussion
----------

You are welcome to start a discussion in the project's
`GitHub issue tracker <https://github.com/python/mypy/issues>`_ if you
have run into behavior in pysteps that you don't understand or
you have found a bug or would like make a feature request.



Contributions workflow
----------------------

Submitting Changes
~~~~~~~~~~~~~~~~~~

We welcome all kind of contributions, from documentation updates, a bug fix,
or a new feature. If your new feature will take a lot of work,
we recommend creating an issue with the **enhancement** tag to encourage
discussions.

We use the usual GitHub pull-request flow, which may be familiar to
you if you've contributed to other projects on GitHub.


First Time Contributors
~~~~~~~~~~~~~~~~~~~~~~~

If you are interested in helping to improve pysteps,
the best way to get started is by looking for "Good First Issue" in the
`issue tracker <https://github.com/pySTEPS/pysteps/issues>`_.


Fork the repository
~~~~~~~~~~~~~~~~~~~

The first step is creating your local copy of the repository where you will
commit your modifications. The steps to follow are:

1. Set up Git in your computer.
2. Create a GitHub account (if you don't have one).
3. Fork the repository in your GitHub.
4. Clone local copy of your fork. For example::

    git clone https://github.com/<your-account>/pysteps.git

Done!, now you have a local copy of pysteps git repository.
If you are new to GitHub, below you can find a list of useful tutorials:

- http://rogerdudler.github.io/git-guide/index.html
- https://www.atlassian.com/git/tutorials


Preparing Changes
~~~~~~~~~~~~~~~~~


**IMPORTANT**

If your changes will take a significant amount of work,
we highly recommend opening an issue first, explaining what do you want
to do and why. It is better to start the discussions early in case that other
contributors disagree with what you would like to do or have ideas
that will help you do it.


Collaborators guidelines
~~~~~~~~~~~~~~~~~~~~~~~~

As a collaborator, all the new contributions that you want should be done in a
new branch under your forked repository.
Working on the master branch is reserved for Core Contributors only.
Core Contributors are developers that actively work and maintain the repository.
They are the only ones who accept pull requests and push commits directly to
the **pysteps** repository.

To include the contributions for collaborators, we use the usual GitHub
pull-request flow. In their simplest form, pull requests are a mechanism for
a collaborator to notify to the pysteps project about completed a feature.

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
The commits will be squashed commits when the pull request is merged.

To keep you forked repository clean, we suggest deleting branches for
once the Pull Requests (PRs) are accepted and merged.

Core developer guidelines
~~~~~~~~~~~~~~~~~~~~~~~~~

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



Testing your changes
~~~~~~~~~~~~~~~~~~~~

Before committing changes or creating pull requests, check that the build-in
tests passed.
See the `Test wiki <https://github.com/pySTEPS/pysteps/wiki/Testing-pysteps>`_
for the instruction to run the tests.


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


Credits
-------

This documents was based in contributors guides of two Python
open source projects:

* Py-Art_: Copyright (c) 2013, UChicago Argonne, LLC.
  `License <https://github.com/ARM-DOE/pyart/blob/master/LICENSE.txt>`_.
* mypy_: Copyright (c) 2015-2016 Jukka Lehtosalo and contributors.
  `MIT License <https://github.com/python/mypy/blob/master/LICENSE>`_.

.. _Py-Art: https://github.com/ARM-DOE/pyart
.. _mypy: https://github.com/python/mypy