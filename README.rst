========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |travis| |requires|
        | |codecov|
    * - package
      - | |version| |wheel| |supported-versions| |supported-implementations|
        | |commits-since|

.. |docs| image:: https://readthedocs.org/projects/chunkblocks/badge/?style=flat
    :target: https://readthedocs.org/projects/chunkblocks
    :alt: Documentation Status

.. |travis| image:: https://travis-ci.org/wongwill86/chunkblocks.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.org/wongwill86/chunkblocks

.. |requires| image:: https://requires.io/github/wongwill86/chunkblocks/requirements.svg?branch=master
    :alt: Requirements Status
    :target: https://requires.io/github/wongwill86/chunkblocks/requirements/?branch=master

.. |codecov| image:: https://codecov.io/github/wongwill86/chunkblocks/coverage.svg?branch=master
    :alt: Coverage Status
    :target: https://codecov.io/github/wongwill86/chunkblocks

.. |version| image:: https://img.shields.io/pypi/v/chunkblocks.svg
    :alt: PyPI Package latest release
    :target: https://pypi.python.org/pypi/chunkblocks

.. |commits-since| image:: https://img.shields.io/github/commits-since/wongwill86/chunkblocks/v0.1.3.svg
    :alt: Commits since latest release
    :target: https://github.com/wongwill86/chunkblocks/compare/v0.1.3...master

.. |wheel| image:: https://img.shields.io/pypi/wheel/chunkblocks.svg
    :alt: PyPI Wheel
    :target: https://pypi.python.org/pypi/chunkblocks

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/chunkblocks.svg
    :alt: Supported versions
    :target: https://pypi.python.org/pypi/chunkblocks

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/chunkblocks.svg
    :alt: Supported implementations
    :target: https://pypi.python.org/pypi/chunkblocks


.. end-badges

Library to facilitate chunked access of large ndarray-like volumes

* Free software: MIT license

Installation
============

::

    pip install chunkblocks

Documentation
=============

https://chunkblocks.readthedocs.io/

Development
===========

To run the all tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
