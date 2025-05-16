.. PyLWL documentation master file, created by
   sphinx-quickstart on Sat May 20 16:59:33 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyLWL's documentation!
=================================

.. image:: https://img.shields.io/badge/release-0.1.0-yellow.svg
   :target: https://github.com/thieu1995/PyLWL/releases

.. image:: https://badge.fury.io/py/pylwl.svg
   :target: https://badge.fury.io/py/pylwl

.. image:: https://img.shields.io/pypi/pyversions/pylwl.svg
   :target: https://www.python.org/

.. image:: https://img.shields.io/pypi/dm/pylwl.svg
   :target: https://img.shields.io/pypi/dm/pylwl.svg

.. image:: https://github.com/thieu1995/PyLWL/actions/workflows/publish-package.yaml/badge.svg
   :target: https://github.com/thieu1995/PyLWL/actions/workflows/publish-package.yaml

.. image:: https://pepy.tech/badge/pylwl
   :target: https://pepy.tech/project/pylwl

.. image:: https://readthedocs.org/projects/pylwl/badge/?version=latest
   :target: https://pylwl.readthedocs.io/en/latest/?badge=latest

.. image:: https://img.shields.io/badge/Chat-on%20Telegram-blue
   :target: https://t.me/+fRVCJGuGJg1mNDg1

.. image:: https://img.shields.io/badge/DOI-10.6084%2Fm9.figshare.29089784-blue
   :target: https://doi.org/10.6084/m9.figshare.29089784

.. image:: https://img.shields.io/badge/License-GPLv3-blue.svg
   :target: https://www.gnu.org/licenses/gpl-3.0


**PyLWL** is an open-source Python library that provides a unified, extensible, and user-friendly
implementation of *Locally Weighted Learning* (LWL) algorithms for supervised learning.
It implements differentiable and gradient-descent-based local models for both classification and regression tasks.

* **Free software:** GNU General Public License (GPL) V3 license
* **Provided Estimators**: `LwClassifier`, `LwRegressor`, `GdLwClassifier`, `GdLwRegressor`
* **Supported Kernel Functions**: Gaussian, Epanechnikov, Cosin...
* **Supported performance metrics**: >= 67 (47 regressions and 20 classifications)
* **Documentation:** https://pylwl.readthedocs.io
* **Python versions:** >= 3.8.x
* **Dependencies:** numpy, scipy, scikit-learn, pandas, permetrics, torch, mealpy


.. toctree::
   :maxdepth: 4
   :caption: Quick Start:

   pages/quick_start.rst

.. toctree::
   :maxdepth: 4
   :caption: Models API:

   pages/pylwl.rst

.. toctree::
   :maxdepth: 4
   :caption: Support:

   pages/support.rst



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
