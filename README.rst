networkinference Documentation
==============================

``networkinference`` is a Python 3 package implementing econometric methods for inference with data exhibiting network dependence or other forms of complex or unknown weak dependence. The package is developed by `Michael P. Leung <https://mpleung.github.io>`_ and distributed under the 3-Clause BSD license. Much of the package is based on work supported by NSF grant SES-1755100.

Links
-----

A tutorial can be found at

  https://nbviewer.org/github/mpleung/networkinference/blob/main/docs/tutorial/tutorial.ipynb

Online documentation, which contains minimal working examples for all functions and classes, is available at

  https://networkinference.readthedocs.io/en/latest/

The latest development version is hosted at

  https://github.com/mpleung/networkinference

Installation
------------

Install via command line using ``pip3``, which is included with Python 3.4+: ::

  $ pip3 install networkinference

Overview
--------

The package contains three main classes:

* ``OLS``: OLS estimator.
* ``TSLS``: 2SLS estimator.
* ``IPW``: Horvitz-Thompson estimator (inverse-probability weighting estimator with known propensity scores).

Each class contains five methods for constructing confidence intervals and two methods implementing scalar equality tests. These methods are based on three classes of inference procedures. The first uses a network HAC variance estimator [1]_ [2]_. The second constructs network clusters using spectral clustering and applies a cluster-robust inference method [4]_. Both require network data and account for a general form of network dependence in which observations are less correlated when further apart in the network (in the sense of shortest path distance) [1]_. The third involves resampling and can be applied to weakly dependent network, clustered, time series, or spatial data (or combination or these) without knowledge of the particular type of dependence [3]_ [5]_. However, it is more computationally intensive and requires a larger amount of data to ensure adequate size control and power.

The methods are also available as independent functions through the ``core`` class for use with custom estimators. 

The ``core`` class and ``utils`` subpackage contain various utilities for computing summary statistics, plotting the spectrum of the Laplacian, constructing friends-of-friends instruments for estimating linear-in-means models, and simulating data.

Example Usage
-------------

  >>> import networkinference as ni               # main package
  >>> from networkinference.utils import FakeData # utilities for generating fake data
  >>> Y, X, W, A = FakeData.tsls()                # simulate data from linear-in-means model
  >>> tsls = ni.TSLS(Y, X, W, A)                  # load data into tsls object
  >>> tsls.network_se()                           # print estimates with network-robust SEs 
  >>> help(tsls)                                  # displays documentation for tsls object

References
----------
.. [1] Kojevnikov, D., V. Marmer, and K. Song, "Limit Theorems for Network Dependent Random Variables," *Journal of Econometrics*, 2021, 222(2), 882-908.
.. [2] Leung, M. "Causal Inference Under Approximate Neighborhood Interference," *Econometrica*, 2022, 90(1), 267-293.
.. [3] Leung, M. "Dependence-Robust Inference Using Resampled Statistics," *Journal of Applied Econometrics*, 2022, 37(2), 270-285.
.. [4] Leung, M., "Network Cluster-Robust Inference," *arXiv preprint arXiv:2103.01470*, 2022.
.. [5] Song, K. "Ordering-Free Inference from Locally Dependent Data," *UBC working paper*, 2016.

Dependencies
------------

* Matplotlib v3.7+
* NetworkX v3.0+
* NumPy v1.24+
* Scikit-learn v1.2+
* SciPy v1.10+
* Seaborn v0.12+
* Tabulate v0.9+
