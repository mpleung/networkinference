import numpy as np
from scipy.linalg import inv
from scipy.stats import norm
from decimal import Decimal
from tabulate import tabulate
from .core import core

class OLS(object):
    """OLS estimator.

    Parameters 
    ----------
    Y : numpy float array
        n-dimensional array of outcomes.
    X : numpy float array
        n x k array of regressors (not including intercept) or n-dimensional array.
    A : NetworkX graph
        Graph on n nodes. NOTE: Assumes nodes are labeled as integers 0, ..., n-1 in A, so that the outcome of node i is given by the ith component of Y. Network can be weighted or directed, although weights and directions are ignored when computing network SEs. Argument not used for dependence robust test or CI. Default value: None.

    Attributes
    ----------
    data : dictionary
        Stores all input data, adding a column of ones to X.
    summands : numpy array
        n-dimensional array of intermediate products used to compute OLS estimator.
    estimate : float
        OLS estimator.
    resid : numpy array
        Regression residuals.
    invhessian : numpy array
        Inverse hessian matrix.
    scores : numpy array
        Regression scores.

    Examples
    --------
    >>> import networkinference as ni
    >>> from networkinference.utils import FakeData
    >>> Y, X, A = FakeData.ols()
    >>> ols_model = ni.OLS(Y, X, A)
    >>> print(ols_model.estimate)
    """

    def __init__(self, Y, X, A=None):
        """Stores inputs, computes estimator.
        """
        if X.ndim == 1:
            Xp = np.vstack([np.ones(X.size), X]).T # n x 2
        elif X.ndim == 2:
            Xp = np.hstack([np.ones(X.shape[0])[:,np.newaxis], X]) # n x (k+1)
        self.invhessian = inv(Xp.T.dot(Xp)) # (k+1) x (k+1), (Xp'Xp)^{-1} matrix
        self.summands = Xp * Y[:,np.newaxis] # n x (k+1), mean of this is Xp'Y matrix
        self.estimate = self.invhessian.dot(self.summands.sum(axis=0)) # (k+1) dimensional, OLS estimate
        self.resid = Y - Xp.dot(self.estimate) # residuals
        self.scores = Xp * self.resid[:,np.newaxis]
        self.data = {'Y':Y, 'X':Xp, 'network':A}

    def network_se(self, b=None, decimals=3, verbose=True, PD_alert=False):
        """Returns standard errors derived from network HAC variance estimator due to [1]_ using bandwidth suggested by [2]_. Setting b=0 outputs the conventional heteroskedasticity-robust variance estimator for i.i.d. data. Network is converted to an unweighted, undirected version by dropping edge weights and directionality of links.

        The default output uses a uniform kernel. If the result is not positive definite, the output is an estimator guaranteed to be positive definite due to the first working paper version of [2]_.
        
        Parameters
        ----------
        b : float
            HAC bandwidth. Recommend keeping b=None, which uses the bandwidth choice recommended by [2]_. Default value: None. 
        decimals : int
            Number of decimals to which to round the output table.
        verbose : boolean
            If True, calling this method prints out the results. Default value: True.
        PD_alert : boolean
            If True, method will print an alert whenever the default estimator is not positive definite.

        Attributes
        ----------
        network_se_vcov : float
            Estimate of variance-covariance matrix.
        network_se_result : float
            Standard errors.

        Examples
        --------
        >>> import networkinference as ni
        >>> from networkinference.utils import FakeData
        >>> Y, X, A = FakeData.ols()
        >>> ols_model = ni.OLS(Y, X, A)
        >>> ols.network_se()

        References
        ----------
        .. [1] Kojevnikov, D., V. Marmer, and K. Song, "Limit Theorems for Network Dependent Random Variables," Journal of Econometrics, 2021, 222 (2), 882-908.
        .. [2] Leung, M. "Causal Inference Under Approximate Neighborhood Interference," Econometrica (forthcoming), 2021.
        """
        PD_failure = False
        if isinstance(self.invhessian, np.ndarray):
            if PD_alert:
                V,_,_,PD_failure = core.network_hac(self.scores, self.data['network'], b, disp=True)
            else:
                V = core.network_hac(self.scores, self.data['network'], b)
            self.network_se_vcov = self.data['Y'].size * self.invhessian.dot(V).dot(self.invhessian)
            self.network_se_result = np.sqrt(np.diag(self.network_se_vcov))
        else:
            if PD_alert:
                self.network_se_vcov,_,_,PD_failure = core.network_hac(self.summands, self.data['network'], b, disp=True)
            else:
                self.network_se_vcov = core.network_hac(self.summands, self.data['network'], b) 
            self.network_se_result = np.sqrt(self.network_se_vcov / self.summands.size)
        if PD_alert and PD_failure: print('Estimator not positive definite. Correction used.')

        if verbose:
            CV = norm.ppf(1-0.05/2)
            if self.estimate.size == 1:
                est = np.array([self.estimate])
                se = np.array([self.network_se_result])
            else:
                est = self.estimate
                se = self.network_se_result
            fmat = '%.' + str(decimals) + 'f'
            table = []
            for k in range(est.size):
                CI = [est[k] - CV * se[k], est[k] + CV * se[k]]
                CI = [float(Decimal(fmat % CI[0])), float(Decimal(fmat % CI[1]))]
                table.append([est[k], se[k], CI])
            print(tabulate(table, headers=['Estimate', 'SE', '95% CI'], floatfmt='.' + str(decimals) + 'f'))

    def drobust_test(self, mu, dimension=0, alpha=0.05, beta=0.01, R=None, L=1000, seed=None, verbose=True):
        """Returns conclusion of dependence-robust test due to [1]_. Note that the output of the test is random by nature. L is the number of simulation draws, and larger values reduce the random variation of the test. 

        Test is implemented using the U-type statistic and randomized confidence function approach due to [2]_ discussed in Remark 2 of [1]_.
        
        Parameters
        ----------
        mu : float
            Null value of the estimand in the specified dimension.
        dimension : int
            Dimension of the estimand being tested. Ignored if estimand is scalar. Default value: 0.
        alpha : float
            Significance level. Default value: 0.05.
        beta : float
            beta in Remark 2 of Leung (2021). The closer this is to alpha, the more conservative the test. Default value: 0.01.
        R : int
            Number of resampling draws for test statistic. Uses default if R=None. Default value: None.
        L : int
            Number of resampling draws for randomized confidence function. The larger the value, the less random the output. Default value: 1000.
        seed : int
            seed for resampling draws. Set to None to not set a seed. Default value: None. 
        verbose : boolean
            If True, calling this method prints out the results. Default value: True.

        Attributes
        ----------
        drobust_test_result : string
            Reject or not reject.

        Examples
        --------
        >>> import networkinference as ni
        >>> from networkinference.utils import FakeData
        >>> Y, X, A = FakeData.ols()
        >>> ols_model = ni.OLS(Y, X, A)
        >>> ols.drobust_test(1, dimension=1)

        References
        ----------
        .. [1] Leung, M. "Dependence-Robust Inference Using Resampled Statistics," Journal of Applied Econometrics (forthcoming), 2021.
        .. [2] Song, K. "Ordering-Free Inference from Locally Dependent Data," UBC working paper, 2016. 
        """
        if isinstance(self.invhessian, np.ndarray):
            dat = self.summands.shape[0] * self.summands.dot(self.invhessian)[:,dimension]
        else:
            dat = self.summands
        self.drobust_test_result = core.drobust_test(dat, mu, alpha, beta, R, L, seed) 
        if verbose: print(f'Conclusion of dependence-robust test: {self.drobust_test_result}')

    def drobust_ci(self, grid_start, grid_stop, dimension=None, grid_size=151, coverage=0.95, \
            beta=0.01, R=None, L=1000, seed=None, decimals=3, verbose=True): 
        """Returns confidence interval (CI) derived from the dependence-robust test due to [1]_. Note that the output of the test is random by nature. L is the number of simulation draws, and larger values reduce the random variation of the test. If the result is a trivial interval, try increasing grid_size.

        Test is implemented using the U-type statistic and randomized confidence function approach due to [2]_ discussed in Remark 2 of [1]_.
        
        Parameters
        ----------
        grid_start : float
            Need to specify a grid of values to test for inclusion in the CI. This is the leftmost point of the grid.
        grid_stop : float
            Rightmost point of the grid. 
        dimension : int
            Dimension of the estimand for which you want the CI. Ignored if estimand is scalar. To generate a table of CIs for all dimensions, set dimension=None. Default value: None.
        grid_size : int
            Number of points in the grid. Default value: 151.
        coverage : float
            Desired coverage. Default value: 0.95.
        beta : float
            beta in Remark 2 of Leung (2021). The closer this is to 1-coverage, the more conservative the CI. Default value: 0.01.
        R : int
            Number of resampling draws for test statistic. Uses default if R=None. Default value: None. 
        L : int
            Number of resampling draws for randomized confidence function. The larger the value, the less random the output. Default value: 1000.
        seed : int
            seed for resampling draws. Set to None to not set a seed. Default value: None. 
        decimals : int
            Number of decimals to which to round the output table.
        verbose : boolean
            If True, calling this method prints out the results. Default value: True.

        Attributes
        ----------
        drobust_ci_result : list 
            Confidence interval.

        Examples
        --------
        >>> import networkinference as ni
        >>> from networkinference.utils import FakeData
        >>> Y, X, A = FakeData.ols()
        >>> ols_model = ni.OLS(Y, X, A)
        >>> ols.drobust_ci(-5, 5)

        References
        ----------
        .. [1] Leung, M. "Dependence-Robust Inference Using Resampled Statistics," Journal of Applied Econometrics (forthcoming), 2021. 
        .. [2] Song, K. "Ordering-Free Inference from Locally Dependent Data," UBC working paper, 2016.
        """ 
        if isinstance(self.invhessian, np.ndarray):
            dims = range(self.estimate.size) if dimension == None else [dimension]
        else:
            dims = [0]

        fmat = '%.' + str(decimals) + 'f'
        table = []
        self.drobust_ci_result = []
        for dim in dims:
            if isinstance(self.invhessian, np.ndarray):
                dat = self.summands.shape[0] * self.summands.dot(self.invhessian)[:,dim]
            else:
                dat = self.summands
            CI = core.drobust_ci(dat, grid_start, grid_stop, grid_size, coverage, beta, R, L, seed)
            CI = [np.around(CI[0],6), np.around(CI[1],6)] # dealing with floating point error
            self.drobust_ci_result.append(CI)
            if verbose:
                CI = [float(Decimal(fmat % CI[0])), float(Decimal(fmat % CI[1]))]
                table.append([dat.mean(), CI])
        if len(self.drobust_ci_result) == 1: self.drobust_ci_result = self.drobust_ci_result[0]
        if verbose: print(tabulate(table, headers=['Estimate', 'CI'], floatfmt='.' + str(decimals) + 'f'))

    def get_clusters(self, num_clusters, clusters=None, seed=None, weight=None, verbose=True):
        """Returns network clusters obtained from normalized spectral clustering algorithm due to [2]_ (also see [3]_). Returns maximal conductance of clusters, a [0,1]-measure of cluster quality that should be at most 0.1 for cluster-robust methods to have good performance (see [1]_). All nodes not in the giant component are grouped into a single cluster.
        
        Parameters
        ----------
        num_clusters : int
            Number of desired clusters in the giant component.
        seed : int
            Seed for k-means clustering initialization. Set to None to not set a seed. Default value: None.
        clusters : numpy array
            Optional array of cluster memberships obtained from the output of this method or spectral_clustering() in the core class. The only purpose of this argument is to load clusters obtained elsewhere into the current object.
        weight : string
            Specifies how edge weights are labeled in A, if A is a weighted graph. Default value: None.
        verbose : boolean
            When set to True, the method prints the maximal conductance of the clusters. Default value: True.

        Attributes
        ----------
        clusters : numpy array
            n-dimensional array of cluster labels from 0 to num_clusters-1, where n is the number of nodes.
        conductance : float
            Maximal conductance of the clusters. 

        Examples
        --------
        >>> import networkinference as ni
        >>> from networkinference.utils import FakeData
        >>> Y, X, A = FakeData.ols(network='RGG')
        >>> ols_model = ni.OLS(Y, X, A)
        >>> ols.get_clusters(10)

        References
        ----------
        .. [1] Leung, M., "Network Cluster-Robust Inference," arXiv preprint arXiv:2103.01470, 2021.
        .. [2] Ng, A., M. Jordan, Y. Weiss, "On Spectral Clustering: Analysis and an Algorithm." Advances in Neural Information Processing Systems, 2002, 849-856.
        .. [3] von Luxburg, U., "A Tutorial on Spectral Clustering," Statistics and Computing, 2007, 17 (4), 395-416.
        """
        if isinstance(clusters, np.ndarray):
            self.clusters = clusters
        else:
            self.clusters = core.spectral_clustering(num_clusters, self.data['network'], seed) 

        self.conductance = core.conductance(self.clusters, self.data['network'], weight)
        if verbose: print(f'Maximal conductance: {self.conductance}')

    def est_by_cluster(self, dimension):
        """Returns array of OLS estimates, one for each cluster. This is a helper method used by arand_test() and arand_ci().

        Parameters
        ----------
        dimension : int
            Dimension of estimand being tested. Ignore if estimand is scalar. Default value: 0.

        Returns
        -------
        thetahat : numpy array
            L-dimensional array of OLS estimates, one for each of the L clusters.
        """
        thetahat = []
        for C in np.unique(self.clusters):
            members = np.where(self.clusters==C)[0]
            Yp = self.data['Y'][members]
            Xp = self.data['X'][members,:]
            thetahat.append( np.linalg.pinv(Xp.T.dot(Xp)).dot(Xp.T.dot(Yp[:,np.newaxis]))[dimension,0] )
        if len(thetahat) == 1:
            thetahat = thetahat[0]
        else:
            thetahat = np.array(thetahat)
        return thetahat

    def trobust_ci(self, dimension=None, num_clusters=5, coverage=0.95, decimals=3, verbose=True):
        """Returns confidence interval (CI) from the t-statistic based cluster-robust procedure due to [1]_. The more clusters, the more powerful the test. However, since the test computes estimates cluster by cluster, the output can be more unstable with a larger number of clusters since the sample size within each cluster can be small.
        
        Parameters
        ----------
        dimension : int
            Dimension of the estimand for which you want the CI. Ignored if estimand is scalar. To generate a table of CIs for all dimensions, set dimension=None. Default value: None.
        num_clusters : int
            Ignored if get_clusters() was already run on this object. If it wasn't, this calls the get_cluster() method, asking for this many clusters. Default value: 5.
        coverage : float
            Desired coverage. Default value: 0.95.
        decimals : int
            Number of decimals to which to round the output table.
        verbose : boolean
            If True, calling this method prints out the results. Default value: True.

        Attributes
        ----------
        trobust_ci_result : list
            Confidence interval.

        Examples
        --------
        >>> import networkinference as ni
        >>> from networkinference.utils import FakeData
        >>> Y, X, A = FakeData.ols(network='RGG')
        >>> ols_model = ni.OLS(Y, X, A)
        >>> ols.get_clusters(10)
        >>> ols.trobust_ci()

        References
        ----------
        .. [1] Ibragimov, R. and U. Mueller, "t-Statistic Based Correlation and Heterogeneity Robust Inference," Journal of Business and Economic Statistics, 2010, 28 (4), 453-468. 
        """
        if not hasattr(self,'clusters'): self.get_clusters(num_clusters)

        if isinstance(self.invhessian, np.ndarray):
            dims = range(self.estimate.size) if dimension == None else [dimension]
        else:
            dims = [0]

        fmat = '%.' + str(decimals) + 'f'
        table = []
        self.trobust_ci_result = []
        for dim in dims:
            if isinstance(self.invhessian, np.ndarray):
                est = self.estimate[dim]
            else:
                est = self.estimate
            thetahat = self.est_by_cluster(dim) 
            CI = core.trobust_ci(thetahat, coverage, False) 
            self.trobust_ci_result.append(CI)
            if verbose:
                CI = [float(Decimal(fmat % CI[0])), float(Decimal(fmat % CI[1]))]
                table.append([est, CI])
        if len(self.trobust_ci_result) == 1: self.trobust_ci_result = self.trobust_ci_result[0]
        if verbose: print(tabulate(table, headers=['Estimate', 'CI'], floatfmt='.' + str(decimals) + 'f'))

    def arand_test(self, mu, dimension=0, num_clusters=5, seed=None, verbose=True):
        """Returns p-value of approximate randomization test [1]_. The test is more powerful with more clusters. However, since the test computes estimates cluster by cluster, the output can be more unstable with a larger number of clusters since the sample size within each cluster can be small.
        
        Parameters
        ----------
        dimension : int
            Dimension of estimand being tested. Ignored if estimand is scalar. Default value: 0.
        mu : float
           Null value of the estimand in the specified dimension. 
        num_clusters : int
            Ignored if get_clusters() was already run on this object. If it wasn't, this calls the get_cluster() method, asking for this many clusters. Default value: 5. 
        seed : int
            Seed for drawing permutations, which is only relevant when there are more than 12 clusters. Set to None to not set a seed. Default value: None.
        verbose : boolean
            If True, calling this method prints out the results. Default value: True.

        Attributes
        ----------
        arand_test_result : float
            P-value.
        arand_test_stat : float
            Test statistic.

        Examples
        --------
        >>> import networkinference as ni
        >>> from networkinference.utils import FakeData
        >>> Y, X, A = FakeData.ols(network='RGG')
        >>> ols_model = ni.OLS(Y, X, A)
        >>> ols.get_clusters(10)
        >>> ols.arand_test(1, dimension=1)

        References
        ----------
        .. [1] Canay, I., J. Romano, and A. Shaikh, "Randomization Tests Under an Approximate Symmetry Assumption," Econometrica, 2017, 85 (3), 1013-1030.
        """
        if not hasattr(self,'clusters'): self.get_clusters(num_clusters)
        thetahat = self.est_by_cluster(dimension)
        self.arand_test_result, self.arand_test_stat = core.arand_test(thetahat, mu, seed)
        if verbose: print(f'P-value of randomization test: {self.arand_test_result}')

    def arand_ci(self, grid_start, grid_stop, dimension=None, grid_size=151, coverage=0.95, \
            num_clusters=5, decimals=3, seed=None, verbose=True):
        """Returns confidence interval (CI) obtained by inverting an approximate randomization test [1]_. If the result is a trivial interval, try increasing grid_size. The CI is narrower with more clusters. However, since the test computes estimates cluster by cluster, the output can be more unstable with a larger number of clusters since the sample size within each cluster can be small.
        
        Parameters
        ----------
        grid_start : float
            Need to specify a grid of values over which to invert the test. This is the leftmost point of the grid.
        grid_stop : float
            Rightmost point of the grid.
        dimension : int
            Dimension of the estimand for which you want the CI. To generate a table of CIs for all dimensions, set dimension=None. Ignored if estimand is scalar. Default value: None. 
        grid_size : int
            Number of points in the grid. Default value: 151.
        coverage : float
            Desired coverage. Default value: 0.95.
        num_clusters : int
            Ignored if get_clusters() was already run on this object. If it wasn't, this calls the get_cluster() method, asking for this many clusters. Default value: 5.
        decimals : int
            Number of decimals to which to round the output table.
        seed : int
            Seed for drawing permutations, which is only relevant when there are more than 12 clusters. Set to None to not set a seed. Default value: None.
        verbose : boolean
            If True, calling this method prints out the results. Default value: True.

        Attributes
        ----------
        arand_ci_result : list
            Confidence interval.

        Examples
        --------
        >>> import networkinference as ni
        >>> from networkinference.utils import FakeData
        >>> Y, X, A = FakeData.ols(network='RGG')
        >>> ols_model = ni.OLS(Y, X, A)
        >>> ols.get_clusters(10)
        >>> ols.arand_ci(-5, 5)

        References
        ----------
        .. [1] Canay, I., J. Romano, and A. Shaikh, "Randomization Tests Under an Approximate Symmetry Assumption," Econometrica, 2017, 85 (3), 1013-1030.
        """
        if not hasattr(self,'clusters'): self.get_clusters(num_clusters)

        if isinstance(self.invhessian, np.ndarray):
            dims = range(self.estimate.size) if dimension == None else [dimension]
        else:
            dims = [0]

        fmat = '%.' + str(decimals) + 'f'
        table = []
        self.arand_ci_result = []
        for dim in dims:
            if isinstance(self.invhessian, np.ndarray):
                est = self.estimate[dim]
            else:
                est = self.estimate
            thetahat = self.est_by_cluster(dim) 
            CI = core.arand_ci(thetahat, grid_start, grid_stop, grid_size, coverage, seed) 
            CI = [np.around(CI[0],6), np.around(CI[1],6)] # dealing with floating point error
            self.arand_ci_result.append(CI)
            if verbose:
                CI = [float(Decimal(fmat % CI[0])), float(Decimal(fmat % CI[1]))]
                table.append([est, CI])
        if len(self.arand_ci_result) == 1: self.arand_ci_result = self.arand_ci_result[0]
        if verbose: print(tabulate(table, headers=['Estimate', 'CI'], floatfmt='.' + str(decimals) + 'f'))

    def cluster_se(self, num_clusters=30, decimals=3, verbose=True):
        """Returns clustered standard errors.
        
        Parameters
        ----------
        num_clusters : int
            Ignored if get_clusters() was already run on this object. If it wasn't, this calls the get_cluster() method, asking for this many clusters. Default value: 30.
        decimals : int
            Number of decimals to which to round the output table.
        verbose : boolean
            If True, calling this method prints out the results. Default value: True.

        Attributes
        ----------
        cluster_se_vcov : float
            Cluster-robust variance estimate.
        cluster_se_result : float
            Clustered standard errors.

        Examples
        --------
        >>> import networkinference as ni
        >>> from networkinference.utils import FakeData
        >>> Y, X, A = FakeData.ols(network='RGG')
        >>> ols_model = ni.OLS(Y, X, A)
        >>> ols.get_clusters(30)
        >>> ols.cluster_se()
        """
        if not hasattr(self,'clusters'): self.get_clusters(num_clusters)
        if isinstance(self.invhessian, np.ndarray):
            self.cluster_se_vcov = self.data['Y'].size * self.invhessian.dot(core.cluster_var(self.scores, self.clusters)).dot(self.invhessian)
            self.cluster_se_result = np.sqrt(np.diag(self.cluster_se_vcov))
        else:
            self.cluster_se_vcov = core.cluster_var(self.summands, self.clusters)
            self.cluster_se_result = np.sqrt(self.cluster_se_vcov / self.summands.size)

        if self.estimate.size == 1:
            est = np.array([self.estimate])
            se = np.array([self.cluster_se_result])
        else:
            est = self.estimate
            se = self.cluster_se_result
        
        if verbose:
            CV = norm.ppf(1-0.05/2)
            fmat = '%.' + str(decimals) + 'f'
            table = []
            for k in range(est.size):
                CI = [est[k] - CV * se[k], est[k] + CV * se[k]]
                CI = [float(Decimal(fmat % CI[0])), float(Decimal(fmat % CI[1]))]
                table.append([est[k], se[k], CI])
            print(tabulate(table, headers=['Estimate', 'SE', '95% CI'], floatfmt='.' + str(decimals) + 'f'))

class TSLS(OLS):
    """2SLS estimator.

    Parameters 
    ----------
    Y : numpy float array
        n-dimensional array of outcomes.
    X : numpy float array
        n x k array of regressors (not including intercept) or n-dimensional array.
    W : numpy float array
        n x d array of instruments for d >= k (not including intercept) or n-dimensional array.
    A : NetworkX undirected graph
        Graph on n nodes. NOTE: Assumes nodes are labeled as integers 0, ..., n-1 in A, so that the outcome of node i is given by the ith component of Y. Network can be weighted or directed, although weights and directions are ignored when computing network SEs. Argument not used for dependence robust test or CI. Default value: None.

    Attributes
    ----------
    data : dictionary
        Stores all input data, adding a column of ones to X and W.
    summands : numpy array
        n-dimensional array of intermediate products used to compute 2SLS estimator.
    estimate : float
        2SLS estimator.
    resid : numpy array
        Regression residuals.
    invhessian : numpy array
        Inverse hessian matrix.
    scores : numpy array
        Regression scores.

    Examples
    --------
    >>> import networkinference as ni
    >>> from networkinference.utils import FakeData
    >>> Y, X, W, A = FakeData.tsls()
    >>> tsls_model = ni.TSLS(Y, X, W, A)
    >>> print(tsls_model.estimate)
    """

    def __init__(self, Y, X, W, A=None):
        """Stores inputs, computes estimator.
        """
        n = Y.size
        if X.ndim == 1:
            Xp = np.vstack([np.ones(n), X]).T
        elif X.ndim == 2:
            Xp = np.hstack([np.ones(n)[:,np.newaxis], X])
        if W.ndim == 1:
            Wp = np.vstack([np.ones(n), W]).T
        elif W.ndim == 2:
            Wp = np.hstack([np.ones(n)[:,np.newaxis], W])
        S = Wp.T.dot(Xp) # (d+1) x (k+1)
        P = inv(Wp.T.dot(Wp)) # (d+1) x (d+1)
        self.invhessian = inv(S.T.dot(P).dot(S)) # (k+1) x (k+1), Xp'Wp(Wp'Wp)^{-1}Wp'Xp matrix
        self.summands = Wp.dot(P).dot(S) * Y[:,np.newaxis] # n x (k+1), mean of this is Xp'Wp(Wp'Wp)^{-1}Wp'Y
        self.estimate = self.invhessian.dot(self.summands.sum(axis=0)) # (k+1) dimensional, 2SLS estimate
        self.resid = Y - Xp.dot(self.estimate) # residuals
        self.scores = Wp.dot(P).dot(S) * self.resid[:,np.newaxis]
        self.data = {'Y':Y, 'X':Xp, 'W':Wp, 'network':A}

    def est_by_cluster(self, dimension):
        """Returns array of 2SLS estimates, one for each cluster. This is a helper method used by arand_test() and arand_ci().

        Parameters
        ----------
        dimension : int
            Dimension of estimand being tested. Ignored if estimand is scalar. Default value: 0.

        Returns
        -------
        thetahat : numpy array
            L-dimensional array of OLS estimates, one for each of the L clusters.
        """
        thetahat = []
        for C in np.unique(self.clusters):
            members = np.where(self.clusters==C)[0]
            Yp = self.data['Y'][members]
            Xp = self.data['X'][members,:]
            Wp = self.data['W'][members,:]
            S = Wp.T.dot(Xp)
            P = np.linalg.pinv(Wp.T.dot(Wp))
            thetahat.append( np.linalg.pinv(S.T.dot(P).dot(S)).dot(S.T.dot(P).dot(Wp.T.dot(Yp[:,np.newaxis])))[dimension,0] )
        if len(thetahat) == 1:
            thetahat = thetahat[0]
        else:
            thetahat = np.array(thetahat)
        return thetahat

class IPW(OLS):
    """Horvitz-Thompson estimator (inverse probability weighting with known propensity scores). See e.g. [1]_ Formula: 

    .. math::
        \\frac{1}{n} \sum_{i=1}^n \left( \\frac{\\text{ind1}_i}{\\text{pscores1}_i} - \\frac{\\text{ind2}_i}{\\text{pscores2}_i} \\right) Y_i.

    Parameters 
    ----------
    Y : numpy float array
        n-dimensional array of outcomes.
    ind1 : numpy int array
        n-dimensional array of indicators for first exposure mapping.
    ind2 : numpy int array
        n-dimensional array of indicators for second exposure mapping.
    pscores1 : numpy float array
        n-dimensional array of propensity scores corresponding to first exposure mapping ind1. The ith component is node i's probability of exposure.
    pscores2 : numpy float array
        n-dimensional array of propensity scores corresponding to second exposure mapping ind2. The ith component is node i's probability of exposure. 
    A : NetworkX undirected graph
        Graph on n nodes. NOTE: Assumes nodes are labeled as integers 0, ..., n-1 in A, so that the outcome for node i is given by the ith component of Y. Network can be weighted or directed, although weights and directions are ignored when computing network SEs. Argument not used for dependence robust test or CI. Default value: None.

    Attributes
    ----------
    data : dictionary
        Stores all input data.
    summands : numpy array
        n-dimensional array whose mean is the IPW estimator.
    estimate : float
        IPW estimator.

    Examples
    --------
    >>> import networkinference as ni
    >>> from networkinference.utils import FakeData
    >>> Y, ind1, ind2, pscores1, pscores2, A = FakeData.ipw()
    >>> ipw_model = ni.IPW(Y, ind1, ind2, pscores1, pscores2, A)
    >>> print(ipw_model.estimate)

    References
    ----------
    .. [1] Leung, M. "Causal Inference Under Approximate Neighborhood Interference," Econometrica (forthcoming), 2021.
    """

    def __init__(self, Y, ind1, ind2, pscores1, pscores2, A=None):
        """Stores inputs, computes estimator.
        """
        self.data = {'Y':Y, 'ind1':ind1, 'ind2':ind2, 'pscores1':pscores1, 'pscores2':pscores2, 'network':A}
        weight1 = self.data['ind1'].copy().astype('float')
        weight2 = self.data['ind2'].copy().astype('float')
        weight1[weight1 == 1] = self.data['ind1'][weight1 == 1] / self.data['pscores1'][weight1 == 1]
        weight2[weight2 == 1] = self.data['ind2'][weight2 == 1] / self.data['pscores2'][weight2 == 1]
        self.summands = self.data['Y'] * (weight1 - weight2)
        self.estimate = self.summands.mean() # IPW estimate
        self.invhessian = 1

    def est_by_cluster(self, dimension):
        """Returns array of IPW estimators, one for each cluster. This is a helper method used by arand_test() and arand_ci().

        Parameters
        ----------
        dimension : int
            Argument ignored.

        Returns
        -------
        thetahat : numpy array
            L-dimensional array of means, one for each of the L clusters.
        """
        thetahat = []
        for C in np.unique(self.clusters):
            members = np.where(self.clusters==C)[0]
            Z = self.summands[members]
            thetahat.append( Z.mean() )
        if len(thetahat) == 1:
            thetahat = thetahat[0]
        else:
            thetahat = np.array(thetahat)
        return thetahat
