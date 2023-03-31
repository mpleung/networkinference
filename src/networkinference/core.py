import numpy as np, networkx as nx, matplotlib.pyplot as plt, seaborn as sns, itertools
from tabulate import tabulate
from scipy.stats import chi2, t
from scipy.sparse import csgraph, csr_matrix
from scipy.linalg import block_diag
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from sklearn.cluster import KMeans

class core:
    """
    """

    ####### Resampled statistics #######

    @staticmethod
    def drobust_test(Z, mu, alpha=0.05, beta=0.01, R=None, L=1000, seed=None):
        """Returns conclusion of dependence-robust test due to [1]_. Note that the output of the test is random by nature. L is the number of simulation draws, and larger values reduce the random variation of the test. 

        Test is implemented using the U-type statistic and randomized confidence function approach due to [2]_ discussed in Remark 2 of [1]_.

        Parameters
        ----------
        Z : numpy array
            n-dimensional array of scalar observations.
        mu : float
            Null hypothesis, e.g. the hypothesized mean of Z.
        alpha : float
            Significance level. Default value: 0.05.
        beta : float
            beta in Remark 2 of Leung (2021). The closer this is to alpha, the more conservative the test. Default value: 0.01.
        R : int
            Number of resampling draws for test statistic. Uses default if R=None. Default value: None. 
        L : int
            Number of resampling draws for randomized confidence function. The larger the value, the less random the output. Default value: 1000.
        seed : int
            Seed for resampling draws. Set to None to not set a seed. Default value: None.

        Returns
        -------
        string
            Reject or not reject.

        Examples
        --------
        >>> import networkinference as ni
        >>> import numpy as np
        >>> Z = np.random.normal(size=500)
        >>> ni.core.drobust_test(Z, 0)

        References
        ----------
        .. [1] Leung, M. "Dependence-Robust Inference Using Resampled Statistics," Journal of Applied Econometrics (forthcoming), 2021.
        .. [2] Song, K. "Ordering-Free Inference from Locally Dependent Data," UBC working paper, 2016. 
        """
        if beta <= 0 or beta > alpha:
            raise ValueError('Must specify a value for beta in (0,alpha].')
        if R==None: R = Z.size
        np.random.seed(seed=seed)
        Zc = (Z - mu) / Z.std() 
        BC = np.sqrt(R) / Z.size
        allPairs = np.array(list(itertools.combinations(Zc, 2))) # array of all pairs of Zc
        draws = np.random.choice(allPairs.prod(axis=1), R*L).reshape(L,R).sum(axis=1) / np.sqrt(R) - BC
        RCF = (np.power(draws,2) <= chi2.ppf(1-alpha+beta,1)).mean() 
        concl = 'Not reject' if RCF >= 1-alpha else 'Reject'
        return concl

    @staticmethod
    def drobust_ci(Z, grid_start, grid_stop, grid_size=151, coverage=0.95, beta=0.01, R=None, L=1000, seed=None): 
        """Returns confidence interval (CI) derived from the dependence-robust test due to [1]_. Note that the output of the test is random by nature. L is the number of simulation draws, and larger values reduce the random variation of the test. If the result is a trivial interval, try increasing grid_size.

        Test is implemented using the U-type statistic and randomized confidence function approach due to [2]_ discussed in Remark 2 of [1]_.

        Parameters
        ----------
        Z : numpy array
            n-dimensional array of scalar observations.
        grid_start : float
            Need to specify a grid of values to test for inclusion in the CI. This is the leftmost point of the grid.
        grid_stop : float
            Rightmost point of the grid.
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
            Seed for resampling draws. Set to None to not set a seed. Default value: None.

        Returns
        -------
        list 
            Confidence interval.

        Examples
        --------
        >>> import networkinference as ni
        >>> import numpy as np
        >>> Z = np.random.normal(size=500)
        >>> ni.core.drobust_ci(Z, -2, 2)

        References
        ----------
        .. [1] Leung, M. "Dependence-Robust Inference Using Resampled Statistics," Journal of Applied Econometrics (forthcoming), 2021. 
        .. [2] Song, K. "Ordering-Free Inference from Locally Dependent Data," UBC working paper, 2016.
        """
        if beta <= 0 or beta > 1-coverage:
            raise ValueError('Must specify a value for beta in (0,1-coverage].') 
        if grid_stop < grid_start: raise IndexError('Grid start point must be smaller than grid endpoint.')
        np.random.seed(seed=seed)
        n = Z.size
        if R==None: R = Z.size
        BC = np.sqrt(R) / n
        sigma = Z.std()
        allPairs = np.array(list(itertools.combinations(Z/sigma, 2))) 
        indices = np.random.choice(range(allPairs.shape[0]), R*L).reshape(L,R)
        CV = chi2.ppf(coverage+beta,1)

        CI_L = grid_start # left endpoint of CI
        grid = np.linspace(grid_start, grid_stop, grid_size)
        for b in range(1,grid_size):
            mu = grid[b]
            ZProd = (allPairs - mu/sigma).prod(axis=1)
            draws = ZProd[indices].sum(axis=1) / np.sqrt(R) - BC
            RCF = (np.power(draws,2) <= CV).mean() 
            if RCF >= coverage: # if not reject
                CI_L = mu
                break

        CI_R = grid_stop # right endpoint of CI
        for b in range(grid_size-1):
            mu = grid[grid_size - 1 - b]
            if mu == CI_L: 
                CI_R = mu
                break
            ZProd = (allPairs - mu/sigma).prod(axis=1)
            draws = ZProd[indices].sum(axis=1) / np.sqrt(R) - BC
            RCF = (np.power(draws,2) <= CV).mean() 
            if RCF >= coverage: 
                CI_R = mu
                break

        return [CI_L,CI_R]

####### HAC #######

    @staticmethod
    def network_hac(Z, A, b=None, disp=False):
        """Returns network HAC variance estimator due to [1]_ (also see [2]_). Setting b=0 and A = any value (e.g. None) outputs the conventional heteroskedasticity-robust variance estimator for i.i.d. data. Network is converted to an unweighted, undirected version by dropping edge weights and directionality of links.

        Parameters
        ----------
        Z : numpy array
            n-dimensional array of scalar observations or n x k matrix of n k-dimensional observations.
        A : NetworkX graph
            Graph on n nodes. NOTE: Assumes nodes are labeled as integers 0, ..., n-1 in A, so that the data for node i is given by the ith component of Z.
        b : float
            HAC bandwidth. Recommend keeping b=None, which uses the bandwidth choice recommended by [2]_. Default value: None.
        disp : boolean
            If False, the function only returns HAC. If True, the function returns (HAC, APL, b, PD_failure). Default value: False.

        Returns
        -------
        HAC : numpy array
            Estimate of variance-covariance matrix.
        APL : float
            Average path length of A.
        b : int
            Bandwidth.
        PD_failure : boolean
            True if substitute positive definite variance estimator needed to be used.

        References
        ----------
        .. [1] Kojevnikov, D., V. Marmer, and K. Song, "Limit Theorems for Network Dependent Random Variables," Journal of Econometrics, 2021, 222 (2), 882-908.
        .. [2] Leung, M. "Causal Inference Under Approximate Neighborhood Interference," Econometrica (forthcoming), 2021.

        Examples
        --------
        >>> import networkinference as ni
        >>> from networkinference.utils import FakeData
        >>> import numpy as np
        >>> Z = np.random.normal(size=500)
        >>> A = FakeData.erdos_renyi(500)
        >>> HAC = ni.core.network_hac(Z, A)
        """
        n = Z.shape[0]
        PD_failure = False

        if b == 0:  # iid SE
            weights = np.eye(n)
            APL = 0
        else:
            G = nx.to_scipy_sparse_array(A.to_undirected(as_view=True), nodelist=range(n), weight=None, format='csr') # sparse matrix representation
            dist_matrix = csgraph.dijkstra(csgraph=G, directed=False, unweighted=True) # path distance matrix
            Gcc = [A.subgraph(c).copy() for c in sorted(nx.connected_components(A), key=len, reverse=True)]
            giant = [i for i in Gcc[0]] # set of nodes in giant component
            APL = dist_matrix[np.ix_(giant,giant)].sum() / len(giant) / (len(giant)-1) # average path length

            if b==None: 
                avg_deg = G.dot(np.ones(n)[:,None]).mean() 
                exp_nbhd = APL < 2 * np.log(n) / np.log(avg_deg)
                b = round(APL/2) if exp_nbhd else round(APL**(1/3)) # default bandwidth

            weights = dist_matrix <= b 

        # default variance estimator (not guaranteed PD)
        Zc = Z - Z.mean(axis=0)
        HAC = Zc.T.dot(weights.dot(Zc)) / n 

        # PD variance estimator from the first (v1) arXiv draft of Leung (2019), "Causal Inference Under Approximate Neighborhood Interference"
        if Z.ndim == 1:
            PD_failure = HAC <= 0
        elif Z.ndim == 2:
            PD_failure = np.any(np.linalg.eigvals(HAC) <= 0)
        if PD_failure:
            if b==None: 
                avg_deg = np.array([i[1] for i in A.degree]).mean()
                exp_nbhd = APL < 2 * np.log(n) / np.log(avg_deg)
                b = round(APL/4) if exp_nbhd else round(APL**(1/3)) # default bandwidth
            b_neighbors = dist_matrix <= b
            row_sums = np.squeeze(b_neighbors.dot(np.ones(Z.shape[0])[:,None]))
            b_norm = b_neighbors / np.sqrt(row_sums)[:,None]
            weights = b_norm.dot(b_norm.T)
            HAC = Zc.T.dot(weights.dot(Zc)) / n

        if disp:
            return HAC, APL, b, PD_failure
        else:
            return HAC 

    ####### Clustering #######

    @staticmethod
    def sumstats(A, decimals=3):
        """Prints table of network summary statistics.

        Parameters
        ----------
        A : NetworkX undirected, unweighted graph
        decimals : int
            Number of decimals to which to round the output.

        Examples
        --------
        >>> import networkinference as ni
        >>> from networkinference.utils import FakeData
        >>> A = FakeData.erdos_renyi(500)
        >>> ni.core.sumstats(A) 
        """
        numnodes = A.number_of_nodes()
        numedges = A.number_of_edges()

        clustering = nx.average_clustering(A)

        Gcc = [A.subgraph(c).copy() for c in sorted(nx.connected_components(A), key=len, reverse=True)]
        giant = Gcc[0].to_undirected()
        diam = nx.diameter(giant)
        APL = nx.average_shortest_path_length(giant)
        giant_size = len(giant)
        ccount = len(Gcc)

        deg_seq = np.array([i[1] for i in A.degree])
        num_isos = (deg_seq == 0).sum()
        max_deg = deg_seq.max()
        avg_deg = deg_seq.mean()

        labels = ['# Units', '# Links', 'Average Degree', 'Max Degree', '# Isolates', 'Giant Size', 'Diameter', 'Average Path Length', '# Components', 'Clustering']
        numbers = [numnodes, numedges, avg_deg, max_deg, num_isos, giant_size, diam, APL, ccount, clustering]
        table = np.vstack([labels, numbers]).T
        print(tabulate(table, headers=['Summary Statistics', ''], floatfmt='.' + str(decimals) + 'f'))

    @staticmethod
    def plot_spectrum(A, giant=True, weight=None, xlim_scat_buffer=0.03, ylim_scat_buffer=0.03, \
            xticks_scat=3, yticks_scat=3, xticks_hist=3, binwidth=None, binrange=None, figsize=(10, 4), \
            title_hist='Histogram', title_scat='Scatterplot', title_y='Eigenvalues', sns_style='dark'):
        """Plots spectrum of the normalized Laplacian in a scatterplot and histogram.
        
        Parameters
        ----------
        A : NetworkX graph
            Can be weighted or directed.
        giant : boolean
            Set to True to plot spectrum of the giant component. Set to False to plot spectrum of the full graph. Default value: True.
        weight : string
            Specifies how edge weights are labeled in A, if A is a weighted graph. Default value: None.
        xlim_scat_buffer : float [0,1]
            Larger value adds more whitespace before and after the leftmost and rightmost points of the scatterplot.
        ylim_scat_buffer : float [0,1]
            Larger value adds more whitespace below and above the bottommost and topmost points of the scatterplot.
        xticks_scat : int
            Number of tick marks on x-axis of scatterplot.
        yticks_scat : int
            Number of tick marks of y-axis of scatterplot.
        xticks_hist : int
            Number of tick marks on x-axis of histogram.
        binwidth : int
            Width of histogram bins.
        binrange : int
            Range of histogram bins.
        figsize : tuple of ints
            Size of figure.
        title_hist : string
            Title of histogram.
        title_scat : string
            Title of scatterplot.
        title_y : string
            Title of scatterplot y-axis.
        sns_style : string
            Seaborn style of figures.

        Examples
        --------
        >>> import networkinference as ni
        >>> from networkinference.utils import FakeData
        >>> A = FakeData.erdos_renyi(500)
        >>> ni.core.plot_spectrum(A)
        """
        if giant:
            Gcc = [A.subgraph(c).copy() for c in sorted(nx.connected_components(A), key=len, reverse=True)]
            G = Gcc[0]
        else:
            G = A
        n = G.number_of_nodes()
        L = csgraph.laplacian(csr_matrix(nx.to_scipy_sparse_array(G, weight=weight, format='csr')), normed=True)
        ivals = eigh(L.todense(), eigvals_only=True)

        sns.set_theme(style='dark')
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        axes[0].set_title(title_scat)
        axes[0].set(ylabel=title_y)
        yscat_buffer = (np.max(ivals)-np.min(ivals))*ylim_scat_buffer
        axes[0].set(ylim=(np.min(ivals)-yscat_buffer, np.max(ivals)+yscat_buffer))
        axes[0].set_yticks(np.linspace(np.min(ivals),np.max(ivals),yticks_scat))
        xscat_buffer = n*xlim_scat_buffer
        axes[0].set(xlim=(-xscat_buffer,n+xscat_buffer))
        axes[0].set_xticks(np.linspace(0,n,xticks_scat))
        sns.scatterplot(x=np.arange(n), y=ivals, linewidth=0, s=10, ax=axes[0])

        maxcount = np.max(np.histogram(ivals)[0])
        axes[1].set(xlim=(np.min(ivals), np.max(ivals)))
        axes[1].set_xticks(np.linspace(np.min(ivals),np.max(ivals),xticks_hist))
        axes[1].set_title(title_hist)
        sns.histplot(data=ivals, binwidth=binwidth, binrange=binrange, ax=axes[1])

        plt.tight_layout()
        plt.show()

    @staticmethod
    def spectrum(A, giant=True, weight=None):
        """Returns spectrum of the normalized Laplacian.

        Parameters
        ----------
        A : NetworkX graph
            Can be weighted or directed.
        giant : boolean
            Set to True to only return spectrum of the giant component. Set to False to return spectrum of the full graph. Default value: True.
        weight : string
            Specifies how edge weights are labeled in A, if A is a weighted graph. Default value: None.

        Returns
        -------
        numpy array
            Eigenvalues.

        Examples
        --------
        >>> import networkinference as ni
        >>> from networkinference.utils import FakeData
        >>> A = FakeData.erdos_renyi(500)
        >>> ivals = ni.core.spectrum(A)
        """
        if giant:
            Gcc = [A.subgraph(c).copy() for c in sorted(nx.connected_components(A), key=len, reverse=True)]
            G = Gcc[0]
        else:
            G = A
        L = csgraph.laplacian(csr_matrix(nx.to_scipy_sparse_array(G, weight=weight, format='csr')), normed=True)
        return eigh(L.todense(), eigvals_only=True)

    @staticmethod
    def spectral_clustering(num_clusters, A, seed=None):
        """Returns network clusters obtained from normalized spectral clustering algorithm due to [1]_ (also see [2]_). All nodes not in the giant component are grouped into a single cluster. NOTE: Assumes nodes are labeled as integers 0, ..., n-1 in A.

        Parameters
        ----------
        num_clusters : int
            Number of desired clusters in the giant component.
        A : NetworkX graph 
            Graph on n nodes. Can be weighted or directed.
        seed : int
            Seed for k-means clustering initialization. Set to None to not set a seed. Default value: None.

        Returns
        -------
        numpy array
            n-dimensional array of cluster labels from 0 to num_clusters-1

        Examples
        --------
        >>> import networkinference as ni
        >>> from networkinference.utils import FakeData
        >>> A = FakeData.erdos_renyi(500)
        >>> clusters = ni.core.spectral_clustering(10, A)

        References
        ----------
        .. [1] Ng, A., M. Jordan, Y. Weiss, "On Spectral Clustering: Analysis and an Algorithm." Advances in Neural Information Processing Systems, 2002, 849-856. 
        .. [2] von Luxburg, U., "A Tutorial on Spectral Clustering," Statistics and Computing, 2007, 17 (4), 395-416.
        """
        A_components = [A.subgraph(c).copy() for c in sorted(nx.connected_components(A), key=len, reverse=True)]
        A_giant = A_components[0]
        L = csgraph.laplacian(csr_matrix(nx.to_scipy_sparse_array(A_giant, format='csr')), normed=True) # sparse laplacian matrix of the network restricted to the giant component

        ivals, ivecs = eigsh(L, k=num_clusters, which='SM')
        ivecs /= np.sqrt( (ivecs**2).sum(axis=1) )[:,None] # row normalize by row norm
        kmeans = KMeans(num_clusters, n_init=30, random_state=seed).fit(ivecs)
        clusters = kmeans.labels_ # spectral clustering only the giant component

        # assign labels to nodes not in the giant
        all_clusters = np.ones(A.number_of_nodes()).astype('float') * clusters.max()+1
        all_clusters[list(A_giant.nodes)] = clusters
        
        return all_clusters 

    @staticmethod
    def conductance(clusters, A, weight=None):
        """Returns maximal conductance of a set of clusters. For cluster-robust methods to work, conductance should be at most 0.1, as recommended by [1]_.

        Parameters
        ----------
        clusters : numpy array
            n-dimensional array of cluster labels for all n nodes, assumed to be 0, ..., L-1 where L is the number of clusters.
        A : NetworkX graph
            Graph on n nodes. Can be weighted or directed.
        weight : string
            Specifies how edge weights are labeled in A, if A is a weighted graph. Default value: None.

        Returns
        -------
        float
            Maximal conductance of the clusters. 

        Examples
        --------
        >>> import networkinference as ni
        >>> from networkinference.utils import FakeData
        >>> A = FakeData.erdos_renyi(500)
        >>> clusters = ni.core.spectral_clustering(10, A)
        >>> ni.core.conductance(clusters, A)

        References
        ----------
        .. [1] Leung, M., "Network Cluster-Robust Inference," arXiv preprint arXiv:2103.01470, 2021.
        """
        num_clusters = int(np.max(clusters) + 1)
        conductances = np.zeros(num_clusters) # record conductance of each cluster
        for i in range(num_clusters):
            S = np.where(clusters==i)[0]
            conductances[i] = nx.cut_size(A, S, weight) / np.maximum(nx.volume(A, S, weight), 1)
        return np.max(conductances)

    @staticmethod
    def cluster_var(Z, clusters):
        """Returns conventional cluster-robust variance estimator.

        Parameters
        ----------
        Z : numpy array
            n-dimensional array of scalar observations or n x k matrix of n k-dimensional observations. 
        clusters : numpy array
            n-dimensional array of cluster labels for all n nodes, assumed to be 0, ..., L-1 where L is the number of clusters. 

        Returns
        -------
        numpy array
            Variance-covariance matrix.

        Examples
        --------
        >>> import networkinference as ni
        >>> from networkinference.utils import FakeData
        >>> import numpy as np
        >>> Z = np.random.normal(size=500)
        >>> A = FakeData.erdos_renyi(500)
        >>> clusters = ni.core.spectral_clustering(10, A)
        >>> var = ni.core.cluster_var(Z, clusters)
        """
        n = Z.shape[0]
        Zc = Z - Z.mean(axis=0)
        cluster_sizes = np.array([(clusters==i).sum() for i in np.unique(clusters)])
        blocks = [np.ones((i,i)) for i in cluster_sizes]
        weights = block_diag(*blocks)
        return Zc.T.dot(weights.dot(Zc)) / n

    @staticmethod
    def trobust_ci(Z, coverage=0.95, verbose=True):
        """Returns CI from the t-statistic based cluster-robust procedure due to [1]_. The larger the dimension of Z (i.e. more clusters), the more powerful the test. However, since the test computes estimates cluster by cluster, the output can be more unstable with a larger number of clusters since the sample size within each cluster can be small.

        Parameters
        ----------
        Z : numpy array
            q-dimensional array of estimates, one for each of the q clusters. 
        coverage : float
            Desired coverage. Default value: 0.95.
        verbose : boolean
            If True, calling this function prints out the results. Default value: True.

        Returns
        -------
        CI : list
            Confidence interval.

        Examples
        --------
        >>> import networkinference as ni
        >>> from networkinference.utils import FakeData
        >>> import numpy as np
        >>> Z = np.random.normal(size=10)
        >>> ni.core.trobust(Z)

        References
        ----------
        .. [1] Ibragimov, R. and U. Mueller, "t-Statistic Based Correlation and Heterogeneity Robust Inference," Journal of Business and Economic Statistics, 2010, 28 (4), 453-468.
        """
        if 1-coverage > 0.08326:
            raise ValueError('alpha must be less than 0.08326.')
        mean, SE = Z.mean(), Z.std() / np.sqrt(Z.size)
        CV = t.ppf(1-(1-coverage)/2, Z.size-1)
        CI = [mean - CV * SE, mean + CV * SE]
        if verbose:
            fmat = '%.' + str(decimals) + 'f'
            CI_formatted = [float(Decimal(fmat % CI[0])), float(Decimal(fmat % CI[1]))]
            print(tabulate([mean, SE, CI_formatted], headers=['Mean', 'SE', '95% CI'], floatfmt='.' + str(decimals) + 'f'))
        return CI

    @staticmethod
    def arand_test(Z, mu, seed=None):
        """Returns p-value and test statistic of approximate randomization test [1]_. The larger the dimension of Z (i.e. more clusters), the more powerful the test. However, since the test computes estimates cluster by cluster, the output can be more unstable with a larger number of clusters since the sample size within each cluster can be small.

        Parameters
        ----------
        Z : numpy array
            q-dimensional array of estimates, one for each of the q clusters.
        mu : float
            Scalar null-hypothesized value of the mean of Z.
        seed : int
            Seed for drawing permutations, which is only relevant when the dimension of Z exceeds 12. Set to None to not set a seed. Default value: None.

        Returns
        -------
        p_value : float
            P-value of test.
        T_wald : float
            Test statistic.

        Examples
        --------
        >>> import networkinference as ni
        >>> import numpy as np
        >>> Z = np.random.normal(size=10)
        >>> ni.core.arand_test(Z, 0)

        References
        ----------
        .. [1] Canay, I., J. Romano, and A. Shaikh, "Randomization Tests Under an Approximate Symmetry Assumption," Econometrica, 2017, 85 (3), 1013-1030.
        """
        q = Z.size
        if q <= 12: # if there are less than 12 clusters, use all possible permutations 
            ph = q*[1]
            G = [tuple(ph), tuple(q*[-1])] # list of permutations
            for j in range(q-1):
                ph[j] = -1
                G += list(perm_unique(ph))
            G = np.array(G)
        else: # if there are more than 12 clusters, sample permutations at random
            np.random.seed(seed=seed)
            G = np.random.binomial(1,0.5,size=(5000,q))
            G[G == 0] = -1

        Zc = Z - mu
        rand_dist = np.array([test_stat(G[g] * Zc) for g in range(G.shape[0])])
        T_wald = test_stat(Zc)
        p_value = (rand_dist >= T_wald).mean()
        return p_value, T_wald 

    @staticmethod
    def arand_ci(Z, grid_start, grid_stop, grid_size=151, coverage=0.95, seed=None): 
        """Returns confidence interval (CI) obtained by inverting an approximate randomization test [1]_. If the result is a trivial interval, try increasing grid_size. The larger the dimension of Z (i.e. more clusters), the narrower the CI. However, since the test computes estimates cluster by cluster, the output can be more unstable with a larger number of clusters since the sample size within each cluster can be small.

        Parameters
        ----------
        Z : numpy array
            q-dimensional array containing estimator for each of the q clusters. 
        grid_start : float
            Need to specify a grid of values over which to invert the test. This is the leftmost point of the grid.
        grid_stop : float
            Rightmost point of the grid.
        grid_size : int
            Number of points in the grid. Default value: 151.
        coverage : float
            Desired coverage. Default value: 0.95.
        seed : int
            Seed for drawing permutations, which is only relevant when the dimension of Z exceeds 12. Set to None to not set a seed. Default value: None.

        Returns
        -------
        list 
            Confidence interval.

        Examples
        --------
        >>> import networkinference as ni
        >>> import numpy as np
        >>> Z = np.random.normal(size=10)
        >>> ni.core.arand_ci(Z, -2, 2)

        References
        ----------
        .. [1] Canay, I., J. Romano, and A. Shaikh, "Randomization Tests Under an Approximate Symmetry Assumption," Econometrica, 2017, 85 (3), 1013-1030. 
        """
        if grid_stop < grid_start: raise IndexError('Grid start point must be smaller than grid endpoint.')

        q = Z.size
        if q <= 12: # if there are less than 12 clusters, use all possible permutations
            ph = q*[1]
            G = [tuple(ph), tuple(q*[-1])] # list of permutations
            for j in range(q-1):
                ph[j] = -1
                G += list(perm_unique(ph))
            G = np.array(G)
        else: # if there are more than 12 clusters, sample permutations at random
            np.random.seed(seed=seed)
            G = np.random.binomial(1,0.5,size=(5000,q))
            G[G == 0] = -1

        # TO DO: improve computational speed using bisection search
        CI_L = grid_start # left endpoint of CI
        grid = np.linspace(grid_start, grid_stop, grid_size)
        for b in range(1,grid_size):
            mu = grid[b]
            Zc = Z - mu
            rand_dist = np.array([test_stat(G[g] * Zc) for g in range(G.shape[0])])
            T_wald = test_stat(Zc)
            pval = (rand_dist >= T_wald).mean()
            if pval > 1-coverage: # if not reject
                CI_L = mu
                break

        CI_R = grid_stop # right endpoint of CI
        for b in range(grid_size-1):
            mu = grid[grid_size - 1 - b]
            if mu == CI_L: 
                CI_R = mu
                break
            Zc = Z - mu
            rand_dist = np.array([test_stat(G[g] * Zc) for g in range(G.shape[0])])
            T_wald = test_stat(Zc)
            pval = (rand_dist >= T_wald).mean()
            if pval > 1-coverage: # if not reject
                CI_R = mu
                break

        return [CI_L,CI_R]

def test_stat(Z):
    # Test statistic for core.arand_test() and core.arand_ci().
    return np.abs(Z.mean()) 

class UniqueElement:
    def __init__(self,value,occurrences):
        self.value = value
        self.occurrences = occurrences

def perm_unique(elements):
    # Function used in core.arand_test() and core.arand_ci().
    # Code from https://stackoverflow.com/questions/6284396/permutations-with-unique-values
    eset=set(elements)
    listunique = [UniqueElement(i,elements.count(i)) for i in eset]
    u=len(elements)
    return perm_unique_helper(listunique,[0]*u,u-1)

def perm_unique_helper(listunique,result_list,d):
    if d < 0:
        yield tuple(result_list)
    else:
        for i in listunique:
            if i.occurrences > 0:
                result_list[d]=i.value
                i.occurrences-=1
                for g in  perm_unique_helper(listunique,result_list,d-1):
                    yield g
                i.occurrences+=1

