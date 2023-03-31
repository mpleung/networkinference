import numpy as np, networkx as nx, math
from scipy.sparse.linalg import inv
from scipy.stats import binom
from scipy.spatial import cKDTree
from scipy.sparse import csgraph, csc_matrix, identity

def nhbr_mean(X, A, distance=1, weight=None):
    """Returns array where the ith entry is the average X of network neighbors at a certain distance from node i.

    This is used, for example, to create a data matrix for estimating linear-in-means models. For distance=2, this returns the average X of network two-neighbors / friends-of-friends, a common instrument used for estimating linear-in-means models (see [1]_, [2]_). Note: the function converts the network to an unweighted, undirected version by dropping edge weights and directionality of links.

    Parameters
    ----------
    X : numpy array
         n-dimensional array of scalar observations or an n x k matrix of n k-dimensional observations.
    A : NetworkX graph
        Network on n nodes. Can be weighted or directed. NOTE: Assumes nodes are labeled 0 through n-1, so that the data for node i is given by the ith component of X. 
    weight : string
            Label of edge weights in A, if A is a weighted graph. Default value: None.

    Returns
    -------
    Xbar : numpy array
        n-dimensional array of scalar observations or an n x k matrix of n k-dimensional observations, where the ith row is the average X of i's friends.

    Examples
    --------
    >>> from networkinference.utils import FakeData, nhbr_mean
    >>> import numpy as np
    >>> A = FakeData.erdos_renyi()
    >>> X = np.random.normal(100)
    >>> Xbar = nhbr_mean(X, A)

    References
    ----------
    .. [1] Bramoullé, Y., H. Djebbari, and B. Fortin, "Identification of Peer Effects Through Social Networks," Journal of Econometrics, 2009, 150 (1), 41-55.
    .. [2] De Giorgi, G., M. Pellizzari, and S. Redaelli, "Identification of Social Interactions Through Partially Overlapping Peer Groups," American Economic Journal: Applied Economics, 2010, 2 (2), 241-75.
    """
    n = A.number_of_nodes()
    if distance == 1:
        A_mat = nx.to_scipy_sparse_array(A, nodelist=range(n), weight=weight, format='csc')
        delta = A_mat.dot(np.ones(n)) # number of friends of each node
        Xbar = (A_mat.dot(X)).astype('float')
    else:
        A_mat = nx.to_scipy_sparse_array(A.to_undirected(as_view=True), nodelist=range(n), weight=None, format='csc')
        dist_matrix = csgraph.dijkstra(csgraph=A_mat, directed=False, unweighted=True)
        delta = (dist_matrix == distance).dot(np.ones(n)) 
        Xbar = ((dist_matrix == distance).dot(X)).astype('float')

    if X.ndim == 1:
        Xbar[delta != 0] = Xbar[delta != 0] / delta[delta != 0]
    elif X.ndim == 2:
        Xbar[delta != 0,:] = Xbar[delta != 0,:] / delta[delta != 0, np.newaxis]
    else:
        raise ValueError('X must be 1 or 2-dimensional.')
    return Xbar

def adjrownorm(A, weight=None):
    """Returns row-normalized adjacency matrix of NetworkX graph.
    
    Parameters
    ----------
    A : NetworkX graph
        Can be weighted or directed.
    weight : string
        Label of edge weights in A, if A is a weighted graph. Default value: None.

    Returns
    -------
    scipy sparse matrix
        Row-normalized adjacency matrix.

    Examples
    --------
    >>> from networkinference.utils import FakeData, adjrownorm 
    >>> A = FakeData.erdos_renyi()
    >>> A_norm = adjrownorm(A)
    """
    A_mat = nx.to_scipy_sparse_array(A, nodelist=range(A.number_of_nodes()), weight=weight, format='csc')
    deg_seq_sim = A_mat.dot(np.ones(A.number_of_nodes()))
    r,c = A_mat.nonzero() 
    rD_sp = csc_matrix(((1.0/np.maximum(deg_seq_sim,1))[r], (r,c)), shape=(A_mat.shape))
    return A_mat.multiply(rD_sp) 

class FakeData:
    """Methods for simulating data.
    """

    @staticmethod
    def linear_in_means(A, theta=np.array([1,0.5,3,1])):
        """Returns a scalar outcome generated from a linear-in-means model with scalar covariate. Covariates and errors are i.i.d. standard normal. Outcomes are generated from the following linear-in-means model:

        .. math::
            Y_i = \\alpha + \\beta \\frac{\sum_{j=1}^n A_{ij} Y_j}{\sum_{j=1}^n A_{ij}} + \delta \\frac{\sum_{j=1}^n A_{ij} X_j}{\sum_{j=1}^n A_{ij}} + \gamma X_i + \\varepsilon_i,

        where :math:`A_{ij}` is an indicator for whether nodes i and j are linked, :math:`\\alpha` is the intercept, :math:`\\beta` the endogenous peer effect, and :math:`\delta` the exogenous peer effect. The default parameter values are 1, 0.5, 3, and 1, respectively.

        Parameters
        ----------
        A : NetworkX graph
            Network on n units. Can be weighted or directed.
        theta : numpy array
            4-dimensional array of model parameters. Default value: np.array([1,0.5,3,1]).

        Returns
        -------
        Y : numpy array
            n-dimensional array of outcomes.
        X : numpy array
            n-dimensional array of covariates.

        Examples
        --------
        >>> from networkinference.utils import FakeData 
        >>> A = FakeData.erdos_renyi()
        >>> Y, X = FakeData.linear_in_means(A)
        """
        n = A.number_of_nodes()
        X = np.random.normal(size=n) 
        errors = np.random.normal(size=n)
        A_norm = adjrownorm(A)
        eps = errors + A_norm.dot(errors)
        LIM_inv = inv( identity(n,format='csc') - theta[1]*A_norm )
        Y = LIM_inv.dot( (theta[0] + theta[2]*A_norm.dot(X) + theta[3]*X + eps) )
        return Y, X

    @staticmethod
    def random_geometric(n=500, avg_deg=5, seed=None):
        """Returns a random geometric graph on n nodes. Nodes are randomly positioned in :math:`[0,1]^2` and form links with all alters within a certain radius.

        Parameters
        ----------
        n : int
            Number of nodes. Default value: 500.
        avg_deg : int
            Desired average degree of output graph. Default value: 5.
        seed : int
            Seed for random positions. Set to None to not set a seed. Default value: None.

        Returns
        -------
        RGG : NetworkX graph
            Unweighted and undirected graph on n nodes.

        Examples
        --------
        >>> from networkinference.utils import FakeData
        >>> A = FakeData.random_geometric()
        """
        np.random.seed(seed=seed)
        r = (avg_deg/math.pi/n)**(1/2)
        positions = np.random.uniform(size=(n,2))
        kdtree = cKDTree(positions)
        pairs = kdtree.query_pairs(r) # Euclidean norm
        RGG = nx.empty_graph(n=positions.shape[0], create_using=nx.Graph())
        RGG.add_edges_from(list(pairs))
        return RGG

    @staticmethod
    def erdos_renyi(n=500, avg_deg=5, seed=None):
        """Returns an Erdos-Renyi graph on n nodes with linking probability avg_deg / n. Just a wrapper for a NetworkX function.

        Parameters
        ----------
        n : int
            Number of nodes. Default value: 500.
        avg_deg : int
            Desired average degree of output graph. Default value: 5.
        seed : int
            Seed for random links. Set to None to not set a seed. Default value: None.

        Returns
        -------
        NetworkX graph
            Undirected and unweighted graph on n nodes.

        Examples
        --------
        >>> from networkinference.utils import FakeData
        >>> A = FakeData.erdos_renyi()
        """
        return nx.fast_gnp_random_graph(n, avg_deg/n, seed=seed)

    @staticmethod
    def ipw(n=500, network='RGG', avg_deg=5, p = 0.15, seed=None):
        """Returns data (Y, ind1, ind2, pscores1, pscores2, A) to input into IPW class. Outcome model is

        .. math::
            Y_i = \left( \\beta_i + \\frac{\sum_{j=1}^n A_{ij} \\beta_j}{\sum_{j=1}^n A_{ij}} \\right) + \mathbf{1}\left\{\sum_{j=1}^n A_{ij} D_j > 0\\right\} + \left( \\varepsilon_i + \\frac{\sum_{j=1}^n A_{ij} \\varepsilon_j}{\sum_{j=1}^n A_{ij}} \\right)

        where :math:`A_{ij}` is an indicator for whether nodes i and j are linked, :math:`\\beta_i \stackrel{iid}\sim \mathcal{N}(1,1)`, :math:`\\varepsilon_i` is i.i.d. standard normal, and :math:`D_i` is i.i.d. Bernoulli with success probability p.

        Parameters
        ----------
        n : int
            Number of observations. Default value: 500.
        network : string
            Type of network to generate. Options: 'ER' (Erdos-Renyi) and 'RGG' (random geometric graph). Default value: 'RGG'.
        avg_deg : int
            Desired average degree of output graph. Default value: 5.
        p : float [0,1]
            Treatment probability.
        seed : int
            Seed for randomness. Set to None to not set a seed. Default value: None.

        Returns
        -------
        Y : numpy array
            n-dimensional array of outcomes generated from a linear-in-means model.
        ind1 : numpy int array
            n-dimensional array of indicators for having at least one treated friend.
        ind2 : numpy int array
            n-dimensional array of indicators for having no treated friends.
        pscores1 : numpy float array
            n-dimensional array of probabilities of having at least one treated friend.
        pscores2 : numpy float array
            n-dimensional array of probabilities of having no treated friends.
        A : NetworkX graph
            Undirected, unweighted graph on n nodes. 

        Examples
        --------
        >>> from networkinference.utils import FakeData
        >>> Y, ind1, ind2, pscores1, pscores2, A = FakeData.ipw()
        """
        if network=='ER':
            A = FakeData.erdos_renyi(n, avg_deg, seed)
        else:
            A = FakeData.random_geometric(n, avg_deg, seed)
        np.random.seed(seed=seed) 
        A_mat = nx.to_scipy_sparse_array(A, nodelist=range(n), format='csc')
        A_norm = adjrownorm(A)
        D = np.random.binomial(1,p,n) # treatments
        b0 = np.random.normal(1,1,size=n)
        eps = np.random.normal(0,1,size=n)
        friends_treated = A_mat.dot(D) # number of friends treated
        Y = (b0+A_norm.dot(b0)) * (friends_treated > 0).astype('float') + (eps+A_norm.dot(eps)) # outcomes

        degrees = A_mat.dot(np.ones(n)) # number of friends
        pscores2 = binom(degrees,p).pmf(0)
        pscores1 = 1 - binom(degrees,p).pmf(0)
        ind1 = (friends_treated > 0).astype('float') # exposure mapping indicators for spillover effect
        ind2 = 1 - ind1

        return Y, ind1, ind2, pscores1, pscores2, A

    @staticmethod
    def ols(n=500, network='RGG', avg_deg=5, seed=None):
        """Returns data (Y, X, A) to input into OLS class. Outcome model is 

        .. math ::
            Y_i = 1 + \left( X_i + \\frac{\sum_{j=1}^n A_{ij} X_j}{\sum_{j=1}^n A_{ij}} \\right) + \left( \\varepsilon_i + \\frac{\sum_{j=1}^n A_{ij} \\varepsilon_j}{\sum_{j=1}^n A_{ij}} \\right)

        where :math:`A_{ij}` is an indicator for whether nodes i and j are linked, :math:`X_i`, and :math:`\\varepsilon_i` are i.i.d. standard normal.

        Parameters
        ----------
        n : int
            Number of observations. Default value: 500.
        network : string
            Type of network to generate. Options: 'ER' (Erdos-Renyi) and 'RGG' (random geometric graph). Default value: 'RGG'.
        avg_deg : int
            Desired average degree of output graph. Default value: 5.
        seed : int
            Seed for randomness. Set to None to not set a seed. Default value: None.

        Returns
        -------
        Y : numpy array
            n-dimensional array of outcomes.
        X : numpy array
            n-dimensional array of covariates.
        A : NetworkX undirected graph
            Undirected, unweighted network on n nodes.

        Examples
        --------
        >>> from networkinference.utils import FakeData
        >>> Y, X, A = FakeData.ols()
        """
        if network=='ER':
            A = FakeData.erdos_renyi(n, avg_deg, seed)
        else:
            A = FakeData.random_geometric(n, avg_deg, seed)
        np.random.seed(seed=seed)
        X = np.random.normal(size=n)
        eps = np.random.normal(size=n)
        A_norm = adjrownorm(A)
        X = X + A_norm.dot(X)
        errors = eps + A_norm.dot(eps)
        Y = 1 + X + errors
        return Y, X, A

    @staticmethod
    def tsls(n=500, network='RGG', avg_deg=5, seed=None):
        """Returns data (Y, X, W, A) to input into TSLS class. Outcomes are generated using the linear_in_means() method of this class. X = (average outcomes of neighbors, average covariate of neighbors, own covariate). W = (average covariate of friends of friends, average covariate of neighbors, own covariate).

        Parameters
        ----------
        n : int
            Number of observations. Default value: 500.
        network : string
            Type of network to generate. Options: 'ER' (Erdos-Renyi) and 'RGG' (random geometric graph). Default value: 'RGG'.
        avg_deg : int
            Desired average degree of output graph. Default value: 5.
        seed : int
            Seed for randomness. Set to None to not set a seed. Default value: None.

        Returns
        -------
        Y : numpy array
            n-dimensional array of outcomes.
        X : numpy array
            n x 3 array of regressors. First column is average outcomes of network neighbors, second is average covariate of neighbors, third is own covariate, where covariates are binary.
        W : numpy array
            n x 3 array of instruments. First column is average covariate of friends of friends, second is average covariate of neighbors, third is own covariate, where covariates are binary.
        A : NetworkX undirected graph
            Undirected, unweighted network on n nodes.

        Examples
        --------
        >>> from networkinference.utils import FakeData
        >>> Y, X, W, A = FakeData.tsls()
        """
        if network=='ER':
            A = FakeData.erdos_renyi(n, avg_deg, seed)
        else:
            A = FakeData.random_geometric(n, avg_deg, seed)
        np.random.seed(seed=seed)
        A_norm = adjrownorm(A)
        Y, D = FakeData.linear_in_means(A)
        Dbar = A_norm.dot(D)
        Ybar = A_norm.dot(Y)
        D2bar = nhbr_mean(D, A, distance=2)
        X = np.vstack([Ybar, Dbar, D]).T
        W = np.vstack([D2bar, Dbar, D]).T
        return Y, X, W, A

