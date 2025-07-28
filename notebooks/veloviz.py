import numpy as np
import scipy.sparse as sp
import scipy.stats as stats
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise


def normalize_depth(x, target_size=1_000_000):
    if isinstance(x, np.ndarray):
        cell_counts = np.sum(x, axis=1, keepdims=True)
    else:  # sp.spmatrix and np.matrix
        cell_counts = np.sum(x, axis=1)

    norm_factor = cell_counts / target_size

    return x / norm_factor


def _mean_variance(x: np.ndarray | sp.csr_matrix, axis: int = None, ddof: int = 0):
    """ Equivalent of np.var that supports sparse and dense matrices. """
    if not sp.issparse(x):
        return np.mean(x, axis), np.var(x, axis, ddof=ddof)

    x_mean = x.mean(axis).A1
    xx_mean = x.multiply(x).mean(axis).A1

    x_var = xx_mean - x_mean ** 2

    # Apply correction for degrees of freedom
    n = np.prod(x.shape) if axis is None else x.shape[axis]
    x_var *= n / (n - ddof)

    return x_mean, x_var


def log1p(X, inplace=False):
    if not inplace:
        X = X.copy()

    if sp.issparse(X):
        X.data = np.log1p(X.data)
    else:
        np.log1p(X, out=X)

    return X


def log10p(X, inplace=False):
    if not inplace:
        X = X.copy()

    if sp.issparse(X):
        X.data += 1
        X.data = np.log10(X.data)
    else:
        X += 1
        np.log10(X, out=X)

    return X


class VelovizNormalization:
    """
    https://github.com/kharchenkolab/pagoda2/blob/main/R/Pagoda2.R#L295-L436
    """
    def __init__(
        self,
        alpha: float = 0.05,
        min_adjusted_variance: float = 1e-3,
        max_adjusted_variance: float = 1e+3,
    ):
        self.alpha = alpha
        self.min_adjusted_variance = min_adjusted_variance
        self.max_adjusted_variance = max_adjusted_variance

        self.overdispersed_mask_ = None
        self.scale_factor_ = None
        self.sqrt_scale_factor_ = None

    def fit(self, x: np.ndarray | sp.spmatrix):
        n_samples, n_genes = x.shape

        if sp.issparse(x):
            x = x.tocsc()

        mu, var = _mean_variance(x, axis=0, ddof=1)

        log_mu = np.log(mu)
        log_var = np.log(var)

        # Fit a linear model to the relationship
        # TODO: In the library, this is done with a GAM
        lr = stats.linregress(log_mu, y=log_var)
        log_var_pred = lr.intercept + lr.slope * log_mu

        log_var_residuals = log_var - log_var_pred
        var_residuals = np.exp(log_var_residuals)

        # The F test compares the variances of two samples
        pvals = stats.f.sf(var_residuals, dfn=n_genes, dfd=n_genes)
        pvals_corr = stats.false_discovery_control(pvals, method="bh")

        self.overdispersed_mask_ = pvals_corr < self.alpha

        # Compute scale factors
        qv = stats.chi2.isf(pvals, n_samples - 1) / n_samples
        qv = np.ma.array(qv, mask=np.isinf(qv))
        self.scale_factor_ = np.clip(
            qv,
            a_min=self.min_adjusted_variance,
            a_max=self.max_adjusted_variance,
        ).filled(1)   # don't rescale inf genes
        self.sqrt_scale_factor_ = np.sqrt(self.scale_factor_)

        return self

    def transform(self, x: np.ndarray | sp.spmatrix) -> np.ndarray:
        if sp.issparse(x):
            x = x.tocsc()

        x = x[:, self.overdispersed_mask_]

        x_mu, x_var = _mean_variance(x, axis=0, ddof=1)
        norm = np.sqrt(x_var) / self.sqrt_scale_factor_[self.overdispersed_mask_]
        x_norm = x / norm[None, :]

        return x_norm

    def fit_transform(self, x):
        return self.fit(x).transform(x)


def veloviz(
    curr: np.ndarray,
    proj: np.ndarray,
    use_pca: bool = True,
    pca_components: int = 10,
    omega: float = 1,
    k_neighbors: int = 10,
    distance_quantile_threshold: float = 1,
    similarity_threshold: float = 0,
    directed: bool = True,
    weighted: bool = True,
) -> sp.spmatrix:
    n_samples = curr.shape[0]

    if use_pca:
        #curr = log10p(curr)
        #proj = log10p(proj)
        mean, var = _mean_variance(curr, axis=0, ddof=1)
        std = np.sqrt(var)
        curr = (curr - mean) / std
        proj = (proj - mean) / std

        pca = PCA(n_components=pca_components, random_state=0)
        curr = pca.fit_transform(curr)
        proj = pca.transform(proj)

    # Compute pairwise distances between predicted endpoints and observed cells
    d_ab = pairwise.euclidean_distances(proj, curr)
    d_ab = 1 / (omega * d_ab + 1)

    # Compute cosine similarities between each cell's velocity vector and all
    # difference vectors between all cells
    angle_similarities = np.empty((n_samples, n_samples), dtype=np.float32)
    for i in range(n_samples):
        angle_similarities[i] = -pairwise.cosine_similarity(
            (proj[i] - curr[i])[None, :], curr - curr[i]
        )

    composite_distance = angle_similarities * d_ab
    # Fill with negative inf so each point is its own nearest neighbor
    # Simplifies finding nearest neighbors later on since distances can be
    # negateive
    np.fill_diagonal(composite_distance, -np.inf)

    # Find nearest neighbors, Because we set d_{i,i} to -inf, each point is its
    # own nearest neighbor, so removing it is simple
    sort_idx = np.argsort(composite_distance, axis=1)
    sort_idx = sort_idx[:, 1:k_neighbors + 1]

    # Determine which NNs pass through the angular and distance thresholds
    nn_composite_dists = np.take_along_axis(composite_distance, sort_idx, axis=1)
    nn_similarities = np.take_along_axis(angle_similarities, sort_idx, axis=1)
    nn_distances = np.take_along_axis(d_ab, sort_idx, axis=1)

    angle_mask = nn_similarities <= -similarity_threshold
    # Distances are filtered based on quantiles
    distance_threshold = np.quantile(nn_distances, distance_quantile_threshold)
    dist_mask = nn_distances <= distance_threshold
    knn_mask = angle_mask & dist_mask

    edge_indices = np.ma.array(sort_idx, mask=~knn_mask)
    edge_weights = np.ma.array(nn_composite_dists, mask=~knn_mask)

    if weighted:
        # Edge weights contain lots of negative distances, while graph weights
        # need to be positive
        edge_weights = edge_weights.max() - edge_weights
        smallest_nonzero = np.ma.masked_less_equal(edge_weights, 0).min()
        edge_weights += 0.1 * smallest_nonzero
        # Fill masked values with zeros, these will be removed later with
        # .eliminate_zeros()
        edge_weights = edge_weights.filled(0)
    else:
        edge_weights = knn_mask.astype(np.float32)

    # Construct an adjacency matrix
    adj = sp.csr_matrix(
        (
            edge_weights.ravel(),
            edge_indices.ravel(),
            range(0, n_samples * k_neighbors + 1, k_neighbors),
        ),
        shape=(n_samples, n_samples),
    )

    if not directed:
        adj = adj.maximum(adj.T)

    adj.eliminate_zeros()

    return adj


def fr_layout_igraph(adj, niter=500, grid=False, init=None):
    import igraph

    g = igraph.Graph.Weighted_Adjacency(adj)
    layout = g.layout_fruchterman_reingold(
        weights="weight", niter=niter, grid=grid, seed=init
    )

    return np.array(layout.coords)


def fr_layout_nx(adj, niter=500, init=None):
    import networkx as nx

    g = nx.from_scipy_sparse_array(adj)

    z = nx.spring_layout(g, iterations=niter, pos={i: xi for i, xi in enumerate(init)})
    z = np.array(list(z.values()))

    return z
