import warnings

import numpy as np


def pairwise_distances_squared(points):
    """
    Compute full pairwise squared Euclidean distance matrix.

    points: (n, 2)
    returns: (n, n) float32
    """
    points = np.asarray(points, dtype=np.float32)

    diff = points[:, None, :] - points[None, :, :]
    dist2 = np.sum(diff * diff, axis=-1)

    return dist2.astype(np.float32)


def top_k_neighbors(dist2, max_number_neighbors=100):
    """
    For each seed, return its kmax nearest neighbors, sorted by distance.
    Assumes diagonal of dist2 is inf.
    """
    n = dist2.shape[0]
    idx = np.argpartition(dist2, kth=max_number_neighbors - 1, axis=1)[:, :max_number_neighbors]
    row = np.arange(n)[:, None]
    order = np.argsort(dist2[row, idx], axis=1)
    return idx[row, order]


def distance_to_next_knn_member(dist2, topk, max_number_neighbors=100, include_seed=True):
    """
    For each seed and each expansion step, compute the distance from the
    newly added cell to the closest member of the current set.

    Parameters
    ----------
    dist2 : (n, n) array
        Pairwise squared distances.
    topk : (n, kmax) array
        Sorted nearest neighbors for each seed, excluding the seed itself.
    max_number_neighbors : int
        Number of nearest neighbors stored in topk.
    include_seed : bool
        If True, the current set starts as {seed}.
        If False, the current set starts as {topk[i, 0]}.

    Returns
    -------
    added_idx : (n, kmax - 1) int32
        The cell added at each step.
    added_d2 : (n, kmax - 1) float32
        Squared distance from the added cell to the closest already-selected cell.
    """
    n = dist2.shape[0]
    n_steps = max_number_neighbors - 1

    added_idx = np.empty((n, n_steps), dtype=np.int32)
    added_d2 = np.empty((n, n_steps), dtype=np.float32)

    for i in range(n):
        nbrs = topk[i, :max_number_neighbors]

        if include_seed:
            selected = np.zeros(n, dtype=bool)
            selected[i] = True
            best = dist2[i].copy()
            start = 0
        else:
            selected = np.zeros(n, dtype=bool)
            selected[nbrs[0]] = True
            best = dist2[nbrs[0]].copy()
            start = 1

        for t in range(n_steps):
            v = nbrs[start + t]

            # distance from the new cell to the current set
            added_idx[i, t] = v
            added_d2[i, t] = best[v]

            # add the new cell to the set and update distances-to-set
            selected[v] = True
            best = np.minimum(best, dist2[v])

    return added_idx, added_d2


def estimate_optimal_number_neighbors_per_cell(adata, basis='umap', max_number_neighbors=500, std_threshold = 4,
                                               min_number_neighbors_selected=20):
    """

    :param adata:
    :param basis:
    :param max_number_neighbors:
    :param std_threshold:
    :param min_number_neighbors_selected: If the heuristic results in a very small number of neighbors,
    we can set a lower bound on the number of neighbors to select.
    This is motivated by the fact that with very few neighbors, the test statistic can become very noisy.
    :return:
    """
    if adata.n_obs > 10000:
        warnings.warn("Estimating optimal number of neighbors for more than 10,000 cells can be very slow. "
                      "A more efficient implementation which doesn't materialize all distances is needed.")

    distances = pairwise_distances_squared(adata.obsm[f'X_{basis}'])
    topk = top_k_neighbors(distances, max_number_neighbors=max_number_neighbors)
    nearest_idx, nearest_d2 = distance_to_next_knn_member(distances, topk, max_number_neighbors=max_number_neighbors)

    mu = nearest_d2.mean(axis=1, keepdims=True)
    sigma = nearest_d2.std(axis=1, keepdims=True)

    stds = (nearest_d2 - mu) / sigma
    bool_array = stds > std_threshold
    number_neighbors_to_sample_from = np.argmax(bool_array, axis=1)
    # Change 0-indexing from np in number neighbors
    number_neighbors_to_sample_from += 1

    # Cover edge case that no neighbor has a distance over the threshold
    no_jump_mask = ~np.any(bool_array, axis=1)
    number_neighbors_to_sample_from[no_jump_mask] = max_number_neighbors

    if min_number_neighbors_selected is not None:
        number_neighbors_to_sample_from = np.maximum(number_neighbors_to_sample_from, min_number_neighbors_selected)

    return number_neighbors_to_sample_from, topk, nearest_d2
