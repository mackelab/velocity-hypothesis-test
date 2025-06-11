from typing import Optional

import numpy as np
import pandas as pd
import torch
from scipy.stats import false_discovery_control

from velotest.neighbors import find_neighbors, find_neighbors_in_direction_of_velocity, \
    find_neighbors_in_direction_of_velocity_multiple
from velotest.test_statistic import mean_cos_directionality_varying_neighborhoods_same_neighbors, \
    mean_cos_directionality_varying_neighbors


#: ArrayLike[NDArray]
def pvals_from_permutation_test(true_statistic, random_statistics):
    """
    Compute p-values using the test statistics from the permutations.

    :param true_statistic: (#cells)
    :param random_statistics: (#cells, #neighborhoods)
    :return:
    """
    if isinstance(true_statistic, list) or isinstance(random_statistics, list):
        return [
            np.mean(random >= true)
            for true, random in zip(true_statistic, random_statistics)
        ]

    elif isinstance(true_statistic, pd.Series) and isinstance(random_statistics, pd.DataFrame):
        pvals = np.mean(random_statistics.values >= true_statistic.values[:, None], axis=1)
        return pd.Series(pvals, index=true_statistic.index)

    else:
        return np.mean(random_statistics >= true_statistic[:, None], axis=1)


# def p_values_list(test_statistics_velocity, test_statistics_random):
#     # test_statistics_velocity, test_statistics_random = test_statistics_velocity.numpy(), test_statistics_random.numpy()
#     return [
#         np.mean(set_of_random_statistics >= observed_statistic)
#         for observed_statistic, set_of_random_statistics in zip(test_statistics_velocity, test_statistics_random)
#     ]


def correct_for_multiple_tests(pvals, correction="bonferroni"):
    if correction is None:
        return pvals

    elif correction == 'benjamini–hochberg':
        pvals_corrected = false_discovery_control(pvals)

    elif correction == 'bonferroni':
        n_tests = len(pvals)
        pvals_corrected = pvals * n_tests
        pvals_corrected = np.clip(pvals_corrected, a_min=None, a_max=1)

    else:
        raise ValueError(
            f"Unknown correction method: '{correction}'. Supported methods "
            f"include 'benjamini–hochberg', 'bonferroni', and None."
        )

    if isinstance(pvals, pd.Series):
        pvals_corrected = pd.Series(pvals_corrected, index=pvals.index)

    return pvals_corrected


#:ArrayLike[int]
def run_hypothesis_test(
    X_expr,
    X_velo_vector,
    Z_expr,
    Z_velo_position,
    number_neighborhoods=1000,
    number_neighbors_to_sample_from=50,
    threshold_degree=22.5,
    exclusion_degree: Optional[float] = None,
    null_distribution='neighbors',
    correction='benjamini–hochberg',
    alpha=0.05,
    cosine_empty_neighborhood=2,
    seed=0,
):
    """
    Samples random neighborhoods for every cell and uses the high-dimensional cosine similarity between
    the velocity of each cell and the cells in the direction of the velocity (in 2D) as test statistic.

    :param X_expr: high-dimensional expressions
    :param X_velo_vector: high-dimensional velocity vector, not position (x+v)
    :param Z_expr: embedding for expressions
    :param Z_velo_position: embedding for velocity position (x+v)
    :param number_neighborhoods: number of neighborhoods used to define null distribution
    :param number_neighbors_to_sample_from: number of neighbors to sample neighborhoods from and
        to look for neighbors in direction of velocity
    :param threshold_degree: angle in degrees to define the cone around the velocity vector
        (angle of cone is 2*threshold_degree),
    :param exclusion_degree: angle in degrees to exclude random velocities which are too similar to
        the visualized velocity. 'None' uses all random velocities.
    :param null_distribution: 'neighbors' or 'velocities'. If 'neighbors', the neighborhoods are uniformly sampled from the neighbors.
        If 'velocities', random velocities are sampled and then the neighborhoods are defined by the neighbors in this direction.
    :param correction: correction method for multiple testing. 'benjamini–hochberg', 'bonferroni' or None
    :param alpha: significance level used for Benjamini-Hochberg or Bonferroni correction.
    :param cosine_empty_neighborhood: if the neighborhood is empty, assign this value to the mean cosine similarity.
        Standard is 2 which is higher than the max of the cosine similarity and will therefore lead to more cells
        where we cannot reject the null hypothesis (Type II error). -2 would lead to Type I errors.
        "None" will ignore empty neighborhoods and then return a variable number of mean cosine similarities per cell.
    :param seed: Random seed for reproducibility.
    :return:
        - ``p_values_`` (p-values from test (not corrected), cells where test couldn't be run are assigned a value of 2),
        - ``h0_rejected``
        - ``test_statistics_velocity``
        - ``test_statistics_random``
        - ``neighborhoods``
    """
    assert not (null_distribution == 'neighbors' and cosine_empty_neighborhood is None)

    np.random.seed(seed)

    if not isinstance(X_expr, torch.Tensor):
        X_expr = torch.tensor(X_expr)
    if not isinstance(X_velo_vector, torch.Tensor):
        X_velo_vector = torch.tensor(X_velo_vector)
    if not isinstance(Z_expr, torch.Tensor):
        Z_expr = torch.tensor(Z_expr)
    if not isinstance(Z_velo_position, torch.Tensor):
        Z_velo_position = torch.tensor(Z_velo_position)

    number_cells = X_expr.shape[0]

    all_neighbors = find_neighbors(Z_expr, k_neighbors=number_neighbors_to_sample_from)
    # Find only neighbors that appear in the direction of the velocity vector
    velocity_neighborhoods = find_neighbors_in_direction_of_velocity(
        Z_expr, Z_velo_position, all_neighbors, threshold_degree
    )
    velocity_neighborhoods = dict(enumerate(velocity_neighborhoods))

    # Some data points do not have any neighbors in the direction of the
    # velocity vector; we can't do much with these, so remove them
    velocity_neighborhoods = {
        cell_idx: neighbors for cell_idx, neighbors in velocity_neighborhoods.items()
        if len(neighbors) > 0
    }

    debug_dict = {}

    if null_distribution == 'neighbors':
        # Generate random neighborhoods for each cell against which to compare
        # the neighborhood appearing in the direction of the velocity vector
        random_neighborhoods = {
            cell_idx: np.random.choice(all_neighbors[cell_idx], size=(number_neighborhoods, len(neighbors)))
            for cell_idx, neighbors in velocity_neighborhoods.items()
        }

        # Prepend the true neighborhood to the random neighborhoods, so they
        # all appear in the same list. The first entry is the true neighborhood
        concat_neighborhoods_ = {
            cell_idx: np.concatenate(
                [velocity_neighborhoods[cell_idx][None, ...], random_neighborhoods[cell_idx]],
                axis=0,
            )
            for cell_idx in random_neighborhoods
        }

        test_statistics = mean_cos_directionality_varying_neighborhoods_same_neighbors(
            X_expr, X_velo_vector, concat_neighborhoods_
        )
        test_statistics = pd.DataFrame.from_dict(test_statistics, orient="index")

    elif null_distribution == 'velocities':
        # Sample number_neighborhoods random velocities on unit circle for each cell and add them to Z_expr
        random_angles = 2 * np.pi * np.random.uniform(0, 1, size=(number_neighborhoods, number_cells))
        random_angle_vectors = np.stack([np.cos(random_angles), np.sin(random_angles)], axis=-1)
        Z_velo_position_random = Z_expr + random_angle_vectors

        debug_dict['Z_velo_position_random'] = Z_velo_position_random

        neighborhoods_random_velocities = find_neighbors_in_direction_of_velocity_multiple(
            Z_expr, torch.tensor(Z_velo_position_random), all_neighbors, threshold_degree
        )
        neighborhoods_random_velocities = dict(enumerate(neighborhoods_random_velocities))

        if exclusion_degree is not None:
            Z_velo_normalized = (Z_velo_position - Z_expr) / (Z_velo_position - Z_expr).norm(dim=1, keepdim=True)
            Z_velo_normalized = Z_velo_normalized.numpy()

            # Compute velocity vector angles
            theta = np.atan2(Z_velo_normalized[:, 1], Z_velo_normalized[:, 0])
            theta[theta < 0] += 2 * torch.pi

            inclusion_mask = np.logical_or(
                random_angles < theta - np.deg2rad(exclusion_degree),
                random_angles > theta + np.deg2rad(exclusion_degree),
            )
            inclusion_mask = inclusion_mask.T
            debug_dict['mask_not_excluded'] = inclusion_mask

            # Select neighborhoods based on inclusion mask
            neighborhoods_random_velocities = {
                cell_idx: [neighborhoods_random_velocities[cell_idx][i] for i in np.where(mask_one_cell)[0]]
                for mask_one_cell, cell_idx in zip(inclusion_mask, neighborhoods_random_velocities)
            }

        # Keep only random neighborhoods for the cells that have a non-empty
        # real neighborhood
        neighborhoods_random_velocities = {
            cell_idx: neighborhoods_random_velocities[cell_idx]
            for cell_idx in velocity_neighborhoods
        }

        # Prepend the true neighborhood to the random neighborhoods, so they
        # all appear in the same list. The first entry is the true neighborhood
        # (list, list, Tensor)
        concat_neighborhoods_ = {}
        for cell_idx in velocity_neighborhoods:
            merged_neighborhoods_cell = [torch.tensor(velocity_neighborhoods[cell_idx])]
            merged_neighborhoods_cell.extend(neighborhoods_random_velocities[cell_idx])
            concat_neighborhoods_[cell_idx] = merged_neighborhoods_cell

        test_statistics = mean_cos_directionality_varying_neighbors(
            X_expr, X_velo_vector, concat_neighborhoods_, cosine_empty_neighborhood
        )
        test_statistics = pd.DataFrame.from_dict(test_statistics, orient="index")

    else:
        raise ValueError(f"Unknown null distribution: {null_distribution}. Use 'neighbors' or 'velocities'.")

    debug_dict['neighborhoods'] = concat_neighborhoods_

    test_statistics_velocity = test_statistics.iloc[:, 0]
    test_statistics_random = test_statistics.iloc[:, 1:]

    pvals = pvals_from_permutation_test(test_statistics_velocity, test_statistics_random)
    pvals_corrected = correct_for_multiple_tests(pvals, correction)

    # Ensure that we have results for all cells, not just the ones that we were
    # able to compute statistics for
    all_cells_index = np.arange(Z_expr.shape[0])
    pvals = pvals.reindex(all_cells_index, fill_value=pd.NA)
    pvals_corrected = pvals_corrected.reindex(all_cells_index, fill_value=pd.NA)

    h0_rejected = pvals_corrected < alpha

    # Because we are unable to test some certain number of cells, these cells
    # are assigned a p-value of NA. However, we can also fill this with a
    # user-specified value
    if cosine_empty_neighborhood is not None:
        pvals.fillna(cosine_empty_neighborhood, inplace=True)

    debug_dict['test_statistics_velocity'] = test_statistics_velocity
    debug_dict['test_statistics_random'] = test_statistics_random

    # TODO: Probably want to switch number_neighborhoods and number_cells dimensions of Z_velo_position_random
    return pvals, h0_rejected, debug_dict


def run_hypothesis_test_on(adata, ekey='Ms', vkey='velocity', basis='umap', **kwargs):
    """
    Runs the hypothesis test using high dimensional expressions, high dimensional velocity,
    and the embeddings from an adata object. For details, see `run_hypothesis_test`.

    :param adata: Anndata object containing high dimensional data and embeddings.
    :param ekey: Name of layer in adata object containing high dimensional expression data.
    :param vkey: Name of layer in adata object containing high dimensional velocity data.
    :param basis: Name of embedding.
    :param kwargs: Additional arguments for `run_hypothesis_test`.
    :return: See `run_hypothesis_test`.
    """
    X_expr = adata.layers[ekey]
    X_velo_vector = adata.layers[vkey]
    Z_expr = adata.obsm[f"X_{basis}"]
    Z_velo_position = Z_expr + adata.obsm[f'velocity_{basis}']

    if Z_expr.shape[1] > 2:
        Z_expr = Z_expr[:, :2]
        Z_velo_position = Z_velo_position[:, :2]
        print("Warning: Your basis has more than two dimensions. "
              "Using only the first two dimensions of the embedding for hypothesis testing like scvelo in "
              "its visualisations.")

    X_expr = torch.tensor(X_expr)
    X_velo_vector = torch.tensor(X_velo_vector)
    Z_expr = torch.tensor(Z_expr)
    Z_velo_position = torch.tensor(Z_velo_position)

    return run_hypothesis_test(X_expr, X_velo_vector, Z_expr, Z_velo_position, **kwargs)
