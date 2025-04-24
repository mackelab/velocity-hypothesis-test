import math

import torch
from sklearn import neighbors


def cos_directionality(expression, velocity_position, expressions_neighbours):
    """
    Calculates the cosine similarity between the velocity of a cell and multiple cells in the neighborhood of the cell.
    Currently used for 2D velocity vectors.
    :param expression: position of the cell
    :param velocity_position: position (x+v) of the tip of the velocity vector
    :param expressions_neighbours: expressions of the neighbors
    :return:
    """
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    return cos(expressions_neighbours - expression, velocity_position - expression)


# Batched version of 'cos_directionality' which can be used on a batch of cells, their velocity vectors and their neighbors
cos_directionality_batched = torch.vmap(cos_directionality)


def find_neighbors(x, k_neighbors=5, metric="euclidean"):
    """
    Finds the k nearest neighbors of each cell in the expression matrix.
    :param x: expression matrix
    :param k_neighbors: number of neighbors
    :param metric: metric to use to calculate distances
    :return: an array of indices of the k nearest neighbors for each cell (#cells, k_neighbors)
    """
    nn = neighbors.NearestNeighbors(metric=metric, n_jobs=-1)
    nn.fit(x)
    indices = nn.kneighbors(n_neighbors=k_neighbors, return_distance=False)

    return indices

def find_neighbors_in_direction_of_velocity(Z_expr, Z_velo_position, nn_indices)-> list:
    """
    Finds the neighbors of each cell in 2D that are in a 2*threshold_degree cone around the velocity vector.
    :param Z_expr: 2D embedding of positions
    :param Z_velo_position: 2D embedding of tip of velocity vectors
    :param nn_indices: indices of the k nearest neighbors for each cell
    :return: return list of neighbours in direction of velocity for each cell
    """
    neighbour_directionality = cos_directionality_batched(Z_expr, Z_velo_position, Z_expr[nn_indices])
    # this allows a derivation of 22.5° (a cone of 45° around velocity vector), 0.707 for 45° (cone of 90°)
    selected_neighbours = neighbour_directionality > 0.9238795325
    neighbours_in_direction_of_velocity = [nn_indices[i][neighbors] for i, neighbors in enumerate(selected_neighbours)]
    return neighbours_in_direction_of_velocity