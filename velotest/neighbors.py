from sklearn import neighbors
import torch

def cos_directionality(expression, velocity_position, expressions_neighbours):
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    return cos(expressions_neighbours - expression, velocity_position - expression)
cos_directionality_batched = torch.vmap(cos_directionality)

def find_neighbors(x, k_neighbors=5, metric="euclidean"):
    nn = neighbors.NearestNeighbors(metric=metric, n_jobs=-1)
    nn.fit(x)
    indices = nn.kneighbors(n_neighbors=k_neighbors, return_distance=False)

    return indices

def find_neighbors_in_direction_of_velocity(Z_expr, Z_velo_position, nn_indices)-> list:
    neighbour_directionality = cos_directionality_batched(Z_expr, Z_velo_position, Z_expr[nn_indices])
    # this allows a derivation of 22.5° (a cone of 45° around velocity vector), 0.707 for 45° (cone of 90°)
    selected_neighbours = neighbour_directionality > 0.9238795325
    neighbours_in_direction_of_velocity = [nn_indices[i][neighbors] for i, neighbors in enumerate(selected_neighbours)]
    return neighbours_in_direction_of_velocity