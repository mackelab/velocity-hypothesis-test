from typing import Optional

import torch
from tqdm import tqdm


def cos_directionality_one_cell_batch_same_neighbors(expression: torch.Tensor, velocity_vector: torch.Tensor,
                                                     expressions_neighbours: torch.Tensor):
    """
    Calculates the cosine similarity between the velocity of a cell and multiple sets of other cells
    (e.g., in the neighborhood of the cell). Every set is assumed to have same number of neighbors.

    :param expression: vector of gene expressions of the cell
    :param velocity_vector: velocity vector of the cell, not the position (x+v) of the velocity
    :param expressions_neighbours: (#neighborhoods, #neighbors, #genes)
    :return:
    """
    number_neighborhoods = expressions_neighbours.shape[0]
    number_neighbors_per_neighborhood = expressions_neighbours.shape[1]

    cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    return cos(expressions_neighbours - expression,
               velocity_vector[None, None, :].expand(number_neighborhoods,
                                                     number_neighbors_per_neighborhood, -1))


def cos_directionality_one_cell_one_neighborhood(expression: torch.Tensor, velocity_vector: torch.Tensor,
                                                 expressions_neighbours: torch.Tensor):
    """
    Calculates the cosine similarity between the velocity of a cell and one set of other cells
    (e.g., in the neighborhood of the cell).

    :param expression: vector of gene expressions of the cell
    :param velocity_vector: velocity vector of the cell, not the position (x+v) of the velocity
    :param expressions_neighbours: (#neighbors, #genes)
    :return:
    """
    cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    number_neighbors = expressions_neighbours.shape[0]
    return cos(expressions_neighbours - expression, velocity_vector[None, :].expand(number_neighbors, -1))


def mean_cos_directionality_varying_neighborhoods_same_neighbors(expression: torch.Tensor,
                                                                 velocity_vector: torch.Tensor,
                                                                 neighborhoods: list, original_indices_cells):
    """
    Mean cos directionality for a varying number of neighbors in a neighborhood across cells but with same number
    of neighbors per cell:
    Calculates the mean cosine similarity between the velocity of a cell and
    multiple sets of other cells (e.g., in the neighborhood of the cell).
    Every set is assumed to have same number of neighbors.

    :param expression: expression of all cells
    :param velocity_vector: velocity vectors of all cells, not position (x+v) of the velocity
    :param neighborhoods: Neighborhoods of selected cells. list of length #cells with (#neighborhoods, #neighbors).
    :param original_indices_cells: indices of the selected cells in the original expression matrix
    :return:
    """
    number_cells = len(neighborhoods)
    number_neighborhoods = neighborhoods[0].shape[0]
    mean_cos_neighborhoods = torch.zeros((number_cells, number_neighborhoods))
    for i, original_index in enumerate(tqdm(original_indices_cells)):
        mean_cos_neighborhoods[i] = torch.mean(
            cos_directionality_one_cell_batch_same_neighbors(expression[original_index],
                                                             velocity_vector[original_index],
                                                             expression[neighborhoods[i]]), dim=-1)
    return mean_cos_neighborhoods


def mean_cos_directionality_varying_neighbors(expression: torch.Tensor,
                                              velocity_vector: torch.Tensor,
                                              neighborhoods: list,
                                              original_indices_cells,
                                              cosine_empty_neighborhood: Optional[float] = 2):
    """
    Mean cos directionality for a varying number of neighbors in the neighborhoods across cells:
    Calculates the cosine similarity between the velocity of a cell and multiple sets of other cells
    (e.g., in the neighborhood of the cell). Every set can have a different number of cells.

    :param expression: expression of all cells
    :param velocity_vector: velocity vectors of all cells, not position (x+v) of the velocity
    :param neighborhoods: Neighborhoods of selected cells. list of length #cells with lists of varying #neighbors.
    :param original_indices_cells: indices of the selected cells in the original expression matrix
    :param cosine_empty_neighborhood: if the neighborhood is empty, assign this value to the mean cosine similarity.
        Standard is 2 which is higher then the max of the cosine similarity and will therefore lead to more cells
        where we cannot reject the null hypothesis (Type II error). -2 would lead to Type I errors.
        "None" will ignore empty neighborhoods and then return a variable number of mean cosine similarities per cell.
    :return:
    """
    if torch.cuda.is_available():
        expression = expression.cuda()
        velocity_vector = velocity_vector.cuda()

    number_cells = len(neighborhoods)
    number_neighborhoods = len(neighborhoods[0])
    if cosine_empty_neighborhood is not None:
        mean_cos_neighborhoods = torch.zeros((number_cells, number_neighborhoods))
    else:
        mean_cos_neighborhoods = []
    for cell, (original_index, neighborhoods_one_cell) in enumerate(zip(tqdm(original_indices_cells), neighborhoods)):
        if cosine_empty_neighborhood is None:
            mean_cos_neighborhoods_cell = []
        for neighborhood_id, neighborhood in enumerate(neighborhoods_one_cell):
            if len(neighborhood) == 0:
                if cosine_empty_neighborhood is not None:
                    mean_cos_neighborhoods[cell, neighborhood_id] = cosine_empty_neighborhood
            else:
                mean_cos_directionality_one_cell_one_neighborhood = torch.mean(
                    cos_directionality_one_cell_one_neighborhood(expression[original_index],
                                                                 velocity_vector[original_index],
                                                                 expression[neighborhood]))
                if cosine_empty_neighborhood is not None:
                    mean_cos_neighborhoods[cell, neighborhood_id] = mean_cos_directionality_one_cell_one_neighborhood
                else:
                    mean_cos_neighborhoods_cell.append(mean_cos_directionality_one_cell_one_neighborhood)
        if cosine_empty_neighborhood is None:
            mean_cos_neighborhoods.append(torch.tensor(mean_cos_neighborhoods_cell))
    return mean_cos_neighborhoods
