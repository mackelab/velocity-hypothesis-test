import torch
from tqdm import tqdm


def cos_directionality_one_cell(expression: torch.Tensor, velocity_vector: torch.Tensor,
                                expressions_neighbours: torch.Tensor):
    """
    Calculates the cosine similarity between the velocity of a cell and multiple sets of cells in the neighborhood of the cell.

    @param expression:
    @param velocity_vector: velocity vector of the cell, not the position (x+v) of the velocity
    @param expressions_neighbours: (#neighborhoods, #neighbors, #genes)
    @return:
    """
    number_neighborhoods = expressions_neighbours.shape[0]
    number_neighbors_per_neighborhood = expressions_neighbours.shape[1]

    cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    return cos(expressions_neighbours - expression,
               velocity_vector.unsqueeze(0).unsqueeze(0).expand(number_neighborhoods,
                                                                number_neighbors_per_neighborhood, -1))


cos_directionality_one_cell_batched = torch.vmap(cos_directionality_one_cell)


def mean_cos_directionality_varying_neighborhoods(expression: torch.Tensor, velocity_vector: torch.Tensor,
                                                  neighborhoods: list, original_indices_cells):
    """
        Calculates the mean cosine similarity between the velocity of a cell and multiple sets of cells in the neighborhood of the cell.

        @param expression: expression of all cells
        @param velocity_vector: velocity vectors of all cells, not position (x+v) of the velocity
        @param neighborhoods: list of length #cells with (#neighborhoods, #neighbors). Neighborhoods of selected cells.
        @param original_indices_cells: indices of the selected cells in the original expression matrix
        @return:
        """
    number_cells = len(neighborhoods)
    number_neighborhoods = neighborhoods[0].shape[0]
    mean_cos_neighborhoods = torch.zeros((number_cells, number_neighborhoods))
    for i, original_index in enumerate(tqdm(original_indices_cells)):
        mean_cos_neighborhoods[i] = torch.mean(
            cos_directionality_one_cell(expression[original_index], velocity_vector[original_index],
                                        expression[neighborhoods[i]]), dim=-1)
    return mean_cos_neighborhoods
