import torch
import numpy as np

from velotest.hypothesis_testing import pvals_from_permutation_test


def test_p_values_same_results():
    test_statistics_velocity = [0.1, 0.2, 0.3]
    test_statistics_velocity_tensor = np.array(test_statistics_velocity)
    test_statistics_random = [[0., 0.1, 0.01], [0.1, 0.1, 0.2], [0.3, 0.3, 0.4]]
    test_statistics_random_tensor = np.array(test_statistics_random)

    test_statistics_velocity_list = [np.array(cell) for cell in test_statistics_velocity]
    test_statistics_random_list = [[np.array(cell) for cell in neighborhood]
                                   for neighborhood in test_statistics_random]

    assert np.allclose(pvals_from_permutation_test(test_statistics_velocity_tensor, test_statistics_random_tensor),
                       pvals_from_permutation_test(test_statistics_velocity_list, test_statistics_random_list),
                       atol=1e-6)
