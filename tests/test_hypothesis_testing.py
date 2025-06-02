import torch
import numpy as np

from velotest.hypothesis_testing import p_values, p_values_list, benjamini_hochberg


def test_p_values_same_results():
    test_statistics_velocity = [0.1, 0.2, 0.3]
    test_statistics_velocity_tensor = torch.tensor(test_statistics_velocity)
    test_statistics_random = [[0., 0.1, 0.01], [0.1, 0.1, 0.2], [0.3, 0.3, 0.4]]
    test_statistics_random_tensor = torch.tensor(test_statistics_random)

    test_statistics_velocity_list = [torch.tensor(cell) for cell in test_statistics_velocity]
    test_statistics_random_list = [[torch.tensor(cell) for cell in neighborhood]
                                   for neighborhood in test_statistics_random]

    assert torch.allclose(p_values(test_statistics_velocity_tensor, test_statistics_random_tensor),
                          p_values_list(test_statistics_velocity_list, test_statistics_random_list),
                          atol=1e-6)


def test_benjamini_hochberg():
    p_values = np.array([0.5, 0.012, 0.5, 0.5, 0.008, 0.5, 0.5, 0.013, 0.5, 0.5])
    rejected_hypothesis = benjamini_hochberg(p_values)
    assert np.all(rejected_hypothesis == np.array([False, True, False, False, True, False, False, True, False, False]))
