import unittest

import anndata
import numpy as np
import torch

from velotest.explicit_hypothesis_testing import run_explicit_test_from
from velotest.hypothesis_testing import run_hypothesis_test_on
from velotest.test_statistic_function import TestStatistic


class TestStatisticTest(unittest.TestCase):
    def test_normalization_factor(self):
        func = TestStatistic(ranges=torch.tensor([[0, 1], [1, 4]]), values=torch.tensor([0, 0.9]))
        exclusion_angle = 0.5
        assert func.normalization_factor(exclusion_angle).item() == 3.5


    def test_p_value(self):
        func = TestStatistic(ranges=torch.tensor([[0, 1], [1, 4]]), values=torch.tensor([0, 0.9]))
        assert func.p_value(-1) == 1.0
        assert func.p_value(0.5) == 0.75
        assert func.p_value(1) == 0


    def test_p_value_equal(self):
        func = TestStatistic(ranges=torch.tensor([[0, 1], [1, 4]]), values=torch.tensor([0, 0.9]))
        assert func.p_value(0.9) == 0.75


    def test_p_value_exclusion(self):
        func = TestStatistic(ranges=torch.tensor([[0, 1], [1, 4]]), values=torch.tensor([0, 1]))
        exclusion_angle = 0.5
        assert func.p_value(0.5, exclusion_angle) == 3 / 3.5


class TestStatisticIntegrationTest(unittest.TestCase):
    __adata: anndata.AnnData

    @classmethod
    def setUpClass(cls):
        import scvelo

        # Load data
        adata = scvelo.datasets.pancreas()
        adata = adata[:50]
        scvelo.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
        scvelo.pp.moments(adata, n_pcs=30, n_neighbors=30)

        # Compute velocity
        scvelo.tl.velocity(adata)

        # Compute 2D embedding of velocity vectors
        scvelo.tl.velocity_graph(adata)
        scvelo.tl.velocity_embedding(adata)
        cls.__adata = adata

    def test_function_start_end(self):
        adata = self.__adata.copy()
        adata = adata[:20]

        _, statistics = run_explicit_test_from(adata, number_neighbors_to_sample_from=15)

        for i, statistic in enumerate(statistics):
            if statistic is not None:
                assert statistic(torch.tensor([0])) == statistic(torch.tensor([2*torch.pi-1e-6])), \
                         f"Test statistic for cell {i} does not match at 0 and 2*pi"

    def test_matching_p_values_old_implementation(self):
        adata = self.__adata.copy()
        adata = adata[:20]
        uncorrected_p_values, _, _ = run_hypothesis_test_on(adata, number_neighborhoods=10000,
                                                            number_neighbors_to_sample_from=15,
                                                            null_distribution="velocities",
                                                            cosine_empty_neighborhood=None,
                                                            exclusion_degree=None)
        p_values_explicit, _ = run_explicit_test_from(adata, number_neighbors_to_sample_from=15)

        assert torch.allclose(torch.tensor(uncorrected_p_values == 2), (p_values_explicit == 2)), \
            "Empty neighborhoods don't match"
        assert np.allclose(torch.tensor(uncorrected_p_values), p_values_explicit, atol=5e-2), \
            "Uncorrected p-values do not match explicit test statistic p-values"

    def test_matching_p_values_old_parallel(self):
        adata = self.__adata.copy()
        adata = adata[:20]
        p_values_parallel, _ = run_explicit_test_from(adata, number_neighbors_to_sample_from=15, parallel=True)
        p_values_serial, _ = run_explicit_test_from(adata, number_neighbors_to_sample_from=15, parallel=False)

        assert np.allclose(p_values_parallel, p_values_serial, atol=1e-6), \
            "Uncorrected p-values do not match explicit test statistic p-values"
