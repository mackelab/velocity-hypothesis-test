import unittest

import anndata
import numpy as np
import torch
from parameterized import parameterized

from velotest.explicit_hypothesis_testing import run_explicit_test_from
from velotest.hypothesis_testing import run_hypothesis_test_on
from velotest.test_statistic_function import TestStatistic


class TestStatisticTest(unittest.TestCase):
    def test_normalization_factor(self):
        func = TestStatistic(ranges=np.array([[0, 1], [1, 4]]), values=np.array([0, 0.9]), offset=0)
        exclusion_angle = 0.5
        assert func.normalization_factor(exclusion_angle).item() == 3.5

    def test_p_value(self):
        func = TestStatistic(ranges=np.array([[0, 1], [1, 4]]), values=np.array([0, 0.9]), offset=0)
        assert func.p_value(-1) == 1.0
        assert func.p_value(0.5) == 0.75
        assert func.p_value(1) == 0

    def test_p_value_equal(self):
        func = TestStatistic(ranges=np.array([[0, 1], [1, 4]]), values=np.array([0, 0.9]), offset=0)
        assert func.p_value(0.9) == 0.75

    def test_p_value_exclusion(self):
        func = TestStatistic(ranges=np.array([[0, 1], [1, 4]]), values=np.array([0, 1]), offset=0)
        exclusion_angle = 0.5
        assert np.allclose(func.p_value(0.5, exclusion_angle), 3 / 3.5)


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
                assert statistic(np.array([0])) == statistic(np.array([2 * np.pi - 1e-6])), \
                    f"Test statistic for cell {i} does not match at 0 and 2*pi"

    @parameterized.expand([[None,], [5,], [10,], [45,]])
    def test_matching_p_values_old_implementation(self, exclusion_deg):
        adata = self.__adata.copy()
        adata = adata[:20]
        uncorrected_p_values, _, _ = run_hypothesis_test_on(adata, number_neighborhoods=10000,
                                                            number_neighbors_to_sample_from=15,
                                                            null_distribution="velocities",
                                                            cosine_empty_neighborhood=None,
                                                            exclusion_degree=exclusion_deg,
                                                            restrict_to_velocity_genes=False)
        p_values_explicit, _ = run_explicit_test_from(adata, number_neighbors_to_sample_from=15,
                                                      exclusion_gamma_deg=exclusion_deg)

        assert np.allclose((uncorrected_p_values == 2), (p_values_explicit == 2)), \
            "Empty neighborhoods don't match"
        # TODO: Why does atol have to grow with exclusion_deg? Faster than linear?
        if exclusion_deg is None or exclusion_deg < 10:
            atol = 2e-2
        elif exclusion_deg < 45:
            atol = 5e-2
        else:
            atol = 5e-1
        assert np.allclose(uncorrected_p_values, p_values_explicit, atol=atol), \
            f"Uncorrected p-values do not match explicit test statistic p-values. Max difference: {np.max(np.abs(uncorrected_p_values - p_values_explicit)):.3f}, "

    def test_matching_p_values_parallel(self):
        adata = self.__adata.copy()
        adata = adata[:20]
        p_values_parallel, _ = run_explicit_test_from(adata, number_neighbors_to_sample_from=15, parallel=True)
        p_values_serial, _ = run_explicit_test_from(adata, number_neighbors_to_sample_from=15, parallel=False)

        assert np.allclose(p_values_parallel, p_values_serial, atol=1e-6), \
            "Uncorrected p-values do not match explicit test statistic p-values"

    def test_matching_max_test_statistic(self):
        adata = self.__adata.copy()
        _, _, debug_dict = run_hypothesis_test_on(adata, number_neighborhoods=500,
                                                  number_neighbors_to_sample_from=40,
                                                  null_distribution="velocities",
                                                  cosine_empty_neighborhood=None)
        test_statistic_all = debug_dict["test_statistic_all"]
        _, _, debug_dict_explicit = run_hypothesis_test_on(adata, number_neighbors_to_sample_from=40,
                                                           null_distribution="velocities-explicit",
                                                           cosine_empty_neighborhood=None)
        statistics = debug_dict_explicit["statistics"]
        statistics = [statistic for statistic in statistics if statistic is not None]  # Remove None statistics

        assert len(statistics)== len(test_statistic_all), \
            f"Number of statistics {len(statistics)} does not match number of cells {len(test_statistic_all)}"
        for i, statistic in enumerate(statistics):
            if statistic is not None:
                assert statistic.get_max_value()[1] + 1e-6 >= torch.max(test_statistic_all[i]).item(), \
                    (f"Max explicit test statistic {statistic.get_max_value()[1]=} for cell {i} "
                     f"is not >= than the max sampled test statistic {torch.max(test_statistic_all[i])}")
