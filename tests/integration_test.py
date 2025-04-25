from velotest.hypothesis_testing import run_hypothesis_test_on
import scvelo
import numpy as np


def test_run_hypothesis_test_on_pancreas():
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

    # Run test
    uncorrected_p_values, h0_rejected, _, _, _ = run_hypothesis_test_on(adata, number_neighborhoods=100, number_neighbors_to_sample_from=20)

    assert np.sum(h0_rejected) >= 1
