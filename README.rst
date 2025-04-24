Velotest is a hypothesis test for how well any 2D embedding of positional and velocity data represents
the original high dimensional data. It's current main application is to help practitioners using 2D embeddings
of single cell RNA sequencing data with RNA velocity decide which 2D velocity vectors are faithfully representing
the high-dimensional data.

Installation
------------------
.. code-block:: bash

   pip install -e "."

Later on:

.. code-block:: bash

   pip install velotest

If you want to build the docs and/or run tests, use

.. code-block:: bash

   pip install -e ".[docs]"

or

.. code-block:: bash

   pip install -e ".[dev]",

respectively.

Usage
----------------

If you have a 2D embedding of any data (see below for scRNA-seq data), you can use our general interface:

.. code-block:: python

   from velotest.hypothesis_testing import run_hypothesis_test

   uncorrected_p_values, h0_rejected, _, _, _ = run_hypothesis_test(high_d_position, high_d_velocity, low_d_position, low_d_velocity_position)

where low_d_velocity_position is the tip's position of the 2D velocity vector, NOT the velocity vector originating in low_d_position.


An application on single-cell sequencing data (runnable notebook: `notebooks/demo.ipynb`) could look like this (following `scvelo's tutorial <https://scvelo.readthedocs.io/en/stable/VelocityBasics.html>`_):

.. code-block:: python

   from velotest.hypothesis_testing import run_hypothesis_test_on
   import scvelo

   adata = scvelo.datasets.pancreas()
   scvelo.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
   scvelo.pp.moments(adata, n_pcs=30, n_neighbors=30)

   # Compute velocity
   scvelo.tl.velocity(adata)

   # Compute 2D embedding of velocity vectors
   scvelo.tl.velocity_graph(adata)
   scvelo.pl.velocity_embedding(adata)

   # Run test
   uncorrected_p_values, h0_rejected, _, _, _ = run_hypothesis_test_on(adata)


For plotting, you can use the `plotting` module. Have a look at `notebooks/demo.ipynb` for an example.


Details
--------------------
For a data point :math:`i`, it uses the mean cosine similarity between the velocity :math:`v_i` and
the difference vector :math:`x_j-x_i` for all :math:`x_j` in a set of neighbors of :math:`x_i` as the test statistic.
This set of neighbors is either chosen based on the points the velocity :math:`\tilde{v}_i` points to in 2D or
is sampled randomly from the 2D neighbors of :math:`\tilde{x}_i`.
:math:`\tilde{v}_i` and :math:`\tilde{x}_i` are the 2D embeddings of :math:`v_i` and :math:`x_i`, respectively.

The null hypothesis is that the observed statistic for the neighbors chosen based on the velocity comes from
the same distribution as random neighbors.
It is rejected if the number of random neighborhoods with a higher statistic as the statistic
from the velocity-based neighborhood exceeds the level we would expect for a certain significance level.

It was originally developed for the analysis of single cell RNA sequencing data,
but can be applied to any application with positional and velocity data.