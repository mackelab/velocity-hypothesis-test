from typing import Dict, List, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects, pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scvelo.plotting.utils import default_arrow
from sklearn.utils import check_random_state
import matplotlib as mpl
from scvelo.tools.velocity_embedding import quiver_autoscale
from sklearn.neighbors import NearestNeighbors
from scipy.stats import norm as normal


def scatter(X_emb: np.ndarray, label_colormap: Union[Dict, List] = None, labels: pd.Series = None,
            ax: matplotlib.axes.Axes = None, title=None, marker="o", size=25, show_labels: bool = True,
            alpha=0.5, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8), dpi=150)

    if title is not None:
        ax.set_title(title)

    if labels is not None and show_labels:
        plot_labels(ax, X_emb, labels)

    glyph_colors = get_glyph_colors(X_emb, labels, label_colormap)
    random_state = check_random_state(0)
    draw_order = random_state.permutation(X_emb.shape[0])
    ax.scatter(*X_emb[draw_order].T, c=glyph_colors[draw_order], s=size, lw=0, alpha=alpha, marker=marker, **kwargs)

    return ax


def arrow_plot(
        X_emb: np.ndarray,
        V_emb: np.ndarray,
        p_values: np.ndarray = None,
        h0_rejected: np.ndarray = None,
        labels: pd.Series = None,
        plot_legend: bool = True,
        label_colormap: Union[Dict, List] = None,
        ax: matplotlib.axes.Axes = None,
        title=None,
        fontsize: int = 12,
        fontweight: str = "bold",
        multiplier = 1,
        box=True,
        vector_friendly: bool = False,
):
    """Plot the arrows defined by X and V."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8), dpi=150)

    hl, hw, hal = default_arrow(3)
    quiver_kwargs = {
        "angles": "xy",
        "scale_units": "xy",
        "edgecolors": "k",
        "scale": 1,
        "width": 0.001,
        "headlength": hl / 2,
        "headwidth": hw / 2,
        "headaxislength": hal / 2,
        "linewidth": 0.2,
        "zorder": 3,
    }

    if h0_rejected is not None or p_values is not None:
        if (h0_rejected is not None) and (p_values is not None):
            significance = h0_rejected.copy().astype(int)
            significance[p_values == 2] = 2
        else:
            raise ValueError("Both `h0_rejected` and `p_values` must be provided.")
    else:
        significance = None

    if significance is None:
        ax.quiver(
            X_emb[:, 0], X_emb[:, 1], V_emb[:, 0] - X_emb[:, 0], V_emb[:, 1] - X_emb[:, 1], **quiver_kwargs
        )
    else:
        significant = significance == 1
        not_significant = significance == 0
        not_tested = significance == 2
        irrelevant_velocities = np.logical_or(not_tested, not_significant)

        irrelevant_quiver = ax.quiver(
            X_emb[irrelevant_velocities][:, 0], X_emb[irrelevant_velocities][:, 1],
            V_emb[irrelevant_velocities][:, 0] - X_emb[irrelevant_velocities][:, 0],
            V_emb[irrelevant_velocities][:, 1] - X_emb[irrelevant_velocities][:, 1],
            facecolor='darkgrey', edgecolor='face', alpha=0.5, **quiver_kwargs
        )
        irrelevant_quiver.set_rasterized(vector_friendly)
        significant_quiver = ax.quiver(
            X_emb[significant][:, 0], X_emb[significant][:, 1], V_emb[significant][:, 0] - X_emb[significant][:, 0],
                                                                V_emb[significant][:, 1] - X_emb[significant][:, 1],
            color='black', **quiver_kwargs
        )
        significant_quiver.set_rasterized(vector_friendly)

    if significance is None:
        scatter(X_emb, label_colormap, labels, ax)
    else:
        # 'multiplier' allows to scale the markers differently for different datasets
        # have to find an automatic way of doing it
        if labels is not None:
            scatter(X_emb[not_tested], label_colormap, labels[not_tested], ax, size=int(multiplier * 10),
                    show_labels=False, alpha=0.1)
            scatter(X_emb[not_significant], label_colormap, labels[not_significant], ax,
                    size=int(multiplier * 10),
                    show_labels=False, alpha=0.1)
            scatter(X_emb[significant], label_colormap, labels[significant], ax, size=int(multiplier * 20),
                    show_labels=False, alpha=1)
        else:
            scatter(X_emb[not_tested], label_colormap, ax=ax, marker="o", size=int(multiplier * 20))
            scatter(X_emb[not_significant], label_colormap, ax=ax, marker="s", size=int(multiplier * 10))
            scatter(X_emb[significant], label_colormap, ax=ax, marker="*", size=int(multiplier * 60))

    if labels is not None and plot_legend:
        plot_labels(ax, X_emb, labels, fontsize=fontsize, fontweight=fontweight)

    ax.set(xticks=[], yticks=[], box_aspect=1)
    if title is not None:
        ax.set_title(title)

    if not box:
        for spine in ax.spines.values():
            spine.set_visible(False)

    return ax


def marker_plot(X_emb: np.ndarray,
                p_values: np.ndarray,
                h0_rejected: np.ndarray,
                ax: matplotlib.axes.Axes = None,
                multiplier_marker_size=1.5):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8), dpi=150)

    significance = h0_rejected
    no_test = p_values == 2

    scatter(X_emb[np.logical_and(~significance, no_test)], marker="o", size=int(multiplier_marker_size * 20), ax=ax,
            label_colormap="grey", label="unable to test")
    scatter(X_emb[np.logical_and(~significance, ~no_test)], marker="s", size=int(multiplier_marker_size * 10), ax=ax,
            label_colormap="lightblue", label="h0 not rejected")
    scatter(X_emb[significance], marker="*", size=int(multiplier_marker_size * 60), ax=ax, label_colormap="orange",
            label="significant")
    ax.set(xticks=[], yticks=[], box_aspect=1)
    ax.legend()

    return ax


def get_glyph_colors(x, labels, label_colormap):
    if labels is not None:
        if isinstance(label_colormap, dict):
            glyph_colors = np.array([label_colormap[v] for v in labels])
        else:
            if label_colormap is None:
                label_colormap = "viridis_r"
            else:
                if not isinstance(label_colormap, str):
                    raise ValueError("`label_colormap` must be either dict or valid cmap string")

            cmap = matplotlib.colormaps.get_cmap(label_colormap)
            glyph_colors = cmap(labels.cat.codes / labels.cat.codes.max())
    else:
        if label_colormap is None:
            glyph_colors = np.full(shape=(x.shape[0]), fill_value="r")
        else:
            glyph_colors = np.full(shape=(x.shape[0]), fill_value=label_colormap)

    return glyph_colors


def plot_labels(
        ax: matplotlib.axes.Axes,
        embedding: np.ndarray,
        point_labels: pd.Series,
        fontoutline: int = 1,
        fontweight: str = "bold",
        fontcolour: str = "black",
        fontsize: int = 12,
) -> list[matplotlib.text.Text]:
    """Plot cluster labels on top of the plot in the same style as scanpy/scvelo."""
    valid_cats = np.where(point_labels.value_counts()[point_labels.cat.categories] > 0)[0]
    categories = np.array(point_labels.cat.categories)[valid_cats]

    texts = []
    for label in categories:
        x_pos, y_pos = np.nanmedian(embedding[point_labels == label, :], axis=0)
        if isinstance(label, str):
            label = label.replace("_", " ")
        kwargs = {"verticalalignment": "center", "horizontalalignment": "center"}
        kwargs.update({"weight": fontweight, "fontsize": fontsize, "color": fontcolour})
        pe = [patheffects.withStroke(linewidth=fontoutline, foreground="w")]
        text = ax.text(x_pos, y_pos, label, path_effects=pe, **kwargs)
        texts.append(text)

    return texts


def plot_optimal_velocity(adata, Z_expr, uncorrected_p_values, labels, label_colormap, test_statistic_all, Z_velo_position_random,
                          used_neighborhoods, name):

    mpl.rc_file('../matplotlibrc-embeddings')
    fig, ax = plt.subplots(dpi=450, constrained_layout=True)
    cm = 1 / 2.54  # centimeters in inches
    fig.set_size_inches(3.17 * cm, 3.17 * cm)

    non_empty_neighborhoods_indices = np.where(uncorrected_p_values != 2)[0]

    best_velocities = compute_optimal_velocity(Z_velo_position_random, adata, non_empty_neighborhoods_indices,
                                               test_statistic_all, used_neighborhoods)
    glyph_colors = get_glyph_colors(Z_expr, labels, label_colormap)
    plt.quiver(*Z_expr[non_empty_neighborhoods_indices].T,
               *((best_velocities - Z_expr[non_empty_neighborhoods_indices]) * 0.4).T, angles='xy', scale_units='xy',
               scale=1, color=glyph_colors[non_empty_neighborhoods_indices], alpha=0.5)
    #ax.set_title("Optimal velocity")
    ax.set(xticks=[], yticks=[], box_aspect=1)
    fig.savefig(f"fig/{name}.pdf", dpi=450)
    plt.show()


def compute_optimal_velocity(Z_velo_position_random, adata, non_empty_neighborhoods_indices, test_statistic_all,
                             used_neighborhoods):
    import torch

    non_empty_random_neighborhoods = used_neighborhoods[:, 1:]
    non_empty_random_neighborhoods_indices = [np.where(neighborhoods_one_cell)[0] for neighborhoods_one_cell in
                                              non_empty_random_neighborhoods]
    best_velocities = np.zeros((len(non_empty_neighborhoods_indices), 2))
    for cell in range(len(adata)):
        converted_cell = torch.where(torch.tensor(non_empty_neighborhoods_indices) == cell)[0]
        # converted_cell = np.where(np.array(non_empty_neighborhoods_indices) == cell)[0]
        if converted_cell.shape[0] > 0:
            good_velocities = np.argsort(test_statistic_all[converted_cell][1:])[-1]
            best_velocities[converted_cell] = \
                Z_velo_position_random[cell][non_empty_random_neighborhoods_indices[converted_cell]][
                    good_velocities]
    return best_velocities


# Code from https://github.com/theislab/scvelo/blob/main/scvelo/plotting/velocity_embedding_grid.py#L28
# Licenses under BSD 3-Clause License
def compute_velocity_on_grid(
        X_emb,
        V_emb,
        density=None,
        smooth=None,
        n_neighbors=None,
        min_mass=None,
        autoscale=True,
        adjust_for_stream=False,
        cutoff_perc=None,
):
    """TODO."""
    # remove invalid cells
    idx_valid = np.isfinite(X_emb.sum(1) + V_emb.sum(1))
    X_emb = X_emb[idx_valid]
    V_emb = V_emb[idx_valid]

    # prepare grid
    n_obs, n_dim = X_emb.shape
    density = 1 if density is None else density
    smooth = 0.5 if smooth is None else smooth

    grs = []
    for dim_i in range(n_dim):
        m, M = np.min(X_emb[:, dim_i]), np.max(X_emb[:, dim_i])
        m = m - 0.01 * np.abs(M - m)
        M = M + 0.01 * np.abs(M - m)
        gr = np.linspace(m, M, int(50 * density))
        grs.append(gr)

    meshes_tuple = np.meshgrid(*grs)
    X_grid = np.vstack([i.flat for i in meshes_tuple]).T

    # estimate grid velocities
    if n_neighbors is None:
        n_neighbors = int(n_obs / 50)
    nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
    nn.fit(X_emb)
    dists, neighs = nn.kneighbors(X_grid)

    scale = np.mean([(g[1] - g[0]) for g in grs]) * smooth
    weight = normal.pdf(x=dists, scale=scale)
    p_mass = weight.sum(1)

    V_grid = (V_emb[neighs] * weight[:, :, None]).sum(1)
    V_grid /= np.maximum(1, p_mass)[:, None]
    if min_mass is None:
        min_mass = 1

    if adjust_for_stream:
        X_grid = np.stack([np.unique(X_grid[:, 0]), np.unique(X_grid[:, 1])])
        ns = int(np.sqrt(len(V_grid[:, 0])))
        V_grid = V_grid.T.reshape(2, ns, ns)

        mass = np.sqrt((V_grid ** 2).sum(0))
        min_mass = 10 ** (min_mass - 6)  # default min_mass = 1e-5
        min_mass = np.clip(min_mass, None, np.max(mass) * 0.9)
        cutoff = mass.reshape(V_grid[0].shape) < min_mass

        if cutoff_perc is None:
            cutoff_perc = 5
        length = np.sum(np.mean(np.abs(V_emb[neighs]), axis=1), axis=1).T
        length = length.reshape(ns, ns)
        cutoff |= length < np.percentile(length, cutoff_perc)

        V_grid[0][cutoff] = np.nan
    else:
        min_mass *= np.percentile(p_mass, 99) / 100
        X_grid, V_grid = X_grid[p_mass > min_mass], V_grid[p_mass > min_mass]

        if autoscale:
            V_grid /= 3 * quiver_autoscale(X_grid, V_grid)

    return X_grid, V_grid


def cosine_similarity(x, y):
    return np.sum(x * y, axis=1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1))


def compute_angle_on_gridplot_between(adata_visualized_velocity, adata_optimal_velocity, basis='umap',):
    X_grid_scvelo, V_grid_scvelo = compute_velocity_on_grid(
        X_emb=adata_visualized_velocity.obsm[f'X_{basis}'],
        V_emb=adata_visualized_velocity.obsm[f'velocity_{basis}'],
        # density=density,
        autoscale=True,
        # smooth=smooth,
        # n_neighbors=n_neighbors,
        # min_mass=min_mass,
    )

    X_grid_best, V_grid_best = compute_velocity_on_grid(
        X_emb=adata_optimal_velocity.obsm[f'X_{basis}'],
        V_emb=adata_optimal_velocity.obsm[f'velocity_{basis}'],
        # density=density,
        autoscale=True,
        # smooth=smooth,
        # n_neighbors=n_neighbors,
        # min_mass=min_mass,
    )

    assert np.allclose(X_grid_scvelo, X_grid_best)

    return np.rad2deg(np.arccos(cosine_similarity(V_grid_scvelo, V_grid_best)))


def plot_uniformity_histogram(samples, number_bins=100, ax=None, density=True):
    if ax is None:
        fig, ax = plt.subplots()

    _, bins, _ = plt.hist(samples, bins=number_bins, label="p-values", density=density)
    N = len(samples)
    p = 1 / number_bins
    normalization = (N * np.diff(bins))[0]
    ax.hlines(N * p / normalization, color="red", linestyle="--", label="Exp. value under Uniformity", xmin=0, xmax=1)
    plt.fill_between(np.arange(0, 2), (N * p - 1.96 * np.sqrt(N * p * (1 - p))) / normalization,
                     (N * p + 1.96 * np.sqrt(N * p * (1 - p))) / normalization, color="red", alpha=0.3, edgecolor=None)


def plot_best_possible_velocities_statistic(Z_expr, best_possible_velocities_statistic, tested_cell_indices,
                                            max_value=None, cbar=True, markersize=1.5, ax=None,
                                            vector_friendly: bool = False):
    if ax is None:
        fig, ax = plt.subplots()

    if max_value is None:
        max_value = np.max(np.abs(best_possible_velocities_statistic))
    sc = ax.scatter(*Z_expr.T, c='grey', s=markersize, label="Not tested", linewidths=0)
    sc.set_rasterized(vector_friendly)
    sc = ax.scatter(*Z_expr[tested_cell_indices].T, c=best_possible_velocities_statistic, cmap='seismic_r',
                    vmax=max_value,
                    vmin=-max_value, s=markersize, label="Best test statistic", linewidths=0)
    sc.set_rasterized(vector_friendly)
    ax.axis('off')
    ax.set(box_aspect=1)
    if cbar:
        cbar = plt.colorbar(sc, ax=ax)
        cbar.locator = plt.MaxNLocator(nbins=5)


def plot_statistic_distribution(x_limited, values_limited, x_excluded, values_excluded,
                                color=None, bins=50, ax=None, vector_friendly: bool = False):
    if color is None:
        color = ["#02ADFF", "#FF776D"]
    if ax is None:
        fig, ax = plt.subplots()
    axScatter = ax
    divider = make_axes_locatable(axScatter)
    axHisty = divider.append_axes("right", 0.6, pad=0.1, sharey=axScatter)

    # make some labels invisible
    axHisty.yaxis.set_tick_params(labelleft=False)
    sc = axScatter.scatter(x_limited, values_limited, s=0.5, c=color[0], marker="s")
    sc.set_rasterized(vector_friendly)
    sc = axScatter.scatter(x_excluded, values_excluded, s=0.5, c=color[1], marker="s")
    sc.set_rasterized(vector_friendly)
    axScatter.set_xlabel("Position on unit circle rel. \nto visualised velocity [rad]")
    axScatter.set_ylabel("Test statistic")
    axScatter.set_xticks([0, np.pi, 2 * np.pi])
    labels = ['$0$', r'$\pi$', r'$2\pi$']
    axScatter.set_xticklabels(labels)
    axScatter.yaxis.set_major_locator(plt.MaxNLocator(3))

    axHisty.hist([values_limited, values_excluded], bins=bins, orientation='horizontal', density=True,
                 color=color, stacked=True)
    axHisty.set_xlabel("Density")
    axHisty.hlines(values_excluded[0], axHisty.get_xlim()[0], axHisty.get_xlim()[1], color='black',
                   linestyle='dashed', linewidth=1)


def plot_neighborhood(cell, Z_expr, neighborhoods, s=None, ax=None, vector_friendly: bool = False):
    if ax is None:
        fig, ax = plt.subplots()
    neighborhood = neighborhoods[cell]
    sc = ax.scatter(*Z_expr.T, c='grey', s=s)
    sc.set_rasterized(vector_friendly)
    sc = ax.scatter(*Z_expr[neighborhood].T, c='blue', label='neighborhood', s=s)
    sc.set_rasterized(vector_friendly)
    ax.axis('off')
