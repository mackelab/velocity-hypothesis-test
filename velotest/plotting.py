from typing import Dict, List, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects
from scvelo.plotting.utils import default_arrow
from sklearn.utils import check_random_state


def scatter(X_emb: np.ndarray, label_colormap: Union[Dict, List] = None, labels: pd.Series = None,
            ax: matplotlib.axes.Axes = None, title=None, marker="o", size=25, show_labels: bool = True, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8), dpi=150)

    if title is not None:
        ax.set_title(title)

    if labels is not None and show_labels:
        plot_labels(ax, X_emb, labels)

    glyph_colors = get_glyph_colors(X_emb, labels, label_colormap)
    random_state = check_random_state(0)
    draw_order = random_state.permutation(X_emb.shape[0])
    ax.scatter(*X_emb[draw_order].T, c=glyph_colors[draw_order], s=size, lw=0, alpha=0.5, marker=marker, **kwargs)

    return ax


def arrow_plot(
        X_emb: np.ndarray,
        V_emb: np.ndarray,
        p_values: np.ndarray = None,
        h0_rejected: np.ndarray = None,
        labels: pd.Series = None,
        label_colormap: Union[Dict, List] = None,
        ax: matplotlib.axes.Axes = None,
        title=None,
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

        ax.quiver(
            X_emb[irrelevant_velocities][:, 0], X_emb[irrelevant_velocities][:, 1],
            V_emb[irrelevant_velocities][:, 0] - X_emb[irrelevant_velocities][:, 0],
            V_emb[irrelevant_velocities][:, 1] - X_emb[irrelevant_velocities][:, 1],
            facecolor='darkgrey', edgecolor='face', **quiver_kwargs
        )
        ax.quiver(
            X_emb[significant][:, 0], X_emb[significant][:, 1], V_emb[significant][:, 0] - X_emb[significant][:, 0],
                                                                V_emb[significant][:, 1] - X_emb[significant][:, 1],
            color='black', **quiver_kwargs
        )

    if significance is None:
        scatter(X_emb, label_colormap, labels, ax)
    else:
        # 'multiplier' allows to scale the markers differently for different datasets
        # have to find an automatic way of doing it
        multiplier = 1
        if labels is not None:
            scatter(X_emb[not_tested], label_colormap, labels[not_tested], ax, marker="o", size=int(multiplier * 20),
                    show_labels=False)
            scatter(X_emb[not_significant], label_colormap, labels[not_significant], ax, marker="s",
                    size=int(multiplier * 10),
                    show_labels=False)
            scatter(X_emb[significant], label_colormap, labels[significant], ax, marker="*", size=int(multiplier * 60),
                    show_labels=False)
        else:
            scatter(X_emb[not_tested], label_colormap, ax=ax, marker="o", size=int(multiplier * 20))
            scatter(X_emb[not_significant], label_colormap, ax=ax, marker="s", size=int(multiplier * 10))
            scatter(X_emb[significant], label_colormap, ax=ax, marker="*", size=int(multiplier * 60))
    if labels is not None:
        plot_labels(ax, X_emb, labels)

    ax.set(xticks=[], yticks=[], box_aspect=1)
    if title is not None:
        ax.set_title(title)

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
