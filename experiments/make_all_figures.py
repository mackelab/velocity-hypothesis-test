import anndata as ad
import hydra
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scvelo
import torch
from hydra.utils import instantiate
from matplotlib.colors import Normalize
from omegaconf import DictConfig

from velotest.explicit_hypothesis_testing import compute_p_values
from velotest.hypothesis_testing import run_hypothesis_test_on, correct_for_multiple_testing
from velotest.neighbors import find_neighbors, find_neighbors_in_direction_of_velocity
from velotest.plotting import arrow_plot, plot_uniformity_histogram, plot_best_possible_velocities_statistic, \
    plot_statistic_distribution, plot_neighborhood, compute_angle_on_gridplot_between


def velocity_embedding_stream(adata, figsize, color_dict, cfg):
    fig, ax = plt.subplots(dpi=450, constrained_layout=True, figsize=figsize)
    ax.set(xticks=[], yticks=[])
    legend_loc = 'on data'
    if cfg.dataset.dataset_name == "organogenesis":
        legend_loc = 'none'
    V = None
    if cfg.dataset.dataset_name == "nystroem":
        V = adata.obsm['velocity_nystroem']
    scvelo.pl.velocity_embedding_stream(adata, basis=cfg.dataset.basis, color=cfg.dataset.scvelo_color, palette=color_dict, title='', frameon=False,
                                        ax=ax, xlabel='', ylabel='', show=False, legend_loc=legend_loc,
                                        legend_fontsize=5, legend_fontweight='normal', size=3, linewidth=0.25,
                                        arrow_size=0.25, add_margin=.07, V=V)  # figsize=(3.17*cm, 3.17*cm), dpi=450,
    ax.set_rasterized(True)
    ax.set_aspect('equal')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(f"fig/05_{cfg.dataset.dataset_name}-velocity_embedding_stream.pdf", transparent=True, dpi=1000)


def test_results(adata, Z_expr, Z_velocity_vector, color_dict, figsize, h0_rejected, labels,
                 uncorrected_p_values_exclusion, name):
    fig, ax = plt.subplots(dpi=450, figsize=figsize)
    ax.set_aspect('equal')
    arrow_plot(adata.obsm[Z_expr], adata.obsm[Z_expr] + adata.obsm[Z_velocity_vector], uncorrected_p_values_exclusion,
               h0_rejected,
               labels=labels, label_colormap=color_dict, ax=ax, fontsize=7, fontweight='normal', multiplier=0.15,
               plot_legend=False,
               box=False,
               vector_friendly=True)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.text(0.975, 0.025, f'{np.sum(h0_rejected) / len(adata) * 100:.1f}%', transform=fig.transFigure,
             fontsize=7, ha="right")
    plt.savefig(name, transparent=True, dpi=1000)


def neighborhoods(Z_expr, adata, cell, cfg, figsize, nn_indices, selected_neighbors):
    _, ax = plt.subplots(dpi=450, constrained_layout=True, figsize=figsize)
    plot_neighborhood(cell, adata.obsm[Z_expr], nn_indices, ax=ax, s=0.3, vector_friendly=True,
                      selected_neighbors=selected_neighbors[cell])
    ax.set_aspect('equal')
    plt.savefig(
        f"fig/05_{cfg.dataset.dataset_name}-neighborhood_{cell}-{cfg.dataset.number_neighbors_to_sample_from}.pdf",
        transparent=True, dpi=1000)


def statistic_function(cell, statistics, figsize, name):
    func = statistics[cell]
    complete_dom = func.get_domain()
    limited_dom = func.get_domain(exclusion_angle=np.deg2rad(10))
    assert complete_dom[0][1] > np.deg2rad(10) and complete_dom[-1][0] < 2 * np.pi - np.deg2rad(
        10), "Original domain is split in exclusion angle, need to implement more extensive logic"
    excluded_dom = [(complete_dom[0][0], limited_dom[0][0]), (limited_dom[-1][1], complete_dom[-1][1])]
    x_limited, values_limited = func.sample_from(limited_dom)
    x_excluded, values_excluded = func.sample_from(excluded_dom)
    fig, ax = plt.subplots(dpi=450, constrained_layout=True,
                           figsize=figsize)
    plot_statistic_distribution(x_limited, values_limited, x_excluded, values_excluded, ax=ax, vector_friendly=True)
    plt.savefig(name, transparent=True, dpi=1000)


def determine_significant_vectors(gamma_rad, non_empty_neighborhoods_indices, statistics):
    uncorrected_p_values = compute_p_values(statistics, exclusion_rad=gamma_rad)
    pvals_corrected = correct_for_multiple_testing(uncorrected_p_values[uncorrected_p_values != 2], 'bonferroni')
    h0_rejected_temp = pvals_corrected < 0.05
    h0_rejected = np.zeros(len(statistics), dtype=bool)
    h0_rejected[non_empty_neighborhoods_indices] = h0_rejected_temp
    return h0_rejected, uncorrected_p_values


@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    print(cfg)
    Z_expr = f"X_{cfg.dataset.basis}"
    Z_velocity_vector = f"velocity_{cfg.dataset.basis}"
    cm = 1 / 2.54  # centimeters in inches
    figsize = (4.4 * cm, 3.3 * cm)

    mpl.rc_file("../matplotlibrc-embeddings")

    adata = instantiate(cfg.dataset)

    # Run test
    uncorrected_p_values_exclusion, h0_rejected, debug_dict = run_hypothesis_test_on(adata,
                                                                                     number_neighbors_to_sample_from=cfg.dataset.number_neighbors_to_sample_from,
                                                                                     null_distribution="velocities-explicit",
                                                                                     cosine_empty_neighborhood=None,
                                                                                     correction="bonferroni",
                                                                                     restrict_to_velocity_genes=True,
                                                                                     exclusion_degree=10,
                                                                                     basis=cfg.dataset.basis)
    statistics = debug_dict['statistics']
    non_empty_neighborhoods_indices = np.where(uncorrected_p_values_exclusion != 2)[0]

    color_dict = dict(zip(adata.obs[cfg.dataset.scvelo_color].cat.categories, adata.uns[cfg.dataset.scvelo_cluster_colors]))
    labels = adata.obs[cfg.dataset.scvelo_color]

    velocity_embedding_stream(adata, figsize, color_dict, cfg)

    ### Test results
    test_results(adata, Z_expr, Z_velocity_vector, color_dict, figsize, h0_rejected, labels,
                 uncorrected_p_values_exclusion,
                 f"fig/05_{cfg.dataset.dataset_name}-test_results-10.pdf")

    rejection_wo_correction = uncorrected_p_values_exclusion < 0.05
    rejection_wo_correction[uncorrected_p_values_exclusion == 2] = False
    test_results(adata, Z_expr, Z_velocity_vector, color_dict, figsize, rejection_wo_correction, labels,
                 uncorrected_p_values_exclusion,
                 f"fig/05_{cfg.dataset.dataset_name}-test_results-10-wo_correction.pdf")

    h0_rejected_5, uncorrected_p_values_5 = determine_significant_vectors(np.deg2rad(5),
                                                                          non_empty_neighborhoods_indices,
                                                                          statistics)
    test_results(adata, Z_expr, Z_velocity_vector, color_dict, figsize, h0_rejected_5, labels, uncorrected_p_values_5,
                 f"fig/05_{cfg.dataset.dataset_name}-test_results-05.pdf")

    h0_rejected_225, uncorrected_p_values_225 = determine_significant_vectors(np.deg2rad(22.5),
                                                                              non_empty_neighborhoods_indices,
                                                                              statistics)
    test_results(adata, Z_expr, Z_velocity_vector, color_dict, figsize, h0_rejected_225, labels,
                 uncorrected_p_values_225,
                 f"fig/05_{cfg.dataset.dataset_name}-test_results-22_5.pdf")

    h0_rejected_45, uncorrected_p_values_45 = determine_significant_vectors(np.deg2rad(45),
                                                                            non_empty_neighborhoods_indices,
                                                                            statistics)
    test_results(adata, Z_expr, Z_velocity_vector, color_dict, figsize, h0_rejected_45, labels, uncorrected_p_values_45,
                 f"fig/05_{cfg.dataset.dataset_name}-test_results-45.pdf")

    h0_rejected_90, uncorrected_p_values_90 = determine_significant_vectors(np.deg2rad(90),
                                                                            non_empty_neighborhoods_indices,
                                                                            statistics)
    test_results(adata, Z_expr, Z_velocity_vector, color_dict, figsize, h0_rejected_90, labels, uncorrected_p_values_90,
                 f"fig/05_{cfg.dataset.dataset_name}-test_results-90.pdf")

    h0_rejected_0, uncorrected_p_values_0 = determine_significant_vectors(None, non_empty_neighborhoods_indices,
                                                                          statistics)
    test_results(adata, Z_expr, Z_velocity_vector, color_dict, figsize, h0_rejected_0, labels, uncorrected_p_values_0,
                 f"fig/05_{cfg.dataset.dataset_name}-test_results-00.pdf")

    rejection_wo_correction_0 = uncorrected_p_values_0 < 0.05
    rejection_wo_correction_0[uncorrected_p_values_0 == 2] = False
    test_results(adata, Z_expr, Z_velocity_vector, color_dict, figsize, rejection_wo_correction_0, labels,
                 uncorrected_p_values_0,
                 f"fig/05_{cfg.dataset.dataset_name}-test_results-0-wo_correction.pdf")

    ### Test-optimal velocity
    best_possible_velocities = [statistic.get_max_value(offset=True)[0] for statistic in statistics if
                                statistic is not None]
    best_velocity_range_mean = []
    for range_best in best_possible_velocities:
        best_velocity_range_mean.append((range_best[0] + range_best[1]) / 2)
    best_velocity_range_mean = np.array(best_velocity_range_mean)
    best_velocities = np.stack([np.cos(best_velocity_range_mean), np.sin(best_velocity_range_mean)], axis=-1)
    adata_best = adata.copy()
    adata_best.obsm[Z_velocity_vector] = np.zeros((adata.shape[0], 2))
    adata_best.obsm[Z_velocity_vector][non_empty_neighborhoods_indices] = best_velocities

    angles = compute_angle_on_gridplot_between(adata, adata_best, basis=cfg.dataset.basis)

    from matplotlib.colors import LinearSegmentedColormap

    norm = Normalize(vmin=0, vmax=90)
    # cmap = plt.get_cmap("Greys")
    cmap = LinearSegmentedColormap.from_list("lightgrey_to_black", ["#eeefff", "#000000"])
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    colors = sm.to_rgba(angles)

    adata_plotting = ad.AnnData(adata_best.X, obs=adata_best.obs.copy())
    adata_plotting.obsm[Z_expr] = adata_best.obsm[Z_expr]
    adata_plotting.obsm[Z_velocity_vector] = adata_best.obsm[Z_velocity_vector]
    adata_plotting.layers['velocity'] = adata_best.layers['velocity']
    adata_plotting.uns[cfg.dataset.scvelo_cluster_colors] = adata_best.uns[cfg.dataset.scvelo_cluster_colors]
    color_dict_plotting = dict(zip(adata_plotting.obs[cfg.dataset.scvelo_color].cat.categories,
                                   adata_plotting.uns[cfg.dataset.scvelo_cluster_colors]))


    fig, ax = plt.subplots(dpi=450, figsize=figsize)
    ax.set(xticks=[], yticks=[])

    pl = scvelo.pl.velocity_embedding_grid(
        adata_plotting, basis=cfg.dataset.basis, dpi=450, title='', ax=ax, frameon=False, xlabel='',
        ylabel='', fontsize=16, show=False, size=3,
        linewidth=.25, arrow_size=2, arrow_length=2, arrow_color=colors, edgecolors=None,
        rasterized=True, figsize=figsize, color=cfg.dataset.scvelo_color, palette=color_dict_plotting)
    ax.set_aspect('equal')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    pl.set_rasterized(True)
    plt.savefig(f"fig/05_{cfg.dataset.dataset_name}-optimal_velocity.pdf", transparent=True, dpi=1000)

    ### Untestable
    fig, ax = plt.subplots(dpi=450, figsize=figsize)

    sc = plt.scatter(*adata.obsm[Z_expr].T, s=0.5, color='lightgrey', edgecolors='none')
    sc.set_rasterized(True)
    sc = plt.scatter(*adata.obsm[Z_expr][uncorrected_p_values_exclusion == 2].T, s=1, color='red', edgecolors='none')
    sc.set_rasterized(True)
    ax.set(xticks=[], yticks=[])
    ax.set_aspect('equal')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(f"fig/05_{cfg.dataset.dataset_name}-untestable.pdf", transparent=True, dpi=1000)
    print(f"Untested cells for {cfg.dataset.dataset_name}: {np.sum(uncorrected_p_values_exclusion == 2)}, {np.sum(uncorrected_p_values_exclusion == 2) / len(adata): .2%}")

    ### Neighborhoods
    nn_indices = find_neighbors(adata.obsm[Z_expr], k_neighbors=cfg.dataset.number_neighbors_to_sample_from)
    selected_neighbors = find_neighbors_in_direction_of_velocity(torch.tensor(adata.obsm[Z_expr]), torch.tensor(
        adata.obsm[Z_expr] + adata.obsm[Z_velocity_vector]), nn_indices)

    for cell in cfg.dataset.interesting_neighborhoods:
        neighborhoods(Z_expr, adata, cell, cfg, figsize, nn_indices, selected_neighbors)

    ### Best test statistic for every cell
    best_possible_velocities_statistic = [statistic.get_max_value()[1] for statistic in statistics if
                                          statistic is not None]
    fig, ax = plt.subplots(dpi=450, constrained_layout=True, figsize=figsize)

    plot_best_possible_velocities_statistic(adata.obsm[Z_expr], best_possible_velocities_statistic,
                                            non_empty_neighborhoods_indices,
                                            ax=ax,
                                            vector_friendly=True,
                                            max_value=0.97)
    ax.set_aspect('equal')
    plt.savefig(f"fig/05_{cfg.dataset.dataset_name}-best_test_statistic-colorbar.pdf", transparent=True, dpi=1000)

    fig, ax = plt.subplots(dpi=450, figsize=figsize)
    ax.set_aspect('equal')
    plot_best_possible_velocities_statistic(adata.obsm[Z_expr], best_possible_velocities_statistic,
                                            non_empty_neighborhoods_indices,
                                            ax=ax,
                                            markersize=0.5,
                                            vector_friendly=True,
                                            max_value=0.97,
                                            cbar=False)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(f"fig/05_{cfg.dataset.dataset_name}-best_test_statistic.pdf", transparent=True, dpi=1000)

    mpl.rc_file("../matplotlibrc")
    fig, ax = plt.subplots(dpi=450, constrained_layout=True,
                           figsize=(3.2 * cm, 1.8 * cm))
    _ = plt.hist(best_possible_velocities_statistic, bins=100, density=True)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.set_xticks([-1, 0, 1], labels=[-1, "", 1])
    ax.yaxis.set_major_locator(plt.MaxNLocator(2))
    assert plt.xlim()[0] >= -1.07
    assert plt.xlim()[1] <= 1.07
    plt.xlim(-1.07, 1.07)
    print(plt.ylim()[1])
    assert plt.ylim()[1] <= 8.38
    plt.ylim(top=8.38)
    # plt.xlabel("Best test statistic among all random velocities")
    # plt.ylabel("Count")
    plt.savefig(f"fig/05_{cfg.dataset.dataset_name}-best_test_statistic-histogram.pdf", transparent=True, dpi=450)

    ### Graphical diagnostic
    mpl.rc_file("../matplotlibrc")
    fig, ax = plt.subplots(dpi=450, constrained_layout=True,
                           figsize=((3.5) * cm, 1.95 * cm))  # figsize=((2.5+0.71) * cm, 1.75 * cm))
    ax.xaxis.set_major_locator(plt.MaxNLocator(2))
    ax.set_xticks([0, 1])
    # ax.set(yticks=[])

    plot_uniformity_histogram(uncorrected_p_values_0[uncorrected_p_values_0 < 2], ax=ax)
    assert plt.ylim()[1] <= 5.43
    plt.ylim(top=5.43)
    # plt.xlabel("p-value")

    plt.savefig(f"fig/05_{cfg.dataset.dataset_name}-uniformity_histogram-wide.pdf", transparent=True, dpi=450)

    ### Statistic function
    for cell in [0, 9, 12, 27, 1244]:
        if statistics[cell] is not None:
            try:
                statistic_function(cell, statistics, (7.2 * cm, 4.75 * cm),
                                   f"fig/05_{cfg.dataset.dataset_name}-test_statistic-explanation-stacked-cell-{cell}.pdf")
            except AssertionError:
                pass


if __name__ == "__main__":
    main()
