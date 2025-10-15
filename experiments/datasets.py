import os
from pathlib import Path

import anndata as ad
import pandas as pd
import scvelo
from scanpy.readwrite import _download


def get_pancreas_stochastic_data(**kwargs):
    adata = scvelo.datasets.pancreas()
    scvelo.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
    scvelo.pp.moments(adata, n_pcs=30, n_neighbors=30)

    # Compute velocity
    scvelo.tl.velocity(adata)

    # Compute 2D embedding of velocity vectors
    scvelo.tl.velocity_graph(adata)
    scvelo.tl.velocity_embedding(adata)
    return adata


def get_pancreas_dynamical_data(**kwargs):
    adata = scvelo.datasets.pancreas()
    scvelo.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
    scvelo.pp.moments(adata, n_pcs=30, n_neighbors=30)

    scvelo.tl.recover_dynamics(adata, n_jobs=7)

    # Compute velocity
    scvelo.tl.velocity(adata, mode='dynamical')

    # Compute 2D embedding of velocity vectors
    scvelo.tl.velocity_graph(adata)
    scvelo.tl.velocity_embedding(adata)
    return adata


def get_dentateyrus_data(**kwargs):
    adata = scvelo.datasets.dentategyrus()
    scvelo.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
    scvelo.pp.moments(adata, n_pcs=30, n_neighbors=30)

    # Compute velocity
    scvelo.tl.velocity(adata)

    # Compute 2D embedding of velocity vectors
    scvelo.tl.velocity_graph(adata)
    scvelo.tl.velocity_embedding(adata)
    return adata


def get_bonemarrow_data(**kwargs):
    adata = scvelo.datasets.bonemarrow()
    scvelo.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
    scvelo.pp.moments(adata, n_pcs=30, n_neighbors=30)

    # Compute velocity
    scvelo.tl.recover_dynamics(adata, n_jobs=8)
    scvelo.tl.velocity(adata, mode="dynamical")

    # Compute 2D embedding of velocity vectors
    scvelo.tl.velocity_graph(adata)
    scvelo.tl.velocity_embedding(adata)
    return adata


def get_covid_data(**kwargs):
    # Reproduction of Wilk et al.'s velocity analysis in "A single-cell atlas of the peripheral immune response in patients with severe COVID-19"
    # https://figshare.com/articles/dataset/Reproduction_of_Wilk_et_al_s_velocity_analysis_in_A_single-cell_atlas_of_the_peripheral_immune_response_in_patients_with_severe_COVID-19_/30369448
    data_dir = os.path.join("data", "covid")
    if not os.path.exists(data_dir):
        try:
            os.makedirs(data_dir)
            urls = ["https://figshare.com/ndownloader/files/58774222",
                    "https://figshare.com/ndownloader/files/58774225",
                    "https://figshare.com/ndownloader/files/58774228",
                    "https://figshare.com/ndownloader/files/58774231",
                    "https://figshare.com/ndownloader/files/58776682"]
            filenames = ["pb1.obsm.csv",
                         "pb1.obs.csv",
                         "pb1.nmat.mtx",
                         "pb1.emat.mtx",
                         "pb1.var_names.txt"]
            for filename, url in zip(filenames, urls):
                _download(url, Path(os.path.join(data_dir, filename)))
        except Exception as e:
            print(f"Could not download data: {e}")
            if os.path.exists(data_dir):
                import shutil
                shutil.rmtree(data_dir)
            raise e

    adata = ad.read_mtx(os.path.join(data_dir, "pb1.emat.mtx"))  # spliced layer
    adata.layers['spliced'] = adata.X
    ndata = ad.read_mtx(os.path.join(data_dir, "pb1.nmat.mtx"))  # unspliced layer
    adata.layers['unspliced'] = ndata.X
    pd_obs = pd.read_csv(os.path.join(data_dir, "pb1.obs.csv"))  # annotation of observations
    pd_obsm = pd.read_csv(os.path.join(data_dir, "pb1.obsm.csv"))  # embedding
    adata.obs = pd_obs
    adata.obsm["X_umap"] = pd_obsm.values
    # pd_obs_names = pd.read_table(os.path.join(dir, "pb.obs_names.txt"), header=None)[0].tolist() #names of observations
    # adata.obs_names = pd_obs_names
    pd_var_names = pd.read_table(os.path.join(data_dir, "pb1.var_names.txt"), header=None)[
        0].tolist()  # names of variables
    adata.var_names = pd_var_names

    scvelo.pp.filter_and_normalize(adata, min_shared_counts=10, n_top_genes=3000, log=False)
    scvelo.pp.moments(adata, n_pcs=30, n_neighbors=20)
    # scvelo.tl.recover_dynamics(adata)
    scvelo.tl.velocity(adata, mode='stochastic')

    print(adata.var.velocity_genes.sum())

    scvelo.tl.velocity_graph(adata)
    scvelo.pl.velocity_embedding_stream(adata, basis='umap', color='cell.type', title='', frameon=False, show=False)
    return adata


def get_gastrulation_erythroid_data(**kwargs):
    adata = scvelo.datasets.gastrulation_erythroid()
    scvelo.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
    scvelo.pp.moments(adata, n_pcs=30, n_neighbors=30)

    # Compute velocity
    scvelo.tl.recover_dynamics(adata, n_jobs=8)
    scvelo.tl.velocity(adata, mode="dynamical")

    # Compute 2D embedding of velocity vectors
    scvelo.tl.velocity_graph(adata)  # , mode_neighbors="connectivities")
    scvelo.tl.velocity_embedding(adata)
    return adata


def get_nystroem_data(**kwargs):
    adata = scvelo.datasets.pancreas()
    scvelo.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
    scvelo.pp.moments(adata, n_pcs=30, n_neighbors=30)

    # Compute velocity
    scvelo.tl.velocity(adata)

    import velocity as nystroem
    adata_nystroem = adata.copy()
    nystroem.project.pca.pca_project(adata_nystroem)
    nystroem.project.nystroem.nystroem_project(adata_nystroem, basis="umap")

    adata.obsm["velocity_nystroem"] = adata_nystroem.obsm["velocity_umap"].copy()
    adata.obsm["X_nystroem"] = adata_nystroem.obsm["X_umap"].copy()

    scvelo.tl.velocity_graph(adata)
    scvelo.pl.velocity_embedding(adata, V=adata.obsm['velocity_nystroem'], basis='nystroem', show=False)
    return adata


def get_developing_mouse_brain_data(**kwargs):
    # Reproduction of Abdelaal et al.'s SIRV velocity embeddings on the developing mouse brain dataset
    # https://figshare.com/articles/dataset/Reproduction_of_Abdelaal_et_al_s_SIRV_velocity_embeddings/30369586
    file = os.path.join("data", "developing_mouse_brain", "HybISS_imputed_velocity.h5ad")
    return scvelo.read(file, backup_url="https://figshare.com/ndownloader/files/58776778")


def get_organogenesis_data(**kwargs):
    # Reproduction of Abdelaal et al.'s SIRV velocity embeddings on the organogenesis dataset
    # https://figshare.com/articles/dataset/Reproduction_of_Abdelaal_et_al_s_SIRV_velocity_embeddings/30369586
    file = os.path.join("data", "organogenesis", "SeqFISH_imputed_velocity.h5ad")
    return scvelo.read(file, backup_url="https://figshare.com/ndownloader/files/58776781")


def get_veloviz_data(**kwargs):
    # Reproduction of Atta et al.'s veloviz velocity embeddings on the pancreas endocrinogenesis dataset
    # https://figshare.com/articles/dataset/Reproduction_of_Atta_et_al_s_veloviz_velocity_embeddings_on_the_pancreas_endocrinogenesis_dataset/30369916?file=58778401
    file = os.path.join("data", "veloviz", "veloviz_pancreas.h5ad")
    adata = scvelo.read(file, backup_url="https://figshare.com/ndownloader/files/58778401")
    del adata.obsm['X_umap'], adata.obsm['X_veloviz2'], adata.obsm['velocity_nystroem'], adata.obsm['velocity_scvelo'], \
        adata.obsm['velocity_veloviz'], adata.obsm['velocity_veloviz2']
    scvelo.pl.velocity_embedding(adata, basis='veloviz', color='clusters', show=False)
    return adata
