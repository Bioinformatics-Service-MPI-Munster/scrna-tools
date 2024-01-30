import os
from functools import wraps
import numpy as np
import scipy
import scanpy as sc
import scanpy.external as sce
import pandas as pd
import matplotlib.pyplot as plt
import copy
from ._core import VData
from .process import batch_correct
from .plotting.celloracle import celloracle_pca_plot
import logging

try:
    import celloracle as co
    from celloracle.applications import Gradient_calculator, Oracle_development_module
    
    with_celloracle = True
except (ModuleNotFoundError, OSError) as e:
    with_celloracle = False

logger = logging.getLogger(__name__)


def requires_celloracle(func):
    """Checks if celloracle is installed in path"""
    
    @wraps(func)
    def wrapper_celloracle(*args, **kwargs):
        if not with_celloracle:
            raise RuntimeError("celloracle is not installed, cannot "
                               "run code that depends on it!")
        return func(*args, **kwargs)
    return wrapper_celloracle


def _estimate_k(adata):
    n_cell = adata.shape[0]
    return int(0.025 * n_cell)


@requires_celloracle
def convert_vdata_to_celloracle(
    vdata, 
    obs_key,
    obsm_key,
    base_grn,
    n_hvg=3000,
    n_pcs=100,
    k=None,
    batch_key='sample',
    batch_algorithm='harmony',
    counts_key='counts',
    size_factor_key='size_factors',
    include_tfs=True,
    use_batch_for_hvg=True,
    cc_regression=False,
    plot_folder=None,
):
    adata = vdata.adata_view.copy()
    
    # subset to HVGs
    sc.pp.highly_variable_genes(
        adata, 
        n_top_genes=n_hvg, 
        batch_key=batch_key,
        subset=False
    )
    
    # subset to TFs
    if include_tfs:
        tfs = set(base_grn.columns)
        is_tf = [True if g in tfs else False for g in adata.var.index.to_list()]
        adata.var['_co_subset'] = np.logical_or(adata.var['highly_variable'], is_tf)
        adata_co = adata[:, adata.var['_co_subset']].copy()
    else:
        adata_co = adata[:, adata.var['highly_variable']].copy()

    # create normalised raw counts that are not log transformed
    adata.X = adata.layers[counts_key].copy()
    adata.X /= adata.obs[size_factor_key].values[:, None]
    adata.X = scipy.sparse.csr_matrix(adata.X)
    #adata.layers['raw_count'] = adata.X.copy()

    # add obs colors in scanpy format
    default_colors = vdata.default_colors(obs_key)
    adata.uns[f'{obs_key}_colors'] = [default_colors[c] for c in adata.obs[obs_key].dtype.categories]
    
    oracle = co.Oracle()
    oracle.import_anndata_as_raw_count(
        adata=adata_co,
        cluster_column_name=obs_key,
        embedding_name=obsm_key,
    )
    oracle.import_TF_data(TF_info_matrix=base_grn)
    
    # run PCA
    sc.tl.pca(
        adata_co, 
        svd_solver='arpack', 
        n_comps=n_pcs, 
        random_state=42
    )
    explained_variance_ratios = adata_co.uns['pca']['variance_ratio']
    
    # batch correct?
    if batch_key is not None:
        batch_correct(
            adata_co,
            batch_algorithm=batch_algorithm,
            batch_key=batch_key,
            use_batch_for_hvg=use_batch_for_hvg,
            cc_regression=cc_regression,
        )
        oracle.pcs = adata_co.obsm['X_integrated']
    else:
        oracle.pcs = adata_co.obsm['X_pca']
    
    n_comps_estimated = np.where(np.diff(np.diff(np.cumsum(explained_variance_ratios))>0.002))[0][0]
    n_comps = min(50, n_comps_estimated)
    
    if plot_folder is not None:
        fig, ax = plt.subplots()
        plotting.celloracle_pca_plot(
            adata_co.uns['pca'], 
            n_comps=n_comps,
            n_comps_estimated=n_comps_estimated,
            max_comps=n_pcs,
            ax=ax,
        )
        fig.savefig(os.path.join(plot_folder, 'pca_explained_variance_chosen_pcs.pdf'))
        plt.close(fig)
    
    logger.info(f'Estimated PCs: {n_comps_estimated}; chosen PCs: {n_comps}')
    
    if k is None:
        k = _estimate_k(adata_co)

    oracle.knn_imputation(
        n_pca_dims=n_comps,
        k=k,
        balanced=True, 
        b_sight=k*8,
        b_maxl=k*4, 
        n_jobs=4
    )
    
    data = {
        'k': k,
        'n_comps': n_comps,
        'n_comps_estimated': n_comps_estimated,
        'pca': adata_co.uns['pca'].copy(),
    }
    
    return oracle, data


def oracle_links(
    oracle,
    obs_key,
    links_alpha=10,
    links_bagging_n=20,
    links_model_method="bagging_ridge",
    links_p=0.001,
    links_threshold_number=10000,
    links_weight='coef_abs',
    fit_alpha=10,
    plot_folder=None,
):
    logger.info("Calculating Links")
    links = oracle.get_links(
        cluster_name_for_GRN_unit=obs_key,
        alpha=links_alpha, 
        verbose_level=10, 
        bagging_number=links_bagging_n,
        model_method=links_model_method
    )
    
    logger.info("Filtering Links")
    links_filtered = copy.deepcopy(links)
    links_filtered.filter_links(
        p=links_p, 
        weight=links_weight, 
        threshold_number=links_threshold_number,
    )
    
    if plot_folder is not None:
        links_filtered.plot_degree_distributions(
            plot_model=True,
            save=plot_folder,
        )
    
    oracle.get_cluster_specific_TFdict_from_Links(
        links_object=links_filtered
    )
    
    oracle.fit_GRN_for_simulation(
        alpha=fit_alpha, 
        use_cluster_specific_TFdict=True
    )

    return oracle, links, links_filtered


def celloracle_simulate_shift(
    oracle,
    perturb_condition,
    celloracle_key,
    n_neighbors=200,
    n_propagation=3,
    sampled_fraction=1,
    knn_random=True,
    sigma_corr=0.05,
    threads_knn=1,
    threads_correlation=1,
    transfer_adata=None,
    copy=True,
):
    
    if copy:
        oracle = oracle.copy()
    
    oracle.simulate_shift(
        perturb_condition=perturb_condition,
        n_propagation=n_propagation,
    )

    oracle.estimate_transition_prob(
        n_neighbors=n_neighbors,
        knn_random=knn_random,
        sampled_fraction=sampled_fraction,
        n_jobs=threads_knn,
        threads=threads_correlation,
    )

    # Calculate embedding
    oracle.calculate_embedding_shift(
        sigma_corr=sigma_corr
    )
    
    if transfer_adata is not None:
        if 'celloracle' not in transfer_adata.uns.keys():
            transfer_adata.uns['celloracle'] = dict()
        
        obsm_key_original = oracle.embedding_name
        layers_key_simulated = f'__celloracle__{celloracle_key}__simulated'
        obsm_key_delta = f'__celloracle__{celloracle_key}__{obsm_key_original}_delta'
        obsp_key_transition_probability = f'__celloracle__{celloracle_key}__transition_probability'
        
        celloracle_data = {
            'type': 'ko',
            'perturb_condition': perturb_condition,
            'genes': genes,
            'key': celloracle_key,
            'obsm_key_original': obsm_key_original,
            'obsm_key_delta': obsm_key_delta,
            'obsp_key_transition_probability': obsp_key_transition_probability,
            'layers_key_simulated': layers_key_simulated,
        }
        
        transfer_adata.layers[layers_key_simulated] = oracle.adata.layers['simulated_count']
        transfer_adata.obsm[obsm_key_delta] = oracle.delta_embedding.copy()
        transfer_adata.obsp[obsp_key_transition_probability] = scipy.sparse.csr_matrix(oracle.transition_prob)
        transfer_adata.uns['celloracle'][celloracle_key] = celloracle_data
    
    return oracle


def celloracle_simulate_knockout(
    oracle,
    genes,
    celloracle_key=None,
    n_neighbors=200,
    n_propagation=3,
    sampled_fraction=1,
    knn_random=True,
    sigma_corr=0.05,
    threads_knn=1,
    threads_correlation=1,
    transfer_adata=None,
    copy=True,
):
    if isinstance(genes, (str, bytes)):
        genes = [genes]

    perturb_condition = {
        gene: 0.0 for gene in genes
    }
    
    celloracle_key = celloracle_key or 'ko__' + '_'.join(genes)
    
    return celloracle_simulate_shift(
        oracle,
        perturb_condition,
        celloracle_key=celloracle_key,
        n_neighbors=n_neighbors,
        n_propagation=n_propagation,
        sampled_fraction=sampled_fraction,
        knn_random=knn_random,
        sigma_corr=sigma_corr,
        threads_knn=threads_knn,
        threads_correlation=threads_correlation,
        transfer_adata=transfer_adata,
        copy=copy,
    )




# base_grn = pd.read_parquet('/Users/kkruse/data/celloracle/TFinfo_data/mm9_mouse_atac_atlas_data_TSS_and_cicero_0.9_accum_threshold_10.5_DF_peaks_by_TFs_v202204.parquet')
# adata = sc.read('/Users/kkruse/project-results/bovay/20210226-embryo61-intestine-mesentery/adata/intestine_e18_and_ec_gfpp_and_mesentery.h5ad')
# from scrna_tools import VData
# from scrna_tools.process import batch_correct
# vdata_ec = VData(adata)
# vdata_ec.add_view_constraint('endothelial')
# vdata_ec.add_categorical_obs_constraint(
#     'ec_sub_celltype',
#     [
#         'Arterial EC 1', 'Arterial EC 2', 'Arterial EC 3',
#         'Esm1+ EC',
#         'Venous EC 1', 'Venous EC 2'
#     ]
# )

# oracle = convert_vdata_to_celloracle(
#     vdata_ec,
#     'ec_sub_celltype',
#     '__view__endothelial__umap',
#     base_grn,
# )
# oracle.to_hdf5('/Users/kkruse/tmp/celloracle-test/test.celloracle.oracle')

# links, links_filtered = oracle_links(
#     oracle,
#     'ec_sub_celltype',
#     plot_folder='/Users/kkruse/tmp/celloracle-test',
# )

# links_filtered.to_hdf5('/Users/kkruse/tmp/celloracle-test/links_filtered.celloracle.links')
# links.to_hdf5('/Users/kkruse/tmp/celloracle-test/links.celloracle.links')

# oracle.get_cluster_specific_TFdict_from_Links(links_object=links_filtered)
# oracle.fit_GRN_for_simulation(alpha=10, use_cluster_specific_TFdict=True)
# oracle.to_hdf5('/Users/kkruse/tmp/celloracle-test/test.celloracle.oracle')

# oracle = co.load_hdf5('/Users/kkruse/tmp/celloracle-test/test.celloracle.oracle')
# # oracle = co.load_hdf5('/home/kkruse/tmp/celloracle-test/test.celloracle.oracle')

# oracle.simulate_shift(
#     perturb_condition={
#         'Sox17': 0.0
#     },
#     n_propagation=3
# )

# oracle.estimate_transition_prob(
#     n_neighbors=200,
#     knn_random=True,
#     sampled_fraction=1,
#     n_jobs=1,
# )

# # Calculate embedding
# oracle.calculate_embedding_shift(sigma_corr=0.05)


# fig, ax = plt.subplots(1, 2,  figsize=[13, 6])
# scale = 15
# # Show quiver plot
# oracle.plot_quiver(scale=scale, ax=ax[0])
# ax[0].set_title(f"Simulated cell identity shift vector: Sox17 KO")

# # Show quiver plot that was calculated with randomized graph.
# oracle.plot_quiver_random(scale=scale, ax=ax[1])
# ax[1].set_title(f"Randomized simulation vector")
# fig.savefig(os.path.join('/Users/kkruse/tmp/celloracle-test', 'quiver_sox17.pdf'))
# plt.close(fig)

# n_grid = 40
# oracle.calculate_p_mass(smooth=0.8, n_grid=n_grid, n_neighbors=200)
# oracle.suggest_mass_thresholds(n_suggestion=8)


# min_mass = 2.
# oracle.calculate_mass_filter(min_mass=min_mass, plot='')
# mass_filter = (oracle.total_p_mass < min_mass)
# fig, ax = plt.subplots(figsize=[5,5])
# ax.scatter(oracle.embedding[:, 0], oracle.embedding[:, 1], c="lightgray", s=10)
# ax.scatter(oracle.flow_grid[~mass_filter, 0],
#         oracle.flow_grid[~mass_filter, 1],
#         c="black", s=0.5)
# ax.set_title("Grid points selected")
# ax.axis("off")
# fig.savefig(os.path.join('/Users/kkruse/tmp/celloracle-test', 'mass_filter.pdf'))



# oracle.mass_filter = (oracle.total_p_mass < min_mass)

# fig, ax = plt.subplots(1, 2,  figsize=[13, 6])
# scale_simulation = 15
# oracle.plot_simulation_flow_on_grid(scale=scale_simulation, ax=ax[0])
# ax[0].set_title(f"Simulated cell identity shift vector: Sox17 KO")
# # Show quiver plot that was calculated with randomized graph.
# oracle.plot_simulation_flow_random_on_grid(scale=scale_simulation, ax=ax[1])
# ax[1].set_title(f"Randomized simulation vector")
# fig.savefig(os.path.join('/Users/kkruse/tmp/celloracle-test', 'vector_field_sox17.pdf'))
# plt.close(fig)

# goi = "Sox17"
# fig = sc.pl.umap(oracle.adata, color=[goi, oracle.cluster_column_name],
#                  layer="raw_count", use_raw=False, cmap="viridis",
#             save=False,
#             return_fig=True)
# fig.savefig(os.path.join('/Users/kkruse/tmp/celloracle-test', 'umap.pdf'))
# plt.close(fig)


# #
# # Comparison to dpt gradient
# #
# gradient = Gradient_calculator(
#     oracle_object=oracle, 
#     pseudotime_key="dpt_ec_non_mitotic"
# )

# gradient.calculate_p_mass(smooth=0.8, n_grid=n_grid, n_neighbors=200)
# gradient.calculate_mass_filter(min_mass=min_mass, plot=True)

# gradient.transfer_data_into_grid(
#     args={"method": "knn", "n_knn":50},
#     plot=False,
# )

# fig, ax = plt.subplots()
# co.visualizations.development_module_visualization.plot_pseudotime_on_grid(
#     gradient, ax=ax, s=40, show_background=True
# )
# fig.savefig(os.path.join('/Users/kkruse/tmp/celloracle-test', 'dpt_gradient.pdf'))
# plt.close(fig)

# gradient.calculate_gradient()

# # Show results
# #gradient.visualize_results(scale=scale, s=5)

# fig, ax = plt.subplots()
# co.visualizations.development_module_visualization.plot_pseudotime_on_grid(gradient, ax=ax, s=5, show_background=True)
# co.visualizations.development_module_visualization.plot_reference_flow_on_grid(gradient, ax=ax, scale=scale*2, show_background=False, s=2)
# fig.savefig(os.path.join('/Users/kkruse/tmp/celloracle-test', 'dpt_gradient_flow.pdf'))
# plt.close(fig)

# gradient.to_hdf5(os.path.join('/Users/kkruse/tmp/celloracle-test', 'dpt.celloracle.gradient'))



# # Make Oracle_development_module to compare two vector field
# dev = Oracle_development_module()

# # Load development flow
# dev.load_differentiation_reference_data(gradient_object=gradient)

# # Load simulation result
# dev.load_perturb_simulation_data(oracle_object=oracle)


# # Calculate inner produc scores
# dev.calculate_inner_product()
# dev.calculate_digitized_ip(n_bins=10)