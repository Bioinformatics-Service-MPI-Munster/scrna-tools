import os
import scanpy as sc
import scanpy.external as sce
from collections import Counter, defaultdict
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy
from patsy import dmatrix
import pandas as pd
import pyarrow
import pyarrow.parquet as pq
import json
from .helpers import markers_to_df, mkdir
import requests
import time
import warnings
from functools import wraps
import logging

try:
    import gseapy
    with_gseapy = True
except (ModuleNotFoundError, OSError):
    with_gseapy = False


def requires_gseapy(func):
    """Checks if gseapy is installed in path"""
    
    @wraps(func)
    def wrapper_gseapy(*args, **kwargs):
        if not with_gseapy:
            raise RuntimeError("gseapy is not installed, cannot "
                               "run code that depends on it!")
        return func(*args, **kwargs)
    return wrapper_gseapy


logger = logging.getLogger('__name__')

try:
    import harmony
    has_harmony = True
except ImportError:
    has_harmony = False
    
try:
    import scvi
    has_scvi = True
except ImportError:
    has_scvi = False
    
try:
    import palantir
    has_palantir = True
except ImportError:
    has_palantir = False

try:
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats
    has_pydeseq2 = True
except ImportError:
    has_pydeseq2 = False


def recluster(
    adata, 
    resolutions=(
        0.05, 0.1, 0.2, 0.3, 0.4,
        0.5, 0.6, 0.7, 0.8, 0.9, 
        1.0, 1.2, 1.4, 1.6, 1.8, 
        2.0, 2.5
    ),
    key_prefix='leiden', 
    **kwargs
):
    for resolution in resolutions:
        logger.info("Resolution: {}".format(resolution))
        sc.tl.leiden(adata, resolution=resolution, key_added='{}_{}'.format(key_prefix, resolution), **kwargs)

    return adata


def batch_correct(
    adata,
    batch_algorithm='harmony',
    batch_key='sample',
    integrated_key='X_integrated',
    use_batch_for_hvg=True,
    cc_regression=False,
    cc_fields=['G2M_score', 'S_score'],
    **kwargs,
):
    if batch_algorithm == 'harmony':
        logger.info("Starting Harmony")
        sce.pp.harmony_integrate(adata, key=batch_key, max_iter_harmony=50, adjusted_basis=integrated_key)
    elif batch_algorithm == 'scvi':
        batch_ix_converter = {b: i for i, b in enumerate(adata.obs[batch_key].dtype.categories)}
        adata.obs['_scvi_tmp_batch'] = pd.Categorical([batch_ix_converter[b] for b in adata.obs[batch_key]])

        scvi.model.SCVI.setup_anndata(
            adata,
            layer="counts",
            batch_key='_scvi_tmp_batch' if use_batch_for_hvg else None,
            continuous_covariate_keys=None if not cc_regression else cc_fields,
        )
        logger.info("Setting up scVI model")
        model = scvi.model.SCVI(
            adata, 
            dispersion=kwargs.get("dispersion", "gene-batch"), 
            n_layers=kwargs.get("n_layers", 2), 
            n_latent=kwargs.get("n_latent", 30), 
            gene_likelihood=kwargs.get("gene_likelihood", "nb"),
        )
        logger.info("Training scVI model")
        model.train(
            max_epochs=kwargs.get('max_epochs', None)
        )

        del adata.obs['_scvi_tmp_batch']
        
        logger.info("Getting scVI latent representation")
        adata.obsm[integrated_key] = model.get_latent_representation()
    
    return adata


def relayout(
    adata, 
    n_pcs=100, 
    batch_algorithm='harmony', 
    batch_key='sample', 
    n_top_genes=3000,
    n_neighbors=30,
    umap=True, 
    tsne=True, 
    fdl=False, 
    diffmap=False,
    use_batch_for_hvg=True,
    hvg=True, 
    scale=False, 
    seurat_hvg=None, 
    cc_regression=False,
    cc_fields=['G2M_score', 'S_score'], 
    key_prefix=None,
    **kwargs
):
    if batch_algorithm == 'scvi' and not has_scvi:
        raise ImportError("scvi-tools are not installed, cannot use scvi batch correction!")
    
    if seurat_hvg is None and batch_algorithm == 'scvi':
        seurat_hvg = True
    else:
        seurat_hvg = False
        
    adata_original = adata
    adata = adata.copy()
    
    if cc_regression and batch_algorithm != 'scvi':
        logger.info("Regressing out cell cycle effects")
        sc.pp.regress_out(adata, cc_fields)
    
    if hvg:
        if not seurat_hvg:
            sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, batch_key=batch_key if use_batch_for_hvg else None,
                                        subset=True)
        else:
            sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, batch_key=batch_key if use_batch_for_hvg else None,
                                        flavor='seurat_v3', layer='counts', min_mean=0.1, max_mean=8, min_disp=1,
                                        subset=True)

    pca_key = 'X_pca' if key_prefix is None else f'X_{key_prefix}_pca'
    sc.tl.pca(adata, svd_solver='arpack', n_comps=n_pcs, random_state=42)
    adata_original.obsm[pca_key] = adata.obsm['X_pca']

    if batch_algorithm is not None:
        integrated_key = 'X_integrated' if key_prefix is None else f'X_{key_prefix}_integrated'
        
        batch_correct(
            adata,
            batch_algorithm=batch_algorithm,
            batch_key=batch_key,
            use_batch_for_hvg=use_batch_for_hvg,
            integrated_key=integrated_key,
            cc_regression=cc_regression,
            cc_fields=cc_fields,
            **kwargs
        )
        if batch_algorithm == 'scvi':
            n_pcs = None
        
        adata_original.obsm[integrated_key] = adata.obsm[integrated_key]
    else:
        integrated_key = 'X_pca'

    logger.info("Calculating neighbors")
    # Gotta fix a scanpy quirk in the Neighbors class init:
    if 'X_diffmap' in adata.obsm:
        del adata.obsm['X_diffmap']
    sc.pp.neighbors(adata, n_pcs=n_pcs, use_rep=integrated_key, knn=True, n_neighbors=n_neighbors)
    neighbors_key = 'neighbors' if key_prefix is None else f'{key_prefix}_neighbors'
    connectivities_key = 'connectivities' if key_prefix is None else f'{key_prefix}_connectivities'
    distances_key = 'distances' if key_prefix is None else f'{key_prefix}_distances'
    adata_original.obsp[distances_key] = adata.obsp['distances']
    adata_original.obsp[connectivities_key] = adata.obsp['connectivities']
    
    # neighbors uns
    neighbors_uns = adata.uns['neighbors'].copy()
    neighbors_uns['connectivities_key'] = connectivities_key
    neighbors_uns['distances_key'] = distances_key
    adata_original.uns[neighbors_key] = neighbors_uns
    
    if scale:
        logger.info("Scaling data")
        sc.pp.scale(adata_original)

    if umap:
        logger.info("Calculating UMAP")
        sc.tl.umap(adata, min_dist=0.3, random_state=42)
        adata_original.obsm['X_umap' if key_prefix is None else f'X_{key_prefix}_umap'] = adata.obsm['X_umap']

    if tsne:
        logger.info("Calculating tSNE")
        sc.tl.tsne(adata, use_rep=integrated_key)
        adata_original.obsm['X_tsne' if key_prefix is None else f'X_{key_prefix}_tsne'] = adata.obsm['X_tsne']

    if diffmap:
        logger.info("Calculating DiffMap")
        sc.tl.diffmap(adata)
        adata_original.obsm['X_diffmap' if key_prefix is None else f'X_{key_prefix}_diffmap'] = adata.obsm['X_diffmap']
        adata_original.uns['diffmap_evals' if key_prefix is None else f'{key_prefix}_diffmap_evals'] = adata.uns['diffmap_evals']

    if fdl:
        if has_harmony and has_palantir:
            logger.info("Running FDL")
            pca_projections = pd.DataFrame(adata.obsm[pca_key], index=adata.obs_names)
            dm_res = palantir.utils.run_diffusion_maps(pca_projections, n_components=5)
            fdl = harmony.plot.force_directed_layout(dm_res['kernel'], adata.obs_names)
            adata_original.obsm['X_fdl' if key_prefix is None else f'X_{key_prefix}_fdl'] = fdl.to_numpy()
        else:
            logger.warning("To use FDL layout you have to install palantir and harmony. Skipping FDL.")

    return adata_original



def recalculate_markers(
    adata, 
    output_file=None, 
    sort_by_abs_score=True, 
    include_groups=None, 
    **kwargs
):
    kwargs.setdefault('key_added', 'rank_genes_groups')
    kwargs.setdefault('method', 'wilcoxon')
    kwargs.setdefault('groupby', 'annotated')
    kwargs.setdefault('pts', True)
    sc.tl.rank_genes_groups(adata, kwargs.pop("groupby"), **kwargs)

    df = markers_to_df(adata, kwargs['key_added'],
                       output_file=output_file,
                       sort_by_abs_score=sort_by_abs_score,
                       include_groups=include_groups)

    return df


def rename_groups(adata, obs_key, rename_dict, key_added=None):
    if key_added is None:
        key_added = obs_key

    obs = np.array(adata.obs[obs_key].copy())
    obs_keys = np.unique(obs)
    
    ixs_to_name = []
    for key, value in rename_dict.items():
        if not str(key) in obs:
            raise ValueError("Key '{}' not in obs! ({})".format(key, ", ".join(obs_keys)))

        ixs = adata.obs[obs_key] == str(key)
        ixs_to_name.append((ixs, str(value)))
    
    for ixs, value in ixs_to_name:
        obs[ixs] = value

    adata.obs[key_added] = obs
    adata.obs[key_added] = adata.obs[key_added].astype('category')
    return adata


def count_groups(
    adata, 
    key=None, 
    splitby=None, 
    split_order=None, 
    group_order=None,
    long=False, 
    relative=False, 
    ignore_nan=True, 
    nan_groups=('NA', 'NaN', 'nan'),
    **kwargs
):
    group_data = adata.obs[key] if key is not None else pd.Series(['all'] * adata.shape[0], index=adata.obs.index, dtype='category')
    split_data = adata.obs[splitby] if splitby is not None else pd.Series(['count'] * adata.shape[0], index=adata.obs.index, dtype='category')
    
    groups = []
    if group_order is None:
        group_order = group_data.unique()
        if group_data.dtype.name == 'category':
            group_order = [g for g in group_data.dtype.categories if g in group_order]
    if split_order is None:
        split_order = split_data.unique()
    for split in split_order:
        groups.append(list(group_data[split_data == split]))
    
    if ignore_nan:
        split_order = [g for g in split_order if str(g) not in nan_groups]
        group_order = [g for g in group_order if str(g) not in nan_groups]

    counts = []
    for group in groups:
        c = Counter(group)
        counts.append([c[g] for g in group_order])
        
    data = []
    for group_name, group in zip(split_order, groups):
        c = Counter(group)
        
        split_count = sum(c[g] for g in group_order)
        for g in group_order:
            count = c[g]
            if relative:
                count /= split_count
            data.append([group_name, g, count])
    
    count_column_name = 'count' if not relative else 'fraction'
    df = pd.DataFrame(data, columns=['split', 'group', count_column_name])

    if not long:
        df = df.pivot(index='split', columns='group', values=count_column_name)
        df = df[group_order]
        df = df.reindex(split_order)

    return df


def cell_cycle_regression(cadata, score_fields=['G2M_score', 'S_score'], suffix='cc_regression', 
                          save_umap=True, save_pca=True, save_counts=False, save_neighbors=True,
                          do_relayout=True, sparse=True,
                          **kwargs):
    if cadata.has_constraints() and (save_umap or save_pca or save_counts or save_neighbors):
        raise ValueError("Cannot save cell cycle regression output to Cadata if it has constraints!")
    
    adata_no_cc = cadata.adata_subset.copy()
    sc.pp.regress_out(adata_no_cc, score_fields)
    if sparse:
        adata_no_cc.X = scipy.sparse.csr_matrix(adata_no_cc.X)

    if do_relayout:
        relayout(adata_no_cc, **kwargs)
        if save_pca:
            cadata.adata.obsm[f'X_pca_{suffix}'] = adata_no_cc.obsm['X_pca']
        if save_umap:
            cadata.adata.obsm[f'X_umap_{suffix}'] = adata_no_cc.obsm['X_umap']
        if save_neighbors:
            cadata.adata.uns[f'neighbors_{suffix}'] = adata_no_cc.uns['neighbors']
            
    if save_counts:
        cadata.adata.layers[f'{suffix}'] = adata_no_cc.X.copy()
    
    return adata_no_cc


def gene_stats(
    vdata, 
    genes=None, 
    layer=None,
    groupby=None, 
    groups=None,
    include_rest=False, 
    ignore_groups=('NA', 'NaN'),
    mean_expression=True, 
    percent_expressed=True, 
    sd_expression=False
):
    vdata = vdata.copy(only_constraints=True)

    if genes is not None:
        vdata.add_index_var_constraint(genes)
        
    group_ixs = {}
    if groupby is not None:
        if groups is None:
            groups = vdata.obs[groupby].dtype.categories
        
        groups = [g for g in groups if g not in ignore_groups]
                
        for group in groups:
            group_ixs[group] = vdata.obs[groupby] ==  group
        
        if include_rest:
            invalid_ixs = (~vdata.obs[groupby].isin(groups))
            group_ixs['rest'] = invalid_ixs
    else:
        group_ixs['all'] = np.array([True] * vdata.obs.shape[0])
    
    if layer is None:
        x = vdata.X
    else:
        x = vdata.layers[layer]
    x = x.toarray()
        
    stats = pd.DataFrame(index=vdata.var.index)
    for group, ixs in group_ixs.items():
        group_clean = group.replace(' ', '_').replace('.', '_')
        x_sub = x[ixs, :]
        
        if percent_expressed:
            is_expressed = np.array(x_sub) > 0
            stats[f'percent_expressed_{group_clean}'] = np.sum(is_expressed, axis=0) / is_expressed.shape[0] * 100
        if mean_expression:
            stats[f'mean_expression_{group_clean}'] = np.array(x_sub.mean(axis=0))
        if sd_expression:
            stats[f'sd_expression_{group_clean}'] = np.array(x_sub.sd(axis=0))
    
    return stats



def fraction_expressed(x):
    return np.sum(x > 0) / len(x)


def _scale(data_df, axis=0, center=True):
    if center:
        df = data_df.subtract(data_df.mean(axis=axis), axis=1 if axis == 0 else 0)
    else:
        df = data_df
    return df.div(np.nanstd(data_df, axis=axis), axis=1 if axis == 0 else 0)


def _reorder_df(df, obs_keys, obs_order=None):
    if isinstance(obs_keys, (str, bytes, int)):
        obs_keys = [obs_keys]
    
    if obs_order is None:
        obs_order = {}
    
    if not isinstance(obs_order, dict):
        obs_order = {
            obs_keys[0]: obs_order
        }
    
    for obs_key in obs_keys:
        if obs_key not in obs_order:
            obs_order[obs_key] = df[obs_key].dtype.categories
        
    df = df.copy()
    order_columns = []
    for obs_key in obs_keys:
        # remove unused obs categories
        df = df.loc[df[obs_key].isin(obs_order[obs_key]), :]
        # convert to numeric for reorder
        custom_order = {c: i for i, c in enumerate(obs_order[obs_key])}
        df[f'_order_{obs_key}'] = df[obs_key].map(custom_order).to_numpy()
        order_columns.append(f'_order_{obs_key}')
    sort_df = df.sort_values(by=order_columns, kind="mergesort")
    return sort_df.drop(order_columns, axis=1)


def expression_data_frame(
    vdata, 
    obs_keys=None, 
    var_keys=None,
    obs_order=None,
    long=False,
    var_name='gene',
    aggregate=False, 
    aggregate_statistic={
        'mean': np.mean,
        'frac': fraction_expressed,
    },
    scale=False, center=True,
    layer=None
):
    if isinstance(obs_keys, (str, bytes, int)):
        obs_keys = [obs_keys]
    if isinstance(var_keys, (str, bytes, int)):
        var_keys = [var_keys]
    single_stat = False
    if aggregate and not isinstance(aggregate_statistic, dict):
        single_stat = True
        aggregate_statistic = {
            'agg': aggregate_statistic
        }
    
    vdata_sub = vdata.copy(only_constraints=True)
    if var_keys is not None:
        vdata_sub.add_index_var_constraint(var_keys)
    
    if layer is None:
        data_matrix = vdata_sub.X
    else:
        data_matrix = vdata_sub.layers[layer]
    
    data_df = pd.DataFrame(data_matrix.toarray(), columns=vdata_sub.var.index)
    data_df = data_df.loc[:, var_keys]
    
    if aggregate:
        if len(obs_keys) != 1:
            raise ValueError(f"Can only aggregate over a single obs_key, you provided {len(obs_keys)}!")
        
        stat_dfs = {}
        for stat_name, stat in aggregate_statistic.items():
            stat_df = data_df.groupby(by=vdata_sub.obs[obs_keys[0]].to_list()).agg(stat)
            if scale:
                stat_df = _scale(stat_df, center=center)
            stat_df[obs_keys[0]] = pd.Categorical(stat_df.index.to_list())
            stat_df = _reorder_df(stat_df, obs_keys[0], obs_order=obs_order)
            stat_dfs[stat_name] = stat_df
        
        if long:
            long_df = None
            for stat_name, stat_df in stat_dfs.items():
                if long_df is None:
                    df = stat_df.melt(id_vars=[obs_keys[0]], var_name=var_name, value_name=stat_name)
                else:
                    df[stat_name] = long_df[stat_name]
            return long_df
        
        if single_stat:
            return stat_dfs['agg']
        
        return stat_dfs
    else:
        if scale:
            data_df = _scale(data_df, center=center)
        
        if obs_keys is not None:
            for i, obs_key in enumerate(obs_keys or []):
                if vdata_sub.obs[obs_key].dtype.name != 'category':
                    raise ValueError(f"All obs_keys must be categorical, but {obs_key} "
                                    f"is not ({vdata_sub.obs[obs_key].dtype})")
                data_df.insert(i, obs_key, pd.Categorical(vdata_sub.obs[obs_key].to_list()))
            data_df = _reorder_df(data_df, obs_keys, obs_order=obs_order)

        if long:
            data_df.insert(0, 'index', vdata_sub.obs.index.to_list())
            return data_df.melt(id_vars=['index'] + obs_keys or [], var_name=var_name)
        
        return data_df


def enrichr(
    df, 
    gene_sets, 
    organism, 
    output_folder,
    padj_cutoff=0.01, 
    log2fc_cutoff=1.5,
    absolute_log2fc=False, 
    make_plots=True,
    enrichr_cutoff=1, 
    max_attempts=3,
    log2_fold_change_field='log2FoldChange',
    padj_field='padj',
    **kwargs
):
    df_sub = df.copy()
    if log2fc_cutoff is not None:
        if log2fc_cutoff >= 0:
            if absolute_log2fc:
                df_sub = df_sub[df_sub[log2_fold_change_field].abs() >= log2fc_cutoff]
            else:
                df_sub = df_sub[df_sub[log2_fold_change_field] >= log2fc_cutoff]
        else:
            df_sub = df_sub[df_sub[log2_fold_change_field] <= log2fc_cutoff]

    if padj_cutoff is not None:
        df_sub = df_sub[df_sub[padj_field] <= padj_cutoff]
    
    print(df)
    print(df_sub)
    
    glist = [g.upper() for g in df_sub.index.str.strip().tolist()]
    bg = [g.upper() for g in df.index.str.strip().tolist()]
    
    logger.info(f"Remaining entries in gene list: {len(glist)}/{len(bg)}")
    
    success = False
    attempt = 0
    while not success:
        attempt += 1
        try:
            enr = gseapy.enrichr(
                gene_list=glist,
                gene_sets=gene_sets,
                organism=organism,
                outdir=output_folder,
                no_plot=not make_plots,
                cutoff=enrichr_cutoff,
                background=bg,
                **kwargs
            )
            success = True
        except requests.exceptions.ConnectionError:
            if attempt > max_attempts:
                raise
            else:
                time.sleep(10)

    return enr


def enrichr_bulk(
    df, 
    output_folder, 
    organism, 
    padj_cutoff=0.01, 
    log2fc_cutoff=0.5,
    gene_sets=[
        'GO_Molecular_Function_2018', 'GO_Cellular_Component_2018', 
        'GO_Biological_Process_2018',
        'KEGG_2019_Mouse', 'WikiPathways_2019_Mouse', 
        'KEGG_2019_Human', 'WikiPathways_2019_Human'
    ],
    deregulated=True, 
    downregulated=True, 
    upregulated=True,
    **kwargs
):
        for gene_set in gene_sets:
            analyses = []
            if deregulated:
                analyses.append(('deregulated', padj_cutoff, log2fc_cutoff, True))
            if upregulated:
                analyses.append(('upregulated', padj_cutoff, log2fc_cutoff, False))
            if downregulated:
                analyses.append(('downregulated', padj_cutoff, -log2fc_cutoff, False))
            
            for name, padj_cutoff_batch, log2fc_cutoff_batch, absolute_log2fc in analyses:
                enrichr_folder = mkdir(output_folder, name, gene_set)
                
                log2fc_cutoff_string = str(log2fc_cutoff_batch)
                if absolute_log2fc:
                    log2fc_cutoff_string = f'+-{log2fc_cutoff_batch}'
                    
                with open(os.path.join(enrichr_folder, 'enrichr_settings.txt'), 'w') as o:
                    o.write(f"padj cutoff: {padj_cutoff_batch}\nlog2fc cutoff: {log2fc_cutoff_string}")
                
                try:
                    kwargs.setdefault('top_term', 20)
                    
                    enrichr(
                        df, 
                        gene_set, 
                        organism, 
                        enrichr_folder,
                        padj_cutoff=padj_cutoff_batch, 
                        log2fc_cutoff=log2fc_cutoff_batch,
                        absolute_log2fc=absolute_log2fc,
                        **kwargs
                    )
                except Exception as e:
                    print(df)
                    print(name)
                    #raise
                    continue


@requires_gseapy
def gsea_from_df(
    df,
    gene_sets,
    pvalue_field='pvalue',
    log2_fold_change_field='log2FoldChange',
    gene_name_field='gene',
    plot_file=None,
    rank_by='pvalue',
    padj_cutoff=1,
    filter_nan_pvalue=True,
    **kwargs
):
    if rank_by in ['pvalue', 'pval']:
        signs = np.sign(df[log2_fold_change_field])
        signs[signs == 0] = 1.
        pvalues = df[pvalue_field]
        pvalues[pvalues == 0] = np.nextafter(0, 1)
        rank_metric = (-np.log10(pvalues))/signs
    elif rank_by in ['log2fc', 'log2_fold_change']:
        rank_metric = df[log2_fold_change_field]
    else:
        raise ValueError(f"Unknown rank_by parameter '{rank_by}'")
    
    # remove NaN p-values
    valid = np.repeat(True, len(rank_metric))
    if filter_nan_pvalue:
        nan_pvalues = np.isnan(df[pvalue_field])
        n_nan_pvalues = sum(nan_pvalues)
        if n_nan_pvalues > 0:
            logger.warning(f"Removing {n_nan_pvalues} that are NaN!")
            valid = ~nan_pvalues
    
    de_prerank = pd.DataFrame.from_dict({
        'gene': df[gene_name_field][valid],
        'metric': rank_metric[valid],
    })
    de_prerank = de_prerank.sort_values('metric', ascending=False)
    
    kwargs.setdefault('processes', 4)
    kwargs.setdefault('min_size', 5)
    kwargs.setdefault('max_size', 5000)
    kwargs.setdefault('permutation_num', 10000)
    kwargs.setdefault('no_plot', True)
    kwargs.setdefault('outdir', None)
    
    pre_res = gseapy.prerank(
        de_prerank, 
        gene_sets=gene_sets, 
        **kwargs
    )

    results_dict = defaultdict(list)
    for key, result in pre_res.results.items():
        results_dict['term'].append(key)
        results_dict['es'].append(result['es'])
        results_dict['nes'].append(result['nes'])
        results_dict['pval'].append(result['pval'])
        results_dict['fdr'].append(result['fdr'])
        results_dict['fwerp'].append(result['fwerp'])
        results_dict['size'].append(0)
        results_dict['matched_size'].append(len(result['matched_genes']))
        results_dict['genes'].append(result['matched_genes'])
        results_dict['leading_edge'].append(result['lead_genes'])
        
        try:
            results_dict['size'][-1] = len(gene_sets[key])
        except:
            results_dict['size'][-1] = len(result['matched_genes'])
    
    results = pd.DataFrame(results_dict)
    results = results.reindex(results.nes.abs().sort_values(ascending=False).index)
    
    if plot_file is not None:
        with PdfPages(plot_file) as pdf:
            for gene_set_name, result in pre_res.results.items():
                if result['fdr'] > padj_cutoff:
                    continue
                
                axes = gseapy.gseaplot(
                    term=gene_set_name, 
                    rank_metric=pre_res.ranking, 
                    **pre_res.results[gene_set_name]
                )
                fig = axes[0].figure
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
    
    return results


def pseudobulk_expression_matrix_and_annotations(
    vdata, 
    obs_key,
    sample1=None,
    sample2=None,
    layer='counts',
    replicate_key=None,
    n_pseudoreplicates=2,
    pseudocount=0,
    random_seed=42,
):
    vdata = vdata.copy(only_constraints=True)
    
    groupby = [obs_key] if replicate_key is None else [obs_key, replicate_key]
    if sample1 is None:
        obs = pd.DataFrame({'group': vdata.obs[obs_key]}, index=vdata.obs.index)
    elif sample2 is not None:
        vdata.add_categorical_obs_constraint(obs_key, [sample1, sample2])
        obs = pd.DataFrame({'group': vdata.obs[obs_key]}, index=vdata.obs.index)
    # or compare one sample vs all other cells (marker)
    else:
        obs = vdata.obs.loc[:, groupby]
        obs[obs_key] = pd.Categorical([x if x == sample1 else 'rest' for x in obs[obs_key]])
        obs = pd.DataFrame({
                'group': pd.Categorical([x if x == sample1 else 'rest' for x in vdata.obs[obs_key]])
            }, index=vdata.obs.index)
        sample2 = 'rest'
    obs_key = 'group'
    
    # generate pseudoreplicates, if no replicate column is provided
    obs[obs_key] = pd.Categorical([v for v in obs[obs_key]])
    if replicate_key is None:
        n_replicates = n_pseudoreplicates
        replicate_key = '__replicate__'
        
        rng = np.random.default_rng(random_seed)
        #np.random.seed(random_seed)
        obs[replicate_key] = pd.Categorical(
            rng.choice(
                [str(i) for i in range(1, n_pseudoreplicates + 1)],
                size=obs.shape[0]
            )
        )
    else:
        obs[replicate_key] = vdata.obs[replicate_key]
        n_replicates = len(obs[replicate_key].unique())
    
    # generate model and count matrices
    mm = dmatrix(f'~0 + {obs_key}:{replicate_key}', obs, return_type='dataframe')
    # ensure unsused replicate / sample combinations are removed
    mm = mm.loc[:, mm.sum(axis=0) > 0]
    
    # generate count matrix
    try:
        expr = vdata[mm.index].layers[layer].toarray()
    except TypeError:
        expr = vdata.adata_subset[mm.index].layers[layer].toarray()
    mat_mm = np.matmul(expr.T, mm)
    #print(type(mat_mm))
    mat_mm.index = vdata.var.index.copy()
    
    mat_mm_int = np.round(mat_mm)
    if not np.all((mat_mm - mat_mm_int) == 0):
        warnings.warn("Counts matrix contains floats! These will be explicitly "
                      "rounded to the nearest integer. Please ensure that this is "
                      "intentional!")
    mat_mm = mat_mm_int.add(pseudocount)

    # get sample annotations
    data = defaultdict(list)
    for name in mm.columns:
        for col_info in name.split(":"):
            groupby, value = col_info[:-1].split("[")
            data[groupby].append(value)
    sample_annotation = pd.DataFrame(data, index=mm.columns)
    
    info = {
        'obs_key': obs_key,
        'sample1': sample1,
        'sample2': sample2,
        'replicate_key': replicate_key,
        'n_replicates': n_replicates,
    }
    return vdata, mat_mm, sample_annotation, info


def de_df_to_parquet(
    df, 
    output_file=None,
    obs_key=None,
    view_key=None,
    categories=None, 
    n_replicates=None,
    replicate_key=None,
    obs_constraints=(),
    var_constraints=(),
    lfc_shrink=False,
    pseudocount=0,
    analysis_name='Pseudobulk DE or Markers',
):
    if output_file is None:
        pqdf = pyarrow.Table.from_pandas(df)
    else:
        df.to_parquet(output_file)
        pqdf = pq.read_table(output_file)
    
    meta = {
        'analysis': analysis_name,
        'obs_key': obs_key,
        'categories': json.dumps(categories or list(df['group'].unique())),
        'obs_constraints': json.dumps([c.to_dict() for c in obs_constraints]),
        'var_constraints': json.dumps([c.to_dict() for c in var_constraints]),
        'lfc_shrink': 'true' if lfc_shrink else 'false',
        'pseudocount': str(pseudocount),
    }
    if view_key is not None:
        meta['view_key'] = view_key
    if n_replicates is not None:
        meta['n_replicates'] = str(n_replicates)
    if replicate_key is not None:
        meta['replicate_key'] = replicate_key
    
    pqdf = pqdf.replace_schema_metadata({**pqdf.schema.metadata, **meta})
    if output_file is not None:
        pq.write_table(pqdf, output_file)
    
    return pqdf


def de(
    vdata,
    obs_key,
    sample1,
    sample2=None,
    replicate_key=None, 
    n_pseudoreplicates=2, 
    layer='counts',
    min_counts_per_gene=0, 
    min_counts_per_sample=0, 
    append_gene_stats=True, 
    stats_layer=None,
    threads=1,
    random_seed=42,
    as_parquet=False,
    view_key=None,
    lfc_shrink=False,
    pseudocount=0,
    _logger=logger,
):
    if not has_pydeseq2:
        raise ImportError("Must install pydeseq2 for pseudobulk DE (or use r.de_pseudobulk")
        
    original_key = obs_key
    
    include_rest = sample2 is None
    
    obs_constraints = vdata.obs_constraints.copy()
    var_constraints = vdata.var_constraints.copy()
    
    if view_key is not None:
        vdata = vdata.copy()
        vdata.add_view_constraint(view_key)
    
    # compare two samples (DE)    
    vdata, mat_mm, sample_annotation, pb_info = pseudobulk_expression_matrix_and_annotations(
        vdata, 
        obs_key=obs_key,
        sample1=sample1,
        sample2=sample2,
        replicate_key=replicate_key,
        n_pseudoreplicates=n_pseudoreplicates,
        layer=layer,
        random_seed=random_seed,
        pseudocount=pseudocount,
    )
    pb_replicate_key = pb_info['replicate_key']
    pb_obs_key = pb_info['obs_key']
    n_replicates = pb_info['n_replicates']
    sample1 = pb_info['sample1']
    sample2 = pb_info['sample2']
    
    # transpose
    mat_mm = mat_mm.T
    
    # filter genes by min counts
    mat_mm = mat_mm.loc[:, mat_mm.sum(axis=0) >= min_counts_per_gene]
    
    # filter samples by min counts
    mat_mm = mat_mm.loc[mat_mm.sum(axis=1) >= min_counts_per_sample, :]
    
    dds = DeseqDataSet(
        counts=mat_mm,
        metadata=sample_annotation,
        design_factors=pb_obs_key,
        refit_cooks=True,
        n_cpus=threads,
        ref_level=(pb_obs_key, sample2),
    )
    dds.deseq2()
    
    contrast = [pb_obs_key, sample1, sample2]
    if sample1 not in dds.obs[pb_obs_key].values and sample1.replace('_', '-') in dds.obs[pb_obs_key].values:
        contrast[1] = sample1.replace('_', '-')
    if sample2 not in dds.obs[pb_obs_key].values and sample2.replace('_', '-') in dds.obs[pb_obs_key].values:
        contrast[2] = sample2.replace('_', '-')
    
    stat_res = DeseqStats(
        dds,
        contrast=contrast,
        alpha=1,
        independent_filter=True,
        cooks_filter=True,
    )
    stat_res.summary()
    
    if lfc_shrink and not stat_res.shrunk_LFCs:
        _logger.info("Applying log-fold shrinkage")
        stat_res.lfc_shrink(coeff=None)
    
    results_df = stat_res.results_df
    
    if append_gene_stats:
        stats = vdata.gene_stats(
            genes=results_df.index, 
            groupby=original_key, 
            groups=[sample1] if sample2 is None else [sample1, sample2],
            include_rest=include_rest,
            sd_expression=False, 
            layer=stats_layer
        )
        for column in stats.columns:
            results_df[column] = stats.loc[results_df.index, column]
    
    results_df = results_df.sort_values(by='log2FoldChange', key=abs, ascending=False, kind='mergesort')
    results_df = results_df.sort_values(by='pvalue', kind='mergesort')
    
    if as_parquet:
        return de_df_to_parquet(
            results_df, 
            obs_key=obs_key,
            view_key=view_key,
            categories=[sample1, sample2],
            n_replicates=n_replicates,
            replicate_key=replicate_key,
            obs_constraints=obs_constraints,
            var_constraints=var_constraints,
            lfc_shrink=lfc_shrink,
            pseudocount=pseudocount,
        )
    
    return results_df


def markers(
    vdata, 
    key, 
    categories=None, 
    view_key=None,
    ignore_categories=['NA', 'nan', 'NaN'],
    as_parquet=False,
    de_func=de,
    _logger=logger,
    **kwargs
):
    _logger.info("Getting categories")
    obs_constraints = vdata.obs_constraints.copy()
    var_constraints = vdata.var_constraints.copy()
    
    vdata = vdata.copy(only_constraints=True)
    if view_key is not None:
        vdata.add_view_constraint(view_key)
    
    original_categories = vdata.obs[key].unique()
    categories_provided = categories is not None
    categories = categories or [g for g in original_categories if g not in ignore_categories]
    
    if Counter(original_categories) != Counter(categories):
        vdata.add_categorical_obs_constraint(key, categories)
        if not categories_provided:
            obs_constraints.append(vdata.obs_constraints[-1])
    
    all_markers = None
    for i, category in enumerate(categories):
        category_clean = category.replace(' ', '_').replace('.', '_')
        _logger.info(f"Calculating markers for {category_clean} ({i+1}/{len(categories)})")
        
        print(category, category_clean, categories)
        
        markers = de_func(vdata, key, category, **kwargs)
        markers['group'] = category
        print(markers.columns)
        markers = markers.rename({
            f'mean_expression_{category_clean}': 'mean_expression_group',
            f'percent_expressed_{category_clean}': 'percent_expressed_group',
        }, axis=1)
        print(markers.columns)

        if all_markers is None:
            all_markers = markers
        else:
            all_markers = pd.concat([all_markers, markers])
    
    if as_parquet:
        _logger.info("Converting to parquet")
        return de_df_to_parquet(
            all_markers, 
            obs_key=key,
            view_key=view_key,
            categories=categories,
            obs_constraints=obs_constraints,
            var_constraints=var_constraints,
        )
    
    return all_markers


def paga(
    vdata,
    paga_key,
    obs_key,
    obs_categories=None,
    neighbors_key=None,
    obsm_key=None,
    do_relayout=False, 
    paga_name=None,
    paga_description=None,
    save_to_adata=True,
    restore_view=None,
    force=False,
    relayout_kwargs={}
):
    if not force and 'trajectory' in vdata.uns_keys() and paga_key in vdata.uns['trajectory']:
        raise ValueError(f"PAGA key {paga_key} already exists in adata. Use 'force=True' to overwrite.")
    
    if restore_view:
        adata_sub = vdata.restore_view(restore_view)
    else:
        adata_sub = vdata.adata_view
    
    if obs_categories is not None:
        from ._core._vdata import VData
        vdata_sub = VData(adata_sub)
        vdata_sub.add_categorical_obs_constraint(obs_key, obs_categories)
        adata_sub = vdata_sub.adata_view
    
    if do_relayout:
        relayout_kwargs.setdefault('umap', False)
        relayout_kwargs.setdefault('tsne', False)
        relayout_kwargs.setdefault('fdl', False)
        relayout_kwargs.setdefault('batch_algorithm', None)
        
        relayout(
            adata_sub, 
            **relayout_kwargs,
        )
    
    sc.tl.paga(
        adata_sub, 
        obs_key, 
        neighbors_key=neighbors_key
    )

    if save_to_adata:
        adata = vdata._parent_adata
        
        view_obs_key = f'__view__{restore_view}__{obs_key}' if restore_view is not None else obs_key
        if view_obs_key not in adata.obs.columns:
            view_obs_key = obs_key
        
        view_obsm_key = f'__view__{restore_view}__{obsm_key}' if restore_view is not None else obsm_key
        if view_obsm_key not in adata.obsm.keys():
            view_obsm_key = obsm_key
            
        paga_results = {
            'type': 'PAGA',
            'key': paga_key,
            'obs_key': view_obs_key,
            'categories': list(adata_sub.obs[obs_key].dtype.categories),
            'obs_constraints': json.dumps([c.to_dict() for c in vdata.obs_constraints]),
            'var_constraints': json.dumps([c.to_dict() for c in vdata.var_constraints]),
            'connectivities': adata_sub.uns['paga']['connectivities'].copy(),
            'connectivities_tree': adata_sub.uns['paga']['connectivities_tree'].copy(),
            'name': paga_name or paga_key,
            'description': paga_description or paga_key,
        }
        if neighbors_key is not None:
            paga_results['neighbors_key'] = neighbors_key
        if obsm_key is not None:
            paga_results['obsm_key'] = view_obsm_key
        
        
        if 'trajectory' not in adata.uns_keys():
            adata.uns['trajectory'] = {}
        adata.uns['trajectory'][paga_key] = paga_results
        
        return vdata
    else:
        return adata_sub
