import sys
from .helpers import mkdir
from .base import ConstrainedAdata
from ._core import VData
from .plotting import scenic_regulon_enrichment_heatmap
import pandas as pd
import numpy as np
import loompy as lp
from scipy.sparse import lil_matrix


def _add_dfs(df1, df2):
    merge = df1.copy()
    df1_columns = set(df1.columns)
    df2_columns = set(df2.columns)

    additional_columns = []
    for column in df2_columns:
        if column not in df1_columns:
            #merge[column] = df2[column]
            additional_columns.append(df2[column])
        else:
            merge[column] += df2[column]

    if len(additional_columns) > 0:
        merge = pd.concat([merge] + additional_columns, axis=1)
    
    return merge

def _add_sq_dfs(df1, df2):
    merge = df1.copy()
    df1_columns = set(df1.columns)
    df2_columns = set(df2.columns)
    
    additional_columns = []
    for column in df2_columns:
        if column not in df1_columns:
            #merge[column] = df2[column]
            additional_columns.append(df2[column])
        else:
            merge[column] += df2[column] * df2[column]
    
    if len(additional_columns) > 0:
        merge = pd.concat([merge] + additional_columns, axis=1)
    
    return merge


def scenic_from_loom(loom_files, adata, key_prefix='scenic', view_key=None):
    from pyscenic.binarization import binarize
    old_recursion_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(10000)
    
    if isinstance(loom_files, (str, bytes)):
        loom_files = [loom_files]
    
    consensus_regulons = dict()
    genes = None
    regulon_names = None
    auc_sum = None
    auc_sum_sq = None
    bin_mtx_sum = None
    all_thresholds = []
    
    for i, loom_file in enumerate(loom_files):
        print(f'{i+1}/{len(loom_files)}')
        
        with lp.connect(loom_file, mode='r+', validate=False) as lf:
            # sum and sd of AUC scores
            auc_mtx = pd.DataFrame(lf.ca.RegulonsAUC, index=lf.ca.CellID)
            if auc_sum is None:
                auc_sum = auc_mtx.copy()
                auc_sum_sq = auc_mtx.copy()
            else:
                assert np.equal(auc_sum.index, auc_mtx.index).all()
                auc_sum = _add_dfs(auc_sum, auc_mtx)
                auc_sum_sq = _add_sq_dfs(auc_sum, auc_mtx)
            
            bin_mtx, thresholds = binarize(auc_mtx, num_workers=4)
            all_thresholds.append(thresholds.to_dict())
            if bin_mtx_sum is None:
                bin_mtx_sum = bin_mtx.copy()
            else:
                bin_mtx_sum = _add_dfs(bin_mtx_sum, bin_mtx)
            
            # ensure genes are identical and in identical order
            if genes is None:
                genes = lf.ra.Gene
            else:
                assert np.equal(genes, lf.ra.Gene).all()
            
            # regulon membership
            regulons = lf.ra.Regulons
            regulon_names = auc_sum.columns.to_numpy()
            current_regulon_names = set(regulons.dtype.names)
            for regulon in regulon_names:
                if regulon not in consensus_regulons:
                    consensus_regulons[regulon] = regulons[regulon]
                elif regulon in current_regulon_names:
                    consensus_regulons[regulon] += regulons[regulon]
    
    auc_mean = auc_sum / len(loom_files)
    auc_sd = np.sqrt(auc_sum_sq / (-(auc_mean*auc_mean) + len(loom_files) ))
    regulon_data = []
    for regulon in auc_mean.columns.to_numpy():
        regulon_data.append(consensus_regulons[regulon])
    
    adata_sub = VData(adata).add_index_obs_constraint(auc_mean.index).adata_view

    gene_ixs = {gene: ix for ix, gene in enumerate(adata_sub.var.index.to_list())}
    regulon_genes = [r[:-3] for r in auc_mean.columns]
    
    # ensure correct cell order
    cell_ixs = {cell: ix for ix, cell in enumerate(adata_sub.obs.index.to_list())}
    o = [cell_ixs[cell] for cell in auc_mean.index]
    auc_mean = auc_mean.iloc[o]
    auc_sd = auc_sd.iloc[o]
    
    auc_mean_pos_layer = lil_matrix(adata_sub.shape, dtype=np.float32)
    auc_mean_neg_layer = lil_matrix(adata_sub.shape, dtype=np.float32)
    auc_sd_pos_layer = lil_matrix(adata_sub.shape, dtype=np.float32)
    auc_sd_neg_layer = lil_matrix(adata_sub.shape, dtype=np.float32)
    auc_sum_pos_layer = lil_matrix(adata_sub.shape, dtype=np.float32)
    auc_sum_neg_layer = lil_matrix(adata_sub.shape, dtype=np.float32)
    target_gene_support_pos = lil_matrix((adata_sub.shape[1], adata_sub.shape[1]), dtype=np.int16)
    target_gene_support_neg = lil_matrix((adata_sub.shape[1], adata_sub.shape[1]), dtype=np.int16)
    for column in auc_mean.columns:
        regulon = column[:-3]
        gene_ix = gene_ixs[regulon]
        if column[-2] == '+':
            auc_mean_pos_layer[:, gene_ix] = auc_mean[column]
            auc_sd_pos_layer[:, gene_ix] = auc_sd[column]
            auc_sum_pos_layer[:, gene_ix] = bin_mtx_sum[column]
            target_gene_support_pos[:, gene_ix] = consensus_regulons[column]
        elif column[-2] == '-':
            auc_mean_neg_layer[:, gene_ix] = auc_mean[column]
            auc_sd_neg_layer[:, gene_ix] = auc_sd[column]
            auc_sum_neg_layer[:, gene_ix] = bin_mtx_sum[column]
            target_gene_support_neg[:, gene_ix] = consensus_regulons[column]
    
    adata_sub.var[f'{key_prefix}_is_regulon'] = np.array([True if g in regulon_genes else False for g in adata_sub.var.index], dtype=np.bool_)
    adata_sub.layers[f'{key_prefix}_aucell_positive_mean'] = auc_mean_pos_layer.tocsr()
    adata_sub.layers[f'{key_prefix}_aucell_positive_sd'] = auc_sd_pos_layer.tocsr()
    adata_sub.layers[f'{key_prefix}_aucell_positive_sum'] = auc_sum_pos_layer.tocsr()
    adata_sub.varm[f'{key_prefix}_target_gene_support_positive'] = target_gene_support_pos.tocsr()
    
    has_negative = False
    neg_sum = auc_sum_neg_layer.tocsr()
    if neg_sum.count_nonzero() > 0:
        has_negative = True
        adata_sub.layers[f'{key_prefix}_aucell_negative_sum'] = neg_sum
        adata_sub.layers[f'{key_prefix}_aucell_negative_mean'] = auc_mean_neg_layer.tocsr()
        adata_sub.layers[f'{key_prefix}_aucell_negative_sd'] = auc_sd_neg_layer.tocsr()
        adata_sub.varm[f'{key_prefix}_target_gene_support_negative'] = target_gene_support_neg.tocsr()
    
    vdata = VData(adata)
    if view_key is None:
        scenic_tmp_prefix = 'scenic_tmp'
        view_prefix = f'__view__{scenic_tmp_prefix}__'
        
        vdata.add_view_var(adata_sub, scenic_tmp_prefix)
        vdata.add_view_varm(adata_sub, scenic_tmp_prefix)
        vdata.add_view_layers(adata_sub, scenic_tmp_prefix)
        for key in list(adata.layers.keys()):
            if key.startswith(view_prefix):
                adata.layers[key[len(view_prefix):]] = adata.layers[key]
                del adata.layers[key]
                
        for key in list(adata.varm.keys()):
            if key.startswith(view_prefix):
                adata.varm[key[len(view_prefix):]] = adata.varm[key]
                del adata.varm[key]
        
        adata.var = adata.var.rename(
            columns={
                f'{view_prefix}{key_prefix}_is_regulon': f'{key_prefix}_is_regulon'
            }
        )
    else:
        vdata.add_view_var(adata_sub, view_key)
        vdata.add_view_uns(adata_sub, view_key)
        vdata.add_view_varm(adata_sub, view_key)
        vdata.add_view_layers(adata_sub, view_key)
    
    scenic_info = adata.uns['scenic'] if 'scenic' in adata.uns_keys() else {}
    scenic_info[key_prefix] = {
        'n_runs': len(loom_files),
        'has_negative': has_negative,
    }
    adata.uns['scenic'] = scenic_info
    
    sys.setrecursionlimit(old_recursion_limit)
    
    return adata


def scenic_regulon_enrichment_scores_old(adata, groupby, 
                                         aucell_key='X_aucell', regulons_key='regulons',
                                         sort=True):
    if not isinstance(adata, (ConstrainedAdata, VData)):
        vdata = VData(adata)
    else:
        vdata = adata
    
    auc_mtx = pd.DataFrame(data=vdata.obsm[aucell_key], index=vdata.obs.index,
                           columns=vdata.uns[regulons_key])
    
    return _scenic_regulon_enrichment_scores_from_df(vdata, auc_mtx, groupby, sort=sort)


def scenic_regulon_enrichment_scores(
    vdata, 
    groupby, 
    key_prefix, 
    layer='aucell_positive_mean',
    sort=True
):
    if not layer.startswith(key_prefix):
        layer = f'{key_prefix}_{layer}'
    
    vdata_sub = vdata.copy(only_constraints=True)
    vdata_sub.add_index_var_constraint(
        [
            v for v, is_regulon in zip(vdata.var.index, vdata.var[f'{key_prefix}_is_regulon'])
            if is_regulon
        ]
    )
    
    df_scores = pd.DataFrame(
        data=vdata_sub.layers[layer].toarray(),
        index=vdata_sub.obs.index,
        columns=vdata_sub.var.index
    )
    
    return _scenic_regulon_enrichment_scores_from_df(vdata, df_scores, groupby, sort=sort)


def _scenic_regulon_enrichment_scores_from_df(vdata, df_scores, groupby, sort=True):
    mean_scores = df_scores.mean()
    std_scores = df_scores.std()
    if isinstance(groupby, (list, tuple)):
        df_scores['group'] = ['--'.join(t) for t in zip(*[vdata.obs[c] for c in groupby])]
    else:
        df_scores['group'] = vdata.obs[groupby]

    df_results = ((df_scores.groupby(by='group').mean() - mean_scores) / std_scores).stack()\
        .reset_index().rename(columns={'level_1': 'regulon', 0: 'Z'})
    
    if sort:
        df_results = df_results.sort_values(['group', 'Z'], ascending=[True, False])
    
    return df_results


def scenic_regulon_enrichment_old(adata, groupby, group, groups=None, 
                                  n_regulons=None, zscore_cutoff=None,
                                  regulon_names=None,
                                  aucell_key='X_aucell', regulons_key='regulons',
                                  plot=False, **plot_kwargs):
    if not isinstance(adata, (ConstrainedAdata, VData)):
        vdata = VData(adata)
    else:
        vdata = adata
    
    df_scores = scenic_regulon_enrichment_scores_old(vdata, groupby, aucell_key=aucell_key,
                                                     regulons_key=regulons_key)
    
    return _scenic_regulon_enrichment_from_df(df_scores, group,
                                              groups=groups,
                                              n_regulons=n_regulons,
                                              zscore_cutoff=zscore_cutoff,
                                              regulon_names=regulon_names,
                                              plot=plot, plot_kwargs=plot_kwargs)
    

def scenic_regulon_enrichment(vdata, groupby, group, groups=None, 
                              n_regulons=None, zscore_cutoff=None,
                              regulon_names=None, ascending=False,
                              key_prefix='scenic', layer='aucell_positive_mean',
                              plot=False, **plot_kwargs):
    df_scores = scenic_regulon_enrichment_scores(
        vdata, 
        groupby, 
        key_prefix=key_prefix, 
        layer=layer
    )
    
    return _scenic_regulon_enrichment_from_df(
        df_scores, 
        group, 
        groups=groups,
        n_regulons=n_regulons,
        zscore_cutoff=zscore_cutoff,
        regulon_names=regulon_names,
        ascending=ascending,
        plot=plot, 
        **plot_kwargs
    )


def _scenic_regulon_enrichment_from_df(
    df_results, 
    group,
    groups=None,
    n_regulons=None,
    zscore_cutoff=None,
    regulon_names=None,
    ascending=False,
    plot=False, 
    **plot_kwargs
):
    if groups is None:
        groups = np.unique(df_results['group'].to_list())
    
    if ascending is None:
        df_group = df_results[df_results['group'] == group].sort_values('Z', key=abs)
    else:
        df_group = df_results[df_results['group'] == group].sort_values('Z', ascending=ascending)
    
    df_sig = df_group.copy()
    if zscore_cutoff is not None:
        if ascending is None:
            df_sig = df_sig[df_sig['Z'].abs() > abs(zscore_cutoff)]
        elif ascending:
            df_sig = df_sig[df_sig['Z'] > zscore_cutoff]
        else:
            df_sig = df_sig[df_sig['Z'] <= zscore_cutoff]
    
    if n_regulons is not None:
        df_sig = df_sig.head(n=n_regulons)
    
    if regulon_names is not None:
        df_sig = df_sig[df_sig['regulon'].isin(regulon_names)]
    
    enriched_regulons = set(df_sig['regulon'])
    df_results_sub = df_results[np.logical_and(df_results['regulon'].isin(enriched_regulons),
                                               df_results['group'].isin(groups))]

    if plot:
        ax = scenic_regulon_enrichment_heatmap(df_results_sub, **plot_kwargs)
        return df_results_sub, ax
    
    return df_results_sub


def scenic_target_gene_confidence(
    vdata, 
    key_prefix='scenic', 
    varm_key='target_gene_support_positive',
    view_key=None
):
    if not varm_key.startswith(key_prefix):
        varm_key = f'{key_prefix}_{varm_key}'
    
    regulon_data = []
    regulons = [
        v for v, is_regulon in zip(
            vdata.var(view_key=view_key).index, 
            vdata.var(view_key=view_key)[f'{key_prefix}_is_regulon']
        )
        if is_regulon
    ]
    target_gene_confidence = vdata.varm(view_key=view_key)[varm_key]
    
    n_runs = vdata.uns(view_key=view_key)['scenic'][key_prefix]['n_runs']
    
    for regulon_name in regulons:
        regulon_tg_support = pd.Series(
            target_gene_confidence[
                :, 
                vdata.var(view_key=view_key).index.get_loc(regulon_name)].toarray().ravel(),
            index=vdata.var(view_key=view_key).index
        )
        for target_gene, r in regulon_tg_support[regulon_tg_support > 0].items():
            regulon_data.append([regulon_name, target_gene, r/n_runs, r, n_runs])
    target_gene_df = pd.DataFrame(regulon_data, columns=['regulon', 'target_gene', 'score', 'count', 'total'])
    
    return target_gene_df

