from .helpers import mkdir
from .base import ConstrainedAdata
from .plotting import scenic_regulon_enrichment_heatmap
import pandas as pd
import numpy as np


def scenic_regulon_enrichment_scores(adata, groupby, 
                                     aucell_key='X_aucell', regulons_key='regulons',
                                     sort=True):
    if not isinstance(adata, ConstrainedAdata):
        cadata = ConstrainedAdata(adata)
    else:
        cadata = adata
    
    auc_mtx = pd.DataFrame(data=cadata.obsm[aucell_key], index=cadata.obs.index,
                           columns=cadata.uns[regulons_key])
    df_scores = auc_mtx.copy()
    mean_scores = df_scores.mean()
    std_scores = df_scores.std()
    df_scores['group'] = cadata.obs[groupby]

    df_results = ((df_scores.groupby(by='group').mean() - mean_scores) / std_scores).stack()\
        .reset_index().rename(columns={'level_1': 'regulon', 0: 'Z'})
    
    if sort:
        df_results = df_results.sort_values(['group', 'Z'], ascending=[True, False])
    
    return df_results
    

def scenic_regulon_enrichment(adata, groupby, group, groups=None, 
                              n_regulons=None, zscore_cutoff=None,
                              regulon_names=None,
                              aucell_key='X_aucell', regulons_key='regulons',
                              plot=False):
    if not isinstance(adata, ConstrainedAdata):
        cadata = ConstrainedAdata(adata)
    else:
        cadata = adata
    
    if groups is None:
        groups = cadata.obs[groupby].dtype.categories.to_list()
        
    df_results = scenic_regulon_enrichment_scores(cadata, groupby, aucell_key=aucell_key, regulons_key=regulons_key)

    df_group = df_results[df_results['group'] == group].sort_values('Z', ascending=False)
    
    enriched_regulons = set()
    df_sig = df_group.copy()
    if zscore_cutoff is not None:
        df_sig = df_sig[df_sig['Z'] > zscore_cutoff]
    
    if n_regulons is not None:
        df_sig = df_sig.head(n=n_regulons)
    
    if regulon_names is not None:
        df_sig = df_sig[df_sig['regulon'].isin(regulon_names)]
    
    enriched_regulons = set(df_sig['regulon'])
    df_results_sub = df_results[np.logical_and(df_results['regulon'].isin(enriched_regulons),
                                               df_results['group'].isin(groups))]

    #df_results_sub['regulon'] = list(map(lambda s: s[:-3], df_results_sub.regulon))
    print(df_results_sub)
    
    if plot:
        ax = scenic_regulon_enrichment_heatmap(df_results_sub)
        return df_results_sub, ax
    
    return df_results_sub
