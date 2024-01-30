import numpy as np

from .plotting import tf_activity_enrichment_heatmap

def tf_activity_enrichment_scores(vdata, tf_activity_df, groupby, sort=True):
    df_scores = tf_activity_df.copy()
    mean_scores = df_scores.mean()
    std_scores = df_scores.std()
    df_scores['group'] = vdata.obs[groupby]

    df_results = ((df_scores.groupby(by='group').mean() - mean_scores) / std_scores).stack()\
        .reset_index().rename(columns={'level_1': 'tf', 0: 'Z'})
    
    if sort:
        df_results = df_results.sort_values(['group', 'Z'], ascending=[True, False])
    
    return df_results


def tf_activity_enrichment(vdata, tf_activity_df, 
                           groupby, group, groups=None, 
                           n_tfs=None, zscore_cutoff=None,
                           regulon_names=None,
                           plot=False, **plot_kwargs):
    if groups is None:
        groups = vdata.obs[groupby].dtype.categories.to_list()
        
    df_results = tf_activity_enrichment_scores(vdata, tf_activity_df, groupby)
    df_group = df_results[df_results['group'] == group].sort_values('Z', ascending=False)
    
    enriched_regulons = set()
    df_sig = df_group.copy()
    if zscore_cutoff is not None:
        df_sig = df_sig[df_sig['Z'] > zscore_cutoff]
    
    if n_tfs is not None:
        df_sig = df_sig.head(n=n_tfs)
    
    if regulon_names is not None:
        df_sig = df_sig[df_sig['tf'].isin(regulon_names)]
    
    enriched_regulons = set(df_sig['tf'])
    df_results_sub = df_results[np.logical_and(df_results['tf'].isin(enriched_regulons),
                                               df_results['group'].isin(groups))]

    if plot:
        ax = tf_activity_enrichment_heatmap(df_results_sub, **plot_kwargs)
        return df_results_sub, ax
    
    return df_results_sub
