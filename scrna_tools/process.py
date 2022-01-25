import scanpy as sc
import scanpy.external as sce
from collections import Counter
import pandas as pd
from .helpers import markers_to_df
import logging

logger = logging.getLogger('__name__')

try:
    import harmony
    has_harmony = True
except ImportError:
    has_harmony = False
    
try:
    import palantir
    has_palantir = True
except ImportError:
    has_palantir = False


def recluster(adata, resolutions=(0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4,
                                  1.6, 1.8, 2.0, 2.5),
              key_prefix='leiden'):
    for resolution in resolutions:
        logger.info("Resolution: {}".format(resolution))
        sc.tl.leiden(adata, resolution=resolution, key_added='{}_{}'.format(key_prefix, resolution))

    return adata


def relayout(adata, n_pcs=100, use_harmony=True, batch_key='sample', n_top_genes=3000,
             umap=True, tsne=True, fdl=True, use_batch_for_hvg=True,
             hvg=True, scale=False, seurat_hvg=False):
    if hvg:
        if not seurat_hvg:
            sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, batch_key=batch_key if use_batch_for_hvg else None)
        else:
            sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, batch_key=batch_key if use_batch_for_hvg else None,
                                        flavor='seurat_v3', layer='counts', min_mean=0.1, max_mean=8, min_disp=1)

    pca_key = 'X_pca'
    sc.tl.pca(adata, svd_solver='arpack', n_comps=n_pcs, use_highly_variable=True, random_state=42)

    if use_harmony:
        pca_key = 'X_pca_harmony'
        sce.pp.harmony_integrate(adata, key=batch_key, max_iter_harmony=50)

    sc.pp.neighbors(adata, n_pcs=n_pcs, use_rep=pca_key, knn=True, n_neighbors=30)
    
    if scale:
        sc.pp.scale(adata)

    if umap:
        sc.tl.umap(adata, min_dist=0.3, random_state=42)

    if tsne:
        sc.tl.tsne(adata, use_rep=pca_key)

    if fdl:
        if has_harmony and has_palantir:
            pca_projections = pd.DataFrame(adata.obsm[pca_key], index=adata.obs_names)
            dm_res = palantir.utils.run_diffusion_maps(pca_projections, n_components=5)
            fdl = harmony.plot.force_directed_layout(dm_res['kernel'], adata.obs_names)
            adata.obsm['X_fdl'] = fdl.to_numpy()
        else:
            logger.warning("To use FDL layout you have to install palantir and harmony. Skipping FDL.")

    return adata


def recalculate_markers(adata, output_file=None, sort_by_abs_score=True, include_groups=None, **kwargs):
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


def count_groups(adata, key, splitby=None, split_order=None, group_order=None, **kwargs):
    groups = []
    if group_order is None:
        group_order = adata.obs[key].unique()
    if splitby is not None:
        if split_order is None:
            split_order = adata.obs[splitby].unique()
        for split in split_order:
            groups.append(list(adata.obs[key][adata.obs[splitby] == split]))
    else:
        split_order = ['count']
        groups = [list(adata.obs[key])]

    counts = []
    for group in groups:
        c = Counter(group)
        counts.append([c[g] for g in group_order])

    df = pd.DataFrame(counts, columns=group_order, index=split_order)
    return df