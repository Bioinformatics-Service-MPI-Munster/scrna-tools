import os
import errno
import pandas as pd
import numpy as np
from future.utils import string_types


def mkdir(*dir_name):
    dir_name = os.path.expanduser(os.path.join(*dir_name))

    try:
        os.makedirs(dir_name)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

    if not dir_name.endswith('/'):
        dir_name += '/'

    return dir_name


def markers_to_df(adata, key, output_file=None, sort_by_abs_score=True,
                  include_groups=None):

    genes = adata.var.index.to_list()

    dfs = []
    for name in adata.uns[key]['names'].dtype.names:
        df = pd.DataFrame.from_dict({
            'gene': list(adata.uns[key]['names'][name]),
            'score': list(adata.uns[key]['scores'][name]),
            'log2fc': list(adata.uns[key]['logfoldchanges'][name]),
            'pval': list(adata.uns[key]['pvals'][name]),
            'padj': list(adata.uns[key]['pvals_adj'][name]),
            'cluster': [name for _ in range(len(list(adata.uns[key]['names'][name])))]
        })
        df = df.set_index('gene')
        df = df.filter(items=genes, axis=0)
        if 'pts' in adata.uns[key].keys():
            df['pts'] = adata.uns[key]['pts'][name].loc[df.index]
        if 'pts_rest' in adata.uns[key].keys():
            df['pts_rest'] = adata.uns[key]['pts_rest'][name].loc[df.index]

        if sort_by_abs_score:
            df = df.iloc[(-df.score.abs()).argsort()]

        dfs.append(df)

    df = pd.concat(dfs)

    if include_groups is not None:
        if isinstance(include_groups, string_types):
            include_groups = [include_groups]
        include_groups = set(include_groups)
        ixs = [c in include_groups for c in df['cluster']]
        df = df[ixs]

    if output_file is not None:
        df.to_csv(output_file, sep="\t")

    return df


def ortholog_dict(ortholog_file, invert=False):
    with open(ortholog_file, 'r') as f:
        ortholog_genes = set()
        converter = {}
        for line in f:
            line = line.rstrip()
            if line == '':
                continue
            
            if invert:
                ortholog, old = line.split("\t")
            else:
                old, ortholog = line.split("\t")

            i = 0
            while ortholog in ortholog_genes:
                ortholog = '{}_{}'.format(ortholog, i)
                i += 1
            converter[old] = ortholog
            ortholog_genes.add(ortholog)
    return converter


def ortholog_converter(gene_names, ortholog_file, invert=False):
    converter = ortholog_dict(ortholog_file, invert=invert)
    return [converter.get(g, g) for g in gene_names]


def ortholog_converter_adata(adata, ortholog_file, invert=False, key_added=None):
    gene_names = ortholog_converter(gene_names=adata.var.index.to_list(), ortholog_file=ortholog_file, invert=invert)
    
    if key_added is not None:
        adata.var[key_added] = gene_names
    
    return gene_names


def to_loom(adata, output_file, layer=None, obs_keys=None, var_keys=None):
    import loompy as lp
    
    if layer is None:
        x = adata.X
    else:
        x = adata.layers[layer]

    row_attrs = {
        "Gene": np.array(adata.var_names),
    }
    if var_keys is not None:
        for key in var_keys:
            row_attrs[key] = adata.var[key].to_numpy()
    
    col_attrs = {
        "CellID": np.array(adata.obs_names),
        "nGene": np.array(np.sum(x.transpose() > 0, axis=0)).flatten(),
        "nUMI": np.array(np.sum(x.transpose(), axis=0)).flatten(),
    }
    if obs_keys is not None:
        for key in obs_keys:
            col_attrs[key] = adata.obs[key].to_numpy()
    
    loom = lp.create(output_file, x.transpose(), row_attrs, col_attrs)
    return loom
