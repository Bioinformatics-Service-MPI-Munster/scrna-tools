import os
import sys
import errno
import pandas as pd
import numpy as np
import io
import re
import gzip
import urllib.request
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


def smart_open(file_name=None, mode='r', compressed=None):
    if compressed is None:
        if file_name.endswith('.gz') or file_name.endswith('.gzip'):
            compressed = True
        else:
            compressed = False

    if file_name is not None and file_name != '-':
        if compressed:
            fh = gzip.open(file_name, mode)
            if mode in {'r', 'rb', 'r+', 'rb+'}:
                fh = io.BufferedReader(fh)
        else:
            fh = open(file_name, mode)
    else:
        fh = sys.stdout

    return fh


def markers_to_df(adata, key, output_file=None, sort_by_abs_score=True,
                  include_groups=None):

    genes = adata.var.index.to_list()

    dfs = []
    for name in adata.uns[key]['names'].dtype.names:
        data = {
            'gene': list(adata.uns[key]['names'][name]),
            'score': list(adata.uns[key]['scores'][name]),
            'log2fc': list(adata.uns[key]['logfoldchanges'][name]),
            'pval': list(adata.uns[key]['pvals'][name]),
            'padj': list(adata.uns[key]['pvals_adj'][name]),
            'cluster': [name for _ in range(len(list(adata.uns[key]['names'][name])))]
        }
        # for k in adata.uns[key].keys():
        #     data[k] = list(adata.uns[key][k][name])
            
        df = pd.DataFrame.from_dict(data)
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


def find_cells_from_coords(adata, key, x, y, delta=0.1):
    coords = adata.obsm[key]
    x_constraint = np.logical_and(
        coords[:, 0] < x + delta,
        coords[:, 0] > x - delta,
    )
    y_constraint = np.logical_and(
        coords[:, 1] < y + delta,
        coords[:, 1] > y - delta,
    )
    return adata.obs.iloc[
        np.where(
            np.logical_and(x_constraint, y_constraint)
        )
    ].index


def merge_var_columns(var, key_prefix, dtype=None):
    columns = [c for c in var.columns if bool(re.match(f'{key_prefix}-\d+', c))]

    final_column = pd.Series(var[columns.pop(0)].to_list(), index=var.index, name=key_prefix)
    for column in columns:
        ixs = final_column.isna()
        final_column[ixs] = var[column][ixs]
    
    if dtype is None:
        return final_column
    return final_column.astype(dtype)


def load_gene_set_from_gmt(file_name):
    genes = []
    with open(file_name, 'r') as f:
        if file_name.endswith('.gmt'):
            for i, line in enumerate(f):
                if i == 0:
                    line = line.rstrip()
                    fields = line.split("\t")
                    genes = fields[2:]
        else:    
            for line in f:
                line = line.rstrip()
                if line == '':
                    continue
                genes.append(line)
    return genes


def load_gene_set_from_msigdb(gene_set_id, organism='human'):
    url = f'https://www.gsea-msigdb.org/gsea/msigdb/{organism}/download_geneset.jsp?geneSetName={gene_set_id}&fileType=gmt'
    with urllib.request.urlopen(url) as f:
        content = f.read().decode('utf-8')
    
    fields = content.split("\t")
    genes = fields[2:]
    return genes
