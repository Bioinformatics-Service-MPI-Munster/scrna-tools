import anndata
import scipy
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import json
import re
from collections import defaultdict, Counter
from future.utils import string_types
from .process import relayout, recluster, recalculate_markers, cell_cycle_regression, gene_stats, rename_groups
from .plotting import color_cycle, embedding_plot, volcano_plot_from_df, violin_plot
from .helpers import markers_to_df
from .trajectory import paga
from .r import de_pseudobulk, slingshot
from functools import wraps
import requests
import time
import logging

try:
    import gseapy
    with_gseapy = True
except (ModuleNotFoundError, OSError):
    with_gseapy = False


logger = logging.getLogger('__name__')


def requires_gseapy(func):
    """Checks if gseapy is installed in path"""
    
    @wraps(func)
    def wrapper_gseapy(*args, **kwargs):
        if not with_gseapy:
            raise RuntimeError("gseapy is not installed, cannot "
                               "run code that depends on it!")
        return func(*args, **kwargs)
    return wrapper_gseapy


class ConstrainedAdata(object):
    def __init__(self, adata, annotations_adata=None,
                 _uns_subset_key='__subsets__',
                 _subset_key_prefix='__subset__'):
        self.adata = adata
        self.annotations_adata = annotations_adata
        self.constraints = []
        self.gene_constraints = []
        self._uns_subset_key = _uns_subset_key
        self._subset_key_prefix = _subset_key_prefix

    @property
    def obs_constraints(self):
        return self.constraints
    
    @property
    def var_constraints(self):
        return self.gene_constraints
    
    @classmethod
    def read(cls, file_name, annotations_file=None, **kwargs):
        adata = sc.read(file_name)
        annotations_adata = sc.read(annotations_file) if annotations_file is not None else None
        return cls(adata, annotations_adata=annotations_adata, **kwargs)

    def __getattr__(self, item):
        return getattr(self.adata[self.valid], item)
    
    def _get_obs_column(self, obs_key, only_valid=True, prefer_annotations=True):
        adata = None
        if self.annotations_adata is not None and obs_key in self.annotations_adata.obs.columns:
            adata = self.annotations_adata
        
        if (adata is None or not prefer_annotations) and obs_key in self.adata.obs.columns:
            adata = self.adata
        
        if adata is None:
            raise KeyError("Cannot find key '{}' in adata or annotations!".format(obs_key))
        
        if only_valid:
            return adata.obs.iloc[self.valid][obs_key]
        return adata.obs[obs_key]

    def _feature_values(self, feature, layer=None, ignore_constraints=False):
        try:
            feature_ix = np.argwhere(self.adata.var.index == feature)[0][0]
            
            if layer is None:
                x = self.adata.X if ignore_constraints else self.X
            else:
                x = self.adata.layers(layer) if ignore_constraints else self.layers(layer)
            if isinstance(x, h5py.Dataset):
                feature_values = x[:, feature_ix][:, 0]
            elif (isinstance(x, anndata._core.sparse_dataset.SparseDataset) or
                  isinstance(x, scipy.sparse.spmatrix)):
                feature_values = x[:, feature_ix].toarray()[:, 0]
            elif (isinstance(x, np.ndarray)):
                feature_values = x[:, feature_ix]
            else:
                raise ValueError("Unknown data structure for AnnData.X: {}".format(type(x)))
        except IndexError:
            try:
                feature_values = self._get_obs_column(feature, only_valid=not ignore_constraints).to_numpy()
            except IndexError:
                raise ValueError("Feature '{}' not in AnnData object".format(feature))

        return feature_values
    
    def values(self, feature, **kwargs):
        return self._feature_values(feature, **kwargs)
    
    @wraps(gene_stats)
    def gene_stats(self, *args, **kwargs):
        return gene_stats(self, *args, **kwargs)
    
    def gene_stats_old(self, genes=None, layer=None, obs=None,
                   groupby=None, groups=None, ignore_groups=('NA', 'NaN'),
                   mean_expression=True, percent_expressed=True, sd_expression=False):
        cadata = self.copy()

        if genes is not None:
            cadata.add_gene_constraint(genes)
        
        if obs is None:
            obs = cadata.obs
        else:
            if obs.shape[0] != cadata.shape[0]:
                raise ValueError("obs shape must match cadata shape!")
        
        group_ixs = {}
        if groupby is not None:
            if groups is None:
                groups = obs[groupby].dtype.categories
            
            groups = [g for g in groups if g not in ignore_groups]
            print(f'GROUPS: {groups}')
            print(f'OBS:')
            print(obs)
            print(obs[groupby].dtype.categories)
            valid = obs[groupby].isin(groups)
            
            cadata.add_index_constraint(obs.index[valid])
            
            print(sum(valid), cadata.shape)
            
            obs_group = obs[groupby]
            for group in groups:
                group_ixs[group] = obs_group == group
        else:
            group_ixs['all'] = np.array([True] * obs.shape[0])
        
        if layer is None:
            x = cadata.X
        else:
            x = cadata.layers[layer]
        x = x.toarray()
        
        print(x.shape, cadata.shape)
        
        stats = pd.DataFrame(index=cadata.var.index)
        for group, ixs in group_ixs.items():
            print(ixs.shape)
            group_clean = group.replace(' ', '_')
            x_sub = x[ixs, :]
            
            if percent_expressed:
                is_expressed = np.array(x_sub) > 0
                stats[f'percent_expressed_{group_clean}'] = np.sum(is_expressed, axis=0) / is_expressed.shape[0] * 100
            if mean_expression:
                stats[f'mean_expression_{group_clean}'] = np.array(x_sub.mean(axis=0))
            if sd_expression:
                stats[f'sd_expression_{group_clean}'] = np.array(x_sub.sd(axis=0))
        
        return stats

    def has_constraints(self):
        return self.has_obs_constraints() or self.has_var_constraints()
    
    def has_obs_constraints(self):
        return len(self.constraints) > 0
    
    def add_categorical_obs_constraint(self, *args, **kwargs):
        # compatibility with AnnDataView
        if 'key' in kwargs and 'group_key' not in kwargs:
            kwargs['group_key'] = kwargs.pop('key')
        if 'categories' in kwargs and 'groups' not in kwargs:
            kwargs['groups'] = kwargs.pop('categories')
            
        return self.add_categorical_constraint(*args, **kwargs)

    def add_categorical_constraint(self, group_key, groups=None, invert=False):
        obs = self._get_obs_column(group_key, only_valid=False)
        
        if not obs.dtype.name == 'category':
            raise ValueError("Subsetting currently only works on "
                             "categorical groups, not {}!".format(obs.dtype.name))

        if isinstance(groups, string_types):
            groups = [groups]

        self.constraints.append(dict(
            type="categorical",
            group_key=group_key,
            groups=groups,
            invert=invert,
        ))
        
        return self

    def _convert_categorical_constraint(self, constraint):
        group_key = constraint.get('group_key')
        groups = constraint.get('groups', None)
        invert = constraint.get('invert', False)
        
        obs_data = self._get_obs_column(group_key, only_valid=False)

        if groups is None:
            groups = obs_data.dtype.categories.to_list()
        groups_set = set(groups)

        subset_bool = np.repeat(True, self.adata.shape[0])
        for i, (index, item) in enumerate(obs_data.iteritems()):
            if item not in groups_set:
                subset_bool[i] = False

        if invert:
            return ~subset_bool
        return subset_bool

    def add_numerical_obs_constraint(self, *args, **kwargs):
        if 'key' in kwargs:
            kwargs['feature'] = kwargs['key']
            del kwargs['key']
        return self.add_numerical_constraint(*args, **kwargs)

    def add_numerical_constraint(self, feature, min_value=None, max_value=None, exclusive=None):
        if min_value == "":
            min_value = None
        if max_value == "":
            max_value = None

        if min_value is None and max_value is None:
            raise ValueError("min_value and max_value cannot both be None")

        self.constraints.append(dict(
            type='numerical',
            feature=feature,
            min_value=float(min_value) if min_value is not None else None,
            max_value=float(max_value) if max_value is not None else None,
            exclusive=exclusive,
        ))
        
        return self

    def _convert_numerical_constraint(self, constraint):
        feature = constraint.get("feature")
        min_value = constraint.get("min_value", None)
        max_value = constraint.get("max_value", None)
        exclusive = constraint.get("exclusive", None)

        feature_values = self._feature_values(feature, ignore_constraints=True)

        subset_bool = np.repeat(True, self.adata.shape[0])
        if min_value is not None:
            if exclusive is not None and exclusive:
                subset_bool = np.logical_and(subset_bool, feature_values > min_value)
            else:
                subset_bool = np.logical_and(subset_bool, feature_values >= min_value)

        if max_value is not None:
            if exclusive is not None and exclusive:
                subset_bool = np.logical_and(subset_bool, feature_values < max_value)
            else:
                subset_bool = np.logical_and(subset_bool, feature_values <= max_value)

        return subset_bool

    def add_subset_constraint(self, subset_key):
        key = subset_key
        if subset_key in self.adata.obs.columns:
            key = subset_key
        elif f'__view__{subset_key}' in self.adata.obs.columns:
            key = f'__view__{subset_key}'
        else:
            key = self._subset_key(subset_key)
        
        self.constraints.append(dict(
            type="subset",
            subset_key=key,
        ))
        
        return self

    def add_view_constraint(self, view_key):
        return self.add_subset_constraint(view_key)

    def _convert_subset_constraint(self, constraint):
        subset_key = constraint.get("subset_key")
        if subset_key in self.adata.obs.columns:
            return self.adata.obs[subset_key]
        
        if subset_key is not None and subset_key != 'all':
            subset_bool = np.repeat(False, self.adata.shape[0])
            subsets = self.saved_subsets()

            for ix in subsets[subset_key]['valid_ixs']:
                subset_bool[ix] = True
        else:
            subset_bool = np.repeat(True, self.adata.shape[0])

        return subset_bool
    
    def add_index_obs_constraint(self, *args, **kwargs):
        return self.add_index_constraint(*args, **kwargs)

    def add_index_constraint(self, ixs):
        self.constraints.append(dict(
            type='index',
            ixs=ixs
        ))
        
        return self

    def _convert_index_constraint(self, constraint):
        ixs = constraint.get("ixs")
        subset_bool = np.repeat(False, self.adata.shape[0])
        is_string_index = False
        for i, ix in enumerate(ixs):
            if isinstance(ix, (bool, np.bool_)):
                subset_bool[i] = ix
            elif isinstance(ix, string_types):
                is_string_index = True
                break
            else:
                subset_bool[ix] = True
        
        if is_string_index:
            subset_bool = self.adata.obs.index.isin(ixs)

        return subset_bool

    def add_zscore_constraint(self, feature, min_score=None, max_score=None, include_zeros=True):
        self.constraints.append(dict(
            type='zscore',
            feature=feature,
            min_score=min_score,
            max_score=max_score,
            include_zeros=include_zeros,
        ))
        
        return self

    def _convert_zscore_constraint(self, constraint):
        feature = constraint.get("feature")
        min_value = constraint.get("min_score", None)
        max_value = constraint.get("max_score", None)
        include_zeros = constraint.get("include_zeros", True)

        feature_values = self._feature_values(feature, ignore_constraints=True)
        if not include_zeros:
            non_zero = feature_values[feature_values > 0]
        else:
            non_zero = np.repeat(True, len(feature_values))

        mean = np.nanmean(feature_values[non_zero])
        sd = np.nanstd(feature_values[non_zero])

        zscores = (feature_values - mean) / sd
        zscores[~non_zero] = np.nan

        subset_bool = non_zero.copy()
        if min_value is not None:
            subset_bool = np.logical_and(subset_bool, zscores > min_value)

        if max_value is not None:
            subset_bool = np.logical_and(subset_bool, zscores < max_value)

        return subset_bool

    def add_constraint(self, constraint):
        if constraint['type'] in {'categorical', 'category', 'group'}:
            return self.add_categorical_constraint(constraint['group_key'],
                                                   groups=constraint.get('groups', None))
        elif constraint['type'] in {'numeric', 'numerical', 'feature'}:
            return self.add_numerical_constraint(constraint['feature'],
                                                 min_value=constraint.get('min_value', None),
                                                 max_value=constraint.get('max_value', None))
        elif constraint['type'] in {'zscore', 'zscores'}:
            return self.add_zscore_constraint(constraint['feature'],
                                              min_score=constraint.get('min_score', None),
                                              max_score=constraint.get('max_score', None),
                                              include_zeros=constraint.get('include_zeros', True))
        elif constraint['type'] in {'subset'}:
            return self.add_subset_constraint(constraint['subset_key'])
        elif constraint['type'] in {'ix', 'index', 'ixs', 'indexes'}:
            return self.add_index_constraint(constraint['ixs'])
        else:
            raise ValueError("Constraint type '{}' not supported".format(constraint['type']))
        
        return self

    def add_index_var_constraint(self, ixs):
        return self.add_gene_constraint(ixs)
    
    def add_gene_constraint(self, genes):
        return self.add_var_constraint(genes)
        
    def add_var_constraint(self, vars):
        if isinstance(vars, string_types) or isinstance(vars, int):
            vars = [vars]
        self.gene_constraints.append(vars)
        
        return self

    @property
    def valid(self):
        subset_bool = np.repeat(True, self.adata.shape[0])
        for constraint in self.constraints:
            if constraint['type'] == 'numerical':
                subset_bool = np.logical_and(subset_bool, self._convert_numerical_constraint(constraint))
            elif constraint['type'] == 'categorical':
                subset_bool = np.logical_and(subset_bool, self._convert_categorical_constraint(constraint))
            elif constraint['type'] == 'subset':
                subset_bool = np.logical_and(subset_bool, self._convert_subset_constraint(constraint))
            elif constraint['type'] == 'index':
                subset_bool = np.logical_and(subset_bool, self._convert_index_constraint(constraint))
            elif constraint['type'] == 'zscore':
                subset_bool = np.logical_and(subset_bool, self._convert_zscore_constraint(constraint))

        return subset_bool

    @property
    def valid_ixs(self):
        v = self.valid
        return np.argwhere(v)[:, 0]
    
    def _adata_gene_subset(self, adata=None):
        if adata is None:
            adata = self.adata
            
        if self.has_var_constraints():
            genes_set = set()
            for genes in self.gene_constraints:
                genes_set = genes_set.union(set(genes))
            adata_sub = adata[:, list(genes_set)]
        else:
            adata_sub = adata
        
        return adata_sub
    
    def has_var_constraints(self):
        return self.gene_constraints is not None and len(self.gene_constraints) > 0
    
    @property
    def var(self):
        adata_sub = self._adata_gene_subset()
        return adata_sub.var

    @property
    def X(self):
        adata_sub = self._adata_gene_subset()

        if self.has_constraints():
            return adata_sub.X[self.valid]
        return adata_sub.X
    
    @property
    def shape(self):
        return (self.obs.shape[0], self.var.shape[0])
    
    @property
    def layers(self):
        class ConstrainedAdataLayer(object):
            def __init__(self, cadata):
                self.cadata = cadata
            
            def __getitem__(self, key):
                adata_sub = self.cadata._adata_gene_subset()

                if self.cadata.has_constraints():
                    return adata_sub.layers[key][self.cadata.valid]
                return adata_sub.layers[key]
            
            def __setitem__(self, key, item):
                if self.cadata.has_constraints() or self.cadata.has_var_constraints():
                    raise ValueError("Cannot set layer item on AnnData with constraints")
                else:
                    self.cadata.adata.layers[key] = item
        return ConstrainedAdataLayer(self)

    @property
    def Xarray(self):
        return self.X[:].toarray()
    sc.read

    @property
    def obs(self):
        obs = self.adata.obs
        if self.annotations_adata is not None:
            obs = self.annotations_adata.obs.combine_first(obs)

        if self.has_constraints():
            return obs[self.valid]
        return obs

    @property
    def obsm(self):
        if len(self.constraints) > 0:
            obsm = self.adata[self.valid].obsm
        else:
            obsm = self.adata.obsm
        
        if self.annotations_adata is not None:
            for key in self.annotations_adata.obsm_keys():
                if len(self.constraints) > 0:
                    obsm[key] = self.annotations_adata[self.valid].obsm[key]
                else:
                    obsm[key] = self.annotations_adata.obsm[key]

        return obsm
    
    @property
    def uns(self):
        if len(self.constraints) > 0:
            uns = self.adata[self.valid].uns
        else:
            uns = self.adata.uns
        
        if self.annotations_adata is not None:
            for key in self.annotations_adata.uns_keys():
                if len(self.constraints) > 0:
                    uns[key] = self.annotations_adata[self.valid].uns[key]
                else:
                    uns[key] = self.annotations_adata.uns[key]

        return uns

    @property
    def adata_subset(self):
        adata_sub = self._adata_gene_subset()
        return adata_sub[self.valid]
    
    @property
    def adata_view(self):
        return self.adata_subset
    
    @property
    def annotations_adata_subset(self):
        annotations_adata_sub = self._adata_gene_subset(adata=self.annotations_adata)
        return annotations_adata_sub[self.valid]

    def copy(self, only_constraints=True):
        if only_constraints:
            copy = ConstrainedAdata(self.adata, annotations_adata=self.annotations_adata)
        else:
            copy = ConstrainedAdata(self.adata.copy(), annotations_adata=self.annotations_adata)
        copy.constraints = self.constraints.copy()
        copy.gene_constraints = self.gene_constraints.copy() if self.gene_constraints is not None else None
        return copy

    def save_subset(self, subset_key, npcs=50, min_gene_cells=None, 
                    cluster=False, umap=False, tsne=False, fdl=False, seurat=False,
                    batch_correct=True, batch_key='sample', batch_algorithm='harmony',
                    hvg=True, n_top_genes=2000, use_batch_for_hvg=True, 
                    leiden_resolutions=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                                        0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5],
                    layer=None, cc_regression=False, subset_name=None):
        if self.annotations_adata is not None:
            adata = self.annotations_adata
        else:
            adata = self.adata
        
        subset_key = self._subset_key(subset_key)
        if subset_name is None:
            subset_name = subset_key

        if self._uns_subset_key not in adata.uns_keys():
            adata.uns[self._uns_subset_key] = dict()
        
        adata.obs[subset_key] = self.valid

        adata.uns[self._uns_subset_key][subset_key] = dict(
            key=subset_key,
            name=subset_name,
            constraints=json.dumps(self.constraints),
            cc_regression=cc_regression,
        )
        if layer is not None:
            adata.uns[self._uns_subset_key][subset_key]['layer'] = layer
        
        if cluster or umap or tsne:
            sub_adata_tmp = self.adata_subset
            sub_adata = sc.AnnData(sub_adata_tmp.X if layer is None else sub_adata_tmp.layers[layer],
                                   obs=sub_adata_tmp.obs, var=sub_adata_tmp.var)
            sub_adata.raw = sub_adata
            sub_adata.layers['counts'] = sub_adata_tmp.layers['counts']
            
            if umap or tsne or fdl:
                if min_gene_cells:
                    sc.pp.filter_genes(sub_adata, min_cells=min_gene_cells)

                sub_adata = relayout(sub_adata, n_pcs=npcs, batch_key=batch_key, batch_algorithm=batch_algorithm,
                                     umap=umap, tsne=tsne, fdl=fdl, n_top_genes=n_top_genes,
                                     use_batch_for_hvg=use_batch_for_hvg, seurat_hvg=seurat, hvg=hvg,
                                     cc_regression=cc_regression)

                embeddings = ['pca']
                if 'X_integrated' in sub_adata.obsm_keys():
                    embeddings.append('integrated')
                if umap and 'X_umap' in sub_adata.obsm_keys():
                    embeddings.append('umap')
                if tsne and 'X_tsne' in sub_adata.obsm_keys():
                    embeddings.append('tsne')
                if fdl and 'X_fdl' in sub_adata.obsm_keys():
                    embeddings.append('fdl')

                # transfer embeddings
                for embedding in embeddings:
                    sub_coords = sub_adata.obsm['X_{}'.format(embedding)]
                    coords = np.empty((self.adata.obs.shape[0], sub_coords.shape[1]))
                    coords[:] = np.nan
                    coords[self.valid] = sub_coords
                    adata.obsm['{}__{}'.format(subset_key, embedding)] = coords
                
                if cc_regression:
                    adata.uns['{}__neighbors_cc_regression'.format(subset_key)] = sub_adata.uns['neighbors']

            if cluster:
                logger.info("Re-clustering")
                sub_adata = recluster(sub_adata, key_prefix='{}__leiden'.format(subset_key),
                                      resolutions=leiden_resolutions)
                
                # subcluster leiden clusters
                for obs_key in sub_adata.obs_keys():
                    if obs_key.startswith('{}__leiden'.format(subset_key)):
                        base_obs = np.repeat('NA', adata.shape[0])
                        base_obs[self.valid] = sub_adata.obs[obs_key]
                        adata.obs[obs_key] = pd.Categorical([x for x in base_obs])

        return adata

    def subset_info(self, subset):
        if not subset.startswith(self._subset_key_prefix):
            subset = self._subset_key(subset)

        return self.saved_subsets()[subset]

    def saved_subsets(self, key=None, _view_prefix=None):
        _subset_key = _view_prefix or self._uns_subset_key
        subsets = dict()
        
        adatas = [self.adata]
        if self.annotations_adata is not None:
            adatas.append(self.annotations_adata)
        
        for adata in adatas:
            if _subset_key not in adata.uns_keys():
                return subsets

            for subset_key, subset in adata.uns[_subset_key].items():
                if key is not None and subset_key != key:
                    continue
                
                s = subset.copy()
                try:
                    s['constraints'] = json.loads(subset['constraints'])
                except KeyError:
                    s['constraints'] = json.loads(subset['obs_constraints'])
                
                if _view_prefix is not None and not subset_key.startswith(_view_prefix):
                    subset_key = f'{_view_prefix}{subset_key}'
                subsets[subset_key] = s

                # check for existing embeddings
                pca_key = '{}__pca'.format(subset_key)
                if pca_key in adata.obsm_keys():
                    s['pca_key'] = pca_key
                pca_key = '{}__pca_harmony'.format(subset_key)
                if pca_key in adata.obsm_keys():
                    s['pca_key'] = pca_key
                integrated_key = '{}__integrated'.format(subset_key)
                if integrated_key in adata.obsm_keys():
                    s['integrated_key'] = integrated_key
                umap_key = '{}__umap'.format(subset_key)
                if umap_key in adata.obsm_keys():
                    s['umap_key'] = umap_key
                tsne_key = '{}__tsne'.format(subset_key)
                if tsne_key in adata.obsm_keys():
                    s['tsne_key'] = tsne_key
                fdl_key = '{}__fdl'.format(subset_key)
                if fdl_key in adata.obsm_keys():
                    s['fdl_key'] = fdl_key

                # check for existing clustering results
                leiden_clusters = dict()
                leiden_key_re = re.compile('{}__leiden_(.+)'.format(subset_key))
                for obs_key in adata.obs_keys():
                    m = re.match(leiden_key_re, obs_key)
                    if m is not None:
                        leiden_clusters[float(m.group(1))] = obs_key
                s['leiden_clusters'] = leiden_clusters
                
                # check for subet key in obs
                try:
                    s['valid'] = adata.obs[subset_key]
                except KeyError:
                    if not 'valid' in s:
                        raise
                s['valid_ixs'] = np.where(s['valid'])[0]
        
        if key is not None:
            return subsets[key]

        return subsets
    
    def view_info(self, *args, **kwargs):
        return self.saved_subsets(*args, **kwargs)

    def _subset_key(self, subset_name):
        if subset_name.startswith(self._subset_key_prefix):
            return subset_name
        return "{}{}".format(self._subset_key_prefix, ''.join(e for e in subset_name if e.isalnum()))
    
    def rename_groups(self, obs_key, rename_dict, key_added=None):
        return rename_groups(self, obs_key=obs_key, rename_dict=rename_dict, key_added=key_added)
    
    def annotate(self, group_key, key_added=None, template_key=None,
                 ignore_groups=('NA', 'nan', 'NaN')):
        valid = self.valid
        groups_obs = self._get_obs_column(group_key, only_valid=False).to_numpy()
        groups = np.unique(groups_obs[valid])
        
        new_groups = ['NA' for _ in range(len(groups_obs))]
        if template_key is not None:
            template = self._get_obs_column(template_key, only_valid=False).to_numpy()
            
            for group in groups:
                ixs = np.logical_and(groups_obs == group, valid)
                template_sub = template[ixs]
                template_counts = Counter(template_sub)
                for g in ignore_groups:
                    template_counts.pop(g, 0)
                highest_vote = max(template_counts.keys(), key=(lambda k: template_counts[k]))
                logger.info("Group '{}' was voted to be '{}' with {} votes"
                            .format(group, highest_vote, template_counts[highest_vote]))
                for i in np.where(ixs)[0]:
                    new_groups[i] = highest_vote
        
        if self.annotations_adata is not None:
            adata = self.annotations_adata
        else:
            adata = self.adata
        
        if key_added is not None:
            adata.obs[key_added] = pd.Categorical(new_groups)
        else:
            adata.obs[group_key] = pd.Categorical(new_groups)
        
        return new_groups
    
    def default_colors(self, groupby, colors=None):
        alt_colors = color_cycle()
        if colors is not None:
            if not isinstance(colors, dict):
                raise ValueError('colors must be a dictionary!')
            
            if '__default_colors__' not in self.adata.uns_keys():
                self.uns['__default_colors__'] = dict()
            
            final_colors = dict()
            groups = self.adata.obs[groupby].dtype.categories
            for group in groups:
                group_no_slash = group.replace('/', '$_SLASH_$')
                if group not in colors:
                    logger.warning(f"Group '{group}' not in colors dictionary, setting default color")
                    final_colors[group_no_slash] = next(alt_colors)
                else:
                    final_colors[group_no_slash] = colors[group]
            self.uns['__default_colors__'][groupby] = final_colors
        
        try:
            d = {}
            for group, color in self.uns['__default_colors__'][groupby].items():
                group_with_slash = group.replace('$_SLASH_$', '/')
                d[group_with_slash] = color
            return d
        except KeyError:
            raise KeyError(f"No default colors found for '{groupby}'")

    def default_colormap(
        self,
        colorby=None,
        cmap=None,
        default='viridis',
    ):
        colorby = colorby or '__default__'
        
        if cmap is not None:
            if not isinstance(cmap, (str, bytes)):
                raise ValueError('cmap must be a string!')
            
            if '__default_colormap__' not in self.uns_keys():
                self.uns['__default_colormap__'] = dict()
            
            self.uns['__default_colomap__'][colorby] = cmap
            return cmap
        else:
            try:
                return self.uns['__default_colormap__'][colorby]
            except KeyError:
                return default
    
    def _de_key(self,
                sample_key, sample1, sample2,
                subset_key=None, 
                group_key=None, groups=None):
        if group_key is not None:
            if groups is None:
                groups = self.obs[group_key].dtype.categories.to_list()
            elif isinstance(groups, string_types):
                groups = [groups]
            group_string = "__{}_{}".format(group_key, "_AND_".join([
                group.replace('-', '_')
                .replace(' ', '_')
                .replace("(", "_")
                .replace(")", "_")
                .replace("?", "_")
                .lower() for group in sorted(groups)
            ]))
        else:
            group_string = ""
        
        de_key = 'de{subset}{group}__{sample}__' \
                 '{top}_VS_{bottom}'.format(
            subset=self._subset_key(subset_key) if subset_key is not None else "",
            group=group_string,
            sample=sample_key,
            top=sample1, bottom=sample2
        )
        return de_key

    def de(self,
           sample_key, sample1, sample2,
           subset_key=None,
           group_key=None, groups=None,
           ignore_constraints=False,
           force=False,
           **kwargs):
        de_key = self._de_key(sample_key, sample1, sample2,
                              subset_key=subset_key,
                              group_key=group_key, groups=groups)

        if not force and de_key in self.uns_keys():
            return markers_to_df(self.adata, de_key, include_groups=sample1, **kwargs)

        if not ignore_constraints and self.has_constraints():
            raise ValueError("Pre-existings constraints ({}) will be ignored for DE analysis. "
                             "Please save these constraints as a subset and re-run this analysis "
                             "using the matching subset key!".format(len(self.constraints)))

        cadata = self.copy()
        if subset_key is not None:
            cadata.add_subset_constraint(subset_key)

        if group_key is not None:
            cadata.add_categorical_constraint(group_key, groups)

        adata_sub = cadata.adata_subset
        df = recalculate_markers(adata_sub, include_groups=sample1,
                                 groupby=sample_key, key_added=de_key,
                                 reference=sample2,
                                 **kwargs)

        self.adata.uns[de_key] = adata_sub.uns[de_key]

        return df
    
    def de_complex(self, analysis_name=None, force=False, **kwargs):
        class DEComplex(object):
            def __init__(self, cadata, name=None, force=False, **kwargs):
                self.cadata = cadata.copy()
                if name == 'tmp':
                    raise ValueError("Analysis name cannot be 'tmp'!")
                self.name = name
                self.force = force
                self.sample1 = cadata.copy()
                self.sample2 = cadata.copy()
                self.kwargs = kwargs
                self._de = None
                self._de_pseudobulk = None
                self._parameters = None
                
                de_key = self._de_key()
                if self.name is not None and self._de_key() in self.cadata.uns_keys():
                    logger.info("Found existing DE analysis. Restoring constraints.")
                    if self.cadata.has_obs_constraints():
                        raise ValueError(f"Cadata already has constraints, cannot restore DE constraints from {name}")
                    self.cadata.constraints = json.loads(self.cadata.uns[de_key]['global_constraints'])
                    self.sample1.constraints = json.loads(self.cadata.uns[de_key]['sample1_constraints'])
                    self.sample2.constraints = json.loads(self.cadata.uns[de_key]['sample2_constraints'])
            
            def _de_key(self):
                if self.name is None:
                    return '__de__tmp'
                return f'__de__{self.name}'
            
            @property
            def de(self):
                if self._de is None:
                    raise ValueError("Must run 'results' before retrieving de")
                return self._de
            
            @property
            def parameters(self):
                if self._parameters is None:
                    raise ValueError("Must run 'results' before retrieving parameters")
                return self._parameters
            
            def _get_de_adata_sub(self):
                de_key = self._de_key()
                    
                sample1_valid = self.sample1.valid
                sample2_valid = self.sample2.valid
                
                overlapping = np.any(np.logical_and(sample1_valid, sample2_valid))
                if overlapping:
                    raise ValueError("The cell groups being compared are at least partially overlapping. "
                                     "Please choose different constraints for each sample!")
                
                labels = ['NA'] * self.cadata.adata.obs.shape[0]
                for i in range(len(labels)):
                    if sample1_valid[i]:
                        labels[i] = '1'
                    elif sample2_valid[i]:
                        labels[i] = '2'
                
                if self.name is None:
                    cadata = ConstrainedAdata(self.cadata.adata.copy())
                    for constraint in self.cadata.constraints:
                        cadata.add_constraint(constraint)
                else:
                    cadata = self.cadata
                
                cadata.adata.obs[de_key] = pd.Categorical(labels)
                
                cadata_copy = cadata.copy()
                cadata_copy.add_index_constraint(np.logical_or(sample1_valid, sample2_valid))
                                
                adata_sub = cadata_copy.adata_subset
                adata_sub.obs[de_key] = pd.Categorical(adata_sub.obs[de_key].to_list())
                
                self._parameters = adata_sub.uns[de_key]
                self._parameters['global_constraints'] = json.dumps(self.cadata.constraints)
                self._parameters['sample1_constraints'] = json.dumps(self.sample1.constraints)
                self._parameters['sample2_constraints'] = json.dumps(self.sample2.constraints)
                self._parameters['sample1_count'] = sum(sample1_valid)
                self._parameters['sample2_count'] = sum(sample2_valid)
                
                return adata_sub
            
            @property
            def results(self):
                de_key = self._de_key()
                if not self.force:
                    if self._de is not None:
                        return self._de
                    
                    elif self.name is not None and de_key in self.cadata.uns_keys():
                        return markers_to_df(self.cadata, de_key, include_groups='1', **self.kwargs)
                
                adata_sub = self._get_de_adata_sub()
                df = recalculate_markers(adata_sub, include_groups='1',
                                        groupby=de_key, key_added=de_key,
                                        reference='2', **self.kwargs)
                
                self._de = df
                
                if self.name is not None:
                    self.cadata.adata.uns[de_key] = self._parameters

                return df
            
            def results_pseudobulk(self, **kwargs):
                de_key = self._de_key()
                if not self.force:
                    if self._de_pseudobulk is not None:
                        return self._de_pseudobulk
                    
                    elif self.name is not None and de_key in self.cadata.uns_keys():
                        return markers_to_df(self.cadata, de_key, include_groups='1', **self.kwargs)
                
                adata_sub = self._get_de_adata_sub()
                df = de_pseudobulk(ConstrainedAdata(adata_sub), de_key, '1', '2', **kwargs)
                
                self._de_pseudobulk = df
                
                if self.name is not None:
                    self.cadata.adata.uns[de_key] = self._parameters

                return df
        return DEComplex(self, analysis_name, force=force, **kwargs)
        
    def de_volcano_plot(self,
                        sample_key, sample1, sample2,
                        subset_key=None,
                        group_key=None, groups=None,
                        ignore_constraints=False,
                        force=False,
                        **kwargs):
        df = self.de(sample_key, sample1, sample2,
                     subset_key=subset_key,
                     group_key=group_key, groups=groups,
                     ignore_constraints=ignore_constraints,
                     force=force)
        
        if 'name' not in df.columns:
            df['name'] = df.index
        
        return volcano_plot_from_df(df, **kwargs)

    @requires_gseapy
    def de_gsea(self, sample_key, sample1, sample2,
                gene_sets,
                subset_key=None,
                group_key=None, groups=None,
                ignore_constraints=False,
                **kwargs):
        de = self.de(sample_key, sample1, sample2,
                     subset_key=subset_key,
                     group_key=group_key, groups=groups,
                     ignore_constraints=ignore_constraints)

        rank_metric = (-np.log10(de['pval'])) / np.sign(de['log2fc'])
        de_prerank = pd.DataFrame.from_dict({
            'gene': [gene.upper() for gene in de.index],
            'metric': rank_metric,
        })
        de_prerank = de_prerank.sort_values('metric', ascending=False)

        pre_res = gseapy.prerank(de_prerank, gene_sets=gene_sets, **kwargs)

        results_dict = defaultdict(list)
        for key, result in pre_res.results.items():
            results_dict['term'].append(key)
            results_dict['es'].append(result['es'])
            results_dict['nes'].append(result['nes'])
            results_dict['pval'].append(result['pval'])
            results_dict['fdr'].append(result['fdr'])
            results_dict['size'].append(result['geneset_size'])
            results_dict['matched_size'].append(result['matched_size'])
            results_dict['genes'].append(result['genes'])
            results_dict['leading_edge'].append(result['ledge_genes'])
        results = pd.DataFrame(results_dict)
        results = results.reindex(results.nes.abs().sort_values(ascending=False).index)

        return pre_res, results

    @requires_gseapy
    def _enrichr(self, df, gene_sets, organism, output_folder,
                 padj_cutoff=0.01, log2fc_cutoff=1.5,
                 absolute_log2fc=False, make_plots=True,
                 enrichr_cutoff=1, max_attempts=3,
                 decription='test', log2_fold_change_field='log2fc',
                 padj_field='padj',
                 **kwargs):
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
        
        glist = [g.upper() for g in df_sub.index.str.strip().tolist()]
        bg = [g.upper() for g in df.index.str.strip().tolist()]
        
        logger.info(f"Remaining entries in gene list: {len(glist)}/{len(bg)}")
        
        success = False
        attempt = 0
        while not success:
            attempt += 1
            try:
                enr = gseapy.enrichr(gene_list=glist,
                                     gene_sets=gene_sets,
                                     organism=organism,
                                     description=decription,
                                     outdir=output_folder,
                                     no_plot=not make_plots,
                                     cutoff=enrichr_cutoff,
                                     background=bg,
                                     **kwargs)
                success = True
            except requests.exceptions.ConnectionError:
                if attempt > max_attempts:
                    raise
                else:
                    time.sleep(10)

        return enr

    def de_enrichr(self, sample_key, sample1, sample2,
                   gene_sets, organism, output_folder,
                   subset_key=None,
                   group_key=None, groups=None,
                   ignore_constraints=False,
                   padj_cutoff=0.01, log2fc_cutoff=1.5,
                   absolute_log2fc=False, make_plots=True,
                   enrichr_cutoff=1, max_attempts=3,
                   description='test',
                   **kwargs):
        de = self.de(sample_key, sample1, sample2,
                     subset_key=subset_key,
                     group_key=group_key, groups=groups,
                     ignore_constraints=ignore_constraints)
        
        return self._enrichr(de, gene_sets, organism, output_folder,
                             padj_cutoff=padj_cutoff, log2fc_cutoff=log2fc_cutoff,
                             absolute_log2fc=absolute_log2fc, make_plots=make_plots,
                             enrichr_cutoff=enrichr_cutoff, max_attempts=max_attempts,
                             description=description)

    def enrichr_markers(self, groupby, group,
                        gene_sets, organism, output_folder,
                        subset=None, force=False,
                        markers_key=None,
                        ignore_constraints=False,
                        padj_cutoff=0.01, log2fc_cutoff=1.5,
                        absolute_log2fc=False, make_plots=True,
                        enrichr_cutoff=1, max_attempts=3,
                        description='test',
                        **kwargs):
        if markers_key is None:
            markers_key = self._markers_key(groupby,
                                            subset_key=subset,
                                            groups=group)
        if not markers_key in self.adata.uns_keys() or force:
            self.markers(groupby=groupby, groups=group, subset=subset)
        
        markers = _markers_to_df(self.adata, markers_key, include_groups=group, **kwargs)
        #print(markers)
        markers_group = markers[markers['cluster'] == group]
        
        return self._enrichr(markers_group, gene_sets, organism, output_folder,
                             padj_cutoff=padj_cutoff, log2fc_cutoff=log2fc_cutoff,
                             absolute_log2fc=absolute_log2fc, make_plots=make_plots,
                             enrichr_cutoff=enrichr_cutoff, max_attempts=max_attempts,
                             description=description)

    def de_pathway_gene_list(self,
                             sample_key, sample1, sample2,
                             gene_sets,
                             subset_key=None,
                             group_key=None, groups=None,
                             ignore_constraints=False,
                             only_significant=False,
                             padj_cutoff=0.01,
                             output_file=None):
        de = self.de(sample_key, sample1, sample2,
                     subset_key=subset_key,
                     group_key=group_key, groups=groups,
                     ignore_constraints=ignore_constraints)

        # List DE genes in pathways
        all_genes = set(de.index.to_list())
        pathway_des = []
        for gene_set_name, genes in gene_sets.items():
            sane_genes = []
            for gene in genes:
                if gene in all_genes:
                    sane_genes.append(gene)

            de_sub = de.loc[sane_genes].copy()
            de_sub['pathway'] = gene_set_name
            de_sub.reset_index(inplace=True)
            pathway_des.append(de_sub)
        df = pd.concat(pathway_des)
        df = df[['pathway', 'gene', 'log2fc', 'pval', 'padj']]
        if only_significant:
            df = df[df['padj'] < padj_cutoff].copy()

        if output_file is not None:
            df.to_csv(output_file, sep="\t", index=False)

        return df

    def umap(self, colorby=None, **kwargs):
        return self.embedding_plot(colorby=colorby, embedding='umap', **kwargs)

    def tsne(self, colorby=None, **kwargs):
        return self.embedding_plot(colorby=colorby, embedding='tsne', **kwargs)

    def embedding_plot(self, colorby=None, embedding='umap', subset=None, **kwargs):
        if subset is not None:
            embedding = "{}__{}".format(self._subset_key(subset), embedding)
        return embedding_plot(self, colorby=colorby, key=embedding, **kwargs)
    
    def _base_key(self, group_key, groups=None,
                  subset_key=None):
        if group_key is not None:
            if groups is None:
                groups = self.obs[group_key].dtype.categories.to_list()
            elif isinstance(groups, string_types):
                groups = [groups]
            group_string = "__{}_{}".format(group_key, "_AND_".join([
                group.replace('-', '_')
                    .replace(' ', '_')
                    .replace("(", "_")
                    .replace(")", "_")
                    .replace("?", "_")
                    .lower() for group in sorted(groups)
            ]))
        else:
            group_string = ""

        base_key = '{subset}{group}'.format(
            subset=self._subset_key(subset_key) if subset_key is not None else "",
            group=group_string,
        )
        return base_key

    def _markers_key(self,
                     group_key, groups=None,
                     subset_key=None):
        base_key = self._base_key(group_key=group_key, groups=groups, subset_key=subset_key)
        return f'markers{base_key}'

    def markers(self, groupby, groups=None,
                subset=None, ignore_constraints=False,
                key_added=None, force=False, **kwargs):
        if key_added is None:
            markers_key = self._markers_key(groupby,
                                            subset_key=subset,
                                            groups=groups)
        else:
            markers_key = key_added

        if markers_key in self.uns_keys() and not force:
            return markers_to_df(self.adata, markers_key, **kwargs)

        if not ignore_constraints and self.has_constraints():
            raise ValueError("Pre-existings constraints ({}) will be ignored for DE analysis. "
                             "Please save these constraints as a subset and re-run this analysis "
                             "using the matching subset key!".format(len(self.constraints)))

        cadata = self.copy()
        if subset is not None:
            cadata.add_subset_constraint(subset)

        adata_sub = cadata.adata_subset
        df = recalculate_markers(adata_sub, groupby=groupby, key_added=markers_key, **kwargs)

        self.adata.uns[markers_key] = adata_sub.uns[markers_key]

        return df

    # def markers_plot(self, groupby, groups=None,
    #                  subset=None, force=False,
    #                  markers_key=None, **kwargs):
    #     if markers_key is None:
    #         markers_key = self._markers_key(groupby,
    #                                         subset_key=subset,
    #                                         groups=groups)
    #     if not markers_key in self.adata.uns_keys() or force:
    #         self.markers(groupby=groupby, groups=groups, subset=subset)

    #     return rank_genes_groups_heatmap(self.adata_subset, groupby=groupby, groups=groups,
    #                                      key=markers_key, **kwargs)
        
    def violin_plot(self, *args, **kwargs):
        return violin_plot(self, *args, **kwargs)

    def _cluster_key(self, resolution, algorithm='leiden', subset=None):
        return "{subset}__{algorithm}_{resolution}".format(
            subset="" if subset is None else self._subset_key(subset),
            algorithm=algorithm,
            resolution=resolution
        )
        
    def paga(self, group_key, groups=None, subset_key=None):
        pass
        
    def slingshot(self, **kwargs):
        return slingshot(self, **kwargs)


class SubAnnData(anndata.AnnData):
    def __init__(self, constraints=None, original_adata=None, *args, **kwargs):
        super(self).__init__(self, *args, **kwargs)
        self._contraints = constraints
        self._original_adata = original_adata