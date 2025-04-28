import collections
import os
import scanpy as sc
from anndata import AnnData
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import seaborn as sns
from collections import defaultdict
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.colors as mcol
import matplotlib.cm as cm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import zscore
from .plotting import barcode_from_embedding_plot, category_colors
from .r import slingshot as slingshot_r
from ._core import VData
from .helpers import find_cells_from_coords
from tqdm import tqdm
from itertools import cycle
import json
import scipy
import logging
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def paga(adata, groupby, pseudotime=True):
    sc.tl.paga(adata, groupby)
    if pseudotime:
        sc.tl.diffmap(adata)
        sc.tl.dpt(adata)
    return adata


def slingshot(
    vdata, 
    obs_key, 
    output_folder, 
    categories=None,
    pseudotime_key_prefix='slingshot',
    pseudotime_name_base='SlingShot pseudotime',
    pseudotime_description_base='SlingShot pseudotime, trajectory',
    save_to_adata=True,
    view_key=None,
    **kwargs
):
    vdata = vdata.copy(only_constraints=True)
    if categories is not None:
        vdata.add_categorical_constraint(obs_key, categories)
    else:
        categories = list(vdata.obs(view_key=view_key)[obs_key].unique())
    
    kwargs['add_obs'] = False
    slingshot_pseudotime_data, slingshot_lineages = slingshot_r(
        vdata, 
        obs_key, 
        output_folder,
        view_key=view_key,
        **kwargs
    )
    
    if save_to_adata:
        adata = vdata._parent_adata
        
        if 'pseudotime' not in adata.uns_keys():
            adata.uns['pseudotime'] = {}
        
        i = 1
        while f'slingPseudotime_{i}' in slingshot_pseudotime_data.columns:
            pseudotime_key = f'{pseudotime_key_prefix}_{i}'
            pseudotime_data = {
                'obs_key': pseudotime_key,
                'name': f'{pseudotime_name_base} {i}',
                'description': f'{pseudotime_description_base} {i}',
            }
            if categories is not None:
                pseudotime_data['categories'] = categories
            
            if view_key is not None:
                pseudotime_data['view_key'] = view_key
            
            adata.obs[pseudotime_key] = slingshot_pseudotime_data[f'slingPseudotime_{i}']
            adata.uns['pseudotime'][pseudotime_key] = pseudotime_data
            i += 1
        
        connectivities = np.zeros((len(categories), len(categories)))
        for lineage in slingshot_lineages:
            for i in range(1, len(lineage)):
                ix1 = categories.index(lineage[i-1])
                ix2 = categories.index(lineage[i])
                connectivities[ix1, ix2] = 1
                connectivities[ix2, ix1] = 1
        
        projection = kwargs.get('projection', 'X_umap')
        trajectories = adata.uns['trajectory'] if 'trajectory' in adata.uns_keys() else {}
        trajectories[pseudotime_key_prefix] = {
            'categories': categories,
            'connectivities': scipy.sparse.csr_matrix(connectivities),
            'description': f'{pseudotime_description_base}',
            'key': pseudotime_key_prefix,
            'name': f'{pseudotime_name_base}',
            'obs_constraints': json.dumps([c.to_dict() for c in vdata.obs_constraints]),
            'var_constraints': json.dumps([c.to_dict() for c in vdata.var_constraints]),
            'obs_key': obs_key if view_key is None else f'__view__{view_key}__{obs_key}',
            'obsm_key': projection if view_key is None else f'__view__{view_key}__{projection}',
            'type': 'SlingShot',
        }
        adata.uns['trajectory'] = trajectories

        return vdata
    else:
        return slingshot_pseudotime_data, slingshot_lineages


def dpt_pseudotime_from_subclusters(
    vdata, 
    obs_key=None, 
    categories=None, 
    starting_cell=None, 
    embedding='X_umap',
    dimensions=(0, 1),
    pseudotime_key='dpt',
    pseudotime_name=None,
    pseudotime_description=None,
    save_diffmap=False,
):
    vdata_sub = vdata.copy(only_constraints=True)
    if categories is not None:
        vdata_sub.add_categorical_obs_constraint(obs_key, categories)
    
    if starting_cell is None:
        starting_cell = barcode_from_embedding_plot(
            vdata_sub, 
            obsm_key=embedding, 
            colorby=obs_key,
            dimensions=dimensions,
        )
    elif isinstance(starting_cell, (list, tuple)):
        barcodes = find_cells_from_coords(
            vdata_sub, 
            embedding, 
            starting_cell[0],
            starting_cell[1],
            dimensions=dimensions
        )
        if len(barcodes) > 0:
            starting_cell = barcodes[0]
        else:
            raise ValueError("No cell found at coordinates.")
    
    adata_sub = vdata_sub.adata_view
    adata_sub.uns['iroot'] = adata_sub.obs.index.get_loc(starting_cell)

    sc.tl.diffmap(adata_sub)
    sc.tl.dpt(adata_sub)
        
    if 'pseudotime' in vdata._parent_adata.uns:
        pseudotime = vdata._parent_adata.uns['pseudotime']
    else:
        pseudotime = {}
    
    vdata._parent_adata.obs[pseudotime_key] = np.nan
    vdata._parent_adata.obs[pseudotime_key][adata_sub.obs.index] = adata_sub.obs['dpt_pseudotime']
    pseudotime[pseudotime_key] = {
        #'key': pseudotime_key,
        'dpt_starting_cell': starting_cell,
        'obs_key': pseudotime_key,
        'name': pseudotime_name or pseudotime_key,
        'description': pseudotime_description or pseudotime_key,
    }
    if categories is not None:
        pseudotime[pseudotime_key]['categories'] = categories
        pseudotime[pseudotime_key]['obs_key_categories'] = obs_key
    
    if save_diffmap:
        parent_ixs = [vdata._parent_adata.obs.index.get_loc(b) for b in adata_sub.obs.index]
        m = np.zeros((vdata._parent_adata.shape[0], adata_sub.obsm['X_diffmap'].shape[1]))
        m[parent_ixs] = adata_sub.obsm['X_diffmap']
        vdata._parent_adata.obsm[pseudotime_key] = m
        pseudotime[pseudotime_key] = {
            'diffmap_obsm_key': pseudotime_key,
        }
    
    vdata._parent_adata.uns['pseudotime'] = pseudotime
    
    return vdata


def pseudotime_binned_observations(
    vdata,
    pseudotime_key,
    var_keys=None,
    obs_keys=None,
    bins=200,
    window_size=200,
    layer=None,
):
    # subset data to non-nan pseudotime values
    vdata_sub = vdata.copy(only_constraints=True)
    pt_all = vdata_sub.obs_vector(pseudotime_key)
    valid_index = [ix for i, ix in enumerate(vdata_sub.obs.index) if not np.isnan(pt_all[i])]
    vdata_sub.add_index_obs_constraint(valid_index)
    
    # subset data to gene / var selection
    if var_keys is not None:
        vdata_sub.add_index_var_constraint(var_keys)
    
    # get valid pseudotime and order of observations
    pseudotime = vdata_sub.obs_vector(pseudotime_key)
    o = np.argsort(pseudotime)

    # get expression data
    x = vdata_sub.X if layer is None else vdata_sub.layers[layer]
    a = x.toarray()
    ex = a[o, :]
    pt = pseudotime[o]
    
    obs_data = dict()
    if obs_keys is not None:
        if isinstance(obs_keys, (str, bytes)):
            obs_keys = [obs_keys]
        for key in obs_keys:
            obs_data[key] = vdata_sub.obs_vector(key)[o]
    
    if bins is not None and len(pt) > 0:
        l = max(1, int((len(ex) - (len(ex)%(bins - 1))) / (bins - 1)))

        if window_size < l:
            warnings.warn(f"Window size ({window_size}) smaller than bin size ({l}). Disabling smoothing.")
            window_size = l
        
        obs_binned = defaultdict(list)
        ex_binned = []
        pt_binned = []
        ct_high = []
        ct_low = []
        for i in range(0, len(ex) - len(ex)%(bins-1), l):
            pt_binned.append(np.nanmean(pt[i:i+window_size]))
            
            m = np.nanmean(ex[i:i+window_size,], axis=0)
            ex_binned.append(m.tolist())
            sd = np.nanstd(ex[i:i+window_size,], axis=0)
            n = ex[i:i+window_size,].shape[0]
            ct_high.append([m[i] + 1.96 * sd[i]/np.sqrt(n) for i in range(len(m))])
            ct_low.append([m[i] - 1.96 * sd[i]/np.sqrt(n) for i in range(len(m))])
            
            for key, data in obs_data.items():
                obs_binned[key].append(max(set(data[i:i+window_size]), key=list(data[i:i+window_size]).count))
        
        # transform back to categorical
        for key, data in obs_data.items():
            if data.dtype.name == 'category':
                obs_binned[key] = pd.Categorical(obs_binned[key], categories=data.dtype.categories)
    else:
        ex_binned = ex
        pt_binned = pt
        ct_high = ex
        ct_low = ex
        obs_binned = obs_data
    
    obs_binned['pseudotime'] = pt_binned
    pseudotime_adata = AnnData(X=np.array(ex_binned),
                               obs=pd.DataFrame(obs_binned),
                               var=vdata_sub.var)
    pseudotime_adata.layers['confidence_low'] = np.array(ct_low)
    pseudotime_adata.layers['confidence_high'] = np.array(ct_high)
    
    return pseudotime_adata


def make_pseudotime_adata(cadata_with_pt, celltype_key, bins=200, 
                          window_size=200, additional_keys=None,
                          pseudotime_key='slingshot_pseudotime'):
    
    if additional_keys is None:
        additional_keys = []
    if isinstance(additional_keys, (str, bytes)):
        additional_keys = [additional_keys]
    additional_keys.append(celltype_key)
    
    pseudotime_adata = pseudotime_binned_observations(
        cadata_with_pt, 
        pseudotime_key,
        var_keys=None,
        obs_keys=additional_keys,
        bins=bins,
        window_size=window_size,
        layer=None
    )
    
    pseudotime_adata.obs['celltype'] = pseudotime_adata.obs[celltype_key].copy()
    pseudotime_adata.obs['color'] = [cadata_with_pt.default_colors.get(ct, '#aaaaaa') 
                                     for ct in pseudotime_adata.obs['celltype']]
    
    return pseudotime_adata


def pseudotime_polar_plot(cpseudotime_adata, gene, ax=None, fig=None):
    if ax is not None and ax.name != 'polar':
        raise ValueError("axis must be polar! (polar=True or projection='polar'")
    else:
        if fig is None:
            fig = plt.gcf()
        ax = fig.add_subplot(111, projection='polar')
    
    pt = cpseudotime_adata.obs['pseudotime']
    exp = cpseudotime_adata.adata_subset[:, gene].X.flatten()
    min_pt = min(pt)
    max_pt = max(pt)
    ct_high = cpseudotime_adata.adata_subset[:, gene].layers['confidence_high'].flatten()
    ct_low = cpseudotime_adata.adata_subset[:, gene].layers['confidence_low'].flatten()
    pt_radians = [2*np.pi*(p - min_pt) / max_pt for p in pt]
    
    # polar plot
    ax.fill_between(pt_radians, 
                    ct_high,
                    ct_low, 
                    color='#000000', alpha=0.05)
    ax.scatter(pt_radians, exp, color=cpseudotime_adata.obs['color'], s=10)
    #ax.set_ylim((0, np.max(ct_high)))
    ax.set_title(gene)
    ax.set_xlabel("Pseudotime (radial)")
    
    return ax


def pseudotime_linear_plot(
    pseudotime_vdatas, 
    gene, 
    obs_key_color=None,
    categories=None,
    colors=None,
    confidence=True, 
    confidence_color='#000000',
    confidence_alpha=0.05,
    z_transform=False,
    legend=None,
    legend_inside=True,
    ylim=None,
    pseudotime_key='pseudotime',
    confidence_high_key='confidence_high',
    confidence_low_key='confidence_low',
    markers=None,
    default_color='#777777',
    ax=None, 
    lax=None,
    legend_kwargs={},
    **scatter_kwargs
):
    ax = ax or plt.gca()
    if legend is None:
        legend = obs_key_color is not None
    scatter_kwargs.setdefault('s', 10)
    
    markers = markers or ['o', '+', 'v', 'd', 's', 'P', '*']
    markers = cycle(markers)
    
    if not isinstance(pseudotime_vdatas, dict):
        pseudotime_vdatas = {'__all__': pseudotime_vdatas}
    
    legend_elements = []
    
    line_markers = []
    for name in pseudotime_vdatas.keys():
        if not isinstance(pseudotime_vdatas[name], VData):
            pseudotime_vdatas[name] = VData(pseudotime_vdatas[name])
        marker = next(markers)
        line_markers.append(marker)
        
        legend_elements.append(
            Line2D(
                [0], [0], 
                marker=marker, 
                color='black', 
                label=name,
                markerfacecolor='black', 
                markersize=5,
                linestyle='none',
            ),
        )
    
    obs_colors = dict()
    if obs_key_color is not None:
        possible_categories = set()
        for pseudotime_vdata in pseudotime_vdatas.values():
            possible_categories = possible_categories.union(pseudotime_vdata.obs[obs_key_color].unique())
        
        if categories is not None:
            categories = set(categories).intersection(possible_categories)
        else:
            categories = possible_categories
        
        obs_colors = category_colors(categories, colors)
        for category in categories:
            legend_elements.append(
                matplotlib.patches.Patch(
                    facecolor=obs_colors[category], 
                    edgecolor=obs_colors[category],
                    label=category
                )
            )
    
    for i, pseudotime_vdata in enumerate(pseudotime_vdatas.values()):
        if obs_key_color is not None:
            pseudotime_vdata.add_categorical_obs_constraint(obs_key_color, list(categories))
        
        pt = pseudotime_vdata.obs[pseudotime_key]
        exp = np.array(pseudotime_vdata.adata_view[:, gene].X).flatten()
        if z_transform:
            exp = zscore(exp)
        if (confidence 
            and confidence_high_key in pseudotime_vdata.layers 
            and confidence_low_key in pseudotime_vdata.layers):
            ct_high = np.array(pseudotime_vdata.adata_view[:, gene].layers[confidence_high_key]).flatten()
            ct_low = np.array(pseudotime_vdata.adata_view[:, gene].layers[confidence_low_key]).flatten()
            
            if z_transform:
                ct_high = zscore(ct_high)
                ct_low = zscore(ct_low)
            
            ax.fill_between(
                pt, 
                ct_high,
                ct_low, 
                color=confidence_color, 
                alpha=confidence_alpha,
            )
    
        if obs_key_color is not None:
            colors = [obs_colors[c] for c in pseudotime_vdata.obs[obs_key_color]]
        elif 'color' in pseudotime_vdata.obs.columns:
            colors = pseudotime_vdata.obs['color']
        else:
            colors = [default_color] * len(pt)
        
        ax.scatter(pt, exp, color=colors, marker=line_markers[i], **scatter_kwargs)
    
    if legend and len(legend_elements) > 1:
        if lax is None:
            lax = ax
            lax_kwargs = dict(bbox_to_anchor=(1.01, 1), loc='upper left')
        else:
            lax_kwargs = dict(loc='center', labelspacing=2, frameon=False)
            lax.axis('off')

        lax_kwargs.update(legend_kwargs)
        if legend_inside:
            lax.legend(handles=legend_elements, **legend_kwargs)
        else:
            lax.legend(handles=legend_elements, **lax_kwargs)
    
    if z_transform:
        ax.set_ylabel('Norm. expression z-score (binned)')
    else:
        ax.set_ylabel('Norm. expression (binned)')
    #ax.set_ylim((0, np.max(ct_high)))
    ax.set_title(gene)
    ax.set_xlabel("Pseudotime")
    
    if ylim is not None:
        ax.set_ylim(ylim)

    return ax
