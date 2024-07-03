import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import matplotlib.colors as mcol

from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D

from matplotlib.backends.backend_pdf import PdfPages
import cmocean
import seaborn as sns
import pandas as pd
import numpy as np
from future.utils import string_types
import functools
import warnings
import re

from adjustText import adjust_text
import textalloc
from datetime import datetime
from itertools import groupby
from pandas.api.types import is_numeric_dtype
import time

from ..helpers import markers_to_df, find_cells_from_coords
from ..process import expression_data_frame

from .helpers import color_cycle, category_colors, get_numerical_and_annotation_columns

import logging
logger = logging.getLogger(__name__)

try:
    import polarity
    with_polarity = True
except (ModuleNotFoundError, OSError) as e:
    logger.debug("Cannot load polarity: {}".format(e))
    with_polarity = False


def scenic_regulon_enrichment_heatmap(regulon_enrichment_df, ax=None, **kwargs):
    return tf_activity_enrichment_heatmap(
        regulon_enrichment_df, 
        ax=ax, 
        tf_column='regulon', 
        **kwargs
    )


def tf_activity_enrichment_heatmap(
    tf_activity_enrichment_df, 
    ax=None, 
    tf_column='tf', 
    remove_directionality=False, 
    **kwargs
):
    if ax is None:
        ax = plt.gca()
        
    kwargs.setdefault('cmap', 'RdBu_r')
    kwargs.setdefault('linecolor', 'gray')
    kwargs.setdefault('annot_kws', {'size': 6})
    kwargs.setdefault('fmt', '.1f')
    kwargs.setdefault('linewidths', .7)
    kwargs.setdefault('cbar', False)
    kwargs.setdefault('annot', True)
    kwargs.setdefault('square', True)
    kwargs.setdefault('xticklabels', True)
    kwargs.setdefault('vmin', -1.5)
    kwargs.setdefault('vmax', 1.5)
    
    tf_activity_enrichment_df.loc[:, tf_column] = list(map(lambda s: s[:-3] if remove_directionality else s, 
                                                       tf_activity_enrichment_df[tf_column]))
    df_heatmap = pd.pivot_table(data=tf_activity_enrichment_df, index='group', columns=tf_column, values='Z')

    sns.heatmap(df_heatmap, ax=ax, **kwargs)
    ax.set_ylabel('')
    
    return ax


def scenic_binarised_plot_old(adata_sub, groups, figsize=(20, 90), colors=None, 
                              obsm_key=None, binarise=True, palette=None,
                              activity_support_cutoff=None, min_cells_active=0,
                              **kwargs):
    if obsm_key is None:
        obsm_key = 'X_aucell_mean' if not binarise else 'X_aucell_bin'
    
    data = pd.DataFrame(data=adata_sub.obsm[obsm_key], index=adata_sub.obs.index,
                        columns=adata_sub.uns['regulons'])
    
    return _scenic_binarised_plot_from_df(adata_sub, data, groups, figsize, colors=colors,
                                          binarise=binarise, palette=palette,
                                          activity_support_cutoff=activity_support_cutoff,
                                          min_cells_active=min_cells_active,
                                          **kwargs)


def scenic_binarised_plot(
    vdata, 
    groups, 
    figsize=(20, 90), 
    key_prefix='scenic',
    layer='aucell_positive_sum', 
    colors=None, 
    binarise=True, 
    palette=None,
    activity_support_cutoff=None, 
    min_cells_active=0,
    view_key=None,
    **kwargs
):
    if not layer.startswith(key_prefix):
        layer = f'{key_prefix}_{layer}'
    
    vdata_sub = vdata.copy(only_constraints=True)
    vdata_sub.add_index_var_constraint(
        [
            v for v, is_regulon in zip(
                vdata.var(view_key=view_key).index, 
                vdata.var(view_key=view_key)[f'{key_prefix}_is_regulon']
            )
            if is_regulon
        ]
    )
    
    data = pd.DataFrame(
        data=vdata_sub.layers(view_key=view_key)[layer].toarray(), 
        index=vdata_sub.obs(view_key=view_key).index,
        columns=vdata_sub.var(view_key=view_key).index
    )
    
    return _scenic_binarised_plot_from_df(vdata, data, groups, figsize, colors=colors,
                                          binarise=binarise, palette=palette,
                                          activity_support_cutoff=activity_support_cutoff,
                                          min_cells_active=min_cells_active,
                                          view_key=view_key,
                                          **kwargs)


def _scenic_binarised_plot_from_df(
    vdata,
    data, 
    groups, 
    figsize=(20, 90), 
    colors=None, 
    binarise=True, 
    palette=None,
    activity_support_cutoff=None, 
    min_cells_active=0,
    view_key=None,
    **kwargs
):
    if activity_support_cutoff is not None:
        valid_columns = []
        for column in data.columns:
            if np.sum(data[column] >= activity_support_cutoff) > min_cells_active:
                valid_columns.append(column)
        data = data[valid_columns]
    
    # N_COLORS = len(adata_sub.obs[group].dtype.categories)
    COLORS = [color['color'] for color in matplotlib.rcParams["axes.prop_cycle"]]

    group_colors = pd.DataFrame(index=vdata.obs(view_key=view_key).index)
    for group in groups:
        try:
            default_colors = vdata.default_colors(group)
        except KeyError:
            default_colors = None
        
        colors_dict = category_colors(
            vdata.obs(view_key=view_key)[group].dtype.categories,
            colors = None if colors is None else colors.get(group, None),
            default_colors=default_colors
        )

        cell_id2cell_type_lut = vdata.obs(view_key=view_key)[group].to_dict()
        group_colors[group] = data.index.map(cell_id2cell_type_lut).map(colors_dict)

    if palette is None:
        palette = sns.xkcd_palette(["white", "black"]) if binarise else 'Reds'

    sns.set()
    sns.set(font_scale=1.0)
    sns.set_style("ticks", {"xtick.minor.size": 1, "ytick.minor.size": 0.1})
    g = sns.clustermap(data.T,
                       col_colors=group_colors,
                       cmap=palette, figsize=figsize,
                       yticklabels=True, **kwargs)
    g.ax_heatmap.set_xticklabels([])
    g.ax_heatmap.set_xticks([])
    g.ax_heatmap.set_xlabel('Cells')
    g.ax_heatmap.set_ylabel('Regulons')

    g.cax.set_visible(False)
    return g.fig


def scenic_target_gene_support_histograms(
    vdata, 
    output_file, 
    key_prefix='scenic', 
    varm_key='target_gene_support_positive',
    view_key=None,
):
    if not varm_key.startswith(key_prefix):
        varm_key = f'{key_prefix}_{varm_key}'
    
    regulons = [
        v for v, is_regulon in zip(
            vdata.var(view_key=view_key).index, 
            vdata.var(view_key=view_key)[f'{key_prefix}_is_regulon']
        )
        if is_regulon
    ]
    target_gene_confidence = vdata.varm(view_key=view_key)[varm_key]
    n_runs = vdata.uns(view_key=view_key)['scenic'][key_prefix]['n_runs']
    
    with PdfPages(output_file) as pdf:
        for regulon in sorted(regulons):
            regulon_tg_support = pd.Series(
                target_gene_confidence[
                    :, 
                    vdata.var(view_key=view_key).index.get_loc(regulon)
                ].toarray().ravel(),
                index=vdata.var(view_key=view_key).index
            )
            
            fig, ax = plt.subplots(figsize=(10, 10))
            bins = list(range(1, n_runs + 2))
            ax.hist(regulon_tg_support[regulon_tg_support > 0], bins=bins, align='left')
            # ax.set_xticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
            # ax.set_xticklabels([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
            ax.set_xticks(list(range(1, n_runs + 1)))
            ax.set_xticklabels(list(range(1, n_runs + 1)))
            ax.set_ylabel("Number of (putative) target genes")
            ax.set_xlabel("Number of SCENIC runs supporting target gene")
            ax.set_title(regulon)
            pdf.savefig(fig)
            plt.close(fig)


def embedding_plot(
    adata, 
    key, 
    colorby=None, 
    groups=None, 
    groupby=None,
    splitby=None, 
    split_groups=None,
    colors=None, 
    shuffle=True, 
    shuffle_seed=42,
    ax=None, 
    cax=None, 
    fig=None, 
    n_cols=2,
    lax=None, 
    legend=True, 
    legend_inside=False, 
    groups_rename=None,
    simple_axes=True, 
    exclude_nan=False, 
    nan_color='#cccccc',
    legend_kwargs={}, 
    show_colorbar=True, 
    colorbar_title='', 
    label_groups=False,
    title=None, 
    view_key=None, 
    layer=None, 
    dimensions=(0, 1),
    **kwargs
):
    if fig is None:
        fig = plt.gcf()
    
    if splitby is not None:
        if split_groups is None:
            split_groups = adata.obs[splitby].dtype.categories
        
        if not isinstance(colorby, (list, tuple)):
            colorby = [colorby]

        max_xlim = None
        max_ylim = None
        axes = []
        if len(colorby) * len(split_groups) == 1:
            n_rows = 1
            n_cols = 1
        else:
            n_rows = max(1, int(np.ceil(len(colorby) * len(split_groups) / n_cols)))
        
        gs = GridSpec(n_rows, n_cols)
        plot_ix = 0
        for i, split_group in enumerate(split_groups):
            cadata_split = adata.copy(only_constraints=True)
            cadata_split.add_categorical_obs_constraint(splitby, split_group)
            
            for j, cb in enumerate(colorby):
                #plot_ix = (i * len(colorby) + j)
                #plot_ix = (j * len(splitby) + i)
                row = int(plot_ix / n_cols)
                col = int(plot_ix % n_cols)

                ax = fig.add_subplot(gs[row, col])
                embedding_plot(cadata_split, key, colorby=cb, groups=groups, groupby=groupby,
                               colors=colors, shuffle=shuffle, shuffle_seed=shuffle_seed,
                               ax=ax, cax=cax, fig=fig, n_cols=n_cols,
                               lax=lax, legend=legend, legend_inside=legend_inside,
                               groups_rename=groups_rename, simple_axes=simple_axes,
                               exclude_nan=exclude_nan, nan_color=nan_color,
                               legend_kwargs=legend_kwargs,
                               colorbar_title=colorbar_title, label_groups=label_groups,
                               view_key=view_key, show_colorbar=show_colorbar, layer=layer,
                               **kwargs)
                if title is None:
                    ax.set_title(f'{split_group} - {cb}')

                if max_xlim is None:
                    max_xlim = list(ax.get_xlim())
                else:
                    xlim = ax.get_xlim()
                    max_xlim[0] = min(max_xlim[0], xlim[0])
                    max_xlim[1] = max(max_xlim[1], xlim[1])

                if max_ylim is None:
                    max_ylim = list(ax.get_ylim())
                else:
                    ylim = ax.get_ylim()
                    max_ylim[0] = min(max_ylim[0], ylim[0])
                    max_ylim[1] = max(max_ylim[1], ylim[1])
                axes.append(ax)
                
                plot_ix += 1
        
        for ax in axes:
            ax.set_xlim(max_xlim)
            ax.set_ylim(max_ylim)
        return fig
    
    if isinstance(colorby, list) or isinstance(colorby, tuple):
        if len(colorby) == 1:
            n_cols = 1
            n_rows = 1
        else:
            n_rows = int(np.ceil(len(colorby)/n_cols))
        gs = GridSpec(n_rows, n_cols)
        for i, cb in enumerate(colorby):
            row = int(i / n_cols)
            col = int(i % n_cols)
            ax = fig.add_subplot(gs[row, col])
            embedding_plot(adata, key, colorby=cb, groups=groups, groupby=groupby,
                           colors=colors, shuffle=shuffle, shuffle_seed=shuffle_seed,
                           ax=ax, cax=cax, fig=fig, n_cols=n_cols,
                           lax=lax, legend=legend, legend_inside=legend_inside,
                           groups_rename=groups_rename, simple_axes=simple_axes,
                           exclude_nan=exclude_nan, nan_color=nan_color,
                           legend_kwargs=legend_kwargs,
                           colorbar_title=colorbar_title, label_groups=label_groups,
                           view_key=view_key, show_colorbar=show_colorbar, layer=layer,
                           **kwargs)
        return fig
    
    start = datetime.now()
    checkpoint = datetime.now()
    
    logger.debug("UMAP START: {}s".format((datetime.now() - checkpoint).total_seconds()))
    kwargs.setdefault('marker', '.')
    kwargs.setdefault('s', 10)
    kwargs.setdefault('linewidth', 0)
    
    if ax is None:
        ax = fig.gca()
    
    obs = adata.obs if view_key is None else adata.obs(view_key)
    obsm = adata.obsm if view_key is None else adata.obsm(view_key)
    
    try:
        if (isinstance(colorby, str) and colorby in obs.columns
            and obs[colorby].dtype.name == 'category'):
                groupby = colorby
                colorby = None
    except ValueError:
        pass

    if groups_rename is None:
        groups_rename = dict()

    if key not in obsm:
        key = 'X_{}'.format(key)
        
    logger.debug("UMAP PREP: {}s".format((datetime.now() - checkpoint).total_seconds()))
    checkpoint = datetime.now()

    coords = obsm[key]
    valid_ixs = np.isfinite(coords[:, 0])
    valid_group_info = None
    if groupby is not None:
        group_info = obs[groupby].to_numpy(dtype='str')
        if exclude_nan:
            valid_ixs = np.logical_and(valid_ixs, group_info != 'nan')
        else:
            group_info[group_info == 'nan'] = 'NA'
        valid_group_info = group_info[valid_ixs]

        all_groups = list(np.unique(valid_group_info))
        if groups is None:
            groups = all_groups
        elif isinstance(groups, string_types):
            groups = [groups]
        else:
            groups = list(groups)
        
        if not exclude_nan and 'NA' not in groups:
            groups.append('NA')

        valid_ixs = np.logical_and(valid_ixs, np.isin(group_info, list(groups)))
        valid_group_info = group_info[valid_ixs]
    valid_coords = coords[valid_ixs]

    logger.debug("UMAP COORDS: {}s".format((datetime.now() - checkpoint).total_seconds()))
    checkpoint = datetime.now()
    
    cmap = None
    legend_elements = None
    if colorby is None:
        if colors is None and groupby is not None:
            try:
                colors = adata.default_colors(groupby)
            except KeyError:
                pass

        if not isinstance(colors, dict):
            color_cycler = color_cycle(colors)
            dict_group_colors = dict()
            if groups is None:
                dict_group_colors['all'] = next(color_cycler)
            else:
                for group in groups:
                    dict_group_colors[group] = next(color_cycler)
            colors = dict_group_colors
        
        colors['NA'] = nan_color

        legend_elements = []
        if groups is None:
            color = colors.get('all', '#aaaaaa')
            valid_colors = [matplotlib.colors.to_rgba(color) for _ in range(valid_coords.shape[0])]
        else:
            valid_colors = [None for _ in range(valid_coords.shape[0])]
            legend_elements = []
            for group in groups:
                color = matplotlib.colors.to_rgba(colors.get(group, '#aaaaaa'))
                ixs = np.argwhere(valid_group_info == group)[:, 0]
                for ix in ixs:
                    valid_colors[ix] = color

                if len(ixs) > 0:
                    label = groups_rename.get(group, group)
                    legend_elements.append(matplotlib.patches.Patch(facecolor=color, edgecolor=color,
                                           label=label))
            if title is None and groupby is not None:
                ax.set_title(groupby.replace('_', ' '))
        valid_colors = np.array(valid_colors)
    else:
        logger.debug("UMAP START COLORBY: {}s".format((datetime.now() - checkpoint).total_seconds()))
        checkpoint = datetime.now()
        if colors is None:
            cmap = 'viridis'
        else:
            cmap = colors
        
        if isinstance(cmap, str):
            if cmap in cmocean.cm.cmapnames:
                cmap = getattr(cmocean.cm, cmap)
            else:
                cmap = matplotlib.colormaps[cmap]
        
        if not exclude_nan:
            cmap.set_bad(nan_color)

        if isinstance(colorby, string_types):
            try:
                checkpoint = datetime.now()
                try:
                    feature_values = adata.obs_vector(colorby, view_key=view_key, layer=layer)[valid_ixs]
                except TypeError:
                    feature_values = adata.obs_vector(colorby, layer=layer)[valid_ixs]
                #feature_values = adata.adata[:, colorby].X.toarray()[:, 0]
                logger.debug("UMAP END VALUES: {}s".format((datetime.now() - checkpoint).total_seconds()))
                checkpoint = datetime.now()
            except IndexError:
                raise ValueError("Feature '{}' not in AnnData object".format(colorby))
            
            if title is None:
                ax.set_title(colorby.replace('_', ' '))
        else:
            feature_values = colorby
        
        logger.debug("UMAP COLORBY FEATURE: {}s".format((datetime.now() - checkpoint).total_seconds()))
        checkpoint = datetime.now()

        valid_colors = np.array(feature_values)
        
        logger.debug("UMAP END COLORBY: {}s".format((datetime.now() - checkpoint).total_seconds()))
        checkpoint = datetime.now()

    if shuffle:
        ixs = np.arange(len(valid_coords))
        np.random.seed(shuffle_seed)
        np.random.shuffle(ixs)
        valid_coords = valid_coords[ixs]
        valid_colors = valid_colors[ixs]
        if valid_group_info is not None:
            valid_group_info = valid_group_info[ixs]
    
    logger.debug("UMAP SHUFFLE: {}s".format((datetime.now() - checkpoint).total_seconds()))
    checkpoint = datetime.now()

    plot = ax.scatter(
        valid_coords[:, dimensions[0]], 
        valid_coords[:, dimensions[1]], 
        c=valid_colors, 
        cmap=cmap, 
        plotnonfinite=not exclude_nan, 
        **kwargs
    )
    
    logger.debug("UMAP PLOT: {}s".format((datetime.now() - checkpoint).total_seconds()))
    checkpoint = datetime.now()

    if legend:
        if lax is None:
            lax = ax
            lax_kwargs = dict(bbox_to_anchor=(1.01, 1), loc='upper left')
        else:
            lax_kwargs = dict(loc='center', labelspacing=2, frameon=False)
            lax.axis('off')

        lax_kwargs.update(legend_kwargs)
        if groups is not None and legend_elements is not None:
            if legend_inside:
                lax.legend(handles=legend_elements, **legend_kwargs)
            else:
                lax.legend(handles=legend_elements, **lax_kwargs)
        elif colorby is not None and show_colorbar:
            if cax is not None:
                cb = plt.colorbar(plot, cax=cax)
            else:
                cb = plt.colorbar(plot, ax=ax)
            cb.set_label(colorbar_title)
            
    logger.debug("UMAP LEGEND: {}s".format((datetime.now() - checkpoint).total_seconds()))
    checkpoint = datetime.now()
    
    if groups is not None and label_groups:
        for group in groups:
            ixs = np.argwhere(valid_group_info == group)[:, 0]
            group_coords = valid_coords[ixs, :]
            label_coords = np.nanmean(group_coords, axis=0)
            ax.annotate(group, label_coords, ha='center')
    
    logger.debug("UMAP ANNOTATE: {}s".format((datetime.now() - checkpoint).total_seconds()))
    checkpoint = datetime.now()

    axis_label = key
    if axis_label.startswith('X_'):
        axis_label = axis_label[2:]
    m = re.match(r'(__subset__|__view__)(.+)__(X_)?(.+)', axis_label)
    if m is not None:
        axis_label = '{}'.format(m.group(4).upper())
    else:
        axis_label = axis_label.upper()
    #axis_label = key.upper() if not key.startswith('X_') else key[2:].upper()
    
    ax.set_xlabel("{}1".format(axis_label), loc='right')
    ax.set_ylabel("{}2".format(axis_label), loc='top')
    
    if simple_axes:
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
    sns.despine(ax=ax, top=True, right=True)
    
    logger.debug("UMAP LABEL: {}s".format((datetime.now() - checkpoint).total_seconds()))
    checkpoint = datetime.now()
    
    logger.debug("UMAP TOTAL: {}s".format((datetime.now() - start).total_seconds()))

    return ax


def _p_to_size(p, min_p=1e-3, base_size=40):
    p = max(min_p, p)
    if p == 1:
        return 10
    lp = -np.log10(p)
    size = (lp + 1) * base_size
    return size


def _lr_get_valid(gene_a_names, gene_b_names,
                  include_genes=None, include_regex=None, include_pairs=None, include_strict=False,
                  exclude_genes=None, exclude_regex=None, exclude_pairs=None, exclude_strict=False,):

    if include_genes is not None:
        valid_a = gene_a_names.isin(include_genes).to_numpy()
        valid_b = gene_b_names.isin(include_genes).to_numpy()
        if include_strict:
            valid_include_gene = np.logical_and(valid_a, valid_b)
        else:
            valid_include_gene = np.logical_or(valid_a, valid_b)
    else:
        valid_include_gene = np.repeat(True, len(gene_a_names))

    if include_regex is not None:
        if isinstance(include_regex, string_types):
            include_regex = [include_regex]

        valid_include_re = np.repeat(False, len(gene_a_names))
        for irx in include_regex:
            rx = re.compile(irx)
            valid_a = [rx.search(str(gene)) is not None for gene in gene_a_names]
            valid_b = [rx.search(str(gene)) is not None for gene in gene_b_names]

            if include_strict:
                local_valid = np.logical_and(valid_a, valid_b)
            else:
                local_valid = np.logical_or(valid_a, valid_b)
            valid_include_re = np.logical_or(local_valid, valid_include_re)
    else:
        valid_include_re = np.repeat(True, len(gene_a_names))

    valid_ix_order = None
    if include_pairs is not None:
        valid_ix_order = []
        pairs = list(zip(gene_a_names, gene_b_names))
        valid_include_pairs = np.repeat(False, len(gene_a_names))
        for pair in include_pairs:
            try:
                ix = pairs.index(pair)
                valid_include_pairs[ix] = True
                valid_ix_order.append(ix)
            except ValueError:
                try:
                    ix = pairs.index((pair[1], pair[0]))
                    valid_include_pairs[ix] = True
                    valid_ix_order.append(ix)
                except ValueError:
                    warnings.warn("Pair {} is not in list of pairs".format(pair))
    else:
        valid_include_pairs = np.repeat(True, len(gene_a_names))

    valid_include = functools.reduce(np.logical_and, [valid_include_gene, valid_include_re, valid_include_pairs])

    if exclude_genes is not None:
        valid_a = ~gene_a_names.isin(exclude_genes).to_numpy()
        valid_b = ~gene_b_names.isin(exclude_genes).to_numpy()
        if exclude_strict:
            valid_exclude_gene = np.logical_and(valid_a, valid_b)
        else:
            valid_exclude_gene = np.logical_or(valid_a, valid_b)
    else:
        valid_exclude_gene = np.repeat(True, len(gene_a_names))

    if exclude_regex is not None:
        if isinstance(exclude_regex, string_types):
            exclude_regex = [exclude_regex]

        valid_exclude_re = np.repeat(False, len(gene_a_names))
        for irx in exclude_regex:
            rx = re.compile(irx)
            valid_a = [rx.search(str(gene)) is not None for gene in gene_a_names]
            valid_b = [rx.search(str(gene)) is not None for gene in gene_b_names]

            if exclude_strict:
                local_valid = np.logical_and(valid_a, valid_b)
            else:
                local_valid = np.logical_or(valid_a, valid_b)
            valid_exclude_re = np.logical_or(local_valid, valid_exclude_re)
    else:
        valid_exclude_re = np.repeat(True, len(gene_a_names))

    if exclude_pairs is not None:
        pairs = list(zip(gene_a_names, gene_b_names))
        valid_exclude_pairs = np.repeat(True, len(gene_a_names))
        for pair in exclude_pairs:
            try:
                ix = pairs.index(pair)
                valid_exclude_pairs[ix] = False
            except ValueError:
                try:
                    ix = pairs.index((pair[1], pair[0]))
                    valid_exclude_pairs[ix] = False
                except ValueError:
                    warnings.warn("Pair {} is not in list of pairs".format(pair))
    else:
        valid_exclude_pairs = np.repeat(True, len(gene_a_names))

    valid_exclude = functools.reduce(np.logical_and, [valid_exclude_gene, valid_exclude_re, valid_exclude_pairs])
    # valid_exclude = np.logical_and(valid_exclude_gene, valid_exclude_re)

    valid = np.logical_and(valid_include, valid_exclude)

    if valid_ix_order is not None:
        valid_ixs = [ix for ix in valid_ix_order if valid[ix]]
    else:
        valid_ixs = [i for i, v in enumerate(valid) if v]
        
    return valid, valid_ixs


def cellphonedb_dot_plot(pvalues_files, means_files, sample_names=None,
                         include_genes=None, include_regex=None, include_pairs=None, include_strict=False,
                         exclude_genes=None, exclude_regex=None, exclude_pairs=None, exclude_strict=False,
                         include_celltype_pairs=None, ct_separator='|',
                         only_significant=False,
                         only_differential=False, pvalue_cutoff=0.05,
                         rl_separator="--", column_gap_size=2,
                         fig=None, cmap='jet', vmin=-10, vmax=5,
                         max_marker_size=30, row_height=.5, column_width=1,
                         invert_x=False):
    if not isinstance(pvalues_files, list) and not isinstance(pvalues_files, tuple):
        pvalues_files = [pvalues_files]

    if not isinstance(means_files, list) and not isinstance(means_files, tuple):
        means_files = [means_files]

    if len(pvalues_files) != len(means_files):
        raise ValueError("Must provide the same number of pvalues "
                         "({}) and means ({}) files!".format(len(pvalues_files), len(means_files)))

    if sample_names is None:
        sample_names = ['Sample-{}'.format(i) for i in range(len(pvalues_files))]
    elif len(sample_names) != len(pvalues_files):
        raise ValueError("Must provide the same number of samples "
                         "({}) and sample names ({}) files!".format(len(pvalues_files), len(sample_names)))

    pvalues_dfs = []
    means_dfs = []
    for i in range(len(pvalues_files)):
        if isinstance(pvalues_files[i], string_types):
            pvalues_df = pd.read_csv(pvalues_files[i], sep="\t")
        else:
            pvalues_df = pvalues_files[i]
        
        if isinstance(pvalues_files[i], string_types):
            means_df = pd.read_csv(means_files[i], sep="\t")
        else:
            means_df = means_files[i]

        # ensure same sort order
        m = {interaction_id: i for i, interaction_id in enumerate(pvalues_df['id_cp_interaction'])}
        o = [m[interaction_id] for interaction_id in means_df['id_cp_interaction']]
        means_df = means_df.iloc[o]

        gene_a_names = []
        for gene, partner in zip(pvalues_df['gene_a'], pvalues_df['partner_a']):
            try:
                partner_type, partner_name = partner.split(":")
            except ValueError:
                partner_type, partner_name1, partner_name2 = partner.split(":")
                partner_name = "{}-{}".format(partner_name1, partner_name2)
                
            if str(gene) == 'nan':
                gene_a_names.append(partner_name)
            else:
                gene_a_names.append(gene)
        gene_a_names = pd.Series(gene_a_names)

        gene_b_names = []
        for gene, partner in zip(pvalues_df['gene_b'], pvalues_df['partner_b']):
            try:
                partner_type, partner_name = partner.split(":")
            except ValueError:
                partner_type, partner_name1, partner_name2 = partner.split(":")
                partner_name = "{}-{}".format(partner_name1, partner_name2)
            
            if str(gene) == 'nan':
                gene_b_names.append(partner_name)
            else:
                gene_b_names.append(gene)
        gene_b_names = pd.Series(gene_b_names)

        valid, valid_ixs = _lr_get_valid(gene_a_names, gene_b_names,
                                         include_genes=include_genes, include_regex=include_regex,
                                         include_pairs=include_pairs, include_strict=include_strict,
                                         exclude_genes=exclude_genes, exclude_regex=exclude_regex,
                                         exclude_pairs=exclude_pairs, exclude_strict=exclude_strict)

        if np.sum(valid) == 0:
            raise ValueError("No valid RL pairs left")

        # do filtering for valid interactions
        pvalues_df = pvalues_df.iloc[valid_ixs]
        means_df = means_df.iloc[valid_ixs]

        pvalues_dfs.append(pvalues_df)
        means_dfs.append(means_df)

    common_interaction_ids = functools.reduce(np.intersect1d, [df['id_cp_interaction'].to_numpy()
                                                               for df in pvalues_dfs])
    common_celltype_pairs = functools.reduce(np.intersect1d, [df.columns.to_numpy()
                                                              for df in pvalues_dfs])

    if include_celltype_pairs is not None:
        include_celltype_pairs = ['{}{}{}'.format(p1, ct_separator, p2) for p1, p2 in include_celltype_pairs]
        not_pairs = []
        new_ct_pairs = []
        for ct in common_celltype_pairs:
            if not ct_separator in ct:
                not_pairs.append(ct)
        for ct in include_celltype_pairs:
            if ct in common_celltype_pairs:
                new_ct_pairs.append(ct)
        common_celltype_pairs = not_pairs + new_ct_pairs
                
    # ensure same sort order of rows and columns
    for i in range(len(pvalues_dfs)):
        m_rows = {interaction_id: i for i, interaction_id in enumerate(pvalues_dfs[i]['id_cp_interaction'])}
        o_rows = [m_rows[interaction_id] for interaction_id in common_interaction_ids]

        if i == 0:
            #o_rows_sorted = sorted(o_rows)
            #pvalues_dfs[i] = pvalues_dfs[i].iloc[o_rows_sorted]
            #means_dfs[i] = means_dfs[i].iloc[o_rows_sorted]
            common_interaction_ids = pvalues_dfs[i]['id_cp_interaction'].to_list()
            ctp = []
            for column in common_celltype_pairs:
                if column in pvalues_dfs[i].columns:
                    ctp.append(column)
            common_celltype_pairs = ctp

        pvalues_dfs[i] = pvalues_dfs[i].iloc[o_rows]
        means_dfs[i] = means_dfs[i].iloc[o_rows]
        pvalues_dfs[i] = pvalues_dfs[i][common_celltype_pairs]
        means_dfs[i] = means_dfs[i][common_celltype_pairs]

    # get only differential interactions
    if only_differential or only_significant:
        valid = np.repeat(False, pvalues_dfs[0].shape[0])
        for row_ix in range(pvalues_dfs[0].shape[0]):
            for col_ix in range(11, pvalues_dfs[0].shape[1]):
                sig, insig = 0, 0
                for i in range(len(pvalues_dfs)):
                    p = pvalues_dfs[i].iloc[row_ix, col_ix]
                    if p < pvalue_cutoff:
                        sig += 1
                    else:
                        insig += 1
                
                if only_differential and sig > 0 and insig > 0:
                    valid[row_ix] = True
                
                if only_significant and sig > 0:
                    valid[row_ix] = True

        for i in range(len(pvalues_dfs)):
            pvalues_dfs[i] = pvalues_dfs[i].iloc[valid]
            means_dfs[i] = means_dfs[i].iloc[valid]

    if not invert_x:  # yes, this is correct, because by default the scatter plot looks "inverted"
        for i in range(len(pvalues_dfs)):
            pvalues_dfs[i] = pvalues_dfs[i].iloc[::-1,]
            means_dfs[i] = means_dfs[i].iloc[::-1,]
    
    # prepare scatter names
    rl_names = []
    for gene_a, partner_a, gene_b, partner_b in zip(pvalues_dfs[0]['gene_a'], pvalues_dfs[0]['partner_a'],
                                                    pvalues_dfs[0]['gene_b'], pvalues_dfs[0]['partner_b']):
        if str(gene_a) == 'nan':
            try:
                partner_type, partner_name = partner_a.split(":")
            except ValueError:
                partner_type, partner_name1, partner_name2 = partner_a.split(":")
                partner_name = "{}-{}".format(partner_name1, partner_name2)
                
            gene_a = partner_name
        if str(gene_b) == 'nan':
            try:
                partner_type, partner_name = partner_b.split(":")
            except ValueError:
                partner_type, partner_name1, partner_name2 = partner_b.split(":")
                partner_name = "{}-{}".format(partner_name1, partner_name2)
                
            gene_b = partner_name
        rl_names.append(str(gene_a) + rl_separator + str(gene_b))

    ct_names = common_celltype_pairs[11:]

    # prepare scatter data
    xs, ys, sizes, colors = [], [], [], []
    for i in range(len(pvalues_dfs)):
        p = pvalues_dfs[i].iloc[:, 11:].to_numpy()
        m = means_dfs[i].iloc[:, 11:].to_numpy()

        for p_i in range(p.shape[0]):
            for p_j in range(p.shape[1]):
                xs.append(p_j * (len(pvalues_dfs) + column_gap_size) + i)
                ys.append(p_i)
                sizes.append(_p_to_size(p[p_i, p_j], base_size=max_marker_size))
                colors.append(np.log2(m[p_i, p_j]))

    if fig is None:
        figsize = (column_width*len(ct_names), row_height*len(rl_names))
        fig = plt.figure(figsize=figsize)

    n_rows = max(10, len(rl_names))
    gs = GridSpec(n_rows, 2, width_ratios=[10, 1])
    ax = plt.subplot(gs[:, 0])
    cb_size = min(int(n_rows/2), 5)
    cax = plt.subplot(gs[0:cb_size, 1])
    lax = plt.subplot(gs[cb_size+3:cb_size*2, 1])

    ax.scatter(xs, ys,
               c=colors, cmap='jet', vmin=vmin, vmax=vmax,
               s=sizes)
    ax.set_ylim((-1, len(rl_names)))
    ax.set_xlim((-column_gap_size, len(ct_names) * (len(pvalues_dfs) + column_gap_size) - 1))
    ax.set_yticks(np.arange(0, len(rl_names)))
    ax.set_yticklabels(rl_names)
    ax.set_xticks([i * (len(pvalues_dfs) + column_gap_size) + len(pvalues_dfs)/2 - .5 for i in range(len(ct_names))])
    ax.set_xticklabels(ct_names, rotation=90)

    # sample ticks
    if len(pvalues_files) > 1:
        sample_ax = ax.secondary_xaxis('top')
        sample_ax_ticks = []
        sample_ax_ticklabels = []
        step_size = len(pvalues_dfs) + column_gap_size
        for i in range(0, step_size*len(ct_names), step_size):
            sample_ax_ticks += [i + x for x in range(len(sample_names))]
            sample_ax_ticklabels += sample_names
        sample_ax.set_xticks(sample_ax_ticks)
        sample_ax.set_xticklabels(sample_ax_ticklabels, rotation=90)

    # colorbar
    if isinstance(cmap, string_types):
        cmap = getattr(matplotlib.cm, cmap)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                 cax=cax, orientation='vertical', label='log2 mean(R, L)')

    # legend
    legend_elements = []
    for p in [1e-3, 1e-2, 1e-1, 1]:
        e = plt.Line2D([0], [0], marker='o', color='black', label='{}'.format(p),
                       markerfacecolor='black', markersize=np.sqrt(_p_to_size(p, base_size=max_marker_size)),
                       linewidth=0)
        legend_elements.append(e)
    lax.legend(handles=legend_elements, loc='center', labelspacing=2, frameon=False)
    lax.axis('off')
    fig.subplots_adjust(left=0.2, bottom=0.3, right=0.9, top=0.8)

    return fig


def volcano_plot_from_df(
    df, plot_adjusted_pvalues=False,
    label_genes=None, 
    label_top_n=0, 
    min_pvalue=2.096468e-309,
    log2fc_cutoff=2, 
    padj_cutoff=1e-10,
    exclude_genes_prefix=None, 
    exclude_genes=None, 
    include_genes=None,
    insignificant_color='#eeeeee', 
    logfc_color='#fceade', 
    padj_color='#25ced1',
    both_color='#ff8a5b',
    label_line_color='#777777',
    label_top_only_significant=True,
    ignore_insignificant_large_z=True,
    hide_insignificant_large_z=True, 
    z_artifact_cutoff=15,
    x_symmetrical=True, 
    log2fc_field='log2fc', 
    pvalue_field='pval', 
    padj_field='padj',
    gene_name_field='name',
    label_size=7,
    adjust_labels=True, 
    adjust_labels_iterations=50, 
    adjust_labels_precision=0.01,
    xlim=None, 
    ylim=None, 
    ax=None, 
    **kwargs
):
    if ax is None:
        ax = plt.gca()

    if include_genes is not None:
        if isinstance(include_genes, string_types):
            include_genes = [include_genes]
        include_genes = set(include_genes)
        df = df.iloc[[gene in include_genes for gene in df[gene_name_field]]]

    if exclude_genes is not None:
        if isinstance(exclude_genes, string_types):
            exclude_genes = [exclude_genes]
        exclude_genes = set(exclude_genes)
        df = df.iloc[[gene not in exclude_genes for gene in df[gene_name_field]]]

    if exclude_genes_prefix is not None:
        if isinstance(exclude_genes_prefix, string_types):
            exclude_genes_prefix = [exclude_genes_prefix]
        exclude_genes_prefix = tuple(exclude_genes_prefix)
        df = df.iloc[[not gene.startswith(exclude_genes_prefix) for gene in df[gene_name_field]]]
    
    if hide_insignificant_large_z:
        df = df[(df[log2fc_field].abs() < z_artifact_cutoff).values | (df[padj_field] < padj_cutoff).values]

    df.loc[df[pvalue_field] == 0, pvalue_field] = min_pvalue
    df.loc[df[padj_field] == 0, padj_field] = min_pvalue

    if plot_adjusted_pvalues:
        pvalues_field = padj_field
    else:
        pvalues_field = pvalue_field

    # insignificant
    df_ins = df.loc[(df[log2fc_field].abs() < log2fc_cutoff) & (df[padj_field] > padj_cutoff)]
    ax.scatter(df_ins[log2fc_field], -1*np.log10(df_ins[pvalues_field]), color=insignificant_color, **kwargs)

    # log2fc > cutoff
    df_log2fc = df.loc[(df[log2fc_field].abs() >= log2fc_cutoff) & (df[padj_field] > padj_cutoff)]
    ax.scatter(df_log2fc[log2fc_field], -1*np.log10(df_log2fc[pvalues_field]), color=logfc_color, **kwargs)

    # padj < cutoff
    df_padj = df.loc[(df[log2fc_field].abs() < log2fc_cutoff) & (df[padj_field] <= padj_cutoff)]
    ax.scatter(df_padj[log2fc_field], -1*np.log10(df_padj[pvalues_field]), color=padj_color, **kwargs)

    # both
    df_both = df.loc[(df[log2fc_field].abs() >= log2fc_cutoff) & (df[padj_field] <= padj_cutoff)]
    ax.scatter(df_both[log2fc_field], -1*np.log10(df_both[pvalues_field]), color=both_color, **kwargs)

    # get labeled genes
    plot_labels = []
    plot_labels_xy = []
    if label_top_n > 0:
        for index, row in df.loc[df[log2fc_field].abs().sort_values(ascending=False).index].iterrows():
            if len(plot_labels) >= label_top_n:
                break
            #if label_top_only_significant and abs(row['log2fc']) < log2fc_cutoff:
            #    break
            if label_top_only_significant and row[padj_field] > padj_cutoff:
                continue
            plot_labels.append(row[gene_name_field])
            plot_labels_xy.append((row[log2fc_field], -1*np.log10(row[pvalues_field])))

    if label_genes is not None:
        for gene in label_genes:
            try:
                row = df.loc[gene]
            except KeyError:
                warnings.warn("Gene '{}' not found in AnnData (subset) or not expressed".format(gene))
                continue
            if row[gene_name_field] not in plot_labels:
                plot_labels.append(row[gene_name_field])
                plot_labels_xy.append((row[log2fc_field], -1*np.log10(row[pvalues_field])))

    if xlim is not None:
        x_min, x_max = xlim
    elif ignore_insignificant_large_z and df[df[padj_field] < padj_cutoff].shape[0] > 0:
        x_max = np.nanmax(df[df[padj_field] < padj_cutoff][log2fc_field]) * 1.1
        x_min = np.nanmin(df[df[padj_field] < padj_cutoff][log2fc_field]) * 1.1
    else:
        x_max = np.nanmax(df[log2fc_field]) * 1.1
        x_min = np.nanmin(df[log2fc_field]) * 1.1

    if x_symmetrical:
        max = np.max([abs(x_min), abs(x_max)])
        x_min, x_max = -max, max

    ax.set_xlim((x_min, x_max))

    if ylim is not None:
        ax.set_ylim(ylim)

    texts = []
    texts_str = []
    for i, label in enumerate(plot_labels):
        #text = ax.annotate(label, plot_labels_xy[i], fontsize=label_size)
        #texts.append(text)
        texts_str.append(label)
    
    if adjust_labels and len(plot_labels_xy) > 0:
        start = datetime.now()
        # adjust_text(
        #     texts, 
        #     arrowprops=dict(arrowstyle='-', color='grey'), 
        #     lim=adjust_labels_iterations,
        #     precision=adjust_labels_precision,
        #     ax=ax
        # )
        
        xy = np.array(plot_labels_xy)
        textalloc.allocate_text(
            ax.figure,
            ax,
            xy[:, 0],
            xy[:, 1],
            texts_str,
            # x_scatter=xy[:, 0],
            # y_scatter=xy[:, 1],
            max_distance=0.2,
            min_distance=0.01,
            margin=0.0009,
            # linewidth=0.5,
            linecolor=label_line_color,
            nbr_candidates=400,
            textsize=label_size,
        )
        end = datetime.now()
        print(f'took: {(end - start).total_seconds()}')

    ax.set_xlabel("Log2-fold change expression")
    ax.set_ylabel("-log10(adjusted p-value)" if plot_adjusted_pvalues else "-log10(p-value)")

    return ax


def volcano_plot(adata, key, group, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()

    kwargs.setdefault('alpha', 0.6)
    kwargs.setdefault('marker', '.')
    kwargs.setdefault('linewidth', 0)
    kwargs.setdefault('s', 50)

    df = markers_to_df(adata, key, include_groups=group)

    return volcano_plot_from_df(df, **kwargs)


def violin_plot_data(
    vdata, 
    feature,
    groupby=None,
    groups=None,
    groups_rename=None,
    ignore_groups=('NA',),
    splitby=None, 
    split_groups=None, 
    split_rename=None, 
    value_key=None,
    layer=None, 
):
    vdata = vdata.copy(only_constraints=True)
    
    value_key = value_key or feature
    
    data = pd.DataFrame.from_dict({
        value_key: vdata.obs_vector(feature, layer=layer),
    })
    
    if groupby is not None:
        data['group'] = pd.Categorical(vdata.obs[groupby])
    
    if splitby is not None:
        data['split'] = pd.Categorical(vdata.obs[splitby])
    
    if ignore_groups is not None and 'group' in data:
        data = data.loc[~data['group'].isin(ignore_groups), :]
    
    if groups is not None and 'group' in data:
        data = data.loc[data['group'].isin(groups), :]
    
    if split_groups is not None and 'split' in data:
        data = data.loc[data['split'].isin(split_groups), :]
    
    if groupby is not None:
        groups = groups or pd.Categorical(data['group'].to_list()).dtype.categories
    
    if splitby is not None:
        split_groups = split_groups or pd.Categorical(data['split'].to_list()).dtype.categories
    
    if groups_rename is not None:
        data['group'] = pd.Categorical(
            [groups_rename.get(g, g) for g in data['group']],
            categories=[groups_rename[g] for g in groups]
        )
    else:
        if 'group' in data:
            data['group'] = pd.Categorical(data['group'], categories=groups)
    
    if split_rename is not None:
        data['split'] = pd.Categorical(
            [split_rename.get(g, g) for g in data['split']],
            categories=[split_rename[g] for g in split_groups]
        )
    else:
        if 'split' in data:
            data['split'] = pd.Categorical(data['split'], categories=split_groups)
    
    return data


def violin_plot_from_df(
    df,
    colors=None,
    scale='width',
    legend=False, 
    legend_inside=True,
    ylim=None, 
    swarm=True, 
    points_size=2, 
    points_color='black', 
    points_alpha=0.4, 
    points_jitter=0.25,
    shuffle_seed=42,
    ax=None, 
    lax=None, 
    legend_kwargs={},
    value_key=None,
    group_key='group',
    split_key='split',
    **kwargs
):
    ax = ax or plt.gca()
    #lax = lax or ax
    value_key = value_key or df.columns[0]
    
    x = None
    order = None
    hue = None
    hue_order = None
    palette = None
    legend_elements = []
    legend_order = None
    
    if group_key in df:
        x = group_key
        order = df[group_key].dtype.categories
        palette = category_colors(order, colors)
        #palette = [palette_dict.get(category, '#777777') for category in order]
        legend_order = order
    
    if split_key in df:
        hue = split_key
        hue_order = list(df[split_key].dtype.categories)
        palette = category_colors(hue_order, colors)
        legend_order = hue_order
        if x is None:
            df = df.copy()
            df['__x'] = ""
            x = '__x'
        split = True
    else:
        split=False

    ax = sns.violinplot(
        x=x,
        y=value_key,
        hue=hue, 
        scale=scale,
        data=df, 
        split=split, 
        order=order,
        hue_order=hue_order,
        palette=palette, 
        ax=ax, 
        #**kwargs
    )
    if swarm:
        np.random.seed(shuffle_seed)
        ax = sns.stripplot(
            x=x, 
            y=value_key, 
            hue=hue,
            data=df, 
            dodge=True, 
            order=order, 
            hue_order=hue_order,
            color=points_color, 
            alpha=points_alpha,
            jitter=points_jitter,
            size=points_size,
            ax=ax, 
            **kwargs
        )
    
    ax.set_xticklabels([label.get_text() for label in ax.get_xticklabels()], rotation=90)
    ax.set_xlabel("")
    ax.set_ylabel(value_key)

    if ylim is not None:
        ax.set_ylim(ylim)
    
    # legend
    if legend_order is not None:
        for category in legend_order:
            legend_elements.append(
                matplotlib.patches.Patch(
                    facecolor=palette[category], 
                    edgecolor=palette[category],
                    label=category
                )
            )
            
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
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return ax


def violin_plot(cadata, feature,
                groupby=None, groups=None, groups_rename={}, colors=None,
                splitby=None, split_groups=None, split_rename={}, 
                scale='width', 
                legend=False, legend_inside=True,
                ylim=None, swarm=True, 
                ignore_groups=('NA',),
                layer=None, 
                points_size=2, points_color='black', points_alpha=0.4, 
                points_jitter=0.25,
                shuffle_seed=42,
                ax=None, lax=None, legend_kwargs={},
                **kwargs):
    if ax is None:
        ax = plt.gca()
    
    cadata = cadata.copy(only_constraints=True)

    if groupby is not None:
        if groups is None:
            groups = cadata.obs[groupby].dtype.categories
        else:
            cadata.add_categorical_obs_constraint(groupby, groups)
        groups = [groups_rename.get(g, g) for g in groups if g not in ignore_groups]

        if splitby is None:
            if colors is None:
                try:
                    colors = cadata.default_colors(groupby)
                except KeyError:
                    pass
            
            if not isinstance(colors, dict):
                colors = color_cycle(colors)
                dict_group_colors = dict()
                for group in groups:
                    dict_group_colors[group] = matplotlib.colors.to_rgba(next(colors))
                colors = dict_group_colors
            elif groups_rename is not None:
                new_group_colors = {}
                for group, color in colors.items():
                    new_group_colors[groups_rename.get(group, group)] = color
                colors = new_group_colors

    split_rename_inverse = {v: k for k, v in split_rename.items()}
    if splitby is not None:
        if split_groups is None:
            split_groups = cadata.obs[splitby].dtype.categories
        else:
            cadata.add_categorical_obs_constraint(splitby, split_groups)
            
        if len(split_groups) != 2:
            raise ValueError(f"Can only split if the number of split_groups ({len(split_groups)}) is exactly 2!")
        
        split_groups = [split_rename.get(s, s) for s in split_groups]

        if colors is None:
            try:
                colors = cadata.default_colors(splitby)
            except KeyError:
                pass
        
        if not isinstance(colors, dict):
            colors = color_cycle(colors)
            dict_split_colors = dict()
            for split in split_groups:
                dict_split_colors[split] = matplotlib.colors.to_rgba(next(colors))
            colors = dict_split_colors
        
        for g in split_groups:
            if g in split_rename_inverse:
                colors[g] = colors[split_rename_inverse[g]]
        
    #sub_adata = cadata.adata_subset

    scale = kwargs.pop('scale', 'width')
    swarm = kwargs.get('inner') == 'swarm' if swarm is None else swarm
    if kwargs.get('inner') == 'swarm':
        kwargs.pop('inner')

    values = cadata.obs_vector(feature, layer=layer)
    if splitby is not None and groupby is not None:
        df_groups = [groups_rename.get(g, g) for g in cadata.obs[groupby]]
        df_splits = [split_rename.get(s, s) for s in cadata.obs[splitby]]
        df = pd.DataFrame.from_dict({
            'value': values,
            'group': df_groups,
            'split': df_splits,
        })
        ax = sns.violinplot(x="group", y="value", hue='split', scale=scale,
                            data=df, split=True, order=groups,
                            hue_order=split_groups,
                            palette=colors, ax=ax, **kwargs)
        if swarm:
            np.random.seed(shuffle_seed)
            ax = sns.stripplot(x="group", y="value", hue='split',
                               data=df, dodge=True, order=groups, hue_order=split_groups,
                               color=points_color, 
                               alpha=points_alpha,
                               jitter=points_jitter,
                               #palette=split_colors, 
                               size=points_size,
                               ax=ax, **kwargs)

    elif groupby is not None:
        df = pd.DataFrame.from_dict({
            'value': values,
            'group': [groups_rename.get(g, g) for g in cadata.obs[groupby]],
        })
        #print([group_colors[g] for g in groups])

        ax = sns.violinplot(x="group", y="value",
                            #hue='group',
                            scale=scale,
                            data=df, split=False, order=groups,
                            color=points_color, alpha=points_alpha,
                            jitter=points_jitter,
                            palette=colors, ax=ax,
                            **kwargs)
        
        if swarm:
            np.random.seed(shuffle_seed)
            ax = sns.stripplot(x="group", y="value",
                               data=df, order=groups,
                               color=points_color, alpha=points_alpha,
                               jitter=points_jitter,
                               #palette=group_colors, 
                               ax=ax,
                               size=points_size,
                               **kwargs)
    elif splitby is not None:
        df = pd.DataFrame.from_dict({
            'group': ['all'] * len(values),
            'value': values,
            'split': [split_rename.get(s, s) for s in cadata.obs[splitby]]
        })
        ax = sns.violinplot(x='group', y="value", hue='split', scale=scale,
                            data=df, split=True, hue_order=split_groups,
                            palette=colors, ax=ax, **kwargs)
        if swarm:
            np.random.seed(shuffle_seed)
            ax = sns.stripplot(x="group", y="value", hue='split',
                               dodge=True,
                               data=df, order=groups, hue_order=split_groups,
                               color=points_color, alpha=points_alpha,
                               jitter=points_jitter,
                               #palette=group_colors, 
                               ax=ax,
                               size=points_size,
                               **kwargs)
    else:
        df = pd.DataFrame.from_dict({
            'group': ['all'] * len(values),
            'value': values,
        })
        ax = sns.violinplot(y="value", scale=scale,
                            data=df, split=False, ax=ax, **kwargs)
        
        if swarm:
            np.random.seed(shuffle_seed)
            ax = sns.stripplot(x="group", y="value",
                               data=df,
                               color=points_color, alpha=points_alpha,
                               jitter=points_jitter,
                               ax=ax,
                               size=points_size,
                               **kwargs)
    ax.set_ylim((np.min(values), np.max(values)))

    if legend:
        handles, labels = ax.get_legend_handles_labels()
        
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        
        if splitby is not None and swarm:
            handles = handles[0:int(len(handles)/2)]
            labels = labels[0:int(len(labels)/2)]
        
        if lax is None:
            lax = ax
            lax_kwargs = dict(bbox_to_anchor=(1.01, 1), loc='upper left')
        else:
            lax_kwargs = dict(loc='center', labelspacing=2, frameon=False)
            lax.axis('off')

        lax_kwargs.update(legend_kwargs)
        if legend_inside:
            print("adding legend inside", handles, labels)
            lax.legend(handles, labels, **legend_kwargs)
        else:
            print("adding legend", handles, labels)
            lax.legend(handles, labels, **lax_kwargs)
    elif ax.get_legend() is not None:
        ax.get_legend().remove()
    # elif splitby is not None and swarm:
    #     handles, labels = ax.get_legend_handles_labels()
    #     ax.legend(handles[0:int(len(handles)/2)], labels[0:int(len(handles)/2)], 
    #                 bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    ax.set_xticklabels([label.get_text() for label in ax.get_xticklabels()], rotation=90)
    ax.set_xlabel("")
    ax.set_ylabel(feature)

    if ylim is not None:
        ax.set_ylim(ylim)

    return ax

#
# CPDB dot plot
#
def _p_to_size(p, min_p=1e-3, max_p=0.1, min_size=1, base_size=40, scale=1., _verbose=False):
    p = max(min_p, p)
    if p > max_p:
        return min_size
    lp = -np.log10(p)
    size = (lp + 1) ** scale * base_size
    if _verbose:
        print(f'p: {p}, lp: {lp}, size: {size}')
    return size


def cpdb_dot_plot_dfs(
    dfs, 
    filter_dfs=None,
    base_marker_size=10, 
    marker_scale=2, 
    min_pvalue=1e-3, 
    max_pvalue=0.1,
    lr_sep='--',
    ct_sep='--',
):
    if not isinstance(dfs, (tuple, list)):
        dfs = [dfs]
    
    df_dots = []
    for df in dfs:

        df_dot = pd.DataFrame({
            'lr': [f'{l}{lr_sep}{r}' for l, r in zip(df['ligand'], df['receptor'])],
            'ct': [f'{lct}{ct_sep}{rct}' for lct, rct in zip(df['ligand_celltype'], df['receptor_celltype'])],
            'pvalue': df['pvalue'],
            'inv_pvalue': 1-df['pvalue'],
            'mean': df['mean'],
            'size': [_p_to_size(p, base_size=base_marker_size, scale=marker_scale,
                                min_p=min_pvalue, max_p=max_pvalue) 
                    for p in df['pvalue']]
        })
        
        if filter_dfs is not None:
            if not isinstance(filter_dfs, (list, tuple)):
                if isinstance(filter_dfs, dict):
                    filter_dfs = list(filter_dfs.values())
                else:
                    filter_dfs = [filter_dfs]
            
            valid_lr = set()
            valid_ct = set()
            for fdf in filter_dfs:
                valid_lr = valid_lr.union(set([f'{l}{lr_sep}{r}' for l, r in zip(fdf['ligand'], fdf['receptor'])]))
                valid_ct = valid_ct.union(set([f'{lct}{ct_sep}{rct}' 
                                for lct, rct in zip(fdf['ligand_celltype'], fdf['receptor_celltype'])]))
            df_dot = df_dot.loc[np.logical_and(df_dot['lr'].isin(valid_lr), df_dot['ct'].isin(valid_ct)), :]
        df_dots.append(df_dot)

    return df_dots


def cpdb_dot_plot(df, filter_df=None,
                  base_marker_size=10, marker_scale=2, legend_size=0.03,
                  min_pvalue=1e-3, max_pvalue=0.1,
                  min_mean=None, max_mean=None,
                  ax=None, lax=None, cax=None, 
                  lrs=None, cts=None,
                  **kwargs):
    ax = ax or plt.gca()
    
    if not isinstance(df, dict):
        dfs = {'__all__': df}
    else:
        dfs = df
    
    df_dots_list = cpdb_dot_plot_dfs(
        [dfs[name] for name in dfs.keys()], 
        filter_dfs=filter_df,
        base_marker_size=base_marker_size, 
        marker_scale=marker_scale, 
        min_pvalue=min_pvalue, 
        max_pvalue=max_pvalue,
    )
    df_dots = {name: df_dots_list[i] for i, name in enumerate(dfs.keys())}
    
    return cpdb_dot_plot_from_prepared(
        df_dots, 
        base_marker_size=base_marker_size, 
        marker_scale=marker_scale, 
        legend_size=legend_size,
        min_pvalue=min_pvalue, 
        max_pvalue=max_pvalue,
        min_mean=min_mean, 
        max_mean=max_mean,
        lrs=lrs,
        cts=cts,
        ax=ax, 
        lax=lax, 
        cax=cax, 
        **kwargs
    )


def cpdb_dot_plot_from_prepared(
    df_dots, 
    base_marker_size=10, 
    marker_scale=2, 
    legend_size=0.03,
    min_pvalue=1e-3, 
    max_pvalue=0.1,
    min_mean=None, 
    max_mean=None,
    ax=None, 
    lax=None, 
    cax=None,
    lrs=None,
    cts=None,
    **kwargs
):
    ax = ax or plt.gca()
    
    if lrs is None or cts is None:
        lrs_all = []
        cts_all = []
        for df_dot in df_dots.values():
            lrs_all += df_dot['lr'].to_list()
            cts_all += df_dot['ct'].to_list()
        lrs = lrs or sorted(np.unique(lrs_all))
        cts = cts or sorted(np.unique(cts_all))
    
    lr_ixs = {lr: i for i, lr in enumerate(lrs)}
    ct_ixs = {ct: i for i, ct in enumerate(cts)}
    
    n_dfs = len(df_dots)
    offset = {0: 0}
    if n_dfs == 2:
        offset = {0: -0.15, 1: 0.15}
    elif n_dfs == 3:
        offset = {0: -0.3, 1: 0, 2: 0.3}
    elif n_dfs == 4:
        offset = {0: -0.3, 1: -0.1, 2: 0.1, 3: 0.3}
    elif n_dfs > 4:
        raise ValueError("Cannot currently compare more than 4 samples.")
    
    for i, (name, df_dot) in enumerate(df_dots.items()):
        x = [ct_ixs[ct] + offset[i] for ct in df_dot['ct']]
        y = [lr_ixs[lr] for lr in df_dot['lr']]
        
        kwargs.setdefault('vmin', min_mean)
        kwargs.setdefault('vmax', max_mean)
        plot_data = ax.scatter(x, y, s=df_dot['size'], c=df_dot['mean'], **kwargs)
        
        if n_dfs > 1:
            labels = [name] * len(x)
            secax = ax.secondary_xaxis('top')
            secax.set_xticks(x)
            secax.set_xticklabels(labels, rotation=90)
    
    ax.set_xticks(range(len(cts)))
    ax.set_yticks(range(len(lrs)))
    ax.set_xticklabels(cts)
    ax.set_yticklabels(lrs)
    ax.set_ylim((-1, len(lrs)))
    ax.set_xlim((-1, len(cts)))
    # default_size_norm = matplotlib.colors.Normalize(vmin=0.9, vmax=0.999)
    # kwargs.setdefault('size_norm', default_size_norm)
    # sns.scatterplot(data=df_dot, x='ct', y='lr', size='size', hue='mean', ax=ax, **kwargs)
    ax.tick_params(axis='x', labelrotation = 90)
    
    # colorbar
    cax = cax or ax.inset_axes([1 + legend_size, 0.5, legend_size, 0.5])
    cb = plt.colorbar(plot_data, ax=ax, cax=cax)
    cb.set_label('Mean L+R\nexpression')
    # legend
    lax = lax or ax.inset_axes([1 + legend_size * 1.7, 0 + legend_size, legend_size, 0.5])
    legend_elements = []
    for p in [1e-3, 1e-2, 1e-1, 1]:
        e = plt.Line2D([0], [0], marker='o', color='black', label='{}'.format(p),
                       markerfacecolor='black', 
                       markersize=np.sqrt(_p_to_size(p, base_size=base_marker_size, 
                                                     scale=marker_scale, 
                                                     min_p=min_pvalue, max_p=max_pvalue,
                                                     _verbose=True)),
                       linewidth=0)
        legend_elements.append(e)
    lax.legend(handles=legend_elements, loc='upper left', labelspacing=2, frameon=False,
               title='P-value')
    lax.axis('off')
    
    return ax


#
# LR plot Circos-style
#
def cpdb_ligand_receptor_circos(cpdb_df, celltypes=None, colors=None, 
                                label_size='xx-small', 
                                plot_gene_names=False,
                                plot_pathway_labels=True,
                                group_by_pathway=True,
                                celltype_label_offset=3.0, pathway_label_offset=1.5,
                                celltype_label_horizontal=True,
                                gene_label_offset=1.5, ylim=(0, 15),
                                inner_radius=10,
                                alpha_min=0.2, alpha_max=0.9,
                                invert_opacity=None,
                                link_opacity_key='mean',
                                link_opacity_value_range=None,
                                link_opacity_value_percentiles=(5, 95),
                                link_color_key=None, link_colors=None, 
                                link_color_value_range=None,
                                link_color_value_percentiles=(0, 100),
                                legend=True, legend_size=0.05, legend_label_size=8,
                                ax=None, fig=None, lax=None,):
    if not with_polarity:
        raise ValueError("Cannot plot circos, please install polarity first.")
    
    if link_opacity_key is not None and link_opacity_key.lower() == 'none':
        link_opacity_key = None
    
    if link_color_key is not None and link_color_key.lower() == 'none':
        link_color_key = None
    
    if invert_opacity is None and link_opacity_key == 'pvalue':
        invert_opacity = True
    
    # remove NaNs
    cpdb_df = cpdb_df.copy()
    if not group_by_pathway:
        cpdb_df['pathway'] = 'NA'
        plot_pathway_labels = False
    else:
        cpdb_df.loc[cpdb_df['pathway'].isna(), 'pathway'] = ''
    
    if isinstance(plot_gene_names, set):
        pass
    elif isinstance(plot_gene_names, list) or isinstance(plot_gene_names, tuple):
        plot_gene_names = set(plot_gene_names)  
    elif not plot_gene_names:
        plot_gene_names = {}
    elif plot_gene_names:
        plot_gene_names = set(cpdb_df['ligand'].to_list()).union(set(cpdb_df['receptor'].to_list()))
        
    pol = polarity.Polarity(inner_radius=inner_radius)
    # set up segments
    if celltypes is None:
        celltypes = sorted(list(set(cpdb_df['ligand_celltype'].unique()).union(set(cpdb_df['receptor_celltype'].unique()))))
    
    if not isinstance(colors, dict):
        cycle_colors = color_cycle(colors)
        colors = {ct: next(cycle_colors) for ct in celltypes}
    
    ct_segments = dict()
    pathway_segments = dict()
    gene_segments = dict()
    for ct in celltypes:
        ct_segment = polarity.Segment(data_span=(0, 1))
        pol.add_segment(ct_segment)
        ct_segment_label = polarity.LabelSegment(label=ct, size=ct_segment, 
                                                 radius_offset=celltype_label_offset, fontweight='bold',
                                                 fontsize=label_size, horizontal=celltype_label_horizontal)
        pol.add_segment(ct_segment_label)
        
        ct_segments[ct] = ct_segment
        pathway_segments[ct] = dict()
        gene_segments[ct] = dict()

        cpdb_ct = cpdb_df[np.logical_or(cpdb_df['ligand_celltype'] == ct,
                                         cpdb_df['receptor_celltype'] == ct)]
        
        total = 0
        unique_pathways = cpdb_ct['pathway'].unique()
        pathway_counts = {}
        for pathway in unique_pathways:
            cpdb_ct_pw = cpdb_ct[cpdb_ct['pathway'] == pathway]
            count = len(set(cpdb_ct_pw['ligand']).union(cpdb_ct_pw['receptor']))
            total += count
            pathway_counts[pathway] = count

        for pathway, count in pathway_counts.items():
            if count > 0:
                gene_segments[ct][pathway] = dict()
                
                pathway_segment = polarity.Segment(size=count/total, radians_padding=0.01)
                if plot_pathway_labels:
                    pathway_segment_label = polarity.LabelSegment(label=pathway if pathway != 'NA' and pathway != '' else 'other',
                                                                size=pathway_segment, 
                                                                radius_offset=pathway_label_offset,
                                                                fontsize=label_size)
                    pol.add_segment(pathway_segment_label)
                
                ct_segment.add_segment(pathway_segment)
                pathway_segments[ct][pathway] = pathway_segment
                
                gene_segments[ct][pathway] = dict()
        
    # add gene segments
    for row in cpdb_df.itertuples():
        pathway = getattr(row, 'pathway', 'NA')
        
        try:
            ligand_gene_segment = gene_segments[row.ligand_celltype][pathway][row.ligand]
        except KeyError:
            ligand_pathway_segment = pathway_segments[row.ligand_celltype][pathway]
            ligand_gene_segment = polarity.ColorSegment(color=colors.get(row.ligand_celltype, '#aaaaaa'), radians_padding=0.001)
            ligand_pathway_segment.add_segment(ligand_gene_segment)
            gene_segments[row.ligand_celltype][pathway][row.ligand] = ligand_gene_segment
            
            if row.ligand in plot_gene_names:
                gene_label_segment = polarity.LabelSegment(label=row.ligand,
                                                           size=ligand_gene_segment, 
                                                           radius_offset=gene_label_offset,
                                                           fontsize=label_size)
                pol.add_segment(gene_label_segment)
        
        try:
            receptor_gene_segment = gene_segments[row.receptor_celltype][pathway][row.receptor]
        except KeyError:
            receptor_pathway_segment = pathway_segments[row.receptor_celltype][pathway]
            receptor_gene_segment = polarity.ColorSegment(color=colors.get(row.receptor_celltype, '#aaaaaa'), radians_padding=0.001)
            receptor_pathway_segment.add_segment(receptor_gene_segment)
            gene_segments[row.receptor_celltype][pathway][row.receptor] = receptor_gene_segment
            
            if row.receptor in plot_gene_names:
                gene_label_segment = polarity.LabelSegment(label=row.receptor,
                                                           size=receptor_gene_segment, 
                                                           radius_offset=gene_label_offset,
                                                           fontsize=label_size)
                pol.add_segment(gene_label_segment)
    
    min_link_color_value, max_link_color_value = None, None
    if link_color_key is not None:
        if link_color_value_range is None:
            min_link_color_value, max_link_color_value = np.nanpercentile(cpdb_df[link_color_key], link_color_value_percentiles)
        else:
            min_link_color_value, max_link_color_value = link_color_value_range
    
    min_link_opacity_value, max_link_opacity_value = None, None
    if link_opacity_key is not None:
        if link_opacity_value_range is None:
            min_link_opacity_value, max_link_opacity_value = np.nanpercentile(cpdb_df[link_opacity_key], link_opacity_value_percentiles)
        else:
            min_link_opacity_value, max_link_opacity_value = link_opacity_value_range

    if link_color_key is not None:
        try:
            link_colors = matplotlib.cm.get_cmap(link_colors)
            has_link_cmap = True
        except ValueError:
            link_color = link_colors
            link_colors = {}
    elif link_colors is not None:
        link_color = link_colors
        link_colors = {}
    else:
        link_colors = colors.copy()
        link_color = "#aaaaaa"
    
    # add links
    for row in cpdb_df.itertuples():
        ligand_segment = gene_segments[row.ligand_celltype][getattr(row, 'pathway', 'NA')][row.ligand]
        receptor_segment = gene_segments[row.receptor_celltype][getattr(row, 'pathway', 'NA')][row.receptor]
        
        if link_opacity_key is not None:
            value = getattr(row, link_opacity_key)
            if value > max_link_opacity_value:
                value = max_link_opacity_value
            elif value < min_link_opacity_value:
                value = min_link_opacity_value
            
            if max_link_opacity_value - min_link_opacity_value > 0:
                norm_value = (value - min_link_opacity_value) / (max_link_opacity_value - min_link_opacity_value)
            else:
                norm_value = 1
            if invert_opacity:
                value = 1 - value
            alpha = alpha_min + (norm_value * (alpha_max - alpha_min))
        else:
            alpha = alpha_min
        
        if link_color_key is None:
            color = link_colors.get(row.ligand_celltype, link_color)
        else:
            value = getattr(row, link_color_key)
            if value > max_link_color_value:
                value = max_link_color_value
            elif value < min_link_color_value:
                value = min_link_color_value
            norm_value = (value - min_link_color_value) / (max_link_color_value - min_link_color_value)
            
            color = link_colors(norm_value)

        pol.add_link(
            polarity.BezierPatchLink(ligand_segment, receptor_segment, alpha=alpha, 
                                     facecolor=color)
        )
    
    grid = False
    ax = pol.plot(show_grid=grid, show_ticks=grid, show_axis=grid, ax=ax, fig=fig)
    ax.set_ylim(ylim)
    
    print("LEGEND ARGS", legend, link_color_key, link_opacity_key)
    if legend and (link_color_key is not None or link_opacity_key is not None):
        print("ADDING LEGEND")
        # legend
        legend_resolution = 20
        if lax is None:
            lax = ax.inset_axes([1 - legend_size, 1 - legend_size, legend_size, legend_size])
        
        im = None
        if link_color_key is not None:
            n_rows = legend_resolution
            if link_opacity_key is None:
                n_rows = 5
            
            color_array = np.tile(
                np.linspace(min_link_color_value, max_link_color_value, legend_resolution),
                n_rows
            ).reshape((n_rows, legend_resolution)).T
            im = lax.imshow(color_array, cmap=link_colors, aspect='equal', origin='lower')
        
        if link_opacity_key is not None:
            if im is None:
                n_rows = 5
                color_array = []
                color_array = np.full((n_rows, legend_resolution), 1)
                tmp_cmap = mcol.LinearSegmentedColormap.from_list("_tmp", [link_color, link_color])
                im = lax.imshow(color_array, aspect='equal', origin='lower', cmap=tmp_cmap)
            
            im_shape = im.get_array().shape
            alpha_array = np.tile(
                np.linspace(min_link_opacity_value, max_link_opacity_value, legend_resolution),
                im_shape[0]
            ).reshape((im_shape[0], legend_resolution))
            alpha_array[alpha_array > max_link_opacity_value] = max_link_opacity_value
            alpha_array[alpha_array < min_link_opacity_value] = min_link_opacity_value
            alpha_array = (alpha_array - min_link_opacity_value) / (max_link_opacity_value - min_link_opacity_value)
            alpha_array[np.isnan(alpha_array)] = 1
            if invert_opacity:
                alpha_array = 1 - alpha_array
            alpha_array = alpha_min + (alpha_array * (alpha_max - alpha_min))
            
            im.set_alpha(alpha_array)
        
        if link_color_key is not None:
            lax.set_yticks([0, im.get_array().shape[0]])
            lax.set_yticklabels([round(min_link_color_value, 2), round(max_link_color_value, 2)])
            lax.set_ylabel(link_color_key)
        else:
            lax.set_yticks([])
            
        if link_opacity_key is not None:
            lax.set_xticks([0, im.get_array().shape[1]])
            lax.set_xticklabels([round(min_link_opacity_value, 2), round(max_link_opacity_value, 2)])
            lax.set_xlabel(link_opacity_key)
        else:
            lax.set_xticks([])

        for l in lax.get_xticklabels() + lax.get_yticklabels():
            l.set_fontsize(legend_label_size)

    return ax


def count_groups_plot(counts_df, ax=None, split_order=None, group_order=None,
                      colors=None, horizontal=True, groups_rename=None,
                      relative=False, add_percentages=False,
                      legend=True, legend_inside=True, lax=None, 
                      legend_kwargs={},
                      **kwargs):
    if ax is None:
        ax = plt.gca()

    if groups_rename is None:
        groups_rename = dict()

    relative_df = (counts_df.T / counts_df.sum(axis=1)).T

    if relative:
        counts_df = relative_df

    if group_order is None:
        group_order = list(counts_df.columns)
    
    if split_order is None:
        if counts_df.shape[0] != 1 or counts_df.index[0] != 'count':
            split_order = list(counts_df.index)

    rowsums = np.zeros(len(group_order))
    
    colors = category_colors(
        categories=split_order or group_order,
        colors=colors,
    )
    
    xticks = np.arange(0, len(group_order))
    if split_order is not None:
        for index in split_order:
            try:
                row = counts_df.loc[index]
            except KeyError:
                values = [0 for _ in group_order]
                continue
            values = []
            for g_ix, g in enumerate(group_order):
                try:
                    values.append(row[g])
                except KeyError:
                    values.append(0)

            label = groups_rename.get(index, index)

            if horizontal:
                ax.barh(xticks, values, left=rowsums, label=label, color=colors[index], **kwargs)
            else:
                ax.bar(xticks, values, bottom=rowsums, label=label, color=colors[index], **kwargs)
            rowsums += values
    else:
        values = counts_df.loc['count']
        for i, (x, value, group) in enumerate(zip(xticks, values, group_order)):
            label = groups_rename.get(group, group)

            if horizontal:
                ax.barh([i], [value],  label=label, color=colors[group], **kwargs)
            else:
                ax.bar([i], [values], label=label, color=colors[group], **kwargs)
        rowsums += values
    
    if add_percentages:
        relative_sum = np.sum(counts_df, axis=0)/ np.sum(np.sum(counts_df, axis=0))
        for g_ix, g in enumerate(group_order):
            relative_label = relative_sum[g] * 100

            if horizontal:
                ax.text(rowsums[g_ix], xticks[g_ix], '{:.1f}%'.format(relative_label), verticalalignment='center',
                fontfamily='sans-serif')
            else:
                ax.text(xticks[g_ix], rowsums[g_ix], '{:.1f}%'.format(relative_label), verticalalignment='center',
                fontfamily='sans-serif')
            
            xmax = np.max(rowsums*1.2)
            ax.set_xlim((0, xmax))

    if legend:
        if lax is None:
            lax = ax
            lax_kwargs = dict(bbox_to_anchor=(1.01, 1), loc='upper left')
        else:
            lax_kwargs = dict(loc='center', labelspacing=2, frameon=False)
            lax.axis('off')

        lax_kwargs.update(legend_kwargs)
        if legend_inside:
            lax.legend(**legend_kwargs)
        else:
            lax.legend(**lax_kwargs)

    if horizontal:
        ax.set_yticks(xticks)
        ax.set_yticklabels([groups_rename.get(g, g) for g in group_order])
        ax.invert_yaxis()
    else:
        ax.set_xticks(xticks)
        ax.set_xticklabels(group_order)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if np.issubdtype(counts_df.to_numpy().dtype, np.integer):
        ax.set_xlabel("Number of cells")
    else:
        if np.max(counts_df.to_numpy()) > 1.:
            ax.set_xlabel("Percentage of cells")
        else:
            ax.set_xlabel("Fraction of cells")

    return ax


def cell_picker_embedding_plot(*args, **kwargs):
    kwargs.setdefault('picker', True)
    kwargs.setdefault('pickradius', 2)
    
    ax = embedding_plot(*args, **kwargs)
    fig = ax.figure
    
    def onpick1(event):
        if isinstance(event.artist, PathCollection):
            ind = event.ind
            data = np.array(event.artist.get_offsets().data)
            ind = event.ind
            print('onpick3 scatter:', ind, data[ind])

    fig.canvas.mpl_connect('pick_event', onpick1)
    return ax



def expression_heatmap(
    expression_df, 
    cmap=None,
    vmin=None, 
    vmax=None,
    annotation_colors=None,
    annotation_columns=None,
    label_first_annotation=True,
    legend=None, 
    legend_rows=2,
    colorbar=True,
    vcenter=None,
    fig=None
):
    if fig is None:
        fig = plt.gcf()
    
    if legend is None:
        legend_first_annotation = False
        legend = True
    elif legend:
        legend_first_annotation = True
    else:
        legend_first_annotation = False
    
    numerical_columns, annotation_columns = get_numerical_and_annotation_columns(
        expression_df, 
        annotation_columns=annotation_columns
    )
    
    if annotation_colors is None:
        annotation_colors = {}
    elif not isinstance(annotation_colors, dict):
        annotation_colors = {c: annotation_colors for c in annotation_columns}
    
    mat = expression_df.loc[:, numerical_columns]
    
    # map annotation colors
    color_mat = []
    legend_annotation_colors = {}
    if len(annotation_columns) > 0:
        for i, annotation_column in enumerate(annotation_columns):
            color_dict = category_colors(expression_df[annotation_column].unique(),
                                         colors=annotation_colors.get(annotation_column, None))
            # default_colors = color_cycle()
            # annotation_color_dict = annotation_colors.get(annotation_column, {})
            # color_dict = {}
            # for category in expression_df[annotation_column].unique():
            #     try:
            #         color_dict[category] = annotation_color_dict[category]
            #     except KeyError:
            #         color_dict[category] = next(default_colors)
            annotation_column_colors = [mcol.to_rgb(color_dict[c]) for c in expression_df[annotation_column]]
            color_mat.append(annotation_column_colors)
            if i > 0 or legend_first_annotation:
                legend_annotation_colors[annotation_column] = color_dict
        
        color_mat = np.array(color_mat).transpose((1, 0, 2))
    
    # heatmap labels
    heatmap_yticks = []
    heatmap_yticklabels = []
    if label_first_annotation and len(annotation_columns) > 0:
        start = 0
        for label, group in groupby(expression_df[annotation_columns[0]]):
            group_size = len(list(group))
            heatmap_yticks.append(start + int(group_size/2))
            heatmap_yticklabels.append(label)
            start += group_size

    n_cols = 2 + int((len(legend_annotation_colors) + 2) / (legend_rows))
    gs = GridSpec(legend_rows, n_cols, width_ratios=[5*len(annotation_columns), 100] + [20] * (n_cols-2), wspace=0.02)
    
    # annotations
    if len(annotation_columns) > 0:
        annotation_ax = fig.add_subplot(gs[0:legend_rows, 0])
        annotation_ax.imshow(color_mat, aspect='auto', interpolation='none')
        annotation_ax.set_xticks(range(0, len(annotation_columns)))
        annotation_ax.set_xticklabels(annotation_columns, rotation=90)
        annotation_ax.set_yticks(heatmap_yticks)
        annotation_ax.set_yticklabels(heatmap_yticklabels)
    
    # norm
    if vcenter is not None:
        norm = mcol.TwoSlopeNorm(vcenter=vcenter, vmin=vmin, vmax=vmax)
    else:
        norm = mcol.Normalize(vmin=vmin, vmax=vmax)
    
    # heatmap
    heatmap_ax = fig.add_subplot(gs[0:legend_rows, 1])
    heatmap = heatmap_ax.imshow(mat, aspect='auto', interpolation='none',
                                cmap=cmap, norm=norm)
    heatmap_ax.set_xticks(range(0, len(numerical_columns)))
    heatmap_ax.set_xticklabels(numerical_columns, rotation=90)
    heatmap_ax.set_yticks([])
    
    # colorbar
    if colorbar:
        gs_cb = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 2], width_ratios=[20, 80])
        cax = fig.add_subplot(gs_cb[0, 0])
        fig.colorbar(heatmap, cax=cax, orientation='vertical', shrink=0.2)
    
    if legend:
        for i, (annotation_column, colors) in enumerate(legend_annotation_colors.items()):
            lax =  fig.add_subplot(gs[(i + 1) % legend_rows, 2 + int((i + 1) / legend_rows)])
            legend_handles = [Line2D([0], [0], color=color, lw=4) for color in colors.values()]
            legend_labels = list(colors.keys())
            lax.legend(legend_handles, legend_labels, title=annotation_column,
                    bbox_to_anchor=(0.05, 1), loc='upper left', borderaxespad=0)
            lax.axis('off')
    
    return fig


def markers_expression_heatmap(
    vdata, 
    markers, 
    obs_key,
    top_n=5,
    percent_expressed_cutoff=70,
    log2_fold_change_cutoff=0,
    abs_log2_fold_change=False,
    cmap='PiYG',
    vmin=None,
    vmax=None,
    vcenter=None,
    sort_categories=True,
    categories_key='group',
    fig=None,
):
    fig = fig or plt.gcf()
    
    try:
        markers = markers.to_pandas()
    except AttributeError:
        pass

    if sort_categories:
        try:
            markers['_group_int'] = [int(c) for c in markers[categories_key]]
            sort_column = '_group_int'
        except ValueError:
            sort_column = categories_key
        
        markers = markers.sort_values(
            'log2FoldChange', 
            kind='mergesort', 
            ascending=False
        ).sort_values(sort_column, kind='mergesort')
    
    if abs_log2_fold_change:
        log2_fold_change_valid = markers['log2FoldChange'].abs() > log2_fold_change_cutoff
    else:
        log2_fold_change_valid = markers['log2FoldChange'] > log2_fold_change_cutoff
        
    top_markers = markers[
        np.logical_and(
            log2_fold_change_valid,
            markers['percent_expressed_group'] > percent_expressed_cutoff,
        )
    ].groupby(categories_key).head(n=top_n).index.unique()

    clusters = [str(c) for c in sorted(markers[categories_key].unique())]
    
    expression_df = expression_data_frame(
        vdata, 
        obs_key, 
        top_markers, 
        scale=True,
        obs_order=clusters
    )

    fig = expression_heatmap(
        expression_df, 
        cmap=cmap, 
        vmin=vmin, 
        vmax=vmax, 
        vcenter=vcenter, 
        fig=fig
    )
    
    return fig
