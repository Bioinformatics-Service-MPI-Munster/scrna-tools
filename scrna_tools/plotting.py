import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import matplotlib.colors as mcol
import matplotlib.cm as cm
import seaborn as sns
import pandas as pd
import numpy as np
from future.utils import string_types
from itertools import cycle
import functools
import warnings
import re
import scanpy as sc
import scanpy.plotting._anndata as scp
import scanpy.plotting._tools as scpt
from adjustText import adjust_text
from datetime import datetime

from .helpers import markers_to_df

import logging
logger = logging.getLogger(__name__)

try:
    import polarity
    with_polarity = True
except (ModuleNotFoundError, OSError) as e:
    logger.debug("Cannot load polarity: {}".format(e))
    with_polarity = False

cm_grey_blue = mcol.LinearSegmentedColormap.from_list("GreyBlue", ["#D3D3D3","#0026F5"])
plt.register_cmap(name='GreyBlue', cmap=cm_grey_blue)


def color_cycle(colors=None):
    if colors is None:
        colors = [
            "#0000ff", "#dc143c", "#00fa9a", "#8b008b", "#f0e68c", "#ff8c00", "#ffc0cb", 
            "#adff2f", 

            "#2f4f4f", "#556b2f", "#a0522d", "#2e8b57", "#800000", "#191970",
            "#708090", "#d2691e", "#9acd32",
            "#20b2aa", "#cd5c5c", "#32cd32", "#8fbc8f", "#b03060", "#9932cc", "#ff0000",
            "#ffd700", "#0000cd", "#00ff00", "#e9967a", "#00ffff",
            "#00bfff", "#9370db", "#a020f0", "#ff6347", "#ff00ff", "#1e90ff",
            "#ffff54", "#dda0dd", "#87ceeb", "#ff1493", "#ee82ee", "#98fb98", "#7fffd4",
            "#ffe4b5", "#ff69b4", "#dcdcdc", "#228b22", "#bc8f8f", "#4682b4", "#000080", 
            "#808000", "#b8860b", 
        ]
    elif isinstance(colors, string_types):
        try:
            colors = plt.get_cmap(colors).colors
        except ValueError:
            colors = [colors]

    colors = cycle(colors)
    return colors


def scenic_regulon_enrichment_heatmap(regulon_enrichment_df, ax=None, **kwargs):
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
    
    regulon_enrichment_df['regulon'] = list(map(lambda s: s[:-3], regulon_enrichment_df.regulon))
    df_heatmap = pd.pivot_table(data=regulon_enrichment_df, index='group', columns='regulon', values='Z')

    sns.heatmap(df_heatmap, ax=ax, **kwargs)
    ax.set_ylabel('')
    
    return ax


def scenic_binarised_plot(adata_sub, groups, figsize=(20, 90), colors=None, 
                          obsm_key=None, binarise=True, palette=None,
                          activity_support_cutoff=None, min_cells_active=0,
                          **kwargs):
    if obsm_key is None:
        obsm_key = 'X_aucell_mean' if not binarise else 'X_aucell_bin'
    
    data = pd.DataFrame(data=adata_sub.obsm[obsm_key], index=adata_sub.obs.index,
                        columns=adata_sub.uns['regulons'])
    
    if activity_support_cutoff is not None:
        valid_columns = []
        for column in data.columns:
            if np.sum(data[column] >= activity_support_cutoff) > min_cells_active:
                valid_columns.append(column)
        data = data[valid_columns]
    
    # N_COLORS = len(adata_sub.obs[group].dtype.categories)
    COLORS = [color['color'] for color in matplotlib.rcParams["axes.prop_cycle"]]

    group_colors = pd.DataFrame(index=adata_sub.obs.index)
    for group in groups:
        if colors is None or colors.get(group, None) is None:
            c = dict(zip(adata_sub.obs[group].dtype.categories, COLORS))
        else:
            c = colors[group]
        # cell_type_color_lut = dict(zip(adata.obs.cell_type.dtype.categories, adata.uns['cell_type_colors']))
        cell_id2cell_type_lut = adata_sub.obs[group].to_dict()
        print(data)
        print(data.index)
        #print(cell_id2cell_type_lut)
        group_colors[group] = data.index.map(cell_id2cell_type_lut).map(c)

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


def embedding_plot(adata, key, colorby=None, 
                   groups=None, groupby=None,
                   splitby=None, split_groups=None,
                   colors=None, shuffle=True, shuffle_seed=42,
                   ax=None, cax=None, fig=None, n_cols=2,
                   lax=None, legend=True, legend_inside=False, groups_rename=None,
                   simple_axes=True, exclude_nan=False, legend_kwargs={}, 
                   colorbar_title='', label_groups=False, 
                   title=None, **kwargs):
    if fig is None:
        fig = plt.gcf()
    
    if splitby is not None:
        if split_groups is None:
            split_groups = adata.obs[splitby].dtype.categories
        
        if not isinstance(colorby, list) and not isinstance(colorby, tuple):
            colorby = [colorby]
        
        max_xlim = None
        max_ylim = None
        axes = []
        n_rows = max(1, int(np.ceil(len(colorby) * len(split_groups) / n_cols)))
        gs = GridSpec(n_rows, n_cols)
        for i, split_group in enumerate(split_groups):
            cadata_split = adata.copy()
            cadata_split.add_categorical_constraint(splitby, split_group)
            
            for j, cb in enumerate(colorby):
                plot_ix = (i * len(colorby) + j)
                row = int(plot_ix / n_cols)
                col = int(plot_ix % n_cols)

                ax = fig.add_subplot(gs[row, col])
                embedding_plot(cadata_split, key, colorby=cb, groups=groups, groupby=groupby,
                               colors=colors, shuffle=shuffle, shuffle_seed=shuffle_seed,
                               ax=ax, cax=cax, fig=fig, n_cols=n_cols,
                               lax=lax, legend=legend, legend_inside=legend_inside,
                               groups_rename=groups_rename, simple_axes=simple_axes,
                               exclude_nan=exclude_nan, legend_kwargs=legend_kwargs,
                               colorbar_title=colorbar_title, label_groups=label_groups,
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
        
        for ax in axes:
            ax.set_xlim(max_xlim)
            ax.set_ylim(max_ylim)
        return fig
    
    if isinstance(colorby, list) or isinstance(colorby, tuple):
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
                           exclude_nan=exclude_nan, legend_kwargs=legend_kwargs,
                           colorbar_title=colorbar_title, label_groups=label_groups,
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
    
    if (colorby is not None and colorby in adata.obs_keys() 
        and adata.obs[colorby].dtype.name == 'category'):
            groupby = colorby
            colorby = None

    if groups_rename is None:
        groups_rename = dict()

    if key not in adata.obsm:
        key = 'X_{}'.format(key)
        
    logger.debug("UMAP PREP: {}s".format((datetime.now() - checkpoint).total_seconds()))
    checkpoint = datetime.now()

    coords = adata.obsm[key]
    valid_ixs = np.isfinite(coords[:, 0])
    valid_group_info = None
    if groupby is not None:
        group_info = adata.obs[groupby].to_numpy(dtype='str')
        if exclude_nan:
            valid_ixs = np.logical_and(valid_ixs, group_info != 'nan')
        else:
            group_info[group_info == 'nan'] = 'NA'
        valid_group_info = group_info[valid_ixs]

        all_groups = np.unique(valid_group_info)
        if groups is None:
            groups = all_groups
        elif isinstance(groups, string_types):
            groups = [groups]

        if groups is not None:
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
            colors = color_cycle(colors)
            dict_group_colors = dict()
            if groups is None:
                dict_group_colors['all'] = next(colors)
            else:
                for group in groups:
                    dict_group_colors[group] = next(colors)
            colors = dict_group_colors

        legend_elements = []
        if groups is None:
            color = colors['all']
            valid_colors = [matplotlib.colors.to_rgba(color) for _ in range(valid_coords.shape[0])]
        else:
            valid_colors = [None for _ in range(valid_coords.shape[0])]
            legend_elements = []
            for group in groups:
                color = matplotlib.colors.to_rgba(colors[group])
                ixs = np.argwhere(valid_group_info == group)[:, 0]
                for ix in ixs:
                    valid_colors[ix] = color

                if len(ixs) > 0:
                    label = groups_rename.get(group, group)
                    legend_elements.append(matplotlib.patches.Patch(facecolor=color, edgecolor=color,
                                           label=label))
            if title is None:
                ax.set_title(groupby.replace('_', ' '))
        valid_colors = np.array(valid_colors)
    else:
        logger.debug("UMAP START COLORBY: {}s".format((datetime.now() - checkpoint).total_seconds()))
        checkpoint = datetime.now()
        if colors is None:
            cmap = 'viridis'
        else:
            cmap = colors

        if isinstance(colorby, string_types):
            try:
                checkpoint = datetime.now()
                feature_values = adata.values(colorby)[valid_ixs]
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

    plot = ax.scatter(valid_coords[:, 0], valid_coords[:, 1], c=valid_colors, cmap=cmap, **kwargs)
    
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
        elif colorby is not None:
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
    m = re.match(r'__subset__(.+)__(.+)', axis_label)
    if m is not None:
        axis_label = '{}'.format(m.group(2).upper())
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


def _colorbar_ax(fig, subplot_spec, max_cbar_height: float = 4.0):
    width, height = fig.get_size_inches()
    if height > max_cbar_height:
        # to make the colorbar shorter, the
        # ax is split and the lower portion is used.
        axs2 = GridSpecFromSubplotSpec(
            2,
            1,
            subplot_spec=subplot_spec,
            height_ratios=[height - max_cbar_height, max_cbar_height],
        )
        heatmap_cbar_ax = fig.add_subplot(axs2[1])
    else:
        heatmap_cbar_ax = fig.add_subplot(subplot_spec)

    return heatmap_cbar_ax


def markers_heatmap(
        adata: scp.AnnData,
        var_names: scp.Union[scp._VarNames, scp.Mapping[str, scp._VarNames]],
        groupby: scp.Union[str, scp.Sequence[str]],
        use_raw: scp.Optional[bool] = None,
        log: bool = False,
        scale_and_center: bool = False,
        num_categories: int = 7,
        dendrogram: scp.Union[bool, str] = False,
        gene_symbols: scp.Optional[str] = None,
        var_group_positions: scp.Optional[scp.Sequence[scp.Tuple[int, int]]] = None,
        var_group_labels: scp.Optional[scp.Sequence[str]] = None,
        var_group_rotation: scp.Optional[float] = None,
        layer: scp.Optional[str] = None,
        standard_scale: scp.Optional[scp.Literal['var', 'obs']] = None,
        swap_axes: bool = False,
        show_gene_labels: scp.Optional[bool] = None,
        groups: scpt.Union[str, scpt.Sequence[str]] = None,
        show: scp.Optional[bool] = None,
        save: scp.Union[str, bool, None] = None,
        figsize: scp.Optional[scp.Tuple[float, float]] = None,
        axes: scp.Union[matplotlib.axes._axes.Axes] = None,
        colors: scp.Union[str, list, None] = None,
        **kwds,
):
    """\
    Heatmap of the expression values of genes.
    If `groupby` is given, the heatmap is ordered by the respective group. For
    example, a list of marker genes can be plotted, ordered by clustering. If
    the `groupby` observation annotation is not categorical the observation
    annotation is turned into a categorical by binning the data into the number
    specified in `num_categories`.
    Parameters
    ----------
    {common_plot_args}
    standard_scale
        Whether or not to standardize that dimension between 0 and 1, meaning for each variable or observation,
        subtract the minimum and divide each by its maximum.
    swap_axes
         By default, the x axis contains `var_names` (e.g. genes) and the y axis the `groupby`
         categories (if any). By setting `swap_axes` then x are the `groupby` categories and y the `var_names`.
    show_gene_labels
         By default gene labels are shown when there are 50 or less genes. Otherwise the labels are removed.
    {show_save_ax}
    axes
        Matplotlib axes for (in this order): heatmap, colorbar, dendrogram, groups, genes
    **kwds
        Are passed to :func:`matplotlib.pyplot.imshow`.
    Returns
    -------
    List of :class:`~matplotlib.axes.Axes`
    Examples
    -------
    >>> import scanpy as sc
    >>> adata = sc.datasets.pbmc68k_reduced()
    >>> markers = ['C1QA', 'PSAP', 'CD79A', 'CD79B', 'CST3', 'LYZ']
    >>> sc.pl.heatmap(adata, markers, groupby='bulk_labels', dendrogram=True, swap_axes=True)
    Using var_names as dict:
    >>> markers = {{'T-cell': 'CD3D', 'B-cell': 'CD79A', 'myeloid': 'CST3'}}
    >>> sc.pl.heatmap(adata, markers, groupby='bulk_labels', dendrogram=True)
    See also
    --------
    rank_genes_groups_heatmap: to plot marker genes identified using the :func:`~scanpy.tl.rank_genes_groups` function.
    """
    if use_raw is None and adata.raw is not None:
        use_raw = True

    var_names, var_group_labels, var_group_positions = scp._check_var_names_type(
        var_names, var_group_labels, var_group_positions
    )

    categories, obs_tidy = scp._prepare_dataframe(
        adata,
        var_names,
        groupby,
        use_raw,
        log,
        num_categories,
        gene_symbols=gene_symbols,
        layer=layer,
    )

    if scale_and_center:
        centered = obs_tidy.subtract(obs_tidy.mean(axis=0), axis=1)
        scaled = centered.div(np.nanstd(centered, axis=0), axis=1)
        obs_tidy = scaled

    if groups is not None:
        ixs = [g in set(groups) for g in obs_tidy.index]
        obs_tidy = obs_tidy[ixs]
        obs_tidy.index = obs_tidy.index.set_categories(groups)
        categories = np.unique(obs_tidy.index)

    if standard_scale == 'obs':
        obs_tidy = obs_tidy.sub(obs_tidy.min(1), axis=0)
        obs_tidy = obs_tidy.div(obs_tidy.max(1), axis=0).fillna(0)
    elif standard_scale == 'var':
        obs_tidy -= obs_tidy.min(0)
        obs_tidy = (obs_tidy / obs_tidy.max(0)).fillna(0)
    elif standard_scale is None:
        pass
    else:
        scp.logg.warning('Unknown type for standard_scale, ignored')

    if groupby is None or len(categories) <= 1:
        categorical = False
        # dendrogram can only be computed  between groupby categories
        dendrogram = False
    else:
        categorical = True
        # get categories colors:
        if colors is None:
            if groupby + "_colors" in adata.uns:
                colors = adata.uns[groupby + "_colors"]
            else:
                colors = color_cycle(colors)

        if groups is None:
            groups = np.unique(adata.obs[groupby])
        
        if isinstance(colors, dict):
            groupby_colors = [colors.get(g) for g in groups]
        else:
            groupby_colors = [next(colors) for _ in range(len(groups))]
    
    print('----->', groupby_colors)

    if dendrogram:
        dendro_data = scp._reorder_categories_after_dendrogram(
            adata,
            groupby,
            dendrogram,
            var_names=var_names,
            var_group_labels=var_group_labels,
            var_group_positions=var_group_positions,
            categories=categories,
        )

        var_group_labels = dendro_data['var_group_labels']
        var_group_positions = dendro_data['var_group_positions']

        # reorder obs_tidy
        if dendro_data['var_names_idx_ordered'] is not None:
            obs_tidy = obs_tidy.iloc[:, dendro_data['var_names_idx_ordered']]
            var_names = [var_names[x] for x in dendro_data['var_names_idx_ordered']]

        obs_tidy.index = obs_tidy.index.reorder_categories(
            [categories[x] for x in dendro_data['categories_idx_ordered']],
            ordered=True,
        )

        # reorder groupby colors
        if groupby_colors is not None:
            groupby_colors = [
                groupby_colors[x] for x in dendro_data['categories_idx_ordered']
            ]

    if show_gene_labels is None:
        if len(var_names) <= 50:
            show_gene_labels = True
        else:
            show_gene_labels = False
            scp.logg.warning(
                'Gene labels are not shown when more than 50 genes are visualized. '
                'To show gene labels set `show_gene_labels=True`'
            )
    if categorical:
        obs_tidy = obs_tidy.sort_index()

    colorbar_width = 0.2

    if not swap_axes:
        # define a layout of 2 rows x 4 columns
        # first row is for 'brackets' (if no brackets needed, the height of this row is zero)
        # second row is for main content. This second row is divided into three axes:
        #   first ax is for the categories defined by `groupby`
        #   second ax is for the heatmap
        #   third ax is for the dendrogram
        #   fourth ax is for colorbar

        dendro_width = 1 if dendrogram else 0
        groupby_width = 0.2 if categorical else 0
        if figsize is None:
            height = 6
            if show_gene_labels:
                heatmap_width = len(var_names) * 0.3
            else:
                heatmap_width = 8
            width = heatmap_width + dendro_width + groupby_width
        else:
            width, height = figsize
            heatmap_width = width - (dendro_width + groupby_width)

        if var_group_positions is not None and len(var_group_positions) > 0:
            # add some space in case 'brackets' want to be plotted on top of the image
            height_ratios = [0.15, height]
        else:
            height_ratios = [0, height]

        width_ratios = [
            groupby_width,
            heatmap_width,
            dendro_width,
            colorbar_width,
        ]

        if axes is None or isinstance(axes, matplotlib.gridspec.SubplotSpec):
            fig = scp.pl.figure(figsize=(width, height))

            if isinstance(axes, matplotlib.gridspec.SubplotSpec) or isinstance(axes, matplotlib.gridspec.GridSpec):
                axs = scp.gridspec.GridSpecFromSubplotSpec(
                    nrows=2,
                    ncols=4,
                    width_ratios=width_ratios,
                    wspace=0.15 / width,
                    hspace=0.13 / height,
                    height_ratios=height_ratios,
                    subplot_spec=axes
                )
            else:
                axs = scp.gridspec.GridSpec(
                    nrows=2,
                    ncols=4,
                    width_ratios=width_ratios,
                    wspace=0.15 / width,
                    hspace=0.13 / height,
                    height_ratios=height_ratios,
                )

            heatmap_ax = fig.add_subplot(axs[1, 1])
            colorbar_ax = _colorbar_ax(fig, axs[1, 3])
            groupby_ax = fig.add_subplot(axs[1, 0]) if categorical else None
            dendro_ax = fig.add_subplot(axs[1, 2], sharey=heatmap_ax) if dendrogram else None
            gene_groups_ax = fig.add_subplot(axs[0, 1], sharex=heatmap_ax) if var_group_positions is not None and \
                                                                              len(var_group_positions) > 0 else None
        else:
            if isinstance(axes, matplotlib.axes._axes.Axes):
                axes = [axes]
            heatmap_ax = axes[0]
            colorbar_ax = axes[1] if len(axes) > 1 else None
            dendro_ax = axes[2] if len(axes) > 2 else None
            groupby_ax = axes[3] if len(axes) > 3 else None
            gene_groups_ax = axes[4] if len(axes) > 4 else None

        kwds.setdefault('interpolation', 'nearest')
        im = heatmap_ax.imshow(obs_tidy.values, aspect='auto', **kwds)

        heatmap_ax.set_ylim(obs_tidy.shape[0] - 0.5, -0.5)
        heatmap_ax.set_xlim(-0.5, obs_tidy.shape[1] - 0.5)
        heatmap_ax.tick_params(axis='y', left=False, labelleft=False)
        heatmap_ax.set_ylabel('')
        heatmap_ax.grid(False)

        # sns.heatmap(obs_tidy, yticklabels="auto", ax=heatmap_ax, cbar_ax=heatmap_cbar_ax, **kwds)
        if show_gene_labels:
            heatmap_ax.tick_params(axis='x', labelsize='small')
            heatmap_ax.set_xticks(np.arange(len(var_names)))
            heatmap_ax.set_xticklabels(var_names, rotation=90)
        else:
            heatmap_ax.tick_params(axis='x', labelbottom=False, bottom=False)

        # plot colorbar
        if colorbar_ax is not None:
            plt.colorbar(im, cax=colorbar_ax)

        if categorical and groupby_ax is not None:
            try:
                ticks, labels, groupby_cmap, norm = scp._plot_categories_as_colorblocks(
                    groupby_ax, obs_tidy, colors=groupby_colors, orientation='left'
                )
            except ValueError:
                label2code, ticks, labels, groupby_cmap, norm = scp._plot_categories_as_colorblocks(
                    groupby_ax, obs_tidy, colors=groupby_colors, orientation='left'
                )

            # add lines to main heatmap
            line_positions = (
                    np.cumsum(obs_tidy.index.value_counts(sort=False))[:-1] - 0.5
            )
            heatmap_ax.hlines(
                line_positions,
                -0.73,
                len(var_names) - 0.5,
                lw=0.6,
                zorder=10,
                clip_on=False,
            )

        if dendrogram and dendro_ax is not None:
            scp._plot_dendrogram(
                dendro_ax, adata, groupby, ticks=ticks, dendrogram_key=dendrogram,
            )

        # plot group legends on top of heatmap_ax (if given)
        if var_group_positions is not None and len(var_group_positions) > 0 and gene_groups_ax is not None:
            scp._plot_gene_groups_brackets(
                gene_groups_ax,
                group_positions=var_group_positions,
                group_labels=var_group_labels,
                rotation=var_group_rotation,
                left_adjustment=-0.3,
                right_adjustment=0.3,
            )

    # swap axes case
    else:
        # define a layout of 3 rows x 3 columns
        # The first row is for the dendrogram (if not dendrogram height is zero)
        # second row is for main content. This col is divided into three axes:
        #   first ax is for the heatmap
        #   second ax is for 'brackets' if any (othwerise width is zero)
        #   third ax is for colorbar

        dendro_height = 0.8 if dendrogram else 0
        groupby_height = 0.13 if categorical else 0
        if figsize is None:
            if show_gene_labels:
                heatmap_height = len(var_names) * 0.18
            else:
                heatmap_height = 4
            width = 10
            height = heatmap_height + dendro_height + groupby_height
        else:
            width, height = figsize
            heatmap_height = height - (dendro_height + groupby_height)

        height_ratios = [dendro_height, heatmap_height, groupby_height]

        if var_group_positions is not None and len(var_group_positions) > 0:
            # add some space in case 'brackets' want to be plotted on top of the image
            width_ratios = [width, 0.14, colorbar_width]
        else:
            width_ratios = [width, 0, colorbar_width]

        if axes is None or isinstance(axes, matplotlib.gridspec.SubplotSpec):
            fig = scp.pl.figure(figsize=(width, height))

            if isinstance(axes, matplotlib.gridspec.SubplotSpec) or isinstance(axes, matplotlib.gridspec.GridSpec):
                axs = scp.gridspec.GridSpecFromSubplotSpec(
                    nrows=3,
                    ncols=3,
                    wspace=0.25 / width,
                    hspace=0.3 / height,
                    width_ratios=width_ratios,
                    height_ratios=height_ratios,
                    subplot_spec=axes
                )
            else:
                axs = scp.gridspec.GridSpec(
                    nrows=3,
                    ncols=3,
                    wspace=0.25 / width,
                    hspace=0.3 / height,
                    width_ratios=width_ratios,
                    height_ratios=height_ratios,
                )

            heatmap_ax = fig.add_subplot(axs[1, 0])
            colorbar_ax = _colorbar_ax(fig, axs[1, 2])
            groupby_ax = fig.add_subplot(axs[2, 0]) if categorical else None
            dendro_ax = fig.add_subplot(axs[0, 0], sharey=heatmap_ax) if dendrogram else None
            gene_groups_ax = fig.add_subplot(axs[1, 1], sharex=heatmap_ax) if var_group_positions is not None and \
                                                                              len(var_group_positions) > 0 else None
        else:
            if isinstance(axes, matplotlib.axes._axes.Axes):
                axes = [axes]
            heatmap_ax = axes[0]
            colorbar_ax = axes[1] if len(axes) > 1 else None
            dendro_ax = axes[2] if len(axes) > 2 else None
            groupby_ax = axes[3] if len(axes) > 3 else None
            gene_groups_ax = axes[4] if len(axes) > 4 else None

        # plot heatmap
        kwds.setdefault('interpolation', 'nearest')
        im = heatmap_ax.imshow(obs_tidy.T.values, aspect='auto', **kwds)
        heatmap_ax.set_xlim(0, obs_tidy.shape[0])
        heatmap_ax.set_ylim(obs_tidy.shape[1] - 0.5, -0.5)
        heatmap_ax.tick_params(axis='x', bottom=False, labelbottom=False)
        heatmap_ax.set_xlabel('')
        heatmap_ax.grid(False)
        if show_gene_labels:
            heatmap_ax.tick_params(axis='y', labelsize='small', length=1)
            heatmap_ax.set_yticks(np.arange(len(var_names)))
            heatmap_ax.set_yticklabels(var_names, rotation=0)
        else:
            heatmap_ax.tick_params(axis='y', labelleft=False, left=False)

        if categorical and groupby_ax is not None:
            ticks, labels, groupby_cmap, norm = scp._plot_categories_as_colorblocks(
                groupby_ax, obs_tidy, colors=groupby_colors, orientation='bottom',
            )
            # add lines to main heatmap
            line_positions = (
                    np.cumsum(obs_tidy.index.value_counts(sort=False))[:-1] - 0.5
            )
            heatmap_ax.vlines(
                line_positions,
                -0.5,
                len(var_names) + 0.35,
                lw=0.6,
                zorder=10,
                clip_on=False,
            )

        if dendrogram and dendro_ax is not None:
            scp._plot_dendrogram(
                dendro_ax,
                adata,
                groupby,
                dendrogram_key=dendrogram,
                ticks=ticks,
                orientation='top',
            )

        # plot group legends next to the heatmap_ax (if given)
        if var_group_positions is not None and len(var_group_positions) > 0 and gene_groups_ax is not None:
            arr = []
            for idx, pos in enumerate(var_group_positions):
                arr += [idx] * (pos[1] + 1 - pos[0])

            gene_groups_ax.imshow(
                np.matrix(arr).T, aspect='auto', cmap=groupby_cmap, norm=norm
            )
            gene_groups_ax.axis('off')

        # plot colorbar
        if colorbar_ax is not None:
            plt.colorbar(im, cax=colorbar_ax)

    return_ax_dict = {'heatmap_ax': heatmap_ax}
    if categorical:
        return_ax_dict['groupby_ax'] = groupby_ax
    if dendrogram:
        return_ax_dict['dendrogram_ax'] = dendro_ax
    if var_group_positions is not None and len(var_group_positions) > 0:
        return_ax_dict['gene_groups_ax'] = gene_groups_ax

    return return_ax_dict


def _rank_genes_groups_plot(
    adata: scpt.AnnData,
    plot_type: str = 'heatmap',
    groups: scpt.Union[str, scpt.Sequence[str]] = None,
    n_genes: int = 10,
    groupby: scpt.Optional[str] = None,
    values_to_plot: scpt.Optional[str] = None,
    min_logfoldchange: scpt.Optional[float] = None,
    key: scpt.Optional[str] = None,
    show: scpt.Optional[bool] = None,
    save: scpt.Optional[bool] = None,
    return_fig: scpt.Optional[bool] = False,
    **kwds,
):
    """\
    Common function to call the different rank_genes_groups_* plots
    """
    if key is None:
        key = 'rank_genes_groups'

    if groupby is None:
        groupby = str(adata.uns[key]['params']['groupby'])
    group_names = adata.uns[key]['names'].dtype.names if groups is None else groups

    var_names = {}  # dict in which each group is the key and the n_genes are the values
    gene_names = []
    for group in group_names:
        if min_logfoldchange is not None:
            df = sc.get.rank_genes_groups_df(adata, group, key=key)
            # select genes with given log_fold change
            genes_list = df[df.logfoldchanges > min_logfoldchange].names.tolist()[
                :n_genes
            ]
        else:
            # get all genes that are 'non-nan'
            genes_list = [
                gene for gene in adata.uns[key]['names'][group] if not pd.isnull(gene)
            ][:n_genes]

        if len(genes_list) == 0:
            scpt.logg.warning(f'No genes found for group {group}')
            continue
        var_names[group] = genes_list
        gene_names.extend(genes_list)

    # by default add dendrogram to plots
    kwds.setdefault('dendrogram', True)

    if plot_type in ['dotplot', 'matrixplot']:
        # these two types of plots can also
        # show score, logfoldchange and pvalues, in general any value from rank
        # genes groups
        title = None
        values_df = None
        if values_to_plot is not None:
            values_df = scpt._get_values_to_plot(adata, values_to_plot, gene_names, key=key)
            title = values_to_plot

        if plot_type == 'dotplot':
            from scanpy.plotting import dotplot

            _pl = dotplot(
                adata,
                var_names,
                groupby,
                dot_color_df=values_df,
                return_fig=True,
                **kwds,
            )

            if title is not None:
                _pl.legend(colorbar_title=title.replace("_", " "))
        elif plot_type == 'matrixplot':
            from scanpy.plotting import matrixplot

            _pl = matrixplot(
                adata, var_names, groupby, values_df=values_df, return_fig=True, **kwds
            )

            if title is not None:
                _pl.legend(title=title.replace("_", " "))

        return scpt._fig_show_save_or_axes(_pl, return_fig, show, save)

    elif plot_type == 'stacked_violin':
        from scanpy.plotting import stacked_violin

        _pl = stacked_violin(adata, var_names, groupby, return_fig=True, **kwds)
        return scpt._fig_show_save_or_axes(_pl, return_fig, show, save)

    elif plot_type == 'heatmap':
        return markers_heatmap(adata, var_names, groupby, groups=group_names, show=show, save=save, **kwds)

    elif plot_type == 'tracksplot':
        from scanpy.plotting import tracksplot

        return tracksplot(adata, var_names, groupby, show=show, save=save, **kwds)


def rank_genes_groups_heatmap(
    adata: scpt.AnnData,
    groups: scpt.Union[str, scpt.Sequence[str]] = None,
    n_genes: int = 10,
    groupby: scpt.Optional[str] = None,
    min_logfoldchange: scpt.Optional[float] = None,
    key: str = None,
    show: scpt.Optional[bool] = None,
    save: scpt.Optional[bool] = None,
    **kwds,
):
    """\
    Plot ranking of genes using heatmap plot (see :func:`~scanpy.pl.heatmap`)
    Parameters
    ----------
    adata
        Annotated data matrix.
    groups
        The groups for which to show the gene ranking.
    n_genes
        Number of genes to show.
    groupby
        The key of the observation grouping to consider. By default,
        the groupby is chosen from the rank genes groups parameter but
        other groupby options can be used.  It is expected that
        groupby is a categorical. If groupby is not a categorical observation,
        it would be subdivided into `num_categories` (see :func:`~scanpy.pl.heatmap`).
    min_logfoldchange
        Value to filter genes in groups if their logfoldchange is less than the
        min_logfoldchange
    key
        Key used to store the ranking results in `adata.uns`.
    **kwds
        Are passed to :func:`~scanpy.pl.heatmap`.
    {show_save_ax}
    """
    return _rank_genes_groups_plot(
        adata,
        plot_type='heatmap',
        groups=groups,
        n_genes=n_genes,
        groupby=groupby,
        key=key,
        min_logfoldchange=min_logfoldchange,
        show=show,
        save=save,
        **kwds,
    )


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


def volcano_plot_from_df(df, plot_adjusted_pvalues=False,
                         label_genes=None, label_top_n=0, min_pvalue=2.096468e-309,
                         log2fc_cutoff=2, padj_cutoff=1e-10,
                         exclude_genes_prefix=None, exclude_genes=None, include_genes=None,
                         insignificant_color='#eeeeee', logfc_color='#fceade', padj_color='#25ced1',
                         both_color='#ff8a5b', label_top_only_significant=True,
                         ignore_insignificant_large_z=True,
                         hide_insignificant_large_z=True, z_artifact_cutoff=15,
                         x_symmetrical=True, log2fc_field='log2fc', pvalue_field='pval', padj_field='padj',
                         gene_name_field='name',
                         label_size='medium',
                         xlim=None, ylim=None, ax=None, **kwargs):    
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
    for i, label in enumerate(plot_labels):
        text = ax.annotate(label, plot_labels_xy[i], fontsize=label_size)
        texts.append(text)
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='grey'), ax=ax, lim=50)

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


def violin_plot(cadata, feature,
                groupby=None, groups=None, groups_rename={}, colors=None,
                splitby=None, split_groups=None, split_rename={}, 
                scale='width', legend=False, ax=None, fig=None,
                ylim=None, swarm=True, 
                ignore_groups=('NA',),
                layer=None, 
                points_size=2, points_color='black', points_alpha=0.4, 
                points_jitter=0.25,
                shuffle_seed=42,
                **kwargs):
    if fig is None:
        fig = plt.gcf()
    
    if ax is None:
        ax = fig.gca()

    if groupby is not None:
        if groups is None:
            groups = cadata.obs[groupby].dtype.categories
        else:
            cadata.add_categorical_constraint(groupby, groups)
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
            cadata.add_categorical_constraint(splitby, split_groups)
            
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

    values = cadata.values(feature, layer=layer)
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

    if not legend and ax.get_legend() is not None:
        ax.get_legend().remove()

    ax.set_xticklabels([label.get_text() for label in ax.get_xticklabels()], rotation=90)
    ax.set_xlabel("")
    ax.set_ylabel(feature)

    if ylim is not None:
        ax.set_ylim(ylim)

    return ax


#
# LR plot Circos-style
#
def cpdb_ligand_receptor_circos(cpdb_df, celltypes=None, colors=None, 
                                label_size='xx-small', plot_gene_names=False,
                                celltype_label_offset=3.0, pathway_label_offset=1.5,
                                gene_label_offset=1.5, ylim=(0, 15),
                                inner_radius=10,
                                ax=None):
    if not with_polarity:
        raise ValueError("Cannot plot circos, please install polarity first.")
        
    # remove NaNs
    cpdb_df = cpdb_df.copy()
    cpdb_df['pathway'][cpdb_df['pathway'].isna()] = ''
    
    if colors is None:
        colors = dict()
    
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
        celltypes = list(set(cpdb_df['ligand_celltype'].unique()).union(set(cpdb_df['receptor_celltype'].unique())))
    
    ct_segments = dict()
    pathway_segments = dict()
    gene_segments = dict()
    for ct in celltypes:
        ct_segment = polarity.Segment(data_span=(0, 1))
        pol.add_segment(ct_segment)
        ct_segment_label = polarity.LabelSegment(label=ct, size=ct_segment, 
                                                 radius_offset=celltype_label_offset, fontweight='bold',
                                                 fontsize=label_size)
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
    
    min_expression, max_expression = np.nanpercentile(cpdb_df['mean'], [5, 95])
    
    # add links
    for row in cpdb_df.itertuples():
        ligand_segment = gene_segments[row.ligand_celltype][getattr(row, 'pathway', 'NA')][row.ligand]
        receptor_segment = gene_segments[row.receptor_celltype][getattr(row, 'pathway', 'NA')][row.receptor]
        
        alpha = min(.9, max(0.2, (row.mean - min_expression) / (max_expression - min_expression)))
        #print(alpha)
        pol.add_link(
            polarity.BezierPatchLink(ligand_segment, receptor_segment, alpha=alpha, 
                                    facecolor=colors.get(row.ligand_celltype, '#aaaaaa')))
    
    grid = False
    ax = pol.plot(show_grid=grid, show_ticks=grid, show_axis=grid, ax=ax)
    ax.set_ylim(ylim)
    
    return ax
