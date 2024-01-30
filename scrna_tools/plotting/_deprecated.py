import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec

import pandas as pd
import numpy as np
from itertools import cycle

import scanpy as sc
import scanpy.plotting._anndata as scp
import scanpy.plotting._tools as scpt

from .helpers import color_cycle

import logging
logger = logging.getLogger(__name__)


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
        axes: matplotlib.axes._axes.Axes = None,
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
        print(colors)
        if colors is None:
            if groupby + "_colors" in adata.uns:
                colors = cycle(adata.uns[groupby + "_colors"])
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