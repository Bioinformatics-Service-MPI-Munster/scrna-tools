import numpy as np
import pandas as pd
import re

import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.colors
import matplotlib.gridspec
import seaborn as sns
from scipy.stats import trim_mean
from time import time

from .base_plotting import category_colors
from ..helpers import find_cells_from_coords


embedding_re = re.compile(r'((__subset__|__view__)(?P<view_key>.+)__)?(X_)?(?P<embedding>.+)_(?P<dimension>\d+)')


def embedding_plot_data(
    vdata,
    obsm_key,
    view_key=None,
    colorby=None,
    categories=None,
    splitby=None,
    split_categories=None,
    layer=None,
    dimensions=(0, 1),
    exclude_nan=False,
    nan_replacement='NA'
):
    from .._core import VData
    
    if not isinstance(vdata, VData):
        vdata = VData(vdata)
    obsm = vdata.obsm(view_key=view_key)
    if obsm_key not in obsm.keys() and f'X_{obsm_key}' in obsm.keys():
        obsm_key = f'X_{obsm_key}'
    
    coords = obsm[obsm_key]
    data = pd.DataFrame(
        coords[:, dimensions],
        index=vdata.obs.index,
        columns=[f'{obsm_key}_{i + 1}' for i in dimensions],
    )
    
    categories = categories or {}
    if colorby is not None:
        if not isinstance(colorby, (list, tuple)):
            if categories is not None and not isinstance(categories, dict):
                categories = {
                    colorby: categories
                }
            colorby = [colorby]
        elif categories is not None and not isinstance(categories, dict):
            raise ValueError("categories must be a dict of the form " \
                             "{obs_key: [category1, category2, ...]} " \
                             "when colorby is a list of keys!")
        
        for obs_key in colorby:
            data[obs_key] = vdata.obs_vector(obs_key, layer=layer, view_key=view_key)
            if obs_key not in categories and data[obs_key].dtype.name == 'category':
                categories[obs_key] = data[obs_key].dtype.categories
        
    if splitby is not None:
        data[splitby] = vdata.obs_vector(splitby, layer=layer, view_key=view_key)
    
    # filter
    data = data.loc[np.isfinite(data.iloc[:, 0]), :]
    
    if colorby is not None:
        for obs_key in colorby:
            if data[obs_key].dtype.name == 'category':
                is_na = np.logical_or(
                    data[obs_key].isna(),
                    data[obs_key] == 'NA'
                )
                
                if exclude_nan:
                    data = data.loc[data[obs_key].isin(categories[obs_key]), :]
                else:
                    data = data.loc[
                        np.logical_or(
                            data[obs_key].isin(categories[obs_key]),
                            is_na,
                        ), 
                        :
                    ]
                new_categories = data[obs_key].to_numpy()
                is_na = np.logical_or(
                    data[obs_key].isna(),
                    data[obs_key] == 'NA'
                )
                new_categories[is_na] = nan_replacement
                
                dtype_categories = categories[obs_key] if not np.any(is_na) else list(categories[obs_key]) + [nan_replacement]
                data[obs_key] = pd.Categorical(
                    new_categories, 
                    categories=dtype_categories,
                )
    
    # split
    if splitby is not None and data[splitby].dtype.name == 'category':
        split_categories = split_categories or data[splitby].dtype.categories
        
        split_data = {}
        for category, sub_data in data.groupby(splitby):
            if category in split_categories:
                split_data[category] = sub_data.drop(splitby, axis=1)
        data = split_data
        
    return data


def embedding_plot_from_df(
    data,
    colors=None,
    vmin=None,
    vmax=None,
    vcenter=None,
    exclude_nan=False,
    nan_color='#cccccc',
    nan_category='NA',
    shuffle=True, 
    shuffle_seed=42,
    z_order_by_magnitude=False,
    simple_axes=True,
    label_groups=False,
    legend=True,
    legend_inside=False,
    colorbar_title='',
    show_title=None,
    x=None,
    y=None,
    xlim=None,
    ylim=None,
    colorby=None,
    ax=None,
    cax=None,
    lax=None,
    fig=None,
    legend_kwargs={},
    **kwargs,
):
    # is this a multi-plot?
    if isinstance(data, dict) or (data.shape[1] - 2 > 1 and colorby is None):
        return embedding_figure_from_df(
            data,
            colors=colors,
            vmin=vmin,
            vmax=vmax,
            vcenter=vcenter,
            exclude_nan=exclude_nan,
            nan_color=nan_color,
            shuffle=shuffle, 
            shuffle_seed=shuffle_seed,
            z_order_by_magnitude=z_order_by_magnitude,
            simple_axes=simple_axes,
            label_groups=label_groups,
            legend=legend,
            legend_inside=legend_inside,
            colorbar_title=colorbar_title,
            show_title=show_title,
            x=x,
            y=y,
            xlim=None,
            ylim=None,
            fig=fig,
            legend_kwargs=legend_kwargs,
            **kwargs,
        )
    
    ax = ax or plt.gca()
    fig = fig or ax.figure
    
    kwargs.setdefault('marker', '.')
    kwargs.setdefault('s', 10)
    kwargs.setdefault('linewidth', 0)
    
    if shuffle:
        data = data.sample(frac=1, random_state=shuffle_seed)
    
    x = x or data.columns[0]
    y = y or data.columns[1]
    if colorby is None and data.shape[1] > 2:
        colorby = data.columns[2]
        if z_order_by_magnitude and data[colorby].dtype.name != 'category':
            data = data.sort_values(colorby, ascending=True)
    
    cmap = None
    norm = None
    legend_elements = None
    if colorby is not None:
        if data[colorby].dtype.name == 'category':
            categories = data[colorby].dtype.categories

            colors_dict = category_colors(
                categories=categories,
                colors=colors,
            )
            colors_dict[nan_category] = nan_color

            colors = [colors_dict[c] for c in data[colorby]]
            
            # legend
            legend_elements = []
            for category in categories:
                legend_elements.append(
                    matplotlib.patches.Patch(
                        facecolor=colors_dict[category], 
                        edgecolor=colors_dict[category],
                        label=category)
                    )
            
            if label_groups:
                for category, data_sub in data.groupby(colorby):
                    label_coords = trim_mean(data_sub.loc[:, [x, y]], 0.2, axis=0)
                    ax.annotate(category, label_coords, ha='center')
        else:
            cmap = colors or 'viridis'

            if isinstance(cmap, (str, bytes)):
                cmap = plt.cm.get_cmap(cmap)

            if not exclude_nan:
                cmap.set_bad(nan_color)
            
            if vcenter:
                norm = matplotlib.colors.TwoSlopeNorm(vcenter=vcenter, vmin=vmin, vmax=vmax)
            else:
                norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
            
            colors = data[colorby]
    else:
        if isinstance(colors, matplotlib.colors.ListedColormap):
            kwargs.setdefault('color', colors(0))
        else:
            kwargs.setdefault('color', colors)
        colors = None
    
    plot = ax.scatter(
        data[x], 
        data[y], 
        c=colors, 
        cmap=cmap,
        norm=norm,
        plotnonfinite=not exclude_nan, 
        **kwargs
    )
    
    if legend:
        if legend_elements is not None:
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
        elif cmap is not None:
            if cax is not None:
                cb = plt.colorbar(plot, cax=cax)
            else:
                if not legend_inside:
                    cb = plt.colorbar(plot, ax=ax)
                else:
                    cax = ax.inset_axes([0.9, 0.8, 0.05, 0.2])
                    cb = plt.colorbar(plot, cax=cax)
            cb.set_label(colorbar_title)
    
    mx = embedding_re.match(x)
    if mx is not None:
        axis_label_1 = "{embedding}{dimension}".format(
            embedding=mx.group("embedding").upper(), 
            dimension=mx.group("dimension")
        )
    else:
        axis_label_1 = x.upper()
        
    my = embedding_re.match(y)
    if my is not None:
        axis_label_2 = "{embedding}{dimension}".format(
            embedding=my.group("embedding").upper(), 
            dimension=my.group("dimension")
        )
    else:
        axis_label_2 = y.upper()
    
    ax.set_xlabel(axis_label_1, loc='right')
    ax.set_ylabel(axis_label_2, loc='top')
    
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    
    if simple_axes:
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
    sns.despine(ax=ax, top=True, right=True)
    
    if show_title and colorby is not None:
        ax.set_title(colorby)
    
    return ax


def embedding_figure_from_df(
    data, 
    n_cols=2,
    preserve_axis_limits=True,
    **kwargs
):
    dimensions = kwargs.get('dimensions', 2)
    show_title = kwargs.get('show_title', None)
    
    fig = kwargs.get('fig', plt.gcf())
    
    if not isinstance(data, dict):
        data = {
            '__all__': data,
        }
    
    n_plots = sum([df.shape[1] - dimensions for df in data.values()])
    n_rows = int(np.ceil(n_plots / n_cols))
    
    gs = matplotlib.gridspec.GridSpec(n_rows, n_cols)
    
    axes = []
    plot_ix = 0
    y_min, y_max, x_min, x_max = None, None, None, None
    for split_key, split_df in data.items():
        for colorby in split_df.columns[dimensions:]:
            row = int(plot_ix / n_cols)
            col = int(plot_ix % n_cols)
            
            ax = fig.add_subplot(gs[row, col])
            axes.append(ax)
            
            embedding_plot_from_df(
                split_df, 
                colorby=colorby, 
                ax=ax, 
                **kwargs
            )
            
            if show_title is None or show_title:
                title = ''
                if not split_key.startswith('_'):
                    title += split_key + ' - '
                title += colorby
                ax.set_title(title)
            
            y_min_sub, y_max_sub = ax.get_ylim()
            y_min = y_min_sub if y_min is None else min(y_min, y_min_sub)
            y_max = y_max_sub if y_max is None else max(y_max, y_max_sub)
            
            x_min_sub, x_max_sub = ax.get_xlim()
            x_min = x_min_sub if x_min is None else min(x_min, x_min_sub)
            x_max = x_max_sub if x_max is None else max(x_max, x_max_sub)
            
            plot_ix += 1
        
    if preserve_axis_limits:
        for ax in axes:
            ax.set_xlim((x_min, x_max))
            ax.set_ylim((y_min, y_max))
    
    return fig


def paga_overlay(
    data, 
    paga_data,
    threshold=0.1,
    use_tree=False,
    categories=None,
    colors=None,
    pointsize_range=(35, 200),
    linewidth_range=(1, 5),
    line_alpha=0.6,
    point_alpha=1.0,
    ax=None
):
    ax = ax or plt.gca()
    obs_key = paga_data['obs_key']
    
    possible_categories = paga_data.get(
        'categories', 
        data[obs_key].dtype.categories,
    )
    if categories is not None:
        categories = [c for c in categories if c in possible_categories]
    else:
        categories = possible_categories
    category_ixs = {c: ix for ix, c in enumerate(possible_categories)}
    
    colors_dict = category_colors(
        categories=categories,
        colors=colors,
    )

    colors = [colors_dict[c] for c in categories]
    
    category_sizes = []
    centroids_x = []
    centroids_y = []
    for category in categories:
        data_sub = data.loc[data[obs_key] == category, :]
        x = data_sub.iloc[:, 0]
        y = data_sub.iloc[:, 1]
        centroids_x.append(trim_mean(x, 0.1))
        centroids_y.append(trim_mean(y, 0.1))
        category_sizes.append(data_sub.shape[0])
    
    #
    # lines
    #
    if not use_tree:
        connectivities = paga_data['connectivities'].toarray().copy()
    else:
        connectivities = paga_data['connectivities_tree'].toarray().copy()
    connectivities[connectivities < threshold] = 0
    
    # subset connectivities
    ixs = np.array([category_ixs[c] for c in categories])
    connectivities = connectivities[ixs, :]
    connectivities = connectivities[:, ixs]
    
    linewidths = connectivities * (linewidth_range[1] - linewidth_range[0]) + linewidth_range[0]
    
    for i in range(len(categories)):
        for j in range(i, len(categories)):
            if connectivities[i, j] == 0:
                continue
            
            ax.plot(
                [centroids_x[i], centroids_x[j]], 
                [centroids_y[i], centroids_y[j]],
                color='black',
                linewidth=linewidths[i, j],
                alpha=line_alpha,
                solid_capstyle='round',
            )
    
    #
    # scatter
    #
    min_size, max_size = min(category_sizes), max(category_sizes)
    sizes_norm = [
        (s - min_size) / (max_size - min_size)
        for s in category_sizes
    ]
    sizes = [
        s * (pointsize_range[1] - pointsize_range[0]) + pointsize_range[0]
        for s in sizes_norm
    ]
    
    ax.scatter(
        centroids_x,
        centroids_y,
        s=sizes,
        c=colors,
        zorder=2,
        alpha=point_alpha,
    )

    return ax


def barcode_from_embedding_plot(
    vdata, 
    *args, 
    **kwargs
):
    kwargs.setdefault('picker', True)
    kwargs.setdefault('pickradius', 2)
    kwargs.setdefault('simple_axes', False)
    kwargs.setdefault('obsm_key', 'X_umap')
    
    obsm_key = kwargs.pop('obsm_key')
    
    df = embedding_plot_data(
        vdata,
        obsm_key=obsm_key,
        view_key=kwargs.pop('view_key', None),
        colorby=kwargs.pop('colorby'),
    )
    ax = embedding_plot_from_df(
        df,
        **kwargs,
    )
    fig = ax.figure
    fig.show()

    while True:
        if not plt.fignum_exists(fig.number):
            return None
        coords = plt.ginput(1)
        barcodes = find_cells_from_coords(vdata, obsm_key, coords[0][0], coords[0][1])
        if len(barcodes) > 0:
            plt.close(fig)
            return barcodes[0]
        time.sleep(0.5)