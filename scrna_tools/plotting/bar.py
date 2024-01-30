import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import seaborn as sns
from .base_plotting import category_colors
from .helpers import get_numerical_and_annotation_columns

from datetime import datetime


def grouped_barplot_from_df(
    expression_df,
    colors=None,
    bar_width=0.3,
    gap_size=0.03,
    # legend=False, 
    # legend_inside=True,
    ylim=None, 
    shuffle_seed=42,
    ax=None, 
    # lax=None, 
    # legend_kwargs={},
    value_key=None,
    group_by=None,
    **kwargs
):
    time = datetime.now()
    
    kwargs.setdefault('linewidths', [bar_width])
    kwargs.setdefault('linestyle', 'solid')
    
    ax = ax or plt.gca()
    #lax = lax or ax
    value_key = value_key or expression_df.columns[0]
    
    numerical_columns, annotation_columns = get_numerical_and_annotation_columns(
        expression_df, 
        annotation_columns=group_by
    )
    
    value_key = numerical_columns[0]
    group_key = None
    if len(annotation_columns) > 0:
        group_key = annotation_columns[0]
    
    # legend_elements = []
    # legend_order = None
    print(group_key)
    if group_key in expression_df:
        categories = expression_df[group_key].dtype.categories
        gb = expression_df.groupby(group_key)
        dfs = {}
        for category in categories:
            try:
                dfs[category] = gb.get_group(category)
            except KeyError:
                continue
        palette = category_colors(categories, colors)
        #legend_order = categories
    else:
        dfs = {'__all__': expression_df.copy()}
        palette = category_colors(['__all__'], colors)
    
    if shuffle_seed is not None:
        dfs = {category: df.sample(frac=1, random_state=shuffle_seed) for category, df in dfs.items()}
    
    offset = 0
    total_values = sum(df.shape[0] for df in dfs.values())
    gap = gap_size * total_values
    total_len = total_values + gap * (len(dfs) - 1)
    
    xticks = []
    xticklabels = []
    for i, (category, df) in enumerate(dfs.items()):
        offset_and_gap = offset + i * gap
        
        xs = np.arange(offset_and_gap, offset_and_gap + df.shape[0])
        ys = df[value_key]
        segments = [[(x, 0), (x, y)] for x, y in zip(xs, ys)]

        line_segments = LineCollection(
            segments,
            color=palette[category],
            **kwargs
        )
        ax.add_collection(line_segments)
        
        # ticks
        xticks.append(offset_and_gap + df.shape[0]/2)
        if not category.startswith('_'):
            xticklabels.append(f'{category}\n{df.shape[0]}')
        else:
            xticklabels.append(f'{df.shape[0]}')
        
        # bottom bar
        ax.axhline(
            y=0, 
            xmin=offset_and_gap/total_len, 
            xmax=(offset_and_gap + df.shape[0])/total_len,
            color='black',
        )
        offset += df.shape[0]
    
    ax.set_xlim((0, total_len))
    ax.set_ylim(ylim or (expression_df[value_key].min(), expression_df[value_key].max()))
    
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    
    ax.set_ylabel(value_key)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    print((datetime.now() - time).total_seconds())
    
    return ax
