import matplotlib.pyplot as plt
import matplotlib.colors as mcol
import numpy as np
from itertools import cycle
from pandas.api.types import is_numeric_dtype


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
    elif isinstance(colors, (str, bytes)):
        try:
            colors = plt.get_cmap(colors)
        except ValueError:
            colors = [colors]
    
    if isinstance(colors, mcol.LinearSegmentedColormap):
        colors = colors(np.arange(0, colors.N))
    elif isinstance(colors, (str, bytes, mcol.ListedColormap)):
        try:
            colors = plt.get_cmap(colors).colors
        except ValueError:
            colors = [colors]
    

    colors = cycle(colors)
    return colors


def category_colors(categories, colors=None, default_colors=None, ignore_na=False, na_values=('NA', 'NaN')):
    category_colors = dict()
    
    if colors is not None and isinstance(colors, dict) and default_colors is None:
        category_colors = colors.copy()
        colors = None
    
    if default_colors is None:
        default_colors = {}
    
    colors = color_cycle(colors)
    for category in categories:
        if ignore_na and category in na_values:
            continue
    
        if category not in category_colors:
            category_colors[category] = default_colors.get(category, next(colors))
    
    return category_colors


def get_numerical_and_annotation_columns(
    expression_df,
    annotation_columns=None,
):
    numerical_columns = []
    possible_annotation_columns = []
    for column in expression_df.columns:
        if is_numeric_dtype(expression_df[column]):
            numerical_columns.append(column)
        elif expression_df[column].dtype.name == 'category':
            possible_annotation_columns.append(column)
    
    if annotation_columns is None:
        annotation_columns = possible_annotation_columns
    else:
        for annotation_column in annotation_columns:
            if annotation_column not in possible_annotation_columns:
                raise ValueError(f"Cannot find categorical column {annotation_column} in data frame")
    
    return numerical_columns, annotation_columns


def default_r_colors(n):
    default_colors = [
        ["#F8766D"],
        ["#F8766D" "#00BFC4"],
        ["#F8766D" "#00BA38" "#619CFF"],
        ["#F8766D" "#7CAE00" "#00BFC4" "#C77CFF"],
        ["#F8766D" "#A3A500" "#00BF7D" "#00B0F6" "#E76BF3"],
        ["#F8766D" "#B79F00" "#00BA38" "#00BFC4" "#619CFF" "#F564E3"],
        ["#F8766D" "#C49A00" "#53B400" "#00C094" "#00B6EB" "#A58AFF" "#FB61D7"],
        ["#F8766D" "#CD9600" "#7CAE00" "#00BE67" "#00BFC4" "#00A9FF" "#C77CFF",
         "#FF61CC"],
        ["#F8766D" "#D39200" "#93AA00" "#00BA38" "#00C19F" "#00B9E3" "#619CFF",
         "#DB72FB" "#FF61C3"],
        ["#F8766D" "#D89000" "#A3A500" "#39B600" "#00BF7D" "#00BFC4" "#00B0F6",
         "#9590FF" "#E76BF3" "#FF62BC"],
        ["#F8766D" "#DB8E00" "#AEA200" "#64B200" "#00BD5C" "#00C1A7" "#00BADE",
         "#00A6FF" "#B385FF" "#EF67EB" "#FF63B6"],
        ["#F8766D" "#DE8C00" "#B79F00" "#7CAE00" "#00BA38" "#00C08B" "#00BFC4",
         "#00B4F0" "#619CFF" "#C77CFF" "#F564E3" "#FF64B0"],
        ["#F8766D" "#E18A00" "#BE9C00" "#8CAB00" "#24B700" "#00BE70" "#00C1AB",
         "#00BBDA" "#00ACFC" "#8B93FF" "#D575FE" "#F962DD" "#FF65AC"],
        ["#F8766D" "#E38900" "#C49A00" "#99A800" "#53B400" "#00BC56" "#00C094",
         "#00BFC4" "#00B6EB" "#06A4FF" "#A58AFF" "#DF70F8" "#FB61D7" "#FF66A8"],
    ]
    
    if n < 1 or n > len(default_colors):
        raise ValueError(f"n must be larger than 0 and smaller than {len(default_colors)}")
    
    return default_colors[n]
