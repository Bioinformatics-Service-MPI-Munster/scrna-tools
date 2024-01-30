import matplotlib

import matplotlib.colors as mcol
import matplotlib.cm
import matplotlib.pyplot as plt

from .base_plotting import *
from .embedding import *
from .bar import *
from .celloracle import *
from ._deprecated import *
from .helpers import *


for cmap_name, colors in [
    ('GreyBlue', ["#D3D3D3", "#0026F5"]),
    ('GreyBlueAlt', ["#bfbfbf","#2166ac"]),
    ('RedWhiteBlue', ["#B22727","#d0d0d0","#104e8b"]),
]:
    try:
        matplotlib.cm.get_cmap(cmap_name)
    except ValueError:
        cm = mcol.LinearSegmentedColormap.from_list(cmap_name, colors)
        plt.register_cmap(name=cmap_name, cmap=cm)
    
    try:
        matplotlib.cm.get_cmap(f'{cmap_name}_r')
    except ValueError:
        cm_r = mcol.LinearSegmentedColormap.from_list(f'{cmap_name}_r', colors[::-1])
        plt.register_cmap(name=f'{cmap_name}_r', cmap=cm_r)

