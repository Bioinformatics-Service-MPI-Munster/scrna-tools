from packaging.version import Version
import matplotlib

import matplotlib.colors as mcol

if Version(matplotlib.__version__) < Version('3.9.0'):
    import matplotlib.cm as cm
    register_cmap = cm.register_cmap
else:
    cm = matplotlib.colormaps
    register_cmap = cm.register

import matplotlib.pyplot as plt

from .base_plotting import *
from .embedding import *
from .bar import *
from .celloracle import *
# from ._deprecated import *
from .helpers import *


for cmap_name, colors in [
    ('GreyBlue', ["#D3D3D3", "#0026F5"]),
    ('GreyBlueAlt', ["#bfbfbf","#2166ac"]),
    ('RedWhiteBlue', ["#B22727","#d0d0d0","#104e8b"]),
]:
    try:
        cm.get_cmap(cmap_name)
    except ValueError:
        cmap = mcol.LinearSegmentedColormap.from_list(cmap_name, colors)
        register_cmap(name=cmap_name, cmap=cmap)
    
    try:
        cm.get_cmap(f'{cmap_name}_r')
    except ValueError:
        cmap_r = mcol.LinearSegmentedColormap.from_list(f'{cmap_name}_r', colors[::-1])
        register_cmap(name=f'{cmap_name}_r', cmap=cmap_r)

