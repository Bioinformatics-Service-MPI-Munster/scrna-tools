import collections
import os
import scanpy as sc
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import seaborn as sns
from collections import defaultdict
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.colors as mcol
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from .r import slingshot as slingshot_r
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

def slingshot(adata, **kwargs):
    return slingshot_r(adata, **kwargs)
