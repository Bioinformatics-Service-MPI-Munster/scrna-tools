from anndataview import AnnDataView
from functools import wraps
import logging

from ..process import rename_groups, gene_stats
from ..plotting import color_cycle, embedding_plot

logger = logging.getLogger(__name__)


class VData(AnnDataView):
    @wraps(rename_groups)
    def rename_groups(self, obs_key, rename_dict, key_added=None):
        return rename_groups(self, 
                             obs_key=obs_key, 
                             rename_dict=rename_dict, 
                             key_added=key_added)
    
    def default_colors(
        self, 
        groupby, 
        colors=None, 
        missing_color=None,
    ):
        groups = self.obs[groupby].dtype.categories
        
        if colors is not None:
            if not isinstance(colors, dict):
                raise ValueError('colors must be a dictionary!')
            
            if '__default_colors__' not in self.uns_keys():
                self.uns['__default_colors__'] = dict()
            
            alt_colors = color_cycle()
            final_colors = []
            for group in groups:
                if group not in colors:
                    if missing_color is not None:
                        final_colors.append(missing_color)
                    else:
                        color = next(alt_colors)
                        logger.warning(f"Group '{group}' not in colors dictionary, "
                                       f"setting default color {color}")
                        final_colors.append(color)
                else:
                    final_colors.append(colors[group])
            self.uns['__default_colors__'][groupby] = final_colors
        
        try:
            default_colors_list = self.uns['__default_colors__'][groupby]
            d = {}
            for i, group in enumerate(groups):
                d[group] = default_colors_list[i]
            return d
        except KeyError:
            raise KeyError(f"No default colors found for '{groupby}'")
        except IndexError:
            raise IndexError(f"Stored default color list incompatible with {groupby} - "
                              "Did the categories change at some point?")
    
    def default_colormap(
        self,
        colorby=None,
        cmap=None,
        default='viridis',
    ):
        colorby = colorby or '__default__'
        
        if cmap is not None:
            if not isinstance(cmap, (str, bytes)):
                raise ValueError('cmap must be a string!')
            
            if '__default_colormap__' not in self.uns_keys():
                self.uns['__default_colormap__'] = dict()
            
            self.uns['__default_colormap__'][colorby] = cmap
            return cmap
        else:
            try:
                return self.uns['__default_colormap__'][colorby]
            except KeyError:
                return default
    
    @wraps(gene_stats)
    def gene_stats(self, *args, **kwargs):
        return gene_stats(self, *args, **kwargs)

    @wraps(embedding_plot)
    def umap(self, colorby=None, **kwargs):
        return embedding_plot(self, colorby=colorby, key='umap', **kwargs)
    
    @wraps(embedding_plot)
    def tsne(self, colorby=None, **kwargs):
        return embedding_plot(self, colorby=colorby, key='tsne', **kwargs)

    @wraps(embedding_plot)
    def embedding_plot(self, key, colorby=None, **kwargs):
        return embedding_plot(self, colorby=colorby, key=key, **kwargs)
