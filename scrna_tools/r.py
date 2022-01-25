import logging
from .plotting import color_cycle
import numpy as np

logger = logging.getLogger(__name__)

try:
    import rpy2
    import rpy2.robjects as ro
    import rpy2.robjects.packages as rpacks
    from rpy2.robjects.conversion import converter as template_converter
    from rpy2.robjects.conversion import Converter, localconverter
    from rpy2.robjects import numpy2ri
    from rpy2.robjects import pandas2ri
    numpy2ri.activate()
    pandas2ri.activate()
    template_converter += numpy2ri.converter
    template_converter += pandas2ri.converter
    with_rpy2 = True
except (ModuleNotFoundError, OSError) as e:
    logger.error("Cannot load rpy2: {}".format(e))
    with_rpy2 = False
    

def r_run(r_code):
    return ro.r(r_code)


def scran_size_factors(data_mat, input_groups, min_mean=0.1):
    
    converter = Converter('ipython conversion',
                          template=template_converter)

    logger.info("Converting variables to R")
    with localconverter(converter) as cv:
        ro.r.assign('input_groups', input_groups)
        ro.r.assign('data_mat', data_mat)

    logger.info("Running scran size factor estimation")
    size_factors = ro.r('''
    library(scran);
    BiocGenerics::sizeFactors(
      scran::computeSumFactors(
        SingleCellExperiment::SingleCellExperiment(list(
          counts=data_mat
        )), 
        clusters=input_groups, 
        min.mean={}
      )
    )
    '''.format(min_mean))
    
    return size_factors
  
  
def slingshot(cadata, groupby, output_folder, projection='X_umap', 
                  start_cluster=None, colors=None, default_color='#eeeeee',
                  output_file_prefix='slingshot'):
  converter = Converter('anndata conversion',
                        template=template_converter)

  if not isinstance(colors, dict):
      group_colors = dict()
      colors = color_cycle(colors)
      for group in cadata.obs[groupby]:
          group_colors[group] = next(colors)
  else:
      group_colors = colors

  color_vector = [group_colors.get(group, default_color) for group in cadata.obs[groupby].to_list()]

  logger.info("Converting variables to R")
  with localconverter(converter) as cv:
      ro.r.assign('logcounts', cadata.X.toarray())
      ro.r.assign('obs', cadata.obs)
      ro.r.assign('var', cadata.var)
      ro.r.assign('projection', cadata.obsm[projection])
      ro.r.assign('color_vector', np.array(color_vector))

  logger.info("Running slingshot conversion")
  command = '''
  library(SingleCellExperiment);
  if(!is.element("slingshot", rownames(installed.packages()))) {{
      if (!requireNamespace("BiocManager", quietly = TRUE))
          install.packages("BiocManager")

      BiocManager::install("slingshot", update=FALSE);
  }}
  library(slingshot);

  sce <- SingleCellExperiment(
      assays      = list(logcounts = t(logcounts)),
      colData     = obs,
      rowData     = var,
      reducedDims = list(projection = projection)
  );

  sce <- slingshot(sce, clusterLabels = '{groupby}', reducedDim = 'projection', start.clus='{start_cluster}');

  png(file=file.path('{output_folder}', '{prefix}_trajectory_{projection}.png'));
  plot(reducedDims(sce)$projection, col=color_vector, pch=16, asp=1);
  lines(SlingshotDataSet(sce), lwd=2, col='black');
  dev.off();

  png(file=file.path('{output_folder}', '{prefix}_lineages_{projection}.png'));
  plot(reducedDims(sce)$projection, col=color_vector, pch=16, asp=1);
  lines(SlingshotDataSet(sce), lwd=2, type='lineages', col='black');
  dev.off();
  
  cdata = colData(sce)
  write.table(x=cdata, file=file.path('{output_folder}', '{prefix}_slingshot_colData.txt'), sep='\t', quote=F)

  cdata
  '''.format(output_folder=output_folder, groupby=groupby, projection=projection, start_cluster=start_cluster,
              prefix=output_file_prefix)

  res = ro.r(command)
