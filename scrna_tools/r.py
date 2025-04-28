import os
import logging

from .plotting import color_cycle
from .process import pseudobulk_expression_matrix_and_annotations, de_df_to_parquet

import numpy as np
import pandas as pd
import json

import pyarrow
import pyarrow.parquet as pq
from collections import defaultdict
import matplotlib.colors
import warnings
from functools import wraps
from collections import Counter

logger = logging.getLogger(__name__)

try:
    import rpy2
    import rpy2.robjects as ro
    import rpy2.robjects.packages as rpacks
    from rpy2.robjects.conversion import converter as template_converter
    from rpy2.robjects.conversion import Converter, localconverter
    from rpy2.robjects import numpy2ri
    from rpy2.robjects import pandas2ri
    
    template_converter += numpy2ri.converter
    template_converter += pandas2ri.converter
    with_rpy2 = True
except (ModuleNotFoundError, OSError) as e:
    with_rpy2 = False
    

def r_run(r_code):
    return ro.r(r_code)


def requires_rpy2(func):
    """Checks if rpy2 is installed in path"""
    
    @wraps(func)
    def wrapper_rpy2(*args, **kwargs):
        if not with_rpy2:
            raise RuntimeError("rpy2 is not installed, cannot "
                               "run code that depends on R!")
        return func(*args, **kwargs)
    return wrapper_rpy2


@requires_rpy2
def scran_size_factors(data_mat, input_groups, min_mean=0.1):
    converter = Converter('ipython conversion',
                          template=template_converter)

    logger.info("Converting variables to R")
    with localconverter(converter) as cv:
        ro.r.assign('input_groups', input_groups)
        ro.r.assign('data_mat', data_mat)

    logger.info("Running scran size factor estimation")
    size_factors = ro.r('''
    print("SCRAN in R")
    if(!is.element("scran", rownames(installed.packages()))) {{
        if (!requireNamespace("BiocManager", quietly = TRUE))
            install.packages("BiocManager")

        BiocManager::install("scran", update=FALSE);
    }}
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


@requires_rpy2
def slingshot(
    vdata, 
    groupby, 
    output_folder, 
    projection='X_umap',
    start_cluster=None, 
    colors=None, 
    default_color='#eeeeee',
    output_file_prefix='slingshot', 
    key_prefix=None, 
    add_obs=True,
    view_key=None
):
    converter = Converter('anndata conversion',
                          template=template_converter)

    if not isinstance(colors, dict):
        group_colors = dict()
        colors = color_cycle(colors)
        for group in vdata.obs(view_key=view_key)[groupby]:
            group_colors[group] = next(colors)
    else:
        group_colors = colors

    color_vector = []
    for group in vdata.obs(view_key=view_key)[groupby].to_list():
        c = group_colors.get(group, default_color)
        if isinstance(c, (str, bytes)) and not c.startswith('#'):
            c = matplotlib.colors.cnames[c]
        color_vector.append(c)

    logger.info("Converting variables to R")
    with localconverter(converter) as cv:
        ro.r.assign('logcounts', vdata.X.toarray())
        ro.r.assign('obs', vdata.obs(view_key=view_key).loc[:, [groupby]])
        ro.r.assign('var', vdata.var(view_key=view_key))
        ro.r.assign('projection', vdata.obsm(view_key=view_key)[projection])
        ro.r.assign('color_vector', np.array(color_vector))

        logger.info("Running slingshot conversion")
        command = '''
        if(!is.element("SingleCellExperiment", rownames(installed.packages()))) {{
            if (!requireNamespace("BiocManager", quietly = TRUE))
                install.packages("BiocManager")

            BiocManager::install("SingleCellExperiment", update=FALSE);
        }}
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

        sce <- slingshot(
            sce, 
            clusterLabels = '{groupby}', 
            reducedDim = 'projection', 
            start.clus='{start_cluster}',
            extend = 'n',
        );

        png(file=file.path('{output_folder}', '{prefix}_trajectory_{projection}.png'));
        plot(reducedDims(sce)$projection, col=color_vector, pch=16, asp=1);
        lines(SlingshotDataSet(sce), lwd=2, col='black');
        dev.off();

        png(file=file.path('{output_folder}', '{prefix}_lineages_{projection}.png'));
        plot(reducedDims(sce)$projection, col=color_vector, pch=16, asp=1);
        lines(SlingshotDataSet(sce), lwd=2, type='lineages', col='black');
        dev.off();
        
        pdf(file=file.path('{output_folder}', '{prefix}_trajectory_{projection}.pdf'));
        plot(reducedDims(sce)$projection, col=color_vector, pch=16, asp=1);
        lines(SlingshotDataSet(sce), lwd=2, col='black');
        dev.off();

        pdf(file=file.path('{output_folder}', '{prefix}_lineages_{projection}.pdf'));
        plot(reducedDims(sce)$projection, col=color_vector, pch=16, asp=1);
        lines(SlingshotDataSet(sce), lwd=2, type='lineages', col='black');
        dev.off();
        
        print(head(colData(sce)[, 'slingshot']))
        
        cdata = SingleCellExperiment::colData(sce)
        cdata <- subset(cdata, select = -c(slingshot))
        write.table(x=cdata, file=file.path('{output_folder}', '{prefix}_slingshot_colData.txt'), sep='\t', quote=F)
        
        print('Writing Lineages...')
        print(file.path('{output_folder}', '{prefix}_slingshot_lineages.txt'))
        print(colData(sce)[, 'slingshot'])
        
        lapply(
            colData(sce)[, 'slingshot']@metadata$lineages, 
            write, 
            file.path('{output_folder}', '{prefix}_slingshot_lineages.txt'),
            append=T, 
            ncolumns=10000, 
            sep="\t"
        )

        as.data.frame(cdata)
        '''.format(output_folder=output_folder, groupby=groupby, projection=projection, start_cluster=start_cluster,
                    prefix=output_file_prefix)

        res = ro.r(command)
    
    if add_obs:
        for column in res.columns:
            if column.startswith('sling'):
                new_column = column if key_prefix is None else f'{key_prefix}_{column}'
                try:
                    vdata.adata.obs(view_key=view_key).loc[res.index, new_column] = res[column]
                except AttributeError:
                    vdata._parent_adata.obs.loc[res.index, new_column] = res[column]
    
    lineages = []
    with open(os.path.join(output_folder, f'{output_file_prefix}_slingshot_lineages.txt'), 'r') as f:
        for line in f:
            lineages.append(line.strip().split('\t'))
    
    return res, lineages


@requires_rpy2
def tricycle(cadata, projection='X_umap', species='mouse', identifier_type='symbol', 
             save=True, key_added_score='tricycle', key_added_phase='tricycle_phase', 
             overwrite=False):
    if save and not overwrite and key_added_score in cadata.obs.columns:
        raise ValueError(f"{key_added_score} already exists in object. Use 'overwrite=True' to overwrite data in column "
                         "or choose a different 'key_added_score'!")
    if save and not overwrite and key_added_phase in cadata.obs.columns:
        raise ValueError(f"{key_added_phase} already exists in object. Use 'overwrite=True' to overwrite data in column "
                         "or choose a different 'key_added_phase'!")

    converter = Converter('anndata conversion',
    template=template_converter)

    logger.info("Converting variables to R")
    with localconverter(converter) as cv:
        ro.r.assign('logcounts', cadata.X.toarray())
        ro.r.assign('obs', cadata.obs)
        ro.r.assign('var', cadata.var)
        ro.r.assign('projection', cadata.obsm[projection])

        logger.info("Running tricycle")
        command = '''
        if(!is.element("SingleCellExperiment", rownames(installed.packages()))) {{
            if (!requireNamespace("BiocManager", quietly = TRUE))
                install.packages("BiocManager")

            BiocManager::install("SingleCellExperiment", update=FALSE);
        }}
        library(SingleCellExperiment);
        if(!is.element("tricycle", rownames(installed.packages()))) {{
            if (!requireNamespace("BiocManager", quietly = TRUE))
                install.packages("BiocManager")

            BiocManager::install("tricycle", update=FALSE);
        }}
        library(tricycle);

        sce <- SingleCellExperiment(
            assays      = list(logcounts = t(logcounts)),
            colData     = obs,
            rowData     = var,
            reducedDims = list(projection = projection)
        );

        species = "{species}"
        sce <- project_cycle_space(sce, species="{species}", gname.type="{identifier_type}")
        sce <- estimate_cycle_position(sce)
        
        cdata = colData(sce)
        as.data.frame(cdata)[, "tricyclePosition", drop=F]
        '''.format(species=species, projection=projection, identifier_type=identifier_type.upper())

        res = ro.r(command)
        
    df = res

    df.columns = [key_added_score]
    phase = []
    for score in df[key_added_score]:
        if 0.5*np.pi <= score < np.pi:
            phase.append('S')
        elif np.pi <= score < 1.75*np.pi:
            phase.append('G2M')
        else:
            phase.append('G1')
    df[key_added_phase] = pd.Categorical(phase)

    if save:
        cadata.adata.obs[key_added_score] = [np.nan] * cadata.adata.obs.shape[0]
        cadata.adata.obs.loc[df[key_added_score].index, key_added_score] = df[key_added_score]
        cadata.adata.obs[key_added_phase] = ['NA'] * cadata.adata.obs.shape[0]
        cadata.adata.obs.loc[df[key_added_phase].index, key_added_phase] = df[key_added_phase]
        cadata.adata.obs[key_added_phase] = pd.Categorical(cadata.adata.obs[key_added_phase].to_list())

    return df


@requires_rpy2
def de_pseudobulk(
    vdata, 
    obs_key, 
    sample1, 
    sample2=None, 
    replicate_key=None, 
    n_pseudoreplicates=2, 
    layer='counts',
    min_counts_per_gene=0, 
    min_counts_per_sample=0, 
    append_gene_stats=True, 
    stats_layer=None,
    random_seed=42,
    as_parquet=False,
    view_key=None,
):
    """
    Pseudobulk differential expression using DESeq2.

    All credit goes to PG Majev.
    """
    original_key = obs_key
    
    obs_constraints = vdata.obs_constraints.copy()
    var_constraints = vdata.var_constraints.copy()
    
    if view_key is not None:
        vdata = vdata.copy()
        vdata.add_view_constraint(view_key)
    
    include_rest = sample2 is None
    
    # compare two samples (DE)    
    vdata, mat_mm, sample_annotation, pb_info = pseudobulk_expression_matrix_and_annotations(
        vdata, 
        obs_key=obs_key,
        sample1=sample1,
        sample2=sample2,
        replicate_key=replicate_key,
        n_pseudoreplicates=n_pseudoreplicates,
        layer=layer,
        random_seed=random_seed
    )
    replicate_key = pb_info['replicate_key']
    pb_obs_key = pb_info['obs_key']
    n_replicates = pb_info['n_replicates']
    sample1 = pb_info['sample1']
    sample2 = pb_info['sample2']
    
    converter = Converter('DEseq conversion', template=template_converter)

    logger.info("Running DESeq2")
    command = '''
    if(!is.element("DESeq2", rownames(installed.packages()))) {{
        if (!requireNamespace("BiocManager", quietly = TRUE))
            install.packages("BiocManager")

        BiocManager::install("DESeq2", update=FALSE);
    }}
    library(DESeq2);
    if(!is.element("IHW", rownames(installed.packages()))) {{
        if (!requireNamespace("BiocManager", quietly = TRUE))
            install.packages("BiocManager")

        BiocManager::install("IHW", update=FALSE);
    }}
    library(IHW);
    
    dds <- DESeqDataSetFromMatrix(countData = count_data, colData = sample_annotation, design = ~ {obs_key})
    
    dds <- dds[rowSums(DESeq2::counts(dds) >= {min_counts_per_gene}) >= {n_replicates},
                colSums(DESeq2::counts(dds)) >= {min_counts_per_sample}]
    
    dds <- estimateSizeFactors(dds)
    dds <- estimateDispersions(dds)
    dds <- nbinomWaldTest(dds, maxit=5000)
    dds <- dds[which(mcols(dds)$betaConv),]
    
    res <- results(dds, contrast=c("{obs_key}", "{sample1}", "{sample2}"), filterFun=ihw)
    results_df <- as.data.frame(res)
    # results_df['gene'] = rownames(dds)
    results_df
    '''.format(
        obs_key=pb_obs_key, 
        sample1=sample1, 
        sample2=sample2,
        min_counts_per_gene=min_counts_per_gene,
        min_counts_per_sample=min_counts_per_sample,
        n_replicates=n_replicates
    )
    
    logger.info("Converting variables to R")
    with localconverter(converter) as cv:
        ro.r.assign('count_data', mat_mm)
        ro.r.assign('sample_annotation', sample_annotation)
        res = ro.r(command)

    if append_gene_stats:
        stats = vdata.gene_stats(genes=res.index, groupby=original_key, 
                                 groups=[sample1] if sample2 is None else [sample1, sample2],
                                 include_rest=include_rest,
                                 sd_expression=False, layer=stats_layer)
        for column in stats.columns:
            res[column] = stats.loc[res.index, column]
    
    res = res.sort_values(by='log2FoldChange', key=abs, ascending=False, kind='mergesort')
    res = res.sort_values(by='pvalue', kind='mergesort')
    
    if as_parquet:
        return de_df_to_parquet(
            res, 
            obs_key=obs_key,
            view_key=view_key,
            categories=[sample1, sample2],
            n_replicates=n_replicates,
            replicate_key=replicate_key,
            obs_constraints=obs_constraints,
            var_constraints=var_constraints,
        )
    
    return res


@requires_rpy2
def markers_pseudobulk(
    vdata, 
    key, 
    groups=None, 
    view_key=None,
    ignore_groups=['NA', 'nan', 'NaN'],
    as_parquet=False,
    **kwargs
):
    obs_constraints = vdata.obs_constraints.copy()
    var_constraints = vdata.var_constraints.copy()
    
    if view_key is not None:
        vdata = vdata.copy()
        vdata.add_view_constraint(view_key)
    
    original_groups = vdata.obs[key].unique()
    groups = groups or [g for g in original_groups if g not in ignore_groups]
    
    if Counter(original_groups) != Counter(groups):
        vdata = vdata.copy(only_constraints=True)
        vdata.add_categorical_obs_constraint(key, groups)
        obs_constraints.append(vdata.obs_constraints[-1])
    
    all_markers = None
    for group in groups:
        group_clean = group.replace(' ', '_').replace('.', '_')
        
        markers = de_pseudobulk(vdata, key, group, **kwargs)
        markers['group'] = group
        markers = markers.rename({
            f'mean_expression_{group_clean}': 'mean_expression_group',
            f'percent_expressed_{group_clean}': 'percent_expressed_group',
        }, axis=1)

        if all_markers is None:
            all_markers = markers
        else:
            all_markers = pd.concat([all_markers, markers])
    
    if as_parquet:
        return de_df_to_parquet(
            all_markers, 
            obs_key=key,
            view_key=view_key,
            categories=groups,
            obs_constraints=obs_constraints,
            var_constraints=var_constraints,
        )
    
    return all_markers


@requires_rpy2
def cluster_distances_pseudobulk(
    vdata, 
    obs_key,
    replicate_key=None, 
    n_pseudoreplicates=2, 
    layer='counts',
    random_seed=42,
    min_counts_per_gene=0,
    min_counts_per_sample=0,
):
    vdata, mat_mm, sample_annotation, pb_info = pseudobulk_expression_matrix_and_annotations(
        vdata, 
        obs_key=obs_key,
        replicate_key=replicate_key,
        n_pseudoreplicates=n_pseudoreplicates,
        layer=layer,
        random_seed=random_seed
    )
    replicate_key = pb_info['replicate_key']
    obs_key = pb_info['obs_key']
    n_replicates = pb_info['n_replicates']
    
    converter = Converter('DEseq2 conversion', template=template_converter)

    logger.info("Running DESeq2 distances")
    command = '''
    if(!is.element("DESeq2", rownames(installed.packages()))) {{
        if (!requireNamespace("BiocManager", quietly = TRUE))
            install.packages("BiocManager")

        BiocManager::install("DESeq2", update=FALSE);
    }}
    library(DESeq2);
    
    dds <- DESeqDataSetFromMatrix(
        countData = count_data, 
        colData = sample_annotation, 
        design = ~ {obs_key}
    )
    
    dds <- dds[rowSums(DESeq2::counts(dds) >= {min_counts_per_gene}) >= {n_replicates},
               colSums(DESeq2::counts(dds)) >= {min_counts_per_sample}]
    
    dds <- estimateSizeFactors(dds)
    dds <- estimateDispersions(dds)
    vst <- DESeq2::vst(dds)
    
    sample_distances <- as.matrix(dist(t(assay(vst))))
    colnames(sample_distances) <- colnames(vst)
    rownames(sample_distances) <- NULL

    as.data.frame(sample_distances)
    '''.format(obs_key=obs_key,
                min_counts_per_gene=min_counts_per_gene,
                min_counts_per_sample=min_counts_per_sample,
                n_replicates=n_replicates)
    
    logger.info("Converting variables to R")
    with localconverter(converter) as cv:
        ro.r.assign('count_data', mat_mm)
        ro.r.assign('sample_annotation', sample_annotation)
        res = ro.r(command)
        
    return res
