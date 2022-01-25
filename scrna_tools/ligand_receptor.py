import pandas as pd
from pandas.testing import assert_series_equal
import numpy as np
import shutil
import os
from .base import ConstrainedAdata
import subprocess
from anndata import AnnData
from .helpers import mkdir, ortholog_converter, to_loom, ortholog_dict
from .r import r_run
import tempfile
from future.utils import string_types


def cellphonedb(adata, groupby, output_folder, is_log=True,
                ortholog_file=None,
                sample_column=None, project_name='project', virtualenv=None):
    if not isinstance(adata, ConstrainedAdata):
        cadata = ConstrainedAdata(adata)
    else:
        cadata = adata.copy()

    X = cadata.Xarray
    # remove log transform
    if is_log:
        X = np.expm1(X)

    # transpose for cellphonedb orientation
    df_expr_matrix = pd.DataFrame(X.T)
    df_expr_matrix.columns = cadata.obs.index
    gene_names = cadata.var.index.to_list()
    if ortholog_file is not None:
        with open(ortholog_file, 'r') as f:
            human_genes = set()
            converter = {}
            for line in f:
                line = line.rstrip()
                if line == '':
                    continue

                old, human = line.split("\t")

                i = 0
                while human in human_genes:
                    human = '{}_{}'.format(human, i)
                    i += 1
                converter[old] = human
                human_genes.add(human)
        gene_names = [converter.get(g, g) for g in gene_names]
    df_expr_matrix.index = gene_names

    meta_data = {'cell': list(cadata.obs.index),
                 'cell_type': list(cadata.obs[groupby])}
    if sample_column is not None:
        meta_data['sample'] = cadata.obs[sample_column]
    df_meta = pd.DataFrame(data=meta_data)
    df_meta.set_index('cell', inplace=True)

    tmp_folder = None
    try:
        tmp_folder = tempfile.mkdtemp()
        print(tmp_folder)
        matrix_file = os.path.join(tmp_folder, 'mat.tsv')
        df_expr_matrix.to_csv(matrix_file, sep='\t')
        meta_file = os.path.join(tmp_folder, 'meta.tsv')
        df_meta.to_csv(meta_file, sep='\t')
        tmp_output_folder = os.path.join(tmp_folder, 'out')

        with open(os.path.join(tmp_folder, 'command.sh'), 'w') as command_file:
            if virtualenv is not None:
                command_file.write('eval "$(pyenv init -)"\n')
                command_file.write('export -f pyenv\n')
                command_file.write("pyenv shell {}\n".format(virtualenv))

            command_file.write("cellphonedb method statistical_analysis --counts-data hgnc_symbol "
                               "--output-path {out} --project-name {project} "
                               "{meta} {mat}\n".format(out=tmp_output_folder, project=project_name,
                                                       meta=meta_file, mat=matrix_file))
            command_file.flush()

            with open(os.path.join(output_folder, '{}_cpdb_log.txt'.format(project_name)), mode='w') as log:
                res = subprocess.call(["sh", command_file.name], stdout=log, stderr=log)
            if res != 0:
                raise RuntimeError("Cellphonedb had non-zero exit status")

        output_folder = mkdir(output_folder)
        shutil.move(os.path.join(tmp_output_folder, project_name), output_folder)
    finally:
        if tmp_folder is not None:
            shutil.rmtree(tmp_folder)
            pass


def cellphonedb_differential_interactions_between_celltypes(
    pvalues_df, means_df, main_group, interacting_group, comparison_groups,
    pvalue_cutoff=.1,
):
    if isinstance(pvalues_df, string_types):
        pvalues_df = pd.read_csv(pvalues_df, sep="\t")
    
    if isinstance(means_df, string_types):
        means_df = pd.read_csv(means_df, sep="\t")
    
    # find LR combinations that have significant interactions
    # in any of the comparison celltype combinations
    sig_forward_comparison = np.array([False] * pvalues_df.shape[0])
    sig_reverse_comparison = np.array([False] * pvalues_df.shape[0])
    for group in comparison_groups:
        forward = '{}|{}'.format(group, interacting_group)
        sig_forward_sub = pvalues_df[forward].to_numpy() < pvalue_cutoff
        sig_forward_comparison = np.logical_or(sig_forward_comparison, sig_forward_sub)
        
        reverse = '{}|{}'.format(interacting_group, group)
        sig_reverse_sub = pvalues_df[reverse].to_numpy() < pvalue_cutoff
        sig_reverse_comparison = np.logical_or(sig_reverse_comparison, sig_reverse_sub)
    
    # find singificant LR interactions in main celltype population of interest
    sig_forward = pvalues_df['{}|{}'.format(main_group, interacting_group)].to_numpy() < pvalue_cutoff
    sig_reverse = pvalues_df['{}|{}'.format(interacting_group, main_group)].to_numpy() < pvalue_cutoff
    
    # find only those LR interactions the are different between main and comparison groups
    forward_diff = np.logical_not(sig_forward == sig_forward_comparison)
    reverse_diff = np.logical_not(sig_reverse == sig_reverse_comparison)
    
    pvalues_df_diff = pvalues_df.iloc[np.logical_or(forward_diff, reverse_diff), :]
    means_df_diff = means_df.iloc[np.logical_or(forward_diff, reverse_diff), :]
    
    return pvalues_df_diff, means_df_diff


def nichenet_ligand_activities(adata, groupby, geneset_of_interest, 
                               sender_populations, receiver_population,
                               orthologs_file=None,
                               only_literature_supported_lr=False,
                               min_expressed_cells_fraction=.1,
                               ligand_target_matrix_file='url("https://zenodo.org/record/3260758/files/ligand_target_matrix.rds")',
                               lr_network_file='url("https://zenodo.org/record/3260758/files/lr_network.rds")',
                               weighted_networks_file='url("https://zenodo.org/record/3260758/files/weighted_networks.rds")',
                               ):
    assert isinstance(adata, AnnData)
    
    adata = adata.copy()
    
    obs_keys = [groupby]
    var_keys = []
    
    if isinstance(sender_populations, string_types):
        sender_populations = [sender_populations]
    
    if orthologs_file is not None:
        adata.var.index = ortholog_converter(adata.var.index, orthologs_file)
        geneset_of_interest = ortholog_converter(geneset_of_interest, orthologs_file)
    
    loom_tmp_file_name = None
    with tempfile.NamedTemporaryFile(delete=False, suffix='.loom') as loom_tmp_file:
        loom_tmp_file.flush()
        loom_tmp_file_name = loom_tmp_file.name
    
    try:
        to_loom(adata, 
                output_file=loom_tmp_file_name,
                obs_keys=obs_keys,
                var_keys=var_keys)
        
        r_code = '''
            # prepare workspace
            installed_packages = rownames(installed.packages())
            
            if(!is.element("tidyverse", installed_packages)) {{
                install.packages("tidyverse")
            }}
            
            if(!is.element("Seurat", installed_packages)) {{
                if (!requireNamespace("BiocManager", quietly = TRUE))
                    install.packages("BiocManager")

                BiocManager::install("Seurat", update=FALSE);
            }}
            
            if(!is.element("SeuratDisk", installed_packages)) {{
                if (!requireNamespace("remotes", quietly = TRUE))
                    install.packages("remotes")
                remotes::install_github("mojaveazure/seurat-disk")
            }}
            
            if(!is.element("nichenetr", installed_packages)) {{
                if (!requireNamespace("devtools", quietly = TRUE))
                    install.packages('devtools')
                library(devtools)
                devtools::install_github("saeyslab/nichenetr")
            }}
            
            library(SeuratDisk)
            library(nichenetr)
            library(Seurat)
            library(tidyverse)
            
            # load NicheNet data
            ligand_target_matrix = readRDS({ligand_target_matrix_file})
            lr_network = readRDS({lr_network_file})
            weighted_networks = readRDS({weighted_networks_file})
            ligands = lr_network %>% pull(from) %>% unique()
            receptors = lr_network %>% pull(to) %>% unique()
            
            if ({only_literature_supported_lr}) {{
                lr_network = lr_network %>% filter(database != "ppi_prediction_go" & database != "ppi_prediction")
            }}
            
            # load single cell data
            loom = Connect('{loom_file}', mode='r')
            seu <- as.Seurat(loom)
            Idents(seu) = '{groupby}'
            expression = GetAssayData(seu, 'counts')
            sample_info = seu@meta.data
            
            geneset_oi = c({geneset_oi})
            sender_populations = c({sender_populations})
            receiver_population = '{receiver_population}'
            
            # get expressed genes in relevant populations
            expressed_genes_receiver = get_expressed_genes(receiver_population, seu, pct = {min_expressed_cells_fraction})
            
            list_expressed_genes_sender = sender_populations %>% unique() %>% lapply(get_expressed_genes, seu, {min_expressed_cells_fraction})
            expressed_genes_sender = list_expressed_genes_sender %>% unlist() %>% unique()

            background_expressed_genes = expressed_genes_receiver %>% .[. %in% rownames(ligand_target_matrix)]
            
            # Define set of potential (expressed) ligands
            expressed_ligands = intersect(ligands, expressed_genes_sender)
            expressed_receptors = intersect(receptors, expressed_genes_receiver)
            lr_network_expressed = lr_network %>% filter(from %in% expressed_ligands & to %in% expressed_receptors)
            potential_ligands = lr_network_expressed %>% pull(from) %>% unique()
            
            # Run ligand activity prediction
            ligand_activities = predict_ligand_activities(geneset = geneset_oi,
                                                          background_expressed_genes = background_expressed_genes, 
                                                          ligand_target_matrix = ligand_target_matrix, 
                                                          potential_ligands = potential_ligands)
            ligand_activities = ligand_activities %>% arrange(-pearson)
            
            as.data.frame(ligand_activities)
        '''.format(
            ligand_target_matrix_file='"{}"'.format(ligand_target_matrix_file) if os.path.exists(ligand_target_matrix_file) else ligand_target_matrix_file,
            lr_network_file='"{}"'.format(lr_network_file) if os.path.exists(lr_network_file) else lr_network_file,
            weighted_networks_file='"{}"'.format(weighted_networks_file) if os.path.exists(weighted_networks_file) else weighted_networks_file,
            only_literature_supported_lr = 'T' if only_literature_supported_lr else 'F',
            loom_file=loom_tmp_file_name,
            groupby=groupby,
            geneset_oi=", ".join(['"{}"'.format(g) for g in geneset_of_interest]),
            sender_populations=", ".join(['"{}"'.format(p) for p in sender_populations]),
            receiver_population=receiver_population,
            min_expressed_cells_fraction=min_expressed_cells_fraction,
        )
        print(r_code)
        
        results = r_run(r_code)
    finally:
        if loom_tmp_file_name is not None:
            os.remove(loom_tmp_file_name)
    
    return pd.DataFrame.from_records(results)


def _cpdb_matrix_to_df(matrix_file):
    columns = ['ligand', 'receptor', 'ligand_celltype', 'receptor_celltype', 'secreted', 'integrin', 'value']
    data = []
    
    celltype_pairs = None
    celltype_cols = dict()
    with open(matrix_file, 'r') as f:
        for i, line in enumerate(f):
            line = line.rstrip()
            fields = line.split("\t")
            if celltype_pairs is None:
                celltype_pairs = []
                for col in range(11, len(fields)):
                    ct1, ct2 = fields[col].split("|")
                    celltype_pairs.append((ct1, ct2))
                    celltype_cols[(ct1, ct2)] = col
            else:
                gene_a, gene_b = fields[4], fields[5]
                is_receptor_a, is_receptor_b = fields[7] == 'True', fields[8] == 'True'
                
                for col in range(11, len(fields)):
                    ix = col - 11
                    # check which gene is receptor
                    if is_receptor_a and not is_receptor_b:
                        ligand, receptor = gene_b, gene_a
                        ct2, ct1 = celltype_pairs[ix]
                    else:
                        ligand, receptor = gene_a, gene_b
                        ct1, ct2 = celltype_pairs[ix]
                    
                    data.append([ligand, receptor, ct1, ct2, fields[6] == 'True', 
                                fields[10] == 'True', float(fields[col])])
    
    df = pd.DataFrame(data, columns=columns)
    return df


def cpdb_to_df(pvalues_file, means_file,
               ortholog_file=None, invert_orthologs=False):
    if ortholog_file is not None:
        oc = ortholog_dict(ortholog_file, invert=invert_orthologs)
    else:
        oc = dict()
    
    p_df = _cpdb_matrix_to_df(pvalues_file)
    m_df = _cpdb_matrix_to_df(means_file)
    
    try:
        assert_series_equal(p_df['ligand'], m_df['ligand'])
        assert_series_equal(p_df['receptor'], m_df['receptor'])
        assert_series_equal(p_df['receptor_celltype'], m_df['receptor_celltype'])
        assert_series_equal(p_df['ligand_celltype'], m_df['ligand_celltype'])
        assert_series_equal(p_df['secreted'], m_df['secreted'])
        assert_series_equal(p_df['integrin'], m_df['integrin'])
    except AssertionError:
        raise ValueError("pvalues and means file are in different order. This "
                         "could mean that they originate from different analyses!")
    
    df = p_df.copy()
    df['mean'] = m_df['value']
    df.columns = list(df.columns[:6]) + ['pvalue', 'mean']
    
    df['ligand'] = [oc.get(g, g) for g in df['ligand']]
    df['receptor'] = [oc.get(g, g) for g in df['receptor']]
    
    return df


def append_lr_annotations(cpdb_df, lr_annotations, annotation_columns=('pathway', 'annotation')):
    if isinstance(lr_annotations, string_types):
        lr_annotations = pd.read_csv(lr_annotations, sep="\t", index_col=0)
    
    annotations = dict()
    for row in lr_annotations.itertuples():
        annotations[(row.ligand, row.receptor)] = [getattr(row, a) for a in annotation_columns]
    
    added_annotations = []
    for row in cpdb_df.itertuples():
        a = annotations.get((row.ligand, row.receptor), ['NA', 'NA'])
        added_annotations.append(a)
    
    a_df = pd.DataFrame(added_annotations, columns=annotation_columns)
    
    return pd.concat([cpdb_df, a_df], axis=1)


def append_expression_information(cpdb_df, adata, groupby, groups=None):
    if groups is None:
        groups = list(adata.obs[groupby].dtype.categories)
    group_info = adata.obs[groupby]
    
    ligands = list(cpdb_df['ligand'].unique())
    receptors = list(cpdb_df['receptor'].unique())
    genes = list(set(ligands).union(set(receptors)))
    try:
        genes.remove('')
    except KeyError:
        pass
    
    gene_expression_by_group = dict()
    fraction_expressed_by_group = dict()
    for group in groups:
        gene_expression_by_group[group] = dict()
        fraction_expressed_by_group[group] = dict()
        
        adata_group = adata[adata.obs[groupby] == group]
        adata_genes = adata_group[:, genes]
        
        means = np.nanmean(adata_genes.X.toarray(), axis=0)
        fractions = np.sum(adata_genes.X.toarray() > 0, axis=0)/adata_genes.shape[0]
        for i, gene in enumerate(genes):
            gene_expression_by_group[group][gene] = means[i]
            fraction_expressed_by_group[group][gene] = fractions[i]

    data_columns = ['mean_ligand_expression', 'mean_receptor_expression', 
                    'percent_expressed_ligand', 'percent_expressed_receptor']
    data = []
    for row in cpdb_df.itertuples():
        data.append([gene_expression_by_group[row.ligand_celltype].get(row.ligand, np.nan), 
                     gene_expression_by_group[row.receptor_celltype].get(row.receptor, np.nan),
                     fraction_expressed_by_group[row.ligand_celltype].get(row.ligand, np.nan), 
                     fraction_expressed_by_group[row.receptor_celltype].get(row.receptor, np.nan)])

    data_df = pd.DataFrame(data, columns=data_columns)
    return pd.concat([cpdb_df, data_df], axis=1)
