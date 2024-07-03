import os
import scanpy as sc
from scanpy._utils import _choose_graph, get_igraph_from_adjacency
import gzip
import pandas as pd
import numpy as np
import scipy
from collections import Counter
import matplotlib.pyplot as plt
from future.utils import string_types

from .helpers import ortholog_dict, mkdir, smart_open
from.process import relayout
from .r import scran_size_factors

import logging
logger = logging.getLogger(__name__)


def read_barcodes(
    barcodes_file, 
    barcodes_suffix=None
):
    if barcodes_suffix is None:
        barcodes_suffix = ''
    else:
        barcodes_suffix = '_' + barcodes_suffix
    
    if isinstance(barcodes_file, (str, bytes)):
        barcodes = []
        with smart_open(barcodes_file, mode='r') as bf:
            for line in bf:
                if isinstance(line, bytes):
                    line = line.decode('utf-8')
                
                line = line.rstrip()
                if line == '' or line.startswith("#"):
                    continue
                barcode = line
                barcodes.append(barcode + barcodes_suffix)
    elif isinstance(barcodes_file, (list, tuple)):
        barcodes = [b + barcodes_suffix for b in barcodes_file]
    else:
        raise ValueError("barcodes_file must be a string (file name) or a list of strings")

    return barcodes


def read_star_data(
    matrix_mtx_file, 
    barcodes_file, 
    features_file, 
    barcodes_suffix=None, 
    use_gene_ids_as_index=False
):
    adata = sc.read_mtx(matrix_mtx_file)

    gene_ids = []
    gene_names = []
    with smart_open(features_file, mode='r') as ff:
        for line in ff:
            if isinstance(line, bytes):
                line = line.decode('utf-8')
            
            line = line.rstrip()
            if line == '' or line.startswith("#"):
                continue

            fields = line.split("\t")
            gene_ids.append(fields[0])
            if len(fields) > 1:
                gene_names.append(fields[1])

    if len(gene_names) == len(gene_ids):
        if not use_gene_ids_as_index:
            var = pd.DataFrame(data={'ensembl': gene_ids, 'symbol': gene_names},
                               index=gene_names)
        else:
            var = pd.DataFrame(data={'ensembl': gene_ids, 'symbol': gene_names},
                               index=gene_ids)
    else:
        var = pd.DataFrame(index=gene_ids)
    
    barcodes = read_barcodes(
        barcodes_file, 
        barcodes_suffix=barcodes_suffix
    )
    obs = pd.DataFrame(index=barcodes)

    final = sc.AnnData(adata.X.T, obs=obs, var=var)
    final.var_names_make_unique()
    return final


def knee_plot(adata, ax=None):
    if ax is None:
        ax = plt.gca()

    logger.info("Calculating stats")
    count_depth_per_cell = np.asarray(np.sum(adata.X, axis=1))[:, 0]
    count_depth_per_cell_sorted = sorted(count_depth_per_cell, reverse=True)
    # count_depth_per_gene = np.sum(adata.X, axis=1)
    # count_depth_per_cell_cumulative = np.cumsum(count_depth_per_cell_sorted)

    logger.info("Plotting knee plot")
    ax.plot(np.arange(1, len(count_depth_per_cell) + 1), count_depth_per_cell_sorted, color='black')
    ax.set_xlabel("Barcode rank")
    ax.set_ylabel("Count depth per cell")
    ax.set_yscale('log')
    ax.set_xscale('log')

    return ax


def count_depth_hist(adata, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()

    logger.info("Calculating stats")
    count_depth_per_cell = np.asarray(np.sum(adata.X, axis=1))[:, 0]

    kwargs.setdefault('bins', np.logspace(np.log10(1), np.log10(np.max(count_depth_per_cell)), 100))

    logger.info("Plotting histogram")
    ax.hist(count_depth_per_cell, **kwargs)
    ax.set_xlabel("Count depth per cell")
    ax.set_ylabel("Frequency")
    ax.set_xscale('log')

    return ax


def gene_count_hist(adata, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()

    logger.info("Calculating stats")
    count_depth_per_gene = np.asarray(np.sum(adata.X, axis=0))[0]

    kwargs.setdefault('bins', np.logspace(np.log10(1), np.log10(np.max(count_depth_per_gene)), 100))

    logger.info("Plotting histogram")
    ax.hist(count_depth_per_gene, **kwargs)
    ax.set_xlabel("Count depth per gene")
    ax.set_ylabel("Frequency")
    ax.set_xscale('log')

    return ax


def cumulative_count_depth_plot(adata, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()

    logger.info("Calculating stats")
    count_depth_per_cell = np.asarray(np.sum(adata.X, axis=1))[:, 0]
    count_depth_per_cell_sorted = sorted(count_depth_per_cell, reverse=True)
    count_depth_per_cell_cumulative = np.cumsum(count_depth_per_cell_sorted)

    logger.info("Plotting histogram")
    ax.plot(np.arange(1, len(count_depth_per_cell_cumulative) + 1), count_depth_per_cell_cumulative, **kwargs)
    ax.set_xlabel("Barcode rank")
    ax.set_ylabel("Cumulative count depth")
    ax.set_xscale('log')
    ax.set_yscale('log')

    return ax


def genes_vs_depth_plot(adata, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()

    kwargs.setdefault('marker', '.')
    kwargs.setdefault('color', 'black')

    count_depth_per_cell = np.asarray(np.sum(adata.X, axis=1))[:, 0]
    genes_per_cell = np.asarray(np.sum(adata.X > 0, axis=1))[:, 0]

    ax.scatter(count_depth_per_cell, genes_per_cell, **kwargs)
    ax.set_ylabel("Number of genes per cell")
    ax.set_xlabel("Count depth per cell")

    return ax


def combined_knee_and_cumulative_count_depth_plot(adata, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()

    ax2 = ax.twinx()

    cumulative_count_depth_plot(adata, ax=ax)
    knee_plot(adata, ax=ax2)
    
    fig = ax.figure
    
    def onpick1(event):
        ind = event.ind
        print('click location:', ind)

    fig.canvas.mpl_connect('pick_event', onpick1)

    return ax, ax2


def cell_cycle_genes(mouse_naming_scheme=True, source='tirosh', orthologs=None):
    if source == 'tirosh':
        s_genes = "MCM5;PCNA;TYMS;FEN1;MCM7;MCM4;RRM1;UNG;GINS2;MCM6;CDCA7;DTL;" \
                  "PRIM1;UHRF1;CENPU;HELLS;RFC2;POLR1B;NASP;RAD51AP1;GMNN;WDR76;" \
                  "SLBP;CCNE2;UBR7;POLD3;MSH2;ATAD2;RAD51;RRM2;CDC45;CDC6;EXO1;" \
                  "TIPIN;DSCC1;BLM;CASP8AP2;USP1;CLSPN;POLA1;CHAF1B;MRPL36;E2F8".split(";")
        g2m_genes = "HMGB2;CDK1;NUSAP1;UBE2C;BIRC5;TPX2;TOP2A;NDC80;CKS2;NUF2;" \
                    "CKS1B;MKI67;TMPO;CENPF;TACC3;PIMREG;SMC4;CCNB2;CKAP2L;CKAP2;" \
                    "AURKB;BUB1;KIF11;ANP32E;TUBB4B;GTSE1;KIF20B;HJURP;CDCA3;JPT1;" \
                    "CDC20;TTK;CDC25C;KIF2C;RANGAP1;NCAPD2;DLGAP5;CDCA2;CDCA8;ECT2;" \
                    "KIF23;HMMR;AURKA;PSRC1;ANLN;LBR;CKAP5;CENPE;CTCF;NEK2;G2E3;" \
                    "GAS2L3;CBX5;CENPA".split(";")
    elif source == 'macosko':
        s_genes = 'ABCC5;ABHD10;ANKRD18A;ASF1B;ATAD2;BBS2;BIVM;BLM;BMI1;BRCA1;BRIP1;C5orf42;' \
                  'C11orf82;CALD1;CALM2;CASP2;CCDC14;CCDC84;CCDC150;CDC7;CDC45;CDCA5;CDKN2AIP;' \
                  'CENPM;CENPQ;CERS6;CHML;COQ9;CPNE8;CREBZF;CRLS1;DCAF16;DEPDC7;DHFR;DNA2;' \
                  'DNAJB4;DONSON;DSCC1;DYNC1LI2;E2F8;EIF4EBP2;ENOSF1;ESCO2;EXO1;EZH2;FAM178A;' \
                  'FANCA;FANCI;FEN1;GCLM;GOLGA8A;GOLGA8B;H1F0;HELLS;HIST1H2AC;HIST1H4C;INTS7;' \
                  'KAT2A;KAT2B;KDELC1;KIAA1598;LMO4;LYRM7;MAN1A2;MAP3K2;MASTL;MBD4;MCM8;MLF1IP;' \
                  'MYCBP2;NAB1;NEAT1;NFE2L2;NRD1;NSUN3;NT5DC1;NUP160;OGT;ORC3;OSGIN2;PHIP;PHTF1;' \
                  'PHTF2;PKMYT1;POLA1;PRIM1;PTAR1;RAD18;RAD51;RAD51AP1;RBBP8;REEP1;RFC2;RHOBTB3;' \
                  'RMI1;RPA2;RRM1;RRM2;RSRC2;SAP30BP;RANGAP1;RCCD1;RDH11;RNF141;SAP30;SKA3;SMC4;' \
                  'STAT1;STIL;STK17B;SUCLG2;TFAP2A;TIMP1;SEPHS1;SETD8;SFPQ;SGOL2;SHCBP1;SMARCB1;' \
                  'SMARCD1;SPAG5;SPTBN1;SRF;SRSF3;SS18;SUV420H1;TACC3;THRAP3;TLE3;TMEM138;TNPO1;' \
                  'TOMM34;TPX2'.split(";")

        g2m_genes = 'ANLN;AP3D1;ARHGAP19;ARL4A;ARMC1;ASXL1;ATL2;AURKB;BCLAF1;BORA;BRD8;BUB3;' \
                    'C2orf69;C14orf80;CASP3;CBX5;CCDC107;CCNA2;CCNF;CDC16;CDC25C;CDCA2;CDCA3;' \
                    'CDCA8;CDK1;CDKN1B;CDKN2C;CDR2;CENPL;CEP350;CFD;CFLAR;CHEK2;CKAP2;CKAP2L;' \
                    'CYTH2;DCAF7;DHX8;DNAJB1;ENTPD5;ESPL1;FADD;FAM83D;FAN1;FANCD2;G2E3;GABPB1;' \
                    'GAS1;GAS2L3;H2AFX;HAUS8;HINT3;HIPK2;HJURP;HMGB2;HN1;HP1BP3;HRSP12;IFNAR1;' \
                    'IQGAP3;KATNA1;KCTD9;KDM4A;KIAA1524;KIF5B;KIF11;KIF20B;KIF22;KIF23;KIFC1;' \
                    'KLF6;KPNA2;LBR;LIX1L;LMNB1;MAD2L1;MALAT1;MELK;MGAT2;MID1;MIS18BP1;MND1;NCAPD3' \
                    ';NCAPH;NCOA5;NDC80;NEIL3;NFIC;NIPBL;NMB;NR3C1;NUCKS1;NUMA1;NUSAP1;PIF1;' \
                    'PKNOX1;POLQ;PPP1R2;PSMD11;PSRC1;PTP4A1;PTPN9;PWP1;QRICH1;RAD51C;RANGAP1;' \
                    'RBM8A;RCAN1;RERE;RNF126;RNF141;RNPS1;RRP1'.split(";")
    else:
        raise ValueError(f"Unknown source '{source}', use 'tirosh' or 'macosko' instead")

    if orthologs is not None:
        if isinstance(orthologs, string_types):
            orthologs = ortholog_dict(orthologs)

        s_genes = [orthologs.get(g, g) for g in s_genes]
        g2m_genes = [orthologs.get(g, g) for g in g2m_genes]
    else:
        if mouse_naming_scheme:
            s_genes = [g.lower().capitalize() for g in s_genes]
            g2m_genes = [g.lower().capitalize() for g in g2m_genes]

    return s_genes, g2m_genes


def activation_score_genes_marsh_et_al(mouse_naming_scheme=True, orthologs=None):
    if mouse_naming_scheme:
        genes = ['Fos', 'Junb', 'Zfp36', 'Jun', 'Hspa1a', 'Socs3', 'Rgs1', 'Egr1', 'Btg2', 
                'Fosb', 'Hist1h1d', 'Ier5', '1500015O10Rik', 'Atf3', 'Hist1h2ac', 
                'Dusp1', 'Hist1h1e', 'Folr1', 'Serpine1']
    else:
        genes = ['FOS', 'JUNB', 'ZFP36', 'JUN', 'HSPA1B', 'SOCS3', 'RGS1', 'EGR1', 'BTG2', 
                 'FOSB', 'HIST1H1D', 'IER5', '1500015O10Rik', 'ATF3', 'HIST1H2AC', 
                 'DUSP1', 'HIST1H1E', 'FOLR1', 'SERPINE1']
    
    if orthologs is not None:
        if isinstance(orthologs, string_types):
            orthologs = ortholog_dict(orthologs)

        genes = [orthologs.get(g, g) for g in genes]
    return genes


def annotate_base_stats(adata, doublets=True, mitochondrial=True, mitochondrial_prefix=None,
                        cell_cycle=True, activation_score=True, genome='mm',
                        output_folder=None, prefix=None, orthologs=None):
    if output_folder is not None:
        output_folder = mkdir(output_folder)

    if prefix is None:
        prefix = ''
    elif not prefix.endswith("_"):
        prefix = prefix + '_'

    if mitochondrial_prefix is None:
        if genome.startswith('mm') or genome.startswith('zf') or genome.startswith('dr'):
            mitochondrial_prefix = 'mt-'
        elif genome.startswith('hg') or genome.startswith('hs'):
            mitochondrial_prefix = 'MT-'

    if doublets:
        sc.external.pp.scrublet(
            adata,
            n_prin_comps=min(30, adata.shape[0] - 1)
        )
        #adata.obs['doublet_scores'], adata.obs['predicted_doublets'] = scrub.scrub_doublets()

        if output_folder is not None:
            fig = sc.external.pl.scrublet_score_distribution(adata, show=False, return_fig=True)
            fig.savefig(os.path.join(output_folder, prefix + 'doublet_scores.pdf'))
            plt.close(fig)

    if mitochondrial:
        adata.var['mt'] = adata.var_names.str.startswith(mitochondrial_prefix)
        sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

        if output_folder is not None:
            fig, axes = plt.subplots(1, 3, figsize=(10, 4))
            sc.pl.violin(adata,
                         'n_genes_by_counts',
                         jitter=0.4, multi_panel=False, ax=axes[0], save=False, show=False)
            sc.pl.violin(adata,
                         'total_counts',
                         jitter=0.4, multi_panel=False, ax=axes[1], save=False, show=False)
            sc.pl.violin(adata,
                         'pct_counts_mt',
                         jitter=0.4, multi_panel=False, ax=axes[2], save=False, show=False)
            fig.savefig(os.path.join(output_folder, prefix + 'mitochondrial.png'))
            plt.close(fig)

    if cell_cycle:
        s_genes, g2m_genes = cell_cycle_genes(
            mouse_naming_scheme=genome.startswith('mm'), 
            orthologs=orthologs
        )
        s_genes_mm_ens = adata.var_names[np.in1d(adata.var_names, s_genes)]
        g2m_genes_mm_ens = adata.var_names[np.in1d(adata.var_names, g2m_genes)]
        sc.tl.score_genes_cell_cycle(adata, s_genes=s_genes_mm_ens, g2m_genes=g2m_genes_mm_ens)
    
    if activation_score:
        stress_activation_genes = activation_score_genes_marsh_et_al(
            mouse_naming_scheme=genome.startswith('mm'), 
            orthologs=orthologs
        )
        stress_activation_genes_ens = adata.var_names[np.in1d(adata.var_names, stress_activation_genes)]
        sc.tl.score_genes(
            adata,
            gene_list=stress_activation_genes_ens,
            score_name='sc_dissociation_signature_score',
        )


def _optimise_scran_clusters(adata, groups, min_group_size=100, directed=False,
                             neighbors_key=None, obsp=None, max_iterations=100):
    adjacency = _choose_graph(adata, obsp, neighbors_key)
    g = get_igraph_from_adjacency(adjacency, directed=directed)

    original_groups = groups
    groups = np.array(list(groups).copy())
    group_counts = Counter(groups)

    i = 0
    while np.min(list(group_counts.values())) < min_group_size and i < max_iterations:
        # get smallest group:
        min_group = min(group_counts, key=group_counts.get)

        # merge with other groups and calculate modularity
        best_modularity = None
        best_group = None
        for group in group_counts.keys():
            if group == min_group:
                continue
            new_groups = groups.copy()
            new_groups[groups == min_group] = group

            m = g.modularity([int(i) for i in new_groups])
            if best_modularity is None or m > best_modularity:
                best_modularity = m
                best_group = group

        groups[groups == min_group] = best_group
        group_counts = Counter(groups)
        i += 1

    return pd.Series([str(g) for g in groups], index=original_groups.index).astype('category')


def scran_norm(
    adata, 
    min_mean=0.1, 
    layer=None, 
    log=True, 
    resolution=0.5, 
    min_count=100, 
    save_raw_counts=None
):
    logger.debug("Running scran norm")
    
    print('A')
    adata_pp = adata.copy()
    if layer is not None:
        adata_pp.X = adata_pp.layers[layer]

    logger.debug("Scran preprocessing")
    print('B')
    sc.pp.normalize_per_cell(adata_pp, counts_per_cell_after=1e6)
    print('B1')
    sc.pp.log1p(adata_pp)
    print('B2')
    sc.pp.pca(adata_pp, n_comps=min(15, adata_pp.shape[0] - 1))
    print('B3')
    sc.pp.neighbors(adata_pp)
    print('B4')
    sc.tl.leiden(adata_pp, key_added='groups', resolution=resolution)
    print('B5')
    
    print('C')
    logger.debug("Scran cluster optimisation")
    adata_pp.obs['groups'] = _optimise_scran_clusters(adata_pp, adata_pp.obs['groups'], min_group_size=min_count)

    input_groups = adata_pp.obs['groups']

    print('D')
    logger.info("Number of different groups passed to scran: {}".format(input_groups.dtype.categories))
    if layer is None:
        data_mat = adata.X.toarray().T
    else:
        data_mat = adata.layers[layer].toarray().T
    data_mat = np.array(data_mat, dtype='float32')
    size_factors = scran_size_factors(data_mat, input_groups, min_mean=min_mean)

    del adata_pp
    
    if layer is None:
        if save_raw_counts or (save_raw_counts is None and 'counts' not in adata.layers.keys()):
            logger.info("Storing raw counts in layer 'counts'")
            adata.layers["counts"] = adata.X.copy()
        adata.obs['size_factors'] = size_factors
        logger.info("Doing normalisation using size factors")
        adata.X /= adata.obs['size_factors'].values[:, None]
        logger.info("Converting matrix back to sparse format")
        adata.X = scipy.sparse.csr_matrix(adata.X)
    else:
        logger.info("Storing raw counts in layer 'counts_{}'".format(layer))
        adata.layers["counts_{}".format(layer)] = adata.layers[layer].copy()
        adata.obs['size_factors_{}'.format(layer)] = size_factors
        logger.info("Doing normalisation using size factors")
        adata.layers[layer] /= adata.obs['size_factors_{}'.format(layer)].values[:, None]
        logger.info("Converting matrix back to sparse format")
        adata.layers[layer] = scipy.sparse.csr_matrix(adata.layers[layer])

    if log:
        logger.info("Log-transforming matrix")
        sc.pp.log1p(adata, layer=layer)
    return adata


def reintegrate(adata, n_top_genes=3000, npcs=100, batch_key='sample', use_batch_for_hvg=True):
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, batch_key=batch_key if use_batch_for_hvg else None)
    sc.tl.pca(adata, svd_solver='arpack', n_comps=npcs, use_highly_variable=True)
    sc.external.pp.harmony_integrate(adata, key=batch_key, max_iter_harmony=50)
    sc.pp.neighbors(adata, n_pcs=npcs, use_rep='X_pca_harmony')
    sc.tl.umap(adata)


def merge_adata(*adatas, join='outer', batch_correct=True, 
                log=False, raw=False, index_unique='-', **kwargs):
    if raw:
        adata = adatas[0].raw
        for i in range(1, len(adatas)):
            adata = adata.concatenate(adatas[i].raw, join=join, index_unique=index_unique)
    else:
        adata = adatas[0].concatenate(adatas[1:], join=join, index_unique=index_unique)

    #sc.pp.filter_genes(adata, min_cells=10)
    if log:
        sc.pp.log1p(adata)

    if batch_correct:
        relayout(adata, **kwargs)

    return adata
