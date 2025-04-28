import os
import regex as re
from future.utils import string_types
from ..helpers import smart_open, mkdir
from ._whitelist_correction import correct_rhapsody_barcode
from tqdm import tqdm
import numpy as np
import pandas as pd
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


BD_RHAPSODY_BARCODE = '(' \
    '(?P<cell_1>.{{9}}){{e<={local_mismatches}}}' \
    '(?P<discard_1>ACTGGCCTGCGA){{e<={local_mismatches}}}' \
    '(?P<cell_2>.{{9}}){{e<={local_mismatches}}}' \
    '(?P<discard_2>GGTAGCGGTGACA){{e<={local_mismatches}}}' \
    '(?P<cell_3>.{{9}}){{e<={local_mismatches}}}' \
    '){{e<={global_mismatches}}}' \
    '(?P<umi_1>.{{8}})' \
    '(TT{{7}}){{s<=2}}.*'


SAMPLE_TAGS = {
    'mm': [
        'AAGAGTCGACTGCCATGTCCCCTCCGCGGGTCCGTGCCCCCCAAG',
        'ACCGATTAGGTGCGAGGCGCTATAGTCGTACGTCGTTGCCGTGCC',
        'AGGAGGCCCCGCGTGAGAGTGATCAATCCAGGATACATTCCCGTC',
        'TTAACCGAGGCGTGAGTTTGGAGCGTACCGGCTTTGCGCAGGGCT',
        'GGCAAGGTGTCACATTGGGCTACCGCGGGAGGTCGACCAGATCCT',
        'GCGGGCACAGCGGCTAGGGTGTTCCGGGTGGACCATGGTTCAGGC',
        'ACCGGAGGCGTGTGTACGTGCGTTTCGAATTCCTGTAAGCCCACC',
        'TCGCTGCCGTGCTTCATTGTCGCCGTTCTAACCTCCGATGTCTCG',
        'GCCTACCCGCTATGCTCGTCGGCTGGTTAGAGTTTACTGCACGCC',
        'TCCCATTCGAATCACGAGGCCGGGTGCGTTCTCCTATGCAATCCC',
        'GGTTGGCTCAGAGGCCCCAGGCTGCGGACGTCGTCGGACTCGCGT',
        'CTGGGTGCCTGGTCGGGTTACGTCGGCCCTCGGGTCGCGAAGGTC',
    ],
    'hg': [
        'ATTCAAGGGCAGCCGCGTCACGATTGGATACGACTGTTGGACCGG',
        'TGGATGGGATAAGTGCGTGATGGACCGAAGGGACCTCGTGGCCGG',
        'CGGCTCGTGCTGCGTCGTCTCAAGTCCAGAAACTCCGTGTATCCT',
        'ATTGGGAGGCTTTCGTACCGCTGCCGCCACCAGGTGATACCCGCT',
        'CTCCCTGGTGTTCAATACCCGATGTGGTGGGCAGAATGTGGCTGG',
        'TTACCCGCAGGAAGACGTATACCCCTCGTGCCAGGCGACCAATGC',
        'TGTCTACGTCGGACCGCAAGAAGTGAGTCAGAGGCTGCACGCTGT',
        'CCCCACCAGGTTGCTTTGTCGGACGAGCCCGCACAGCGCTAGGAT',
        'GTGATCCGCGCAGGCACACATACCGACTCAGATGGGTTGTCCAGG',
        'GCAGCCGGCGTCGTACGAGGCACAGCGGAGACTAGATGAGGCCCC',
        'CGCGTCCAATTTCCGAAGCCCCGCCCTAGGAGTTCCCCTGCGTGC',
        'GCCCATTCATTGCACCCGCCAGTGATCGACCCTAGTGGAGCTAAG',
    ]
}

barcode_res = {
    'legacy': re.compile(
        '(?P<barcode>.{9}ACTGGCCTGCGA.{9}GGTAGCGGTGACA.{9}){{e<=2}}'
    ),
    'enhanced': re.compile(
        '(?P<barcode>.{9}GTGA.{9}GACA.{9})'
    ),
}


class BarcodeCorrect(object):
    def __init__(self, whitelist):
        if isinstance(whitelist, string_types):
            whitelist_file = whitelist
            whitelist = []
            with open(whitelist_file, 'r') as f:
                for line in f:
                    line = line.rstrip()
                    if line == '':
                        continue
                    whitelist.append(line)

        self.whitelist = set(whitelist)

    def known(self, barcodes):
        """
        The subset of `words` that appear in the dictionary of WORDS.
        """
        return set(b for b in barcodes if b in self.whitelist)

    def candidates(self, barcode):
        """
        Generate possible corrections for barcode.
        """
        return self.known([barcode]) or self.known(self.edits1(barcode)) or None

    def edits1(self, barcode):
        """
        All edits that are one edit away from `barcode`.
        """
        letters = 'ACGT'
        splits = [(barcode[:i], barcode[i:]) for i in range(len(barcode) + 1)]
        # deletes = [L + R[1:] for L, R in splits if R]
        # transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        # inserts = [L + c + R for L, R in splits for c in letters]
        # return set(deletes + transposes + replaces + inserts)
        return set(replaces)


def rhapsody_extract_barcodes(fastq_1_file, output_1_file,
                              fastq_2_file=None, output_2_file=None,
                              expected_barcode_length=27,
                              barcodes_output_file=None, whitelist_file=None,
                              whitelist_cutoff=10):
    logger.info("Maching barcode pattern to read")
    barcode_pattern = BD_RHAPSODY_BARCODE.format(local_mismatches=2,
                                                 global_mismatches=5)

    bc = None
    if whitelist_file is not None:
        bc = BarcodeCorrect(whitelist_file)

    barcode_re = re.compile(barcode_pattern)
    whitespace_re = re.compile(r"\s+")
    valid = 0
    total = 0

    barcode_counts = defaultdict(int)
    count_barcodes = barcodes_output_file is not None
    file1 = None
    file1_iter = None
    out1 = None
    file2 = None
    file2_iter = None
    out2 = None
    try:
        with smart_open(fastq_1_file, 'rb') as f:
            logger.info("Determining file size...")
            n_entries = int(sum(1 for _ in f)/4)

        file1 = smart_open(fastq_1_file, 'rb')
        file1_iter = iter(file1)
        out1 = smart_open(output_1_file, 'wb')
        if fastq_2_file is not None:
            file2 = smart_open(fastq_2_file, 'rb') if fastq_2_file is not None else None
            file2_iter = iter(file2)
            out2 = smart_open(output_2_file, 'wb')

        try:
            current_entry_1 = [None, None, None, None]
            current_entry_2 = [None, None, None, None]
            ix = 0
            with tqdm(total=n_entries, desc='Barcodes') as pb:
                while True:
                    line1 = next(file1_iter).decode('utf-8').rstrip()
                    current_entry_1[ix] = line1

                    if file2 is not None:
                        line2 = next(file2_iter).decode('utf-8').rstrip()
                        current_entry_2[ix] = line2

                    current_valid = False
                    if ix == 3:
                        total += 1
                        pb.update(total)
                        ix = -1

                        m = barcode_re.match(current_entry_1[1])
                        if m is not None:
                            current_valid = True

                            # write barcode and read to file
                            # cell
                            cell_counter = 1
                            cell_spans = []
                            try:
                                while True:
                                    g = m.span('cell_{}'.format(cell_counter))
                                    cell_spans.append(g)
                                    cell_counter += 1
                            except IndexError:
                                pass

                            barcode = "".join([current_entry_1[1][s[0]:s[1]] for s in cell_spans])
                            barcode_quality = "".join([current_entry_1[3][s[0]:s[1]] for s in cell_spans])
                            barcode_length = len(barcode)
                            if barcode_length > expected_barcode_length:
                                current_valid = False
                            if barcode_length < expected_barcode_length:
                                barcode += 'N' * (barcode_length - expected_barcode_length)
                                barcode_quality += 'A' * (barcode_length - expected_barcode_length)

                            if bc is not None:
                                candidates = bc.candidates(barcode)
                                if candidates is None or len(candidates) > 1:
                                    current_valid = False
                                else:
                                    barcode = candidates.pop()

                            umi_counter = 1
                            umi_spans = []
                            try:
                                while True:
                                    g = m.span('umi_{}'.format(umi_counter))
                                    umi_spans.append(g)
                                    umi_counter += 1
                            except IndexError:
                                pass

                            umi = "".join([current_entry_1[1][s[0]:s[1]] for s in umi_spans])
                            umi_quality = "".join([current_entry_1[3][s[0]:s[1]] for s in umi_spans])

                            discard_counter = 1
                            discard_spans = []
                            try:
                                while True:
                                    g = m.span('discard_{}'.format(discard_counter))
                                    discard_spans.append(g)
                                    discard_counter += 1
                            except IndexError:
                                pass

                            sequence = "{barcode}{umi}".format(barcode=barcode, umi=umi)
                            quality = "{barcode}{umi}".format(barcode=barcode_quality, umi=umi_quality)

                            header1 = whitespace_re.split(current_entry_1[0])
                            header1[0] = "{}_{}_{}".format(header1[0], barcode, umi)

                            if current_valid:
                                out1.write("{}\n{}\n+\n{}\n".format(" ".join(header1),
                                                                    sequence, quality).encode('utf-8'))
                                valid += 1

                                if out2 is not None:
                                    header2 = whitespace_re.split(current_entry_2[0])
                                    header2[0] = "{}_{}_{}".format(header2[0], barcode, umi)

                                    if header1[0] != header2[0]:
                                        raise ValueError("Paired FASTQ files are not "
                                                            "in the same order!\n{}\n{}".format(
                                            header1[0], header2[0]
                                        ))

                                    out2.write("{}\n{}\n+\n{}\n".format(" ".join(header2), current_entry_2[1],
                                                                        current_entry_2[3]).encode('utf-8'))

                                if count_barcodes:
                                    barcode_counts[barcode] += 1

                    ix += 1
        except StopIteration:
            pass
    finally:
        file1.close()
        out1.close()
        if file2 is not None:
            file2.close()
            out2.close()

    if barcodes_output_file is not None:
        with open(barcodes_output_file, 'w') as o:
            for barcode, count in barcode_counts.items():
                if whitelist_cutoff is None or count >= whitelist_cutoff:
                    o.write("{}\n".format(barcode))

    logger.info("Total: {}; matching barcode regex: {}".format(total, valid))

    return total, valid


def rhapsody_demultiplex(
    barcodes_files, 
    reads_files, 
    genome, 
    output_folder=None, 
    prefix='sample',
    mismatches=2, 
    hq_singlet_cutoff=0.75,
    expected_noise_plot=None, 
    sample_tag_counts_plot=None,
    statistics_file=None,
    barcode_version='enhanced',
    whitelists=None,
):
    if isinstance(barcodes_files, string_types):
        barcodes_files = [barcodes_files]
    
    if isinstance(reads_files, string_types):
        reads_files = [reads_files]
    
    if len(barcodes_files) != len(reads_files):
        raise ValueError("Number of barcode files does not match number of reads files.")
    
    sample_tags = SAMPLE_TAGS[genome]
    
    sample_tag_option_string = "(" + ")|(".join(sample_tags) + ")"
    sample_tag_re = re.compile(
        "(GTTGTCAAGATGCTACCGTTCAGAG){{e<={mismatches}}}"
        "(?P<sample_tag>{sample_tag_option_string}){{e<={mismatches}}}"
        "AAA*".format(
            mismatches=mismatches,
            sample_tag_option_string=sample_tag_option_string,
        )
    )
    
    barcode_re = barcode_res[barcode_version]
    
    barcode_matches = 0
    barcode_matches_with_sample_tag = 0
    cell_sample_tags = defaultdict(lambda: np.array([0] * len(sample_tags), dtype=int))
    
    for barcodes_file, reads_file in zip(barcodes_files, reads_files):
        if not os.path.exists(barcodes_file):
            raise ValueError("Barcodes file {} does not exist.".format(barcodes_file))
        if not os.path.exists(reads_file):
            raise ValueError("Reads file {} does not exist.".format(reads_file))
    
        # start demultiplexing
        with smart_open(barcodes_file, 'rb') as f:
            logger.info("Determining file size...")
            n_lines = int(sum(1 for _ in f))

        with smart_open(barcodes_file, 'rb') as file1:
            with smart_open(reads_file, 'rb') as file2:
                try:
                    file1_iter = iter(file1)
                    file2_iter = iter(file2)

                    i = 0
                    with tqdm(total=n_lines, desc='DMX') as pb:
                        while True:
                            line1 = next(file1_iter).decode('utf-8').rstrip()
                            line2 = next(file2_iter).decode('utf-8').rstrip()
                            
                            if i % 4 == 1:
                                barcode_match = barcode_re.search(line1)
                                if barcode_match is not None:
                                    barcode_tmp = barcode_match.group(0)
                                    if barcode_version == 'enhanced':
                                        barcode = f'{barcode_tmp[:9]}_{barcode_tmp[13:22]}_{barcode_tmp[26:]}'
                                    else:
                                        barcode = f'{barcode_tmp[:9]}_{barcode_tmp[21:30]}_{barcode_tmp[43:]}'
                                    barcode_matches += 1
                                    
                                    sample_tag_match = sample_tag_re.match(line2)
                                    if sample_tag_match is not None:
                                        sample_tag_ix = None
                                        for ix, group in enumerate(sample_tag_match.groups()[2:]):
                                            if group is not None:
                                                sample_tag_ix = ix
                                                break
                                        cell_sample_tags[barcode][sample_tag_ix] += 1
                                        barcode_matches_with_sample_tag += 1
                            
                            pb.update()
                            i += 1
                except StopIteration:
                    pass

    print("Barcodes detected: ", barcode_matches)
    print("Barcodes with sample tag: ", barcode_matches_with_sample_tag)
    
    if whitelists is not None:
        n_failures = 0
        cell_sample_tags_corrected = defaultdict(lambda: np.array([0] * len(sample_tags)))
        for barcode, counts in tqdm(cell_sample_tags.items(), desc='Correcting barcodes'):
            corrected_barcode = correct_rhapsody_barcode(barcode, *whitelists)
            if corrected_barcode is not None:
                cell_sample_tags_corrected[corrected_barcode] += counts
            else:
                n_failures += 1
                cell_sample_tags_corrected[barcode] += counts
        print("Barcode correction failures: ", n_failures)
        cell_sample_tags = cell_sample_tags_corrected
    
    sample_tag_statistics = pd.DataFrame(
        [
            list(counts) + [sum(counts)]
            for counts in cell_sample_tags.values()
        ],
        columns=['sample_tag_{}'.format(i + 1) for i in range(len(sample_tags))] + ['total'],
        index=[barcode for barcode in cell_sample_tags.keys()],
    )
    sample_tag_statistics['final_sample_tag'] = -1
    sample_tag_statistics['type'] = pd.Categorical(
        ['undetermined'] * sample_tag_statistics.shape[0],
        categories=['undetermined', 'hq', 'recovered', 'multiplet'],
    )
    
    print("Identifying high quality singlets...")
    hq = 0
    cell_samples = dict()
    # identify high quality singlets
    lq_cell_sample_tags = {}
    sample_tags_hq_counts = [[] for _ in range(len(sample_tags))]
    sample_tags_noise_counts = [[] for _ in range(len(sample_tags))]
    total_noise = 0
    for barcode, counts in tqdm(cell_sample_tags.items()):
        fractions = counts / sum(counts)
        hq_ixs = np.where(fractions > hq_singlet_cutoff)[0]
        if len(hq_ixs) == 1:
            hq_ix = hq_ixs[0]
            cell_samples[barcode] = hq_ix
            sample_tag_statistics.loc[barcode, 'final_sample_tag'] = hq_ix
            sample_tag_statistics.loc[barcode, 'type'] = 'hq'
            hq_count = counts[hq_ix]
            hq += 1
            noise_count = np.sum(np.delete(counts, hq_ixs))
            total_noise += noise_count
            sample_tags_hq_counts[hq_ix].append(hq_count)
            sample_tags_noise_counts[hq_ix].append(noise_count)
        else:
            lq_cell_sample_tags[barcode] = counts

    minimum_sample_tag_counts = [np.min(counts) if len(counts) > 0 else 999999999999999
                                    for counts in sample_tags_hq_counts]
    percentage_noise = [sum(noise)/total_noise for noise in sample_tags_noise_counts]

    print("Plotting sample tag counts...")
    if sample_tag_counts_plot is not None:
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        fig = plt.figure(figsize=(8.27, 11.69))
        gs = GridSpec(4, 3)
        for i in range(len(sample_tags_hq_counts)):
            counts = sample_tags_hq_counts[i]
            ax = plt.subplot(gs[int(i / 3), i % 3])
            ax.hist(counts, bins=100)
            ax.set_xlabel("Sample Tag read count")
            ax.set_ylabel("Number of putative cells")
            ax.set_yscale('log')
            ax.set_title(f'Tag {i + 1}')
        fig.tight_layout()
        fig.savefig(sample_tag_counts_plot)

    print("Fitting expected noise trendline...")
    # fit expected noise trendline
    per_cell_tag_counts = []
    per_cell_total_counts = []
    per_cell_noise_counts = []
    for barcode, counts in cell_sample_tags.items():
        tag_ixs = []
        for i in range(len(counts)):
            if counts[i] >= minimum_sample_tag_counts[i]:
                tag_ixs.append(i)
        tag_count = np.sum(counts[tag_ixs])
        noise_count = np.sum(np.delete(counts, tag_ixs))
        per_cell_tag_counts.append(tag_count)
        per_cell_total_counts.append(tag_count + noise_count)
        per_cell_noise_counts.append(noise_count)

    m, b = np.polyfit(per_cell_total_counts, per_cell_noise_counts, 1)

    if expected_noise_plot is not None:
        fig, ax = plt.subplots()
        ax.scatter(per_cell_total_counts, per_cell_noise_counts)
        ax.plot(per_cell_total_counts, m * np.array(per_cell_total_counts) + b, color='red')
        ax.set_xlabel("Total Sample Tag counts")
        ax.set_ylabel("Noise Sample Tag counts")
        fig.tight_layout()
        fig.savefig(expected_noise_plot)

    multiplets = 0
    undetermined = 0
    recovered = 0
    for barcode, counts in lq_cell_sample_tags.items():
        total = np.sum(counts)
        expected_noise = m * total + b
        corrected_counts = counts - expected_noise
        ixs = []
        for i in range(len(corrected_counts)):
            if corrected_counts[i] > minimum_sample_tag_counts[i]:
                ixs.append(i)

        if len(ixs) == 1:
            recovered += 1
            cell_samples[barcode] = ixs[0]
            sample_tag_statistics.loc[barcode, 'final_sample_tag'] = ixs[0]
            sample_tag_statistics.loc[barcode, 'type'] = 'recovered'
        elif len(ixs) > 1:
            multiplets += 1
            sample_tag_statistics.loc[barcode, 'type'] = 'multiplet'
        else:
            undetermined += 1

    logger.info("HQ: {}".format(hq))
    logger.info("Recovered: {}".format(recovered))
    logger.info("Multiplets: {}".format(multiplets))
    logger.info("Undetermined: {}".format(undetermined))

    # fix for natural numbering
    sample_tag_statistics['final_sample_tag'] += 1

    if statistics_file is not None:
        sample_tag_statistics.to_csv(statistics_file, sep="\t")
    
    if output_folder is None:
        return cell_samples, None

    output_folder = mkdir(output_folder)
    output_files = {}
    output_file_names = {}
    
    try:
        for barcodes_file, reads_file in zip(barcodes_files, reads_files):
            with smart_open(barcodes_file, 'rb') as file1:
                with smart_open(reads_file, 'rb') as file2:
                    try:
                        file1_iter = iter(file1)
                        file2_iter = iter(file2)

                        i = 0
                        current_sample_name = -1
                        current_entry1 = [None, None, None, None]
                        current_entry2 = [None, None, None, None]
                        with tqdm(total=n_lines, desc='DMX-Split') as pb:
                            while True:
                                line1 = next(file1_iter).decode('utf-8').rstrip()
                                line2 = next(file2_iter).decode('utf-8').rstrip()

                                current_entry1[i % 4] = line1
                                current_entry2[i % 4] = line2

                                if i % 4 == 1:
                                    barcode_match = barcode_re.search(line1)
                                    if barcode_match is not None:
                                        barcode_tmp = barcode_match.group(0)
                                        if barcode_version == 'enhanced':
                                            barcode = f'{barcode_tmp[:9]}_{barcode_tmp[13:22]}_{barcode_tmp[26:]}'
                                        else:
                                            barcode = f'{barcode_tmp[:9]}_{barcode_tmp[21:30]}_{barcode_tmp[43:]}'
                                        
                                        barcode = correct_rhapsody_barcode(barcode, *whitelists) or barcode
                                        
                                        current_sample_name = cell_samples.get(barcode, -1)
                                    else:
                                        current_sample_name = -1

                                if i % 4 == 3:
                                    if current_sample_name in output_files:
                                        f1, f2 = output_files[current_sample_name]
                                    else:
                                        file1_name = os.path.join(
                                            output_folder,
                                            '{}_{}_R1.fastq.gz'.format(prefix, current_sample_name + 1)
                                        )
                                        file2_name = os.path.join(
                                            output_folder,
                                            '{}_{}_R2.fastq.gz'.format(prefix, current_sample_name + 1)
                                        )
                                        output_file_names[current_sample_name] = (file1_name, file2_name)
                                        f1 = smart_open(file1_name, 'wb')
                                        f2 = smart_open(file2_name, 'wb')
                                        output_files[current_sample_name] = (f1, f2)

                                    for line in current_entry1:
                                        f1.write("{}\n".format(line).encode('utf-8'))
                                    for line in current_entry2:
                                        f2.write("{}\n".format(line).encode('utf-8'))
                                    current_sample_name = None

                                i += 1
                                pb.update()
                    except StopIteration:
                        pass
    finally:
        for f1, f2 in output_files.values():
            f1.close()
            f2.close()

    return cell_samples, output_file_names

