import os
import regex as re
from future.utils import string_types
from ..helpers import smart_open, mkdir
from tqdm import tqdm
import numpy as np
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


def rhapsody_demultiplex(barcodes_file, reads_file, genome, output_folder=None, prefix='sample',
                         sample_names=None, common_mismatches=2, hq_singlet_cutoff=0.75,
                         expected_noise_plot=None, sample_tag_counts_plot=None):
    if sample_names is None:
        sample_names = {1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6',
                        7: '7', 8: '8', 9: '9', 10: '10', 11: '11', 12: '12'}

    # compile regex
    sample_tag_re = re.compile("(GTTGTCAAGATGCTACCGTTCAGAG){{e<={common_mismatches}}}"
                                "(?P<sample_tag>.{{45}})"
                                "AAA*".format(common_mismatches=common_mismatches))
    sample_tags = {}
    sample_tag_to_ix = {}
    sample_ix_to_tag = {}
    for i, (sample_tag_ix, sample_name) in enumerate(sample_names.items()):
        sample_tag = SAMPLE_TAGS[genome[:2]][int(sample_tag_ix) - 1]
        sample_tags[sample_tag] = sample_name
        sample_tag_to_ix[sample_tag] = i
        sample_ix_to_tag[i] = sample_tag

    sample_tags_correct = BarcodeCorrect(sample_tag_to_ix.keys())

    # start demultiplexing
    with smart_open(barcodes_file, 'rb') as f:
        logger.info("Determining file size...")
        n_lines = int(sum(1 for _ in f))

    cell_sample_tags = defaultdict(lambda: np.array([0] * len(sample_names)))
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
                            barcode = line1[:27]
                            m = sample_tag_re.match(line2)
                            if m is not None:
                                sample_tag = m.group(2)
                                sample_tag_candidates = sample_tags_correct.candidates(sample_tag)
                                if sample_tag_candidates is not None and len(sample_tag_candidates) == 1:
                                    ix = sample_tag_to_ix.get(sample_tag_candidates.pop(), None)
                                    if ix is not None:
                                        cell_sample_tags[barcode][ix] += 1

                        i += 1
                        pb.update(1)
            except StopIteration:
                pass

    hq = 0
    cell_samples = dict()
    # identify high quality singlets
    lq_cell_sample_tags = {}
    sample_tags_hq_counts = [[] for _ in range(len(sample_names))]
    sample_tags_noise_counts = [[] for _ in range(len(sample_names))]
    total_noise = 0
    for barcode, counts in cell_sample_tags.items():
        fractions = counts / sum(counts)
        hq_ixs = np.where(fractions > hq_singlet_cutoff)[0]
        if len(hq_ixs) == 1:
            hq_ix = hq_ixs[0]
            cell_samples[barcode] = hq_ix
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

    if sample_tag_counts_plot is not None:
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        fig = plt.figure(figsize=(8.27, 11.69))
        gs = GridSpec(4, 3)
        for i in range(len(sample_tags_hq_counts)):
            counts = sample_tags_hq_counts[i]
            sample_tag = sample_ix_to_tag[i]
            sample_name = sample_tags[sample_tag]
            ax = plt.subplot(gs[int(i / 3), i % 3])
            ax.hist(counts, bins=100)
            ax.set_xlabel("Sample Tag read count")
            ax.set_ylabel("Number of putative cells")
            ax.set_title(sample_name)
        fig.tight_layout()
        fig.savefig(sample_tag_counts_plot)

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
    unique = 0
    for barcode, counts in lq_cell_sample_tags.items():
        total = np.sum(counts)
        expected_noise = m * total + b
        corrected_counts = counts - expected_noise
        ixs = []
        for i in range(len(corrected_counts)):
            if corrected_counts[i] > minimum_sample_tag_counts[i]:
                ixs.append(i)

        if len(ixs) == 1:
            cell_samples[barcode] = ixs[0]
        elif len(ixs) > 1:
            multiplets += 1
        else:
            undetermined += 1

    logger.info("HQ: {}".format(hq))
    logger.info("Recovered: {}".format(unique))
    logger.info("Multiplets: {}".format(multiplets))
    logger.info("Undetermined: {}".format(undetermined))

    cell_sample_names = {barcode: sample_tags[sample_ix_to_tag[ix]] for barcode, ix in cell_samples.items()}

    if output_folder is None:
        return cell_sample_names

    output_folder = mkdir(output_folder)
    output_files = {}
    output_file_names = {}
    try:
        with smart_open(barcodes_file, 'rb') as file1:
            with smart_open(reads_file, 'rb') as file2:
                try:
                    file1_iter = iter(file1)
                    file2_iter = iter(file2)

                    i = 0
                    current_sample_name = None
                    current_entry1 = [None, None, None, None]
                    current_entry2 = [None, None, None, None]
                    with tqdm(total=n_lines, desc='DMX-Split') as pb:
                        while True:
                            line1 = next(file1_iter).decode('utf-8').rstrip()
                            line2 = next(file2_iter).decode('utf-8').rstrip()

                            current_entry1[i % 4] = line1
                            current_entry2[i % 4] = line2

                            if i % 4 == 1:
                                barcode = line1[:27]
                                current_sample_name = cell_sample_names.get(barcode, None)

                            if i % 4 == 0 and current_sample_name is not None:
                                if current_sample_name in output_files:
                                    f1, f2 = output_files[current_sample_name]
                                else:
                                    file1_name = os.path.join(output_folder,
                                                                '{}_{}_R1.fastq.gz'.format(prefix,
                                                                                            current_sample_name))
                                    file2_name = os.path.join(output_folder,
                                                                '{}_{}_R2.fastq.gz'.format(prefix,
                                                                                            current_sample_name))
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

    return cell_sample_names, output_file_names

