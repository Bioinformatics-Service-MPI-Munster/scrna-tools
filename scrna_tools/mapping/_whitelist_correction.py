
def known(words, WORDS): 
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)


def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'ACGT'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    return set(replaces)


def edits2(word): 
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))


def candidates(word, WORDS): 
    "Generate possible spelling corrections for word."
    return (known([word], WORDS) or known(edits1(word), WORDS) or known(edits2(word), WORDS))


def correct_rhapsody_barcode(
    barcode,
    *whitelists,
):
    if isinstance(barcode, str):
        barcode_parts = barcode.split('_')
    else:
        barcode_parts = barcode
    
    corrected_barcode = []
    for (b, wl) in zip(barcode_parts, whitelists):
        corrections = candidates(b, wl)
        if len(corrections) == 1:
            corrected_barcode.append(corrections.pop())
        else:
            return None

    return '_'.join(corrected_barcode)
