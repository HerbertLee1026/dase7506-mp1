"""Build compact bigram/trigram counts using only the supplied training split."""

from pathlib import Path

import numpy as np

from common import load_data, sha


VOCAB = 2048
OUTPUT = Path('assets/train_ngrams.npz')


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    ids = load_data()['train'][0].numpy().astype(np.int64)
    bigrams = np.bincount(
        ids[:-1] * VOCAB + ids[1:], minlength=VOCAB * VOCAB
    ).reshape(VOCAB, VOCAB)
    triples = (ids[:-2] * VOCAB + ids[1:-1]) * VOCAB + ids[2:]
    keys, counts = np.unique(triples, return_counts=True)
    context_keys, starts, lengths = np.unique(
        keys // VOCAB, return_index=True, return_counts=True
    )
    offsets = np.append(starts, len(keys))
    context_totals = np.add.reduceat(counts, starts)
    assert bigrams.max() <= np.iinfo(np.uint16).max
    assert counts.max() <= np.iinfo(np.uint16).max
    assert context_totals.max() <= np.iinfo(np.uint16).max
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUTPUT,
        bigram_counts=bigrams.astype(np.uint16),
        bigram_totals=bigrams.sum(axis=1).astype(np.int32),
        context_keys=context_keys.astype(np.uint32),
        context_offsets=offsets.astype(np.int32),
        context_totals=context_totals.astype(np.uint16),
        next_tokens=(keys % VOCAB).astype(np.uint16),
        trigram_counts=counts.astype(np.uint16),
    )
    print({'path': str(OUTPUT), 'bytes': OUTPUT.stat().st_size,
           'sha256': sha(OUTPUT), 'trigrams': len(keys),
           'contexts': len(context_keys)})


if __name__ == '__main__':
    main()
