"""Build compact train-only trigram, 4-gram and 5-gram lookup tables."""

from pathlib import Path

import numpy as np

from common import load_data, sha


VOCAB = 2048
SOURCE = Path('assets/train_ngrams.npz')
OUTPUT = Path('assets/train_ngrams_3to5.npz')


def later_order(ids, order):
    contexts = np.zeros(len(ids) - order + 1, dtype=np.int64)
    for offset in range(order - 1):
        contexts = contexts * VOCAB + ids[offset:offset + len(contexts)]
    distinct, inverse, totals = np.unique(
        contexts, return_inverse=True, return_counts=True)
    del distinct
    selected = totals[inverse] >= 2
    del totals, inverse
    keys, counts = np.unique(
        contexts[selected] * VOCAB + ids[order - 1:][selected], return_counts=True)
    distinct_contexts, starts = np.unique(keys // VOCAB, return_index=True)
    offsets = np.append(starts, len(keys))
    context_totals = np.add.reduceat(counts, starts)
    assert counts.max() <= np.iinfo(np.uint16).max
    assert context_totals.max() <= np.iinfo(np.uint16).max
    prefix = ('four' if order == 4 else 'five')
    return {
        f'{prefix}_context_keys': distinct_contexts.astype(np.uint64),
        f'{prefix}_context_offsets': offsets.astype(np.int32),
        f'{prefix}_context_totals': context_totals.astype(np.uint16),
        f'{prefix}_next_tokens': (keys % VOCAB).astype(np.uint16),
        f'{prefix}_counts': counts.astype(np.uint16),
    }


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    train_ids = load_data()['train'][0].numpy().astype(np.int64)
    with np.load(SOURCE) as saved:
        tables = {name: saved[name] for name in (
            'context_keys', 'context_offsets', 'context_totals',
            'next_tokens', 'trigram_counts')}
    tables.update(later_order(train_ids, 4))
    tables.update(later_order(train_ids, 5))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUTPUT, **tables)
    print({'path': str(OUTPUT), 'bytes': OUTPUT.stat().st_size,
           'sha256': sha(OUTPUT),
           'four_contexts': len(tables['four_context_keys']),
           'five_contexts': len(tables['five_context_keys'])})


if __name__ == '__main__':
    main()
