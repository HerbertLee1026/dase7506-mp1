"""Remove train-trigram contexts skipped by the 128-branch inference rule."""

from pathlib import Path

import numpy as np

from common import sha


SOURCE = Path('assets/train_ngrams_3to5.npz')
OUTPUT = Path('assets/train_ngrams_3to5_pruned128.npz')


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    with np.load(SOURCE) as saved:
        arrays = {name: saved[name] for name in saved.files}
    offsets = arrays['context_offsets']
    lengths = np.diff(offsets)
    keep_context = lengths <= 128
    keep_entry = np.repeat(keep_context, lengths)
    arrays['context_keys'] = arrays['context_keys'][keep_context]
    arrays['context_totals'] = arrays['context_totals'][keep_context]
    arrays['context_offsets'] = np.append(
        0, np.cumsum(lengths[keep_context])).astype(np.int32)
    arrays['next_tokens'] = arrays['next_tokens'][keep_entry]
    arrays['trigram_counts'] = arrays['trigram_counts'][keep_entry]
    np.savez(OUTPUT, **arrays)
    print({'path': str(OUTPUT), 'bytes': OUTPUT.stat().st_size,
           'sha256': sha(OUTPUT), 'contexts_removed': int((~keep_context).sum()),
           'entries_removed': int((~keep_entry).sum())})


if __name__ == '__main__':
    main()
