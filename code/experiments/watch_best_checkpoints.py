"""Preserve best-so-far checkpoints from a long training run without mutation."""

import argparse
import json
from pathlib import Path
import shutil
import time

from common import sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--interval', type=float, default=20)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error('interval must be positive')
    dest = args.run_dir / 'snapshots'
    dest.mkdir(exist_ok=True)
    while True:
        progress_path = args.run_dir / 'progress.json'
        if progress_path.exists():
            try:
                progress = json.loads(progress_path.read_text())
                step = progress['best_step']
                expected = progress['checkpoint_sha256']
                source = args.run_dir / 'checkpoint.pt'
                target = dest / f'checkpoint_step{step}.pt'
                if not target.exists() and source.exists() and sha(source) == expected:
                    shutil.copy2(source, target)
                    if sha(target) != expected:
                        target.unlink(missing_ok=True)
                        raise RuntimeError('Copied checkpoint hash differs from source')
                    print(json.dumps({'snapshot_step': step, 'sha256': expected}), flush=True)
            except (json.JSONDecodeError, OSError, KeyError):
                pass
        if (args.run_dir / 'metrics.json').exists():
            break
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
