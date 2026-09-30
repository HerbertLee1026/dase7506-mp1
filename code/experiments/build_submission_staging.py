"""Prepare a small, reviewable code repository and matching checkpoint bundle."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT.parent
CHECKPOINT = ROOT / 'runs/hidden_cache_nocopy_a002_b10_dim128_d7_w320_sam1200_s83/checkpoint.pt'
ASSET = ROOT / 'assets/train_ngrams_3to5_pruned128.npz'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def copy_file(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if digest(CHECKPOINT) != '9cde65239e390907ff853fd653186d175a56ea116328e52044f1de6d5edf525c':
        raise ValueError('Candidate checkpoint changed.')
    if digest(ASSET) != '2b233a7b87ff145221c44d493c8cb3db4176e2daf149180af82eceb6836d88d7':
        raise ValueError('Candidate n-gram asset changed.')
    repository = output / 'repository'
    code = repository / 'code'
    bundle = output / 'checkpoint_bundle'
    for source in ROOT.glob('*.py'):
        copy_file(source, code / source.name)
    for name in ('README.md', 'REPRODUCE.md', 'requirements.txt',
                 '.gitignore', 'PACKAGE_MANIFEST.json', 'RUN_LOG_TEMPLATE.csv'):
        copy_file(ROOT / name, code / name)
    copy_file(PACKAGE_ROOT / 'GUIDE.md', repository / 'GUIDE.md')
    for directory in ('configs', 'data', 'tests'):
        for source in (ROOT / directory).rglob('*'):
            if source.is_file() and '__pycache__' not in source.parts and source.suffix != '.pyc':
                copy_file(source, code / source.relative_to(ROOT))
    for source in (ROOT / 'experiments').glob('*.py'):
        copy_file(source, code / source.relative_to(ROOT))
    for name in ('EXPERIMENT_LOG.md',):
        source = ROOT / 'experiments' / name
        copy_file(source, code / 'experiments' / name)
    copy_file(ROOT / 'REPORT.md', code / 'REPORT.md')
    for run in (ROOT / 'runs').iterdir():
        if run.is_dir():
            for source in run.glob('*.json'):
                copy_file(source, code / 'experiments/run_records' / run.name / source.name)
    copy_file(ASSET, code / 'assets' / ASSET.name)
    copy_file(CHECKPOINT, bundle / 'checkpoint.pt')
    copy_file(ROOT / 'runs/hidden_cache_nocopy_a002_b10_dim128_d7_w320_sam1200_s83/ancestry.json', bundle / 'ancestry.json')
    copy_file(ROOT / 'runs/hidden_cache_nocopy_a002_b10_dim128_d7_w320_sam1200_s83/test_cpu_fp32_frozen.json', bundle / 'test_cpu_fp32.json')
    manifest = {
        'status': 'frozen predictor; immutable release links pending',
        'checkpoint_sha256': digest(bundle / 'checkpoint.pt'),
        'ngram_asset_sha256': digest(code / 'assets' / ASSET.name),
        'implementation_sha256': digest(code / 'student_hidden_variable_ngram.py'),
        'checkpoint_bytes': CHECKPOINT.stat().st_size,
        'ngram_asset_bytes': ASSET.stat().st_size,
        'inference_asset_bytes': CHECKPOINT.stat().st_size + ASSET.stat().st_size,
        'validation_bpb': 1.4626716440329555,
        'test_bpb': 1.4782461576027206,
    }
    (bundle / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (repository / 'SUBMISSION_STATUS.md').write_text(
        'Frozen predictor. See `code/REPRODUCE.md` and `code/REPORT.md`. '
        'Full-test CPU FP32 BPB: 1.4782461576027206. '
        'Immutable code and checkpoint URLs must be added through the course form.\n')
    print(json.dumps({'output': str(output), **manifest}, indent=2))


if __name__ == '__main__':
    main()
