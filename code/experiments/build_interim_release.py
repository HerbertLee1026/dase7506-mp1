"""Build an independently frozen interim release from already measured artifacts."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]

def copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run=args.run.resolve();out=args.output.resolve()
    if out.exists():raise FileExistsError(out)
    summary=json.loads((run/'release_summary.json').read_text())
    freeze=json.loads((run/'freeze_before_test.json').read_text())
    assert summary['passed_resources']
    assert sha(run/'checkpoint.pt')==summary['checkpoint_sha256']
    for name,digest in freeze['source_sha256'].items():
        assert sha(ROOT/name)==digest,name
    repo=out/'repository';code=repo/'code';bundle=out/'checkpoint_bundle'
    for source in ROOT.glob('*.py'):copy(source,code/source.name)
    for name in ['requirements.txt','PACKAGE_MANIFEST.json','RUN_LOG_TEMPLATE.csv','.gitignore','README.md']:
        copy(ROOT/name,code/name)
    copy(ROOT.parent/'GUIDE.md',repo/'GUIDE.md')
    for directory in ['configs','data','tests']:
        for source in (ROOT/directory).rglob('*'):
            if source.is_file() and '__pycache__' not in source.parts and source.suffix!='.pyc':
                copy(source,code/source.relative_to(ROOT))
    for source in (ROOT/'experiments').iterdir():
        if source.is_file() and source.suffix in ('.py','.md','.txt'):
            copy(source,code/'experiments'/source.name)
    for directory in (ROOT/'runs').iterdir():
        if directory.is_dir():
            for source in directory.glob('*.json'):
                copy(source,code/'experiments/run_records'/directory.name/source.name)
    for source in run.iterdir():
        if source.is_file() and source.name!='checkpoint.pt':
            copy(source,code/'evidence'/source.name)
    for source in (run/'source').iterdir():copy(source,code/'evidence/frozen_source'/source.name)
    cache_run=ROOT/'runs/cache_objective_d7_w320_s101_lr3e5'
    for source in (cache_run/'source').iterdir():
        if source.is_file():copy(source,code/'experiments/archive_cache_objective'/source.name)
    copy(run/'checkpoint.pt',bundle/'submitted_checkpoint/checkpoint.pt')
    asset=ROOT/'assets/train_ngrams_3to5_pruned128.npz'
    copy(asset,bundle/'assets'/asset.name)
    copy(ROOT/'runs/baseline/checkpoint.pt',out/'comparison_only/baseline_checkpoint.pt')
    manifest=dict(summary,asset_sha256=sha(asset),asset_bytes=asset.stat().st_size,
                  checkpoint_bytes=(run/'checkpoint.pt').stat().st_size,
                  implementation='student_fast_hidden_ngram',
                  implementation_sha256=sha(code/'student_fast_hidden_ngram.py'),
                  protocol='7506-mp1-wt2-v2',bundle_extract_destination='repository/code/',
                  inference_files=['submitted_checkpoint/checkpoint.pt','assets/'+asset.name])
    (bundle/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (out/'RELEASE_SUMMARY.json').write_text(json.dumps(manifest,indent=2)+'\n')
    recipe_source=(ROOT/'experiments/finetune_sam.py').read_text()
    recipe_source=recipe_source.replace("    args = parser.parse_args()", "    parser.add_argument('--stop-after', type=int, default=1200)\n    args = parser.parse_args()")
    recipe_source=recipe_source.replace('range(1, args.steps + 1)', 'range(1, min(args.steps, args.stop_after) + 1)')
    (code/'experiments/replay_sam1200.py').write_text(recipe_source)
    (code/'experiments/convert_fast_release.py').write_text('''"""No training: convert compatible weights to equivalent fast local scatter."""
import argparse
from pathlib import Path
import torch
from common import sha
p=argparse.ArgumentParser()
p.add_argument('--parent',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
c=torch.load(a.parent,map_location='cpu',weights_only=True)
c['implementation']='student_fast_hidden_ngram'
c['config']=dict(c['config'],hidden_cache_dim=320)
c['conversion_parent_checkpoint_sha256']=sha(a.parent)
a.output.parent.mkdir(parents=True,exist_ok=True)
torch.save(c,a.output)
''')
    print(json.dumps({'output':str(out),**summary},indent=2))

if __name__=='__main__':main()
