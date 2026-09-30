"""No training: convert compatible weights to equivalent fast local scatter."""
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
