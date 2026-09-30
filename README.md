# DASE7506 MP1 - interim frozen release

**Full-test CPU FP32 BPB: 1.478713291044518**. Validation BPB: 1.4622831529104345.

Selected on validation and frozen before this test evaluation. An earlier separately frozen model scored1.4782461576027206 on test; the latest candidate is slightly worse on test despite better validation. This release does not claim an improved test score. Previous results are disclosed in the report.

- [Report PDF](code/REPORT.pdf) and [report source](code/REPORT.md)
- [Installation and exact evaluation commands](code/REPRODUCE.md)
- [Training recipe](code/TRAINING_RECIPE.md)
- [Resource measurements and freeze record](code/evidence/)
- [Assignment guide](GUIDE.md)

Download the matching MP1_checkpoint_bundle.zip and extract it **inside code/**. It supplies both submitted_checkpoint/checkpoint.pt and assets/train_ngrams_3to5_pruned128.npz. The code archive alone deliberately does not contain the inference asset or model weights.

From code/, with Python3.12 and requirements.txt installed:

```bash
python evaluate.py --checkpoint submitted_checkpoint/checkpoint.pt --device cpu --precision fp32 --threads 4 --split test
```

Checkpoint SHA256: `47103d970645bd1858cc5fee40fefe5bfb880c7c5b29e20652ab8d959b373f39`.

OpenAI Codex provided substantial assistance with model code, experiments, debugging, measurements and documentation. No external training text or pretrained weights were used. Results and resource constraints are discussed critically in the report. This repository includes historical records; the root README, current REPORT, and evidence/release_summary.json define this release.
