# Reproduce this interim frozen predictor

## Install in a separate reviewer environment

Use Python3.12. On Linux CPU, run from code/:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install numpy==2.5.3 tokenizers==0.21.4
```

For macOS use the default PyPI torch2.7.1 wheel instead of the Linux CPU index. These reviewer commands do not reinstall the author's already working environment. The original code/README.md also documents installation.

## Restore inference files

Extract MP1_checkpoint_bundle.zip into code/. The resulting files must be:

- submitted_checkpoint/checkpoint.pt
- assets/train_ngrams_3to5_pruned128.npz
- manifest.json

Checkpoint SHA256: `47103d970645bd1858cc5fee40fefe5bfb880c7c5b29e20652ab8d959b373f39`.
Asset SHA256: `2b233a7b87ff145221c44d493c8cb3db4176e2daf149180af82eceb6836d88d7`.
Implementation: student_fast_hidden_ngram.py, SHA256 `ef1f89f31fe8fcacb7875e7d5782aa3428a2bbc59a9229666293cc94cba78471`.
All model dependencies, fixed data and tokenizer are in the code archive. No retraining or network access is needed after installation and downloading the two archives.

## Score the frozen model

```bash
.venv/bin/python evaluate.py --checkpoint submitted_checkpoint/checkpoint.pt --device cpu --precision fp32 --threads 4 --split validation --output reproduced_validation.json
.venv/bin/python evaluate.py --checkpoint submitted_checkpoint/checkpoint.pt --device cpu --precision fp32 --threads 4 --split test --output reproduced_test.json
```

Reference validation BPB: 1.4622831529104345; full-test BPB: 1.478713291044518. Test scoring covers428,405 targets and1,292,013 UTF-8 bytes. The evaluator creates a JSON result and a per-window loss NPY file. Submit the full-test bpb value, never token_ppl, training loss or validation BPB. Tiny floating-point differences across hardware are possible.

## Measure the resource budget

MP1_baseline_comparison.zip contains the original baseline weights for timing only; they are not read by the submitted model. Extract it into code/ and score comparison_only/baseline_checkpoint.pt using the same CPU, FP32, threads4 and test split. Run the baseline and candidate serially when the machine is idle. On macOS use /usr/bin/time -l; on Linux use /usr/bin/time -v and convert its maximum RSS from KiB to bytes.

Recorded test scoring seconds: candidate23.5479606250301, baseline5.219540417077951, ratio4.511500772746755. Candidate peak RSS: 1563082752 bytes. Required inference assets: 66394918 bytes. The extra comparison checkpoint, historical evidence and any training recovery state are not inference dependencies.

## Training and disclosure

See TRAINING_RECIPE.md and experiments/run_records/. The last fitting stage used MPS; CPU score reproduction uses no MPS/NPU code. The prepared but untrained learned-pointer experiment and server migration work are excluded from this frozen predictor. Earlier test exposure and the lack of test improvement are disclosed in REPORT.md. AI assistance is disclosed in the repository README.
