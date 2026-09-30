# Training recipe for the selected ancestry

Direct evaluation requires no training: use the released checkpoint. The commands below document the selected ancestry in a fresh checkout; do not run them over existing results. Run from code/, set PYTHONPATH=., and use a Python3.12 interpreter with the documented dependencies. The original CPU stages use four threads internally. These commands have been checked against the recorded arguments and script interfaces, but the entire expensive ancestry was not rerun while packaging.

## 1. Original controls

```bash
PYTHONPATH=. python train.py --implementation model --config configs/baseline.json --device cpu --precision fp32 --threads 4 --seed 17 --steps 1200 --batch-size 32 --eval-every 0 --run-dir runs/replay_baseline
PYTHONPATH=. python train.py --implementation student --config configs/baseline.json --device cpu --precision fp32 --threads 4 --seed 17 --steps 1200 --batch-size 32 --eval-every 0 --run-dir runs/replay_swiglu1200
```

Both process9,830,400 targets; original recorded validation values are2.071087821 and2.007882703. These are controls, not ancestors of the final model. Historical metric files record the RoPE equal-target comparison as well.

## 2. Six-layer training and continuation

```bash
PYTHONPATH=. python experiments/train_dropout_arch.py --config configs/rope_dropout_depth6_width320_p020.json --implementation student_rope_dropout --run-dir runs/replay_d6 --steps 6000 --batch-size 32 --seed 17 --eval-every 600 --threads 4
PYTHONPATH=. python experiments/finetune_dropout.py --parent runs/replay_d6/checkpoint.pt --run-dir runs/replay_d6_ft --steps 3000 --batch-size 32 --seed 43 --learning-rate 0.00005 --eval-every 300
```

## 3. Identity layer and seven-layer continuation

```bash
PYTHONPATH=. python experiments/build_identity_depth.py --parent runs/replay_d6_ft/checkpoint.pt --output runs/replay_identity
PYTHONPATH=. python experiments/finetune_dropout.py --parent runs/replay_identity/checkpoint.pt --run-dir runs/replay_d7 --steps 3600 --batch-size 32 --seed 71 --learning-rate 0.00005 --last-block-lr-multiplier 4 --eval-every 300
```

The historical best was step3300; checkpoint.pt always stores the validation-best weights. Selection can differ if numerical differences change the trajectory.

## 4. SAM continuation

```bash
PYTHONPATH=. python experiments/replay_sam1200.py --parent runs/replay_d7/checkpoint.pt --run-dir runs/replay_sam --steps 1800 --stop-after 1200 --batch-size 32 --seed 83 --learning-rate 0.00003 --rho 0.05 --eval-every 300
```

replay_sam1200.py is the included finetune_sam.py with a stop-after boundary added for reproducibility: the cosine schedule still uses1800 planned steps, and updates stop at1200 as in the historical run. Changing --steps to1200 would change the learning-rate schedule. SAM counts two forward/backward target exposures per sample.

## 5. Construct the training-only count asset and combined predictor

In a fresh source checkout before extracting the checkpoint bundle, no assets exist yet. If the final asset was already extracted, use it and skip the build commands; builders refuse to overwrite results.

```bash
PYTHONPATH=. python experiments/build_train_ngrams.py
PYTHONPATH=. python experiments/build_variable_ngrams.py
PYTHONPATH=. python experiments/prune_variable_ngrams.py
PYTHONPATH=. python experiments/build_cache_checkpoint.py --parent runs/replay_sam/checkpoint.pt --output runs/replay_cache --temperature 1.15 --unigram-weight 0.05 --bigram-weight 0.25
PYTHONPATH=. python experiments/build_variable_checkpoint.py --parent runs/replay_cache/checkpoint.pt --output runs/replay_ngram --asset assets/train_ngrams_3to5_pruned128.npz --trigram-weight 0.05 --fourgram-weight 0.05 --fivegram-weight 0.08 --max-branches 128 --later-max-branches 1000
PYTHONPATH=. python experiments/build_hidden_checkpoint.py --parent runs/replay_ngram/checkpoint.pt --output runs/replay_hidden128 --weight 0.02 --beta 10 --dim 128
python evaluate.py --checkpoint runs/replay_hidden128/checkpoint.pt --device cpu --precision fp32 --threads 4 --split validation --output replay_parent_validation.json
```

The asset SHA is authoritative; NPZ ZIP metadata can differ on regeneration even when arrays are identical. Builders calculate and embed the actual regenerated asset hash. Keep the matching file with its checkpoint.

## 6. Cache-objective continuation and inference conversion

The final fitting stage ran on Mac MPS. The snapshot trainer below is the exact source saved for that run; it differs from the later experimental trainer that added Muon/CUDA options. Determine the replay parent's full validation BPB from the preceding command, and replace PARENT_VALIDATION_BPB below with that numeric result (historically1.4626716440329555).

```bash
PYTHONPATH=. python experiments/archive_cache_objective/finetune_cache_objective.py --parent runs/replay_hidden128/checkpoint.pt --run-dir runs/replay_cache_objective --parent-bpb PARENT_VALIDATION_BPB --steps 1200 --batch-size 16 --seed 101 --learning-rate 0.00003 --eval-every 200 --patience 3 --min-delta 0.0005 --train-device mps
PYTHONPATH=. python experiments/convert_fast_release.py --parent runs/replay_cache_objective/checkpoint.pt --output runs/replay_final/checkpoint.pt
```

The historical run stopped at600 and selected step200. On non-Mac hardware the snapshot accepts cpu, but that changes numerical execution; the later NPU adaptation is not part of this frozen release. The conversion changes only the implementation wrapper and hidden cache dimension128->320; it performs no training. It was selected and resource-checked on validation before this release's test evaluation.

The included frozen weights and exact source/asset hashes are the route to reproducing the reported score. Historical scripts evolved during search, and retraining on other platforms is not claimed to recreate byte-identical weights. All seeds, parent hashes, validation histories, and search costs are retained in experiments/run_records/.
