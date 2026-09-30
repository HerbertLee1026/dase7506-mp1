# Validation-only optimization resumed on 29 September 2026

The user requested further optimization and will handle submission. A previous candidate already has a disclosed full-test score. This round uses only validation to select methods, checkpoints and hyperparameters. No new test evaluation will be run during this development round. The previously frozen code and checkpoint remain separately preserved.

Reference validation BPB: 1.4626716440329555, checkpoint `runs/hidden_cache_nocopy_a002_b10_dim128_d7_w320_sam1200_s83/checkpoint.pt`.

## 1. Train against the within-window cache mixture

Hypothesis: neural-only training overinvests in predictions already solved by local copying. Train its causal cache mixture likelihood using training windows, while evaluating the complete unchanged predictor on validation. No training-set n-gram counts are used in the training loss, avoiding their especially optimistic in-sample probabilities.

- Script: `experiments/finetune_cache_objective.py`.
- Seed 101, MPS FP32 training, batch 16, learning rate 3e-5, CPU FP32 full validation every 200 steps.
- Planned maximum 1,200 steps; stop after three validations without a 0.0005 material improvement.
- Exact agreement of the training target-probability formula with the existing cache predictor was checked before training; finite nonzero gradients were verified.
- Results: step 200 1.462499357; step 400 1.462924202; step 600 1.463429286. Automatically stopped at 600. Best weights retained, but insufficient gain to select as the new reference.

## 2. Fused SwiGLU projection

Hypothesis: merge gate/value linear operations to free inference time without increasing parameter count. A CPU batch probe checked the converted model against the original and alternated timing order. Maximum log-probability delta was 1.335e-5. Median time improved only from 0.8602 to 0.8526 seconds during concurrent MPS training. This is not an official budget measurement; the small speed change does not justify adopting it or enlarging the model on that basis.

## 3. Revisit existing width-384 weights within all inference limits

The earlier six-layer width-384 model stopped at 3,600 steps with dropout 0.1. Its standalone BPB at that training stage did not establish its eventual quality after the stronger dropout and continuation that helped width 320. Reuse those existing weights instead of restarting training.

The original full 3/4/5-gram asset would exceed the size budget with this model. Removing only the trigram table creates a train-derived 4/5-gram asset of 21,053,142 bytes. The neural checkpoint plus this asset totals 66,833,227 bytes, under 64 MiB. The complete candidate retains local token/bigram copy and 128-dimensional hidden copy. Its CPU full-validation speed and RAM must pass a paired baseline check before training is launched.

If the resource gate passes, continue the existing weights with dropout 0.2, using complete-predictor validation for checkpoint selection and early stopping. No score claim will be made from raw neural BPB alone.

Resource gate for width 384 passed before continuation: full validation 1.510592843, scoring 21.603906 seconds versus paired baseline 4.457268 seconds (4.847x), RSS 1,990,606,848 bytes; combined checkpoint and asset 66,833,227 bytes. Continuing with dropout 0.2, AdamW learning rate 0.00015, seed 109, batch 16, maximum 4,000 steps; full validation every 500 and stop after three without 0.001 material gain. Run: `runs/w384_d6_p020_continue_s109_lr15e5`.

## 4. New training optimizer: Muon

`experiments/muon_local.py` adapts the original MIT-licensed single-device algorithm from https://github.com/KellerJordan/Muon ; attribution and license are preserved in `experiments/MUON_LICENSE.txt`. Use FP32 Newton-Schulz operations for MPS, Muon only for hidden block matrices, and AdamW for embeddings and nonmatrix parameters. CPU and MPS checks both decreased a quadratic objective over 30 steps; rectangular/square matrices, zero gradients and descent direction passed. This has no inference parameter or operation overhead. Validation will determine whether continuation helps this already-trained model.

## 5. Learned causal prefix pointer

New module `student_learned_pointer.py` and trainer `experiments/train_learned_pointer.py`. Inspired by the pointer-mixture idea in Merity et al., https://arxiv.org/abs/1609.07843 ; this implementation uses new query/key projections and a learned gate over the existing Transformer hidden states. It can copy only known next tokens of strictly earlier positions within the same window. It replaces the old untrained hidden-similarity cache and retains existing local exact-match and training-only ngram components. Only 41,282 additional parameters at width 320. Training uses training windows and local copy likelihood, excluding in-sample training ngram lookup from the loss.

Correctness checks on the complete predictor: normalization error 1.19e-7, prefix equivalence maximum log-probability difference 7.63e-6, future-token independence, batch independence and repeated-call difference all zero. Training target-probability formula versus complete no-ngram inference differed by 6.98e-10. CPU and MPS pointer gradients finite. Full-validation benefit and idle resource limits remain to be checked after training.

Width-384 continuation stopped after the 1,500-step validation: 500/1000/1500 BPB = 1.497381972 / 1.494339022 / 1.491618712. Best checkpoint retained. Recent gains are too small relative to the 0.028947 gap to the current 320-width reference. SIGINT ended the process after that checkpoint; any subsequent unvalidated updates are discarded. The metrics trainer hash was corrected to the startup source snapshot (the working script had been extended for future experiments).

Muon continuation started from the reference model: seed 113, AdamW auxiliary LR 3e-5, Muon hidden-matrix LR 0.002, warmup 100 steps, batch 16, up to 1,800 steps, complete validation every 300, stop after three without a 0.0007 material gain. Run `runs/muon_d7_w320_s113_lr2e3`.

Muon continuation: validation at steps 300 and 600 was 1.481615102 and 1.480570783, both worse than the 1.462671644 parent. Stopped at the 600-step validation boundary at user request for local pause and SSH migration. `checkpoint.pt` remains the unchanged parent. No live optimizer/RNG snapshot exists; do not claim exact mid-run resumption. Next priority is the prepared learned-pointer experiment, not repeating this unsuccessful continuation.

Fast local scatter microbenchmark: exact synthetic batch log-probability agreement; median 1.820315 -> 1.562138 seconds (14.2% less) during simultaneous training. `runs/fast_local_scatter_d7_w320_microbenchmark`. This is exploratory, not an idle official timing. Natural-prefix equivalence verification was interrupted and must be rerun before adoption.

SSH pause handoff: no further real model training started. A small mocked trainer verified SIGTERM -> atomic snapshot -> new-directory resume against an uninterrupted four-step toy run; all parameter values were exactly equal. Snapshot stores optimizer, CPU/device RNG, sampling RNG, scheduler position, history and ancestry. New pointer trainer supports CPU/MPS/CUDA; CUDA remains untested on this Mac. Old Muon run last logged step650 after the600 validation; those unvalidated updates were discarded, so actual search cost is at least650 steps (2,662,400 target exposures).

Migration verification completed: natural-text old-vs-fast-scatter maximum log-probability delta 1.90735e-6, normalization 1.19209e-7, prefix delta zero. The combined learned-pointer + fast-scatter predictor passed small CPU checks: future, batch and repeat differences zero, prefix max difference 7.62939e-6, normalization1.19209e-7. Results saved in `migration/verification.json`. No real learned-pointer training or new full/test scoring was run. See project-root `SSH_HANDOFF.md` for the transfer procedure and exact next experiment.

## Interim release requested before SSH migration

User authorized one frozen full-test evaluation and preparation of local submission files; training remains paused. Existing cache-objective step200 weights plus equivalent fast local scatter and full-width320 hidden cache achieved complete validation 1.4622831529104345. Resource and contract checks passed, and sources/checkpoint were frozen at 2026-09-29T20:15:26.741879+08:00 before test. Full test 1.478713291044518, slightly worse than the earlier frozen1.478246158. This is not claimed as a test improvement. No further test-guided changes or candidate comparisons were made. Run `release_cache200_fast320_20260929_201339`; checkpoint `47103d970645bd1858cc5fee40fefe5bfb880c7c5b29e20652ab8d959b373f39`; test CPU ratio 4.5115, peak RSS 1563082752, inference bytes 66394918. The older release remains separately preserved.
