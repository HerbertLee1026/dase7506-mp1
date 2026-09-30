# MP1 Small Language Model Challenge - report draft

**Status:** validation-only draft. Add the final frozen test score and immutable download links before submission. Keep the submitted report at no more than 10 pages.

## 1. Task and protocol

The task is next-token prediction on the supplied WikiText-2 raw split with the fixed training-fitted BPE-2048 tokenizer. I used only the supplied training split for model weights and n-gram counts. I selected architecture, checkpoint, calibration and interpolation parameters on the validation split. The supplied evaluator, tokenizer and data were unchanged. The scorer resets state at each independent 256-target window. The reported ranking metric is the full-test bits per UTF-8 byte (BPB), measured with CPU FP32. No external text, pretrained checkpoint, validation-derived retrieval entry or test-based tuning was used.

## 2. Baseline and equal-token comparison

The original four-layer, width-128 GPT with seed 17, batch size 32 and 1,200 steps processed 9,830,400 training targets. It achieved validation BPB 2.071088 and full-test BPB 2.101266. At the same 9,830,400 targets, a SwiGLU replacement achieved validation BPB 2.007883. The difference is an equal-token architecture comparison, not a claim that the longer runs have equal training cost.

At 19,660,800 targets, SwiGLU alone achieved validation BPB 1.829998, while adding rotary position embeddings (RoPE) achieved 1.739580. This paired comparison isolates the effect of RoPE at the same processed-token count under this training setup. These early results motivated the later RoPE architecture, but architecture and training duration both changed in subsequent experiments.

## 3. Current best validation method

The current validation candidate uses a width-320, seven-layer RoPE and SwiGLU Transformer with 0.2 residual and embedding dropout. Its seventh block was initialized to an identity residual update from a trained six-layer model, then all layers were fine-tuned at a lower learning rate. A later 1,200-step sharpness-aware (SAM) continuation produced raw neural validation BPB 1.530811. Its cumulative count is 120,422,400 training-target forward/backward exposures, including the six-layer ancestor and two passes on each SAM batch. Checkpoint reuse and identity initialization did not reset the cost.

At inference, a temperature of 1.15 calibrates neural probabilities. A causal cache within each scoring window mixes in empirical successors after a repeated one-token context (weight 0.05) or two-token context (weight 0.25). Compact train-derived 3-, 4- and 5-gram distributions are then interpolated with weights 0.05, 0.05 and 0.08. The n-gram asset was built solely from the supplied training token sequence; the trigram table skips contexts with more than 128 continuations, and the 4-/5-gram tables keep contexts seen at least twice. Finally, normalized final-layer hidden vectors from 128 dimensions identify similar earlier positions in the same window. Their observed successors contribute weight 0.02, with similarity scale 10. Every cache uses only earlier positions, and no information passes between windows.

The unchanged evaluator confirmed complete-validation BPB **1.462672** for this combined predictor. The final test score remains to be measured after method freeze.

## 4. Ablation and critical analysis

Before SAM, the seven-layer neural model alone scored 1.531929 on validation. With temperature and the causal window cache, the same weights scored 1.481934. Adding train-derived n-grams with trigram weight 0.10 gave 1.467413; retuning that weight to 0.05 gave 1.465788. On the later SAM checkpoint, the neural model scored 1.530811 and the cache plus train n-grams scored 1.464959. Adding 128-dimensional hidden-copy inference then reached 1.462672. Each inference ablation uses one fixed neural checkpoint at a time. The window cache makes the largest single difference, consistent with repeated phrases in the local context. Train n-grams and hidden copy add smaller gains at the expense of file size and CPU time.

Several alternatives did not beat the current candidate: directly averaging separately trained dropout-0.1 and dropout-0.2 weights damaged validation BPB; an untied-output continuation and token-specific output biases yielded negligible full-predictor gains; and two ensemble distillation experiments did not improve the resource-compliant predictor. A width-384, eight-layer model violated the CPU timing limit and was stopped. A two-view dropout consistency continuation improved raw validation BPB but almost not the complete predictor. These negative results limit claims about general benefit. Hyperparameter selection on one validation split may overfit it, so the frozen full-test evaluation is the only final ranking claim.

## 5. Compute and inference resources

The present predictor has 9,287,462 trainable parameters. Its checkpoint is 37,186,294 bytes and its required train-derived n-gram asset is 29,208,496 bytes, totaling 66,394,790 bytes (below 64 MiB = 67,108,864 bytes). Idle same-machine CPU FP32 four-thread complete-validation scoring took 20.429174 seconds versus 4.492142 seconds for the original baseline (4.55 times). Peak resident memory was 2,082,521,088 bytes, below 4 GiB. Validation timing measured while another training process was active was excluded from the budget comparison.

Training used the supplied `.venv312` Python, mostly CPU FP32 with four threads. One discarded consistency experiment used MPS for training but CPU FP32 for validation. The experiment log and run records give seeds, steps, target exposures, checkpoint parent hashes and outcomes. Summing recorded run times gives a lower bound of approximately 12.3 CPU-hours and 0.11 MPS-hours for model search, excluding some intermediate probes and packaging. The selected checkpoint's own ancestor training took approximately 4.8 CPU-hours. This cost is much larger than the 20-second inference pass.

## 6. Reproducibility and AI assistance

The final repository must pin the code revision, include all model modules and exact evaluation commands, and link a matching checkpoint bundle containing every required inference asset. The evaluator reconstructs the predictor from the checkpoint's implementation module and configuration. The checkpoint and implementation hashes in the evaluator output identify the exact predictor used for the submitted score.

I used OpenAI Codex substantially for architecture implementation, experiment scripting, debugging, validation analysis, resource measurement, and preparation of this report. I reviewed the code, causal masking, training-data provenance and reported measurements. [Add any other reused work or assistance before final submission.]
