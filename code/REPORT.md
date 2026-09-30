# MP1 Small Language Model Challenge - interim frozen report

Full-test CPU FP32 BPB: **1.478713291044518**. Full-validation BPB: **1.4622831529104345**. This predictor was selected on validation and frozen at 2026-09-29T 20:15:26.741879+08:00 before its test evaluation. This is an interim release, not a claim that the target of 1.4 was reached.

## 1. Protocol, data and test exposure

The supplied protocol is 7506-mp1-wt2-v2: WikiText-2 raw text, the supplied train-fitted BPE-2048 tokenizer, and independent causal windows of 256 targets. Only the supplied training split was used to fit weights and n-gram statistics. Validation selected architectures, checkpoints and interpolation settings. The data, tokenizer, common.py and evaluate.py remain unchanged. Each prediction uses only its own window prefix; no state survives between windows. No external text, pretrained weights or evaluation-network access was used.

An earlier separately frozen submission candidate had already been tested at BPB 1.4782461576027206. After that exposure, the user requested additional development, which used validation only. This new candidate improved validation but scored 1.478713291044518 on test, a deterioration of 0.0004671334. We disclose both evaluations, do not claim a test improvement, and did not alter or reselect this frozen candidate after seeing its test score. Repeated development on one validation split may overfit it; the earlier test exposure means this is not a never-seen holdout history.

## 2. Baseline and equal-target evidence

The supplied four-layer width 128 GPT, seed 17, batch 32 and 1200 updates processed 9,830,400 targets. It achieved validation BPB 2.071087821 and test BPB 2.101265676. At the same target count, replacing the feed-forward network with SwiGLU achieved validation 2.007882703. At 19,660,800 targets, SwiGLU alone scored 1.829997811 and adding RoPE scored 1.739580. These are equal-target architecture comparisons; the final longer-trained model is not an equal-compute comparison to the initial baseline.

## 3. Model and training ancestry

The neural model has width 320, depth 7,10 attention heads, tied token/output embeddings, LayerNorm, RoPE positions and SwiGLU feed-forward blocks. Training uses residual and embedding dropout 0.2. A six-layer model trained 6000 updates at batch 32 and seed 17, followed by 3000 low-learning-rate updates at seed 43. An identity-initialized seventh block was appended;3600 updates at seed 71 selected the step 3300 checkpoint. A SAM continuation at seed 83 selected step 1200 of an 1800-step learning-rate schedule. SAM counts two forward/backward passes per sampled batch.

Those ancestors account for 120,422,400 forward/backward target exposures. The latest cache-objective continuation used seed 101, batch 16, learning rate 3e-5, MPS FP32 training and CPU FP32 validation every 200 updates. It stopped at 600 updates because progress was negligible; the best step 200 contributes 819,200 inherited exposures, giving 121,241,600 in the released checkpoint. All 600 attempted updates count toward search cost. The objective is the likelihood of the neural and strictly causal local-copy mixture. In-sample training n-gram lookups are excluded from this training loss. This continuation's 128-dimensional hidden-cache predictor scored validation 1.462499357.

## 4. Complete predictor and ablations

Neural logits use temperature 1.15. A window-local empirical successor distribution after repeated one-token contexts receives weight 0.05; repeated two-token contexts receive 0.25, only where a match exists. Training-derived 3/4/5-gram distributions use weights 0.05/0.05/0.08. Trigram contexts with more than 128 successors are pruned;4/5-gram contexts must occur at least twice. All counts come only from the training sequence. Hidden-state cosine similarity copies observed successors from strictly earlier window positions with weight 0.02 and scale 10.

The present release uses all 320 hidden dimensions. An algebraically equivalent implementation accumulates the two local-copy distributions directly into one vocabulary-sized buffer, avoiding extra large intermediate probability arrays. It adds no parameters. On training-text probes its maximum log-probability difference from the original formula was 0.00000191. The faster implementation provides enough measured CPU margin for the full-width hidden cache. The unchanged evaluator confirmed validation 1.462283153 before freezing.

Mechanism ablations on fixed ancestor weights are available in the records. Before SAM: raw neural 1.531929; temperature plus window copy 1.481934; adding train n-grams with retuned weights 1.465788. On the SAM ancestor: raw neural 1.530811; window copy plus n-grams 1.464959; hidden copy over 128 dimensions 1.462672; hidden copy over 320 dimensions 1.462478. The current cache-objective weights with 128 dimensions give 1.462499, versus 1.462283 with 320 dimensions. Small late differences are not evidence of broad generalization gains; the newest test result did not improve.

Negative searches include separate-solution weight averaging, untied outputs, head biases, two ensemble distillation attempts, wider models, and dropout consistency. A two-model ensemble and a wider eight-layer model were excluded for resource cost. A recent width 384 continuation stopped at validation 1.491619. Muon continuation reached 1.481615/1.480571 at 300/600 updates, worse than its parent, and was paused for migration. A learned pointer head is prepared but untrained and is not part of this release. No new training was run while preparing this release.

## 5. Resources and search cost

The released model has 9,287,462 parameters. Its checkpoint is 37,186,422 bytes; the required n-gram asset is 29,208,496 bytes. Their sum is 66,394,918 bytes, below 64 MiB(67,108,864 bytes). Code and weights do not read any optimizer or training-state file at inference.

On the author's idle Mac, Python 3.12.14/PyTorch 2.7.1, CPU FP32 with 4 threads: full-validation scoring 20.879442 seconds versus baseline 4.975568 seconds; full-test scoring 23.547961 seconds versus baseline 5.219540 seconds, ratio 4.511501. Candidate peak test RSS was 1,563,082,752 bytes, below 4 GiB. Measurements are from serial independent processes using the unchanged scorer and /usr/bin/time -l. They are not a speed guarantee for other CPUs; reviewers should measure the same paired setup on their own machine.

The previous search inventory estimated lower bounds of 12.3 CPU-hours and 0.11 MPS-hours, excluding some probes. The resumed round adds at least 19 minutes of recorded mixed MPS-training/CPU-validation elapsed time, plus unvalidated interrupted updates and probes. Accelerator active time was not separately instrumented, so this wall time is not relabeled as exact NPU/GPU hours. The selected ancestor chain took roughly 4.8 CPU-hours before the brief MPS continuation. Detailed seeds, timings, checkpoint hashes and negative outcomes are included in experiments/run_records and OPTIMIZATION_ROUND2.md.

## 6. Reproduction, limitations and assistance

Extract the separate checkpoint bundle inside code/ so that submitted_checkpoint/checkpoint.pt and assets/train_ngrams_3to5_pruned128.npz exist. Run the unchanged evaluate.py with --device cpu --precision fp32 --threads 4 --split test. REPRODUCE.md gives exact commands and TRAINING_RECIPE.md gives the ancestry recipe. Source and asset hashes plus the pre-test freeze record are included. Training from scratch across platforms is not promised to be bitwise identical; exact reported-score reproduction uses the supplied frozen checkpoint.

Checkpoint SHA256: 47103d970645bd1858cc5fee40fefe5bfb880c7c5b29e20652ab8d959b373f39

Substantial OpenAI Codex assistance was used for implementations, scripts, debugging, model selection analysis, measurement, documentation and this report. The student must understand and take responsibility for the submitted work. The course baseline and fixed scorer were reused. Muon code in the negative-search scripts is adapted from Keller Jordan's MIT-licensed implementation; its license is included and Muon is not part of the final predictor. The supplied WikiText-2 attribution and licenses remain in code/README.md.
