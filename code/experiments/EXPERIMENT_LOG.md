# MP1 experiment log (validation selection)

All new-model selection uses the supplied validation split. A test score is reported here only for the two previously frozen checkpoints. The evaluator, tokenizer, and benchmark splits are unchanged. `runs/` contains the exact checkpoints and per-run JSON records.

| Experiment | Training target tokens | Validation BPB | Test BPB | Notes |
| --- | ---: | ---: | ---: | --- |
| Original baseline, 1,200 steps | 9,830,400 | 2.071088 | 2.101266 | Original `model`, seed 17 |
| SwiGLU, 1,200 steps | 9,830,400 | 2.007883 | — | Equal-token baseline comparison |
| SwiGLU, 2,400 steps | 19,660,800 | 1.829998 | — | More training |
| SwiGLU + RoPE, 2,400 steps | 19,660,800 | 1.739580 | — | RoPE ablation at equal tokens |
| RoPE, width 192, depth 6, 3,600 steps | 29,491,200 | 1.639879 | 1.670922 | Previous frozen neural candidate |
| Previous neural + causal window cache | Same checkpoint | 1.581994 | — | Validation-only cache selection |
| Previous neural + window cache + train ngrams | Same checkpoint | 1.553444 | — | Pruned trigram asset |
| Width 192 dropout fine-tune, 3,000 extra steps | 54,067,200 cumulative | 1.613067 | — | From 3,600-step self-trained parent |
| Fine-tuned width 192 + window cache + train ngrams | Same checkpoint | 1.537481 | — | Validation only |
| Width 320, dropout 0.1, 6,000 steps | 49,152,000 | 1.569453 | — | Self-trained from scratch |
| Width 320 + window cache + train ngrams | Same checkpoint | 1.496020 | — | CPU validation 18.71 s when idle; baseline 4.54 s |
| Width 320 + temperature 1.2 + window cache + train ngrams | Same checkpoint | 1.485631 | — | Previous best; idle resource timing pending |
| Width 320, dropout 0.2, 6,000 steps | 49,152,000 | 1.544189 | — | Stronger regularization improved validation after step 3,000 |
| Width 320, dropout 0.2, extra 3,000 steps at learning rate 0.00005 | 73,728,000 cumulative | 1.536716 | — | Best at final step; seed 43; parent was the 6,000-step model |
| Width 320, dropout 0.2 + temperature 1.1 | Same checkpoint | 1.529038 | — | Temperature selected on validation |
| Width 320, dropout 0.2 + temperature 1.15 + window cache + train trigram | Same checkpoint | 1.484265 | — | Previous best complete validation score |
| Width 320, dropout 0.2 + temperature 1.15 + cache + train 3/4/5-grams | Same checkpoint | 1.477393 | — | Previous best complete validation; compact train-only asset |
| Fine-tuned width 320 + temperature 1.15 + cache + train 3/4/5-grams | 73,728,000 cumulative | 1.471416 | — | Previous best complete validation; CPU score 18.77 s vs baseline 4.33 s (4.34x), peak RSS 1.90 GB, inference assets 61,851,531 bytes |
| Ensemble-distilled width 320, 1,800 extra steps | 81,100,800 cumulative | 1.527787 raw | — | Raw neural score improved from 1.536716; with the compact cache and ngram recipe, validation-only probe 1.474463, worse than current best |
| Raw-neural-ensemble-distilled width 320, stopped after 700 steps | 76,595,200 cumulative at stop | 1.542245 raw at step 300; 1.474210 with cache and train ngrams | — | Validation at steps 300 and 600 did not improve the 1.536716 raw parent; combined predictor also worse than the seven-layer candidate |
| Seven-layer width 320 SAM continuation, stopped after 1,200 validated steps | 120,422,400 cumulative forward/backward target exposures at last validation | 1.530811 raw; 1.464959 with compact cache and ngrams | — | Improvement over the 1.465788 seven-layer parent predictor was only 0.000829 BPB; stopped for low marginal return |
| Seven-layer consistency continuation from SAM checkpoint, stopped after 600 steps | 125,337,600 cumulative target exposures at last validation | 1.527259 raw; 1.462628 full predictor at step 300 | — | Raw improved 0.000706 from steps 300 to 600, while the complete predictor at step 300 improved only 0.000044; stopped for low marginal return |
| Matched hard-only continuation, 900 extra steps | 77,414,400 cumulative at stop | 1.537164 raw | — | Same parent, seed, batch and learning-rate schedule as distillation; stopped after three validations; distillation was 1.529264 at 900 steps |
| Dropout 0.3 continuation from fine-tuned width 320 | 81,100,800 cumulative at stop | 1.552496 best at step 300 | — | Stopped after step 900; all three validations worse than 1.536716 parent |
| Token-specific output bias fitted with neural weights frozen | 78,643,200 cumulative | 1.535284 raw | — | Raw score slightly improved; compact cache and ngram validation-only probe 1.471867, worse than current best |
| Untied input/output weights, extra 900 steps | 81,100,800 cumulative at stop | 1.536824 best at step 900 | — | Stopped: raw score still worse than tied parent; calibrated window-cache probe 1.486683 vs parent 1.486937, marginal gain before train ngrams |
| Seven-layer identity-initialized width 320 before further training | Same 73,728,000 parent targets | 1.471416 with compact cache and ngrams | — | Exactly matches six-layer logits; CPU scoring 20.761 s vs baseline 4.327 s (4.80x); checkpoint plus asset 66,788,620 bytes |
| Seven-layer width 320, 1,500 extra steps + compact cache and ngrams | 86,016,000 cumulative | 1.470214 | — | Earlier seven-layer checkpoint; pruned train-only asset; checkpoint plus asset 66,394,662 bytes |
| Seven-layer width 320, best of 3,600 extra steps + compact cache and ngrams | 100,761,600 cumulative at best step 3,300 | 1.467413 | — | Earlier full-validation result; original evaluator CPU FP32 20.571 s vs same-machine baseline 4.327 s (4.75x); peak RSS 1,577,517,056 bytes; checkpoint plus pruned asset 66,394,662 bytes |
| Same seven-layer checkpoint with reduced trigram weight | Same 100,761,600 targets | 1.465788 | — | Earlier full-validation best; original evaluator confirmed validation-only weight search |
| SAM-fine-tuned seven-layer checkpoint + compact cache, ngrams and 128-dimensional hidden copy | 120,422,400 cumulative training target exposures | **1.462672** | **1.478246** | Frozen final predictor; CPU FP32 full-test scoring 23.226833 s versus baseline 5.102700 s (4.55x); checkpoint plus asset 66,394,790 bytes |
| Fine-tuned width 320 + equal blend with dropout 0.1 predictor | Two self-trained checkpoints | 1.446520 | — | Validation-only teacher target for distillation; dual inference does not meet the current single-model resource plan |
| Equal blend of dropout 0.2 and dropout 0.1 width-320 predictors | Two self-trained checkpoints | 1.450419 | — | Validation-only upper bound; direct dual-model inference likely exceeds CPU time limit |
| Blend of width-320 and width-192 predictors | Two self-trained checkpoints | 1.466770 | — | Validation-only probe; resource and final implementation pending |
| Width 320, dropout 0.1, extra 1,200 steps | 58,982,400 cumulative | 1.570856 | — | Stopped: worse than parent |
| Width 384, dropout 0.1, 3,600 steps | 29,491,200 | 1.597948 | — | Stopped: worse than width 320 at same step, tighter CPU budget |
| Width 384, depth 8, dropout 0.25, 600 validated steps | 4,915,200 validated targets | 1.975183 | — | Stopped after first validation: four-thread CPU FP32 scoring 26.768 s exceeds 5x same-machine baseline 21.637 s; eight-thread idle rerun was 21.193 s vs eight-thread baseline 3.829 s (5.53x); no test evaluation |
| Window-reset LSTM, width 512, 3 layers, 1,200 steps | 9,830,400 | 1.874561 | — | Stopped: far worse than Transformer at equal tokens |
| Two-component softmax head, 600 extra steps | 54,067,200 cumulative | 1.544073 best at step 200 | — | Stopped: no stable gain over 1.544189 parent |
| Width 192, depth 8, 4,800 steps | 39,321,600 | 1.655641 best at step 3,600 | — | Stopped: later validation worsened; no checkpoint saved by original trainer |

Training-derived ngram counts are in `assets/train_ngrams.npz` and were built only from the supplied training token sequence. The inference predictor resets all temporary state between scoring windows. The active low-learning-rate continuation and any later runs are recorded in their own `runs/` directories.

The compact 3/4/5-gram asset `assets/train_ngrams_3to5.npz` was also built only from the training split. The complete-validation 1.477393 score was reproduced by the original unchanged evaluator with `runs/variable_ngram_full45_t115_dropout_p020_d6_w320_6000_s17/checkpoint.pt`. The long-gram interpolation order was checked token by token against the validation-only probe. Both ensemble results are exploratory and are not submission candidates until the inference limits are verified.

The fine-tuned compact 3/4/5-gram candidate is `runs/variable_ngram_t115_dropout_p020_d6_w320_ft3000_lr5e5_s43/checkpoint.pt`. A same-machine idle measurement on the complete validation split used the unchanged evaluator, CPU, FP32 and four threads for both models: baseline scoring 4.327469 s, candidate scoring 18.773789 s. `/usr/bin/time -l` gave maximum resident set sizes 1,811,103,744 and 1,900,855,296 bytes respectively. The candidate checkpoint plus asset totals 61,851,531 bytes, below 64 MiB (67,108,864 bytes).

An earlier full-validation predictor is `runs/variable_ngram_pruned_t115_dropout_p020_d7_w320_best3300_s71/checkpoint.pt`. The seven-layer training continued from an identity-initialized seventh block derived from the six-layer checkpoint, for 3,600 planned extra steps. Its best raw validation score was 1.531929 at step 3,300 (not the final 3,600), and the cache plus training-only ngrams improved full validation to 1.467413. The idle CPU FP32 four-thread scoring took 20.571420 s, under five times the same-machine 4.327469 s baseline; peak RSS was 1,577,517,056 bytes. The checkpoint plus pruned ngram asset is 66,394,662 bytes, below 64 MiB.

For the seven-layer checkpoint, a fresh validation-only grid over neural temperature and causal window-cache weights again selected temperature 1.15, unigram weight 0.05, and bigram weight 0.25. The cache-only BPB was 1.481934 at that point.

A validation-only grid over compact train-derived trigram, 4-gram, and 5-gram weights selected 0.05, 0.05, and 0.08 instead of the previous 0.10, 0.05, and 0.08. The original unchanged evaluator confirmed full-validation BPB 1.4657878742034793 for `runs/variable_ngram_pruned_t115_tri005_dropout_p020_d7_w320_best3300_s71/checkpoint.pt`. This check ran concurrently with a separate training job, so its reported scoring time is not used for the inference limit; the previous idle 20.571420 s measurement has the same architecture and operations, but a final idle measurement remains necessary.

A hidden-state similarity copy rule, applied only to earlier positions in the same window, improved the seven-layer full predictor to 1.463264 validation BPB at weight 0.02 and similarity scale 10. The unchanged evaluator reproduced this result with `runs/hidden_cache_a002_b10_d7_w320_best3300_s71/checkpoint.pt`. A same-machine paired four-thread idle run measured 23.160208 s for this candidate and 4.760200 s for baseline (4.87x); this is close to the 5x limit and should be remeasured after all training stops. The model plus asset is 66,394,726 bytes and peak RSS was 1,642,741,760 bytes. A zero-weight implementation control exactly reproduced its 1.465787874 parent BPB.

SAM continuation of that seven-layer model lowered its raw validation BPB to 1.530811 at step 1,200, but the complete cache and ngram predictor improved only to 1.464959. It was stopped for low marginal return. Adding the hidden cache to this checkpoint achieved 1.462478 with all 320 hidden dimensions. Restricting similarity to 128 dimensions gave 1.462672, with only a 0.000194 BPB loss. An in-place hidden probability mixture reduced idle four-thread scoring to 21.370099 s and peak RSS to 1,800,044,544 bytes; same-machine baseline measurements ranged 4.327-4.760 s, leaving a small timing margin. Skipping final numerical renormalization did not reliably improve wall time, so it was rejected. The selected 128-dimensional candidate is `runs/hidden_cache_fast_a002_b10_dim128_d7_w320_sam1200_s83/checkpoint.pt` and totals 66,394,790 inference bytes.

A single validation-only probe compared hidden cache representations from each of the seven Transformer layers, with 128 and 320 dimensions. The final layer was best (1.462478 at 320 dimensions); layer 6 reached 1.462541 at 320 dimensions. No layer change was adopted.

The 128-dimensional hidden candidate was optimized further by scaling the existing probability table in place and adding 3-/4-/5-gram values without a full-size copy. The unchanged evaluator confirmed full-validation BPB 1.462671644 with `runs/hidden_cache_nocopy_a002_b10_dim128_d7_w320_sam1200_s83/checkpoint.pt`. A later idle paired measurement took 20.429174 s versus 4.492142 s for baseline (4.55x), and peak RSS was 2,082,521,088 bytes. A separate two-view dropout consistency continuation used MPS training but CPU FP32 validation. Its raw BPB improved from 1.530811 to 1.527259 by step 600, but the full predictor at step 300 was only 1.462628, a negligible 0.000044 gain over the parent. Training was stopped after the second validation point.

One validation-only probe of longer within-window repeated contexts on the current best predictor gave 1.462362 with local trigram and four-gram weights 0.05 and 0.10, versus 1.462672 without them. The 0.00031 gain was not adopted because it is small relative to the remaining target gap and would add another inference rule near the time limit.

The method was frozen after packaging and complete-validation reproduction from the independent staging directory, which exactly reproduced 1.462671644 BPB and the checkpoint, implementation and asset hashes. The unchanged evaluator then scored the frozen checkpoint once on full test: 1.4782461576027206 BPB, 23.226833 s, CPU FP32 four threads. The original baseline scored 2.1012656764238318 BPB and 5.102700 s in the paired full-test run. No later model adjustment was made from test feedback. The full-test evaluation records are in the release bundle and baseline run records.

Direct weight interpolation of the dropout 0.1 and 0.2 models was rejected on validation: second-model weights of 0.25, 0.5, and 0.75 yielded BPB 1.741886, 2.041519, and 1.661866 respectively. The two trained weight solutions cannot be averaged directly.

Weight interpolation between steps 2,400 and 3,300 of the same seven-layer continuation also failed to beat the step-3,300 model's raw validation BPB 1.531929: interpolation fractions 0.25, 0.5, and 0.75 toward step 3,300 yielded 1.532546, 1.532202, and 1.531994. The unaveraged step-3,300 checkpoint remains selected.

A causal window-cache weight adjustment based on repeated bigram count changed the fine-tuned model's validation BPB only from 1.486937 to 1.486912 at its best tested setting. This negligible gain was rejected to keep the inference rule simple.

Exponential recency weighting of within-window cache matches was also rejected. A joint validation-only grid of unigram/bigram weights and decay scales 32, 64, and 128 on the seven-layer raw checkpoint was best with no recency decay at all: cache-only BPB 1.481934. All tested positive recency-decay choices scored worse.

Increasing 4/5-gram interpolation weights with training-context count worsened validation BPB from 1.471415 (fixed weights) to 1.472657 at the smallest positive gain tried; the fixed interpolation remains selected.

A six-token train-derived lookup improved the 1.471416 predictor to 1.468821 when singleton contexts were retained, but that full index would exceed the remaining asset budget. Keeping only contexts seen at least twice improved it to merely 1.471001, so the longer lookup was not incorporated.
