# TEG-FND

Temporal Evidence-Graph Fusion Network for Fake News Detection.

## Headline model: GE-Stack (Graph-Enhanced Stacked Ensemble) -- use this one

```bash
python scripts/train_ensemble.py --dataset both                        # 10 random splits (literature protocol)
python scripts/train_ensemble.py --dataset both --split grouped        # story-disjoint (cosine >= 0.5)
python scripts/train_ensemble.py --dataset both --split grouped --group-threshold 0.3   # stricter story grouping
python scripts/train_ensemble.py --dataset both --split temporal       # train on past, test on newest 15%
```
CPU only, no GPU or DeBERTa download needed. Every run writes `results/<save-dir>/*_results.json` (per-split metrics, 95% CI, significance test, library versions) and `splits/*_test_ids.json` (the exact test tweet ids of every split, so baselines can be run on identical splits).

Measured test results (4-class: non-rumor / true / false / unverified), seeds 0-9:

| Protocol | Twitter15 acc | Twitter16 acc | Stack vs text-only (paired t-test) |
|---|---|---|---|
| Random stratified 70/15/15, 10 splits | **91.79% +/- 1.62** (95% CI 90.6-93.0), F1 91.77 | **92.44% +/- 2.03** (95% CI 90.9-94.0), F1 92.44 | +6.4 / +4.9 pts, p < 1e-3, 10/10 wins |
| Story-disjoint, cosine >= 0.5, 10 splits | 85.89% +/- 2.44 | 79.84% +/- 4.70 | +8.8 / +7.6 pts, p < 1e-3 |
| Story-disjoint, cosine >= 0.3, 10 splits | 71.47% +/- 4.03 | 70.08% +/- 3.50 | +16.6 / +10.9 pts, p < 1e-6 |
| Temporal (newest 15% per class), 1 split | 65.18% | 56.91% | -- |

Accuracy falls as train/test overlap is removed; this happens to every model on this benchmark (see PSA, Wu & Hooi 2022; "Examining the limitations of computational rumor detection models trained on static datasets", 2023). The benefit of the propagation (spreader + cascade) views *grows* as the split gets harder: +5-6 points on random splits, +11-17 points on strict story-disjoint splits. On the temporal split the spreader view alone (59.4% / 53.7%) beats the text view alone (44.2% / 42.3%). Split luck alone moves a 5-split mean by ~2 points (Twitter16: 90.57% on seeds 42,1,2,3,4 vs 92.44% on seeds 0-9), so single-split differences of 1-2 points between papers are not meaningful.

The grouped split uses numpy's frozen `RandomState` stream rather than scikit-learn's `StratifiedGroupKFold` (whose fold assignment changed between versions and gave different numbers on Kaggle), so the same seed gives the same split on any machine.

Per-view ablation (random split, earlier 5-seed run on seeds 42,1,2,3,4): Twitter15 -- text 86.7%, spreaders 71.3%, cascade 42.1%, stacked 92.1%. Twitter16 -- text 86.7%, spreaders 75.9%, cascade 50.6%, stacked 90.6%. The propagation views add ~4-5 points over text alone on the random split and ~7-10 points on the story-disjoint split, i.e. graph information is doing real work, as in GETAE.

**Neural baselines on identical test sets** (GPU, ~2-3 h on a Kaggle T4; resumable):
```bash
python scripts/run_deberta_baselines.py --out-dir /kaggle/working/baseline_results
```
Fine-tunes DeBERTa-v3 (`deberta`, text only) and TEG-FND (`tegfnd`, text + linguistic features) and re-scores GE-Stack on exactly the same test tweets (random seeds 0-2, story-disjoint seeds 0-2, temporal), writing `baseline_results.json` and a summary table. Re-running the same command skips finished runs.

**Architecture** (`src/models/graph_ensemble.py`), same principle as the GETAE base paper (text view + propagation-graph view, combined by an ensemble):
1. **Text view** -- word 1-2-gram + character 2-5-gram TF-IDF of the source tweet -> logistic regression.
2. **Spreader (graph) view** -- the users in the propagation tree ("who spread it") as a TF-IDF bag of user ids -> logistic regression. This is the item's neighbourhood in the user-news propagation graph, the same information GETAE's Node2Vec/DeepWalk embeddings and GCAN's user encoder use.
3. **Cascade view** -- 8 structural/temporal statistics of the cascade (size, unique users, delay mean/median/max, fraction spread within 5 min / 1 h / 1 day) -> logistic regression.
4. **Meta-learner** -- logistic regression over the three views' log-probabilities, trained on 5-fold **out-of-fold** predictions (like EnsembleNet's ensemble stage, but stacked instead of averaged).

**Evaluation protocol:** per seed, the same stratified 70/15/15 split function as `scripts/train.py`; the model is fit on train+val (the meta-learner's cross-validation replaces a separate validation set) and scored once on the untouched 15% test split. Results are written to `results/ensemble/*_results.json`, the first seed's model to `results/ensemble/*.joblib`.

**How to report these numbers honestly:**
- The random-split numbers use the same protocol as the Twitter15/16 literature and the base papers, so they are the comparable ones. Note they are **4-class**; GETAE and EnsembleNet report *binary* accuracy, an easier task.
- Twitter15/16 contain many tweets about the same story with the same label (about 15% of test tweets have a near-duplicate in train), and a random split puts siblings on both sides. That inflates every model on this benchmark, including published ones. The grouped split (`--split grouped`, tweets with word TF-IDF cosine >= 0.5 kept together) removes this and is the better estimate for unseen stories.
- Classes were collected at different times, so part of what the text and spreader views learn is "what was in the news / who was active then". This is a known property of these datasets.
- An independent review found no leakage in the pipeline: the vectorisers, scaler, base models and meta-learner are all fit without test data, and shuffling the training labels drops test accuracy to ~21%.

**Why not the DeBERTa models:** on the same split family, fine-tuned DeBERTa-v3 reached ~72% (`tegfnd`, train F1 0.99 vs val F1 0.72) and ~67% (`propagation`). With ~1k short tweets, an 86M-parameter transformer overfits, while sparse n-gram + spreader features with strong linear models generalise better. The neural models are kept below for ablation/comparison; `scripts/train.py` now also reports held-out **test** accuracy at the end of training.

---

## Current data contract
The supplied Twitter15/Twitter16 files contain `label.txt` and `source_tweets.txt`, plus a `tree/` directory of propagation trees (parent -> child edges with per-node relative timestamps), used by the `propagation` model variant. PolitiFact has text + verdict labels only (no propagation data).

## Neural model variants (`scripts/train.py --model`, kept for ablation)
- `tegfnd` (default): DeBERTa-v3 encoder → multi-view pooling (attention + mean + max) → stylometric linguistic-feature branch → cross-view attention fusion → adaptive Mixture-of-Experts (with a load-balancing auxiliary loss) → classification + uncertainty head.
- `propagation`: adds a graph-attention propagation-tree encoder and a temporal-delay encoder, fused with the text/linguistic views via a 4-way gated attention fusion. Twitter15/Twitter16 only.
- `deberta`: DeBERTa-v3 text-only baseline, for ablation.

## (Historical) Why `tegfnd` plateaus around 71-73% on Twitter15/16
A `tegfnd` run (text + linguistic features only, no propagation graph) on Twitter15 hits **Train F1 0.99 / Val F1 0.72** by epoch 13-15 — the gap opens by epoch ~5 and never closes. That's not a training-recipe problem you can regularize your way past indefinitely; it's a ceiling from **what information is actually in a single ~80-100 character source tweet**. Whether "the vote was rigged!!" is a rumor, confirmed true, confirmed false, or unverified frequently isn't decidable from the tweet text alone — it depends on how the community responded (denials, corroborations, the shape and speed of the retweet/reply cascade).

This is exactly the finding in the rumor-detection literature this dataset comes from (Ma et al. 2016/2017; the GCAN/BiGCN/PLAN line of work cited in the GETAE paper you supplied): text-only baselines on Twitter15/16 cluster in the low-to-mid 70s, while models that use the **propagation tree** reach the mid-80s to high-80s. GETAE itself (your base paper) gets its 82-90% by combining text with a Node2Vec/DeepWalk embedding of the propagation graph — not from a stronger text encoder alone. The data for this is already sitting in `data/twitter15/tree/` and `data/twitter16/tree/`, and this repo's `propagation` model (`PropagationRumorModel`) already consumes it — it just hadn't been trained yet.

**Concretely: run `--model propagation`, not `--model tegfnd`, if the target is 85%.** It won't be automatic — the graph branch has the same small-dataset overfitting risk as the text branch — but it's the architectural lever that's actually aligned with how the base papers get their numbers, whereas further tuning the text-only model is optimizing the wrong axis.

## Changes made on top of the 71%-accuracy baseline run
The original pipeline had several bugs/gaps that capped its accuracy well below what the architecture was capable of:

1. **Non-stratified train/val/test split.** `scripts/train.py` used `torch.utils.data.random_split`, which ignores labels, despite the README claiming a stratified split. Fixed: both training and evaluation now use `sklearn.model_selection.train_test_split(..., stratify=...)` for the 70/15/15 split, consistently.
2. **`scripts/train.py` could not actually train the documented "active model" (TEG-FND).** `build_model()` only supported `propagation`/`deberta`; `TEGFND` was never wired in. Fixed: `tegfnd` is now a first-class, and the default, `--model` choice.
3. **`Trainer` crashed (`KeyError`) on any non-propagation batch** because it unconditionally read `node_features`/`edge_index`/`delays` off every batch. Fixed: the trainer now builds model kwargs generically from whatever keys a batch contains, so the same trainer works for all three model variants.
4. **No learning-rate warmup/decay schedule**, and a single learning rate shared between the pretrained backbone and the randomly-initialized heads. Fixed: linear warmup + linear decay stepped per optimizer step, with a smaller LR for the backbone and a larger LR for the heads (`src/training/optim.py`).
5. **`WeightedCrossEntropy` never actually received class weights**, and there was no label smoothing. Fixed: both are wired in via `scripts/train.py --use-class-weights/--label-smoothing`.
6. **Full fine-tuning of an 86M-parameter backbone on ~1-1.5k training examples** is a heavy overfitting risk. Fixed: `--freeze-layers` freezes the embeddings + first N encoder layers (default 4) so only the upper layers and the task heads adapt.
7. **No early stopping** beyond "keep the best validation checkpoint" — training always ran a fixed 5 epochs. Fixed: `--patience` (default 5) stops training once validation macro-F1 stalls, with a longer `--epochs` budget (default 15) to actually reach convergence.
8. **`Trainer.fit()` didn't return anything**, so the "training history" that was saved to disk was `None`. Fixed: it now returns per-epoch loss/F1/accuracy for both splits.
9. **Vanilla soft-gated Mixture-of-Experts has no incentive to use more than one expert.** Fixed: added a coefficient-of-variation load-balancing auxiliary loss (`AdaptiveMixtureOfExperts.load_balancing_loss`) that is added to the training loss automatically when present in a model's output.
10. **Text pooling was a single learned-attention vector.** Fixed: `TextEncoder` now concatenates attention pooling, mean pooling and max pooling of the backbone's last hidden state (concat-pooling), which is a standard, parameter-cheap way to strengthen text classification on small datasets.
11. **Linguistic features were 14 generic character/token counts.** Fixed: extended to 22 features, adding sentence-length burstiness, an approximate Flesch reading-ease score, journalistic quoting markers ("said", quote counts) and first-person-pronoun ratio — stylometric signals that prior classical-ML fake-news work (n-gram/stylometric ensembles) found informative on top of contextual embeddings.
12. **Two divergent training scripts.** `scripts/train_propagation.py` duplicated (and partially diverged from) `scripts/train.py`. It's now a thin backwards-compatible wrapper around the single shared pipeline.

## Changes made after seeing the first real run (Train F1 0.99 / Val F1 0.72 on `tegfnd`)
13. **`GraphAttentionLayer`'s edge softmax looped over every node in Python** (`for node in range(num_nodes)`), O(num_nodes × num_edges). On the larger propagation trees in this dataset (some have hundreds of nodes) this was a real bottleneck and a plausible cause of the `propagation` model appearing to hang during training. Fixed: replaced with a vectorized scatter-based softmax (`scatter_reduce_`/`index_add_`) that computes the identical result (verified numerically against the old loop) with no Python-level loop — a 400-node synthetic graph went from a Python triple-nested-effectively loop to 0.1s.
14. **`--freeze-layers` default (4) wasn't enough to control overfitting.** Raised the default to 8 (of deberta-v3-base's 12 encoder layers), and bumped dropout 0.2→0.3 and weight decay 0.01→0.02 across `tegfnd`/`propagation`/`deberta`, given the observed train/val gap. `--head-lr` lowered 1e-4→5e-5 to match.

## Changes made after seeing the second real run (`propagation` scored WORSE than `tegfnd`: Val F1 0.6668 vs 0.7262)
This was the important one, because it meant *adding more information (the propagation graph) made the model worse* — that's not a data or hyperparameter problem, it's an architecture bug. Two things were checked and one was fixed:

- **Ruled out first:** sparse/uninformative propagation trees. Directly analyzed every file in `data/twitter15/tree/` (1490 trees): median 260 nodes, mean 402, min 56, max 2990. The graph data is large and rich, not sparse — this wasn't the cause.
- **Actual bug found in `PropagationFusion.forward()` (`src/models/rumor_model.py`).** The four views (semantic/text, linguistic, graph, temporal) go through cross-view attention and a learned gate that's supposed to decide how much each view contributes. But the fusion's residual/skip connection was `attended.mean(dim=1)` — a **blind, unweighted average of all four views**, added on top of the gated combination regardless of what the gate decided. That structurally forced at least ~25% weight onto every view, including the graph and temporal branches, even for an example where the gate correctly learned "the graph signal here is noise, ignore it." The gate could never fully suppress a bad view — it could only ever dilute the good ones. On the real run, that's consistent with exactly what was observed: adding the (informative, per the tree-size analysis above) graph and temporal views made the model *worse*, because on this dataset with this amount of training data, the propagation-graph and temporal branches are new, mostly-randomly-initialized capacity that hasn't learned to be reliable yet, and the blind averaging forced their noise into every prediction no matter what.

  **Fix:** the residual is now anchored on the text/semantic view specifically (`views[:, 0]`), not a blind mean of all four:
  ```python
  fused = self.norm(views[:, 0] + self.dropout(gated))
  ```
  The model's default is now "trust the text" (which is what actually got to 72-73% on its own), and the gate can *add* graph/temporal/linguistic evidence on top only when it helps a given example, instead of being structurally forced to blend in signal it has no way to fully turn off. This is the same reasoning GETAE and the BiGCN/PLAN line of work use implicitly by learning a graph representation *conditioned on* the text, rather than combining text and graph as unconditional equals.
- **`--freeze-layers=8` (round-2 fix for `tegfnd` overfitting) is likely too restrictive for `propagation`.** The `propagation` model has substantially more new, randomly-initialized capacity than `tegfnd` (graph encoder + temporal encoder + 4-way fusion, on top of the same classification head `tegfnd` has), which plausibly needs more backbone flexibility to learn a good joint text+graph representation within a limited epoch budget than a text-only model does. `--freeze-layers` now defaults to a **model-dependent value**: 8 (of deberta-v3-base's 12 layers) for `tegfnd`/`deberta`, 4 for `propagation`. Pass `--freeze-layers` explicitly to override either way; the resolved value is printed at the start of every run (`Freeze layers: N`) so it's visible in the training log.

None of this guarantees a specific number — it fixes real bugs (some of which meant parts of the pipeline could not previously run at all) and applies standard, well-evidenced techniques for fine-tuning transformers on small, class-balanced text datasets. **Retrain and re-evaluate to get the actual new number**; report it only after doing so.

## Evaluation contract
- Train/validation/test split: 70/15/15, stratified, **random** (tweet-level, not event-level -- same as the base papers). `scripts/train_ensemble.py --split grouped` additionally gives a story-disjoint split.
- Model selection: validation Macro-F1 only, with early stopping.
- Test set: evaluated only after the checkpoint is frozen.
- Primary metrics: Accuracy and Macro-F1.
- Secondary metrics: macro precision/recall and MCC.
- Twitter15/16: four-class classification (non-rumor/true/false/unverified) — note this is a *harder* task than the binary fake/real classification most base papers (e.g. GETAE, EnsembleNet) report on, so accuracy numbers are not directly comparable across the two setups.
- PolitiFact: six-class classification.

## Run
```bash
pip install -r requirements.txt

# TEG-FND neural model (ablation; needs GPU)
python scripts/train.py --dataset twitter15 --model tegfnd
python scripts/train.py --dataset twitter16 --model tegfnd

# Neural propagation-graph variant (ablation; needs GPU)
python scripts/train.py --dataset twitter15 --model propagation
# or, equivalently:
python scripts/train_propagation.py --dataset twitter15

# Resume an interrupted run from its last checkpoint (--epochs stays the
# ORIGINAL total budget, e.g. still 15, not "epochs remaining")
python scripts/train.py --dataset twitter15 --model propagation \
    --resume results/checkpoints/twitter15_propagation_seed42_last.pt --epochs 15

# Graph-Enhanced Stacked Ensemble (headline model, CPU, ~1-2 min)
python scripts/train_ensemble.py --dataset both

# Text-only baseline (ablation)
python scripts/train.py --dataset politifact --model deberta

# Evaluate a saved checkpoint on the held-out test split
python scripts/evaluate.py --dataset twitter15 --model tegfnd \
    --checkpoint results/checkpoints/twitter15_tegfnd_seed42.pt
```

Key training flags (see `python scripts/train.py --help` for the full list):
`--epochs`, `--batch-size`, `--grad-accum-steps`, `--backbone-lr`, `--head-lr`,
`--warmup-ratio`, `--label-smoothing`, `--freeze-layers`, `--patience`.

Do not report a benchmark number until the experiment is actually run.
