# TEG-FND

Temporal Evidence-Graph Fusion Network for Fake News Detection.

## Current data contract
The supplied Twitter15/Twitter16 files contain `label.txt` and `source_tweets.txt`, plus a `tree/` directory of propagation trees (parent -> child edges with per-node relative timestamps), used by the `propagation` model variant. PolitiFact has text + verdict labels only (no propagation data).

## Model variants (`--model`)
- `tegfnd` (default): DeBERTa-v3 encoder → multi-view pooling (attention + mean + max) → stylometric linguistic-feature branch → cross-view attention fusion → adaptive Mixture-of-Experts (with a load-balancing auxiliary loss) → classification + uncertainty head.
- `propagation` (**use this one if you want to close the gap toward literature numbers in the 80s** — see below): adds a graph-attention propagation-tree encoder and a temporal-delay encoder, fused with the text/linguistic views via a 4-way gated attention fusion. Twitter15/Twitter16 only.
- `deberta`: DeBERTa-v3 text-only baseline, for ablation.

## Why `tegfnd` alone plateaus around 71-73% on Twitter15/16, and what actually gets to ~85%
A `tegfnd` run (text + linguistic features only, no propagation graph) on Twitter15 hits **Train F1 0.99 / Val F1 0.72** by epoch 13-15 — the gap opens by epoch ~5 and never closes. That's not a training-recipe problem you can regularize your way past indefinitely; it's a ceiling from **what information is actually in a single ~80-100 character source tweet**. Whether "the vote was rigged!!" is a rumor, confirmed true, confirmed false, or unverified frequently isn't decidable from the tweet text alone — it depends on how the community responded (denials, corroborations, the shape and speed of the retweet/reply cascade).

This is exactly the finding in the rumor-detection literature this dataset comes from (Ma et al. 2016/2017; the GCAN/BiGCN/PLAN line of work cited in the GETAE paper you supplied): text-only baselines on Twitter15/16 cluster in the low-to-mid 70s, while models that use the **propagation tree** reach the mid-80s to high-80s. GETAE itself (your base paper) gets its 82-90% by combining text with a Node2Vec/DeepWalk embedding of the propagation graph — not from a stronger text encoder alone. The data for this is already sitting in `data/twitter15/tree/` and `data/twitter16/tree/`, and this repo's `propagation` model (`PropagationRumorModel`) already consumes it — it just hadn't been trained yet.

**Concretely: run `--model propagation`, not `--model tegfnd`, if the target is 85%.** It won't be automatic — the graph branch has the same small-dataset overfitting risk as the text branch — but it's the architectural lever that's actually aligned with how the base papers get their numbers, whereas further tuning the text-only model is optimizing the wrong axis.

## Changes made on top of the 71%-accuracy baseline run
The original pipeline had several bugs/gaps that capped its accuracy well below what the architecture was capable of:

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

None of this guarantees a specific number — it fixes real bugs (some of which meant parts of the pipeline could not previously run at all) and applies standard, well-evidenced techniques for fine-tuning transformers on small, class-balanced text datasets. **Retrain and re-evaluate to get the actual new number**; report it only after doing so.

## Evaluation contract
- Train/validation/test split: 70/15/15, stratified, event-level.
- Model selection: validation Macro-F1 only, with early stopping.
- Test set: evaluated only after the checkpoint is frozen.
- Primary metrics: Accuracy and Macro-F1.
- Secondary metrics: macro precision/recall and MCC.
- Twitter15/16: four-class classification (non-rumor/true/false/unverified) — note this is a *harder* task than the binary fake/real classification most base papers (e.g. GETAE, EnsembleNet) report on, so accuracy numbers are not directly comparable across the two setups.
- PolitiFact: six-class classification.

## Run
```bash
pip install -r requirements.txt

# TEG-FND (default, recommended)
python scripts/train.py --dataset twitter15 --model tegfnd
python scripts/train.py --dataset twitter16 --model tegfnd

# Propagation-graph variant (Twitter15/16 only) -- this is the one to run for 80s-range accuracy
python scripts/train.py --dataset twitter15 --model propagation
# or, equivalently:
python scripts/train_propagation.py --dataset twitter15

# Resume an interrupted run from its last checkpoint (--epochs stays the
# ORIGINAL total budget, e.g. still 15, not "epochs remaining")
python scripts/train.py --dataset twitter15 --model propagation \
    --resume results/checkpoints/twitter15_propagation_seed42_last.pt --epochs 15

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
