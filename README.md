# Fake News Detection on Twitter: Graph-Enhanced Stacked Ensemble

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E?logo=scikitlearn&logoColor=white)
![Transformers](https://img.shields.io/badge/HF%20Transformers-DeBERTa--v3-FFD21E?logo=huggingface&logoColor=black)

Rumour detection on **Twitter15** and **Twitter16**, using both **what a tweet says** and **how it spreads**.

The headline model, **GE-Stack**, combines three views of every source tweet (its text, the users who spread it, and the timing and shape of its retweet cascade) in a stacked ensemble. On the standard 4-class task it reaches **~92% accuracy**, and it outperforms a fine-tuned DeBERTa-v3 by about 20 points on identical test sets.

---

## Results

4-class task (non-rumour / true / false / unverified). Mean ± std over **10 seeds** (0–9); test sets are never used for fitting.

| Evaluation protocol | Twitter15 | Twitter16 | Gain over text-only |
|---|---|---|---|
| Random stratified 70/15/15 (literature protocol) | **91.79% ± 1.62** (F1 91.77) | **92.44% ± 2.03** (F1 92.44) | +6.4 / +4.9 pts, p < 1e-3 |
| Story-disjoint split (cosine ≥ 0.5) | 85.89% ± 2.44 | 79.84% ± 4.70 | +8.8 / +7.6 pts |
| Story-disjoint split (cosine ≥ 0.3, stricter) | 71.47% ± 4.03 | 70.08% ± 3.50 | +16.6 / +10.9 pts |
| Temporal (train on past, test on newest 15%) | 65.18% | 56.91% | n/a |

### Against neural baselines (same test tweets, Kaggle T4)

| Split | GE-Stack | DeBERTa-v3 | TEG-FND (DeBERTa + MoE) |
|---|---|---|---|
| Twitter15 random | **92.26** | 71.88 | 68.01 |
| Twitter15 story-disjoint | **84.52** | 63.84 | 60.57 |
| Twitter16 random | **92.14** | 70.46 | 70.46 |
| Twitter16 story-disjoint | **82.93** | 62.87 | 60.98 |

On GETAE's **binary** protocol (true vs false, ten 80/20 splits), GE-Stack scores **95.37% / 95.78%**, compared with 82.7% / 89.6% reported in the GETAE paper.

### What the experiments show

- **Propagation features do real work.** They add 5–6 points on random splits and **11–17 points on story-disjoint splits**, so the harder the split, the more they help. On the temporal split, the spreader view alone (59.4%) beats the text view alone (44.2%).
- **Large transformers overfit here.** With about 1k short tweets, the 86M-parameter DeBERTa-v3-based TEG-FND reaches train F1 0.99 but only 0.72 on validation. Sparse n-gram and spreader features with linear models generalise far better.
- **Random-split scores are optimistic.** About 15% of test tweets have a near-duplicate in the training set, which inflates every model on this benchmark, including published ones. The story-disjoint split gives a more honest estimate for unseen stories.

---

## How it works

```
                 ┌─────────────────────────────┐
source tweet ──► │ Text view                   │ word 1-2-gram + char 2-5-gram TF-IDF → LogReg ─┐
                 └─────────────────────────────┘                                                │
                 ┌─────────────────────────────┐                                                ▼
propagation ───► │ Spreader view ("who")       │ bag of user IDs in the cascade → LogReg ──► Meta-learner ──► label
tree             └─────────────────────────────┘                                          (LogReg over
                 ┌─────────────────────────────┐                                           out-of-fold
             ──► │ Cascade view ("how fast")   │ size, unique users, delays, % spread      log-probs)
                 └─────────────────────────────┘ within 5 min / 1 h / 1 day → LogReg ───────────┘
```

The meta-learner is trained on **5-fold out-of-fold predictions**, so it learns how much to trust each view without seeing leaked base-model outputs. Vectorisers, scalers and every model are fit without test data. As a sanity check, shuffling the training labels drops test accuracy to about 21%, roughly chance for 4 classes.

The repo also includes the **neural models** used for comparison and ablation:

- **TEG-FND**: DeBERTa-v3 with multi-view pooling, a 22-feature stylometric branch, cross-view attention, and an adaptive Mixture-of-Experts with a load-balancing loss and an uncertainty head
- **Propagation model**: adds a graph-attention encoder over the propagation tree and a temporal-delay encoder, combined through gated 4-way fusion
- **LIME** explanations for individual predictions

---

## Quick start

```bash
pip install -r requirements.txt

# Headline model: CPU only, ~1-2 minutes, no GPU or model download
python scripts/train_ensemble.py --dataset both                   # random splits
python scripts/train_ensemble.py --dataset both --split grouped   # story-disjoint
python scripts/train_ensemble.py --dataset both --split temporal  # past -> future

# Ablations and comparisons
python scripts/ablation_views.py twitter15 random      # leave-one-view-out
python scripts/binary_getae_protocol.py                # GETAE's binary setup
python scripts/run_deberta_baselines.py --out-dir out  # neural baselines (GPU, ~2-3 h)
```

Each run writes per-split metrics, 95% confidence intervals, paired significance tests and the exact test tweet IDs to `results/`, so every number above can be reproduced and compared on identical splits.

## Repository layout

```
configs/        dataset configs (Twitter15, Twitter16, PolitiFact)
data/           Twitter15/16 source tweets, labels and propagation trees; PolitiFact claims
scripts/        training, evaluation, ablation and baseline scripts
src/data/       loaders, propagation-tree parsing, feature extraction
src/models/     GE-Stack ensemble, TEG-FND, graph/temporal encoders, MoE
src/training/   trainer, losses, LR schedules, seeding
src/evaluation/ metrics, uncertainty, evaluation
src/explainability/  LIME explainer
docs/           full experiment log and neural-architecture notes
```

## Notes

- Twitter15/16 results are **4-class**. Many papers (GETAE, EnsembleNet) report binary accuracy, which is an easier task, so compare with care.
- The development history is in [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md): what was tried, which bugs were found and fixed in the neural pipeline, and why the final model is a stacked ensemble rather than a transformer.

## Datasets

Twitter15 and Twitter16 come from Ma et al., *Detect Rumors in Microblog Posts Using Propagation Structure via Kernel Learning* (ACL 2017). PolitiFact claims are used for text-only experiments.
