"""Extended GE-Stack analysis: per-class results, view complementarity, story-threshold
sensitivity, user-masking stress test, seed variation and class collection periods.

    python scripts/extended_analysis.py                 # run everything (CPU, ~1-1.5 h)
    python scripts/extended_analysis.py --stage analyse  # re-use cached base predictions
    python scripts/extended_analysis.py --stage figures  # redraw figures from the summary

How it works
------------
For every split (same split functions, seeds and hyper-parameters as
``scripts/train_ensemble.py``) the three base classifiers are fitted once and their
out-of-fold log-probabilities and test probabilities are cached in
``results/extended/<dataset>/<protocol>_seed<k>.npz``. Every stack variant (full stack,
leave-one-view-out, masked spreaders) is then a logistic-regression meta-learner on a
column subset of the same cached predictions, which is exactly what ``GraphEnhancedStack``
computes, so the numbers match ``train_ensemble.py`` and ``ablation_views.py``.

Protocols: random; story-disjoint with tau in {0.7, 0.6, 0.5, 0.4, 0.3, 0.2}; temporal.

User-masking stress test: a random 25/50/75/100 % of the users in the spreader vocabulary
are removed from all test trees and the trained model is applied without re-training
(100 % simulates a test set spread only by users never seen in training).

Outputs: ``docs/extended_results.json`` (summary) and ``docs/figures/*.png``.
"""
import argparse
import datetime
import glob
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scipy import stats  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

from src.data.loaders import load_dataset  # noqa: E402
from src.models.graph_ensemble import (  # noqa: E402
    _cascade_model, _spreader_model, _text_model, load_propagation_views,
)
from train_ensemble import (  # noqa: E402
    create_grouped_splits, create_splits, create_temporal_split, story_groups,
)

VIEWS = ["text", "spreaders", "cascade"]
FACTORY = {"text": _text_model, "spreaders": _spreader_model, "cascade": _cascade_model}
TAUS = (0.7, 0.6, 0.5, 0.4, 0.3, 0.2)
PROTOCOLS = ["random"] + [f"story{t}" for t in TAUS] + ["temporal"]
MASKED = ("random", "story0.5", "story0.3", "temporal")   # protocols with the user-masking test
LOO = {"-cascade": ["text", "spreaders"], "-spreaders": ["text", "cascade"], "-text": ["spreaders", "cascade"]}
CLASSES = ["N", "T", "F", "U"]


def log(p):
    return np.log(np.clip(p, 1e-6, 1.0))


def load(dataset):
    df = load_dataset(dataset, str(ROOT / "data" / dataset))
    y = df["label"].to_numpy()
    sp, ca = load_propagation_views(str(ROOT / "data" / dataset / "tree"), df["id"].tolist())
    views = {"text": df["text"].to_numpy(dtype=object), "spreaders": sp, "cascade": ca}
    return views, y, df["id"].astype(str).to_numpy()


# --------------------------------------------------------------------------- stage 1: base predictions
def run(dataset, cache):
    out = cache / dataset
    out.mkdir(parents=True, exist_ok=True)
    views, y, ids = load(dataset)
    for prot in PROTOCOLS:
        tau = float(prot[5:]) if prot.startswith("story") else None
        groups = story_groups(views["text"], threshold=tau) if tau else None
        for seed in ([0] if prot == "temporal" else range(10)):
            fn = out / f"{prot}_seed{seed}.npz"
            if fn.exists():
                continue
            t0 = time.time()
            if prot == "random":
                tr, va, te = create_splits(y, seed)
                fit = np.concatenate([tr, va])
            elif prot == "temporal":
                fit, te = create_temporal_split(y, ids)
            else:
                fit, te = create_grouped_splits(y, groups, seed)
            yf, yt = y[fit], y[te]
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            save = {}
            for v in VIEWS:
                X = views[v][fit]
                oof = np.zeros((len(yf), 4))
                for a, b in skf.split(np.zeros(len(yf)), yf):
                    oof[b] = clone(FACTORY[v]()).fit(X[a], yf[a]).predict_proba(X[b])
                full = clone(FACTORY[v]()).fit(X, yf)
                save[f"oof_{v}"] = log(oof)
                save[f"pt_{v}"] = full.predict_proba(views[v][te])
                if v == "spreaders" and prot in MASKED:
                    vocab = full.steps[0][1].vocabulary_
                    known = np.array(sorted(vocab))
                    docs = [d.split() for d in views[v][te]]
                    save["rec_known"] = np.array([np.mean([u in vocab for u in d]) if d else 0.0 for d in docs])
                    perm = np.random.RandomState(1000 + seed).permutation(len(known))
                    for frac in (25, 50, 75, 100):
                        hidden = set(known[perm[: int(round(frac / 100 * len(known)))]])
                        masked = np.array([" ".join(u for u in d if u not in hidden) for d in docs], dtype=object)
                        save[f"mask{frac}"] = full.predict_proba(masked)
            np.savez(fn, yf=yf, yt=yt, te=te, n_groups=(len(np.unique(groups)) if tau else -1),
                     largest=(int(np.bincount(groups).max()) if tau else -1), **save)
            print(f"{dataset} {prot} seed {seed}: n_test={len(te)} ({time.time() - t0:.0f}s)", flush=True)


# --------------------------------------------------------------------------- stage 2: analysis
def stack_pred(z, use=VIEWS, spr_key=None):
    meta = LogisticRegression(C=1, max_iter=3000).fit(np.hstack([z[f"oof_{v}"] for v in use]), z["yf"])
    Xt = np.hstack([log(z[spr_key] if (v == "spreaders" and spr_key) else z[f"pt_{v}"]) for v in use])
    return meta.classes_[meta.predict_proba(Xt).argmax(1)]


def summ(a):
    a = np.asarray(a, float) * 100
    d = {"mean": a.mean(), "min": a.min(), "max": a.max(), "n": len(a), "vals": a.round(2).tolist()}
    if len(a) > 1:
        d["sd"] = a.std(ddof=1)
        h = stats.t.ppf(0.975, len(a) - 1) * d["sd"] / np.sqrt(len(a))
        d["ci"] = [a.mean() - h, a.mean() + h]
    return d


def class_periods(dataset):
    """Posting dates per class from the Twitter snowflake ids."""
    _, y, ids = load(dataset)
    ts = lambda i: datetime.datetime.utcfromtimestamp(((int(i) >> 22) + 1288834974657) / 1000)
    dates = np.array([ts(i) for i in ids])
    _, te = create_temporal_split(y, ids)
    is_test = np.zeros(len(y), bool)
    is_test[te] = True
    out = {}
    for c, name in enumerate(CLASSES):
        d = np.sort(dates[y == c])
        t = dates[(y == c) & is_test]
        out[name] = {"p05": str(d[int(0.05 * len(d))].date()), "median": str(d[len(d) // 2].date()),
                     "p95": str(d[int(0.95 * len(d)) - 1].date()),
                     "test_from": str(t.min().date()), "test_to": str(t.max().date())}
    return out


def analyse(datasets, cache):
    R = {}
    for ds in datasets:
        R[ds] = {"class_periods": class_periods(ds)}
        for prot in PROTOCOLS:
            files = sorted(glob.glob(str(cache / ds / f"{prot}_seed*.npz")))
            if not files:
                continue
            acc, f1, pr, rc = [], [], [], []
            va = {v: [] for v in VIEWS}
            loo = {k: [] for k in LOO}
            pc, comp, known = [], [], []
            mask, cm = {}, np.zeros((4, 4), int)
            for fn in files:
                z = np.load(fn)
                yt = z["yt"]
                p = stack_pred(z)
                P, Rc, F, _ = precision_recall_fscore_support(yt, p, labels=range(4), zero_division=0)
                acc.append(accuracy_score(yt, p)); f1.append(F.mean()); pr.append(P.mean()); rc.append(Rc.mean())
                pc.append(np.stack([P, Rc, F])); cm += confusion_matrix(yt, p, labels=range(4))
                vp = {v: z[f"pt_{v}"].argmax(1) for v in VIEWS}
                for v in VIEWS:
                    va[v].append(accuracy_score(yt, vp[v]))
                for k, use in LOO.items():
                    loo[k].append(accuracy_score(yt, stack_pred(z, use)))
                ct, cs, cc = vp["text"] == yt, vp["spreaders"] == yt, vp["cascade"] == yt
                comp.append([np.mean(ct & cs), np.mean(ct & ~cs), np.mean(~ct & cs), np.mean(~ct & ~cs),
                             np.mean(ct | cs | cc), np.mean(vp["text"] == vp["spreaders"])])
                if "mask100" in z.files:
                    for fr in (25, 50, 75, 100):
                        mask.setdefault(fr, []).append(accuracy_score(yt, stack_pred(z, spr_key=f"mask{fr}")))
                    known.append([z["rec_known"].mean(), np.mean(z["rec_known"] > 0)])
            d = {"acc": summ(acc), "macro_f1": summ(f1), "macro_precision": summ(pr), "macro_recall": summ(rc),
                 "single_views": {v: summ(va[v]) for v in VIEWS}, "leave_one_out": {k: summ(loo[k]) for k in LOO},
                 "per_class_precision_recall_f1": (np.mean(pc, 0) * 100).round(2).tolist(),
                 "confusion_sum": cm.tolist(), "n_clusters": int(z["n_groups"]), "largest_cluster": int(z["largest"]),
                 "complementarity": dict(zip(["both", "only_text", "only_spreaders", "neither", "any_view", "agreement"],
                                             (np.mean(comp, 0) * 100).round(2).tolist()))}
            if len(files) > 1:
                a, t = np.array(acc), np.array(va["text"])
                d["stack_vs_text"] = {"paired_t_p": float(stats.ttest_rel(a, t).pvalue), "wins": int((a > t).sum())}
            if mask:
                d["user_masking"] = {str(fr): summ(mask[fr]) for fr in mask}
                d["known_users"] = {"share_of_test_records": float(np.mean(known, 0)[0] * 100),
                                    "share_of_test_trees_with_known_user": float(np.mean(known, 0)[1] * 100)}
            R[ds][prot] = d
            print(f"{ds:9s} {prot:9s} acc {d['acc']['mean']:.2f} ± {d['acc'].get('sd', 0):.2f}  "
                  f"views {[float(round(d['single_views'][v]['mean'], 1)) for v in VIEWS]}  "
                  f"loo {[float(round(d['leave_one_out'][k]['mean'], 1)) for k in LOO]}  "
                  f"mask {[float(round(v['mean'], 1)) for v in d.get('user_masking', {}).values()]}")
    (ROOT / "docs").mkdir(exist_ok=True)
    json.dump(R, open(ROOT / "docs" / "extended_results.json", "w"), indent=1)
    print("wrote docs/extended_results.json")


# --------------------------------------------------------------------------- stage 3: figures
def figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    R = json.load(open(ROOT / "docs" / "extended_results.json"))
    out = ROOT / "docs" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    names = {"twitter15": "Twitter15", "twitter16": "Twitter16"}
    dss = [d for d in names if d in R]
    col = {"stack": "#1F4E79", "text": "#2E75B6", "spreaders": "#548235", "cascade": "#BF9000"}

    fig, axs = plt.subplots(1, len(dss), figsize=(5.2 * len(dss), 3.8), sharey=True, squeeze=False)
    for ax, ds in zip(axs[0], dss):
        keys = ["random"] + [f"story{t}" for t in TAUS]
        x = np.arange(len(keys))
        ax.errorbar(x, [R[ds][k]["acc"]["mean"] for k in keys], yerr=[R[ds][k]["acc"]["sd"] for k in keys],
                    color=col["stack"], marker="o", lw=2, capsize=3, label="GE-Stack")
        for v in VIEWS:
            ax.plot(x, [R[ds][k]["single_views"][v]["mean"] for k in keys], color=col[v], marker="s", label=f"{v} view")
        ax.set_xticks(x, ["random"] + [str(t) for t in TAUS]); ax.set_xlabel("story-similarity threshold (stricter →)")
        ax.set_title(names[ds]); ax.grid(axis="y", alpha=0.3)
    axs[0][0].set_ylabel("accuracy (%)"); axs[0][0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(out / "threshold_sensitivity.png", dpi=200); plt.close(fig)

    fig, axs = plt.subplots(1, len(dss), figsize=(5.2 * len(dss), 3.8), sharey=True, squeeze=False)
    for ax, ds in zip(axs[0], dss):
        for prot, c in zip(MASKED, ["#1F4E79", "#2E75B6", "#548235", "#BF9000"]):
            d = R[ds][prot]
            ax.plot([0, 25, 50, 75, 100], [d["acc"]["mean"]] + [d["user_masking"][k]["mean"] for k in ("25", "50", "75", "100")],
                    color=c, marker="o", label=prot)
            ax.plot([100], [d["leave_one_out"]["-spreaders"]["mean"]], marker="*", ms=12, color=c, mec="black", ls="none")
        ax.set_xlabel("known spreaders hidden at test time (%)"); ax.set_title(names[ds]); ax.grid(axis="y", alpha=0.3)
    axs[0][0].set_ylabel("GE-Stack accuracy (%)  (★ = stack without spreader view)"); axs[0][0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(out / "user_masking.png", dpi=200); plt.close(fig)

    fig, axs = plt.subplots(len(dss), 3, figsize=(10, 3.5 * len(dss)), squeeze=False)
    for i, ds in enumerate(dss):
        for j, prot in enumerate(["random", "story0.5", "temporal"]):
            M = np.array(R[ds][prot]["confusion_sum"]); Rn = M / M.sum(1, keepdims=True); ax = axs[i][j]
            ax.imshow(Rn, cmap="Blues", vmin=0, vmax=1)
            for a in range(4):
                for b in range(4):
                    ax.text(b, a, f"{M[a, b]}\n({100 * Rn[a, b]:.0f}%)", ha="center", va="center", fontsize=8,
                            color="white" if Rn[a, b] > 0.55 else "black")
            ax.set_xticks(range(4), CLASSES); ax.set_yticks(range(4), CLASSES); ax.set_title(f"{names[ds]}, {prot}", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "confusion_matrices.png", dpi=200); plt.close(fig)
    print(f"wrote figures to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["twitter15", "twitter16", "both"], default="both")
    ap.add_argument("--stage", choices=["run", "analyse", "figures", "all"], default="all")
    ap.add_argument("--cache", default=str(ROOT / "results" / "extended"))
    args = ap.parse_args()
    dsets = ["twitter15", "twitter16"] if args.dataset == "both" else [args.dataset]
    os.chdir(ROOT)
    if args.stage in ("run", "all"):
        for ds_ in dsets:
            run(ds_, Path(args.cache))
    if args.stage in ("analyse", "all"):
        analyse(dsets, Path(args.cache))
    if args.stage in ("figures", "all"):
        figures()
