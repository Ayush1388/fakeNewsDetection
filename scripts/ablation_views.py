"""Leave-one-view-out ablation of GE-Stack (10 seeds).

    python scripts/ablation_views.py twitter15 random   # or: grouped
"""
import os, sys, json, numpy as np, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"scripts"))
from src.data.loaders import load_dataset
from src.models.graph_ensemble import GraphEnhancedStack, load_propagation_views
from train_ensemble import create_splits, create_grouped_splits, story_groups, subset
from sklearn.metrics import accuracy_score, f1_score
ds, prot = sys.argv[1], sys.argv[2]
df = load_dataset(ds, str(ROOT/"data"/ds)); y = df.label.to_numpy()
sp, ca = load_propagation_views(str(ROOT/"data"/ds/"tree"), df.id.tolist())
views = {"text": df.text.to_numpy(dtype=object), "spreaders": sp, "cascade": ca}
groups = story_groups(views["text"]) if prot == "grouped" else None
os.makedirs(ROOT/'results',exist_ok=True)
out = {}
for combo in [("text","spreaders"), ("text","cascade"), ("spreaders","cascade")]:
    accs, f1s = [], []
    for seed in range(10):
        if prot == "grouped": fit, te = create_grouped_splits(y, groups, seed)
        else:
            tr, va, te = create_splits(y, seed); fit = np.concatenate([tr, va])
        m = GraphEnhancedStack(use_views=combo, seed=seed).fit(subset(views, fit), y[fit])
        p = m.predict(subset(views, te)); accs.append(accuracy_score(y[te], p)); f1s.append(f1_score(y[te], p, average="macro"))
    out["+".join(combo)] = {"acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)), "f1_mean": float(np.mean(f1s)), "accs": accs}
    print(ds, prot, combo, round(np.mean(accs)*100,2), flush=True)
json.dump(out, open(ROOT/"results"/f"ablation_{ds}_{prot}.json","w"), indent=1)
