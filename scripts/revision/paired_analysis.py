"""Paired tests of neural baselines vs GE-Stack from saved per-tweet predictions."""
import glob, json, os, re, sys, numpy as np, pandas as pd
from sklearn.metrics import f1_score
PRED = sys.argv[1] if len(sys.argv) > 1 else 'results/baselines/predictions'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'results/baselines/paired_vs_gestack.csv'
rng = np.random.RandomState(0); B = 4000
files = {}
for f in glob.glob(f'{PRED}/*.csv'):
    m = re.match(r'(twitter1[56])_(.+)_seed(\d+)_(gestack|deberta|tegfnd)\.csv', os.path.basename(f))
    if m: files[(m[1], m[2], int(m[3]), m[4])] = pd.read_csv(f)
def holm(p):
    p = np.asarray(p); o = np.argsort(p); adj = np.empty(len(p)); run = 0
    for r, i in enumerate(o): run = max(run, (len(p) - r) * p[i]); adj[i] = min(1, run)
    return adj
rows = []
for ds in ('twitter15', 'twitter16'):
    for proto in sorted({k[1] for k in files if k[0] == ds}):
        seeds = sorted({k[2] for k in files if k[0] == ds and k[1] == proto})
        for model in ('deberta', 'tegfnd'):
            pairs = []
            for s in seeds:
                g, n = files.get((ds, proto, s, 'gestack')), files.get((ds, proto, s, model))
                if g is None or n is None: continue
                g, n = g.sort_values('id').reset_index(drop=True), n.sort_values('id').reset_index(drop=True)
                assert (g.id.values == n.id.values).all() and (g.label.values == n.label.values).all(), 'test sets differ'
                pairs.append((g.label.values, g.pred.values, n.pred.values))
            if not pairs: continue
            # per-seed 0/1 correctness differences (GE-Stack minus neural)
            d = [((p == y).astype(float) - (q == y).astype(float)) for y, p, q in pairs]
            obs = np.mean([x.mean() for x in d]) * 100
            # paired bootstrap: resample tweets within each seed, average over seeds
            bs = np.array([np.mean([x[rng.randint(0, len(x), len(x))].mean() for x in d]) for _ in range(B)]) * 100
            # paired permutation (sign flips of per-tweet differences)
            perm = np.array([np.mean([(x * rng.choice([-1, 1], len(x))).mean() for x in d]) for _ in range(B)]) * 100
            p = (np.sum(np.abs(perm) >= abs(obs) - 1e-12) + 1) / (B + 1)
            f1d = np.mean([f1_score(y, p_, average='macro') - f1_score(y, q_, average='macro') for y, p_, q_ in pairs]) * 100
            rows.append(dict(dataset=ds, protocol=proto, model=model, seeds=len(pairs), n_test=len(pairs[0][0]),
                             gestack_acc=np.mean([(p_ == y).mean() for y, p_, _ in pairs]) * 100,
                             neural_acc=np.mean([(q_ == y).mean() for y, _, q_ in pairs]) * 100,
                             delta_acc=obs, ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5),
                             delta_macro_f1=f1d, p_perm=p))
df = pd.DataFrame(rows); df['p_holm'] = holm(df.p_perm.values) if len(df) else []
df['significant_holm_0.05'] = df.p_holm < 0.05
df.round(3).to_csv(OUT, index=False); print(df.round(3).to_string(index=False)); print('saved', OUT)
