"""Statistics for the revision: one primary framework (paired bootstrap on identical test instances + paired
permutation test, Holm-corrected within each table), secondary Wilcoxon across repeated splits."""
import sys, json, glob, os, math, itertools, numpy as np, warnings
warnings.filterwarnings('ignore')
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
B_BOOT = 4000
VN = ['text', 'spreaders', 'cascade']
log = lambda p: np.log(np.clip(p, 1e-6, 1.0))
E1, E2 = './exp', './exp2'
DSS = ['twitter15', 'twitter16']
PROTS = {'random': range(10), 'story0.5': range(10), 'story0.3': range(10), 'temporal': [0]}
OUT = {}

def meta_fit(cols_f, yf, cols_t):
    m = LogisticRegression(C=1, max_iter=3000).fit(np.hstack(cols_f), yf)
    p = m.predict_proba(np.hstack(cols_t)); return m.classes_[p.argmax(1)], p

def macro_f1(y, p):
    cm = np.bincount(y * 4 + p, minlength=16).reshape(4, 4); tp = np.diag(cm).astype(float)
    den = cm.sum(0) + cm.sum(1); f = np.divide(2 * tp, den, out=np.zeros(4), where=den > 0)
    return f.mean()
def bal_acc(y, p):
    r = precision_recall_fscore_support(y, p, labels=range(4), zero_division=0)[1]
    return r[np.bincount(y, minlength=4) > 0].mean()
def tci(v):
    v = np.asarray(v, float); n = len(v)
    if n < 2: return [float(v.mean()), float('nan'), float('nan'), float('nan')]
    s = v.std(ddof=1); h = stats.t.ppf(0.975, n - 1) * s / math.sqrt(n)
    return [float(v.mean()), float(s), float(v.mean() - h), float(v.mean() + h)]
def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); adj = np.empty_like(ps); run = 0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (len(ps) - r) * ps[i])); adj[i] = run
    return adj.tolist()
def cohen_h(p1, p2): return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))

# --------------------------------------------------------------------------- predictions of every model, per split
def load_split(ds, prot, seed):
    z = np.load(f'{E1}/{ds}/{prot}_seed{seed}.npz'); yt, yf = z['yt'], z['yf']
    P, PR = {}, {}
    pred, prob = meta_fit([z[f'oof_{v}'] for v in VN], yf, [log(z[f'pt_{v}']) for v in VN])
    P['GE-Stack'], PR['GE-Stack'] = pred, prob
    for v in VN: P[v] = z[f'pt_{v}'].argmax(1); PR[v] = z[f'pt_{v}']
    pt = np.stack([z[f'pt_{v}'] for v in VN])                      # 3 x n x 4
    votes = pt.argmax(2); avg = pt.mean(0); mv = []
    for i in range(votes.shape[1]):
        c = np.bincount(votes[:, i], minlength=4)
        mv.append(np.flatnonzero(c == c.max())[np.argmax(avg[i][c == c.max()])])
    P['Majority voting'] = np.array(mv); P['Soft voting'] = avg.argmax(1); PR['Soft voting'] = avg
    P['Log-probability averaging'] = log(pt).mean(0).argmax(1)
    for name, use in [('Text + spreaders (stack)', ['text', 'spreaders']), ('Propagation only (stack)', ['spreaders', 'cascade'])]:
        P[name] = meta_fit([z[f'oof_{v}'] for v in use], yf, [log(z[f'pt_{v}']) for v in use])[0]
    fb = f'{E2}/{ds}/base_{prot}_seed{seed}.npz'
    if os.path.exists(fb):
        b = np.load(fb)
        P['LR, text'] = P['text']; P['Linear SVM, text'] = b['svm_text']; P['Naive Bayes, text'] = b['nb_text']
        P['LR, concatenated views'] = b['lr_concat']; P['Linear SVM, concatenated views'] = b['svm_concat']; P['Naive Bayes, concatenated views'] = b['nb_concat']
        P['Time only'] = b['pt_time'].argmax(1)
        P['Text + time (stack)'] = meta_fit([z['oof_text'], b['oof_time']], yf, [log(z['pt_text']), log(b['pt_time'])])[0]
        P['GE-Stack + time (stack)'] = meta_fit([z[f'oof_{v}'] for v in VN] + [b['oof_time']], yf, [log(z[f'pt_{v}']) for v in VN] + [log(b['pt_time'])])[0]
    fc = f'{E2}/{ds}/cf_{prot}_seed{seed}.npz'
    if os.path.exists(fc):
        c = np.load(fc)
        for k, nm in [('story', 'Story-aware cross-fitting'), ('time', 'Temporal cross-fitting')]:
            P[nm], PR[nm] = meta_fit([c[f'oof_{k}_{v}'] for v in VN], yf, [log(z[f'pt_{v}']) for v in VN])
    return yt, P, PR

SPL = {}
for ds in DSS:
    for prot, seeds in PROTS.items():
        for s in seeds: SPL[(ds, prot, s)] = load_split(ds, prot, s)

def splits(ds, prot): return [SPL[(ds, prot, s)] for s in PROTS[prot]]

def paired_compare(ds, prot, a, b, metric='acc', B=B_BOOT, seed=0):
    """Mean over splits of metric(a) - metric(b); paired bootstrap CI (instances resampled within each split,
    identical indices for both models) and paired permutation p-value (labels of the two models swapped per instance)."""
    rng = np.random.RandomState(seed); S = splits(ds, prot)
    f = (lambda y, p: (y == p).mean()) if metric == 'acc' else macro_f1
    obs = np.mean([f(y, P[a]) - f(y, P[b]) for y, P, _ in S]) * 100
    boot = np.zeros(B); perm = np.zeros(B)
    for r in range(B):
        db = []; dp = []
        for y, P, _ in S:
            n = len(y); i = rng.randint(0, n, n); db.append(f(y[i], P[a][i]) - f(y[i], P[b][i]))
            sw = rng.rand(n) < 0.5; pa = np.where(sw, P[b], P[a]); pb = np.where(sw, P[a], P[b]); dp.append(f(y, pa) - f(y, pb))
        boot[r] = np.mean(db) * 100; perm[r] = np.mean(dp) * 100
    p = (np.sum(np.abs(perm) >= abs(obs) - 1e-12) + 1) / (B + 1)
    ma = np.mean([(y == P[a]).mean() for y, P, _ in S]); mb = np.mean([(y == P[b]).mean() for y, P, _ in S])
    w = None
    if len(S) > 1:
        da = [f(y, P[a]) - f(y, P[b]) for y, P, _ in S]
        try: w = float(stats.wilcoxon(da).pvalue)
        except ValueError: w = None
    return dict(delta=float(obs), lo=float(np.percentile(boot, 2.5)), hi=float(np.percentile(boot, 97.5)), p=float(p),
                h=float(cohen_h(ma, mb)), wilcoxon=w)

def summary(ds, prot, model):
    S = splits(ds, prot)
    acc = [(y == P[model]).mean() * 100 for y, P, _ in S]; f1v = [macro_f1(y, P[model]) * 100 for y, P, _ in S]
    ba = [bal_acc(y, P[model]) * 100 for y, P, _ in S]
    pr = [precision_recall_fscore_support(y, P[model], labels=range(4), average='macro', zero_division=0)[0] * 100 for y, P, _ in S]
    return dict(acc=tci(acc), f1=tci(f1v), bacc=tci(ba), prec=tci(pr))

# --------------------------------------------------------------------------- 1. overall table with CIs (item 3)
R = {}
for ds in DSS:
    R[ds] = {p: summary(ds, p, 'GE-Stack') for p in PROTS}
OUT['overall'] = R

# --------------------------------------------------------------------------- 2. bootstrap CIs for single-split temporal + per-class (items 9, 30)
def boot_metrics(y, p, B=10000, seed=1):
    rng = np.random.RandomState(seed); n = len(y); acc = []; f1s = []; pc = []
    for _ in range(B):
        i = rng.randint(0, n, n); acc.append((y[i] == p[i]).mean()); f1s.append(macro_f1(y[i], p[i]))
        pc.append(np.stack(precision_recall_fscore_support(y[i], p[i], labels=range(4), zero_division=0)[:3]))
    pc = np.array(pc)
    q = lambda a: [float(np.percentile(a, 2.5) * 100), float(np.percentile(a, 97.5) * 100)]
    base = np.stack(precision_recall_fscore_support(y, p, labels=range(4), zero_division=0)[:3])
    return dict(acc=[float((y == p).mean() * 100)] + q(np.array(acc)), f1=[float(macro_f1(y, p) * 100)] + q(np.array(f1s)),
                bacc=float(bal_acc(y, p) * 100),
                perclass=[[[float(base[m, c] * 100)] + q(pc[:, m, c]) for m in range(3)] for c in range(4)],
                support=np.bincount(y, minlength=4).tolist())
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [(c - h) * 100, (c + h) * 100]
PC = {}
for ds in DSS:
    PC[ds] = {}
    for prot in PROTS:
        S = splits(ds, prot)
        if prot == 'temporal':
            y, P, _ = S[0]; PC[ds][prot] = boot_metrics(y, P['GE-Stack'])
            r = (P['GE-Stack'] == y) & (y == 1); PC[ds][prot]['true_recall_wilson'] = wilson(int(r.sum()), int((y == 1).sum())) + [int(r.sum()), int((y == 1).sum())]
        else:
            arr = np.array([np.stack(precision_recall_fscore_support(y, P['GE-Stack'], labels=range(4), zero_division=0)[:3]) for y, P, _ in S]) * 100
            PC[ds][prot] = dict(perclass=[[tci(arr[:, m, c]) for m in range(3)] for c in range(4)],
                                support=np.sum([np.bincount(y, minlength=4) for y, _, _ in S], 0).tolist(),
                                support_per_split=np.mean([np.bincount(y, minlength=4) for y, _, _ in S], 0).tolist())
OUT['perclass'] = PC

# --------------------------------------------------------------------------- 3. protocol transitions paired by seed (items 6, 16)
TR = {}
allp = []
for ds in DSS:
    acc = {p: np.array([(y == P['GE-Stack']).mean() * 100 for y, P, _ in splits(ds, p)]) for p in PROTS}
    TR[ds] = {}
    for a, b in [('random', 'story0.5'), ('story0.5', 'story0.3'), ('random', 'story0.3')]:
        d = acc[b] - acc[a]; m, s, lo, hi = tci(d)
        signs = np.array(list(itertools.product([1, -1], repeat=len(d))))
        pe = float(np.mean(np.abs((signs * d).mean(1)) >= abs(d.mean()) - 1e-12))
        rng = np.random.RandomState(3); bs = [rng.choice(d, len(d)).mean() for _ in range(10000)]
        TR[ds][f'{a}->{b}'] = dict(delta=m, sd=s, lo=lo, hi=hi, blo=float(np.percentile(bs, 2.5)), bhi=float(np.percentile(bs, 97.5)),
                                   p=pe, dz=float(m / s), wilcoxon=float(stats.wilcoxon(d).pvalue))
        allp.append((ds, f'{a}->{b}', pe))
    # temporal drop: bootstrap over temporal test instances and over random seeds
    y, P, _ = splits(ds, 'temporal')[0]; c = (y == P['GE-Stack']).astype(float); rng = np.random.RandomState(4)
    for a in ['random', 'story0.5', 'story0.3']:
        bs = [c[rng.randint(0, len(c), len(c))].mean() * 100 - rng.choice(acc[a], len(acc[a])).mean() for _ in range(10000)]
        dlt = c.mean() * 100 - acc[a].mean()
        pb = float(min(1.0, 2 * min(np.mean(np.array(bs) >= 0), np.mean(np.array(bs) <= 0)) + 1 / 10001))
        TR[ds][f'{a}->temporal'] = dict(delta=float(dlt), blo=float(np.percentile(bs, 2.5)), bhi=float(np.percentile(bs, 97.5)), p=pb,
                                        h=float(cohen_h(c.mean(), acc[a].mean() / 100)))
        allp.append((ds, f'{a}->temporal', pb))
adj = holm([x[2] for x in allp])
for (ds, k, _), a in zip(allp, adj): TR[ds][k]['p_holm'] = a
OUT['transitions'] = TR

# --------------------------------------------------------------------------- 4. fusion / single-classifier / time / cross-fitting comparisons vs GE-Stack
FAM = {
 'fusion': ['text', 'spreaders', 'cascade', 'Majority voting', 'Soft voting', 'Log-probability averaging', 'LR, concatenated views'],
 'linear': ['LR, text', 'Linear SVM, text', 'Naive Bayes, text', 'LR, concatenated views', 'Linear SVM, concatenated views', 'Naive Bayes, concatenated views'],
 'time': ['text', 'Propagation only (stack)', 'Time only', 'Text + time (stack)', 'GE-Stack + time (stack)'],
 'cf': ['Story-aware cross-fitting', 'Temporal cross-fitting'],
}
CMP = {}
for fam, models in FAM.items():
    CMP[fam] = {}; flat = []
    for ds in DSS:
        CMP[fam][ds] = {}
        prots = ['random', 'story0.3', 'temporal'] if fam == 'cf' else list(PROTS)
        for prot in prots:
            CMP[fam][ds][prot] = {'GE-Stack': summary(ds, prot, 'GE-Stack')}
            for mdl in models:
                if mdl not in splits(ds, prot)[0][1]: continue
                r = summary(ds, prot, mdl); r['vs'] = paired_compare(ds, prot, mdl, 'GE-Stack')
                if fam == 'cf': r['vs_f1'] = paired_compare(ds, prot, mdl, 'GE-Stack', metric='f1')
                CMP[fam][ds][prot][mdl] = r; flat.append((ds, prot, mdl, r['vs']['p']))
    adj = holm([x[3] for x in flat])
    for (ds, prot, mdl, _), a in zip(flat, adj): CMP[fam][ds][prot][mdl]['vs']['p_holm'] = a
    print('family', fam, 'tests', len(flat), flush=True)
OUT['compare'] = CMP

# GE-Stack vs its text view under the same framework (Table 7 column)
G = {}; flat = []
for ds in DSS:
    G[ds] = {}
    for prot in PROTS:
        G[ds][prot] = paired_compare(ds, prot, 'GE-Stack', 'text'); flat.append((ds, prot, G[ds][prot]['p']))
for (ds, prot, _), a in zip(flat, holm([x[2] for x in flat])): G[ds][prot]['p_holm'] = a
OUT['gain_text'] = G

# --------------------------------------------------------------------------- 5. calibration (item 29)
def ece(y, prob, bins=15):
    conf = prob.max(1); pred = prob.argmax(1); e = 0.0; edges = np.linspace(0, 1, bins + 1); rel = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean()); rel.append([float(conf[m].mean()), float((pred[m] == y[m]).mean()), int(m.sum())])
    return e * 100, rel
def brier(y, prob): return float(np.mean(np.sum((prob - np.eye(4)[y]) ** 2, 1)))
CAL = {}
for ds in DSS:
    CAL[ds] = {}
    for prot in ['random', 'story0.5', 'story0.3', 'temporal']:
        CAL[ds][prot] = {}
        for mdl in ['GE-Stack', 'text', 'Soft voting']:
            S = splits(ds, prot)
            e = [ece(y, PR[mdl])[0] for y, P, PR in S]; br = [brier(y, PR[mdl]) for y, P, PR in S]
            yy = np.concatenate([y for y, _, _ in S]); pp = np.concatenate([PR[mdl] for _, _, PR in S])
            conf = [float(PR[mdl].max(1).mean() * 100) for y, P, PR in S]
            CAL[ds][prot][mdl] = dict(ece=tci(e), brier=tci(br), conf=tci(conf), reliability=ece(yy, pp)[1])
OUT['calibration'] = CAL

# --------------------------------------------------------------------------- 6. confusion matrices (item 10)
CM = {}
for ds in DSS:
    CM[ds] = {p: [confusion_matrix(y, P['GE-Stack'], labels=range(4)).tolist() for y, P, _ in splits(ds, p)] for p in PROTS}
OUT['confusion'] = CM

# --------------------------------------------------------------------------- 7. early detection (item 28)
EA = {}
for ds in DSS:
    EA[ds] = {}
    ms = json.load(open(f'{E2}/{ds}/early_meansize.json')) if os.path.exists(f'{E2}/{ds}/early_meansize.json') else {}
    EA[ds]['mean_records'] = ms
    for prot, seeds in [('random', range(5)), ('story0.3', range(5)), ('temporal', [0])]:
        fs = [f'{E2}/{ds}/early_{prot}_seed{s}.npz' for s in seeds]
        if not all(os.path.exists(f) for f in fs): continue
        EA[ds][prot] = {}
        for key in ['time5', 'time15', 'time30', 'time60', 'time180', 'time360', 'nodes10', 'nodes25', 'nodes50', 'nodes100', 'full']:
            accs, accs_s, f1s = [], [], []
            for s, f in zip(seeds, fs):
                z = np.load(f'{E1}/{ds}/{prot}_seed{s}.npz'); yf, yt = z['yf'], z['yt']
                if key == 'full':
                    cf = [z[f'oof_{v}'] for v in VN]; ct = [log(z[f'pt_{v}']) for v in VN]; ps = z['pt_spreaders']
                else:
                    e = np.load(f)
                    cf = [z['oof_text'], e[f'oof_{key}_spreaders'], e[f'oof_{key}_cascade']]
                    ct = [log(z['pt_text']), log(e[f'pt_{key}_spreaders']), log(e[f'pt_{key}_cascade'])]; ps = e[f'pt_{key}_spreaders']
                pred = meta_fit(cf, yf, ct)[0]
                accs.append((pred == yt).mean() * 100); f1s.append(macro_f1(yt, pred) * 100); accs_s.append((ps.argmax(1) == yt).mean() * 100)
            text = [(np.load(f'{E1}/{ds}/{prot}_seed{s}.npz')['pt_text'].argmax(1) == np.load(f'{E1}/{ds}/{prot}_seed{s}.npz')['yt']).mean() * 100 for s in seeds]
            EA[ds][prot][key] = dict(acc=tci(accs), f1=tci(f1s), spr=tci(accs_s), text=tci(text))
OUT['early'] = EA

# --------------------------------------------------------------------------- 8. rolling-origin temporal (item 7)
RO = {}
for ds in DSS:
    RO[ds] = {}
    for o in (55, 70, 85):
        f = f'{E2}/{ds}/rolling_{o}.npz'
        if not os.path.exists(f): continue
        z = np.load(f); yt, yf = z['yt'], z['yf']
        pred = meta_fit([z[f'oof_{v}'] for v in VN], yf, [log(z[f'pt_{v}']) for v in VN])[0]
        r = dict(n_train=int(len(yf)), n_test=int(len(yt)), acc=float((pred == yt).mean() * 100), f1=float(macro_f1(yt, pred) * 100),
                 acc_ci=boot_metrics(yt, pred, B=2000)['acc'][1:], recall_true=float(((pred == yt) & (yt == 1)).sum() / max(1, (yt == 1).sum()) * 100))
        for v in VN: r[v] = float((z[f'pt_{v}'].argmax(1) == yt).mean() * 100)
        r['time'] = float((z['pt_time'].argmax(1) == yt).mean() * 100)
        RO[ds][o] = r
OUT['rolling'] = RO

# --------------------------------------------------------------------------- 9. neural baselines from seed-level summaries (item 8)
NEU = {  # mean, sd over seeds 0-2 (temporal: single split, sd None); accuracy, from the Kaggle run
 'twitter15': {'random': {'GE-Stack': (92.26, 1.43), 'DeBERTa-v3': (71.88, 1.18), 'TEG-FND': (68.01, 2.11)},
               'story0.5': {'GE-Stack': (84.52, 2.54), 'DeBERTa-v3': (63.84, 6.04), 'TEG-FND': (60.57, 3.23)},
               'temporal': {'GE-Stack': (65.18, None), 'DeBERTa-v3': (49.11, None), 'TEG-FND': (45.09, None)}},
 'twitter16': {'random': {'GE-Stack': (92.14, 1.87), 'DeBERTa-v3': (70.46, 4.62), 'TEG-FND': (70.46, 7.81)},
               'story0.5': {'GE-Stack': (82.93, 1.63), 'DeBERTa-v3': (62.87, 3.28), 'TEG-FND': (60.98, 3.26)},
               'temporal': {'GE-Stack': (56.91, None), 'DeBERTa-v3': (43.90, None), 'TEG-FND': (43.09, None)}}}
NT = {'twitter15': 224, 'twitter16': 123}
NR = {}; flat = []
for ds in DSS:
    NR[ds] = {}
    for prot, d in NEU[ds].items():
        NR[ds][prot] = {}
        g, gs = d['GE-Stack']
        for mdl in ['GE-Stack', 'DeBERTa-v3', 'TEG-FND']:
            m, s = d[mdl]; r = {'mean': m, 'sd': s}
            if s is not None:
                h = stats.t.ppf(0.975, 2) * s / math.sqrt(3); r['ci'] = [m - h, m + h]
            else:
                k = round(m / 100 * NT[ds]); r['ci'] = wilson(k, NT[ds])
            if mdl != 'GE-Stack':
                dlt = g - m
                if s is not None:   # Welch interval (conservative: the seed-level pairing is not used)
                    se = math.sqrt(gs ** 2 / 3 + s ** 2 / 3); dfw = se ** 4 / ((gs ** 2 / 3) ** 2 / 2 + (s ** 2 / 3) ** 2 / 2)
                    tq = stats.t.ppf(0.975, dfw); r['delta_ci'] = [dlt - tq * se, dlt + tq * se]
                    r['p'] = float(2 * stats.t.sf(abs(dlt) / se, dfw)); r['d'] = dlt / math.sqrt((gs ** 2 + s ** 2) / 2)
                else:               # Newcombe hybrid score interval for a difference of two proportions
                    n = NT[ds]; p1, p2 = g / 100, m / 100; l1, u1 = [x / 100 for x in wilson(round(p1 * n), n)]; l2, u2 = [x / 100 for x in wilson(round(p2 * n), n)]
                    dd = p1 - p2; r['delta_ci'] = [100 * (dd - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)), 100 * (dd + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2))]
                    pp = (p1 + p2) / 2; z = dd / math.sqrt(2 * pp * (1 - pp) / n); r['p'] = float(2 * stats.norm.sf(abs(z)))
                    r['h'] = cohen_h(p1, p2)
                r['delta'] = dlt; flat.append((ds, prot, mdl, r['p']))
            NR[ds][prot][mdl] = r
for (ds, prot, mdl, _), a in zip(flat, holm([x[3] for x in flat])): NR[ds][prot][mdl]['p_holm'] = a
OUT['neural'] = NR

json.dump(OUT, open('docs/revision/rev_results.json', 'w'), indent=1, default=float)
print('saved')
