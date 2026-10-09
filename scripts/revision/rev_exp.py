"""Revision experiments for GE-Stack (professor's review, round 3).
Stages (python rev_exp.py <dataset> <stage>):
  base   : single-classifier baselines (LR/SVM/NB on text and on concatenated views) and a time-only view
           (OOF + test probabilities) on the cached splits (random, story0.5, story0.3, temporal)
  cf     : stacking with story-aware and temporal (blocked) cross-fitting folds on random, story0.3, temporal
  early  : early detection: spreader/cascade views rebuilt from truncated trees (time deadlines, node budgets)
  rolling: expanding-window temporal evaluation with three chronological origins
All splits are read from the cached npz files of run_exp.py, so test sets are identical to the paper."""
import sys, os, json, time, re, warnings, numpy as np
warnings.filterwarnings('ignore')
ROOT = '/home/claude/fakeNewsDetection'
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + '/scripts'); os.chdir(ROOT)
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.pipeline import make_pipeline, make_union
from sklearn.compose import ColumnTransformer
from scipy.sparse import hstack, csr_matrix
from src.data.loaders import load_dataset
from src.models.graph_ensemble import load_propagation_views, _text_model, _spreader_model, _cascade_model
from train_ensemble import story_groups

ds, stage = sys.argv[1], sys.argv[2]
EXP = f'results/revision/exp/{ds}'; OUT = f'results/revision/exp2/{ds}'; os.makedirs(OUT, exist_ok=True)
df = load_dataset(ds, f'data/{ds}'); y = df['label'].to_numpy()
ids = df['id'].astype(str).to_numpy()
text = df['text'].to_numpy(dtype=object)
sp, ca = load_propagation_views(f'data/{ds}/tree', df['id'].tolist())
VIEWS = {'text': text, 'spreaders': sp, 'cascade': ca}
FACT = {'text': _text_model, 'spreaders': _spreader_model, 'cascade': _cascade_model}
VN = ['text', 'spreaders', 'cascade']
log = lambda p: np.log(np.clip(p, 1e-6, 1.0))
ts = np.array([((int(i) >> 22) + 1288834974657) / 1000.0 for i in ids])  # posting time (unix s)
import datetime
dt = [datetime.datetime.utcfromtimestamp(t) for t in ts]
TIME = np.array([[d.year, d.year * 100 + d.month, d.weekday(), d.isocalendar()[1]] for d in dt], dtype=float)
TIME = np.hstack([TIME, ts[:, None]])
SPLITS = [('random', s) for s in range(10)] + [('story0.5', s) for s in range(10)] + [('story0.3', s) for s in range(10)] + [('temporal', 0)]

def time_model():
    ct = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), [0, 1, 2, 3]), ('num', StandardScaler(), [4])])
    return make_pipeline(ct, LogisticRegression(C=1, max_iter=3000))

def oof_and_test(make, Xf, yf, Xt, folds):
    o = np.zeros((len(yf), 4))
    for a, b in folds:
        m = clone(make()).fit(Xf[a], yf[a]); o[b] = m.predict_proba(Xf[b])
    full = clone(make()).fit(Xf, yf)
    return log(o), full.predict_proba(Xt)

def save(fn, **kw): np.savez_compressed(fn, **kw)

if stage == 'base':
    for prot, seed in SPLITS:
        fn = f'{OUT}/base_{prot}_seed{seed}.npz'
        if os.path.exists(fn): continue
        t0 = time.time(); z = np.load(f'{EXP}/{prot}_seed{seed}.npz'); fit, te = z['fit'], z['te']; yf = y[fit]
        tv = make_union(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True), TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), sublinear_tf=True))
        Xt_f = tv.fit_transform(text[fit]); Xt_t = tv.transform(text[te])
        sv = TfidfVectorizer(token_pattern=r'\S+', min_df=2, sublinear_tf=True)
        Xs_f = sv.fit_transform(sp[fit]); Xs_t = sv.transform(sp[te])
        sc = StandardScaler().fit(ca[fit]); Xc_f = csr_matrix(sc.transform(ca[fit])); Xc_t = csr_matrix(sc.transform(ca[te]))
        mm = MinMaxScaler().fit(ca[fit]); Xm_f = csr_matrix(mm.transform(ca[fit])); Xm_t = csr_matrix(np.clip(mm.transform(ca[te]), 0, None))
        C_f = hstack([Xt_f, Xs_f, Xc_f]).tocsr(); C_t = hstack([Xt_t, Xs_t, Xc_t]).tocsr()
        N_f = hstack([Xt_f, Xs_f, Xm_f]).tocsr(); N_t = hstack([Xt_t, Xs_t, Xm_t]).tocsr()
        out = {}
        out['svm_text'] = LinearSVC(C=1).fit(Xt_f, yf).predict(Xt_t)
        out['nb_text'] = ComplementNB().fit(Xt_f, yf).predict(Xt_t)
        lrc = LogisticRegression(C=10, max_iter=3000).fit(C_f, yf); out['lr_concat'] = lrc.predict(C_t); out['lr_concat_p'] = lrc.predict_proba(C_t)
        out['svm_concat'] = LinearSVC(C=1).fit(C_f, yf).predict(C_t)
        out['nb_concat'] = ComplementNB().fit(N_f, yf).predict(N_t)
        # time-only view with the same folds as the cached OOF predictions
        skf = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(np.zeros(len(yf)), yf))
        out['oof_time'], out['pt_time'] = oof_and_test(time_model, TIME[fit], yf, TIME[te], skf)
        out['nnz'] = np.array([C_f.shape[1], int((lrc.coef_ != 0).sum())])
        save(fn, **out)
        print(ds, 'base', prot, seed, f'{time.time()-t0:.0f}s', flush=True)

if stage == 'cf':
    for prot, seed in [s for s in SPLITS if s[0] in ('random', 'story0.3', 'temporal')]:
        fn = f'{OUT}/cf_{prot}_seed{seed}.npz'
        if os.path.exists(fn): continue
        t0 = time.time(); z = np.load(f'{EXP}/{prot}_seed{seed}.npz'); fit, te = z['fit'], z['te']; yf = y[fit]
        g = story_groups(text[fit], threshold=0.3)                    # computed on training tweets only
        story_folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(np.zeros(len(yf)), yf, g))
        tf = np.zeros(len(yf), int)                                   # blocked chronological folds within each class
        for c in range(4):
            idx = np.flatnonzero(yf == c); idx = idx[np.argsort(ts[fit][idx], kind='stable')]
            tf[idx] = (np.arange(len(idx)) * 5) // len(idx)
        time_folds = [(np.flatnonzero(tf != k), np.flatnonzero(tf == k)) for k in range(5)]
        out = {}
        for name, folds in [('story', story_folds), ('time', time_folds)]:
            for v in VN:
                X = VIEWS[v][fit]; o = np.zeros((len(yf), 4))
                for a, b in folds:
                    m = clone(FACT[v]()).fit(X[a], yf[a]); o[b] = m.predict_proba(X[b])
                out[f'oof_{name}_{v}'] = log(o)
        out['n_story_groups'] = np.array(len(np.unique(g)))
        save(fn, **out)
        print(ds, 'cf', prot, seed, f'{time.time()-t0:.0f}s', flush=True)

def trunc_views(mode, k):
    """Spreader documents and cascade statistics from trees truncated at a deadline (minutes) or a node budget."""
    S, Cc = [], []
    NODE = re.compile(r"\['([^']*)', '([^']*)', '([^']*)'\]")
    for tid in df['id'].tolist():
        users, delays = [], []
        with open(f'data/{ds}/tree/{tid}.txt', encoding='utf-8') as f:
            for line in f:
                if '->' not in line: continue
                mt = NODE.search(line.split('->', 1)[1])
                if mt is None: continue
                try: dl = max(float(mt.group(3)), 0.0)
                except ValueError: dl = 0.0
                users.append(mt.group(1)); delays.append(dl)
        o = np.argsort(np.array(delays), kind='stable'); users = [users[i] for i in o]; delays = [delays[i] for i in o]
        keep = [i for i, d in enumerate(delays) if d <= k] if mode == 'time' else list(range(min(k, len(delays))))
        u = [users[i] for i in keep]; d = np.array([delays[i] for i in keep] or [0.0])
        S.append(' '.join(u))
        Cc.append([np.log1p(len(u)), np.log1p(len(set(u))), np.log1p(d.mean()), np.log1p(np.median(d)), np.log1p(d.max()),
                   (d <= 60).mean(), (d <= 5).mean(), (d <= 1440).mean()])
    return np.asarray(S, dtype=object), np.array(Cc)

if stage == 'early':
    SETTINGS = [('time', 5), ('time', 15), ('time', 30), ('time', 60), ('time', 180), ('time', 360),
                ('nodes', 10), ('nodes', 25), ('nodes', 50), ('nodes', 100)]
    TV = {f'{m}{k}': trunc_views(m, k) for m, k in SETTINGS}
    json.dump({kk: float(np.mean([len(s.split()) for s in v[0]])) for kk, v in TV.items()}, open(f'{OUT}/early_meansize.json', 'w'))
    for prot, seed in [('random', s) for s in range(5)] + [('story0.3', s) for s in range(5)] + [('temporal', 0)]:
        fn = f'{OUT}/early_{prot}_seed{seed}.npz'
        if os.path.exists(fn): continue
        t0 = time.time(); z = np.load(f'{EXP}/{prot}_seed{seed}.npz'); fit, te = z['fit'], z['te']; yf = y[fit]
        skf = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(np.zeros(len(yf)), yf))
        out = {}
        for key, (S, Cc) in TV.items():
            for v, X in (('spreaders', S), ('cascade', Cc)):
                out[f'oof_{key}_{v}'], out[f'pt_{key}_{v}'] = oof_and_test(FACT[v], X[fit], yf, X[te], skf)
        save(fn, **out)
        print(ds, 'early', prot, seed, f'{time.time()-t0:.0f}s', flush=True)

if stage == 'rolling':
    for origin in (0.55, 0.70, 0.85):
        fn = f'{OUT}/rolling_{int(origin*100)}.npz'
        if os.path.exists(fn): continue
        t0 = time.time(); fit, te = [], []
        for c in range(4):
            idx = np.flatnonzero(y == c); idx = idx[np.argsort(ts[idx], kind='stable')]
            a = int(round(origin * len(idx))); b = int(round((origin + 0.15) * len(idx)))
            fit += list(idx[:a]); te += list(idx[a:b])
        fit, te = np.array(fit), np.array(te); yf = y[fit]
        skf = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(np.zeros(len(yf)), yf))
        out = {'fit': fit, 'te': te, 'yt': y[te], 'yf': yf}
        for v in VN:
            out[f'oof_{v}'], out[f'pt_{v}'] = oof_and_test(FACT[v], VIEWS[v][fit], yf, VIEWS[v][te], skf)
        out['oof_time'], out['pt_time'] = oof_and_test(time_model, TIME[fit], yf, TIME[te], skf)
        save(fn, **out)
        print(ds, 'rolling', origin, len(fit), len(te), f'{time.time()-t0:.0f}s', flush=True)
print('DONE', ds, stage, flush=True)
