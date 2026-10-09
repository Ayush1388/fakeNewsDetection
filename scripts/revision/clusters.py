import sys, os, json, numpy as np, warnings; warnings.filterwarnings('ignore')
ROOT='/home/claude/fakeNewsDetection'; sys.path.insert(0,ROOT); sys.path.insert(0,ROOT+'/scripts'); os.chdir(ROOT)
from src.data.loaders import load_dataset
from train_ensemble import story_groups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
out={}
rng=np.random.RandomState(7); samples={}
for ds in ['twitter15','twitter16']:
    df=load_dataset(ds,f'data/{ds}'); y=df.label.to_numpy(); txt=df.text.to_numpy(dtype=object)
    X=TfidfVectorizer().fit_transform(txt); S=(X@X.T).toarray(); D=np.clip(1-S,0,None); np.fill_diagonal(D,0)
    out[ds]={}
    G={}
    for tau in [0.7,0.6,0.5,0.4,0.3,0.2]:
        g=story_groups(txt,threshold=tau); G[tau]=g
        u,c=np.unique(g,return_counts=True)
        multi=u[c>1]
        pur=[len(set(y[g==k]))==1 for k in multi]
        # tweet-weighted purity: share of tweets whose label equals the majority label of its cluster
        maj=sum(np.bincount(y[g==k]).max() for k in u)/len(y)
        # same-label rate of co-clustered pairs
        pairs=same=0
        for k in multi:
            lab=y[g==k]; n=len(lab); pairs+=n*(n-1)/2; same+=sum(v*(v-1)/2 for v in np.bincount(lab))
        sil=silhouette_score(D,g,metric='precomputed') if 1<len(u)<len(y) else float('nan')
        out[ds][str(tau)]=dict(clusters=int(len(u)),mean_size=float(c.mean()),mean_size_multi=float(c[c>1].mean()),singletons=float((c==1).mean()*100),
            in_multi=float(c[c>1].sum()/len(y)*100),largest=int(c.max()),silhouette=float(sil),pure_clusters=float(np.mean(pur)*100),
            majority_purity=float(maj*100),pair_same_label=float(same/pairs*100) if pairs else float('nan'),pairs=int(pairs))
        print(ds,tau,out[ds][str(tau)],flush=True)
    if ds=='twitter15':
        # precision samples: random co-clustered pairs at tau 0.5/0.3/0.2 (distinct texts)
        for tau in [0.5,0.3,0.2]:
            g=G[tau]; cand=[]
            for k in np.unique(g):
                idx=np.flatnonzero(g==k)
                if len(idx)>1: cand+= [(a,b) for ii,a in enumerate(idx) for b in idx[ii+1:]]
            cand=[p for p in cand if txt[p[0]].strip().lower()!=txt[p[1]].strip().lower()]
            pick=[cand[i] for i in rng.choice(len(cand),min(60,len(cand)),replace=False)]
            samples[f'prec_{tau}']=[dict(a=int(a),b=int(b),cos=float(S[a,b]),ta=txt[a],tb=txt[b]) for a,b in pick]
        # recall pool: directly similar pairs (cos>=0.2, distinct texts)
        iu=np.triu_indices(len(y),1); m=(S[iu]>=0.2)
        cand=[(a,b) for a,b in zip(iu[0][m],iu[1][m]) if txt[a].strip().lower()!=txt[b].strip().lower()]
        pick=[cand[i] for i in rng.choice(len(cand),80,replace=False)]
        samples['recall_pool']=[dict(a=int(a),b=int(b),cos=float(S[a,b]),ta=txt[a],tb=txt[b],co={str(t):bool(G[t][a]==G[t][b]) for t in [0.7,0.6,0.5,0.4,0.3,0.2]}) for a,b in pick]
        samples['n_recall_pool']=len(cand)
json.dump(out,open('results/revision/clusters.json','w'),indent=1)
json.dump(samples,open('results/revision/cluster_samples.json','w'),indent=1,ensure_ascii=False)
