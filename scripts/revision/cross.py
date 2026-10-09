import sys, os, json, numpy as np, warnings; warnings.filterwarnings('ignore')
ROOT='/home/claude/fakeNewsDetection'; sys.path.insert(0,ROOT); sys.path.insert(0,ROOT+'/scripts'); os.chdir(ROOT)
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from src.data.loaders import load_dataset
from src.models.graph_ensemble import load_propagation_views, _text_model, _spreader_model, _cascade_model
FACT={'text':_text_model,'spreaders':_spreader_model,'cascade':_cascade_model}; VN=['text','spreaders','cascade']
log=lambda p: np.log(np.clip(p,1e-6,1))
D={}
for ds in ['twitter15','twitter16']:
    df=load_dataset(ds,f'data/{ds}'); sp,ca=load_propagation_views(f'data/{ds}/tree',df['id'].tolist())
    D[ds]=dict(y=df.label.to_numpy(),ids=df.id.astype(str).to_numpy(),V={'text':df.text.to_numpy(dtype=object),'spreaders':sp,'cascade':ca})
res={}
for A,Bn in [('twitter15','twitter16'),('twitter16','twitter15')]:
    a,b=D[A],D[Bn]; ya=a['y']
    skf=list(StratifiedKFold(5,shuffle=True,random_state=0).split(np.zeros(len(ya)),ya))
    oof={}; pt={}
    for v in VN:
        o=np.zeros((len(ya),4))
        for tr,va in skf: o[va]=clone(FACT[v]()).fit(a['V'][v][tr],ya[tr]).predict_proba(a['V'][v][va])
        oof[v]=log(o); pt[v]=clone(FACT[v]()).fit(a['V'][v],ya).predict_proba(b['V'][v])
    meta=LogisticRegression(C=1,max_iter=3000).fit(np.hstack([oof[v] for v in VN]),ya)
    pred=meta.predict(np.hstack([log(pt[v]) for v in VN]))
    pv={v:pt[v].argmax(1) for v in VN}
    idoverlap=np.isin(b['ids'],a['ids'])
    tv=TfidfVectorizer().fit(np.concatenate([a['V']['text'],b['V']['text']]))
    S=(tv.transform(b['V']['text'])@tv.transform(a['V']['text']).T).max(1).toarray().ravel()
    masks={'all':np.ones(len(b['y']),bool),'id_disjoint':~idoverlap,'id_story_disjoint':(~idoverlap)&(S<0.5),'id_story_disjoint_0.3':(~idoverlap)&(S<0.3)}
    r={}
    for k,m in masks.items():
        yt=b['y'][m]
        r[k]=dict(n=int(m.sum()),acc=accuracy_score(yt,pred[m])*100,f1=f1_score(yt,pred[m],average='macro')*100,
                 **{f'acc_{v}':accuracy_score(yt,pv[v][m])*100 for v in VN}, support=np.bincount(yt,minlength=4).tolist())
    # also: known-user share of test trees
    vocab=clone(FACT['spreaders']()).fit(a['V']['spreaders'],ya).steps[0][1].vocabulary_
    r['known_tree_share']=float(np.mean([any(u in vocab for u in d.split()) for d in b['V']['spreaders']])*100)
    r['id_overlap']=int(idoverlap.sum())
    res[f'{A}->{Bn}']=r; print(A,'->',Bn,json.dumps(r),flush=True)
    np.savez(f'results/revision/cross_{A}_{Bn}.npz',pred=pred,yt=b['y'],**{f'p_{k}':m for k,m in masks.items()})
json.dump(res,open('results/revision/cross.json','w'),indent=1)
