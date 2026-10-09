import sys, os, json, numpy as np, warnings; warnings.filterwarnings('ignore')
ROOT='/home/claude/fakeNewsDetection'; sys.path.insert(0,ROOT); sys.path.insert(0,'.')
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
os.chdir(ROOT)
from src.data.loaders import load_dataset
from src.models.graph_ensemble import load_propagation_views, _text_model, _spreader_model, _cascade_model
import pheme_load
FACT={'text':_text_model,'spreaders':_spreader_model,'cascade':_cascade_model}; VN=['text','spreaders','cascade']
log=lambda p: np.log(np.clip(p,1e-6,1))
P=pheme_load.build(); yp=P['y']
def sub(V,i): return {k:V[k][i] for k in V}
def fit_stack(V,y,seed=0):
    skf=list(StratifiedKFold(5,shuffle=True,random_state=seed).split(np.zeros(len(y)),y)); oof={}
    for v in VN:
        o=np.zeros((len(y),4))
        for tr,va in skf:
            m=clone(FACT[v]()).fit(V[v][tr],y[tr]); o[va][:,:]=0; pr=m.predict_proba(V[v][va]); o[np.ix_(va,m.classes_)]=pr
        oof[v]=log(o)
    base={v:clone(FACT[v]()).fit(V[v],y) for v in VN}
    meta=LogisticRegression(C=1,max_iter=3000).fit(np.hstack([oof[v] for v in VN]),y)
    return base,meta
def predict(base,meta,V):
    pr={}
    for v in VN:
        p=np.zeros((len(V[v]),4)); p[:,base[v].classes_]=base[v].predict_proba(V[v]); pr[v]=p
    ps=meta.predict_proba(np.hstack([log(pr[v]) for v in VN])); full=np.zeros((len(ps),4)); full[:,meta.classes_]=ps
    return full.argmax(1),{v:pr[v].argmax(1) for v in VN}
def boot(yt,pred,n=2000,seed=0):
    r=np.random.RandomState(seed); a=[]
    for _ in range(n):
        i=r.randint(0,len(yt),len(yt)); a.append(accuracy_score(yt[i],pred[i])*100)
    return float(np.percentile(a,2.5)),float(np.percentile(a,97.5))
def summ(yt,pred,pv=None,labels=(0,1,2,3)):
    d=dict(n=int(len(yt)),acc=accuracy_score(yt,pred)*100,ci=boot(yt,pred),f1=f1_score(yt,pred,labels=list(labels),average='macro')*100,
      majority=float(np.bincount(yt,minlength=4).max()/len(yt)*100))
    if pv: d.update({f'acc_{v}':accuracy_score(yt,pv[v])*100 for v in VN})
    return d
res={}
# ---- A. transfer Twitter15/16 -> PHEME
T={}
for ds in ['twitter15','twitter16']:
    df=load_dataset(ds,f'data/{ds}'); sp,ca=load_propagation_views(f'data/{ds}/tree',df['id'].tolist())
    T[ds]=dict(y=df.label.to_numpy(),V={'text':df.text.to_numpy(dtype=object),'spreaders':sp,'cascade':ca})
T['both']=dict(y=np.concatenate([T['twitter15']['y'],T['twitter16']['y']]),V={k:np.concatenate([T['twitter15']['V'][k],T['twitter16']['V'][k]]) for k in VN})
rum=yp>0
for name in ['twitter15','twitter16','both']:
    base,meta=fit_stack(T[name]['V'],T[name]['y'])
    pred,pv=predict(base,meta,P['V'])
    vocab=base['spreaders'].steps[0][1].vocabulary_
    known=float(np.mean([any(u in vocab for u in d.split()) for d in P['V']['spreaders']])*100)
    # rumour-only veracity: restrict prediction to classes 1-3
    pr3=np.where(pred==0,1,pred)
    r=dict(all4=summ(yp,pred,pv),rumours_veracity3=summ(yp[rum],predict_mask:=None) if False else None)
    # 3-class: re-argmax excluding non-rumour using meta probs
    pm=np.hstack([log(np.eye(4)[pv[v]]) for v in VN]) if False else None
    res[f'A_{name}->pheme']=dict(all4=summ(yp,pred,pv),rumours_pred_as_4=summ(yp[rum],pred[rum]),known_user_share=known)
    print('A',name,json.dumps(res[f'A_{name}->pheme']),flush=True)
# ---- B. in-PHEME: random 5-fold vs leave-one-event-out
for scheme in ['random5','leave_one_event_out']:
    pred=np.zeros(len(yp),int); pvs={v:np.zeros(len(yp),int) for v in VN}
    if scheme=='random5': folds=[(tr,te) for tr,te in StratifiedKFold(5,shuffle=True,random_state=0).split(np.zeros(len(yp)),yp)]
    else:
        folds=[]
        for e in ['charliehebdo','sydneysiege','ferguson','ottawashooting','germanwings-crash','putinmissing','prince-toronto','gurlitt']:
            te=np.where(P['ev']==e)[0]; tr=np.where(P['ev']!=e)[0]; folds.append((tr,te))
    per={}
    for tr,te in folds:
        base,meta=fit_stack(sub(P['V'],tr),yp[tr]); p,pv=predict(base,meta,sub(P['V'],te)); pred[te]=p
        for v in VN: pvs[v][te]=pv[v]
        if scheme!='random5': per[str(P['ev'][te][0])]=dict(n=int(len(te)),acc=accuracy_score(yp[te],p)*100)
    m=np.ones(len(yp),bool) if scheme=='random5' else np.isin(P['ev'],list(per))
    res[f'B_{scheme}']=dict(all4=summ(yp[m],pred[m],{v:pvs[v][m] for v in VN}),rumours_only=summ(yp[m&rum],pred[m&rum]),per_event=per)
    print('B',scheme,json.dumps(res[f'B_{scheme}']),flush=True)
json.dump(res,open('docs/revision/pheme.json','w'),indent=1)
