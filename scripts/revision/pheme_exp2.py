exec(open('data/pheme_exp.py').read().split('res={}')[0])
from sklearn.feature_extraction.text import TfidfVectorizer
T={}
for ds in ['twitter15','twitter16']:
    df=load_dataset(ds,f'data/{ds}'); sp,ca=load_propagation_views(f'data/{ds}/tree',df['id'].tolist())
    T[ds]=dict(y=df.label.to_numpy(),V={'text':df.text.to_numpy(dtype=object),'spreaders':sp,'cascade':ca})
T['both']=dict(y=np.concatenate([T['twitter15']['y'],T['twitter16']['y']]),V={k:np.concatenate([T['twitter15']['V'][k],T['twitter16']['V'][k]]) for k in VN})
rum=yp>0; res={}
def pred3(base,meta,V):
    pr={}
    for v in VN:
        p=np.zeros((len(V[v]),4)); p[:,base[v].classes_]=base[v].predict_proba(V[v]); pr[v]=p
    ps=meta.predict_proba(np.hstack([log(pr[v]) for v in VN])); full=np.zeros((len(ps),4)); full[:,meta.classes_]=ps
    full4=full.argmax(1); full[:,0]=-1; return full4, full.argmax(1), {v:np.where(np.arange(4)[None]==0,-1,pr[v]).argmax(1) for v in VN}
for name in ['twitter15','twitter16','both']:
    tv=TfidfVectorizer().fit(np.concatenate([T[name]['V']['text'],P['V']['text']]))
    S=(tv.transform(P['V']['text'])@tv.transform(T[name]['V']['text']).T).max(1).toarray().ravel()
    base,meta=fit_stack(T[name]['V'],T[name]['y']); p4,p3,pv3=pred3(base,meta,P['V'])
    r={}
    for k,m in {'all':np.ones(len(yp),bool),'story_disjoint_0.5':S<0.5,'story_disjoint_0.3':S<0.3}.items():
        mr=m&rum
        r[k]=dict(share_of_pheme=float(m.mean()*100),rumours_n=int(mr.sum()),
          veracity3=summ(yp[mr],p3[mr],{v:pv3[v][mr] for v in VN},labels=(1,2,3)) if mr.sum()>20 else None,
          four_class=summ(yp[m],p4[m]) if m.sum()>20 else None,
          nonrumour_recall=float((p4[m&~rum]==0).mean()*100) if (m&~rum).sum() else None)
    res[name]=r; print(name,json.dumps(r),flush=True)
json.dump(res,open('docs/revision/pheme_transfer.json','w'),indent=1)
