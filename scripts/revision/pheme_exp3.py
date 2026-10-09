exec(open('data/pheme_exp.py').read().split('res={}')[0])
rum=yp>0; idx=np.where(rum)[0]; Vr=sub(P['V'],idx); yr=yp[idx]; er=P['ev'][idx]; res={}
def fit3(V,y):
    skf=list(StratifiedKFold(5,shuffle=True,random_state=0).split(np.zeros(len(y)),y)); oof={}
    for v in VN:
        o=np.zeros((len(y),4))
        for tr,va in skf:
            m=clone(FACT[v]()).fit(V[v][tr],y[tr]); o[np.ix_(va,m.classes_)]=m.predict_proba(V[v][va])
        oof[v]=log(o)
    base={v:clone(FACT[v]()).fit(V[v],y) for v in VN}
    return base,LogisticRegression(C=1,max_iter=3000).fit(np.hstack([oof[v] for v in VN]),y)
for scheme in ['random5','leave_one_event_out']:
    pred=np.zeros(len(yr),int); pvs={v:np.zeros(len(yr),int) for v in VN}; per={}
    if scheme=='random5': folds=list(StratifiedKFold(5,shuffle=True,random_state=0).split(np.zeros(len(yr)),yr))
    else: folds=[(np.where(er!=e)[0],np.where(er==e)[0]) for e in ['charliehebdo','sydneysiege','ferguson','ottawashooting','germanwings-crash','putinmissing','prince-toronto','gurlitt']]
    for tr,te in folds:
        b,m=fit3(sub(Vr,tr),yr[tr]); p,pv=predict(b,m,sub(Vr,te)); pred[te]=p
        for v in VN: pvs[v][te]=pv[v]
        if scheme!='random5': per[str(er[te][0])]=dict(n=int(len(te)),acc=accuracy_score(yr[te],p)*100,majority=float(np.bincount(yr[te],minlength=4).max()/len(te)*100))
    mk=np.ones(len(yr),bool) if scheme=='random5' else np.isin(er,list(per))
    res[scheme]=dict(**summ(yr[mk],pred[mk],{v:pvs[v][mk] for v in VN},labels=(1,2,3)),per_event=per)
    print(scheme,json.dumps(res[scheme]),flush=True)
json.dump(res,open('results/revision/pheme_rumours.json','w'),indent=1)
