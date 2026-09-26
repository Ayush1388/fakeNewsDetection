"""GE-Stack on GETAE's binary protocol (true vs false tweets, ten stratified 80/20 splits).

    python scripts/binary_getae_protocol.py
"""
import os, sys, json, numpy as np, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"scripts"))
from src.data.loaders import load_dataset
from src.models.graph_ensemble import GraphEnhancedStack, load_propagation_views
from train_ensemble import subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
os.makedirs(ROOT/'results',exist_ok=True)
out={}
for ds in ["twitter15","twitter16"]:
    df=load_dataset(ds,str(ROOT/"data"/ds)); df=df[df.label.isin([1,2])].reset_index(drop=True)   # 1=true, 2=false
    y=(df.label.to_numpy()==2).astype(int)   # 1 = false/fake
    sp,ca=load_propagation_views(str(ROOT/"data"/ds/"tree"),df.id.tolist())
    views={"text":df.text.to_numpy(dtype=object),"spreaders":sp,"cascade":ca}
    res=[]; txt=[]
    for seed in range(10):
        tr,te=train_test_split(np.arange(len(y)),test_size=0.2,random_state=seed,stratify=y)
        m=GraphEnhancedStack(seed=seed).fit(subset(views,tr),y[tr]); p=m.predict(subset(views,te))
        res.append([accuracy_score(y[te],p),precision_score(y[te],p,average="macro"),recall_score(y[te],p,average="macro"),f1_score(y[te],p,average="macro")])
        pt=m.base_models["text"].predict(views["text"][te]); txt.append(accuracy_score(y[te],pt))
        print(ds,seed,round(res[-1][0],4),flush=True)
    r=np.array(res)
    out[ds]={"n":int(len(y)),"acc_mean":r[:,0].mean(),"acc_sd":r[:,0].std(ddof=1),"prec":r[:,1].mean(),"rec":r[:,2].mean(),"f1_mean":r[:,3].mean(),"f1_sd":r[:,3].std(ddof=1),"text_acc":float(np.mean(txt)),"per_seed_acc":r[:,0].tolist()}
    print(ds, {k:(round(v,4) if isinstance(v,float) else v) for k,v in out[ds].items() if k!="per_seed_acc"},flush=True)
json.dump(out,open(ROOT/"results"/"binary_getae_protocol.json","w"),indent=1)
