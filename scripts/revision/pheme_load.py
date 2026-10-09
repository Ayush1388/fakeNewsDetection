import json, glob, os, re, numpy as np, sys
from datetime import datetime
sys.path.insert(0,'data/pheme'); from convert_veracity_annotations import convert_annotations
R='data/pheme/all-rnr-annotated-threads'
def ts(s): return datetime.strptime(s,'%a %b %d %H:%M:%S %z %Y').timestamp()
LAB={'non-rumor':0,'true':1,'false':2,'unverified':3}
rows=[]
for ev in sorted(x for x in os.listdir(R) if x.endswith('-all-rnr-threads') and not x.startswith('.')):
    e=ev.replace('-all-rnr-threads','')
    for kind in ['rumours','non-rumours']:
        for tid in [t for t in os.listdir(f'{R}/{ev}/{kind}') if not t.startswith('.')]:
            p=f'{R}/{ev}/{kind}/{tid}'
            if not os.path.isdir(p): continue
            try:
                src=json.load(open(f'{p}/source-tweets/{tid}.json'))
                if kind=='non-rumours': lab='non-rumor'
                else:
                    import io,contextlib
                    with contextlib.redirect_stdout(io.StringIO()):
                        lab=convert_annotations(json.load(open(f'{p}/annotation.json')))
                    if lab is None: continue
                t0=ts(src['created_at']); users=[]; delays=[]
                for f in glob.glob(f'{p}/reactions/*.json'):
                    r=json.load(open(f)); users.append(r['user']['id_str']); delays.append((ts(r['created_at'])-t0)/60)
                rows.append(dict(id=tid,event=e,label=LAB[lab],text=re.sub(r'\s+',' ',src['text']).lower(),users=users,delays=delays,t0=t0))
            except Exception as ex: pass
def casc(users,delays):
    d=np.clip(np.asarray(delays or [0.0],float),0,None)
    return np.array([np.log1p(len(users)),np.log1p(len(set(users))),np.log1p(d.mean()),np.log1p(np.median(d)),np.log1p(d.max()),(d<=60).mean(),(d<=5).mean(),(d<=1440).mean()])
def build():
    y=np.array([r['label'] for r in rows]); ev=np.array([r['event'] for r in rows])
    return dict(y=y,ev=ev,ids=np.array([r['id'] for r in rows]),
      V={'text':np.array([r['text'] for r in rows],dtype=object),'spreaders':np.array([' '.join(r['users']) for r in rows],dtype=object),'cascade':np.vstack([casc(r['users'],r['delays']) for r in rows])})
if __name__=='__main__':
    D=build(); import collections
    print(len(rows),collections.Counter(D['y']),collections.Counter(D['ev']))
    print('empty reply sets',np.mean([len(r['users'])==0 for r in rows]))
