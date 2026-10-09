"""Hardware-specific cost of GE-Stack on this machine: training (incl. 5-fold cross-fitting), inference, peak memory,
model size and coefficient counts. Run in a fresh process on an otherwise idle machine."""
import sys, os, time, json, pickle, resource, platform, numpy as np, warnings; warnings.filterwarnings('ignore')
ROOT='/home/claude/fakeNewsDetection'; sys.path.insert(0,ROOT); sys.path.insert(0,ROOT+'/scripts'); os.chdir(ROOT)
import sklearn, scipy
from src.data.loaders import load_dataset
from src.models.graph_ensemble import load_propagation_views, GraphEnhancedStack
ds=sys.argv[1]
t0=time.perf_counter(); df=load_dataset(ds,f'data/{ds}'); sp,ca=load_propagation_views(f'data/{ds}/tree',df['id'].tolist()); t_load=time.perf_counter()-t0
z=np.load(f'docs/revision/exp/{ds}/random_seed0.npz'); fit,te=z['fit'],z['te']; y=df.label.to_numpy()
V={'text':df.text.to_numpy(dtype=object),'spreaders':sp,'cascade':ca}
Vf={k:v[fit] for k,v in V.items()}; Vt={k:v[te] for k,v in V.items()}
times=[]
for rep in range(3):
    m=GraphEnhancedStack(seed=0); t0=time.perf_counter(); m.fit(Vf,y[fit]); times.append(time.perf_counter()-t0)
t0=time.perf_counter()
for rep in range(5): p=m.predict(Vt)
t_inf=(time.perf_counter()-t0)/5
acc=float((p==y[te]).mean())
size=len(pickle.dumps(m))
coef={}; nnz={}; nfeat={}
for k,pipe in m.base_models.items():
    lr=pipe.steps[-1][1]; coef[k]=int(lr.coef_.size+lr.intercept_.size); nnz[k]=int((lr.coef_!=0).sum()+(lr.intercept_!=0).sum()); nfeat[k]=int(lr.coef_.shape[1])
coef['meta']=int(m.meta.coef_.size+m.meta.intercept_.size); nnz['meta']=int((m.meta.coef_!=0).sum()+(m.meta.intercept_!=0).sum()); nfeat['meta']=int(m.meta.coef_.shape[1])
cpu=[l.split(':',1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name')]
r=dict(dataset=ds,n_train=int(len(fit)),n_test=int(len(te)),load_s=t_load,train_s=times,train_s_median=float(np.median(times)),
       infer_s=t_inf,infer_ms_per_tweet=t_inf/len(te)*1000,acc=acc,model_bytes=size,coef=coef,nonzero=nnz,features=nfeat,
       peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,cpu=cpu[0] if cpu else platform.processor(),cores=os.cpu_count(),
       python=platform.python_version(),sklearn=sklearn.__version__,numpy=np.__version__,scipy=scipy.__version__,threads=os.environ.get('OMP_NUM_THREADS','default'))
print(json.dumps(r)); json.dump(r,open(f'docs/revision/eff_{ds}.json','w'),indent=1)
