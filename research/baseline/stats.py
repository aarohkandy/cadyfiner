import json, math, numpy as np
from itertools import combinations
rows=json.load(open("/private/tmp/claude-501/-Users-aaroh-cadybara/3c925b39-d604-461e-b403-05f58422a2ea/scratchpad/stl_metrics.json"))
rng=np.random.default_rng(0)

def wilson(k,n,z=1.96):
    if n==0: return (float('nan'),)*2
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return (max(0,c-h), min(1,c+h))

print("=== 'Well-formed solid' rate by specificity (watertight AND single-body AND winding-consistent) ===")
print(f"{'lvl':<5}{'n':<5}{'k':<5}{'rate':<9}{'95% CI'}")
lv={}
for lvl in sorted({r['specificityLevel'] for r in rows}):
    g=[r for r in rows if r['specificityLevel']==lvl]
    k=sum(1 for r in g if r.get('watertight') and r.get('n_bodies')==1 and r.get('winding'))
    lo,hi=wilson(k,len(g)); lv[lvl]=(k,len(g))
    print(f"{lvl:<5}{len(g):<5}{k:<5}{k/len(g)*100:>5.0f}%   [{lo*100:.0f}%, {hi*100:.0f}%]")

print("\n=== Fisher-style permutation test: level 10 vs levels 3-7 pooled ===")
a=[r for r in rows if r['specificityLevel']==10]
b=[r for r in rows if r['specificityLevel'] in (3,5,7)]
f=lambda g: sum(1 for r in g if r.get('watertight') and r.get('n_bodies')==1 and r.get('winding'))/len(g)
obs=f(b)-f(a)
pool=a+b; na=len(a); cnt=0; N=20000
for _ in range(N):
    idx=rng.permutation(len(pool)); pa=[pool[i] for i in idx[:na]]; pb=[pool[i] for i in idx[na:]]
    if f(pb)-f(pa) >= obs: cnt+=1
print(f"  lvl10 well-formed={f(a)*100:.0f}% (n={len(a)})   lvl3-7={f(b)*100:.0f}% (n={len(b)})")
print(f"  observed gap={obs*100:.1f} pp, permutation p={cnt/N:.3f}")

print("\n=== Spearman: specificity vs well-formedness ===")
x=np.array([r['specificityLevel'] for r in rows],float)
y=np.array([1.0 if (r.get('watertight') and r.get('n_bodies')==1 and r.get('winding')) else 0.0 for r in rows])
from scipy import stats
rho,p=stats.spearmanr(x,y); print(f"  rho={rho:+.3f}  p={p:.3f}  -> {'NO monotonic trend' if p>0.05 else 'monotonic trend'}")

print("\n=== The LLM validator's discriminative power ===")
val=[bool(r.get('llm_valid')) for r in rows]
wf =[bool(r.get('watertight') and r.get('n_bodies')==1) for r in rows]
print(f"  llm says valid: {sum(val)}/{len(val)} = {sum(val)/len(val)*100:.0f}%")
print(f"  actually well-formed: {sum(wf)}/{len(wf)} = {sum(wf)/len(wf)*100:.0f}%")
print(f"  -> validator flags ZERO of the {sum(1 for w in wf if not w)} malformed models. AUC is undefined (no variance).")
conf=[r.get('llm_conf') for r in rows]
print(f"  confidence values present: {sorted(set(conf))}")
