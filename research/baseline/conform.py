import json, math, numpy as np
rows=json.load(open("/private/tmp/claude-501/-Users-aaroh-cadybara/3c925b39-d604-461e-b403-05f58422a2ea/scratchpad/stl_metrics.json"))

# The wall_planter ladder states explicit targets from level 6 up:
# OD 80mm, height 90mm, back panel 100mm tall x 90mm wide -> overall bbox ~ 100(H) x 90(W) x ~84(D)
TARGET={"height":100.0,"width":90.0,"depth":84.0}
print("=== wall_planter: does stated geometry appear in the STL? (spec says 80mm OD, 90mm tall body, 100mm back panel) ===")
print(f"{'level':<7}{'id':<46}{'bbox (sorted mm)':<30}{'maxdim err vs 100mm':<20}{'watertight'}")
pl=[r for r in rows if r['family']=='wall_planter']
for r in sorted(pl,key=lambda r:(r['specificityLevel'],r['id'])):
    if not r.get('ok'): continue
    dims=sorted([r['bx'],r['by'],r['bz']],reverse=True)
    err=abs(dims[0]-100.0)/100.0*100
    print(f"{r['specificityLevel']:<7}{r['id'][:44]:<46}{'x'.join(f'{d:.1f}' for d in dims):<30}{err:>8.0f}%           {'Y' if r['watertight'] else 'N'}")

print("\n=== Dimensional conformance summary (wall_planter, |maxdim - 100mm| tolerance bands) ===")
for lvl in sorted({r['specificityLevel'] for r in pl}):
    g=[r for r in pl if r['specificityLevel']==lvl and r.get('ok')]
    errs=[abs(max(r['bx'],r['by'],r['bz'])-100.0)/100.0 for r in g]
    within10=sum(1 for e in errs if e<=0.10); within25=sum(1 for e in errs if e<=0.25)
    print(f"  lvl {lvl:<3} n={len(g):<3} median_err={np.median(errs)*100:>5.1f}%   within10%={within10}/{len(g)}   within25%={within25}/{len(g)}")

print("\n=== Scale sanity across ALL families (is anything absurdly sized?) ===")
allr=[r for r in rows if r.get('ok')]
md=[r['maxdim'] for r in allr]
print(f"  maxdim: min={min(md):.1f}mm  p25={np.percentile(md,25):.1f}  median={np.median(md):.1f}  p75={np.percentile(md,75):.1f}  max={max(md):.1f}mm")
absurd=[r for r in allr if r['maxdim']<10 or r['maxdim']>500]
print(f"  absurd scale (<10mm or >500mm): {len(absurd)}")
for r in absurd: print(f"    {r['id'][:60]} maxdim={r['maxdim']:.1f}mm lvl={r['specificityLevel']}")

print("\n=== Thin-wall / printability proxy: volume-to-area ratio (mean wall thickness proxy = 2V/A) ===")
for lvl in sorted({r['specificityLevel'] for r in allr}):
    g=[r for r in allr if r['specificityLevel']==lvl and r.get('watertight') and not math.isnan(r.get('volume',float('nan')))]
    t=[2*r['volume']/r['area'] for r in g if r['area']>0]
    if t: print(f"  lvl {lvl:<3} n={len(t):<3} median 2V/A = {np.median(t):>5.2f} mm   (spec asks 3mm walls; <1mm suggests unprintable shell, >8mm suggests solid blob)")
