import json, math, numpy as np
rows=json.load(open("/private/tmp/claude-501/-Users-aaroh-cadybara/3c925b39-d604-461e-b403-05f58422a2ea/scratchpad/stl_metrics.json"))
wt=[r for r in rows if r.get('watertight') and not math.isnan(r.get('bbox_fill',float('nan')))]

print("=== HOLLOWNESS: bbox_fill = volume / bounding-box volume ===")
print("    A hollow 3mm-wall planter should fill roughly 15-30% of its bbox. A solid blob approaches 60-100%.\n")
print(f"{'lvl':<5}{'n':<5}{'median bbox_fill':<20}{'median 2V/A (mm)':<20}{'n with fill>0.55 (blob)'}")
for lvl in sorted({r['specificityLevel'] for r in wt}):
    g=[r for r in wt if r['specificityLevel']==lvl]
    fills=[r['bbox_fill'] for r in g]; th=[2*r['volume']/r['area'] for r in g]
    print(f"{lvl:<5}{len(g):<5}{np.median(fills):<20.2f}{np.median(th):<20.2f}{sum(1 for f in fills if f>0.55)}/{len(g)}")

print("\n=== wall_planter only: spec demands hollow, 3mm walls, open top ===")
print(f"{'lvl':<6}{'id':<44}{'bbox_fill':<12}{'2V/A mm':<10}{'genus':<8}{'verdict'}")
for r in sorted([r for r in wt if r['family']=='wall_planter'],key=lambda r:(r['specificityLevel'],r['id'])):
    th=2*r['volume']/r['area']; f=r['bbox_fill']
    v = "SOLID BLOB - not hollowed" if f>0.55 or th>8 else ("hollow-ish" if th<6 else "borderline")
    print(f"{r['specificityLevel']:<6}{r['id'][:42]:<44}{f:<12.2f}{th:<10.2f}{int(r['genus']):<8}{v}")

print("\n=== The two effects are SEPARABLE ===")
pl=[r for r in rows if r['family']=='wall_planter' and r.get('ok')]
for lvl in sorted({r['specificityLevel'] for r in pl}):
    g=[r for r in pl if r['specificityLevel']==lvl]
    gw=[r for r in g if r.get('watertight') and not math.isnan(r.get('bbox_fill',float('nan')))]
    dim=np.median([abs(max(r['bx'],r['by'],r['bz'])-100.0)/100.0 for r in g])*100
    blob=sum(1 for r in gw if r['bbox_fill']>0.55 or 2*r['volume']/r['area']>8)
    print(f"  lvl {lvl:<3}  dimensional error={dim:>5.1f}%   |   solid-blob failures={blob}/{len(gw)} watertight   |   watertight={len([r for r in g if r.get('watertight')])}/{len(g)}")
