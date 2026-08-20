import json, math, numpy as np
rows=json.load(open("/private/tmp/claude-501/-Users-aaroh-cadybara/3c925b39-d604-461e-b403-05f58422a2ea/scratchpad/stl_metrics.json"))
wt=[r for r in rows if r.get('watertight') and not math.isnan(r.get('bbox_fill',float('nan')))]
print("=== bbox_fill and wall-thickness proxy BY FAMILY x LEVEL ===")
print("    (snowman SHOULD be solid; planter/hook should be shelled or thin)\n")
for fam in ['wall_planter','wall_hook','snowman']:
    print(f"  {fam}:")
    for lvl in sorted({r['specificityLevel'] for r in wt}):
        g=[r for r in wt if r['family']==fam and r['specificityLevel']==lvl]
        if not g: continue
        print(f"    lvl {lvl:<3} n_wt={len(g):<3} bbox_fill={np.median([r['bbox_fill'] for r in g]):>5.2f}  2V/A={np.median([2*r['volume']/r['area'] for r in g]):>6.2f}mm")
    print()

print("=== SUCCESS RATE (watertight) vs FIDELITY (when it works) — the tradeoff ===")
print(f"{'family':<14}{'lvl':<6}{'watertight rate':<18}{'wall-thick err vs 3mm (watertight only)'}")
for fam in ['wall_planter']:
    for lvl in sorted({r['specificityLevel'] for r in rows}):
        g=[r for r in rows if r['family']==fam and r['specificityLevel']==lvl and r.get('ok')]
        gw=[r for r in g if r.get('watertight')]
        if not g: continue
        th=[2*r['volume']/r['area'] for r in gw if r['area']>0]
        e=f"{np.median([abs(t-3.0)/3*100 for t in th]):.0f}%" if th else "n/a"
        print(f"{fam:<14}{lvl:<6}{len(gw)}/{len(g)} = {len(gw)/len(g)*100:>3.0f}%        {e}")

print("\n=== Snowman check: is 'solid' correct there? ===")
sn=[r for r in wt if r['family']=='snowman']
print(f"  snowman median bbox_fill={np.median([r['bbox_fill'] for r in sn]):.2f} -> a solid stacked-sphere form, as expected. NOT a failure.")
print(f"  => bbox_fill/2V-A is only a defect signal CONDITIONED on intended hollowness. Oracle must be spec-aware, not generic.")
