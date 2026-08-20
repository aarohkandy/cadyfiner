import json, os, math, warnings
warnings.filterwarnings("ignore")
import numpy as np, trimesh

ROOT = "/Users/aaroh/cadybara/cad_grade"
items = json.load(open(f"{ROOT}/public/data/items.json"))["items"]

rows = []
for it in items:
    p = os.path.join(ROOT, "public", it["stlUrl"].lstrip("/"))
    r = {k: it.get(k) for k in ("id","family","seedId","specificityLevel","repetition","experimentId","latencyMs")}
    r["llm_valid"] = it["validation"].get("valid")
    r["llm_conf"] = it["validation"].get("confidence")
    r["attempts"] = it["validation"].get("attempt_count")
    r["n_issues"] = len(it["validation"].get("issues") or [])
    try:
        m = trimesh.load(p, force="mesh")
        r["ok"] = True
        r["bytes"] = os.path.getsize(p)
        r["faces"] = len(m.faces); r["verts"] = len(m.vertices)
        r["watertight"] = bool(m.is_watertight)
        r["winding"] = bool(m.is_winding_consistent)
        r["volume"] = float(m.volume) if m.is_watertight else float("nan")
        r["area"] = float(m.area)
        ext = m.extents
        r["bx"],r["by"],r["bz"] = [float(v) for v in ext]
        r["maxdim"] = float(max(ext)); r["mindim"] = float(min(ext))
        r["n_bodies"] = int(m.body_count)
        r["euler"] = int(m.euler_number)
        # genus only meaningful if watertight & single body
        r["genus"] = (2 - m.euler_number)/2 if m.is_watertight else float("nan")
        cv = m.convex_hull.volume
        r["convexity"] = float(m.volume/cv) if (m.is_watertight and cv>0) else float("nan")
        r["degenerate"] = int((m.area_faces < 1e-12).sum())
        r["bbox_fill"] = float(m.volume/(ext[0]*ext[1]*ext[2])) if (m.is_watertight and np.all(ext>0)) else float("nan")
    except Exception as e:
        r["ok"] = False; r["err"] = str(e)[:120]
    rows.append(r)

json.dump(rows, open("/private/tmp/claude-501/-Users-aaroh-cadybara/3c925b39-d604-461e-b403-05f58422a2ea/scratchpad/stl_metrics.json","w"), indent=1)

n = len(rows); okr=[r for r in rows if r.get("ok")]
print(f"parsed {len(okr)}/{n}")
print("\n=== BY SPECIFICITY LEVEL ===")
print(f"{'lvl':<5}{'n':<5}{'watertight':<12}{'1body':<8}{'genus0':<8}{'llm_valid':<11}{'medvol_cm3':<12}{'med_maxdim':<11}{'med_attempts'}")
for lvl in sorted({r['specificityLevel'] for r in okr}):
    g=[r for r in okr if r['specificityLevel']==lvl]
    wt=[r for r in g if r['watertight']]
    vols=[r['volume']/1000 for r in wt if not math.isnan(r['volume'])]
    print(f"{lvl:<5}{len(g):<5}{len(wt)/len(g)*100:>5.0f}%      {sum(r['n_bodies']==1 for r in g)/len(g)*100:>4.0f}%   "
          f"{sum(1 for r in wt if r['genus']==0)/max(len(wt),1)*100:>4.0f}%   {sum(bool(r['llm_valid']) for r in g)/len(g)*100:>5.0f}%    "
          f"{np.median(vols) if vols else float('nan'):>9.1f}   {np.median([r['maxdim'] for r in g]):>8.1f}   {np.median([r['attempts'] for r in g]):>6.1f}")

print("\n=== BY FAMILY ===")
for fam in sorted({r['family'] for r in okr}):
    g=[r for r in okr if r['family']==fam]
    wt=[r for r in g if r['watertight']]
    print(f"{fam:<14} n={len(g):<4} watertight={len(wt)/len(g)*100:>3.0f}%  1body={sum(r['n_bodies']==1 for r in g)/len(g)*100:>3.0f}%  med_maxdim={np.median([r['maxdim'] for r in g]):>7.1f}mm")

print("\n=== HOLE / TOPOLOGY SIGNAL (genus>0 means through-holes exist) ===")
for lvl in sorted({r['specificityLevel'] for r in okr}):
    g=[r for r in okr if r['specificityLevel']==lvl and r['watertight']]
    if not g: continue
    gen=[r['genus'] for r in g if not math.isnan(r['genus'])]
    print(f"  lvl {lvl:<3} n_wt={len(g):<4} genus dist={dict(sorted((int(x), gen.count(x)) for x in set(gen)))}")
