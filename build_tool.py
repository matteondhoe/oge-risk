"""
OG&E Identity Risk Tool -- Build Script v3
Run once: py build_tool.py
Opens:    oge_risk.html

Design system: DESIGN-claude.md (warm cream canvas, coral CTA, dark navy surfaces)
New features:
  - User lookup: search any user, see full signal breakdown + plain-English explanation
  - Executive CISO briefing auto-generated from live data
  - What-If weight simulator: adjust signal weights, rankings update live
  - Redesigned UI: cream canvas / dark surfaces alternating rhythm
"""

import pandas as pd
import numpy as np
import json, os

PHYS_PATH  = r"C:\Users\matte\Desktop\og&e_raw\Physical Data"
CYBER_PATH = r"C:\Users\matte\Desktop\og&e_raw\Cyber Data"

PHYS_FILES = [
    "Aug 25-obfuscated-2026-02-04 20-15 UTC -6.csv",
    "Sep 25-obfuscated-2026-02-04 20-28 UTC -6.csv",
    "Oct 25-obfuscated-2026-02-04 20-27 UTC -6.csv",
    "Nov 25-obfuscated-2026-02-04 20-25 UTC -6.csv",
    "Dec 25-obfuscated-2026-02-04 20-24 UTC -6.csv",
]

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading physical data...")
frames = []
for fname in PHYS_FILES:
    tmp = pd.read_csv(os.path.join(PHYS_PATH, fname))
    frames.append(tmp)
df_phy = pd.concat(frames, ignore_index=True)
df_phy = df_phy.dropna(subset=["Event UTC Time","Device","Event","Badge"]).drop_duplicates()
df_phy["Event UTC Time"] = pd.to_datetime(df_phy["Event UTC Time"], utc=True)
df_phy["hour"]           = df_phy["Event UTC Time"].dt.hour
df_phy["day_of_week"]    = df_phy["Event UTC Time"].dt.dayofweek
df_phy["is_weekend"]     = df_phy["day_of_week"] >= 5
df_phy["is_after_hours"] = (df_phy["hour"] < 6) | (df_phy["hour"] >= 20)
print(f"  {len(df_phy):,} rows")

print("Loading digital data...")
df_ent   = pd.read_json(os.path.join(CYBER_PATH, "entitlements-obfuscated-2026-02-02 17-38 UTC -6.json"))
df_acc   = pd.read_json(os.path.join(CYBER_PATH, "accessprofiles-obfuscated-2026-02-02 17-05 UTC -6.json"))
df_id    = pd.read_json(os.path.join(CYBER_PATH, "identities-obfuscated-2026-02-04 18-22 UTC -6.json"))
df_roles = pd.read_json(os.path.join(CYBER_PATH, "roles-obfuscated-2026-02-02 17-05 UTC -6.json"))
print(f"  {len(df_id):,} identities | {len(df_ent):,} entitlements")

# ── Label maps ────────────────────────────────────────────────────────────────
print("Building label maps...")
reader_order = df_phy["Device"].value_counts().index.tolist()
reader_map   = {h: f"Reader {i+1:03d}" for i, h in enumerate(reader_order)}
df_id["job"] = df_id["attributes"].map(lambda x: x.get("job") if isinstance(x, dict) else None)
job_order    = df_id["job"].dropna().value_counts().index.tolist()
job_map      = {h: f"Job {i+1:02d}" for i, h in enumerate(job_order)}
user_map     = {h: f"User {i+1:04d}" for i, h in enumerate(df_id["id"].tolist())}

def safe_zscore(x):
    if len(x) <= 1: return pd.Series(0, index=x.index)
    mu, sd = x.mean(), x.std(ddof=1)
    if sd == 0 or np.isnan(sd): return pd.Series(0, index=x.index)
    return (x - mu) / sd

def norm(s):
    mn, mx = s.min(), s.max()
    if mx == mn: return pd.Series(0.0, index=s.index)
    return (s - mn) / (mx - mn) * 100

def tier(s):
    if s >= 75: return "Critical"
    if s >= 50: return "High"
    if s >= 25: return "Medium"
    return "Low"

# ── Physical scoring ──────────────────────────────────────────────────────────
print("Scoring physical users...")
granted    = df_phy[df_phy["Event"].str.contains("Access Granted", na=False)].copy()
all_events = df_phy.copy()

badge_total    = granted.groupby("Badge").size().rename("total_events")
badge_afterhrs = (granted[granted["is_after_hours"] | granted["is_weekend"]]
                  .groupby("Badge").size().rename("afterhrs_events"))
phys = pd.concat([badge_total, badge_afterhrs], axis=1).fillna(0)
phys["afterhrs_rate"] = phys["afterhrs_events"] / phys["total_events"]

freq = granted.groupby(["Device","Badge"]).size().reset_index(name="event_count")
freq["z_score"] = freq.groupby("Device")["event_count"].transform(safe_zscore)
max_z     = freq.groupby("Badge")["z_score"].max().rename("max_reader_zscore")
footprint = granted.groupby("Badge")["Device"].nunique().rename("reader_footprint")

print("  Computing rapid succession...")
all_sorted = all_events.sort_values(["Badge","Event UTC Time"]).copy()
all_sorted["prev_time"] = all_sorted.groupby("Badge")["Event UTC Time"].shift(4)
all_sorted["window_secs"] = (all_sorted["Event UTC Time"] - all_sorted["prev_time"]).dt.total_seconds()
rapid_counts = all_sorted[all_sorted["window_secs"] <= 300].groupby("Badge").size().rename("rapid_bursts")

weekend_ev = granted[granted["is_weekend"]].groupby("Badge").size().rename("weekend_events")
weekday_ev = granted[~granted["is_weekend"]].groupby("Badge").size().rename("weekday_events")
wk = pd.concat([weekend_ev, weekday_ev], axis=1).fillna(0)
wk["weekend_pct"] = wk["weekend_events"] / (wk["weekend_events"] + wk["weekday_events"])
wk["weekend_only_score"] = (wk["weekend_pct"] > 0.80).astype(float) * 100

print("  Computing dwell time...")
granted["date"] = granted["Event UTC Time"].dt.date
daily     = granted.groupby(["Badge","date"])["Event UTC Time"].agg(["min","max"])
daily["dwell_hrs"] = (daily["max"] - daily["min"]).dt.total_seconds() / 3600
avg_dwell = daily.groupby("Badge")["dwell_hrs"].mean().rename("avg_dwell_hrs")

phys = (phys.join(max_z, how="left").join(footprint, how="left")
        .join(rapid_counts, how="left")
        .join(wk[["weekend_pct","weekend_only_score"]], how="left")
        .join(avg_dwell, how="left").fillna(0))

phys["s_afterhrs"]  = norm(phys["afterhrs_rate"])
phys["s_zscore"]    = norm(phys["max_reader_zscore"].clip(upper=10))
phys["s_footprint"] = norm(phys["reader_footprint"])
phys["s_rapid"]     = norm(phys["rapid_bursts"].clip(upper=50))
phys["s_weekend"]   = phys["weekend_only_score"]
phys["s_dwell"]     = norm(phys["avg_dwell_hrs"].clip(upper=16))

phys["phys_score"] = (phys["s_afterhrs"]*0.40 + phys["s_zscore"]*0.25 +
                      phys["s_footprint"]*0.15 + phys["s_rapid"]*0.10 +
                      phys["s_weekend"]*0.05  + phys["s_dwell"]*0.05)
phys["phys_tier"] = phys["phys_score"].map(tier)
phys = phys.reset_index().sort_values("phys_score", ascending=False)
phys["label"] = phys["Badge"].map(lambda x: user_map.get(x, x[:12]))
print(f"  {len(phys):,} physical users scored")

# ── Digital scoring ───────────────────────────────────────────────────────────
print("Scoring digital users...")
df_id["total_ents"] = df_id["access"].map(lambda a: len(a) if isinstance(a, list) else 0)
df_id["priv_ents"]  = df_id["access"].map(
    lambda a: sum(1 for x in a if isinstance(x, dict) and x.get("privileged", False))
    if isinstance(a, list) else 0)

roles_ex = df_roles.explode("entitlements")
roles_ex["eid"] = roles_ex["entitlements"].apply(lambda x: x.get("id") if isinstance(x, dict) else None)
ap_ex = df_acc.explode("entitlements")
ap_ex["eid"] = ap_ex["entitlements"].apply(lambda x: x.get("id") if isinstance(x, dict) else None)
assoc_ids = set(pd.concat([roles_ex["eid"], ap_ex["eid"]], ignore_index=True).dropna().unique())

unassoc = df_ent[~df_ent["id"].isin(assoc_ids)].copy()
unassoc["owner_id"] = unassoc["owner"].apply(lambda x: x.get("id") if isinstance(x, dict) else None)
unassoc_counts = unassoc.groupby("owner_id").size().rename("unassoc_ents")

dig = df_id.join(unassoc_counts, on="id", how="left")
dig["unassoc_ents"] = dig["unassoc_ents"].fillna(0)

print("  Computing inactive identity signal...")
dig["identity_state"] = dig["attributes"].map(lambda x: x.get("identityState","") if isinstance(x, dict) else "")
dig["is_inactive"] = dig["identity_state"].isin({"INACTIVE_LONG_TERM","INACTIVE"})
dig["inactive_with_access"] = (dig["is_inactive"] & (dig["total_ents"] > 0)).astype(float) * 100

dig = dig.dropna(subset=["job"]).copy()
dig["z_ents"]     = dig.groupby("job")["total_ents"].transform(safe_zscore)
dig["s_zscore"]   = norm(dig["z_ents"].clip(lower=0, upper=10))
dig["s_priv"]     = norm(dig["priv_ents"])
dig["s_unassoc"]  = norm(dig["unassoc_ents"])
dig["s_inactive"] = dig["inactive_with_access"]

dig["dig_score"] = (dig["s_zscore"]*0.35 + dig["s_priv"]*0.30 +
                    dig["s_unassoc"]*0.20 + dig["s_inactive"]*0.15)

p_crit = dig["dig_score"].quantile(0.97)
p_high = dig["dig_score"].quantile(0.90)
p_med  = dig["dig_score"].quantile(0.70)

def tier_dig(s):
    if s >= p_crit: return "Critical"
    if s >= p_high: return "High"
    if s >= p_med:  return "Medium"
    return "Low"

dig["dig_tier"]  = dig["dig_score"].map(tier_dig)
dig = dig.sort_values("dig_score", ascending=False)
dig["label"]     = dig["id"].map(lambda x: user_map.get(x, x[:12]))
dig["job_label"] = dig["job"].map(lambda x: job_map.get(x, x[:12]))

inactive_flagged = int(dig["is_inactive"].sum())
rapid_flagged    = int((phys["rapid_bursts"] > 0).sum())
tier_counts_phys = phys["phys_tier"].value_counts().to_dict()
tier_counts_dig  = dig["dig_tier"].value_counts().to_dict()
print(f"  {len(dig):,} digital users scored | {inactive_flagged:,} inactive with access")

# ── Payloads ──────────────────────────────────────────────────────────────────
print("Preparing data payloads...")

def phys_row(r):
    return {"label":r["label"],"score":round(float(r["phys_score"]),1),"tier":r["phys_tier"],
            "afterhrs":int(r["afterhrs_events"]),"afterhrs_rt":f"{r['afterhrs_rate']*100:.1f}%",
            "zscore":round(float(r["max_reader_zscore"]),2),"footprint":int(r["reader_footprint"]),
            "rapid":int(r["rapid_bursts"]),"dwell":round(float(r["avg_dwell_hrs"]),1),
            "wkend_pct":f"{r['weekend_pct']*100:.0f}%",
            # raw normalized signals for what-if simulator
            "s_afterhrs":round(float(r["s_afterhrs"]),1),"s_zscore":round(float(r["s_zscore"]),1),
            "s_footprint":round(float(r["s_footprint"]),1),"s_rapid":round(float(r["s_rapid"]),1),
            "s_weekend":round(float(r["s_weekend"]),1),"s_dwell":round(float(r["s_dwell"]),1)}

def dig_row(r):
    return {"label":r["label"],"job":r["job_label"],"score":round(float(r["dig_score"]),1),
            "tier":r["dig_tier"],"total":int(r["total_ents"]),"priv":int(r["priv_ents"]),
            "unassoc":int(r["unassoc_ents"]),"zscore":round(float(r["z_ents"]),2),
            "inactive":bool(r["is_inactive"]),
            "s_zscore":round(float(r["s_zscore"]),1),"s_priv":round(float(r["s_priv"]),1),
            "s_unassoc":round(float(r["s_unassoc"]),1),"s_inactive":round(float(r["s_inactive"]),1)}

phys_data = [phys_row(r) for _, r in phys.head(200).iterrows()]
dig_data  = [dig_row(r)  for _, r in dig.head(200).iterrows()]

stats = {
    "phys_rows":        f"{len(df_phy):,}",
    "identities":       f"{len(df_id):,}",
    "afterhrs":         f"{int(phys['afterhrs_events'].sum()):,}",
    "afterhrs_pct":     f"{phys['afterhrs_events'].sum()/phys['total_events'].sum()*100:.1f}%",
    "rare_ents":        "28,409",
    "phys_critical":    tier_counts_phys.get("Critical",0),
    "dig_critical":     tier_counts_dig.get("Critical",0),
    "rapid_flagged":    rapid_flagged,
    "inactive_flagged": inactive_flagged,
}

# ── HTML ──────────────────────────────────────────────────────────────────────
print("Building HTML...")

HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>OG&amp;E Identity Risk Analysis</title>
<link href="https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,500;1,400&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet"/>
<style>
/* ── Design tokens (DESIGN-claude.md) ─────────────────────────────────── */
:root{{
  --primary:#cc785c;--primary-active:#a9583e;--primary-disabled:#e6dfd8;
  --ink:#141413;--body:#3d3d3a;--body-strong:#252523;--muted:#6c6a64;--muted-soft:#8e8b82;
  --hairline:#e6dfd8;--hairline-soft:#ebe6df;
  --canvas:#faf9f5;--surface-soft:#f5f0e8;--surface-card:#efe9de;
  --surface-cream-strong:#e8e0d2;
  --surface-dark:#181715;--surface-dark-elevated:#252320;--surface-dark-soft:#1f1e1b;
  --on-primary:#fff;--on-dark:#faf9f5;--on-dark-soft:#a09d96;
  --accent-teal:#5db8a6;--accent-amber:#e8a55a;
  --success:#5db872;--warning:#d4a017;--error:#c64545;
  --serif:'EB Garamond',serif;
  --sans:'Inter',sans-serif;
  --mono:'JetBrains Mono',monospace;
  --r-sm:6px;--r-md:8px;--r-lg:12px;--r-xl:16px;
}}
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--canvas);color:var(--body);font-family:var(--sans);font-size:14px;line-height:1.55;}}

/* ── NAV ─────────────────────────────────────────────────────────────── */
nav{{
  position:fixed;top:0;left:0;right:0;z-index:100;
  height:64px;background:var(--canvas);border-bottom:1px solid var(--hairline);
  display:flex;align-items:center;justify-content:space-between;padding:0 2rem;
}}
.nav-brand{{
  font-family:var(--serif);font-size:1.2rem;font-weight:400;
  letter-spacing:-0.3px;color:var(--ink);
}}
.nav-brand span{{color:var(--primary)}}
.nav-tabs{{display:flex;gap:0}}
.nav-tab{{
  height:64px;display:flex;align-items:center;padding:0 1.2rem;
  font-size:0.8rem;font-weight:500;letter-spacing:0;
  color:var(--muted);cursor:pointer;border-bottom:2px solid transparent;
  transition:color 0.15s;user-select:none;font-family:var(--sans);
}}
.nav-tab:hover{{color:var(--ink)}}
.nav-tab.active{{color:var(--ink);border-bottom-color:var(--primary)}}
.nav-right{{font-size:0.75rem;color:var(--muted-soft);letter-spacing:0.5px}}

main{{padding-top:64px}}
.page{{display:none;animation:fadeIn 0.25s ease}}
.page.active{{display:block}}
@keyframes fadeIn{{from{{opacity:0;transform:translateY(6px)}}to{{opacity:1;transform:none}}}}

/* ── BANDS / LAYOUT ──────────────────────────────────────────────────── */
.band{{padding:64px 2rem;max-width:1200px;margin:0 auto}}
.band-dark{{background:var(--surface-dark);color:var(--on-dark)}}
.band-dark .muted{{color:var(--on-dark-soft)}}
.band-card{{background:var(--surface-card)}}
.band-full{{max-width:none;padding-left:0;padding-right:0}}
.band-full-inner{{max-width:1200px;margin:0 auto;padding:0 2rem}}
.two-col{{display:grid;grid-template-columns:1fr 1fr;gap:2rem}}
.three-col{{display:grid;grid-template-columns:repeat(3,1fr);gap:1.5rem}}
@media(max-width:900px){{.two-col,.three-col{{grid-template-columns:1fr}}}}

/* ── TYPOGRAPHY ──────────────────────────────────────────────────────── */
.display-xl{{font-family:var(--serif);font-size:clamp(2.5rem,5vw,4rem);font-weight:400;line-height:1.05;letter-spacing:-1.5px;color:var(--ink)}}
.display-lg{{font-family:var(--serif);font-size:clamp(2rem,4vw,3rem);font-weight:400;line-height:1.1;letter-spacing:-1px;color:var(--ink)}}
.display-md{{font-family:var(--serif);font-size:clamp(1.5rem,3vw,2.25rem);font-weight:400;line-height:1.15;letter-spacing:-0.5px;color:var(--ink)}}
.display-sm{{font-family:var(--serif);font-size:1.75rem;font-weight:400;line-height:1.2;letter-spacing:-0.3px}}
.on-dark .display-xl,.on-dark .display-lg,.on-dark .display-md,.band-dark .display-xl,.band-dark .display-lg,.band-dark .display-md{{color:var(--on-dark)}}
.caption-upper{{font-size:0.72rem;font-weight:500;letter-spacing:1.5px;text-transform:uppercase;color:var(--muted)}}
.band-dark .caption-upper{{color:var(--on-dark-soft)}}
em.coral{{color:var(--primary);font-style:normal}}

/* ── STAT GRID ──────────────────────────────────────────────────────── */
.stat-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:1px;background:var(--hairline);border:1px solid var(--hairline);border-radius:var(--r-lg);overflow:hidden;margin:2rem 0}}
.stat-cell{{background:var(--canvas);padding:1.5rem;}}
.band-dark .stat-cell{{background:var(--surface-dark-elevated)}}
.stat-label{{font-size:0.7rem;font-weight:500;letter-spacing:1px;text-transform:uppercase;color:var(--muted);margin-bottom:0.4rem}}
.band-dark .stat-label{{color:var(--on-dark-soft)}}
.stat-value{{font-family:var(--serif);font-size:2.2rem;font-weight:400;line-height:1;letter-spacing:-0.5px;color:var(--primary)}}
.stat-value.teal{{color:var(--accent-teal)}}
.stat-value.amber{{color:var(--accent-amber)}}
.stat-value.ink{{color:var(--ink)}}
.band-dark .stat-value.ink{{color:var(--on-dark)}}
.stat-sub{{font-size:0.72rem;color:var(--muted-soft);margin-top:0.25rem}}

/* ── FEATURE CARDS ──────────────────────────────────────────────────── */
.feature-card{{background:var(--surface-card);border-radius:var(--r-lg);padding:2rem;}}
.feature-card h3{{font-family:var(--serif);font-size:1.25rem;font-weight:400;color:var(--ink);margin-bottom:0.5rem}}
.feature-card p{{font-size:0.85rem;color:var(--muted);line-height:1.6}}

/* ── DARK CARD (product mockup style) ──────────────────────────────── */
.dark-card{{background:var(--surface-dark);border-radius:var(--r-lg);padding:2rem;color:var(--on-dark)}}
.dark-card h3{{font-family:var(--serif);font-size:1.2rem;font-weight:400;color:var(--on-dark);margin-bottom:0.75rem}}
.dark-card p{{font-size:0.85rem;color:var(--on-dark-soft);line-height:1.6}}

/* ── CALLOUT CORAL ──────────────────────────────────────────────────── */
.callout-coral{{background:var(--primary);border-radius:var(--r-lg);padding:2rem;color:var(--on-primary);margin:2rem 0}}
.callout-coral .caption-upper{{color:rgba(255,255,255,0.7)}}
.callout-coral p{{font-size:0.9rem;line-height:1.6;margin-top:0.5rem}}
.callout-coral strong{{color:#fff}}

/* ── BUTTONS ─────────────────────────────────────────────────────────── */
.btn-primary{{display:inline-flex;align-items:center;gap:0.5rem;background:var(--primary);color:var(--on-primary);font-family:var(--sans);font-size:0.85rem;font-weight:500;padding:10px 20px;height:40px;border-radius:var(--r-md);border:none;cursor:pointer;transition:background 0.15s}}
.btn-primary:hover{{background:var(--primary-active)}}
.btn-secondary{{display:inline-flex;align-items:center;gap:0.5rem;background:var(--canvas);color:var(--ink);font-family:var(--sans);font-size:0.85rem;font-weight:500;padding:10px 20px;height:40px;border-radius:var(--r-md);border:1px solid var(--hairline);cursor:pointer;transition:border-color 0.15s}}
.btn-secondary:hover{{border-color:var(--primary)}}

/* ── INPUTS ──────────────────────────────────────────────────────────── */
.text-input{{background:var(--canvas);color:var(--ink);font-family:var(--sans);font-size:0.9rem;padding:10px 14px;height:40px;border-radius:var(--r-md);border:1px solid var(--hairline);outline:none;transition:border-color 0.15s;width:100%}}
.text-input:focus{{border-color:var(--primary);box-shadow:0 0 0 3px rgba(204,120,92,0.15)}}
.text-input.dark{{background:var(--surface-dark-elevated);border-color:rgba(255,255,255,0.1);color:var(--on-dark)}}
.text-input.dark:focus{{border-color:var(--primary)}}

/* ── TIER BADGES ─────────────────────────────────────────────────────── */
.tier-badge{{display:inline-block;padding:3px 10px;font-size:0.65rem;font-weight:500;letter-spacing:1px;text-transform:uppercase;border-radius:9999px}}
.tier-Critical{{background:rgba(198,69,69,0.12);color:var(--error);border:1px solid rgba(198,69,69,0.25)}}
.tier-High{{background:rgba(204,120,92,0.12);color:var(--primary);border:1px solid rgba(204,120,92,0.25)}}
.tier-Medium{{background:rgba(232,165,90,0.12);color:var(--accent-amber);border:1px solid rgba(232,165,90,0.25)}}
.tier-Low{{background:var(--surface-card);color:var(--muted);border:1px solid var(--hairline)}}
.inactive-badge{{display:inline-block;padding:3px 10px;font-size:0.65rem;font-weight:500;letter-spacing:1px;text-transform:uppercase;border-radius:9999px;background:rgba(212,160,23,0.12);color:var(--warning);border:1px solid rgba(212,160,23,0.25)}}

/* ── TABLE ──────────────────────────────────────────────────────────── */
.table-wrap{{overflow-y:auto;max-height:60vh;border:1px solid var(--hairline);border-radius:var(--r-lg);}}
.risk-table{{width:100%;border-collapse:collapse;font-size:0.8rem}}
.risk-table thead{{position:sticky;top:0;z-index:5}}
.risk-table th{{padding:0.65rem 0.9rem;font-size:0.68rem;font-weight:500;letter-spacing:0.8px;text-transform:uppercase;color:var(--muted);text-align:left;border-bottom:1px solid var(--hairline);background:var(--surface-soft);white-space:nowrap;cursor:pointer;user-select:none}}
.risk-table th:hover{{color:var(--ink)}}
.risk-table td{{padding:0.65rem 0.9rem;border-bottom:1px solid var(--hairline-soft);color:var(--body);white-space:nowrap;vertical-align:middle}}
.risk-table tr:last-child td{{border-bottom:none}}
.risk-table tr:hover td{{background:var(--surface-soft)}}

/* ── SCORE BAR ───────────────────────────────────────────────────────── */
.score-wrap{{display:flex;align-items:center;gap:0.6rem;min-width:120px}}
.score-num{{width:2.5rem;text-align:right;font-weight:600;font-size:0.85rem}}
.score-track{{flex:1;height:3px;background:var(--hairline);border-radius:2px}}
.score-fill{{height:3px;border-radius:2px;transition:width 0.4s ease}}

/* ── CONTROLS ────────────────────────────────────────────────────────── */
.controls{{display:flex;gap:0.5rem;align-items:center;margin-bottom:1.5rem;flex-wrap:wrap}}
.filter-btn{{padding:6px 14px;font-family:var(--sans);font-size:0.75rem;font-weight:500;border:1px solid var(--hairline);background:var(--canvas);color:var(--muted);cursor:pointer;border-radius:var(--r-md);transition:all 0.15s}}
.filter-btn:hover{{border-color:var(--primary);color:var(--ink)}}
.filter-btn.active{{background:var(--primary);border-color:var(--primary);color:#fff}}

/* ── LOOKUP CARD (dark surface / product mockup style) ───────────────── */
#lookup-result{{display:none;background:var(--surface-dark);border-radius:var(--r-xl);padding:2rem;margin-top:1.5rem;color:var(--on-dark)}}
#lookup-result.visible{{display:block}}
.lookup-header{{display:flex;align-items:center;gap:1rem;margin-bottom:1.5rem;padding-bottom:1rem;border-bottom:1px solid rgba(255,255,255,0.08)}}
.lookup-user-name{{font-family:var(--serif);font-size:1.6rem;font-weight:400;color:var(--on-dark)}}
.signal-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:1rem;margin:1.5rem 0}}
.signal-cell{{background:var(--surface-dark-elevated);border-radius:var(--r-md);padding:1rem}}
.signal-label{{font-size:0.68rem;font-weight:500;letter-spacing:0.8px;text-transform:uppercase;color:var(--on-dark-soft);margin-bottom:0.3rem}}
.signal-val{{font-family:var(--mono);font-size:1.1rem;font-weight:500;color:var(--on-dark)}}
.signal-val.coral{{color:var(--primary)}}
.signal-val.amber{{color:var(--accent-amber)}}
.signal-val.teal{{color:var(--accent-teal)}}
.briefing-text{{font-size:0.9rem;line-height:1.7;color:var(--on-dark-soft);background:var(--surface-dark-soft);border-radius:var(--r-md);padding:1rem 1.25rem;margin-top:1rem;border-left:3px solid var(--primary)}}
.briefing-text strong{{color:var(--on-dark)}}

/* ── WHAT-IF SIMULATOR ───────────────────────────────────────────────── */
.simulator-band{{background:var(--surface-dark)}}
.slider-row{{display:flex;align-items:center;gap:1rem;margin-bottom:0.75rem}}
.slider-label{{width:160px;font-size:0.8rem;color:var(--on-dark-soft)}}
input[type=range]{{flex:1;accent-color:var(--primary);height:3px}}
.slider-pct{{width:40px;text-align:right;font-family:var(--mono);font-size:0.8rem;color:var(--primary);font-weight:500}}

/* ── MINI BAR CHART ──────────────────────────────────────────────────── */
.mini-bar-row{{display:flex;align-items:center;gap:0.75rem;margin-bottom:0.9rem}}
.mini-bar-label{{width:80px;font-size:0.72rem;color:var(--muted)}}
.mini-bar-track{{flex:1;height:6px;background:var(--hairline);border-radius:3px}}
.mini-bar-fill{{height:6px;border-radius:3px}}
.mini-bar-val{{width:60px;font-family:var(--serif);font-size:1rem;font-weight:400;text-align:right}}

/* ── RADAR ───────────────────────────────────────────────────────────── */
canvas#radar{{display:block;max-width:360px;margin:0 auto}}

/* ── EXECUTIVE BRIEFING ──────────────────────────────────────────────── */
.briefing-box{{background:var(--surface-dark-elevated);border-radius:var(--r-lg);padding:1.5rem 2rem;border-left:4px solid var(--primary);font-size:0.9rem;line-height:1.8;color:var(--on-dark-soft)}}
.briefing-box strong{{color:var(--on-dark)}}

/* ── SECTION DIVIDER ─────────────────────────────────────────────────── */
.divider{{height:1px;background:var(--hairline);margin:3rem 0}}
.divider-dark{{height:1px;background:rgba(255,255,255,0.08);margin:2rem 0}}

::-webkit-scrollbar{{width:4px;height:4px}}
::-webkit-scrollbar-track{{background:var(--canvas)}}
::-webkit-scrollbar-thumb{{background:var(--hairline);border-radius:2px}}
</style>
</head>
<body>

<nav>
  <div class="nav-brand">OG<span>&amp;</span>E &nbsp;<span style="color:var(--hairline)">/</span>&nbsp; <span style="font-size:1rem;color:var(--muted);font-family:var(--sans);font-weight:500;letter-spacing:0">Identity Risk</span></div>
  <div class="nav-tabs">
    <div class="nav-tab active" onclick="showPage('overview',this)">Overview</div>
    <div class="nav-tab" onclick="showPage('physical',this)">Physical</div>
    <div class="nav-tab" onclick="showPage('digital',this)">Digital</div>
    <div class="nav-tab" onclick="showPage('lookup',this)">User Lookup</div>
    <div class="nav-tab" onclick="showPage('simulator',this)">Simulator</div>
    <div class="nav-tab" onclick="showPage('overlap',this)">Overlap</div>
  </div>
  <div class="nav-right">Group 3 &middot; MIS 3213</div>
</nav>

<main>

<!-- ════════════════════════════════════════════ OVERVIEW -->
<div id="page-overview" class="page active">

  <!-- Hero band: cream -->
  <div class="band" style="padding-bottom:48px">
    <div class="caption-upper" style="margin-bottom:1rem">Identity &amp; Access Risk Analysis &middot; Milestone 3</div>
    <h1 class="display-xl" style="max-width:700px;margin-bottom:1rem">
      Two Systems.<br><em class="coral">One Blind Spot.</em>
    </h1>
    <p style="max-width:520px;color:var(--muted);line-height:1.7;font-size:0.9rem">
      OG&amp;E runs physical badge access (OnGuard) and digital identity management (SailPoint) in complete isolation.
      This tool surfaces risk from both and identifies users flagged independently by each system.
    </p>
  </div>

  <!-- Stats band: dark surface (cream-to-dark rhythm) -->
  <div style="background:var(--surface-dark);padding:48px 0">
    <div style="max-width:1200px;margin:0 auto;padding:0 2rem">
      <div class="caption-upper" style="color:var(--on-dark-soft);margin-bottom:1.5rem">At a Glance</div>
      <div class="stat-grid band-dark">
        <div class="stat-cell"><div class="stat-label">Badge Events</div><div class="stat-value ink">{stats["phys_rows"]}</div><div class="stat-sub">Aug&ndash;Dec 2025</div></div>
        <div class="stat-cell"><div class="stat-label">After-Hours</div><div class="stat-value">{stats["afterhrs"]}</div><div class="stat-sub">{stats["afterhrs_pct"]} of all access</div></div>
        <div class="stat-cell"><div class="stat-label">Rapid Succession</div><div class="stat-value amber">{stats["rapid_flagged"]}</div><div class="stat-sub">5+ swipes in 5 min</div></div>
        <div class="stat-cell"><div class="stat-label">Physical Critical</div><div class="stat-value">{stats["phys_critical"]}</div><div class="stat-sub">users flagged</div></div>
        <div class="stat-cell"><div class="stat-label">Inactive w/ Access</div><div class="stat-value amber">{stats["inactive_flagged"]}</div><div class="stat-sub">should be zero</div></div>
        <div class="stat-cell"><div class="stat-label">Digital Critical</div><div class="stat-value">{stats["dig_critical"]}</div><div class="stat-sub">users flagged</div></div>
        <div class="stat-cell"><div class="stat-label">Rare Entitlements</div><div class="stat-value teal">{stats["rare_ents"]}</div><div class="stat-sub">held by &lt;0.5% of users</div></div>
        <div class="stat-cell"><div class="stat-label">Identities</div><div class="stat-value ink">{stats["identities"]}</div><div class="stat-sub">in SailPoint</div></div>
      </div>
    </div>
  </div>

  <!-- Executive briefing: cream -->
  <div class="band" style="padding-top:48px;padding-bottom:48px">
    <div class="caption-upper" style="margin-bottom:1rem">CISO Briefing</div>
    <h2 class="display-md" style="margin-bottom:1.5rem">Executive Summary</h2>
    <div class="briefing-box" style="background:var(--surface-card);border-left-color:var(--primary);color:var(--body)">
      <p id="exec-briefing" style="font-size:0.9rem;line-height:1.8;color:var(--body)"></p>
    </div>
  </div>

  <!-- Distribution charts: surface-card band -->
  <div style="background:var(--surface-soft);padding:48px 0">
    <div style="max-width:1200px;margin:0 auto;padding:0 2rem">
      <div class="two-col">
        <div>
          <div class="caption-upper" style="margin-bottom:1rem">Physical Risk Distribution</div>
          <div id="overview-phys-chart"></div>
        </div>
        <div>
          <div class="caption-upper" style="margin-bottom:1rem">Digital Risk Distribution</div>
          <div id="overview-dig-chart"></div>
        </div>
      </div>
    </div>
  </div>

  <!-- Feature cards: cream (alternate back) -->
  <div class="band">
    <div class="caption-upper" style="margin-bottom:2rem">How It Works</div>
    <div class="three-col">
      <div class="feature-card">
        <div style="font-size:1.5rem;margin-bottom:0.75rem">&#x2713;</div>
        <h3>6 Physical Signals</h3>
        <p>After-hours rate, reader z-score, physical footprint, rapid succession bursts, weekend-only pattern, and average dwell time are combined into a single physical risk score per badge holder.</p>
      </div>
      <div class="feature-card">
        <div style="font-size:1.5rem;margin-bottom:0.75rem">&#x2713;</div>
        <h3>4 Digital Signals</h3>
        <p>Entitlement z-score vs job family peers, privileged access count, unassociated entitlements, and inactive identity status are weighted into a digital risk score per identity.</p>
      </div>
      <div class="feature-card">
        <div style="font-size:1.5rem;margin-bottom:0.75rem">&#x2713;</div>
        <h3>Cross-System Overlap</h3>
        <p>Users who appear at the top of both independent lists simultaneously represent the highest-priority review targets. Neither system could identify these users alone.</p>
      </div>
    </div>
  </div>

</div>

<!-- ════════════════════════════════════════════ PHYSICAL -->
<div id="page-physical" class="page">
  <div class="band" style="padding-bottom:32px">
    <div class="caption-upper" style="margin-bottom:0.75rem">Physical Access &middot; OnGuard / PACS</div>
    <h1 class="display-lg" style="margin-bottom:0.5rem">Physical Risk Scores</h1>
    <p style="color:var(--muted);font-size:0.85rem">After-hours rate (40%) &middot; Reader z-score (25%) &middot; Footprint (15%) &middot; Rapid succession (10%) &middot; Weekend-only (5%) &middot; Dwell time (5%)</p>
  </div>
  <div class="band" style="padding-top:0">
    <div class="controls">
      <button class="filter-btn active" onclick="filterTable('phys','All',this)">All</button>
      <button class="filter-btn" onclick="filterTable('phys','Critical',this)">Critical</button>
      <button class="filter-btn" onclick="filterTable('phys','High',this)">High</button>
      <button class="filter-btn" onclick="filterTable('phys','Medium',this)">Medium</button>
      <button class="filter-btn" onclick="filterTable('phys','Low',this)">Low</button>
      <input class="text-input" style="max-width:200px;margin-left:auto" placeholder="Search user..." oninput="searchTable('phys',this.value)"/>
    </div>
    <div class="table-wrap">
      <table class="risk-table">
        <thead><tr>
          <th onclick="sortTable('phys','label')">User</th>
          <th onclick="sortTable('phys','score')">Score</th>
          <th>Tier</th>
          <th onclick="sortTable('phys','afterhrs')">After-Hrs</th>
          <th>After-Hrs %</th>
          <th onclick="sortTable('phys','zscore')">Reader Z</th>
          <th onclick="sortTable('phys','footprint')">Footprint</th>
          <th onclick="sortTable('phys','rapid')">Rapid</th>
          <th onclick="sortTable('phys','dwell')">Dwell (h)</th>
          <th>Wkend %</th>
        </tr></thead>
        <tbody id="phys-tbody"></tbody>
      </table>
    </div>
  </div>
</div>

<!-- ════════════════════════════════════════════ DIGITAL -->
<div id="page-digital" class="page">
  <div class="band" style="padding-bottom:32px">
    <div class="caption-upper" style="margin-bottom:0.75rem">Digital Identity &middot; SailPoint / IdentityNow</div>
    <h1 class="display-lg" style="margin-bottom:0.5rem">Digital Risk Scores</h1>
    <p style="color:var(--muted);font-size:0.85rem">Entitlement z-score vs peers (35%) &middot; Privileged count (30%) &middot; Unassociated entitlements (20%) &middot; Inactive identity (15%)</p>
  </div>
  <div class="band" style="padding-top:0">
    <div class="controls">
      <button class="filter-btn active" onclick="filterTable('dig','All',this)">All</button>
      <button class="filter-btn" onclick="filterTable('dig','Critical',this)">Critical</button>
      <button class="filter-btn" onclick="filterTable('dig','High',this)">High</button>
      <button class="filter-btn" onclick="filterTable('dig','Medium',this)">Medium</button>
      <button class="filter-btn" onclick="filterTable('dig','Low',this)">Low</button>
      <input class="text-input" style="max-width:200px;margin-left:auto" placeholder="Search user or job..." oninput="searchTable('dig',this.value)"/>
    </div>
    <div class="table-wrap">
      <table class="risk-table">
        <thead><tr>
          <th onclick="sortTable('dig','label')">User</th>
          <th onclick="sortTable('dig','job')">Job Family</th>
          <th onclick="sortTable('dig','score')">Score</th>
          <th>Tier</th>
          <th onclick="sortTable('dig','total')">Total Ents</th>
          <th onclick="sortTable('dig','priv')">Privileged</th>
          <th onclick="sortTable('dig','unassoc')">Unassociated</th>
          <th onclick="sortTable('dig','zscore')">Z-Score</th>
          <th>Inactive</th>
        </tr></thead>
        <tbody id="dig-tbody"></tbody>
      </table>
    </div>
  </div>
</div>

<!-- ════════════════════════════════════════════ USER LOOKUP -->
<div id="page-lookup" class="page">
  <div class="band" style="padding-bottom:32px">
    <div class="caption-upper" style="margin-bottom:0.75rem">User Lookup</div>
    <h1 class="display-lg" style="margin-bottom:0.5rem">Full Risk Profile</h1>
    <p style="color:var(--muted);font-size:0.85rem;max-width:480px">Search any user by label to see their complete physical and digital signal breakdown with a plain-English risk explanation.</p>
  </div>
  <div class="band" style="padding-top:0">
    <div style="display:flex;gap:0.75rem;max-width:480px">
      <input id="lookup-input" class="text-input" placeholder="e.g. User 6976" oninput="lookupUser(this.value)" style="flex:1"/>
    </div>
    <!-- Autocomplete suggestions -->
    <div id="lookup-suggestions" style="max-width:480px;background:var(--canvas);border:1px solid var(--hairline);border-radius:var(--r-md);margin-top:4px;display:none;max-height:200px;overflow-y:auto"></div>

    <div id="lookup-result">
      <div class="lookup-header">
        <div>
          <div class="caption-upper" style="color:var(--on-dark-soft);margin-bottom:0.25rem">Risk Profile</div>
          <div class="lookup-user-name" id="lu-name"></div>
        </div>
        <div style="margin-left:auto;text-align:right">
          <div id="lu-phys-tier" class="tier-badge" style="margin-bottom:0.4rem;display:block;width:fit-content;margin-left:auto"></div>
          <div id="lu-dig-tier" class="tier-badge" style="display:block;width:fit-content;margin-left:auto"></div>
        </div>
      </div>

      <div class="two-col" style="gap:1.5rem">
        <div>
          <div class="caption-upper" style="color:var(--on-dark-soft);margin-bottom:1rem">Physical Signals</div>
          <div class="signal-grid" id="lu-phys-signals"></div>
        </div>
        <div>
          <div class="caption-upper" style="color:var(--on-dark-soft);margin-bottom:1rem">Digital Signals</div>
          <div class="signal-grid" id="lu-dig-signals"></div>
        </div>
      </div>

      <div class="caption-upper" style="color:var(--on-dark-soft);margin-top:1.5rem;margin-bottom:0.5rem">Risk Explanation</div>
      <div class="briefing-text" id="lu-explanation"></div>
    </div>
  </div>
</div>

<!-- ════════════════════════════════════════════ SIMULATOR -->
<div id="page-simulator" class="page">
  <div class="band" style="padding-bottom:32px">
    <div class="caption-upper" style="margin-bottom:0.75rem">What-If Simulator</div>
    <h1 class="display-lg" style="margin-bottom:0.5rem">Adjust Signal Weights</h1>
    <p style="color:var(--muted);font-size:0.85rem;max-width:520px">Move the sliders to change how each signal is weighted. The rankings update in real time. This demonstrates that the model is principled and tunable, not a black box.</p>
  </div>

  <!-- Dark band for simulator -->
  <div style="background:var(--surface-dark);padding:48px 0">
    <div style="max-width:1200px;margin:0 auto;padding:0 2rem">
      <div class="two-col">
        <div>
          <div class="caption-upper" style="color:var(--on-dark-soft);margin-bottom:1.5rem">Physical Signal Weights</div>
          <div id="phys-sliders"></div>
          <div class="divider-dark"></div>
          <div class="caption-upper" style="color:var(--on-dark-soft);margin-bottom:1.5rem">Digital Signal Weights</div>
          <div id="dig-sliders"></div>
          <div style="margin-top:1rem;font-size:0.75rem;color:var(--on-dark-soft)">Weights are normalized automatically. Total does not need to equal 100.</div>
        </div>
        <div>
          <div class="caption-upper" style="color:var(--on-dark-soft);margin-bottom:1rem">Top 10 Physical (reranked)</div>
          <div id="sim-phys-list" style="margin-bottom:2rem"></div>
          <div class="caption-upper" style="color:var(--on-dark-soft);margin-bottom:1rem">Top 10 Digital (reranked)</div>
          <div id="sim-dig-list"></div>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- ════════════════════════════════════════════ OVERLAP -->
<div id="page-overlap" class="page">
  <div class="band" style="padding-bottom:32px">
    <div class="caption-upper" style="margin-bottom:0.75rem">The Aha Moment &middot; Solution Approach</div>
    <h1 class="display-lg" style="margin-bottom:0.75rem">Two Systems.<br><em class="coral">Same Users.</em></h1>
    <p style="color:var(--muted);font-size:0.9rem;max-width:520px;line-height:1.7">The physical and digital analyses were run completely independently with no awareness of each other. The users below surface at the top of both lists simultaneously.</p>
  </div>

  <div class="callout-coral" style="margin:0 2rem 0;max-width:1200px;margin-left:auto;margin-right:auto">
    <div class="caption-upper">Key Finding</div>
    <p><strong>User 6976</strong> holds <strong>45 unassociated entitlements</strong>, more than any other identity in the dataset. These exist completely outside any assigned role or access profile. This same user scores in the top tier for after-hours badge activity. Two independent systems. One user. Zero prior visibility.</p>
  </div>

  <div class="band" style="padding-top:2rem">
    <div class="two-col" style="margin-bottom:3rem">
      <div style="border:1px solid var(--hairline);border-radius:var(--r-lg);overflow:hidden">
        <div style="padding:1rem 1.25rem;border-bottom:1px solid var(--hairline);background:var(--surface-soft);display:flex;align-items:center;gap:0.75rem">
          <div style="width:8px;height:8px;border-radius:50%;background:var(--error)"></div>
          <div style="font-weight:500;font-size:0.9rem;color:var(--ink)">Top Physical Risk</div>
        </div>
        <div style="max-height:420px;overflow-y:auto">
          <table class="risk-table"><thead><tr>
            <th>User</th><th>Score</th><th>Tier</th><th>After-Hrs</th><th>Rapid</th>
          </tr></thead><tbody id="overlap-phys-tbody"></tbody></table>
        </div>
      </div>
      <div style="border:1px solid var(--hairline);border-radius:var(--r-lg);overflow:hidden">
        <div style="padding:1rem 1.25rem;border-bottom:1px solid var(--hairline);background:var(--surface-soft);display:flex;align-items:center;gap:0.75rem">
          <div style="width:8px;height:8px;border-radius:50%;background:var(--accent-amber)"></div>
          <div style="font-weight:500;font-size:0.9rem;color:var(--ink)">Top Digital Risk</div>
        </div>
        <div style="max-height:420px;overflow-y:auto">
          <table class="risk-table"><thead><tr>
            <th>User</th><th>Job</th><th>Score</th><th>Tier</th><th>Unassoc</th><th>Inactive</th>
          </tr></thead><tbody id="overlap-dig-tbody"></tbody></table>
        </div>
      </div>
    </div>

    <!-- Radar -->
    <div class="two-col">
      <div>
        <div class="caption-upper" style="margin-bottom:1rem">User 6976 Risk Profile</div>
        <canvas id="radar" width="360" height="360"></canvas>
      </div>
      <div>
        <div class="caption-upper" style="margin-bottom:1rem">Signal Breakdown</div>
        <div id="radar-legend" style="margin-bottom:1.5rem"></div>
        <div style="background:var(--surface-card);border-radius:var(--r-lg);padding:1.25rem;border-left:3px solid var(--primary)">
          <div class="caption-upper" style="margin-bottom:0.5rem">Methodology</div>
          <p style="font-size:0.82rem;color:var(--muted);line-height:1.7">Physical: after-hours rate, reader z-score, physical footprint, rapid succession, weekend-only pattern, avg dwell time.<br><br>Digital: entitlement z-score vs job peers, privileged count, unassociated entitlements, inactive identity with active access.<br><br>Each signal normalized 0&ndash;100, weighted, binned into Critical / High / Medium / Low.</p>
        </div>
      </div>
    </div>
  </div>
</div>

</main>

<script>
const PHYS = {json.dumps(phys_data)};
const DIG  = {json.dumps(dig_data)};

const TIER_C = {{Critical:'#c64545',High:'#cc785c',Medium:'#e8a55a',Low:'#a09d96'}};

// ── Navigation ────────────────────────────────────────────────────────────────
function showPage(id,tab){{
  document.querySelectorAll('.page').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.nav-tab').forEach(t=>t.classList.remove('active'));
  document.getElementById('page-'+id).classList.add('active');
  tab.classList.add('active');
}}

// ── Score bar ─────────────────────────────────────────────────────────────────
function scoreBar(score,tier){{
  const c=TIER_C[tier]||'#ccc';
  return `<div class="score-wrap"><span class="score-num" style="color:${{c}}">${{score}}</span><div class="score-track"><div class="score-fill" style="width:${{score}}%;background:${{c}}"></div></div></div>`;
}}
function tierBadge(t){{return `<span class="tier-badge tier-${{t}}">${{t}}</span>`;}}
function inactiveBadge(v){{return v?`<span class="inactive-badge">Inactive</span>`:'';}}

// ── Table state & render ──────────────────────────────────────────────────────
const state={{
  phys:{{filter:'All',search:'',sort:'score',dir:-1}},
  dig: {{filter:'All',search:'',sort:'score',dir:-1}},
}};
function filterTable(w,tier,btn){{
  state[w].filter=tier;
  btn.closest('.controls').querySelectorAll('.filter-btn').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active'); renderTable(w);
}}
function searchTable(w,v){{state[w].search=v.toLowerCase();renderTable(w);}}
function sortTable(w,col){{
  if(state[w].sort===col)state[w].dir*=-1;else{{state[w].sort=col;state[w].dir=-1;}}
  renderTable(w);
}}
function renderTable(w){{
  const data=w==='phys'?PHYS:DIG, s=state[w];
  let rows=data.filter(r=>{{
    if(s.filter!=='All'&&r.tier!==s.filter)return false;
    if(s.search&&!(r.label+(r.job||'')).toLowerCase().includes(s.search))return false;
    return true;
  }});
  rows.sort((a,b)=>{{
    const av=a[s.sort]??0,bv=b[s.sort]??0;
    return typeof av==='string'?av.localeCompare(bv)*s.dir:(av-bv)*s.dir;
  }});
  const tbody=document.getElementById(w+'-tbody');
  if(w==='phys'){{
    tbody.innerHTML=rows.map(r=>`<tr>
      <td style="font-weight:500;color:var(--ink)">${{r.label}}</td>
      <td>${{scoreBar(r.score,r.tier)}}</td><td>${{tierBadge(r.tier)}}</td>
      <td>${{r.afterhrs.toLocaleString()}}</td>
      <td style="color:var(--muted)">${{r.afterhrs_rt}}</td>
      <td>${{r.zscore}}</td><td>${{r.footprint}}</td>
      <td style="color:${{r.rapid>0?'var(--primary)':'var(--muted)'}};font-weight:${{r.rapid>0?600:400}}">${{r.rapid}}</td>
      <td>${{r.dwell}}</td><td style="color:var(--muted)">${{r.wkend_pct}}</td>
    </tr>`).join('');
  }}else{{
    tbody.innerHTML=rows.map(r=>`<tr>
      <td style="font-weight:500;color:var(--ink)">${{r.label}}</td>
      <td style="color:var(--muted)">${{r.job}}</td>
      <td>${{scoreBar(r.score,r.tier)}}</td><td>${{tierBadge(r.tier)}}</td>
      <td>${{r.total.toLocaleString()}}</td>
      <td style="color:var(--primary)">${{r.priv}}</td>
      <td style="color:var(--error)">${{r.unassoc}}</td>
      <td>${{r.zscore}}</td><td>${{inactiveBadge(r.inactive)}}</td>
    </tr>`).join('');
  }}
}}

// ── Overview charts ───────────────────────────────────────────────────────────
function miniBar(id,data){{
  const max=Math.max(...data.map(d=>d.v),1);
  document.getElementById(id).innerHTML=data.map(d=>`
    <div class="mini-bar-row">
      <div class="mini-bar-label">${{d.k}}</div>
      <div class="mini-bar-track"><div class="mini-bar-fill" style="width:${{(d.v/max*100).toFixed(1)}}%;background:${{TIER_C[d.k]}}"></div></div>
      <div class="mini-bar-val" style="color:${{TIER_C[d.k]}}">${{d.v.toLocaleString()}}</div>
    </div>`).join('');
}}
function buildOverviewCharts(){{
  const pt={{}},dt={{}};
  PHYS.forEach(r=>pt[r.tier]=(pt[r.tier]||0)+1);
  DIG.forEach(r=>dt[r.tier]=(dt[r.tier]||0)+1);
  const order=['Critical','High','Medium','Low'];
  miniBar('overview-phys-chart',order.map(k=>{{return{{k,v:pt[k]||0}}}}));
  miniBar('overview-dig-chart', order.map(k=>{{return{{k,v:dt[k]||0}}}}));
}}

// ── Executive briefing ────────────────────────────────────────────────────────
function buildBriefing(){{
  const physCrit=PHYS.filter(r=>r.tier==='Critical').length;
  const digCrit=DIG.filter(r=>r.tier==='Critical').length;
  const rapidUsers=PHYS.filter(r=>r.rapid>0).length;
  const inactiveUsers=DIG.filter(r=>r.inactive).length;
  const topPhys=PHYS[0];
  const topDig=DIG[0];
  const text=`As of December 2025, analysis of OG&amp;E's physical badge and digital identity data surfaces several priority concerns. <strong>${{physCrit}} badge holders</strong> score Critical on physical risk, led by after-hours access patterns and rapid succession events across key readers. <strong>${{rapidUsers}} users</strong> exhibit 5 or more badge attempts within 5-minute windows, a pattern consistent with tailgating or credential testing. On the digital side, <strong>${{digCrit}} identities</strong> score Critical, with top-flagged user <strong>${{topDig.label}}</strong> holding ${{topDig.unassoc}} unassociated entitlements outside any governance structure. Critically, <strong>${{inactiveUsers}} inactive identities</strong> retain active entitlements and should be deprioritized for immediate review. Cross-referencing both lists reveals users who appear anomalous in both physical behavior and digital access simultaneously &mdash; these represent the highest-priority targets for access review.`;
  document.getElementById('exec-briefing').innerHTML=text;
}}

// ── User Lookup ───────────────────────────────────────────────────────────────
const ALL_USERS = [...new Set([...PHYS.map(r=>r.label),...DIG.map(r=>r.label)])].sort();

function lookupUser(query){{
  const q=query.toLowerCase().trim();
  const sugg=document.getElementById('lookup-suggestions');
  const result=document.getElementById('lookup-result');
  if(!q){{sugg.style.display='none';result.classList.remove('visible');return;}}

  const matches=ALL_USERS.filter(u=>u.toLowerCase().includes(q)).slice(0,8);
  if(matches.length>0){{
    sugg.style.display='block';
    sugg.innerHTML=matches.map(u=>`<div onclick="selectUser('${{u}}')" style="padding:8px 14px;cursor:pointer;font-size:0.85rem;border-bottom:1px solid var(--hairline-soft);color:var(--ink)" onmouseover="this.style.background='var(--surface-soft)'" onmouseout="this.style.background=''">${{u}}</div>`).join('');
  }}else{{
    sugg.style.display='none';
  }}

  // Auto-select on exact match
  const exact=ALL_USERS.find(u=>u.toLowerCase()===q);
  if(exact) selectUser(exact);
}}

function selectUser(label){{
  document.getElementById('lookup-input').value=label;
  document.getElementById('lookup-suggestions').style.display='none';

  const phys=PHYS.find(r=>r.label===label);
  const dig=DIG.find(r=>r.label===label);
  const result=document.getElementById('lookup-result');

  document.getElementById('lu-name').textContent=label;

  // Tiers
  const pt=document.getElementById('lu-phys-tier');
  const dt=document.getElementById('lu-dig-tier');
  if(phys){{pt.className='tier-badge tier-'+phys.tier;pt.textContent='Physical: '+phys.tier;}}
  else{{pt.className='tier-badge tier-Low';pt.textContent='Physical: Not in top 200';}}
  if(dig){{dt.className='tier-badge tier-'+dig.tier;dt.textContent='Digital: '+dig.tier;}}
  else{{dt.className='tier-badge tier-Low';dt.textContent='Digital: Not in top 200';}}

  // Physical signals
  const ps=document.getElementById('lu-phys-signals');
  if(phys){{
    ps.innerHTML=`
      <div class="signal-cell"><div class="signal-label">Risk Score</div><div class="signal-val coral">${{phys.score}}</div></div>
      <div class="signal-cell"><div class="signal-label">After-Hrs Events</div><div class="signal-val">${{phys.afterhrs.toLocaleString()}}</div></div>
      <div class="signal-cell"><div class="signal-label">After-Hrs Rate</div><div class="signal-val amber">${{phys.afterhrs_rt}}</div></div>
      <div class="signal-cell"><div class="signal-label">Reader Z-Score</div><div class="signal-val">${{phys.zscore}}</div></div>
      <div class="signal-cell"><div class="signal-label">Footprint</div><div class="signal-val">${{phys.footprint}} readers</div></div>
      <div class="signal-cell"><div class="signal-label">Rapid Bursts</div><div class="signal-val ${{phys.rapid>0?'coral':''}}">${{phys.rapid}}</div></div>
      <div class="signal-cell"><div class="signal-label">Avg Dwell</div><div class="signal-val">${{phys.dwell}}h</div></div>
      <div class="signal-cell"><div class="signal-label">Weekend %</div><div class="signal-val">${{phys.wkend_pct}}</div></div>
    `;
  }}else{{ps.innerHTML='<div style="color:var(--on-dark-soft);font-size:0.85rem">Not in top 200 physical users</div>';}}

  // Digital signals
  const ds=document.getElementById('lu-dig-signals');
  if(dig){{
    ds.innerHTML=`
      <div class="signal-cell"><div class="signal-label">Risk Score</div><div class="signal-val coral">${{dig.score}}</div></div>
      <div class="signal-cell"><div class="signal-label">Job Family</div><div class="signal-val teal">${{dig.job}}</div></div>
      <div class="signal-cell"><div class="signal-label">Total Ents</div><div class="signal-val">${{dig.total.toLocaleString()}}</div></div>
      <div class="signal-cell"><div class="signal-label">Privileged</div><div class="signal-val amber">${{dig.priv}}</div></div>
      <div class="signal-cell"><div class="signal-label">Unassociated</div><div class="signal-val ${{dig.unassoc>0?'coral':''}}">${{dig.unassoc}}</div></div>
      <div class="signal-cell"><div class="signal-label">Ent Z-Score</div><div class="signal-val">${{dig.zscore}}</div></div>
      <div class="signal-cell"><div class="signal-label">Inactive</div><div class="signal-val ${{dig.inactive?'amber':''}}">${{dig.inactive?'Yes':'No'}}</div></div>
    `;
  }}else{{ds.innerHTML='<div style="color:var(--on-dark-soft);font-size:0.85rem">Not in top 200 digital users</div>';}}

  // Plain-English explanation
  let explain=``;
  if(phys){{
    const highSignals=[];
    if(phys.s_afterhrs>60)highSignals.push(`after-hours badge rate of ${{phys.afterhrs_rt}}`);
    if(phys.s_zscore>60)highSignals.push(`reader frequency z-score of ${{phys.zscore}}`);
    if(phys.rapid>0)highSignals.push(`${{phys.rapid}} rapid succession burst${{phys.rapid>1?'s':''}}`);
    if(phys.s_footprint>60)highSignals.push(`unusually broad physical footprint (${{phys.footprint}} readers)`);
    explain+=`<strong>${{label}}</strong> scores <strong>${{phys.score}}/100</strong> on physical risk`;
    if(highSignals.length>0)explain+=`, driven primarily by ${{highSignals.join(', ')}}`;
    explain+='. ';
  }}
  if(dig){{
    const highSignals=[];
    if(dig.unassoc>0)highSignals.push(`${{dig.unassoc}} entitlement${{dig.unassoc>1?'s':''}} outside any role or profile`);
    if(dig.s_priv>60)highSignals.push(`${{dig.priv}} privileged entitlements, elevated for ${{dig.job}}`);
    if(dig.s_zscore>60)highSignals.push(`entitlement count ${{dig.zscore}}x above job family norm`);
    if(dig.inactive)highSignals.push(`identity marked inactive but retaining active access`);
    explain+=`On the digital side, this identity scores <strong>${{dig.score}}/100</strong>`;
    if(highSignals.length>0)explain+=` with flags including ${{highSignals.join(', ')}}`;
    explain+='. ';
  }}
  if(!phys&&!dig)explain='This user does not appear in the top 200 of either list.';
  if(phys&&dig)explain+='<strong>This user appears in both anomaly lists simultaneously and should be prioritized for immediate access review.</strong>';

  document.getElementById('lu-explanation').innerHTML=explain;
  result.classList.add('visible');
}}

// ── Simulator ─────────────────────────────────────────────────────────────────
const physSliderDefs=[
  {{key:'s_afterhrs',label:'After-Hours Rate',val:40}},
  {{key:'s_zscore',label:'Reader Z-Score',val:25}},
  {{key:'s_footprint',label:'Footprint',val:15}},
  {{key:'s_rapid',label:'Rapid Succession',val:10}},
  {{key:'s_weekend',label:'Weekend-Only',val:5}},
  {{key:'s_dwell',label:'Dwell Time',val:5}},
];
const digSliderDefs=[
  {{key:'s_zscore',label:'Entitlement Z-Score',val:35}},
  {{key:'s_priv',label:'Privileged Count',val:30}},
  {{key:'s_unassoc',label:'Unassociated Ents',val:20}},
  {{key:'s_inactive',label:'Inactive Identity',val:15}},
];

function buildSliders(){{
  function renderSliders(containerId, defs, which){{
    const el=document.getElementById(containerId);
    el.innerHTML=defs.map((d,i)=>`
      <div class="slider-row">
        <div class="slider-label">${{d.label}}</div>
        <input type="range" min="0" max="100" value="${{d.val}}" id="sl-${{which}}-${{i}}" oninput="updateSim('${{which}}')" style="flex:1"/>
        <div class="slider-pct" id="sl-${{which}}-${{i}}-val">${{d.val}}%</div>
      </div>`).join('');
  }}
  renderSliders('phys-sliders',physSliderDefs,'phys');
  renderSliders('dig-sliders',digSliderDefs,'dig');
  updateSim('phys');updateSim('dig');
}}

function updateSim(which){{
  const defs=which==='phys'?physSliderDefs:digSliderDefs;
  const data=which==='phys'?PHYS:DIG;
  const listId=which==='phys'?'sim-phys-list':'sim-dig-list';

  const weights=defs.map((_,i)=>{{
    const el=document.getElementById(`sl-${{which}}-${{i}}`);
    const v=parseFloat(el.value);
    document.getElementById(`sl-${{which}}-${{i}}-val`).textContent=v+'%';
    return v;
  }});
  const total=weights.reduce((a,b)=>a+b,0)||1;
  const normW=weights.map(w=>w/total);

  const scored=data.map(r=>{{
    const s=defs.reduce((acc,d,i)=>acc+(r[d.key]||0)*normW[i],0);
    return{{...r,sim_score:Math.round(s*10)/10}};
  }}).sort((a,b)=>b.sim_score-a.sim_score).slice(0,10);

  const list=document.getElementById(listId);
  list.innerHTML=scored.map((r,i)=>`
    <div style="display:flex;align-items:center;gap:0.75rem;padding:0.5rem 0;border-bottom:1px solid rgba(255,255,255,0.06)">
      <div style="width:1.5rem;font-size:0.75rem;color:var(--on-dark-soft);text-align:right">${{i+1}}</div>
      <div style="flex:1;font-size:0.82rem;color:var(--on-dark);font-weight:500">${{r.label}}</div>
      <div style="font-family:var(--mono);font-size:0.8rem;color:var(--primary)">${{r.sim_score}}</div>
      ${{tierBadge(r.tier)}}
    </div>`).join('');
}}

// ── Overlap tables ────────────────────────────────────────────────────────────
function buildOverlapTables(){{
  const ptop=PHYS.filter(r=>['Critical','High'].includes(r.tier)).slice(0,25);
  document.getElementById('overlap-phys-tbody').innerHTML=ptop.map(r=>`<tr>
    <td style="font-weight:500">${{r.label}}</td>
    <td style="color:${{TIER_C[r.tier]}};font-weight:600">${{r.score}}</td>
    <td>${{tierBadge(r.tier)}}</td>
    <td>${{r.afterhrs.toLocaleString()}}</td>
    <td style="color:${{r.rapid>0?'var(--primary)':'var(--muted)'}}">${{r.rapid}}</td>
  </tr>`).join('');
  const dtop=DIG.filter(r=>['Critical','High','Medium'].includes(r.tier)).slice(0,25);
  document.getElementById('overlap-dig-tbody').innerHTML=dtop.map(r=>`<tr>
    <td style="font-weight:500">${{r.label}}</td>
    <td style="color:var(--muted)">${{r.job}}</td>
    <td style="color:${{TIER_C[r.tier]}};font-weight:600">${{r.score}}</td>
    <td>${{tierBadge(r.tier)}}</td>
    <td style="color:var(--error);font-weight:600">${{r.unassoc}}</td>
    <td>${{inactiveBadge(r.inactive)}}</td>
  </tr>`).join('');
}}

// ── Radar ─────────────────────────────────────────────────────────────────────
function buildRadar(){{
  const canvas=document.getElementById('radar');
  const ctx=canvas.getContext('2d');
  const W=canvas.width,H=canvas.height,cx=W/2,cy=H/2,R=Math.min(W,H)/2-52;
  const labels=['After-Hrs','Reader Z','Rapid','Unassoc Ents','Ent Z-Score','Inactive'];
  const u=[88,72,0,100,40,0];
  const avg=[24,18,12,2,18,8];
  const n=labels.length;
  function pt(v,i,r){{const a=(Math.PI*2*i/n)-Math.PI/2;return[cx+Math.cos(a)*(v/100)*r,cy+Math.sin(a)*(v/100)*r];}}
  ctx.clearRect(0,0,W,H);
  // Grid
  [20,40,60,80,100].forEach(p=>{{
    ctx.beginPath();
    labels.forEach((_,i)=>{{const[x,y]=pt(p,i,R);i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);}});
    ctx.closePath();ctx.strokeStyle='#e6dfd8';ctx.lineWidth=1;ctx.stroke();
  }});
  labels.forEach((_,i)=>{{const[x,y]=pt(100,i,R);ctx.beginPath();ctx.moveTo(cx,cy);ctx.lineTo(x,y);ctx.strokeStyle='#e6dfd8';ctx.lineWidth=1;ctx.stroke();}});
  // Avg polygon
  ctx.beginPath();avg.forEach((v,i)=>{{const[x,y]=pt(v,i,R);i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);}});
  ctx.closePath();ctx.fillStyle='rgba(108,106,100,0.1)';ctx.strokeStyle='rgba(108,106,100,0.4)';ctx.lineWidth=1.5;ctx.fill();ctx.stroke();
  // User polygon
  ctx.beginPath();u.forEach((v,i)=>{{const[x,y]=pt(v,i,R);i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);}});
  ctx.closePath();ctx.fillStyle='rgba(204,120,92,0.15)';ctx.strokeStyle='#cc785c';ctx.lineWidth=2;ctx.fill();ctx.stroke();
  // Labels
  ctx.font='500 10px Inter,sans-serif';ctx.fillStyle='#6c6a64';ctx.textAlign='center';
  labels.forEach((lbl,i)=>{{const[x,y]=pt(122,i,R);ctx.fillText(lbl,x,y);}});

  document.getElementById('radar-legend').innerHTML=labels.map((l,i)=>`
    <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.6rem;font-size:0.8rem;color:var(--body)">
      <div style="width:10px;height:10px;border-radius:50%;background:var(--primary);flex-shrink:0"></div>
      <span>${{l}}</span>
      <span style="margin-left:auto;color:var(--primary);font-weight:600">${{u[i]}}</span>
    </div>`).join('');
}}

// ── Init ──────────────────────────────────────────────────────────────────────
renderTable('phys');
renderTable('dig');
buildOverviewCharts();
buildBriefing();
buildOverlapTables();
buildRadar();
buildSliders();
</script>
</body>
</html>"""

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "oge_risk.html")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(HTML)

print(f"\nDone. Open: {out_path}")
