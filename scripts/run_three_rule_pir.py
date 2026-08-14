"""Three reference-rule PIR sensitivity for the Minnesota study.

Reports, for each budget scenario, the realized PIR under the three reference rules
of manuscript Section (framework) §reference:
    (i)   socially best-PNE reference   u_i^dagger = u_i(best PNE)
    (ii)  average reference             u_i^dagger = mean_k u_i(PNE_k)
    (iii) minimum reference             u_i^dagger = min_k  u_i(PNE_k)

Only the BSW model is re-solved (8 solves total: 4 scenarios x {avg, min};
the best-PNE column is the value already in the paper). BZR and SW are NOT re-run:
    * the collected PNE strategy profiles are read from all_PNE_sets_{pct}.pkl;
    * per-county utilities are reconstructed deterministically with compute_utility
      (no optimization) -- this matches the BSW IR constraint's destination-county
      incoming-coverage accounting exactly;
    * SW is read from df_test_minnesota_SW_enhancing_{pct}.csv.

Feasibility handling: if a reference model is INFEASIBLE it is reported
explicitly as "INFEASIBLE", never silently replaced. The script also verifies
computationally that the average and minimum floors lie weakly below the best-PNE
utilities coordinate-wise, and whether the best PNE is individually dominant.

Usage
-----
    python scripts/run_three_rule_pir.py \
        --results-dir nash_informed/ebmc/results/minnesota \
        --data-dir data \
        --percentage 50% 75% 100% 125% \
        --bsw-time-limit 600 --threads 16

Output
------
    <results-dir>/three_rule_pir.csv   (one row per scenario x rule)
    printed summary table for tab:pir_rules
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from nash_informed.ebmc import load_minnesota_instance, compute_utility, solve_bsw

# Gurobi status codes we treat as "a solution exists"
_OK_STATUS = {2, 9, 13}      # OPTIMAL, TIME_LIMIT (with incumbent), SUBOPTIMAL
_INFEASIBLE_STATUS = {3, 4}  # INFEASIBLE, INF_OR_UNBD


def _load_pne_profiles(pkl_path: Path) -> list[list]:
    """Return a list of PNE strategy profiles (each = list of selected lake ids).

    Handles both the old pkl layout {'minnesota': {'BPNE': [...], 'WPNE': [...]}}
    where each element is a flat list/array of lake ids, and the newer layout
    {'BPNE': [...], 'WPNE': [...], 'deduped': [...]} where each element may be a
    dict with a 'selected_lakes' key. Profiles are de-duplicated by lake set.
    """
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    inner = d["minnesota"] if isinstance(d, dict) and "minnesota" in d else d

    raw = []
    if isinstance(inner, dict):
        for key in ("deduped", "BPNE", "WPNE"):
            raw.extend(inner.get(key, []) or [])
    elif isinstance(inner, list):
        raw = inner

    profiles, seen = [], set()
    for rec in raw:
        if isinstance(rec, dict):
            lakes = rec.get("selected_lakes") or rec.get("x") or []
        else:
            lakes = list(rec)
        key = frozenset(int(x) for x in lakes)
        if key and key not in seen:
            seen.add(key)
            profiles.append([int(x) for x in lakes])
    return profiles


def _read_sw_phi(results_dir: Path, pct: str) -> float:
    csv = results_dir / f"df_test_minnesota_SW_enhancing_{pct}.csv"
    df = pd.read_csv(csv, index_col=0)
    return float(df["SW"].iloc[0])


def _build_reference_vectors(u_by_pne: list[dict], counties: list):
    """Given per-county utilities of each collected PNE, build the three floors."""
    best_idx = int(np.argmax([sum(u.values()) for u in u_by_pne]))
    best = {c: float(u_by_pne[best_idx].get(c, 0.0)) for c in counties}
    avg = {c: float(np.mean([u.get(c, 0.0) for u in u_by_pne])) for c in counties}
    mn = {c: float(min(u.get(c, 0.0) for u in u_by_pne)) for c in counties}
    return best, avg, mn, best_idx


def _solve_and_score(inst, ref_u, sw_phi, warm_start, cfg, tlim):
    res = solve_bsw(inst, ir_ref=ref_u, time_limit=tlim,
                    only_playing=True, warm_start=warm_start, config=cfg)
    status = res.get("status")
    solcount_ok = res.get("phi") == res.get("phi")  # not NaN
    if status in _INFEASIBLE_STATUS or not solcount_ok:
        return {"feasible": False, "bsw_phi": None, "pir": None,
                "status": status, "mipgap": None}
    bsw_phi = float(res["phi"])
    pir = sw_phi / bsw_phi if bsw_phi > 0 else float("nan")
    return {"feasible": True, "bsw_phi": bsw_phi, "pir": pir,
            "status": status, "mipgap": res.get("mipgap")}


def run(args):
    results_dir = Path(args.results_dir)
    cfg = {"gurobi_threads": args.threads}
    rows = []

    for pct in args.percentage:
        print(f"\n{'='*70}\n[{pct}] three-rule PIR\n{'='*70}")
        inst = load_minnesota_instance(data_dir=args.data_dir, percentage=pct)
        counties = list(inst.counties_playing)

        profiles = _load_pne_profiles(results_dir / f"all_PNE_sets_{pct}.pkl")
        if not profiles:
            print(f"  [WARN] no PNE profiles found for {pct}; skipping.")
            continue
        # per-county utilities for each collected PNE (deterministic, no MIP)
        u_by_pne = [compute_utility(inst, prof, only_playing=True)[0] for prof in profiles]
        print(f"  collected PNEs: {len(profiles)}")

        best, avg, mn, best_idx = _build_reference_vectors(u_by_pne, counties)
        warm = profiles[best_idx]
        sw_phi = _read_sw_phi(results_dir, pct)
        print(f"  SW phi = {sw_phi:.0f}")

        # --- diagnostic checks: best-PNE dominance and floor ordering ---
        dominant = all(
            best[c] >= u.get(c, 0.0) - 1e-6 for u in u_by_pne for c in counties
        )
        avg_below = max((avg[c] - best[c]) for c in counties)   # want <= ~0
        min_below = max((mn[c] - best[c]) for c in counties)    # want <= 0
        print(f"  best PNE individually dominant: {dominant}")
        print(f"  max(avg_floor - best): {avg_below:+.3f}   "
              f"max(min_floor - best): {min_below:+.3f}")

        for rule, ref in (("(i) best-PNE", best),
                          ("(ii) average", avg),
                          ("(iii) minimum", mn)):
            sc = _solve_and_score(inst, ref, sw_phi, warm, cfg, args.bsw_time_limit)
            tag = ("INFEASIBLE" if not sc["feasible"]
                   else f"PIR={sc['pir']:.4f}  BSW={sc['bsw_phi']:.0f}")
            print(f"    {rule:16s}: {tag}")
            rows.append({
                "percentage": pct, "rule": rule,
                "n_pne": len(profiles), "sw_phi": sw_phi,
                "bsw_phi": sc["bsw_phi"], "pir": sc["pir"],
                "feasible": sc["feasible"], "status": sc["status"],
                "mipgap": sc["mipgap"],
                "best_pne_dominant": dominant,
                "max_avg_minus_best": avg_below,
                "max_min_minus_best": min_below,
            })

    out = results_dir / "three_rule_pir.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nWrote {out}")

    # tidy PIR matrix for tab:pir_rules
    df = pd.DataFrame(rows)
    if not df.empty:
        piv = df.pivot(index="percentage", columns="rule", values="pir")
        print("\nPIR by scenario x rule (fill tab:pir_rules with these):")
        print(piv.to_string(float_format=lambda x: f"{x:.4f}"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir",
                    default=str(REPO_ROOT / "nash_informed" / "ebmc" / "results" / "minnesota"))
    ap.add_argument("--data-dir", default=str(REPO_ROOT / "data"))
    ap.add_argument("--percentage", nargs="+", default=["50%", "75%", "100%", "125%"])
    ap.add_argument("--bsw-time-limit", type=int, default=600)
    ap.add_argument("--threads", type=int, default=16)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
