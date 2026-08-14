"""Fix the Table 13 z^BSW > z^SW artifact for the two affected KPG Type B instances.

Only two Type B instances are affected (n_players = 10 and 20). There
z^BSW slightly exceeds z^SW, which is impossible at true optimality because
BSW = SW + per-player IR constraints, so the BSW feasible region is a subset of
the SW feasible region and z^BSW <= z^SW must hold. The excess is a solver
termination artifact: SW stopped at its MIP gap / time limit with an incumbent
slightly below the true optimum.

Fix: use the BSW solution as an SW incumbent (MIP start) and rerun
the SW model. A BSW-optimal profile is SW-feasible, so warm-starting SW with it
guarantees the reported z^SW >= z^BSW while never exceeding the true SW optimum;
SW is used in the paper only as the welfare upper bound (POS, PIR denominators),
never as a computational benchmark against BSW, so the warm start is principled.

Minimal scope: the instances are fixed bundled files (type_B/{n}-100-5-cij.txt),
and the reference-utility vector is read from the existing results CSV, so BZR is
NOT re-run. Per instance: solve BSW (to obtain its item selection) then solve SW
warm-started with that selection -> 4 solves total. MIP gap is left at the study
default (1e-4) for consistency with the rest of Table 13.

Usage
-----
    python scripts/rerun_kpg_table13.py \
        --results-csv nash_informed/kpg/results/kpg_results.csv \
        --category B --n-players 10 20 --ratio-pct 5 --n-items 100 \
        --time-limit 600 --threads 16
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from nash_informed.kpg import load_kpg_bundled, solve_bsw, solve_sw


def _parse_ref(cell, players) -> dict:
    """Parse a stored u_player_ref cell into dict[player -> float] keyed to
    the instance's player ids."""
    d = ast.literal_eval(cell) if isinstance(cell, str) else dict(cell)
    # coerce keys to the instance player type (usually int)
    out = {}
    for p in players:
        if p in d:
            out[p] = float(d[p])
        elif int(p) in d:
            out[p] = float(d[int(p)])
        elif str(p) in d:
            out[p] = float(d[str(p)])
        else:
            raise KeyError(f"reference has no entry for player {p!r}; keys={list(d)[:5]}...")
    return out


def _lookup_row(df, category, n, ratio_pct, n_items):
    """Find the CSV row for this (category, n, ratio, items) instance.

    The results CSV does not store 'category' explicitly, so match on the
    instance_meta filename (…/type_{cat}/{n}-{items}-{ratio}-…txt)."""
    tag = f"type_{category}"
    fstub = f"{n}-{n_items}-{ratio_pct}-"
    for _, r in df.iterrows():
        meta = str(r.get("instance_meta", ""))
        if tag in meta and fstub in meta and int(r["n_players"]) == n:
            return r
    # fallback: match on n_players + ratio + items (first occurrence)
    cand = df[(df["n_players"] == n) & (df.get("ratio_pct", ratio_pct) == ratio_pct)
              & (df["n_items"] == n_items)]
    if len(cand):
        return cand.iloc[0]
    raise LookupError(f"no CSV row for category={category} n={n} ratio={ratio_pct}")


def run(args):
    df = pd.read_csv(args.results_csv)
    cfg = {"gurobi_threads": args.threads}
    rows = []

    for n in args.n_players:
        print(f"\n{'='*66}\n[type {args.category}  n={n}]\n{'='*66}")
        inst = load_kpg_bundled(category=args.category, n_players=n,
                                ratio_pct=args.ratio_pct, n_items=args.n_items)
        row = _lookup_row(df, args.category, n, args.ratio_pct, args.n_items)
        sw_old = float(row["sw_phi"])
        bsw_old = float(row["bsw_phi"])
        ref_u = _parse_ref(row["u_player_ref"], inst.players)

        # 1) re-solve BSW with the stored reference to obtain its item selection
        bsw = solve_bsw(inst, ir_ref=ref_u, time_limit=args.time_limit, config=cfg)
        if bsw is None:
            print("  [ERROR] BSW infeasible under stored reference; check ref vector.")
            continue
        bsw_phi = bsw["phi"]

        # 2) re-solve SW warm-started with the BSW profile
        sw = solve_sw(inst, time_limit=args.time_limit,
                      warm_start=bsw["profile"], config=cfg)
        sw_phi = sw["phi"]

        pir_old = sw_old / bsw_old if bsw_old > 0 else float("nan")
        pir_new = sw_phi / bsw_phi if bsw_phi > 0 else float("nan")
        ok = sw_phi >= bsw_phi - 1e-6

        print(f"  BSW: stored {bsw_old:.1f}  ->  recomputed {bsw_phi:.1f}"
              f"  (gap={bsw['mipgap']:.2e}, status={bsw['status']})")
        print(f"  SW : stored {sw_old:.1f}  ->  warm-started {sw_phi:.1f}"
              f"  (gap={sw['mipgap']:.2e}, status={sw['status']})")
        print(f"  PIR: {pir_old:.4f}  ->  {pir_new:.4f}   "
              f"[z^SW >= z^BSW ? {'YES' if ok else 'NO'}]")
        if not ok:
            print("  [WARN] ordering still violated; raise --time-limit or report "
                  "incumbent+bound instead of a point estimate.")

        rows.append({
            "category": args.category, "n_players": n,
            "ratio_pct": args.ratio_pct, "n_items": args.n_items,
            "sw_old": sw_old, "bsw_old": bsw_old, "pir_old": pir_old,
            "sw_new": sw_phi, "bsw_new": bsw_phi, "pir_new": pir_new,
            "sw_mipgap": sw["mipgap"], "bsw_mipgap": bsw["mipgap"],
            "sw_status": sw["status"], "bsw_status": bsw["status"],
            "ordering_ok": ok,
        })

    out = Path(args.results_csv).with_name("kpg_table13_rerun.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nWrote {out}")
    print("\nUpdate Table 13 (EC G) rows n=10 and n=20 with sw_new / bsw_new / pir_new above.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-csv",
                    default=str(REPO_ROOT / "nash_informed" / "kpg" / "results" / "kpg_results.csv"))
    ap.add_argument("--category", default="B")
    ap.add_argument("--n-players", type=int, nargs="+", default=[10, 20])
    ap.add_argument("--ratio-pct", type=int, default=5)
    ap.add_argument("--n-items", type=int, default=100)
    ap.add_argument("--time-limit", type=int, default=600)
    ap.add_argument("--threads", type=int, default=16)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
