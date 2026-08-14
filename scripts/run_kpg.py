"""Driver: KPG Nash-informed planning on bundled type-A/B/C instances.

Loads ``nash_informed/kpg/datasets/type_{A,B,C}/{n}-{items}-{ratio}*.txt``
and runs the BZR + BSW pipeline. Reports POS, POA, PIR per (category,
n_players, ratio_pct) and saves incremental CSV.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from nash_informed.kpg import load_kpg_bundled, run_pipeline_bzr, list_bundled_instances


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--categories", nargs="+", default=["A", "B", "C"],
                        choices=["A", "B", "C"])
    parser.add_argument("--player-sizes", type=int, nargs="+",
                        default=[2, 3, 5, 8, 10, 15, 20])
    parser.add_argument("--ratio-pcts", type=int, nargs="+", default=[2, 5, 8],
                        help="Budget ratio labels (interpreted as 0.2 / 0.5 / 0.8).")
    parser.add_argument("--n-items", type=int, default=100)
    parser.add_argument("--bzr-time-limit", type=int, default=600)
    parser.add_argument("--bsw-time-limit", type=int, default=600)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--no-strategic", action="store_true",
                        help="Disable strategic dominance/payoff inequalities.")
    parser.add_argument("--strategic-cap", type=int, default=15,
                        help="Skip strategic inequalities when n_players > cap.")
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "nash_informed" / "kpg" / "results"),
    )
    parser.add_argument("--output-name", default="kpg_results.csv")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / args.output_name
    cfg = {"gurobi_threads": args.threads}

    rows: list[dict] = []
    start = time.time()

    for cat in args.categories:
        available = list_bundled_instances(cat)
        for n in args.player_sizes:
            for r in args.ratio_pcts:
                label = f"{cat}/{n}-{args.n_items}-{r}"
                if (n, args.n_items, r) not in available:
                    print(f"[{label}] not bundled; skip"); continue
                print(f"\n[{label}]")
                try:
                    inst = load_kpg_bundled(category=cat, n_players=n,
                                             ratio_pct=r, n_items=args.n_items)
                except Exception as e:
                    print(f"  LOAD FAIL: {e}"); continue
                print(f"  {inst.summary()}")

                res = run_pipeline_bzr(
                    inst,
                    bzr_time_limit=args.bzr_time_limit,
                    bsw_time_limit=args.bsw_time_limit,
                    use_strategic_inequalities=not args.no_strategic,
                    strategic_player_cap=args.strategic_cap,
                    config=cfg,
                    verbose=args.verbose,
                )
                if "error" in res:
                    print(f"  ERROR: {res['error']}"); continue

                res["ratio_pct"] = r
                rows.append(res)
                pd.DataFrame(rows).to_csv(csv_path, index=False)

                print(
                    f"  SW={res['sw_phi']:.1f}  bestPNE={res['best_pne_phi']:.1f}  "
                    f"worstPNE={res['worst_pne_phi']:.1f}  BSW={res['bsw_phi']:.1f}  "
                    f"POS={res['POS']:.4f}  POA={res['POA']:.4f}  PIR={res['PIR']:.4f}  "
                    f"wall={res['wall_seconds']:.1f}s"
                )

    elapsed = time.time() - start
    print(f"\nDone in {elapsed/60:.1f} min. {len(rows)} runs.  CSV: {csv_path}")


if __name__ == "__main__":
    main()
