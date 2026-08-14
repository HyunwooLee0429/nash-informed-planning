"""Driver: EBMC Nash-informed planning on the random CSV dataset.

Loads pre-generated random EBMC instances from
``nash_informed/ebmc/datasets/ebmc_random/{n}_{lakes_per_county}_random.csv``.
Budgets are not stored in the CSVs; they are regenerated here via the
``--budget-ratio`` flag.

Usage:
    python scripts/run_ebmc_random.py --county-sizes 2 3 5 8 10 15 20 \
        --bzr-time-limit 600 --bsw-time-limit 600 --seed 0
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

from nash_informed.ebmc import load_random_ebmc_csv, run_pipeline_bzr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="nash_informed/ebmc/datasets/ebmc_random")
    parser.add_argument("--county-sizes", type=int, nargs="+",
                        default=[2, 3, 5, 8, 10, 15, 20])
    parser.add_argument("--lakes-per-county", type=int, default=30)
    parser.add_argument(
        "--budget-ratio", default="random",
        help="'random' or a float in (0, 1] for proportional budget.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bzr-time-limit", type=int, default=600)
    parser.add_argument("--bsw-time-limit", type=int, default=600)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--include-bc", action="store_true")
    parser.add_argument("--include-bgc", action="store_true")
    parser.add_argument("--include-gc", action="store_true")
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "nash_informed" / "ebmc" / "results" / "ebmc_random"),
    )
    parser.add_argument("--output-name", default="ebmc_random_results.csv")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    try:
        bratio = float(args.budget_ratio)
    except ValueError:
        bratio = args.budget_ratio  # 'random'

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / args.output_name
    cfg = {"gurobi_threads": args.threads}

    rows: list[dict] = []
    start = time.time()
    for n in args.county_sizes:
        csv_file = Path(args.data_dir) / f"{n}_{args.lakes_per_county}_random.csv"
        if not csv_file.exists():
            print(f"[{n}_{args.lakes_per_county}] missing CSV {csv_file}; skip"); continue
        print(f"\n[random] {csv_file}")
        instance = load_random_ebmc_csv(csv_file, budget_ratio=bratio, seed=args.seed)
        print(f"  {instance.summary()}")

        res = run_pipeline_bzr(
            instance,
            only_playing=False,
            bzr_time_limit=args.bzr_time_limit,
            bsw_time_limit=args.bsw_time_limit,
            include_bc=args.include_bc,
            include_bgc=args.include_bgc,
            include_gc=args.include_gc,
            config=cfg,
            verbose=args.verbose,
        )
        if "error" in res:
            print(f"  ERROR: {res['error']}"); continue

        res["county_size"] = n
        res["lakes_per_county"] = args.lakes_per_county
        res["budget_ratio"] = args.budget_ratio
        res["seed"] = args.seed
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
