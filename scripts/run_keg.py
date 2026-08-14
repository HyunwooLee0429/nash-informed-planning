"""Driver: KEG Nash-informed planning on bundled Constantino instances.

Default workflow loads the bundled ``small`` or ``big`` dataset from
``nash_informed/keg/datasets/``. Override with ``--instance-dir`` if you have
your own copy elsewhere. Writes incremental CSV named after the dataset.

Usage:
    # Bundled small dataset (default)
    python scripts/run_keg.py --dataset small --sizes 60 80 100 --instances 1 2 3 4 5

    # Bundled big dataset
    python scripts/run_keg.py --dataset big --sizes 100 200 300 400 500 --instances 1 2 3
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

from nash_informed.keg import (
    load_constantino_instance, run_pipeline_bzr,
    bundled_dataset_dir, list_bundled_sizes,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=["small", "big"], default="small",
        help="Bundled dataset name (resolves to nash_informed/keg/datasets/<dataset>/).",
    )
    parser.add_argument(
        "--instance-dir", default=None,
        help="Override bundled directory with a custom path.",
    )
    parser.add_argument("--sizes", type=int, nargs="+", default=None,
                        help="Sizes to run; default = all bundled sizes for the dataset.")
    parser.add_argument(
        "--instances", type=int, nargs="+", default=list(range(1, 6)),
        help="Instance ids within each size.",
    )
    parser.add_argument(
        "--partitions", nargs="+",
        default=["uniform", "geographic", "blood_type"],
    )
    parser.add_argument("--n-players", type=int, default=3)
    parser.add_argument("--l-max", type=int, default=3)
    parser.add_argument("--time-limit", type=int, default=600,
                        help="Seconds per solve (SW, each BZR direction, BSW).")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "nash_informed" / "keg" / "results"),
        help="Output directory.",
    )
    parser.add_argument(
        "--output-name", default=None,
        help="CSV filename; defaults to 'keg_<dataset>_results.csv'.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    instance_dir = Path(args.instance_dir) if args.instance_dir else bundled_dataset_dir(args.dataset)
    if not instance_dir.exists():
        sys.exit(f"Instance directory not found: {instance_dir}")

    sizes = args.sizes if args.sizes else list_bundled_sizes(args.dataset)
    if not sizes:
        sys.exit(f"No bundled sizes for dataset {args.dataset!r}.")

    out_name = args.output_name or f"keg_{args.dataset}_results.csv"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / out_name

    results: list[dict] = []
    failed: list[dict] = []
    n_runs = len(sizes) * len(args.instances) * len(args.partitions)
    print(f"Dataset: {args.dataset}  ({instance_dir})")
    print(f"Sizes: {sizes}")
    print(f"Total runs: {n_runs}")
    print(f"Writing to: {csv_path}\n")

    start = time.time()
    for n_nodes in sizes:
        for inst_id in args.instances:
            for partition in args.partitions:
                label = f"{n_nodes}_{inst_id:02d} {partition:11s}"
                try:
                    res = run_pipeline_bzr(
                        load_constantino_instance, instance_dir, n_nodes, inst_id,
                        n_players=args.n_players,
                        L_max=args.l_max,
                        partition_strategy=partition,
                        bzr_time_limit=args.time_limit,
                        verbose=args.verbose,
                        config={"gurobi_threads": args.threads},
                    )
                except Exception as e:
                    failed.append({"label": label, "error": str(e)})
                    print(f"  {label}: EXCEPTION  {e}")
                    continue
                if "error" in res:
                    failed.append({"label": label, "error": res["error"]})
                    print(f"  {label}: ERROR  {res['error']}")
                    continue
                res["dataset"] = args.dataset
                res["n_nodes"]     = n_nodes
                res["instance_id"] = inst_id
                res["partition"]   = partition
                results.append(res)
                pd.DataFrame(results).to_csv(csv_path, index=False)
                triv = "  TRIVIAL" if res["POS"] == 1.0 else ""
                print(
                    f"  {label}: |C|={res['n_cycles']:5d}, "
                    f"tri={res['n_trilateral']:4d}, "
                    f"SW={res['sw_phi']:3d}, bestPNE={res['best_pne_phi']:3d}, "
                    f"worstPNE={res['worst_pne_phi']:3d}, BSW={res['bsw_phi']:3d}, "
                    f"POS={res['POS']:.4f}, PIR={res['PIR']:.4f}, "
                    f"mode={res['ref_mode']:9s}, t={res['wall_seconds']:5.1f}s{triv}"
                )

    elapsed = time.time() - start
    print(f"\nDone in {elapsed/60:.1f} min. Completed={len(results)}, Failed={len(failed)}.")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
