# nash_informed

Nash-informed planning for integer programming games (IPGs). A unified
Python package containing the solvers, BZR algorithm, and BSW pipeline used
in the MSOM manuscript for three game classes:

- **`nash_informed.ebmc`** — Edge-weighted Budgeted Maximum Coverage (Minnesota AIS) — *planned*
- **`nash_informed.keg`** — Kidney Exchange Game (Constantino et al. 2013 instances)
- **`nash_informed.kpg`** — Knapsack Problem Game (type A/B/C instances) — *planned*

Game-agnostic code lives in **`nash_informed.common`** (config, Gurobi helpers,
manuscript §3.3 reference-utility construction).

## Install

From the repository root:

```bash
pip install -e .
```

Or just keep the repo root on `PYTHONPATH`; the driver scripts/notebooks
auto-prepend it via `sys.path`.

## Quick start (KEG)

```python
from nash_informed.keg import (
    load_constantino_instance, run_pipeline_bzr, bundled_dataset_dir,
)

result = run_pipeline_bzr(
    load_constantino_instance,
    bundled_dataset_dir('small'), 60, 1,   # dataset dir, |V|, instance id
    n_players=3,
    L_max=3,
    partition_strategy='blood_type',
    bzr_time_limit=600,
)
print(f"POS={result['POS']:.4f}, PIR={result['PIR']:.4f}")
```

For a full sweep over many instances see:
- `scripts/run_keg.py` (CLI)
- `notebooks/keg.ipynb` (mirror notebook)

## Package layout

```
nash_informed/
├── common/
│   ├── config.py              # DEFAULT_CONFIG (seed, Gurobi limits)
│   ├── gurobi_utils.py        # set_gurobi_params
│   └── reference_utility.py   # construct_reference_utility (§3.3)
├── keg/
│   ├── instances.py           # load_constantino_instance
│   ├── model.py               # enumerate_cycles, partition_pairs, categorize_cycles,
│   │                          # solve_sw, best_response
│   ├── bzr.py                 # build_master_keg, make_bzr_callback, find_pne_bzr
│   ├── bsw.py                 # solve_bsw (wraps solve_sw with ir_ref)
│   └── pipeline.py            # run_pipeline_bzr
├── ebmc/                      # planned
└── kpg/                       # planned
```
