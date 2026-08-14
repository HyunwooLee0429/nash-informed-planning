"""Constantino et al. (2013) KEP instance loader.

Format
------
`{N}_{XX}.input.gz`
    Header: `<n_nodes> <n_edges>`.
    Rows: `<from> <to> <weight>` (weights are all 1 in the dataset).
    A sentinel row `-1 -1 -1` may appear; it is silently skipped.

`{N}_{XX}-type-information.input.gz` (optional - absent for some small sizes)
    Header: `PAIR PATIENT DONOR WIFE-P? %PRA IN-DEG OUT-DEG`.
    Per-pair attributes (blood types, %PRA, wife-pregnancy flag).

Bundled datasets (under ``nash_informed/keg/datasets/``):
    ``small/`` - sizes 10..100, 50 instances/size.
    ``big/``   - sizes 70, 100, 200, 300, 400, 500; 10 instances/size.

Use :func:`bundled_dataset_dir` (or the ``dataset`` keyword on the loader) to
resolve these without hard-coding paths.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import networkx as nx


_DATASETS_ROOT = Path(__file__).parent / "datasets"
_BUNDLED_DATASETS = {
    "small": _DATASETS_ROOT / "small",
    "big": _DATASETS_ROOT / "big",
}


def bundled_dataset_dir(name: str) -> Path:
    """Return the absolute path of a bundled KEG dataset directory.

    Parameters
    ----------
    name : {'small', 'big'}
    """
    if name not in _BUNDLED_DATASETS:
        raise ValueError(
            f"Unknown bundled dataset {name!r}; choose from {list(_BUNDLED_DATASETS)}"
        )
    return _BUNDLED_DATASETS[name]


def list_bundled_sizes(name: str) -> list:
    """List available sizes in a bundled dataset."""
    d = bundled_dataset_dir(name)
    sizes = set()
    for f in d.glob("*.input.gz"):
        stem = f.name
        if "-type-information" in stem:
            continue
        try:
            sizes.add(int(stem.split("_")[0]))
        except ValueError:
            continue
    return sorted(sizes)


def list_bundled_instance_ids(name: str, n_nodes: int) -> list:
    """List instance ids present for a given size in a bundled dataset."""
    d = bundled_dataset_dir(name)
    ids = []
    for f in d.glob(f"{n_nodes}_*.input.gz"):
        if "-type-information" in f.name:
            continue
        try:
            ids.append(int(f.stem.split(".")[0].split("_")[1]))
        except (ValueError, IndexError):
            continue
    return sorted(ids)


def _pra_band(pra: float, thresh: tuple = (0.25, 0.80)) -> str:
    """Map a numeric %PRA value to a coarse band (low / med / high)."""
    if pra < thresh[0]:
        return "low"
    if pra < thresh[1]:
        return "med"
    return "high"


def load_constantino_instance(
    instance_dir=None,
    n_nodes: int = 0,
    instance_id: int = 0,
    *,
    dataset: str | None = None,
):
    """Load a Constantino et al. (2013) KEP instance.

    Parameters
    ----------
    instance_dir : str | Path | None
        Directory containing the instance files. Ignored if ``dataset`` is given.
    n_nodes : int
        Declared instance size (must match the file header).
    instance_id : int
        Instance number within size (1..50 for ``small``, 1..10 for ``big``).
    dataset : {'small', 'big'} | None
        If given, resolves ``instance_dir`` from the bundled datasets under
        ``nash_informed/keg/datasets/``.

    Returns
    -------
    G : networkx.DiGraph
        Compatibility graph; nodes 0..n_nodes-1. Node attrs populated only
        when the type-information file is present.
    pair_info : dict[int, dict]
        Per-pair attributes (empty dicts when type-info missing).
    """
    if dataset is not None:
        instance_dir = bundled_dataset_dir(dataset)
    if instance_dir is None:
        raise ValueError("Provide either `instance_dir` or `dataset`.")
    instance_dir = Path(instance_dir)
    stem = f"{n_nodes}_{instance_id:02d}"

    graph_file = instance_dir / f"{stem}.input.gz"
    type_file = instance_dir / f"{stem}-type-information.input.gz"
    if not graph_file.exists():
        raise FileNotFoundError(f"Missing graph file for {stem} in {instance_dir}")

    pair_info: dict = {i: {} for i in range(n_nodes)}
    if type_file.exists():
        with gzip.open(type_file, "rt") as f:
            _ = f.readline()
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                idx = int(parts[0])
                pair_info[idx] = {
                    "patient_blood": parts[1],
                    "donor_blood": parts[2],
                    "wife": int(parts[3]),
                    "patient_pra_raw": float(parts[4]),
                    "patient_pra": _pra_band(float(parts[4])),
                }

    G = nx.DiGraph()
    for i in range(n_nodes):
        G.add_node(i, **pair_info[i])
    with gzip.open(graph_file, "rt") as f:
        first = f.readline().strip().split()
        n_decl = int(first[0])
        if n_decl != n_nodes:
            raise ValueError(f"Header says n={n_decl} but expected {n_nodes}")
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            u, v = int(parts[0]), int(parts[1])
            if u < 0 or v < 0 or u >= n_nodes or v >= n_nodes:
                continue
            G.add_edge(u, v)
    return G, pair_info
