"""KPG (Knapsack Potential Game) instance container, parser, and generator.

Instance file format (matches Dragotto & Scatamacchia 2023 KP generator format):
    Line 1:    "<n_players> <n_items>"
    Line 2:    "<budget_0> <budget_1> ... <budget_{n-1}>"
    Lines 3+:  one per item j:
               "<j> <profit_0> <weight_0> <profit_1> <weight_1> ... <profit_{n-1}> <weight_{n-1}> "
               "<interaction[(0,1)]> <interaction[(0,2)]> ... <interaction[(n-1, n-2)]>"
               where interactions enumerate over permutations(players, 2) in fixed order.

Interaction categories (per generator):
    'A' : potential game (each player has a single shared interaction value across items / pairs)
    'B' : independent positive interactions (Uniform{1..R})
    'C' : negative interactions (Uniform{-R/5..0}) — strategic complements

The bundled dataset lives under ``nash_informed/kpg/datasets/type_{A,B,C}/`` and
was generated with `R = 100`, `m = 100`, `n ∈ {2,3,5,8,10,15,20,25,30}`,
budget ratios in {0.2, 0.5, 0.8}.

Public API:
    KPGInstance              : dataclass bundle
    load_kpg_instance(path)  : parse a .txt instance file
    load_kpg_bundled(category, n_players, ratio_pct=20, n_items=100)
                             : load from the bundled dataset by parameters
    list_bundled_instances(category=None) : enumerate available bundled files
    generate_kpg_instances(...) : write new instances to disk
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from itertools import permutations
from pathlib import Path


# Bundled dataset location (relative to this file)
_BUNDLED_DIR = Path(__file__).parent / "datasets"


@dataclass
class KPGInstance:
    """All derived data for a single KPG game.

    Attributes:
        n_players          : number of players
        n_items            : number of items
        players            : list[int] = list(range(n_players))
        items              : list[int] = list(range(n_items))
        players_interaction: list of (i, j) tuples == list(permutations(players, 2))
        budgets            : list[float], one per player
        profits            : dict[player -> dict[item -> int]]
        weights            : dict[player -> dict[item -> int]]
        interactions       : dict[(i, j) -> dict[item -> int]]
        category           : 'A' | 'B' | 'C' | None
        meta               : provenance dict
    """
    n_players: int
    n_items: int
    players: list
    items: list
    players_interaction: list
    budgets: list
    profits: dict
    weights: dict
    interactions: dict
    category: str | None = None
    meta: dict = field(default_factory=dict)

    @property
    def total_budget(self) -> float:
        return sum(self.budgets)

    def summary(self) -> str:
        cat = self.category or "?"
        return (
            f"KPGInstance(category={cat}, n_players={self.n_players}, "
            f"n_items={self.n_items}, "
            f"total_budget={self.total_budget:.0f}, "
            f"n_interactions={len(self.players_interaction)})"
        )


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def load_kpg_instance(path: str | Path, category: str | None = None) -> KPGInstance:
    """Parse a KPG instance file. ``category`` is recorded in meta; pass 'A'/'B'/'C'
    when known."""
    path = Path(path)
    with open(path, "r") as f:
        lines = f.readlines()

    n_players, n_items = map(int, lines[0].split())
    items = list(range(n_items))
    players = list(range(n_players))
    players_interaction = list(permutations(players, 2))
    budgets = list(map(float, lines[1].split()))

    profits = {p: {} for p in players}
    weights = {p: {} for p in players}
    interactions = {a: {} for a in players_interaction}

    for line in lines[2:]:
        data = list(map(int, line.split()))
        if not data:
            continue
        item = data[0]
        for p in players:
            profits[p][item] = data[2 * p + 1]
            weights[p][item] = data[2 * p + 2]
        offset = 2 * n_players + 1
        for idx, action in enumerate(players_interaction):
            interactions[action][item] = data[offset + idx]

    # Infer category from filename if not provided
    if category is None:
        name = path.name
        if name.endswith("-pot.txt"):
            category = "A"
        elif name.endswith("-cij.txt"):
            category = "B"
        elif name.endswith("-cij-sn.txt"):
            category = "C"

    return KPGInstance(
        n_players=n_players,
        n_items=n_items,
        players=players,
        items=items,
        players_interaction=players_interaction,
        budgets=budgets,
        profits=profits,
        weights=weights,
        interactions=interactions,
        category=category,
        meta={"file": str(path)},
    )


# ---------------------------------------------------------------------------
# Bundled dataset accessors
# ---------------------------------------------------------------------------

_RATIO_LABEL = {2: 2, 5: 5, 8: 8}  # canonical labels in filenames
_CATEGORY_TO_LABEL = {"A": "pot", "B": "cij", "C": "cij-sn"}
_LABEL_TO_CATEGORY = {v: k for k, v in _CATEGORY_TO_LABEL.items()}


def load_kpg_bundled(
    category: str,
    n_players: int,
    ratio_pct: int = 2,
    n_items: int = 100,
    dataset_dir: str | Path | None = None,
) -> KPGInstance:
    """Load a bundled KPG instance by parameters.

    Parameters
    ----------
    category   : 'A' (pot), 'B' (cij), 'C' (cij-sn)
    n_players  : number of players (must match an available file)
    ratio_pct  : 2 / 5 / 8 — budget ratio label in filename (interpreted as 20%/50%/80%)
    n_items    : number of items (bundled = 100)
    dataset_dir: override bundled location (default: ``nash_informed/kpg/datasets``)
    """
    if category not in _CATEGORY_TO_LABEL:
        raise ValueError(f"category must be 'A', 'B', or 'C', got {category!r}")
    base = Path(dataset_dir) if dataset_dir else _BUNDLED_DIR
    label = _CATEGORY_TO_LABEL[category]
    fname = f"{n_players}-{n_items}-{ratio_pct}-{label}.txt"
    path = base / f"type_{category}" / fname
    if not path.exists():
        raise FileNotFoundError(f"Bundled instance not found: {path}")
    return load_kpg_instance(path, category=category)


def list_bundled_instances(category: str | None = None,
                            dataset_dir: str | Path | None = None) -> list[dict]:
    """Enumerate bundled instance files. Returns dicts with parsed parameters."""
    base = Path(dataset_dir) if dataset_dir else _BUNDLED_DIR
    if not base.exists():
        return []
    cats = [category] if category else ["A", "B", "C"]
    out = []
    for c in cats:
        type_dir = base / f"type_{c}"
        if not type_dir.exists():
            continue
        for f in sorted(type_dir.iterdir()):
            if not f.name.endswith(".txt"):
                continue
            stem = f.stem
            parts = stem.split("-")
            try:
                n = int(parts[0]); m = int(parts[1]); r = int(parts[2])
            except (IndexError, ValueError):
                continue
            label = "-".join(parts[3:])
            out.append({
                "category": _LABEL_TO_CATEGORY.get(label, c),
                "n_players": n,
                "n_items": m,
                "ratio_pct": r,
                "path": str(f),
            })
    return out


# ---------------------------------------------------------------------------
# Generator (Dragotto & Scatamacchia 2023, with type-C reduced range)
# ---------------------------------------------------------------------------

def generate_kpg_instances(
    output_dir: str | Path,
    player_sizes: list[int] = (2, 3, 5, 8, 10, 15, 20, 25, 30),
    item_sizes: list[int] = (100,),
    budget_ratios: list[int] = (2, 5, 8),
    interaction_types: list[int] = (0, 1, 2),
    R: int = 100,
    seed: int | None = None,
) -> list[str]:
    """Generate synthetic KPG instances and write them to disk.

    Adapted from Dragotto & Scatamacchia (2023) C++ generator. For interaction
    type 2 (Type C), the interaction range is **reduced** to [-R/5, 0] (rather
    than [-R, R]) to keep large-scale instances feasible.

    Parameters
    ----------
    output_dir       : root directory (will create ``type_A/``, ``type_B/``, ``type_C/`` subdirs)
    player_sizes     : iterable of player counts
    item_sizes       : iterable of item counts
    budget_ratios    : ratio labels {2, 5, 8} interpreted as 0.2, 0.5, 0.8
    interaction_types: subset of {0, 1, 2}
    R                : range parameter for profits/weights/positive interactions
    seed             : optional RNG seed (deterministic per (n, type, m, ratio) tuple)

    Returns
    -------
    list[str] : paths of generated files.
    """
    out_root = Path(output_dir)
    files: list[str] = []
    rng = random.Random(seed) if seed is not None else random

    for n in player_sizes:
        for itype in interaction_types:
            for m in item_sizes:
                for ratio in budget_ratios:
                    sum_weights = [0] * n
                    CPotential = [rng.randint(1, R) for _ in range(n)]
                    items: list[list[int]] = []

                    for j in range(m):
                        item_row: list[int] = []
                        for i in range(n):
                            profit = rng.randint(1, R)
                            weight = rng.randint(1, R)
                            item_row.extend([profit, weight])
                            sum_weights[i] += weight
                        for p in range(n):
                            for o in range(n):
                                if p != o:
                                    if itype == 0:
                                        interaction = CPotential[p]
                                    elif itype == 1:
                                        interaction = rng.randint(1, R)
                                    elif itype == 2:
                                        interaction = -rng.randint(1, R // 5)
                                    else:
                                        raise ValueError(f"itype={itype}")
                                    item_row.append(interaction)
                        items.append(item_row)

                    label = {0: "pot", 1: "cij", 2: "cij-sn"}[itype]
                    cat = {0: "A", 1: "B", 2: "C"}[itype]
                    instance_name = f"{n}-{m}-{ratio}-{label}.txt"
                    sub = out_root / f"type_{cat}"
                    sub.mkdir(parents=True, exist_ok=True)
                    out_path = sub / instance_name

                    budgets = [round((ratio / 10) * tot) for tot in sum_weights]
                    with open(out_path, "w") as f:
                        f.write(f"{n} {m}\n")
                        f.write(" ".join(map(str, budgets)) + "\n")
                        for j, row in enumerate(items):
                            f.write(f"{j} {' '.join(map(str, row))}\n")
                    files.append(str(out_path))
    return files
