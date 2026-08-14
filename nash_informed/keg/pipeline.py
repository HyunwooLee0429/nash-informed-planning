"""End-to-end Nash-informed planning pipeline for KEG.

Workflow:
    1. Load an instance.
    2. Enumerate cycles, partition pairs, categorize.
    3. Solve SW (no IR).
    4. Run BZR max and BZR min; deduplicate the collected PNE set.
    5. Construct reference utility per manuscript section 3.3 (dominant_or_average).
    6. Solve BSW with IR lower bounds; on infeasibility fall back to min_pne.
    7. Compute POS, POA, PIR; return all per-player utility vectors.
"""

from __future__ import annotations

import time

from ..common.config import DEFAULT_CONFIG
from ..common.reference_utility import construct_reference_utility
from .model import (
    enumerate_cycles,
    partition_pairs,
    categorize_cycles,
    solve_sw,
)
from .bzr import find_pne_bzr
from .bsw import solve_bsw


def run_pipeline_bzr(
    instance_loader,
    *loader_args,
    n_players: int = 3,
    L_max: int = 3,
    partition_strategy: str = "uniform",
    bzr_time_limit: int = 600,
    verbose: bool = False,
    config: dict | None = None,
    **loader_kwargs,
):
    """End-to-end BZR-based BSW pipeline for KEG.

    SW, BZR max/min, and BSW all share ``bzr_time_limit`` by default.

    Parameters
    ----------
    instance_loader   : callable returning (G, pair_info).
    *loader_args      : forwarded to instance_loader.
    n_players         : number of players to partition the pairs into.
    L_max             : cycle-length cap.
    partition_strategy: 'uniform' | 'geographic' | 'blood_type'.
    bzr_time_limit    : seconds per solve (SW, each BZR direction, and BSW).
    verbose           : if True, print intermediate diagnostics.
    config            : optional override of DEFAULT_CONFIG.
    **loader_kwargs   : forwarded to instance_loader.

    On failure, returns {'error': '...'}.
    """
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    cfg["L_max"] = L_max
    t0 = time.time()

    G, info = instance_loader(*loader_args, **loader_kwargs)
    cycles = enumerate_cycles(G, L_max)
    seed = loader_kwargs.get("seed", loader_args[-1] if loader_args else 0)
    partition = partition_pairs(G, n_players, strategy=partition_strategy, seed=seed)
    cycle_info = categorize_cycles(cycles, partition)
    players = list(range(n_players))

    n_internal = sum(1 for inf in cycle_info.values() if len(inf["players"]) == 1)
    n_bilateral = sum(1 for inf in cycle_info.values() if len(inf["players"]) == 2)
    n_trilateral = sum(1 for inf in cycle_info.values() if len(inf["players"]) == 3)

    if verbose:
        print(
            f"  |V|={G.number_of_nodes()}, |E|={G.number_of_edges()}, "
            f"|C|={len(cycles)} (internal={n_internal}, bilateral={n_bilateral}, "
            f"trilateral={n_trilateral})"
        )

    sw_cfg = dict(cfg); sw_cfg["gurobi_time_limit"] = bzr_time_limit
    if verbose:
        print(f"[pipeline] SW baseline (time_limit={bzr_time_limit}s) ...")
    sw = solve_sw(cycle_info, players, partition, config=sw_cfg, ir_ref=None)
    if sw is None:
        return {"error": "SW failed"}

    pne_max, info_max = find_pne_bzr(
        cycle_info, partition, players, direction="max",
        time_limit=bzr_time_limit, config=cfg, verbose=verbose,
    )
    pne_min, info_min = find_pne_bzr(
        cycle_info, partition, players, direction="min",
        time_limit=bzr_time_limit, config=cfg, verbose=verbose,
    )

    seen: set = set()
    pne_list: list = []
    for p in pne_max + pne_min:
        k = tuple(sorted(p["z"].items()))
        if k not in seen:
            seen.add(k)
            pne_list.append(p)
    if not pne_list:
        return {"error": "BZR found no PNE"}

    best_pne = max(pne_list, key=lambda d: d["phi"])
    worst_pne = min(pne_list, key=lambda d: d["phi"])

    ref_u, ref_mode, _, _ = construct_reference_utility(
        pne_list, players, scheme="dominant_or_average"
    )
    bsw_cfg = dict(cfg); bsw_cfg["gurobi_time_limit"] = bzr_time_limit
    bsw = solve_bsw(cycle_info, players, partition, ir_ref=ref_u, config=bsw_cfg)
    if bsw is None:
        ref_u, ref_mode, _, _ = construct_reference_utility(
            pne_list, players, scheme="min_pne"
        )
        bsw = solve_bsw(cycle_info, players, partition, ir_ref=ref_u, config=bsw_cfg)
        if bsw is None:
            return {"error": "BSW infeasible even with min"}

    sw_phi = sw["phi"]
    bp = best_pne["phi"]; wp = worst_pne["phi"]; bsw_phi = bsw["phi"]
    pos = sw_phi / bp if bp > 0 else float("nan")
    poa = sw_phi / wp if wp > 0 else float("nan")
    pir = sw_phi / bsw_phi if bsw_phi > 0 else float("nan")

    return {
        "V_size": G.number_of_nodes(),
        "E_size": G.number_of_edges(),
        "n_cycles": len(cycles),
        "n_internal": n_internal,
        "n_bilateral": n_bilateral,
        "n_trilateral": n_trilateral,
        "sw_phi": sw_phi,
        "best_pne_phi": bp,
        "worst_pne_phi": wp,
        "n_pne": len(pne_list),
        "ref_mode": ref_mode,
        "bsw_phi": bsw_phi,
        "POS": pos,
        "POA": poa,
        "PIR": pir,
        "bzr_max_cuts": info_max.get("EI_cuts", 0),
        "bzr_min_cuts": info_min.get("EI_cuts", 0),
        "bzr_max_runtime": info_max.get("runtime", 0.0),
        "bzr_min_runtime": info_min.get("runtime", 0.0),
        "wall_seconds": time.time() - t0,
        "u_player_sw":   sw.get("u_player"),
        "u_player_bpne": best_pne.get("u_player"),
        "u_player_ref":  ref_u,
        "u_player_bsw":  bsw.get("u_player"),
    }
