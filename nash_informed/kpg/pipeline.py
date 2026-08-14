"""End-to-end Nash-informed planning pipeline for KPG.

Workflow:
    1. Load a KPGInstance.
    2. Solve SW (no IR) - baseline.
    3. Run BZR max + BZR min; dedupe collected PNE set.
    4. Construct IR reference utility per manuscript section 3.3.
    5. Solve BSW with IR lower bounds (fallback to min_pne on infeasibility).
    6. Report POS, POA, PIR.
"""

from __future__ import annotations

import time

from ..common.config import DEFAULT_CONFIG
from ..common.reference_utility import construct_reference_utility
from .instances import KPGInstance
from .model import solve_sw, compute_utility
from .bzr import find_pne_bzr
from .bsw import solve_bsw


def _pne_to_simple(pne_records: list) -> list:
    return [{"u_player": r["u_player"], "phi": r["phi"]} for r in pne_records]


def run_pipeline_bzr(
    instance: KPGInstance,
    bzr_time_limit: int = 600,
    bsw_time_limit: int | None = None,
    use_strategic_inequalities: bool = True,
    strategic_player_cap: int = 15,
    config: dict | None = None,
    verbose: bool = False,
) -> dict:
    """Run the full BZR -> BSW pipeline on a single KPGInstance.

    SW, BZR max/min, and BSW all share ``bzr_time_limit`` by default; pass
    ``bsw_time_limit`` to override the BSW budget separately.

    Returns a result dict with keys:
        sw_phi, best_pne_phi, worst_pne_phi, n_pne, ref_mode, bsw_phi,
        POS, POA, PIR,
        bzr_max_runtime, bzr_min_runtime, bzr_max_cuts, bzr_min_cuts,
        bzr_max_extra_cuts, bzr_min_extra_cuts,
        bzr_max_dom_cuts, bzr_min_dom_cuts,
        bzr_max_payoff_cuts, bzr_min_payoff_cuts,
        wall_seconds, plus per-player utility vectors.

    On failure: {'error': '...'}.
    """
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    if bsw_time_limit is None:
        bsw_time_limit = bzr_time_limit

    t0 = time.time()

    # 1. SW baseline
    sw_time_limit = bzr_time_limit
    if verbose: print(f"[pipeline] SW baseline (time_limit={sw_time_limit}s) ...")
    sw = solve_sw(instance, time_limit=sw_time_limit, config=cfg)
    if sw is None:
        return {"error": "SW failed"}

    # 2. BZR max + min
    if verbose: print(f"[pipeline] BZR max (time_limit={bzr_time_limit}s) ...")
    pne_max, info_max = find_pne_bzr(
        instance, direction="max", time_limit=bzr_time_limit,
        config=cfg, use_strategic_inequalities=use_strategic_inequalities,
        strategic_player_cap=strategic_player_cap, verbose=verbose,
    )
    if verbose: print(f"[pipeline] BZR min (time_limit={bzr_time_limit}s) ...")
    pne_min, info_min = find_pne_bzr(
        instance, direction="min", time_limit=bzr_time_limit,
        config=cfg, use_strategic_inequalities=use_strategic_inequalities,
        strategic_player_cap=strategic_player_cap, verbose=verbose,
    )

    seen: set = set()
    pne_records: list = []
    for r in pne_max + pne_min:
        key = tuple(tuple(sorted(r["profile"][p])) for p in instance.players)
        if key not in seen:
            seen.add(key)
            pne_records.append(r)
    if not pne_records:
        return {"error": "BZR found no PNE"}

    best_pne = max(pne_records, key=lambda d: d["phi"])
    worst_pne = min(pne_records, key=lambda d: d["phi"])

    # 3. Reference utility per section 3.3
    simple = _pne_to_simple(pne_records)
    ref_u, ref_mode, _, _ = construct_reference_utility(
        simple, instance.players, scheme="dominant_or_average"
    )
    if verbose: print(f"[pipeline] reference utility mode = {ref_mode}")

    # 4. BSW
    bsw = solve_bsw(
        instance, ir_ref=ref_u,
        time_limit=bsw_time_limit,
        warm_start=best_pne["profile"], config=cfg,
    )
    if bsw is None:
        if verbose: print("[pipeline] BSW infeasible under dominant_or_average; "
                          "falling back to min_pne.")
        ref_u, ref_mode, _, _ = construct_reference_utility(
            simple, instance.players, scheme="min_pne"
        )
        bsw = solve_bsw(
            instance, ir_ref=ref_u,
            time_limit=bsw_time_limit,
            warm_start=best_pne["profile"], config=cfg,
        )
        if bsw is None:
            return {"error": "BSW infeasible even with min_pne reference"}

    # 5. Metrics
    sw_phi = sw["phi"]
    bp = best_pne["phi"]; wp = worst_pne["phi"]
    pos = sw_phi / bp if bp != 0 else float("nan")
    poa = sw_phi / wp if wp != 0 else float("nan")
    pir = sw_phi / bsw["phi"] if bsw["phi"] != 0 else float("nan")

    u_bsw_per_player, _ = compute_utility(instance, bsw["profile"])

    return {
        "category":      instance.category,
        "n_players":     instance.n_players,
        "n_items":       instance.n_items,
        "total_budget":  instance.total_budget,

        "sw_phi":        sw_phi,
        "best_pne_phi":  bp,
        "worst_pne_phi": wp,
        "n_pne":         len(pne_records),
        "ref_mode":      ref_mode,
        "bsw_phi":       bsw["phi"],

        "POS": pos, "POA": poa, "PIR": pir,

        "bzr_max_runtime": info_max["runtime"],
        "bzr_min_runtime": info_min["runtime"],
        "bzr_max_cuts":    info_max["EI_cuts"],
        "bzr_min_cuts":    info_min["EI_cuts"],
        "bzr_max_extra_cuts": info_max.get("additional_cuts", 0),
        "bzr_min_extra_cuts": info_min.get("additional_cuts", 0),
        "bzr_max_dom_cuts": info_max.get("dominance_cuts", 0),
        "bzr_min_dom_cuts": info_min.get("dominance_cuts", 0),
        "bzr_max_payoff_cuts": info_max.get("payoff_cuts", 0),
        "bzr_min_payoff_cuts": info_min.get("payoff_cuts", 0),

        "wall_seconds":  time.time() - t0,

        "u_player_bsw":  u_bsw_per_player,
        "u_player_bpne": best_pne["u_player"],
        "u_player_ref":  ref_u,
    }
