"""End-to-end Nash-informed planning pipeline for EBMC.

Workflow (mirrors the MSOM manuscript and the KEG pipeline):
    1. Load an EBMCInstance.
    2. Solve SW (no IR) — baseline welfare.
    3. Run BZR max + BZR min; dedupe collected PNE set.
    4. Construct IR reference utility per manuscript §3.3
       (welfare-best PNE if individually dominant; else per-county average;
        fallback to min-over-collected if BSW infeasible).
    5. Solve BSW (and optionally BC / BGC / GC) with IR lower bounds.
    6. Report POS = SW / bestPNE, POA = SW / worstPNE,
              PIR = SW / BSW, PI_BC = (BC - SW) / SW, PI_BGC = (BGC - SW) / SW.
"""

from __future__ import annotations

import time
from typing import Iterable

from ..common.config import DEFAULT_CONFIG
from ..common.reference_utility import construct_reference_utility
from .instances import EBMCInstance
from .model import compute_utility
from .bzr import find_pne_bzr
from .bsw import solve_gc, solve_bgc, solve_bsw, solve_bc


def _pne_records_to_simple(pne_records: list) -> list:
    """Convert BZR-style PNE records to the (u_player, phi)-only format expected by
    common.reference_utility.construct_reference_utility."""
    return [{"u_player": r["u_player"], "phi": r["phi"]} for r in pne_records]


def run_pipeline_bzr(
    instance: EBMCInstance,
    only_playing: bool = True,
    bzr_time_limit: int = 600,
    bsw_time_limit: int | None = None,
    include_bc: bool = False,
    include_bgc: bool = False,
    include_gc: bool = False,
    bc_kwargs: dict | None = None,
    config: dict | None = None,
    verbose: bool = False,
) -> dict:
    """Run the full Nash-informed pipeline on a single EBMCInstance.

    Parameters
    ----------
    instance        : EBMCInstance (load_minnesota_instance / load_random_ebmc_csv / generator).
    only_playing    : if True, operate on counties_playing / lakes_playing / arcs_playing
                      (default for Minnesota).
    bzr_time_limit  : seconds per BZR direction (max + min). Default 3600s.
    bsw_time_limit  : seconds for each downstream BSW/BC/BGC/GC solve.
                      If None, falls back to config['gurobi_time_limit'].
    include_bc/bgc/gc : optional additional models to solve alongside BSW.
    bc_kwargs       : extra keyword args forwarded to solve_bc (tailored variants).
    config          : optional override of DEFAULT_CONFIG.
    verbose         : if True, print intermediate diagnostics.

    Returns a result dict with keys:
        sw_phi, best_pne_phi, worst_pne_phi, n_pne, ref_mode,
        bsw_phi, [bc_phi, bgc_phi, gc_phi] when requested,
        POS, POA, PIR, [PI_BC, PI_BGC, PI_GC],
        bzr_max_runtime, bzr_min_runtime, bzr_max_cuts, bzr_min_cuts,
        wall_seconds, plus per-county utility vectors u_player_*.

    On failure, returns {'error': '...'}.
    """
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    if bsw_time_limit is None:
        bsw_time_limit = cfg["gurobi_time_limit"]

    counties_use = (
        instance.counties_playing if only_playing else instance.counties
    )
    t0 = time.time()

    # ----- 1. SW baseline -----
    from .model import solve_sw  # local import to avoid circular at module load
    sw_time_limit = bzr_time_limit
    if verbose:
        print(f"[pipeline] SW baseline (time_limit={sw_time_limit}s) ...")
    sw = solve_sw(instance, time_limit=sw_time_limit,
                   only_playing=only_playing, config=cfg)
    if sw is None:
        return {"error": "SW failed"}

    # ----- 2. BZR max + min -----
    if verbose:
        print(f"[pipeline] BZR max (time_limit={bzr_time_limit}s) ...")
    pne_max, info_max = find_pne_bzr(
        instance, direction="max",
        time_limit=bzr_time_limit, only_playing=only_playing,
        config=cfg, verbose=verbose,
    )
    if verbose:
        print(f"[pipeline] BZR min (time_limit={bzr_time_limit}s) ...")
    pne_min, info_min = find_pne_bzr(
        instance, direction="min",
        time_limit=bzr_time_limit, only_playing=only_playing,
        config=cfg, verbose=verbose,
    )

    # Deduplicate PNE set across directions
    seen: set = set()
    pne_records: list = []
    for r in pne_max + pne_min:
        key = tuple(sorted(r["selected_lakes"]))
        if key not in seen:
            seen.add(key)
            pne_records.append(r)

    if not pne_records:
        return {"error": "BZR found no PNE"}

    best_pne = max(pne_records, key=lambda d: d["phi"])
    worst_pne = min(pne_records, key=lambda d: d["phi"])

    # ----- 3. Reference utility (manuscript §3.3) -----
    simple = _pne_records_to_simple(pne_records)
    ref_u, ref_mode, _, _ = construct_reference_utility(
        simple, counties_use, scheme="dominant_or_average"
    )
    if verbose:
        print(f"[pipeline] reference utility mode = {ref_mode}")

    # ----- 4. BSW -----
    bsw = solve_bsw(
        instance, ir_ref=ref_u,
        time_limit=bsw_time_limit, only_playing=only_playing,
        warm_start=best_pne["selected_lakes"], config=cfg,
    )
    if bsw is None or bsw.get("status") not in {2, 9, 13}:  # OPTIMAL, TIME_LIMIT, SUBOPTIMAL
        if verbose:
            print("[pipeline] BSW infeasible under dominant_or_average; "
                  "falling back to min_pne.")
        ref_u, ref_mode, _, _ = construct_reference_utility(
            simple, counties_use, scheme="min_pne"
        )
        bsw = solve_bsw(
            instance, ir_ref=ref_u,
            time_limit=bsw_time_limit, only_playing=only_playing,
            warm_start=best_pne["selected_lakes"], config=cfg,
        )
        if bsw is None:
            return {"error": "BSW infeasible even with min_pne reference"}

    # ----- 5. Optional BC / BGC / GC -----
    bc_phi = bgc_phi = gc_phi = None
    bc_res = bgc_res = gc_res = None
    if include_bc:
        kw = dict(bc_kwargs or {})
        bc_res = solve_bc(
            instance, ir_ref=ref_u,
            time_limit=bsw_time_limit, only_playing=only_playing,
            warm_start=best_pne["selected_lakes"], config=cfg,
            **kw,
        )
        if bc_res and bc_res.get("phi") == bc_res.get("phi"):
            bc_phi = bc_res["phi"]
    if include_bgc:
        bgc_res = solve_bgc(
            instance, ir_ref=ref_u,
            time_limit=bsw_time_limit, only_playing=only_playing,
            warm_start=best_pne["selected_lakes"], config=cfg,
        )
        if bgc_res and bgc_res.get("phi") == bgc_res.get("phi"):
            bgc_phi = bgc_res["phi"]
    if include_gc:
        gc_res = solve_gc(
            instance, time_limit=bsw_time_limit,
            only_playing=only_playing, config=cfg,
        )
        if gc_res and gc_res.get("phi") == gc_res.get("phi"):
            gc_phi = gc_res["phi"]

    # ----- 6. Metrics -----
    sw_phi = sw["phi"]
    bp = best_pne["phi"]
    wp = worst_pne["phi"]
    pos = sw_phi / bp if bp > 0 else float("nan")
    poa = sw_phi / wp if wp > 0 else float("nan")
    pir = sw_phi / bsw["phi"] if bsw["phi"] > 0 else float("nan")
    pi_bc = ((bc_phi - sw_phi) / sw_phi) if bc_phi is not None and sw_phi > 0 else None
    pi_bgc = ((bgc_phi - sw_phi) / sw_phi) if bgc_phi is not None and sw_phi > 0 else None
    pi_gc = ((gc_phi - sw_phi) / sw_phi) if gc_phi is not None and sw_phi > 0 else None

    # Per-county utility at BSW solution
    u_bsw_per_county, _ = compute_utility(
        instance, bsw["selected_lakes"], only_playing=only_playing
    )

    out = {
        "dataset": instance.dataset,
        "n_counties_playing": len(counties_use),
        "n_lakes": len(instance.lakes),
        "n_arcs": len(instance.arcs),
        "total_budget": instance.total_budget,

        "sw_phi":        sw_phi,
        "best_pne_phi":  bp,
        "worst_pne_phi": wp,
        "n_pne":         len(pne_records),
        "ref_mode":      ref_mode,
        "bsw_phi":       bsw["phi"],
        "bc_phi":        bc_phi,
        "bgc_phi":       bgc_phi,
        "gc_phi":        gc_phi,

        "POS": pos,
        "POA": poa,
        "PIR": pir,
        "PI_BC":  pi_bc,
        "PI_BGC": pi_bgc,
        "PI_GC":  pi_gc,

        "bzr_max_runtime": info_max["runtime"],
        "bzr_min_runtime": info_min["runtime"],
        "bzr_max_cuts":    info_max["EI_cuts"],
        "bzr_min_cuts":    info_min["EI_cuts"],
        "bzr_max_extra_cuts": info_max["additional_cuts"],
        "bzr_min_extra_cuts": info_min["additional_cuts"],
        "wall_seconds":    time.time() - t0,

        "u_player_bsw":   u_bsw_per_county,
        "u_player_bpne":  best_pne["u_player"],
        "u_player_ref":   ref_u,

        "instance_meta":  instance.meta,
    }
    return out
