"""EBMC primitives: utility computations, SW MIP, single-player best response, helpers.

These functions operate on an :class:`EBMCInstance` — all data lives on the
instance, no module-level globals. Players are counties; a strategy is a set
of selected lake ids per county.

API:
    compute_utility(instance, selected_lakes, only_playing=True)
        -> (per-county dict, global float)
    compute_player_utility(instance, selected_lakes, county)
        -> float
    solve_sw(instance, time_limit=3600, only_playing=True, config=None)
        -> result dict
    build_best_response_model(instance, county, config=None)
        -> Gurobi model (reusable across BRD iterations)
    solve_best_response(model, instance, county, opponents_selection,
                        current_self_selection=None)
        -> (utility, selected_lakes for `county`, model)
    solve_non_game(instance, county, config=None)
        -> (utility, selected_lakes) — each county acts alone
    generate_feasible_strategy(instance, trial_index, use_max_on_odd=True, rng=None)
        -> dict[county -> list of lakes]
    flatten_selection(selection_by_county)
        -> flat list of lakes
    check_individual_best(instance, pne_list, only_playing=True)
        -> (is_individually_best, best_pne, worst_pne)
"""

from __future__ import annotations

import random
import time
from typing import Iterable

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from ..common.config import DEFAULT_CONFIG
from ..common.gurobi_utils import set_gurobi_params
from .instances import EBMCInstance


# ---------------------------------------------------------------------------
# Utility computation (game-theoretic objective evaluation)
# ---------------------------------------------------------------------------

def compute_utility(
    instance: EBMCInstance,
    selected_lakes: Iterable,
    only_playing: bool = True,
) -> tuple[dict, float]:
    """Compute per-county and global utility for a strategy profile.

    An arc (u, v) is *covered* iff either endpoint is in `selected_lakes`.
    Each covered arc contributes `n_uv * t_uv` to global welfare and to the
    destination county's utility (the county that owns v).
    """
    counties_use = instance.counties_playing if only_playing else instance.counties
    arcs_use = instance.arcs_playing if only_playing else instance.arcs
    selected_set = set(selected_lakes)

    selfish_obj: dict = {c: 0.0 for c in counties_use}
    global_obj = 0.0
    for (u, v) in arcs_use:
        if u in selected_set or v in selected_set:
            val = instance.n[(u, v)] * instance.t[(u, v)]
            global_obj += val
            cv = instance.lake_county[v]
            if cv in selfish_obj:
                selfish_obj[cv] += val
    return selfish_obj, float(global_obj)


def compute_player_utility(
    instance: EBMCInstance,
    selected_lakes: Iterable,
    county,
) -> float:
    """Compute a single county's utility from a strategy profile.

    Counted arcs: internal (both endpoints in the county) and incoming external
    (only destination in the county). This matches the EBMC payoff rule.
    """
    selected_set = set(selected_lakes)
    total = 0.0
    for arc in instance.arcs_c[county] + instance.arcs_plus_c[county]:
        if arc[0] in selected_set or arc[1] in selected_set:
            total += instance.n[arc] * instance.t[arc]
    return float(total)


def flatten_selection(selection_by_county: dict) -> list:
    """Flatten {county: [lake, ...]} into a single list of lakes."""
    return [lid for picks in selection_by_county.values() for lid in picks]


# ---------------------------------------------------------------------------
# Social welfare MIP
# ---------------------------------------------------------------------------

def solve_sw(
    instance: EBMCInstance,
    time_limit: int | None = None,
    only_playing: bool = True,
    config: dict | None = None,
) -> dict:
    """Build and solve the SW maximization MIP for EBMC.

    Variables
        x[i] : lake selection (binary), for i in instance.lakes
        y[a] : arc covered (binary), for a in arcs (active set chosen by `only_playing`)

    Constraints
        budget                : sum_{i in I_c[c]} x[i] <= county_budget[c], for c in playing
        arc activation        : y[a] <= x[a.origin] + x[a.dest]
        forced-zero (only_playing=False): x[i] = 0 for i not in lakes_playing

    Objective
        max sum_{a in arcs_use} t[a] * n[a] * y[a]

    Returns dict with keys: phi, selected_lakes, runtime, mipgap, status, model.
    """
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    if time_limit is not None:
        cfg["gurobi_time_limit"] = time_limit

    counties_use = instance.counties_playing if only_playing else instance.counties
    arcs_obj = instance.arcs_playing if only_playing else list(instance.arcs)

    t0 = time.time()
    model = gp.Model("SW")
    set_gurobi_params(model, cfg)

    x = model.addVars(instance.lakes, vtype=GRB.BINARY, name="x")
    y = model.addVars(instance.arcs, vtype=GRB.BINARY, name="y")

    model.addConstrs(
        (gp.quicksum(x[i] for i in instance.I_c[c]) <= instance.county_budget[c]
         for c in counties_use),
        name="Budget",
    )
    model.addConstrs(
        (y[a] <= x[a[0]] + x[a[1]] for a in instance.arcs),
        name="Link",
    )

    # When only_playing=True, force non-playing lakes to 0. This prevents the
    # optimizer from "freely" turning on non-playing lakes to cover incoming
    # arcs at zero budget cost.
    if only_playing:
        lakes_playing_set = set(instance.lakes_playing)
        for i in instance.lakes:
            if i not in lakes_playing_set:
                model.addConstr(x[i] == 0)

    model.setObjective(
        gp.quicksum(instance.t[a] * instance.n[a] * y[a] for a in arcs_obj),
        GRB.MAXIMIZE,
    )
    model.optimize()

    selected_lakes = [i for i in instance.lakes if model.SolCount > 0 and x[i].X > 0.5]
    elapsed = time.time() - t0

    return {
        "phi": float(model.ObjVal) if model.SolCount > 0 else float("nan"),
        "selected_lakes": selected_lakes,
        "runtime": model.Runtime,
        "wall_seconds": elapsed,
        "mipgap": model.MIPGap if model.SolCount > 0 else None,
        "status": model.Status,
        "model": model,
    }


# ---------------------------------------------------------------------------
# Best response (per-county MIP) — reusable across BRD iterations
# ---------------------------------------------------------------------------

def build_best_response_model(
    instance: EBMCInstance,
    county,
    config: dict | None = None,
) -> gp.Model:
    """Build a reusable single-county best-response MIP.

    The model has decision variables x[i] for *all* lakes (own + complement).
    Caller fixes opponents' lake selections via `var.lb, var.ub = 1, 1` or `0, 0`
    on the complement lakes, then calls `model.optimize()`. The arc activation
    structure references both own and complement variables.

    Stored model attributes:
        model._x             : dict[lake -> Var]
        model._relevant_arcs : list of arcs whose destination is in I_c[county]
        model._county        : county id
    """
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)

    model = gp.Model(f"BR_{county}")
    set_gurobi_params(model, cfg)
    model.Params.MIPGap = cfg.get("br_mipgap", 1e-6)

    own_and_compl = instance.I_c[county] + instance.I_c_complement[county]
    x = model.addVars(own_and_compl, vtype=GRB.BINARY, name="x")

    relevant_arcs = [a for a in instance.arcs if a[1] in set(instance.I_c[county])]
    y = model.addVars(relevant_arcs, vtype=GRB.BINARY, name="y")

    model.addConstr(
        gp.quicksum(x[i] for i in instance.I_c[county]) <= instance.county_budget[county],
        name="Budget",
    )
    model.addConstrs((y[a] <= x[a[0]] + x[a[1]] for a in relevant_arcs), name="Link")

    model.setObjective(
        gp.quicksum(instance.t[a] * instance.n[a] * y[a] for a in relevant_arcs),
        GRB.MAXIMIZE,
    )

    model._x = x
    model._relevant_arcs = relevant_arcs
    model._county = county
    model.update()
    return model


def solve_best_response(
    model: gp.Model,
    instance: EBMCInstance,
    county,
    opponents_selection: Iterable,
    current_self_selection: Iterable | None = None,
) -> tuple[float, list]:
    """Optimize a pre-built best-response model with opponents' selections fixed.

    Parameters
    ----------
    model               : built via build_best_response_model.
    instance            : same EBMCInstance the model was built from.
    county              : county id.
    opponents_selection : iterable of lakes (in I_c_complement) that opponents have selected.
    current_self_selection : optional warm start for own variables.

    Returns
    -------
    (utility, selected_lakes_for_county).
    """
    x = model._x
    opp_set = set(opponents_selection)
    for i in instance.I_c_complement[county]:
        var = x[i]
        if i in opp_set:
            var.lb, var.ub = 1, 1
        else:
            var.lb, var.ub = 0, 0
    if current_self_selection is not None:
        self_set = set(current_self_selection)
        for i in instance.I_c[county]:
            x[i].start = 1 if i in self_set else 0
    model.update()
    model.optimize()
    if model.SolCount == 0:
        return 0.0, []
    selected = [i for i in instance.I_c[county] if x[i].X > 0.5]
    return float(model.ObjVal), selected


# ---------------------------------------------------------------------------
# Non-game best response (each county alone, ignoring opponents)
# ---------------------------------------------------------------------------

def solve_non_game(
    instance: EBMCInstance,
    county,
    config: dict | None = None,
) -> tuple[float, list]:
    """Solve the non-game best response for `county`: each county acts alone.

    Variables for complement lakes are forced to 0 (opponents do nothing).
    Returns (utility, selected_lakes).
    """
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    cfg.setdefault("gurobi_time_limit", 60)

    model = gp.Model(f"NonGame_{county}")
    set_gurobi_params(model, cfg)
    model.Params.MIPGap = cfg.get("br_mipgap", 1e-6)

    own_and_compl = instance.I_c[county] + instance.I_c_complement[county]
    x = model.addVars(own_and_compl, vtype=GRB.BINARY, name="x")
    relevant_arcs = [a for a in instance.arcs if a[1] in set(instance.I_c[county])]
    y = model.addVars(relevant_arcs, vtype=GRB.BINARY, name="y")

    model.addConstr(
        gp.quicksum(x[i] for i in instance.I_c[county]) <= instance.county_budget[county]
    )
    model.addConstrs((y[a] <= x[a[0]] + x[a[1]] for a in relevant_arcs), name="Link")
    model.addConstrs((x[i] == 0 for i in instance.I_c_complement[county]), name="NoOpponent")
    model.setObjective(
        gp.quicksum(instance.t[a] * instance.n[a] * y[a] for a in relevant_arcs),
        GRB.MAXIMIZE,
    )
    model.optimize()
    if model.SolCount == 0:
        return 0.0, []
    selected = [i for i in instance.I_c[county] if x[i].X > 0.5]
    return float(model.ObjVal), selected


# ---------------------------------------------------------------------------
# Random feasible strategy generation (RR-BRD warm starts)
# ---------------------------------------------------------------------------

def generate_feasible_strategy(
    instance: EBMCInstance,
    trial_index: int,
    use_max_on_odd: bool = True,
    only_playing: bool = True,
    rng: random.Random | None = None,
) -> dict:
    """Generate a random feasible strategy profile.

    Alternates between two modes (controlled by trial parity):
      * 'maximal'      : each county picks exactly min(BG, |I_c|) lakes.
      * 'full-support' : each county picks b ~ Uniform{0, ..., min(BG, |I_c|)} lakes.

    `use_max_on_odd=True` (default): odd trial -> maximal, even trial -> full-support.

    Returns {county: [chosen_lake_ids]}.
    """
    if rng is None:
        rng = random
    counties_use = instance.counties_playing if only_playing else instance.counties

    sel: dict = {}
    odd = (trial_index % 2 == 1)
    maximal_turn = (odd and use_max_on_odd) or ((not odd) and not use_max_on_odd)
    for c in counties_use:
        lakes_c = instance.I_c[c]
        Bp = min(int(instance.county_budget[c]), len(lakes_c))
        b = Bp if maximal_turn else rng.randint(0, Bp)
        sel[c] = rng.sample(lakes_c, b) if b > 0 else []
    return sel


# ---------------------------------------------------------------------------
# PNE collection analysis
# ---------------------------------------------------------------------------

def check_individual_best(
    instance: EBMCInstance,
    pne_list: list,
    only_playing: bool = True,
) -> tuple[bool, list, list]:
    """For a list of PNEs (each = list/set of selected lakes), determine whether
    the welfare-best PNE is also individually best for every county.

    Returns
    -------
    (is_individually_best, best_pne, worst_pne).
    """
    if not pne_list:
        return False, [], []

    counties_use = instance.counties_playing if only_playing else instance.counties

    evaluated = []
    for pne in pne_list:
        selfish_obj, global_obj = compute_utility(instance, pne, only_playing=only_playing)
        evaluated.append({"pne": pne, "selfish": selfish_obj, "global": global_obj})

    evaluated.sort(key=lambda d: -d["global"])  # best first
    best = evaluated[0]
    worst = evaluated[-1]

    is_best_individually_best = True
    for c in counties_use:
        max_c = max(e["selfish"][c] for e in evaluated)
        if best["selfish"][c] < max_c - 1e-9:
            is_best_individually_best = False
            break

    return is_best_individually_best, best["pne"], worst["pne"]


# ---------------------------------------------------------------------------
# Solution extraction helper
# ---------------------------------------------------------------------------

def print_solution(model: gp.Model, x_vars, lake_ids, threshold: float = 0.1) -> list:
    """Return list of lakes whose x-variable is set in the model's incumbent."""
    if model.SolCount == 0:
        return []
    return [i for i in lake_ids if x_vars[i].X > threshold]
