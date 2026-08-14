"""BZR (Best-response-incorporated Zero-Regret) for EBMC.

Master MIP variables
    x[i]  : lake selection (binary), for i in instance.lakes
    y[a]  : arc covered    (binary), for a in instance.arcs

Constraints
    Budget        : sum_{i in I_c[c]} x[i] <= county_budget[c]   for c in playing
    Link (max)    : y[a] <= x[a[0]] + x[a[1]]
    Link (min)    : also y[a] >= x[a[0]] and y[a] >= x[a[1]]      (forces y = OR(x_endpoints))
    Forced zero   : x[i] = 0 for i not in lakes_playing            (when only_playing=False)

Objective
    max or min  sum_{a in arcs_obj} t[a] * n[a] * y[a]

Equilibrium-inequality (EI) lazy cut, added when player p has a profitable
deviation BR_p against the current incumbent's opponent profile:

    sum_{arc in arcs_c[p]: arc[0] in BR_p or arc[1] in BR_p}  t * n
  + sum_{arc in arcs_plus_c[p]: arc[1] in BR_p}               t * n
  + sum_{arc in arcs_plus_c[p]: arc[1] not in BR_p}           t * n * x[arc[0]]
  <= sum_{arc: arc[1] in I_c[p]} t * n * y[arc]

(LHS = utility under deviation BR_p against master's opponents;
 RHS = player's master utility. Holds at every PNE.)

Additionally, when a violation is found, RRR-BRD (random-restart BRD) is
invoked to discover further PNEs; for small player counts (<=30) cuts from
the entire BRD trajectory are also added.
"""

from __future__ import annotations

import random
import time
from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB

from ..common.config import DEFAULT_CONFIG
from ..common.gurobi_utils import set_gurobi_params
from .instances import EBMCInstance
from .model import (
    compute_utility,
    compute_player_utility,
    flatten_selection,
    build_best_response_model,
    generate_feasible_strategy,
)


# ---------------------------------------------------------------------------
# Master MIP builder
# ---------------------------------------------------------------------------

def build_master_ebmc(
    instance: EBMCInstance,
    obj_type: str = "max",
    only_playing: bool = True,
    time_limit: int | None = None,
    config: dict | None = None,
):
    """Build the BZR master MIP for EBMC. Returns (model, x_vars, y_vars).

    For obj_type='min', adds tightening `y >= x[a[0]]` and `y >= x[a[1]]` so
    the minimizer cannot set y=0 when an endpoint is selected.
    """
    assert obj_type in ("max", "min")
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    if time_limit is not None:
        cfg["gurobi_time_limit"] = time_limit

    counties_use = instance.counties_playing if only_playing else instance.counties
    arcs_obj = instance.arcs_playing if only_playing else list(instance.arcs)
    arcs_all = list(instance.arcs)

    model = gp.Model(f"BZR_{obj_type}_EBMC")
    set_gurobi_params(model, cfg)
    model.Params.LazyConstraints = 1

    x_vars = model.addVars(instance.lakes, vtype=GRB.BINARY, name="x")
    y_vars = model.addVars(arcs_all, vtype=GRB.BINARY, name="y")

    model.addConstrs(
        (gp.quicksum(x_vars[i] for i in instance.I_c[c]) <= instance.county_budget[c]
         for c in counties_use),
        name="Budget",
    )
    model.addConstrs(
        (y_vars[a] <= x_vars[a[0]] + x_vars[a[1]] for a in arcs_all),
        name="Link",
    )
    if obj_type == "min":
        model.addConstrs((y_vars[a] >= x_vars[a[0]] for a in arcs_all), name="Link_min0")
        model.addConstrs((y_vars[a] >= x_vars[a[1]] for a in arcs_all), name="Link_min1")

    # Force non-playing lakes to 0 when only_playing=True.
    if only_playing:
        lakes_playing_set = set(instance.lakes_playing)
        for i in instance.lakes:
            if i not in lakes_playing_set:
                model.addConstr(x_vars[i] == 0)

    sense = GRB.MAXIMIZE if obj_type == "max" else GRB.MINIMIZE
    model.setObjective(
        gp.quicksum(instance.t[a] * instance.n[a] * y_vars[a] for a in arcs_obj),
        sense,
    )
    return model, x_vars, y_vars


# ---------------------------------------------------------------------------
# Equilibrium-inequality (EI) cut LHS expression
# ---------------------------------------------------------------------------

def _ei_cut_lhs(instance: EBMCInstance, county, best_response_lakes, x_master):
    """Build the EI cut LHS expression for a county and its best response.

    Constant part (BR-only): arcs whose coverage is determined by BR alone.
    Parameterized part: incoming external arcs with dest NOT in BR — they
    contribute t*n*x[arc[0]] (coverage depends on the other county's selection).
    """
    br_set = set(best_response_lakes)
    const_sum = 0.0
    for a in instance.arcs_c[county]:
        if a[0] in br_set or a[1] in br_set:
            const_sum += instance.t[a] * instance.n[a]
    for a in instance.arcs_plus_c[county]:
        if a[1] in br_set:
            const_sum += instance.t[a] * instance.n[a]
    lhs = gp.LinExpr(const_sum)
    for a in instance.arcs_plus_c[county]:
        if a[1] not in br_set:
            lhs += instance.t[a] * instance.n[a] * x_master[a[0]]
    return lhs


def _ei_cut_rhs(instance: EBMCInstance, county, y_master):
    """Master-side RHS expression: county's utility = sum over destination-in-I_c arcs."""
    Ic_set = set(instance.I_c[county])
    return gp.quicksum(
        instance.t[a] * instance.n[a] * y_master[a]
        for a in instance.arcs if a[1] in Ic_set
    )


# ---------------------------------------------------------------------------
# RRR-BRD (Random-Restart Best-Response Dynamics)
# ---------------------------------------------------------------------------

def rrr_brd(
    instance: EBMCInstance,
    x_current: list,
    selfish_models: dict,
    max_init: int = 3,
    max_iteration: int = 20,
    only_playing: bool = True,
    config: dict | None = None,
    rng: random.Random | None = None,
) -> tuple[dict, bool, int, dict]:
    """Random-Restart Best-Response Dynamics for EBMC.

    Returns
    -------
    selected_lakes : dict[int -> dict[county -> list]] — BRD path.
    pne_found      : True if BRD converged to a fixed point.
    total_iterations : total best-response updates performed.
    last_solution  : dict[county -> utility] at the last iteration.
    """
    if config is None:
        config = DEFAULT_CONFIG
    if rng is None:
        rng = random

    counties_use = instance.counties_playing if only_playing else instance.counties

    TOL = 1e-8
    selected_lakes: dict = {0: {}}
    for c in counties_use:
        selected_lakes[0][c] = [i for i in x_current if instance.lake_county.get(i) == c]
    init_obj, _ = compute_utility(instance, x_current, only_playing=only_playing)
    solution: dict = {0: init_obj}

    playing_sequence = list(counties_use)
    rng.shuffle(playing_sequence)
    total_iterations = 0

    for num_init in range(1, max_init + 1):
        if num_init > 1:
            sel_dict = generate_feasible_strategy(
                instance, num_init, use_max_on_odd=True,
                only_playing=only_playing, rng=rng,
            )
            x_current = flatten_selection(sel_dict)
            selected_lakes = {0: {}}
            for c in counties_use:
                selected_lakes[0][c] = [i for i in x_current if instance.lake_county.get(i) == c]
            init_obj, _ = compute_utility(instance, x_current, only_playing=only_playing)
            solution = {0: init_obj}

        for num in range(1, max_iteration + 1):
            total_iterations += 1
            selected_lakes[num] = {c: selected_lakes[num - 1][c] for c in counties_use}
            solution[num] = {c: 0.0 for c in counties_use}

            for c in playing_sequence:
                if c not in selfish_models:
                    selfish_models[c] = build_best_response_model(instance, c, config)
                m = selfish_models[c]

                current_profile_flat = [lid for cc, ls in selected_lakes[num].items() for lid in ls]
                u_inc = compute_player_utility(instance, current_profile_flat, c)

                cur_set = set(current_profile_flat)
                for i in instance.I_c_complement[c]:
                    var = m._x[i]
                    var.lb, var.ub = (1, 1) if i in cur_set else (0, 0)

                incumbent_self = selected_lakes[num][c]
                for i in instance.I_c[c]:
                    m._x[i].start = 1 if i in incumbent_self else 0

                m.update()
                m.optimize()
                u_br = float(m.ObjVal) if m.SolCount > 0 else 0.0

                if u_br > u_inc + TOL:
                    selected_lakes[num][c] = [i for i in instance.I_c[c] if m._x[i].X > 0.5]
                    solution[num][c] = u_br
                else:
                    selected_lakes[num][c] = incumbent_self
                    solution[num][c] = u_inc

            # PNE convergence check
            if selected_lakes[num] == selected_lakes[num - 1]:
                return selected_lakes, True, total_iterations, solution[num]

            playing_sequence = list(counties_use)
            rng.shuffle(playing_sequence)

    return selected_lakes, False, total_iterations, {c: 0.0 for c in counties_use}


# ---------------------------------------------------------------------------
# MIPSOL callback (EI cuts + RR-BRD primal heuristic + PNE collection)
# ---------------------------------------------------------------------------

def make_bzr_callback(
    instance: EBMCInstance,
    x_master,
    y_master,
    only_playing: bool = True,
    config: dict | None = None,
    state: dict | None = None,
    verbose: bool = False,
):
    """Build the Gurobi callback for EBMC BZR separation.

    `state` (dict) accumulates:
        'EI_cuts'         : int — number of primary EI cuts added
        'additional_cuts' : int — number of extra cuts from BRD trajectory
        'PNE_sets'        : list of PNEs (each a list of lakes)
        'first_PNE_obj'   : welfare of the first PNE encountered
        'first_time'      : wall-clock at first PNE
        'start_time'      : passed in to compute elapsed timing
        'selfish_models'  : dict[county -> Gurobi model] (cached BR models)
        'BR_sets'         : dict[county -> list of BR profiles already cut]
    """
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    if state is None:
        state = {}
    state.setdefault("EI_cuts", 0)
    state.setdefault("additional_cuts", 0)
    state.setdefault("PNE_sets", [])
    state.setdefault("first_PNE_obj", None)
    state.setdefault("first_time", None)
    state.setdefault("start_time", time.time())
    state.setdefault("selfish_models", {})
    state.setdefault("BR_sets", {})
    state.setdefault("time_limit", cfg["gurobi_time_limit"])
    counties_use = instance.counties_playing if only_playing else instance.counties
    for c in counties_use:
        state["BR_sets"].setdefault(c, [])

    rng = random.Random(cfg.get("seed", 0))

    def callback(model, where):
        if where != GRB.Callback.MIPSOL:
            return

        # Extract current incumbent
        x_current = [i for i in instance.lakes if model.cbGetSolution(x_master[i]) > 0.3]
        curr_selfish_obj, curr_global_obj = compute_utility(
            instance, x_current, only_playing=only_playing
        )

        violated = False
        for c in counties_use:
            # Build / reuse selfish model
            if c not in state["selfish_models"]:
                state["selfish_models"][c] = build_best_response_model(instance, c, cfg)
            m = state["selfish_models"][c]

            cur_set = set(x_current)
            for i in instance.I_c_complement[c]:
                var = m._x[i]
                var.lb, var.ub = (1, 1) if i in cur_set else (0, 0)
            incumbent_self = [i for i in x_current if instance.lake_county.get(i) == c]
            for i in instance.I_c[c]:
                m._x[i].start = 1 if i in incumbent_self else 0

            m.optimize()
            u_br = float(m.ObjVal) if m.SolCount > 0 else 0.0
            if u_br > curr_selfish_obj[c] + 1e-6:
                # Profitable deviation -> add EI cut
                br_lakes = [i for i in instance.I_c[c] if m._x[i].X > 0.1]
                lhs = _ei_cut_lhs(instance, c, br_lakes, x_master)
                rhs = _ei_cut_rhs(instance, c, y_master)
                model.cbLazy(lhs <= rhs)
                state["EI_cuts"] += 1
                violated = True

        if not violated and state["first_time"] is None:
            state["first_PNE_obj"] = curr_global_obj
            state["first_time"] = time.time() - state["start_time"]

        # Decide whether to invoke RR-BRD primal heuristic
        time_approx_BRD = (1.5 * state["first_time"]) if state["first_time"] is not None else 200.0
        time_used = time.time() - state["start_time"]

        if not violated:
            PNE, BR_PNE_found, current_sol = x_current, True, curr_selfish_obj
        else:
            BR_PNE_found = False
            PNE = []
            current_sol = {c: 0.0 for c in counties_use}
            if state["time_limit"] - time_used > time_approx_BRD:
                BRD_lakes, BR_PNE_found, total_iter, current_sol = rrr_brd(
                    instance, x_current, state["selfish_models"],
                    max_init=cfg.get("brd_max_init", 3),
                    max_iteration=cfg.get("brd_max_iter", 20),
                    only_playing=only_playing, config=cfg, rng=rng,
                )
                length_BRD = len(BRD_lakes)

                # Additional cuts from BRD trajectory (only for small player counts)
                if len(counties_use) <= cfg.get("additional_cut_player_cap", 30):
                    max_extra = len(instance.counties)
                    extra_added = 0
                    for num in reversed(range(length_BRD)):
                        for c in counties_use:
                            br_c = BRD_lakes[num].get(c, [])
                            if br_c not in state["BR_sets"][c] and extra_added < max_extra:
                                lhs = _ei_cut_lhs(instance, c, br_c, x_master)
                                rhs = _ei_cut_rhs(instance, c, y_master)
                                model.cbLazy(lhs <= rhs)
                                state["additional_cuts"] += 1
                                state["BR_sets"][c].append(br_c)
                                extra_added += 1

                if BR_PNE_found:
                    PNE = flatten_selection(BRD_lakes[length_BRD - 1])

        # Inject a recovered PNE as MIP start
        if BR_PNE_found and PNE not in state["PNE_sets"]:
            PNE_set = set(PNE)
            for i in instance.lakes:
                model.cbSetSolution(x_master[i], 1.0 if i in PNE_set else 0.0)
            arcs_iter = instance.arcs_playing if only_playing else instance.arcs
            for (u, v) in arcs_iter:
                model.cbSetSolution(y_master[(u, v)], 1.0 if (u in PNE_set or v in PNE_set) else 0.0)
            model.cbUseSolution()
            state["PNE_sets"].append(PNE)
            if state["first_PNE_obj"] is None:
                state["first_PNE_obj"] = (
                    sum(current_sol[c] for c in counties_use) if isinstance(current_sol, dict) else 0.0
                )
                state["first_time"] = time.time() - state["start_time"]
            if verbose:
                print(f"    [PNE injected] |PNE|={len(PNE)}")

    return callback, state


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def find_pne_bzr(
    instance: EBMCInstance,
    direction: str = "max",
    time_limit: int | None = None,
    only_playing: bool = True,
    config: dict | None = None,
    verbose: bool = False,
):
    """Run BZR on an EBMCInstance; return (pne_list, info_dict).

    `pne_list` is sorted welfare-first under the chosen direction (max -> best first;
    min -> worst first). `info_dict` reports runtime, EI/additional cut counts, first-PNE
    timing, status, and ObjVal.
    """
    assert direction in ("max", "min")
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    if time_limit is not None:
        cfg["gurobi_time_limit"] = time_limit

    start = time.time()
    model, x_master, y_master = build_master_ebmc(
        instance, obj_type=direction, only_playing=only_playing,
        time_limit=time_limit, config=cfg,
    )

    state = {"start_time": start, "time_limit": cfg["gurobi_time_limit"]}
    callback, state = make_bzr_callback(
        instance, x_master, y_master,
        only_playing=only_playing, config=cfg, state=state, verbose=verbose,
    )

    model.optimize(callback)
    runtime = time.time() - start

    # Evaluate collected PNEs and sort
    pne_records = []
    for pne in state["PNE_sets"]:
        selfish_obj, global_obj = compute_utility(
            instance, pne, only_playing=only_playing
        )
        pne_records.append({
            "selected_lakes": pne,
            "u_player": selfish_obj,
            "phi": float(global_obj),
        })
    pne_records.sort(key=lambda d: -d["phi"] if direction == "max" else d["phi"])

    info = {
        "runtime": runtime,
        "EI_cuts": state["EI_cuts"],
        "additional_cuts": state["additional_cuts"],
        "first_PNE_obj": state["first_PNE_obj"],
        "first_time": state["first_time"],
        "status": model.Status,
        "obj_val": model.ObjVal if model.SolCount > 0 else None,
        "mipgap": model.MIPGap if model.SolCount > 0 else None,
    }
    if verbose:
        best_phi = pne_records[0]["phi"] if pne_records else None
        print(
            f"  BZR {direction}: status={model.Status}, runtime={runtime:.2f}s, "
            f"EI={state['EI_cuts']}, extra={state['additional_cuts']}, "
            f"PNEs={len(pne_records)}, best phi={best_phi}"
        )
    return pne_records, info
