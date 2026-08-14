"""BZR (Best-response-incorporated Zero-Regret) for KPG.

Master MIP variables:
    x[p][i]     : binary, player p selects item i
    y[(p,o)][i] : binary, both p and o select item i (interaction active)

Constraints
    Budget per player; McCormick lift for y; equilibrium-inequality (EI) lazy cuts
    via callback; optional strategic-dominance and strategic-payoff inequalities
    (relevant for Type C / negative interactions).

EI cut (KPG-specific):

    sum_{k in BR_p}  profits[p][k]
  + sum_{o != p, k in BR_p}  interactions[(p, o)][k] * x[o][k]
  <= sum_k profits[p][k] * x[p][k]
   + sum_{o != p, k}  interactions[(p, o)][k] * y[(p, o)][k]

LHS = utility under deviation BR_p against master's opponent profile.
RHS = master's own utility for player p.

Direction support:
    direction='max'  -> max sum_p u_p  (welfare-best PNE)
    direction='min'  -> min sum_p u_p  (welfare-worst PNE)
"""

from __future__ import annotations

import random
import time
from itertools import chain, combinations, permutations

import gurobipy as gp
from gurobipy import GRB

from ..common.config import DEFAULT_CONFIG
from ..common.gurobi_utils import set_gurobi_params
from .instances import KPGInstance
from .model import (
    compute_utility,
    compute_player_utility,
    build_best_response_model,
    update_best_response_objective,
    generate_feasible_strategy,
)


# ---------------------------------------------------------------------------
# Master MIP builder
# ---------------------------------------------------------------------------

def build_master_kpg(
    instance: KPGInstance,
    obj_type: str = "max",
    time_limit: int | None = None,
    config: dict | None = None,
):
    """Build the BZR master MIP for KPG. Returns (model, x_vars, y_vars).

    `x_vars` is a dict[player -> dict[item -> Var]]; `y_vars` is dict[(p,o) -> dict[item -> Var]].
    """
    assert obj_type in ("max", "min")
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    if time_limit is not None:
        cfg["gurobi_time_limit"] = time_limit

    model = gp.Model(f"BZR_{obj_type}_KPG")
    set_gurobi_params(model, cfg)
    model.Params.LazyConstraints = 1

    x = {p: {} for p in instance.players}
    for p in instance.players:
        for i in instance.items:
            x[p][i] = model.addVar(vtype=GRB.BINARY, name=f"x_{p}_{i}")

    y = {a: {} for a in instance.players_interaction}
    for a in instance.players_interaction:
        for i in instance.items:
            y[a][i] = model.addVar(vtype=GRB.BINARY, name=f"y_{a[0]}_{a[1]}_{i}")

    # Knapsack budget per player
    for p in instance.players:
        model.addConstr(
            gp.quicksum(instance.weights[p][i] * x[p][i] for i in instance.items)
            <= instance.budgets[p],
            name=f"Budget_{p}",
        )

    # McCormick: y = AND(x_p, x_o)
    for a in instance.players_interaction:
        p, o = a
        for i in instance.items:
            model.addConstr(y[a][i] <= x[p][i])
            model.addConstr(y[a][i] <= x[o][i])
            model.addConstr(y[a][i] >= x[p][i] + x[o][i] - 1)

    sense = GRB.MAXIMIZE if obj_type == "max" else GRB.MINIMIZE
    model.setObjective(
        gp.quicksum(instance.profits[p][i] * x[p][i]
                     for p in instance.players for i in instance.items)
        + gp.quicksum(instance.interactions[a][i] * y[a][i]
                       for a in instance.players_interaction for i in instance.items),
        sense,
    )
    return model, x, y


# ---------------------------------------------------------------------------
# Strategic inequalities (Type C — negative-interaction strengthening cuts)
# ---------------------------------------------------------------------------

def _powerset(it):
    s = list(it)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def _minimal_neg_subset_indices(scalar: float, values: list) -> tuple | None:
    """Smallest-size subset of `values` whose scalar + subset_sum < 0; else None."""
    min_idx = None
    min_len = float("inf")
    for idx in _powerset(range(len(values))):
        s = scalar + sum(values[i] for i in idx)
        if s < 0 and len(idx) < min_len:
            min_idx = idx
            min_len = len(idx)
            if min_len == 0:
                break
    return min_idx


def strategic_inequalities(instance: KPGInstance) -> tuple[list, list]:
    """Compute strategic dominance + payoff inequalities.

    Returns
    -------
    dominance_relationships         : list of (player, item_dominant, item_dominated)
    strategic_payoff_relationships  : list of [player, item, subset_indices, subset_size]
                                       (only when category == 'C')
    """
    # Strategic dominance
    profit_interactions: dict = {p: {} for p in instance.players}
    profit_min: dict = {p: {} for p in instance.players}
    profit_max: dict = {p: {} for p in instance.players}
    items_pair = list(permutations(instance.items, 2))

    for p in instance.players:
        for k in instance.items:
            inter_vals = [instance.interactions[a][k]
                          for a in instance.players_interaction if a[0] == p]
            profit_interactions[p][k] = inter_vals
            min_val, max_val = float("inf"), float("-inf")
            for subset in _powerset(inter_vals):
                total = instance.profits[p][k] + sum(subset)
                if total < min_val: min_val = total
                if total > max_val: max_val = total
            profit_min[p][k] = min_val
            profit_max[p][k] = max_val

    dominance = []
    for p in instance.players:
        for i1, i2 in items_pair:
            if profit_min[p][i1] > profit_max[p][i2]:
                if instance.weights[p][i1] <= instance.weights[p][i2]:
                    dominance.append((p, i1, i2))

    # Strategic payoff (Type C only)
    strategic_payoff: list = []
    if instance.category == "C":
        for p in instance.players:
            for k in instance.items:
                sub = _minimal_neg_subset_indices(
                    instance.profits[p][k], profit_interactions[p][k]
                )
                if sub:
                    strategic_payoff.append([p, k, list(sub), len(sub)])
        # Adjust indices to the underlying interaction order (skip self).
        for entry in strategic_payoff:
            pivot = entry[0]
            for i in range(len(entry[2])):
                if pivot <= entry[2][i]:
                    entry[2][i] += 1
    return dominance, strategic_payoff


# ---------------------------------------------------------------------------
# Equilibrium-inequality (EI) cut helpers
# ---------------------------------------------------------------------------

def _ei_cut_lhs(instance: KPGInstance, player: int, br_items: list, x_master):
    """Constant own-profit term + sum_{o != p, k in BR} interactions[(p,o)][k] * x[o][k]."""
    const_sum = sum(instance.profits[player][k] for k in br_items)
    lhs = gp.LinExpr(const_sum)
    for (a0, a1) in instance.players_interaction:
        if a0 != player:
            continue
        for k in br_items:
            lhs += instance.interactions[(a0, a1)][k] * x_master[a1][k]
    return lhs


def _ei_cut_rhs(instance: KPGInstance, player: int, x_master, y_master):
    """Master-side player utility expression."""
    rhs = gp.quicksum(
        instance.profits[player][k] * x_master[player][k]
        for k in instance.items
    )
    for (a0, a1) in instance.players_interaction:
        if a0 != player:
            continue
        rhs += gp.quicksum(
            instance.interactions[(a0, a1)][k] * y_master[(a0, a1)][k]
            for k in instance.items
        )
    return rhs


# ---------------------------------------------------------------------------
# Pure separation callback (ZR — basic, no RR-BRD heuristic)
# ---------------------------------------------------------------------------

def make_separation_callback(
    instance: KPGInstance,
    x_master,
    y_master,
    config: dict | None = None,
    state: dict | None = None,
    verbose: bool = False,
):
    """Build the Gurobi callback for ZR (basic separation only)."""
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    if state is None:
        state = {}
    state.setdefault("EI_cuts", 0)
    state.setdefault("PNE_sets", [])
    state.setdefault("PNE_keys", set())
    state.setdefault("selfish_models", {})
    state.setdefault("start_time", time.time())
    state.setdefault("first_PNE_obj", None)
    state.setdefault("first_time", None)

    def callback(model, where):
        if where != GRB.Callback.MIPSOL:
            return

        # Extract incumbent profile
        profile = {}
        for p in instance.players:
            profile[p] = [i for i in instance.items
                           if model.cbGetSolution(x_master[p][i]) > 0.3]
        curr_selfish, curr_global = compute_utility(instance, profile)

        violated = False
        for p in instance.players:
            if p not in state["selfish_models"]:
                state["selfish_models"][p] = build_best_response_model(instance, p, cfg)
            m = state["selfish_models"][p]
            update_best_response_objective(m, instance, profile)
            # Warm start
            inc = set(profile[p])
            for k in instance.items:
                m._x[k].start = 1 if k in inc else 0
            m.optimize()
            br_val = float(m.ObjVal) if m.SolCount > 0 else 0.0

            if br_val > curr_selfish[p] + 1e-8:
                br_items = [k for k in instance.items if m._x[k].X > 0.5]
                lhs = _ei_cut_lhs(instance, p, br_items, x_master)
                rhs = _ei_cut_rhs(instance, p, x_master, y_master)
                model.cbLazy(lhs <= rhs)
                state["EI_cuts"] += 1
                violated = True

        if not violated and state["first_PNE_obj"] is None:
            state["first_PNE_obj"] = float(curr_global)
            state["first_time"] = time.time() - state["start_time"]

        if not violated:
            key = tuple(tuple(sorted(profile[p])) for p in instance.players)
            if key not in state["PNE_keys"]:
                state["PNE_keys"].add(key)
                state["PNE_sets"].append({
                    "profile": {p: list(profile[p]) for p in instance.players},
                    "u_player": dict(curr_selfish),
                    "phi": float(curr_global),
                })

    return callback, state


# ---------------------------------------------------------------------------
# RR-BRD (Random-Restart Best-Response Dynamics for KPG)
# ---------------------------------------------------------------------------

def rrr_brd(
    instance: KPGInstance,
    x_current: dict[int, list],
    selfish_models: dict | None = None,
    max_init: int = 3,
    max_iteration: int = 20,
    config: dict | None = None,
    rng: random.Random | None = None,
) -> tuple[dict, bool, int, dict]:
    """KPG Random-Restart Best-Response Dynamics.

    Returns (selected_items, pne_found, total_iterations, last_solution).
    """
    if config is None:
        config = DEFAULT_CONFIG
    if rng is None:
        rng = random
    if selfish_models is None:
        selfish_models = {}

    TOL = 1e-8
    total_iterations = 0

    selected: dict = {0: {p: list(x_current[p]) for p in instance.players}}
    solution: dict = {0: compute_utility(instance, selected[0])[0]}

    playing_seq = list(instance.players)
    rng.shuffle(playing_seq)

    for num_init in range(1, max_init + 1):
        if num_init > 1:
            x_current = generate_feasible_strategy(instance, num_init, rng=rng)
            selected = {0: {p: list(x_current[p]) for p in instance.players}}
            solution = {0: compute_utility(instance, selected[0])[0]}

        for num in range(1, max_iteration + 1):
            total_iterations += 1
            selected[num] = {}
            solution[num] = {}
            for p in instance.players:
                solution[num][p] = 0.0
                selected[num][p] = selected[num - 1][p]

            for p in playing_seq:
                if p not in selfish_models:
                    selfish_models[p] = build_best_response_model(instance, p, config)
                m = selfish_models[p]
                # Build profile from latest selections
                cur_profile = {q: selected[num][q] for q in instance.players}
                update_best_response_objective(m, instance, cur_profile)
                inc = set(selected[num][p])
                for k in instance.items:
                    m._x[k].start = 1 if k in inc else 0
                m.optimize()
                br_val = float(m.ObjVal) if m.SolCount > 0 else 0.0
                u_inc = compute_player_utility(instance, cur_profile, p)
                if br_val > u_inc + TOL:
                    selected[num][p] = [k for k in instance.items if m._x[k].X > 0.5]
                    solution[num][p] = br_val
                else:
                    solution[num][p] = u_inc

            if all(sorted(selected[num][p]) == sorted(selected[num - 1][p])
                   for p in instance.players):
                return selected, True, total_iterations, solution[num]

            playing_seq = list(instance.players)
            rng.shuffle(playing_seq)

    return selected, False, total_iterations, {p: 0.0 for p in instance.players}


# ---------------------------------------------------------------------------
# BZR callback (separation + RR-BRD primal heuristic + additional cuts)
# ---------------------------------------------------------------------------

def make_bzr_callback(
    instance: KPGInstance,
    x_master,
    y_master,
    config: dict | None = None,
    state: dict | None = None,
    verbose: bool = False,
):
    """BZR callback: EI separation + RR-BRD invocation on violation + extra cuts."""
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    if state is None:
        state = {}
    state.setdefault("EI_cuts", 0)
    state.setdefault("additional_cuts", 0)
    state.setdefault("PNE_sets", [])
    state.setdefault("PNE_keys", set())
    state.setdefault("BR_sets", {p: [] for p in instance.players})
    state.setdefault("selfish_models", {})
    state.setdefault("start_time", time.time())
    state.setdefault("first_PNE_obj", None)
    state.setdefault("first_time", None)
    state.setdefault("time_limit", cfg["gurobi_time_limit"])

    rng = random.Random(cfg.get("seed", 0))

    def callback(model, where):
        if where != GRB.Callback.MIPSOL:
            return

        profile = {
            p: [i for i in instance.items
                if model.cbGetSolution(x_master[p][i]) > 0.3]
            for p in instance.players
        }
        curr_selfish, curr_global = compute_utility(instance, profile)

        violated = False
        for p in instance.players:
            if p not in state["selfish_models"]:
                state["selfish_models"][p] = build_best_response_model(instance, p, cfg)
            m = state["selfish_models"][p]
            update_best_response_objective(m, instance, profile)
            inc = set(profile[p])
            for k in instance.items:
                m._x[k].start = 1 if k in inc else 0
            m.optimize()
            br_val = float(m.ObjVal) if m.SolCount > 0 else 0.0

            if br_val > curr_selfish[p] + 1e-8:
                br_items = [k for k in instance.items if m._x[k].X > 0.5]
                lhs = _ei_cut_lhs(instance, p, br_items, x_master)
                rhs = _ei_cut_rhs(instance, p, x_master, y_master)
                model.cbLazy(lhs <= rhs)
                state["EI_cuts"] += 1
                violated = True

        if not violated and state["first_PNE_obj"] is None:
            state["first_PNE_obj"] = float(curr_global)
            state["first_time"] = time.time() - state["start_time"]

        # Invoke RR-BRD when there's time budget remaining
        time_used = time.time() - state["start_time"]
        time_approx = (1.5 * state["first_time"]) if state["first_time"] else 200.0

        if not violated:
            ok_pne = True
            pne_profile = profile
        else:
            ok_pne = False
            if state["time_limit"] - time_used > time_approx:
                br_lakes, ok_pne, _, _ = rrr_brd(
                    instance, profile, selfish_models=state["selfish_models"],
                    max_init=cfg.get("brd_max_init", 3),
                    max_iteration=cfg.get("brd_max_iter", 20),
                    config=cfg, rng=rng,
                )
                length = len(br_lakes)
                if instance.n_players <= cfg.get("additional_cut_player_cap", 30):
                    max_extra = instance.n_players
                    extra = 0
                    for num in reversed(range(length)):
                        for p in instance.players:
                            br_items = br_lakes[num].get(p, [])
                            if br_items not in state["BR_sets"][p] and extra < max_extra:
                                lhs = _ei_cut_lhs(instance, p, br_items, x_master)
                                rhs = _ei_cut_rhs(instance, p, x_master, y_master)
                                model.cbLazy(lhs <= rhs)
                                state["additional_cuts"] += 1
                                state["BR_sets"][p].append(br_items)
                                extra += 1
                if ok_pne:
                    pne_profile = br_lakes[length - 1]

        if ok_pne:
            key = tuple(tuple(sorted(pne_profile[p])) for p in instance.players)
            if key not in state["PNE_keys"]:
                state["PNE_keys"].add(key)
                sf, gl = compute_utility(instance, pne_profile)
                state["PNE_sets"].append({
                    "profile": {p: list(pne_profile[p]) for p in instance.players},
                    "u_player": dict(sf),
                    "phi": float(gl),
                })
                # Inject as MIP solution
                for p in instance.players:
                    sel = set(pne_profile[p])
                    for i in instance.items:
                        model.cbSetSolution(x_master[p][i], 1.0 if i in sel else 0.0)
                for (a0, a1) in instance.players_interaction:
                    s0 = set(pne_profile[a0]); s1 = set(pne_profile[a1])
                    for i in instance.items:
                        model.cbSetSolution(
                            y_master[(a0, a1)][i],
                            1.0 if (i in s0 and i in s1) else 0.0,
                        )
                model.cbUseSolution()
                if state["first_PNE_obj"] is None:
                    state["first_PNE_obj"] = float(gl)
                    state["first_time"] = time.time() - state["start_time"]

    return callback, state


# ---------------------------------------------------------------------------
# Top-level drivers (ZR and BZR; max and min direction)
# ---------------------------------------------------------------------------

def _add_strategic_inequalities(model, instance, x_vars, dominance, payoff):
    """Add strategic dominance + (optional) payoff cuts to the master MIP."""
    n_dom = 0
    for (p, i1, i2) in dominance:
        # x[p][i1] >= x[p][i2]   when i1 dominates i2 for player p
        model.addConstr(x_vars[p][i1] >= x_vars[p][i2], name=f"Dom_{p}_{i1}_{i2}")
        n_dom += 1
    n_pay = 0
    for entry in payoff:
        p, k, subset, _ = entry
        # Payoff inequality: x[p][k] is suppressed when the cumulative
        # interaction sum from a subset of opponents is so negative that
        # picking k cannot be profitable. The constraint encodes: if all
        # listed others pick k, then x[p][k] = 0.
        if subset:
            others = list(instance.players_interaction)
            sub_idx = [others[idx][1] for idx in subset if idx < len(others)]
            model.addConstr(
                x_vars[p][k]
                + gp.quicksum(x_vars[o][k] for o in sub_idx)
                <= len(sub_idx),
                name=f"Payoff_{p}_{k}",
            )
            n_pay += 1
    return n_dom, n_pay


def find_pne_zr(
    instance: KPGInstance,
    direction: str = "max",
    time_limit: int | None = None,
    config: dict | None = None,
    use_strategic_inequalities: bool = True,
    strategic_player_cap: int = 15,
    verbose: bool = False,
):
    """Run ZR (basic EI separation only). Returns (pne_records, info).

    Strategic inequalities are added when ``use_strategic_inequalities=True`` and
    ``n_players <= strategic_player_cap`` (default 15).
    """
    assert direction in ("max", "min")
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    if time_limit is not None:
        cfg["gurobi_time_limit"] = time_limit

    start = time.time()
    model, x_vars, y_vars = build_master_kpg(
        instance, obj_type=direction, time_limit=time_limit, config=cfg,
    )

    n_dom = n_pay = 0
    if use_strategic_inequalities and instance.n_players <= strategic_player_cap:
        dom, pay = strategic_inequalities(instance)
        n_dom, n_pay = _add_strategic_inequalities(model, instance, x_vars, dom, pay)

    state = {"start_time": start, "time_limit": cfg["gurobi_time_limit"]}
    callback, state = make_separation_callback(
        instance, x_vars, y_vars, config=cfg, state=state, verbose=verbose,
    )
    model.optimize(callback)
    runtime = time.time() - start

    pne_records = sorted(
        state["PNE_sets"],
        key=lambda d: -d["phi"] if direction == "max" else d["phi"],
    )
    return pne_records, {
        "runtime": runtime,
        "EI_cuts": state["EI_cuts"],
        "dominance_cuts": n_dom,
        "payoff_cuts": n_pay,
        "first_PNE_obj": state["first_PNE_obj"],
        "first_time": state["first_time"],
        "status": model.Status,
        "obj_val": model.ObjVal if model.SolCount > 0 else None,
        "mipgap": model.MIPGap if model.SolCount > 0 else None,
    }


def find_pne_bzr(
    instance: KPGInstance,
    direction: str = "max",
    time_limit: int | None = None,
    config: dict | None = None,
    use_strategic_inequalities: bool = True,
    strategic_player_cap: int = 15,
    verbose: bool = False,
):
    """Run BZR (EI separation + RR-BRD heuristic + extra cuts). Same return shape
    as :func:`find_pne_zr`."""
    assert direction in ("max", "min")
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    if time_limit is not None:
        cfg["gurobi_time_limit"] = time_limit

    start = time.time()
    model, x_vars, y_vars = build_master_kpg(
        instance, obj_type=direction, time_limit=time_limit, config=cfg,
    )

    n_dom = n_pay = 0
    if use_strategic_inequalities and instance.n_players <= strategic_player_cap:
        dom, pay = strategic_inequalities(instance)
        n_dom, n_pay = _add_strategic_inequalities(model, instance, x_vars, dom, pay)

    state = {"start_time": start, "time_limit": cfg["gurobi_time_limit"]}
    callback, state = make_bzr_callback(
        instance, x_vars, y_vars, config=cfg, state=state, verbose=verbose,
    )
    model.optimize(callback)
    runtime = time.time() - start

    pne_records = sorted(
        state["PNE_sets"],
        key=lambda d: -d["phi"] if direction == "max" else d["phi"],
    )
    return pne_records, {
        "runtime": runtime,
        "EI_cuts": state["EI_cuts"],
        "additional_cuts": state["additional_cuts"],
        "dominance_cuts": n_dom,
        "payoff_cuts": n_pay,
        "first_PNE_obj": state["first_PNE_obj"],
        "first_time": state["first_time"],
        "status": model.Status,
        "obj_val": model.ObjVal if model.SolCount > 0 else None,
        "mipgap": model.MIPGap if model.SolCount > 0 else None,
    }
