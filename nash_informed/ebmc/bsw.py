"""Bounded social-welfare family of MIPs for EBMC.

Models
------
solve_gc(instance, ...)               : Grand Coalition — pool all budgets, no IR
solve_bgc(instance, ir_ref, ...)      : Bounded GC = GC + IR
solve_bsw(instance, ir_ref, ...)      : Bounded SW (each county keeps own budget) + IR
solve_bc(instance, ir_ref, ...)       : Bounded Coalition — flexible coalition assignment
                                         with optional tailored constraints.

All return a result dict with: phi, selected_lakes, runtime, mipgap, status, model,
plus 'coalition_assignment' for solve_bc.

Reference-utility constraint (IR)
---------------------------------
    sum_{a in arcs_use: a[1] in I_c[c]} t[a] * n[a] * y[a]   >=  ir_ref[c]
    for every c in counties_use.

The arc-coverage rule y[a] <= x[a[0]] + x[a[1]] is shared with the SW model.
"""

from __future__ import annotations

import time
from typing import Iterable

import gurobipy as gp
from gurobipy import GRB

from ..common.config import DEFAULT_CONFIG
from ..common.gurobi_utils import set_gurobi_params
from .instances import EBMCInstance


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _arcs_obj_and_all(instance: EBMCInstance, only_playing: bool):
    """Return (arcs_in_objective, list of all arcs) for the requested mode."""
    arcs_obj = instance.arcs_playing if only_playing else list(instance.arcs)
    return arcs_obj, list(instance.arcs)


def _counties_in_scope(instance: EBMCInstance, only_playing: bool):
    return instance.counties_playing if only_playing else instance.counties


def _add_link_constraints(model, instance, x_vars, y_vars):
    """y[a] <= x[a[0]] + x[a[1]] for every arc."""
    model.addConstrs(
        (y_vars[a] <= x_vars[a[0]] + x_vars[a[1]] for a in instance.arcs),
        name="Link",
    )


def _add_force_zero_non_playing(model, instance, x_vars):
    """When only_playing=True, force x[i] = 0 for lakes outside playing counties."""
    lakes_playing_set = set(instance.lakes_playing)
    for i in instance.lakes:
        if i not in lakes_playing_set:
            model.addConstr(x_vars[i] == 0)


def _add_ir_constraints(model, instance, y_vars, ir_ref, counties_use, arcs_obj):
    """sum_{a in arcs_obj: a[1] in I_c[c]} t*n*y[a] >= ir_ref[c], for c in counties_use."""
    Ic_sets = {c: set(instance.I_c[c]) for c in counties_use}
    model.addConstrs(
        (
            gp.quicksum(
                instance.t[a] * instance.n[a] * y_vars[a]
                for a in arcs_obj if a[1] in Ic_sets[c]
            ) >= ir_ref[c]
            for c in counties_use
        ),
        name="IR",
    )


def _apply_warm_start(x_vars, warm_start):
    """Warm-start: list of lake ids (selected) -> x[lake].start = 1 for each."""
    if not warm_start:
        return
    for lid in warm_start:
        if lid in x_vars:
            x_vars[lid].start = 1


def _selected_lakes_from_x(instance, x_vars, threshold: float = 0.5):
    return [i for i in instance.lakes if x_vars[i].X > threshold]


def _result(model, selected_lakes, t_wall, extra: dict | None = None):
    out = {
        "phi": float(model.ObjVal) if model.SolCount > 0 else float("nan"),
        "selected_lakes": selected_lakes,
        "runtime": model.Runtime,
        "wall_seconds": t_wall,
        "mipgap": model.MIPGap if model.SolCount > 0 else None,
        "status": model.Status,
        "model": model,
    }
    if extra:
        out.update(extra)
    return out


# ---------------------------------------------------------------------------
# GC — Grand Coalition (single pooled budget, no IR)
# ---------------------------------------------------------------------------

def solve_gc(
    instance: EBMCInstance,
    time_limit: int | None = None,
    only_playing: bool = True,
    config: dict | None = None,
) -> dict:
    """Grand Coalition: total budget = sum of counties' budgets, no IR."""
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    if time_limit is not None:
        cfg["gurobi_time_limit"] = time_limit
    counties_use = _counties_in_scope(instance, only_playing)
    arcs_obj, arcs_all = _arcs_obj_and_all(instance, only_playing)
    total_budget = sum(instance.county_budget[c] for c in counties_use)

    t0 = time.time()
    model = gp.Model("GC")
    set_gurobi_params(model, cfg)

    x_vars = model.addVars(instance.lakes, vtype=GRB.BINARY, name="x")
    y_vars = model.addVars(arcs_all, vtype=GRB.BINARY, name="y")

    _add_link_constraints(model, instance, x_vars, y_vars)
    if only_playing:
        model.addConstr(
            gp.quicksum(x_vars[i] for i in instance.lakes_playing) <= total_budget
        )
        _add_force_zero_non_playing(model, instance, x_vars)
    else:
        model.addConstr(gp.quicksum(x_vars[i] for i in instance.lakes) <= total_budget)

    model.setObjective(
        gp.quicksum(instance.t[a] * instance.n[a] * y_vars[a] for a in arcs_obj),
        GRB.MAXIMIZE,
    )
    model.optimize()
    selected = _selected_lakes_from_x(instance, x_vars) if model.SolCount > 0 else []
    return _result(model, selected, time.time() - t0)


# ---------------------------------------------------------------------------
# BGC — Bounded Grand Coalition (GC + IR)
# ---------------------------------------------------------------------------

def solve_bgc(
    instance: EBMCInstance,
    ir_ref: dict,
    time_limit: int | None = None,
    only_playing: bool = True,
    warm_start: Iterable | None = None,
    config: dict | None = None,
) -> dict:
    """Bounded Grand Coalition: GC + per-county IR lower bounds (ir_ref)."""
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    if time_limit is not None:
        cfg["gurobi_time_limit"] = time_limit
    counties_use = _counties_in_scope(instance, only_playing)
    arcs_obj, arcs_all = _arcs_obj_and_all(instance, only_playing)
    total_budget = sum(instance.county_budget[c] for c in counties_use)

    t0 = time.time()
    model = gp.Model("BGC")
    set_gurobi_params(model, cfg)
    model.setParam("MIPGap", cfg.get("bsw_mipgap", 1e-5))

    x_vars = model.addVars(instance.lakes, vtype=GRB.BINARY, name="x")
    y_vars = model.addVars(arcs_all, vtype=GRB.BINARY, name="y")

    _add_link_constraints(model, instance, x_vars, y_vars)
    if only_playing:
        model.addConstr(
            gp.quicksum(x_vars[i] for i in instance.lakes_playing) <= total_budget
        )
        _add_force_zero_non_playing(model, instance, x_vars)
    else:
        model.addConstr(gp.quicksum(x_vars[i] for i in instance.lakes) <= total_budget)

    _add_ir_constraints(model, instance, y_vars, ir_ref, counties_use, arcs_obj)
    _apply_warm_start(x_vars, warm_start)

    model.setObjective(
        gp.quicksum(instance.t[a] * instance.n[a] * y_vars[a] for a in arcs_obj),
        GRB.MAXIMIZE,
    )
    model.optimize()
    selected = _selected_lakes_from_x(instance, x_vars) if model.SolCount > 0 else []
    return _result(model, selected, time.time() - t0)


# ---------------------------------------------------------------------------
# BSW — Bounded Social Welfare (each county keeps own budget + IR)
# ---------------------------------------------------------------------------

def solve_bsw(
    instance: EBMCInstance,
    ir_ref: dict,
    time_limit: int | None = None,
    only_playing: bool = True,
    warm_start: Iterable | None = None,
    config: dict | None = None,
) -> dict:
    """Bounded Social Welfare: SW with per-county budget + per-county IR (ir_ref)."""
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    if time_limit is not None:
        cfg["gurobi_time_limit"] = time_limit
    counties_use = _counties_in_scope(instance, only_playing)
    arcs_obj, arcs_all = _arcs_obj_and_all(instance, only_playing)

    t0 = time.time()
    model = gp.Model("BSW")
    set_gurobi_params(model, cfg)

    x_vars = model.addVars(instance.lakes, vtype=GRB.BINARY, name="x")
    y_vars = model.addVars(arcs_all, vtype=GRB.BINARY, name="y")

    _add_link_constraints(model, instance, x_vars, y_vars)
    model.addConstrs(
        (gp.quicksum(x_vars[i] for i in instance.I_c[c]) <= instance.county_budget[c]
         for c in counties_use),
        name="Budget",
    )
    if only_playing:
        _add_force_zero_non_playing(model, instance, x_vars)

    _add_ir_constraints(model, instance, y_vars, ir_ref, counties_use, arcs_obj)
    _apply_warm_start(x_vars, warm_start)

    model.setObjective(
        gp.quicksum(instance.t[a] * instance.n[a] * y_vars[a] for a in arcs_obj),
        GRB.MAXIMIZE,
    )
    model.optimize()
    selected = _selected_lakes_from_x(instance, x_vars) if model.SolCount > 0 else []
    return _result(model, selected, time.time() - t0)


# ---------------------------------------------------------------------------
# BC — Bounded Coalition (flexible coalitional structure + IR)
# ---------------------------------------------------------------------------

def solve_bc(
    instance: EBMCInstance,
    ir_ref: dict,
    time_limit: int | None = None,
    only_playing: bool = True,
    num_groups: int | None = None,
    warm_start: Iterable | None = None,
    config: dict | None = None,
    # --- tailored-variant parameters (all optional) ---
    non_coalitions: Iterable | None = None,    # counties acting alone in group 0 (own-budget cap)
    retention_counties: Iterable | None = None,  # counties subject to the budget-retention constraint
    retention_alpha: int = 2,                  # max budget units deployable outside the own coalition
    max_num_groups: int | None = None,         # cap on # active groups (size-limited)
    max_group_size: int | None = None,         # cap on # counties per group (size-limited)
    fixed_groupings: dict | None = None,       # {group_id -> list[county]} (predetermined groupings)
) -> dict:
    """Bounded Coalition: counties choose coalitions g in {0..num_groups-1};
    each coalition has its own budget = sum of member counties' budgets; arc coverage
    requires at least one endpoint selected by some group.

    Per-county IR is enforced on the destination-county's incoming arcs.

    Tailored constraints (independent — combine the kwargs the variant needs;
    ready-made kwarg builders live in `datasets.minnesota_groups`):
      * `non_coalitions`         : listed counties act alone in group 0 with an
                                    own-budget cap; all other counties are kept out
                                    of group 0. (Opt-out, BC_1; also NVC/BC_NVC.)
      * `retention_counties` + `retention_alpha` : each listed county deploys at
                                    least (budget - alpha) of its own lakes, i.e.
                                    at most alpha units leave its coalition's
                                    selection. (BC_1 non-opt-out counties; all
                                    counties in BC_2 and BC_3.)
      * `max_num_groups`+`max_group_size` : limit on active groups and coalition
                                    cardinality. (Size-limited, BC_2.)
      * `fixed_groupings`        : dict mapping group id to the list of counties
                                    forced into that group. (Predetermined
                                    groupings, BC_3; also NVC/BC_NVC coalitions.)
    """
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    if time_limit is not None:
        cfg["gurobi_time_limit"] = time_limit

    counties_use = _counties_in_scope(instance, only_playing)
    arcs_obj, arcs_all = _arcs_obj_and_all(instance, only_playing)

    if num_groups is None:
        num_groups = len(counties_use)
    groups = range(num_groups)

    t0 = time.time()
    model = gp.Model("BC")
    set_gurobi_params(model, cfg)
    model.setParam("MIPGap", cfg.get("bc_mipgap", 1e-3))

    x_vars = model.addVars(instance.lakes, groups, vtype=GRB.BINARY, name="x")
    y_vars = model.addVars(arcs_all, vtype=GRB.BINARY, name="y")
    z_vars = model.addVars(instance.counties, groups, vtype=GRB.BINARY, name="z")

    # Single-group assignment per playing county
    model.addConstrs(
        (gp.quicksum(z_vars[c, g] for g in groups) == 1 for c in counties_use),
        name="single_group",
    )
    # x[i, g] <= z[county_of_i, g]
    model.addConstrs(
        (
            x_vars[i, g] <= z_vars[c, g]
            for g in groups
            for c in counties_use
            for i in instance.I_c[c]
        ),
        name="x_z_link",
    )
    # Arc coverage from any group's selection
    model.addConstrs(
        (
            y_vars[a] <= gp.quicksum(x_vars[a[0], g] + x_vars[a[1], g] for g in groups)
            for a in arcs_all
        ),
        name="arc_coverage",
    )
    # Coalition budget = sum of member counties' budgets
    if only_playing:
        model.addConstrs(
            (
                gp.quicksum(x_vars[i, g] for i in instance.lakes_playing)
                <= gp.quicksum(instance.county_budget[c] * z_vars[c, g] for c in counties_use)
                for g in groups
            ),
            name="coalition_budget",
        )
        for i in instance.lakes:
            if i not in set(instance.lakes_playing):
                model.addConstrs((x_vars[i, g] == 0 for g in groups), name=f"force_zero_{i}")
    else:
        model.addConstrs(
            (
                gp.quicksum(x_vars[i, g] for i in instance.lakes)
                <= gp.quicksum(instance.county_budget[c] * z_vars[c, g] for c in counties_use)
                for g in groups
            ),
            name="coalition_budget",
        )

    # IR (uses y[a] which already sums across groups)
    _add_ir_constraints(model, instance, y_vars, ir_ref, counties_use, arcs_obj)

    # ----- Tailored variants -----
    if non_coalitions is not None:
        nc_set = set(non_coalitions)
        # Counties in nc_set: act alone in group 0, capped at their own budget.
        # All other counties are kept out of group 0.
        for c in counties_use:
            if c in nc_set:
                model.addConstr(z_vars[c, 0] == 1, name=f"opt_out_z_{c}")
                model.addConstr(
                    gp.quicksum(x_vars[i, 0] for i in instance.I_c[c])
                    <= instance.county_budget[c],
                    name=f"opt_out_x_{c}",
                )
            else:
                model.addConstr(z_vars[c, 0] == 0, name=f"not_opt_out_z_{c}")

    if retention_counties is not None:
        # Budget retention: each listed county deploys at least
        # (budget - retention_alpha) units on its own lakes (summed over groups;
        # the x-z link confines the selection to the county's coalition).
        for c in retention_counties:
            model.addConstr(
                gp.quicksum(
                    x_vars[i, g] for i in instance.I_c[c] for g in groups
                ) >= instance.county_budget[c] - retention_alpha,
                name=f"retention_{c}",
            )

    if max_num_groups is not None or max_group_size is not None:
        # Auxiliary: is group g active?
        g_active = model.addVars(groups, vtype=GRB.BINARY, name="g_active")
        # g_active[g] = 1 iff some county is in group g
        model.addConstrs(
            (g_active[g] >= z_vars[c, g] for c in counties_use for g in groups),
            name="g_active_lower",
        )
        # weak upper bound (not strictly needed unless we want exact)
        model.addConstrs(
            (
                g_active[g] <= gp.quicksum(z_vars[c, g] for c in counties_use)
                for g in groups
            ),
            name="g_active_upper",
        )
        if max_num_groups is not None:
            model.addConstr(
                gp.quicksum(g_active[g] for g in groups) <= max_num_groups,
                name="cap_num_groups",
            )
        if max_group_size is not None:
            model.addConstrs(
                (
                    gp.quicksum(z_vars[c, g] for c in counties_use) <= max_group_size
                    for g in groups
                ),
                name="cap_group_size",
            )

    if fixed_groupings is not None:
        for g_id, members in fixed_groupings.items():
            for c in members:
                model.addConstr(z_vars[c, g_id] == 1, name=f"fixed_{c}_{g_id}")

    # Warm start: place warm_start lakes into group 0
    if warm_start:
        for lid in warm_start:
            if (lid, 0) in x_vars:
                x_vars[lid, 0].start = 1

    model.setObjective(
        gp.quicksum(instance.t[a] * instance.n[a] * y_vars[a] for a in arcs_obj),
        GRB.MAXIMIZE,
    )
    model.optimize()

    if model.SolCount == 0:
        return _result(model, [], time.time() - t0,
                        extra={"coalition_assignment": None})

    # Extract selected lakes (union over groups) and coalition assignment
    selected_set = set()
    selected_by_group: dict = {g: [] for g in groups}
    for i in instance.lakes:
        for g in groups:
            if x_vars[i, g].X > 0.5:
                selected_set.add(i)
                selected_by_group[g].append(i)
    selected_lakes = sorted(selected_set)

    coalition_assignment: dict = {}
    for c in counties_use:
        for g in groups:
            if z_vars[c, g].X > 0.5:
                coalition_assignment[c] = g
                break

    return _result(
        model, selected_lakes, time.time() - t0,
        extra={
            "coalition_assignment": coalition_assignment,
            "selected_by_group": selected_by_group,
        },
    )


# ---------------------------------------------------------------------------
# Convenience callback — terminate Gurobi on first feasible solution
# ---------------------------------------------------------------------------

def make_first_feasible_callback(verbose: bool = False):
    """Returns a Gurobi callback that terminates the solve as soon as one
    feasible solution is found. Useful for quickly checking BSW/BC feasibility.
    """
    def cb(model, where):
        if where == GRB.Callback.MIP:
            solcnt = model.cbGet(GRB.Callback.MIP_SOLCNT)
            if solcnt > 0:
                if verbose:
                    print("First feasible solution found, terminating.")
                model.terminate()
    return cb
