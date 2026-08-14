"""Bounded Social Welfare (BSW) for KPG.

KPG BSW maximizes total social welfare subject to per-player IR lower bounds.
Parallels the EBMC and KEG BSW patterns.

Variables and constraints share the SW model's structure
(see :func:`nash_informed.kpg.model.solve_sw`): per-player knapsack budgets,
McCormick lift for y, and the bilinear-via-y SW objective.

The IR constraint per player p:

    sum_k profits[p][k] * x[p][k]
  + sum_{o != p, k} interactions[(p, o)][k] * y[(p, o)][k]
    >= ir_ref[p]
"""

from __future__ import annotations

import time

import gurobipy as gp
from gurobipy import GRB

from ..common.config import DEFAULT_CONFIG
from ..common.gurobi_utils import set_gurobi_params
from .instances import KPGInstance


def solve_bsw(
    instance: KPGInstance,
    ir_ref: dict,
    time_limit: int | None = None,
    warm_start: dict | None = None,
    config: dict | None = None,
) -> dict:
    """Solve KPG BSW: SW maximization with per-player IR lower bounds.

    Parameters
    ----------
    instance   : KPGInstance.
    ir_ref     : dict[player -> reference utility lower bound].
    time_limit : seconds.
    warm_start : optional dict[player -> list of selected item ids].
    config     : optional config override.

    Returns dict with: phi, profile, runtime, mipgap, status, model.
    None if infeasible.
    """
    if ir_ref is None:
        raise ValueError("solve_bsw requires ir_ref; use solve_sw for unconstrained SW.")
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    if time_limit is not None:
        cfg["gurobi_time_limit"] = time_limit

    t0 = time.time()
    model = gp.Model("KPG_BSW")
    set_gurobi_params(model, cfg)

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

    # McCormick lift for y
    for a in instance.players_interaction:
        p, o = a
        for i in instance.items:
            model.addConstr(y[a][i] <= x[p][i])
            model.addConstr(y[a][i] <= x[o][i])
            model.addConstr(y[a][i] >= x[p][i] + x[o][i] - 1)

    # IR constraints
    tol = cfg.get("ir_tolerance", 1e-6)
    for p in instance.players:
        u_p = gp.quicksum(instance.profits[p][k] * x[p][k] for k in instance.items)
        for (a0, a1) in instance.players_interaction:
            if a0 != p:
                continue
            u_p += gp.quicksum(
                instance.interactions[(a0, a1)][k] * y[(a0, a1)][k]
                for k in instance.items
            )
        model.addConstr(u_p >= ir_ref[p] - tol, name=f"IR_{p}")

    # Warm start
    if warm_start:
        for p in instance.players:
            sel = set(warm_start.get(p, []))
            for i in instance.items:
                x[p][i].start = 1 if i in sel else 0

    model.setObjective(
        gp.quicksum(instance.profits[p][i] * x[p][i]
                     for p in instance.players for i in instance.items)
        + gp.quicksum(instance.interactions[a][i] * y[a][i]
                       for a in instance.players_interaction for i in instance.items),
        GRB.MAXIMIZE,
    )
    model.optimize()
    if model.SolCount == 0:
        return None

    profile = {p: [i for i in instance.items if x[p][i].X > 0.5]
                for p in instance.players}
    return {
        "phi": float(model.ObjVal),
        "profile": profile,
        "runtime": model.Runtime,
        "wall_seconds": time.time() - t0,
        "mipgap": model.MIPGap,
        "status": model.Status,
        "model": model,
    }
