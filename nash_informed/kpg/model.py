"""KPG primitives: utility computations, SW MIP, single-player best response, helpers.

These functions operate on a :class:`KPGInstance` (no global variables).
A player's utility:

    u_i(x) = sum_k profits[i][k] * x^i_k
           + sum_{j != i} sum_k interactions[(i,j)][k] * x^i_k * x^j_k

The bilinear cross-term x^i_k * x^j_k is McCormick-linearized via auxiliary
y[(i,j)][k] in SW and BR formulations.
"""

from __future__ import annotations

import random
import time
from typing import Iterable

import gurobipy as gp
from gurobipy import GRB

from ..common.config import DEFAULT_CONFIG
from ..common.gurobi_utils import set_gurobi_params
from .instances import KPGInstance


# ---------------------------------------------------------------------------
# Utility computation (game-theoretic objective evaluation)
# ---------------------------------------------------------------------------

def compute_utility(
    instance: KPGInstance,
    profile: dict[int, Iterable],
) -> tuple[dict, float]:
    """Compute per-player and global utility for a strategy profile.

    `profile` maps each player to an iterable of selected item ids.
    """
    selfish_obj: dict = {p: 0.0 for p in instance.players}
    # Direct profit
    for p in instance.players:
        selfish_obj[p] += sum(instance.profits[p][i] for i in profile[p])
    # Interaction bonuses accrue to a0 only.
    for (a0, a1) in instance.players_interaction:
        items0 = set(profile[a0])
        items1 = set(profile[a1])
        common = items0 & items1
        for i in common:
            selfish_obj[a0] += instance.interactions[(a0, a1)][i]
    global_obj = sum(selfish_obj[p] for p in instance.players)
    return selfish_obj, float(global_obj)


def compute_player_utility(
    instance: KPGInstance,
    profile: dict[int, Iterable],
    player: int,
) -> float:
    """Compute a single player's utility under a profile."""
    my_items = set(profile[player])
    util = sum(instance.profits[player][i] for i in my_items)
    for (a0, a1) in instance.players_interaction:
        if a0 != player:
            continue
        other_items = set(profile[a1])
        for i in my_items & other_items:
            util += instance.interactions[(a0, a1)][i]
    return float(util)


# ---------------------------------------------------------------------------
# Social welfare MIP
# ---------------------------------------------------------------------------

def solve_sw(
    instance: KPGInstance,
    time_limit: int | None = None,
    warm_start: dict | None = None,
    config: dict | None = None,
) -> dict:
    """Build and solve the SW maximization MIP for KPG.

    Variables
        x[p][i]     : binary, player p selects item i
        y[(p,o)][i] : binary, both p and o select item i (interaction active)

    Constraints
        budget       : sum_i weights[p][i] * x[p][i] <= budgets[p]   per player p
        McCormick    : y[(p,o)][i] <= x[p][i], y[(p,o)][i] <= x[o][i],
                       y[(p,o)][i] >= x[p][i] + x[o][i] - 1

    Returns dict with keys: phi, profile, runtime, mipgap, status, model.
    """
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    if time_limit is not None:
        cfg["gurobi_time_limit"] = time_limit

    t0 = time.time()
    model = gp.Model("KPG_SW")
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

    # McCormick lift for y = AND(x_p, x_o)
    for a in instance.players_interaction:
        p, o = a
        for i in instance.items:
            model.addConstr(y[a][i] <= x[p][i])
            model.addConstr(y[a][i] <= x[o][i])
            model.addConstr(y[a][i] >= x[p][i] + x[o][i] - 1)

    # Warm start (optional MIP start): dict[player -> list of selected item ids].
    # A BSW-optimal profile is SW-feasible (SW drops the IR constraints), so it is a
    # valid incumbent; it guarantees the reported SW objective is >= that profile's
    # value and never exceeds the true SW optimum.
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
    elapsed = time.time() - t0
    return {
        "phi": float(model.ObjVal),
        "profile": profile,
        "runtime": model.Runtime,
        "wall_seconds": elapsed,
        "mipgap": model.MIPGap,
        "status": model.Status,
        "model": model,
    }


# ---------------------------------------------------------------------------
# Best response (per-player MIP) — reusable across BRD iterations
# ---------------------------------------------------------------------------

def build_best_response_model(
    instance: KPGInstance,
    player: int,
    config: dict | None = None,
) -> gp.Model:
    """Build a reusable single-player best-response MIP.

    Returns the Gurobi model with the player's own decision variables x[i]
    accessible via ``model._x``. The objective coefficient depends on opponents'
    choices and is set/updated separately via :func:`update_best_response_objective`.
    """
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)

    model = gp.Model(f"KPG_BR_{player}")
    set_gurobi_params(model, cfg)
    model.Params.MIPGap = cfg.get("br_mipgap", 1e-6)

    x = {i: model.addVar(vtype=GRB.BINARY, name=f"x_{i}") for i in instance.items}
    model.addConstr(
        gp.quicksum(instance.weights[player][i] * x[i] for i in instance.items)
        <= instance.budgets[player],
        name="Budget",
    )
    model._x = x
    model._player = player
    model.setObjective(0.0, GRB.MAXIMIZE)  # placeholder; overwritten by caller
    model.update()
    return model


def update_best_response_objective(
    model: gp.Model,
    instance: KPGInstance,
    profile: dict[int, Iterable],
) -> None:
    """Update a pre-built BR model's objective using the current profile of opponents.

    The objective absorbs opponents' choices into the linear coefficient of x[k]:
        own profits[i][k]
      + sum_{o != i, k in profile[o]} interactions[(i, o)][k]
    """
    i = model._player
    x = model._x

    obj = gp.quicksum(instance.profits[i][k] * x[k] for k in instance.items)
    for (a0, a1) in instance.players_interaction:
        if a0 != i:
            continue
        for k in profile[a1]:
            obj += instance.interactions[(a0, a1)][k] * x[k]
    model.setObjective(obj, GRB.MAXIMIZE)
    model.update()


# ---------------------------------------------------------------------------
# Random feasible strategy generation
# ---------------------------------------------------------------------------

def _generate_feasible_maximal(
    instance: KPGInstance,
    rng: random.Random,
) -> dict[int, list]:
    """Greedy random-order, weight-feasible 'maximal' fill."""
    profile: dict[int, list] = {}
    items_list = list(instance.items)
    for p in instance.players:
        remaining = instance.budgets[p]
        shuffled = items_list[:]
        rng.shuffle(shuffled)
        k = rng.randrange(len(shuffled)) if shuffled else 0
        if rng.random() < 0.5:
            seq = shuffled[k:] + shuffled[:k]
        else:
            seq = list(reversed(shuffled[k:] + shuffled[:k]))
        chosen: list = []
        for item in seq:
            w = instance.weights[p][item]
            if w <= remaining:
                chosen.append(item)
                remaining -= w
        profile[p] = chosen
    return profile


def _generate_feasible_full_support(
    instance: KPGInstance,
    rng: random.Random,
    eps: float = 1e-3,
    max_rejections: int | None = None,
) -> dict[int, list]:
    """Bernoulli sampling with rejection."""
    profile: dict[int, list] = {}
    for p in instance.players:
        B = float(instance.budgets[p])
        W = sum(float(instance.weights[p][i]) for i in instance.items) + 1e-12
        prob = max(eps, min(1.0 - eps, B / W))
        trials = 0
        while True:
            trials += 1
            chosen = [i for i in instance.items if rng.random() < prob]
            total_w = sum(float(instance.weights[p][i]) for i in chosen)
            if total_w <= B:
                profile[p] = chosen
                break
            if max_rejections is not None and trials >= max_rejections:
                profile[p] = []
                break
    return profile


def generate_feasible_strategy(
    instance: KPGInstance,
    trial_index: int,
    use_max_on_odd: bool = True,
    eps: float = 1e-3,
    max_rejections: int | None = None,
    rng: random.Random | None = None,
) -> dict[int, list]:
    """Alternating warm-start generator for RR-BRD:
      * odd trial  -> maximal (default)
      * even trial -> full-support Bernoulli with rejection
    Reverse with ``use_max_on_odd=False``.
    """
    if rng is None:
        rng = random
    odd = (trial_index % 2 == 1)
    if (odd and use_max_on_odd) or ((not odd) and (not use_max_on_odd)):
        return _generate_feasible_maximal(instance, rng)
    return _generate_feasible_full_support(instance, rng, eps=eps,
                                            max_rejections=max_rejections)
