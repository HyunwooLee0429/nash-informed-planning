"""KEG primitives: cycle enumeration, partitioning, SW / BR MIPs.

The KEG IPG is a cycle-packing game. Cycles are simple directed cycles in the
compatibility graph; a cycle "executes" iff every player whose pairs appear in
it agrees. The McCormick lift introduces `z_c = AND of x[(p, c)] over p in P_c`
so the social-welfare maximization remains a pure binary program.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import networkx as nx
import gurobipy as gp
from gurobipy import GRB

from ..common.config import DEFAULT_CONFIG
from ..common.gurobi_utils import set_gurobi_params


# ---------------------------------------------------------------------------
# Cycle enumeration, partitioning, categorization
# ---------------------------------------------------------------------------

def enumerate_cycles(G: nx.DiGraph, L_max: int) -> list[tuple]:
    """Return canonicalized simple directed cycles of length 2..L_max.

    Each returned tuple is rotated so its smallest node appears first.
    Duplicates from rotation are removed.
    """
    try:
        cycles_iter = nx.simple_cycles(G, length_bound=L_max)
    except TypeError:
        cycles_iter = (c for c in nx.simple_cycles(G) if len(c) <= L_max)

    seen: set[tuple] = set()
    out: list[tuple] = []
    for cyc in cycles_iter:
        if len(cyc) < 2 or len(cyc) > L_max:
            continue
        m = cyc.index(min(cyc))
        canon = tuple(cyc[m:] + cyc[:m])
        if canon not in seen:
            seen.add(canon)
            out.append(canon)
    return out


def partition_pairs(G: nx.DiGraph, n_players: int, strategy: str = "uniform",
                     seed: int | None = None) -> dict[int, int]:
    """Assign each node in G to one of n_players groups.

    Strategies
    ----------
    'uniform'    : random round-robin.
    'geographic' : 1D layout in [0,1]; contiguous slices.
    'blood_type' : group by patient blood type (O/A/B/AB → mod n_players).
                   Requires node attribute `patient_blood`; falls back to 'O'.
    """
    nodes = list(G.nodes())
    rng = np.random.default_rng(seed if seed is not None else DEFAULT_CONFIG["seed"])
    if strategy == "uniform":
        rng.shuffle(nodes)
        return {v: i % n_players for i, v in enumerate(nodes)}
    if strategy == "geographic":
        pos = rng.uniform(0, 1, len(nodes))
        order = [v for _, v in sorted(zip(pos, nodes))]
        per = max(1, len(nodes) // n_players)
        return {v: min(i // per, n_players - 1) for i, v in enumerate(order)}
    if strategy == "blood_type":
        bts = ["O", "A", "B", "AB"]
        bt_to_p = {bt: i % n_players for i, bt in enumerate(bts)}
        return {v: bt_to_p[G.nodes[v].get("patient_blood", "O")] for v in nodes}
    raise ValueError(f"Unknown partition strategy: {strategy}")


def categorize_cycles(cycles: list[tuple], partition: dict[int, int]) -> dict:
    """Annotate each cycle with its player-set and per-player pair counts.

    Returns a dict[cycle_index -> info_dict] where info_dict has keys
        'cycle'  : tuple of node ids
        'length' : len(cycle)
        'players': frozenset of player ids involved
        'w_p'    : dict[player -> count of player's pairs in cycle]
    """
    cycle_info: dict[int, dict] = {}
    for idx, cyc in enumerate(cycles):
        w: dict[int, int] = defaultdict(int)
        for v in cyc:
            w[partition[v]] += 1
        cycle_info[idx] = {
            "cycle": cyc,
            "length": len(cyc),
            "players": frozenset(w.keys()),
            "w_p": dict(w),
        }
    return cycle_info


# ---------------------------------------------------------------------------
# Social welfare (and BSW = SW + IR) solver
# ---------------------------------------------------------------------------

def solve_sw(
    cycle_info: dict,
    players: list,
    partition: dict[int, int],
    config: dict | None = None,
    ir_ref: dict | None = None,
):
    """Solve SW maximization. When `ir_ref` is provided, the model becomes BSW
    (IR-constrained social welfare).

    Variables: `x[(p, c)]` (player p selects cycle c), `z[c]` (cycle c executes).
    Constraints: McCormick lift for z; capacity (each pair in at most one executed cycle);
    optional IR lower bounds `u_p(x, z) >= ir_ref[p]`.

    Returns a result dict, or None if Gurobi found no feasible solution.
    """
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)

    m = gp.Model("SW" if ir_ref is None else "BSW")
    set_gurobi_params(m, cfg)

    x_vars: dict = {}
    z_vars: dict = {}
    for c, info in cycle_info.items():
        z_vars[c] = m.addVar(vtype=GRB.BINARY, name=f"z_{c}")
        for p in info["players"]:
            x_vars[(p, c)] = m.addVar(vtype=GRB.BINARY, name=f"x_{p}_{c}")
        # McCormick: z = AND of x's over P_c
        for p in info["players"]:
            m.addConstr(z_vars[c] <= x_vars[(p, c)])
        m.addConstr(
            z_vars[c] >= gp.quicksum(x_vars[(p, c)] for p in info["players"])
            - (len(info["players"]) - 1)
        )

    # Capacity: each pair in at most one executed cycle
    pair_to_cycles: dict[int, list[int]] = defaultdict(list)
    for c, info in cycle_info.items():
        for v in info["cycle"]:
            pair_to_cycles[v].append(c)
    for v, cs in pair_to_cycles.items():
        m.addConstr(gp.quicksum(z_vars[c] for c in cs) <= 1, name=f"cap_{v}")

    # Per-player utility expression
    u_expr = {
        p: gp.quicksum(cycle_info[c]["w_p"][p] * z_vars[c]
                       for c, info in cycle_info.items() if p in info["players"])
        for p in players
    }

    if ir_ref is not None:
        tol = cfg.get("ir_tolerance", 1e-6)
        for p, lb in ir_ref.items():
            m.addConstr(u_expr[p] >= lb - tol, name=f"IR_{p}")

    m.setObjective(
        gp.quicksum(info["length"] * z_vars[c] for c, info in cycle_info.items()),
        GRB.MAXIMIZE,
    )
    m.optimize()

    if m.SolCount == 0:
        return None

    x_sol = {k: int(round(v.X)) for k, v in x_vars.items()}
    z_sol = {c: int(round(v.X)) for c, v in z_vars.items()}
    u_player = {p: int(round(u_expr[p].getValue())) for p in players}
    return {
        "x": x_sol,
        "z": z_sol,
        "phi": int(round(m.ObjVal)),
        "u_player": u_player,
        "runtime": m.Runtime,
        "mipgap": m.MIPGap,
        "status": m.Status,
    }


# ---------------------------------------------------------------------------
# Best response (per-player MIP)
# ---------------------------------------------------------------------------

def best_response(
    player_id,
    x_others: dict,
    cycle_info: dict,
    partition: dict[int, int],
    config: dict | None = None,
):
    """Solve player_id's best response given opponents' fixed selections.

    A cycle is *active* for player_id iff every opponent in P_c\\{player_id}
    has x[(q, c)] == 1 in x_others. Only active cycles can contribute to BR utility;
    non-active cycles are set to 1 in the returned `x` for maximality (Carvalho 2022,
    Lemma E.2), since they cannot execute regardless and selecting them costs no capacity.

    Returns
    -------
    dict with keys 'x' (full player selection), 'u' (BR utility), 'runtime'.
    """
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)

    active_cycles: list = []
    all_my_cycles: list = []
    for c, info in cycle_info.items():
        if player_id not in info["players"]:
            continue
        all_my_cycles.append(c)
        opps = info["players"] - {player_id}
        if all(x_others.get((q, c), 0) == 1 for q in opps):
            active_cycles.append(c)

    my_pairs = [v for v in partition if partition[v] == player_id]
    pair_to_active: dict[int, list] = defaultdict(list)
    for c in active_cycles:
        info = cycle_info[c]
        for v in info["cycle"]:
            if partition[v] == player_id:
                pair_to_active[v].append(c)

    m = gp.Model(f"BR_{player_id}")
    set_gurobi_params(m, cfg)
    x = {c: m.addVar(vtype=GRB.BINARY, name=f"x_{c}") for c in active_cycles}
    for v in my_pairs:
        if pair_to_active[v]:
            m.addConstr(gp.quicksum(x[c] for c in pair_to_active[v]) <= 1)
    m.setObjective(
        gp.quicksum(cycle_info[c]["w_p"][player_id] * x[c] for c in active_cycles),
        GRB.MAXIMIZE,
    )
    m.optimize()

    out_x: dict = {}
    for c in all_my_cycles:
        if c in active_cycles:
            out_x[(player_id, c)] = int(round(x[c].X))
        else:
            out_x[(player_id, c)] = 1  # maximality fill-in
    u = float(m.ObjVal) if m.SolCount > 0 else 0.0
    return {"x": out_x, "u": u, "runtime": m.Runtime}
