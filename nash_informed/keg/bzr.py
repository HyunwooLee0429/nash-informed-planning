"""BZR (Best-response-incorporated Zero-Regret) for KEG.

Master MIP variables: x[(p,c)], z[c], z_tilde[(p,c)] (only for |P_c| >= 3).
On every MIPSOL the callback solves each player's best response and adds an
equilibrium-inequality (EI) lazy cut if a profitable deviation exists. When no
deviation is found, the profile is recorded as a certified PNE.

EI cut (KEG-specific):
    sum_{c in BR_p, internal}    w_c^p
  + sum_{c in BR_p, bilateral}   w_c^p * x[(opp, c)]
  + sum_{c in BR_p, trilateral}  w_c^p * z_tilde[(p, c)]
  <= sum_{c containing p} w_c^p * z[c]

LHS = utility under deviation BR_p against master's opponent profile.
RHS = master's own utility for player p. Must hold at every PNE.
"""

from __future__ import annotations

import time
from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB

from ..common.config import DEFAULT_CONFIG
from ..common.gurobi_utils import set_gurobi_params
from .model import best_response


# ---------------------------------------------------------------------------
# Master MIP
# ---------------------------------------------------------------------------

def build_master_keg(
    cycle_info: dict,
    players: list,
    partition: dict,
    obj_type: str = "max",
    config: dict | None = None,
):
    """Build the BZR master MIP for KEG and return (model, x, z, z_tilde) dicts."""
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)

    m = gp.Model(f"BZR_{obj_type}")
    set_gurobi_params(m, cfg)
    m.Params.LazyConstraints = 1

    x_vars: dict = {}
    z_vars: dict = {}
    z_tilde_vars: dict = {}

    for c, info in cycle_info.items():
        z_vars[c] = m.addVar(vtype=GRB.BINARY, name=f"z_{c}")
        for p in info["players"]:
            x_vars[(p, c)] = m.addVar(vtype=GRB.BINARY, name=f"x_{p}_{c}")
        # McCormick: z = AND of x over P_c
        for p in info["players"]:
            m.addConstr(z_vars[c] <= x_vars[(p, c)])
        m.addConstr(
            z_vars[c] >= gp.quicksum(x_vars[(p, c)] for p in info["players"])
            - (len(info["players"]) - 1)
        )
        # z_tilde for trilateral (or higher) cycles only
        if len(info["players"]) >= 3:
            for p in info["players"]:
                opponents = info["players"] - {p}
                z_tilde_vars[(p, c)] = m.addVar(vtype=GRB.BINARY, name=f"zt_{p}_{c}")
                for q in opponents:
                    m.addConstr(z_tilde_vars[(p, c)] <= x_vars[(q, c)])
                m.addConstr(
                    z_tilde_vars[(p, c)] >= gp.quicksum(x_vars[(q, c)] for q in opponents)
                    - (len(opponents) - 1)
                )

    # Capacity constraints
    pair_to_cycles: dict[int, list[int]] = defaultdict(list)
    for c, info in cycle_info.items():
        for v in info["cycle"]:
            pair_to_cycles[v].append(c)
    for v, cs in pair_to_cycles.items():
        m.addConstr(gp.quicksum(z_vars[c] for c in cs) <= 1)

    sense = GRB.MAXIMIZE if obj_type == "max" else GRB.MINIMIZE
    m.setObjective(
        gp.quicksum(info["length"] * z_vars[c] for c, info in cycle_info.items()),
        sense,
    )
    return m, x_vars, z_vars, z_tilde_vars


# ---------------------------------------------------------------------------
# MIPSOL callback
# ---------------------------------------------------------------------------

def make_bzr_callback(
    cycle_info: dict,
    players: list,
    partition: dict,
    x_vars,
    z_vars,
    z_tilde_vars,
    config: dict | None = None,
    state: dict | None = None,
    verbose: bool = False,
):
    """Build a Gurobi callback for BZR equilibrium-inequality separation.

    `state` (dict) accumulates:
        'EI_cuts'  : int — number of EI cuts added
        'PNE_sets' : list of dicts (each with 'x', 'z', 'u_player', 'phi')
        'PNE_keys' : set for dedup
    """
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    if state is None:
        state = {"EI_cuts": 0, "PNE_sets": [], "PNE_keys": set()}

    cycles_with_p = {
        p: [c for c, info in cycle_info.items() if p in info["players"]] for p in players
    }

    def callback(model, where):
        if where != GRB.Callback.MIPSOL:
            return
        x_curr = {k: int(round(model.cbGetSolution(v))) for k, v in x_vars.items()}
        z_curr = {c: int(round(model.cbGetSolution(v))) for c, v in z_vars.items()}

        violated_any = False
        for p in players:
            u_curr = sum(cycle_info[c]["w_p"][p] * z_curr[c] for c in cycles_with_p[p])
            x_others = {k: v for k, v in x_curr.items() if k[0] != p}
            br = best_response(p, x_others, cycle_info, partition, cfg)
            u_br = int(round(br["u"]))

            if u_br > u_curr + 1e-6:
                # BR's selections among active cycles
                active_cycles = []
                for c in cycles_with_p[p]:
                    opps = cycle_info[c]["players"] - {p}
                    if all(x_curr.get((q, c), 0) == 1 for q in opps):
                        active_cycles.append(c)
                BR_cycles = [c for c in active_cycles if br["x"].get((p, c), 0) == 1]

                lhs = gp.LinExpr()
                for c in BR_cycles:
                    info = cycle_info[c]
                    w_p = info["w_p"][p]
                    np_c = len(info["players"])
                    if np_c == 1:
                        lhs += w_p
                    elif np_c == 2:
                        q = next(iter(info["players"] - {p}))
                        lhs += w_p * x_vars[(q, c)]
                    else:
                        lhs += w_p * z_tilde_vars[(p, c)]
                rhs = gp.quicksum(cycle_info[c]["w_p"][p] * z_vars[c] for c in cycles_with_p[p])
                model.cbLazy(lhs <= rhs)
                state["EI_cuts"] += 1
                violated_any = True

        if not violated_any:
            key = tuple(sorted(z_curr.items()))
            if key not in state["PNE_keys"]:
                state["PNE_keys"].add(key)
                u_player = {
                    p: sum(cycle_info[c]["w_p"][p] * z_curr[c] for c in cycles_with_p[p])
                    for p in players
                }
                phi = sum(cycle_info[c]["length"] * z_curr[c] for c in cycle_info)
                state["PNE_sets"].append({
                    "x": dict(x_curr),
                    "z": dict(z_curr),
                    "u_player": u_player,
                    "phi": int(phi),
                })
                if verbose:
                    print(f"    [PNE found] phi={phi}")

    return callback, state


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def find_pne_bzr(
    cycle_info: dict,
    partition: dict,
    players: list,
    direction: str = "max",
    time_limit: float | None = None,
    config: dict | None = None,
    verbose: bool = False,
):
    """Run BZR; return (sorted pne_list, info_dict).

    direction='max' -> welfare-best PNE; 'min' -> welfare-worst PNE.
    Returned pne_list is sorted best-first under the chosen direction.
    """
    assert direction in ("max", "min")
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    if time_limit is not None:
        cfg["gurobi_time_limit"] = time_limit

    m, x_vars, z_vars, z_tilde_vars = build_master_keg(
        cycle_info, players, partition, obj_type=direction, config=cfg
    )
    callback, state = make_bzr_callback(
        cycle_info, players, partition, x_vars, z_vars, z_tilde_vars,
        config=cfg, verbose=verbose,
    )
    t0 = time.time()
    m.optimize(callback)
    runtime = time.time() - t0

    pne_list = list(state["PNE_sets"])
    pne_list.sort(key=lambda d: -d["phi"] if direction == "max" else d["phi"])
    if verbose:
        best_phi = pne_list[0]["phi"] if pne_list else None
        print(
            f"  BZR {direction}: status={m.Status}, runtime={runtime:.2f}s, "
            f"EI cuts={state['EI_cuts']}, PNEs={len(pne_list)}, best phi={best_phi}"
        )
    return pne_list, {
        "runtime": runtime,
        "EI_cuts": state["EI_cuts"],
        "status": m.Status,
        "obj_val": m.ObjVal if m.SolCount > 0 else None,
    }
