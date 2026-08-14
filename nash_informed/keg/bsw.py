"""Bounded Social Welfare (BSW) for KEG.

KEG's BSW is structurally the same model as SW plus IR lower bounds. The
KEG SW solver (`solve_sw` in `model.py`) already accepts an `ir_ref` keyword
argument; this module provides a thin wrapper with a clearer name for callers.
"""

from __future__ import annotations

from .model import solve_sw


def solve_bsw(
    cycle_info: dict,
    players: list,
    partition: dict,
    ir_ref: dict,
    config: dict | None = None,
):
    """Solve KEG BSW: SW maximization with per-player IR lower bounds.

    Parameters
    ----------
    cycle_info  : output of categorize_cycles
    players     : iterable of player ids
    partition   : dict[node -> player]
    ir_ref      : dict[player -> reference utility lower bound]
    config      : optional override of DEFAULT_CONFIG

    Returns the same result dict as solve_sw, or None if infeasible.
    """
    if ir_ref is None:
        raise ValueError("solve_bsw requires ir_ref; use solve_sw for unconstrained SW.")
    return solve_sw(cycle_info, players, partition, config=config, ir_ref=ir_ref)
