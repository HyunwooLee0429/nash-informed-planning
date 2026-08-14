"""Construct IR-reference utility vectors from a collected set of pure Nash equilibria.

Implements manuscript §3.3:
    (i)   single PNE (welfare-best if individually dominant)
    (ii)  average over two extremes
    (iii) average / minimum across all collected PNEs

The construction is game-agnostic; each PNE dict must have keys
'u_player' (dict[player -> int|float]) and 'phi' (scalar social welfare).
"""

from __future__ import annotations

import numpy as np


def construct_reference_utility(
    pne_list: list[dict],
    players: list,
    scheme: str = "dominant_or_average",
):
    """Build a reference utility vector from a collected PNE set.

    Parameters
    ----------
    pne_list : list of dicts, each with 'u_player' (dict[player -> value]) and 'phi'.
    players  : iterable of player identifiers.
    scheme   : selection scheme.
        - 'dominant_or_average' (manuscript §3.3 default):
              best PNE if individually dominant; else per-player mean.
        - 'best_pne'  : welfare-best PNE's utility vector (always feasible).
        - 'min_pne'   : per-player minimum across collected (most conservative).

    Returns
    -------
    (ref_utility, mode, best_pne, per_player_utilities_if_average)
        ref_utility : dict[player -> value], or None if pne_list is empty.
        mode        : 'dominant' | 'average' | 'best_pne' | 'min' | 'empty'.
        best_pne    : the welfare-best PNE dict, or None.
        per_player_utilities_if_average : list of u_player dicts when mode='average',
                                          else None.
    """
    if not pne_list:
        return None, "empty", None, None

    best = max(pne_list, key=lambda d: d["phi"])

    # Dominance check: does best PNE individually dominate every other PNE?
    dominant = True
    for x in pne_list:
        for p in players:
            if best["u_player"][p] < x["u_player"][p] - 1e-9:
                dominant = False
                break
        if not dominant:
            break

    if scheme == "dominant_or_average":
        if dominant:
            return dict(best["u_player"]), "dominant", best, None
        avg = {p: float(np.mean([x["u_player"][p] for x in pne_list])) for p in players}
        return avg, "average", best, [x["u_player"] for x in pne_list]

    if scheme == "best_pne":
        return dict(best["u_player"]), "best_pne", best, None

    if scheme == "min_pne":
        mins = {p: min(x["u_player"][p] for x in pne_list) for p in players}
        return mins, "min", best, [x["u_player"] for x in pne_list]

    raise ValueError(f"Unknown scheme: {scheme}")
