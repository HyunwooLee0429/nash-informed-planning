"""Deterministic fixed-list groupings for Minnesota EBMC BC tailored variants.

These fixed lists let any driver script or notebook reproduce the manuscript's
BC_1 (opt-out), BC_2 (size-limited), and BC_3 / BC3_coal (predetermined
groupings) results bit-for-bit without seed-dependent randomness.

Usage:
    from nash_informed.ebmc import (
        MN_NON_COALITIONS, MN_BC3_COAL1, MN_BC3_COAL2, get_bc3_noncoal,
    )

Notes:
- `MN_NON_COALITIONS` is the BC_1 fixed list (10 counties opting out).
- `MN_BC3_COAL1`, `MN_BC3_COAL2` are the two predetermined coalitions used
  by the BC3_coal (BC_3) variant. `BC3_noncoal` is derived from these and
  the current `counties_playing` set via `get_bc3_noncoal(...)`.
- BC3_coal in its literature definition carries no retention constraint;
  the Nash-informed extension (pooling + IR) keeps `min_retention=0` for
  BC_3 by default. BC_1 uses `min_retention=2`.
"""

from __future__ import annotations


# BC_1 (Option 1): opt-out counties.
MN_NON_COALITIONS: list[str] = [
    "chisago",
    "scott",
    "anoka",
    "meeker",
    "douglas",
    "ramsey",
    "otter tail",
    "saint louis",
    "waseca",
    "wadena",
]

# BC_3 / BC3_coal predetermined groupings.
MN_BC3_COAL1: list[str] = ["ramsey", "washington", "anoka"]
MN_BC3_COAL2: list[str] = [
    "crow wing", "hennepin", "stearns", "carver", "wright", "cass",
]


def get_bc3_noncoal(counties_playing) -> list:
    """Compute BC3_noncoal = counties_playing - (BC3_coal1 ∪ BC3_coal2).

    Returns the list of playing counties that are neither in BC3_coal1 nor
    BC3_coal2 (each acts alone in coalition group 0).
    """
    coal = set(MN_BC3_COAL1 + MN_BC3_COAL2)
    return [c for c in counties_playing if c not in coal]


# Default BC variants registry — convenience for drivers.
BC_VARIANTS: dict[str, dict] = {
    "standard": {},                                # no tailored constraints
    "BC_1": {                                       # opt-out + minimum retention
        "non_coalitions": MN_NON_COALITIONS,
        "min_retention": 2,
    },
    "BC_2": {                                       # size-limited
        "max_num_groups": 10,
        "max_group_size": 10,
    },
    # BC_3 (BC3_coal) needs counties_playing at call time -> see helper below
}


def bc3_coal_kwargs(counties_playing) -> dict:
    """Return solve_bc kwargs for the BC3_coal (BC_3) variant on a given
    counties_playing list. Pairs `non_coalitions=BC3_noncoal` (each alone)
    with `fixed_groupings={1: BC3_coal1, 2: BC3_coal2}` and `min_retention=0`."""
    return {
        "non_coalitions": get_bc3_noncoal(counties_playing),
        "min_retention": 0,
        "fixed_groupings": {1: list(MN_BC3_COAL1), 2: list(MN_BC3_COAL2)},
    }
