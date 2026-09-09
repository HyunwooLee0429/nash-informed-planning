"""Deterministic fixed-list groupings for the Minnesota EBMC tailored variants.

These fixed lists let any driver script or notebook reproduce the manuscript's
tailored coalition variants bit-for-bit without seed-dependent randomness:

  BC_1  (opt-out)          : counties in `MN_NON_COALITIONS` act alone in group 0
                             with an own-budget cap; every other county is subject
                             to the budget-retention constraint (alpha = 2).
  BC_2  (size-limited)     : at most 10 coalitions, each of size at most 10;
                             every county is subject to budget retention (alpha = 2).
  BC_3  (predetermined)    : counties are grouped by the four Minnesota DNR
                             administrative regions (`MN_DNR_REGIONS`); every
                             county is subject to budget retention (alpha = 2).
  NVC / BC_NVC (association-based) : two fixed multi-county coalitions
                             (`MN_NVC_COAL1`, `MN_NVC_COAL2`); all remaining
                             counties act alone in group 0 with an own-budget cap;
                             no retention constraint. NVC omits the IR constraint,
                             BC_NVC includes it.

Usage:
    from nash_informed.ebmc import bc1_kwargs, bc2_kwargs, bc3_kwargs, nvc_kwargs
    res = solve_bc(inst, ir_ref=ref_u, **bc3_kwargs(inst.counties_playing))
"""

from __future__ import annotations


# BC_1: opt-out counties.
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

# BC_3: Minnesota DNR administrative regions (group id -> member counties).
MN_DNR_REGIONS: dict[int, list[str]] = {
    1: [
        "becker", "beltrami", "cass", "clay", "clearwater", "douglas", "grant",
        "hubbard", "kittson", "lake of the woods", "mahnomen", "marshall", "norman",
        "otter tail", "pennington", "polk", "pope", "red lake", "roseau", "stevens",
        "traverse", "wadena", "wilkin",
    ],
    2: [
        "aitkin", "carlton", "cook", "crow wing", "itasca", "koochiching",
        "lake", "pine", "saint louis",
    ],
    3: [
        "anoka", "benton", "carver", "chisago", "dakota", "fillmore", "goodhue",
        "hennepin", "houston", "isanti", "kanabec", "mille lacs", "morrison", "olmsted",
        "ramsey", "scott", "sherburne", "stearns", "todd", "wabasha", "washington",
        "winona", "wright",
    ],
    4: [
        "big stone", "blue earth", "brown", "chippewa", "cottonwood", "dodge",
        "faribault", "freeborn", "jackson", "kandiyohi", "lac qui parle", "le sueur",
        "lincoln", "lyon", "martin", "mcleod", "meeker", "mower", "murray", "nicollet",
        "nobles", "pipestone", "redwood", "renville", "rice", "rock", "sibley", "steele",
        "swift", "waseca", "watonwan", "yellow medicine",
    ],
}

# NVC / BC_NVC: the two association-based coalitions.
MN_NVC_COAL1: list[str] = ["ramsey", "washington", "anoka"]
MN_NVC_COAL2: list[str] = [
    "crow wing", "hennepin", "stearns", "carver", "wright", "cass",
]


def get_nvc_noncoal(counties_playing) -> list:
    """Playing counties outside both NVC coalitions (each acts alone in group 0)."""
    coal = set(MN_NVC_COAL1 + MN_NVC_COAL2)
    return [c for c in counties_playing if c not in coal]


# ---------------------------------------------------------------------------
# solve_bc kwarg builders (one per tailored variant)
# ---------------------------------------------------------------------------

def bc1_kwargs(counties_playing, alpha: int = 2) -> dict:
    """BC_1 (opt-out): `MN_NON_COALITIONS` act alone in group 0 with an
    own-budget cap; all other playing counties are subject to budget
    retention with the given alpha."""
    nc = set(MN_NON_COALITIONS)
    return {
        "non_coalitions": MN_NON_COALITIONS,
        "retention_counties": [c for c in counties_playing if c not in nc],
        "retention_alpha": alpha,
    }


def bc2_kwargs(counties_playing, alpha: int = 2) -> dict:
    """BC_2 (size-limited): at most 10 coalitions of size at most 10; every
    playing county is subject to budget retention with the given alpha."""
    return {
        "max_num_groups": 10,
        "max_group_size": 10,
        "retention_counties": list(counties_playing),
        "retention_alpha": alpha,
    }


def bc3_kwargs(counties_playing, alpha: int = 2) -> dict:
    """BC_3 (predetermined groupings): playing counties fixed to their
    Minnesota DNR region (group ids 1-4); every playing county is subject
    to budget retention with the given alpha."""
    playing = set(counties_playing)
    return {
        "fixed_groupings": {
            region: [c for c in members if c in playing]
            for region, members in MN_DNR_REGIONS.items()
        },
        "retention_counties": list(counties_playing),
        "retention_alpha": alpha,
    }


def nvc_kwargs(counties_playing) -> dict:
    """NVC / BC_NVC (association-based): the two fixed coalitions
    (`MN_NVC_COAL1` -> group 1, `MN_NVC_COAL2` -> group 2); every other
    playing county acts alone in group 0 with an own-budget cap; no
    retention constraint. Pass `ir_ref` of zeros for NVC and the reference
    utilities for BC_NVC."""
    return {
        "non_coalitions": get_nvc_noncoal(counties_playing),
        "fixed_groupings": {1: list(MN_NVC_COAL1), 2: list(MN_NVC_COAL2)},
    }


# Registry of kwarg builders — convenience for drivers.
BC_VARIANTS: dict[str, object] = {
    "BC_1": bc1_kwargs,
    "BC_2": bc2_kwargs,
    "BC_3": bc3_kwargs,
    "NVC": nvc_kwargs,
    "BC_NVC": nvc_kwargs,
}
