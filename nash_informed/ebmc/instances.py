"""EBMC instance loading and random generation.

The EBMC game lives on a directed boat-movement network where:
  * Nodes are *lakes*, partitioned into *counties* (the players).
  * Arc (i, j) has a *weight* n_ij (boats moved) and a *susceptibility* t_ij = b_ij
    (per-boat risk that an uninspected boat carries infestation from i to j).
  * Each county has a *budget* limiting how many lakes it can inspect.
  * Social welfare = sum over arcs (i, j) of n_ij * t_ij * y_ij, where
    y_ij = 1 iff either endpoint is inspected.

This module defines `EBMCInstance` (a frozen bundle of all derived data) plus
entry points:
    load_minnesota_instance(...)         : real Minnesota AIS dataset
    generate_random_ebmc_instance(...)   : synthetic multi-AIS random instance
    write_random_ebmc_csv(...)           : same, but serialized to bundled CSV
    load_random_ebmc_csv(...) / load_random_ebmc(...) : reload bundled CSVs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Instance container
# ---------------------------------------------------------------------------

@dataclass
class EBMCInstance:
    """All the derived data for a single EBMC game.

    Public attributes:
      counties           : list[str | int]               - all county ids (players)
      lakes              : list                           - all lake ids
      lake_county        : dict[lake -> county]
      I_c                : dict[county -> list of lakes in county]
      I_c_complement     : dict[county -> list of lakes outside county]

      arcs               : set of (i, j) tuples           - directed arcs with t > 0
      n                  : dict[arc -> int|float]         - boats per arc
      t                  : dict[arc -> int|float]         - bij coefficient per arc
      arcs_c             : dict[county -> internal arcs]
      arcs_plus_c        : dict[county -> incoming external arcs]
      arcs_minus_c       : dict[county -> outgoing external arcs]

      infested_lakes     : list of lake ids
      county_budget      : dict[county -> int]            - inspection budget
      counties_playing   : list                           - counties with budget > 0
      lakes_playing      : list                           - lakes owned by playing counties
      arcs_playing       : list of arcs                   - incoming to a playing county

      dataset            : str                            - 'minnesota' or 'random'
      meta               : dict                           - provenance / parameters
    """

    counties: list
    lakes: list
    lake_county: dict
    I_c: dict
    I_c_complement: dict

    arcs: set
    n: dict
    t: dict
    arcs_c: dict
    arcs_plus_c: dict
    arcs_minus_c: dict

    infested_lakes: list
    county_budget: dict
    counties_playing: list
    lakes_playing: list
    arcs_playing: list

    dataset: str = "unknown"
    meta: dict = field(default_factory=dict)

    @property
    def n_counties(self) -> int:
        return len(self.counties)

    @property
    def n_counties_playing(self) -> int:
        return len(self.counties_playing)

    @property
    def total_budget(self) -> int:
        return sum(self.county_budget.values())

    def summary(self) -> str:
        return (
            f"EBMCInstance(dataset={self.dataset}, "
            f"counties={len(self.counties)} (playing={len(self.counties_playing)}), "
            f"lakes={len(self.lakes)}, "
            f"infested={len(self.infested_lakes)}, "
            f"arcs={len(self.arcs)}, "
            f"total_budget={self.total_budget})"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _derive_arc_partitions(arcs, lake_county, counties):
    """Partition arcs per county into (internal, incoming, outgoing) buckets.

    Linear in |arcs| (each arc visited once) and avoids the O(|arcs| * |counties|)
    naive pattern.
    """
    arcs_c: dict = {c: [] for c in counties}
    arcs_plus_c: dict = {c: [] for c in counties}
    arcs_minus_c: dict = {c: [] for c in counties}
    for arc in arcs:
        u, v = arc
        cu = lake_county.get(u)
        cv = lake_county.get(v)
        if cu is None or cv is None:
            continue
        if cu == cv:
            arcs_c[cu].append(arc)
        else:
            arcs_plus_c[cv].append(arc)   # incoming to cv
            arcs_minus_c[cu].append(arc)  # outgoing from cu
    return arcs_c, arcs_plus_c, arcs_minus_c


def _playing_subsets(counties, county_budget, I_c, arcs):
    counties_playing = [c for c in counties if county_budget.get(c, 0) > 0]
    lakes_playing = [lake for c in counties_playing for lake in I_c[c]]
    lakes_playing_set = set(lakes_playing)
    arcs_playing = [a for a in arcs if a[1] in lakes_playing_set]
    return counties_playing, lakes_playing, arcs_playing


# ---------------------------------------------------------------------------
# Minnesota loader
# ---------------------------------------------------------------------------

# Default species columns in the Minnesota lake_info.csv
MN_SPECIES_COLUMNS = ("zm2019", "ss2019", "ew2019", "sf2019")


def load_minnesota_instance(
    data_dir: str | Path = "data",
    percentage: str = "100%",
    species_columns: tuple = MN_SPECIES_COLUMNS,
    lake_info_csv: str = "lake_info.csv",
    boats_csv: str = "boats1.csv",
    budget_xlsx: str = "Estimation.xlsx",
) -> EBMCInstance:
    """Load the Minnesota EBMC instance at the requested budget percentage.

    Parameters
    ----------
    data_dir    : directory containing lake_info.csv, boats1.csv, Estimation.xlsx
    percentage  : '100%' | '50%' | '75%' | '125%' - sheet name in Estimation.xlsx
    species_columns : AIS infestation columns in lake_info.csv (default = 4 species, 2019)
    """
    data_dir = Path(data_dir)

    df_lakes_info = pd.read_csv(data_dir / lake_info_csv)
    counties = list(df_lakes_info["county_name"].unique())
    lakes = list(df_lakes_info["DOW"].unique())
    lake_county = dict(zip(df_lakes_info["DOW"], df_lakes_info["county_name"]))

    I_c = df_lakes_info.groupby("county_name")["DOW"].apply(list).to_dict()
    lakes_set = set(lakes)
    I_c_complement = {c: list(lakes_set - set(I_c[c])) for c in counties}

    species_status = df_lakes_info.set_index("DOW")[list(species_columns)].astype(int)

    df_edge = pd.read_csv(data_dir / boats_csv)
    df_edge = df_edge.loc[df_edge["dow_origin"] != df_edge["dow_destination"]].copy()
    df_edge = (
        df_edge.groupby(["dow_origin", "dow_destination"], as_index=False)["weight"].sum()
    )

    origin_status = species_status.rename(columns=lambda s: f"o_{s}")
    dest_status = species_status.rename(columns=lambda s: f"d_{s}")
    df_edge = df_edge.merge(
        origin_status, left_on="dow_origin", right_index=True, how="inner"
    )
    df_edge = df_edge.merge(
        dest_status, left_on="dow_destination", right_index=True, how="inner"
    )

    o_cols = [f"o_{s}" for s in species_columns]
    d_cols = [f"d_{s}" for s in species_columns]
    df_edge["t"] = (
        df_edge[o_cols].to_numpy() * (1 - df_edge[d_cols].to_numpy())
    ).sum(axis=1)
    df_edge = df_edge.loc[df_edge["t"] > 0, ["dow_origin", "dow_destination", "weight", "t"]]

    arcs = set(zip(df_edge["dow_origin"], df_edge["dow_destination"]))
    n = dict(zip(zip(df_edge["dow_origin"], df_edge["dow_destination"]),
                  df_edge["weight"].tolist()))
    t = dict(zip(zip(df_edge["dow_origin"], df_edge["dow_destination"]),
                  df_edge["t"].astype(float).tolist()))

    arcs_c, arcs_plus_c, arcs_minus_c = _derive_arc_partitions(
        arcs, lake_county, counties
    )

    species_mask = df_lakes_info[list(species_columns)].sum(axis=1) > 0
    infested_lakes = list(df_lakes_info.loc[species_mask, "DOW"].unique())

    budget_df = pd.read_excel(data_dir / budget_xlsx, sheet_name=percentage, header=0)
    budget_df["Number of inspection stations"] = budget_df["Number of inspection stations"].round(0)
    budget_df["County"] = budget_df["County"].str.lower()

    county_budget = {}
    for c in counties:
        row = budget_df.loc[budget_df["County"] == c, "Number of inspection stations"]
        county_budget[c] = int(row.values[0]) if len(row) else 0

    counties_playing, lakes_playing, arcs_playing = _playing_subsets(
        counties, county_budget, I_c, arcs
    )

    return EBMCInstance(
        counties=counties,
        lakes=lakes,
        lake_county=lake_county,
        I_c=I_c,
        I_c_complement=I_c_complement,
        arcs=arcs,
        n=n,
        t=t,
        arcs_c=arcs_c,
        arcs_plus_c=arcs_plus_c,
        arcs_minus_c=arcs_minus_c,
        infested_lakes=infested_lakes,
        county_budget=county_budget,
        counties_playing=counties_playing,
        lakes_playing=lakes_playing,
        arcs_playing=arcs_playing,
        dataset="minnesota",
        meta={"percentage": percentage, "species_columns": list(species_columns)},
    )


# ---------------------------------------------------------------------------
# Random EBMC generator (multi-AIS)
# ---------------------------------------------------------------------------

# Default per-county per-species infestation rates for the multi-AIS generator.
# Species: zebra mussel, spiny water flea, Eurasian watermilfoil, starry stonewort.
DEFAULT_SPECIES_PROBS: dict = {
    "zm2019": 0.5,
    "ss2019": 0.4,
    "ew2019": 0.3,
    "sf2019": 0.2,
}


def _two_letter_county_codes(n: int) -> list:
    """Two-letter county codes AA, AB, ..., ZZ. Caps at 100 counties."""
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    codes = [a + b for a in alphabet for b in alphabet][:100]
    if n > len(codes):
        raise ValueError(f"n_counties={n} exceeds 100 (max 2-letter codes).")
    return codes[:n]


def _build_random_ebmc_data(
    n_counties: int,
    lakes_per_county: int,
    species_probs: dict,
    edge_drop_rate: float,
    weight_low: int,
    weight_high: int,
    budget_ratio,
    seed: int,
) -> tuple:
    """Build the raw multi-AIS dataset components shared by the in-memory and
    on-disk generators.

    Returns
    -------
    (counties, lakes, lake_county, infestation_status, df_edge, county_budget)
        where ``df_edge`` already has the random row drop and ``bij >= 1`` filter
        applied, and matches the bundled CSV column layout.
    """
    import random as _random
    from itertools import permutations

    rng = _random.Random(seed)
    counties = _two_letter_county_codes(n_counties)
    lakes = [c + str(i) for c in counties for i in range(1, lakes_per_county + 1)]
    lake_county = {
        c + str(i): c for c in counties for i in range(1, lakes_per_county + 1)
    }
    county_lakes = {
        c: [c + str(i) for i in range(1, lakes_per_county + 1)] for c in counties
    }

    species_list = list(species_probs.keys())

    infestation_status: dict = {lake: {} for lake in lakes}
    for c in counties:
        for species in species_list:
            n_inf = int(lakes_per_county * species_probs[species])
            infested = set(rng.sample(county_lakes[c], n_inf))
            for lake in county_lakes[c]:
                infestation_status[lake][species] = int(lake in infested)

    infested_lakes = [
        lake for lake, vals in infestation_status.items()
        if any(v > 0 for v in vals.values())
    ]

    rows = []
    for o, d in permutations(lakes, 2):
        co = lake_county[o]
        cd = lake_county[d]
        vals_o = [infestation_status[o][s] for s in species_list]
        vals_d = [infestation_status[d][s] for s in species_list]
        risks = [int(vals_o[k] == 1 and vals_d[k] == 0) for k in range(len(species_list))]
        bij = sum(risks)
        weight = rng.randint(weight_low, weight_high)
        rows.append([o, d, weight, co, *vals_o, cd, *vals_d, *risks, bij])

    columns = (
        ["dow_origin", "dow_destination", "weight", "county_name.origin"]
        + [f"{s}.origin" for s in species_list]
        + ["county_name.destination"]
        + [f"{s}.destination" for s in species_list]
        + [f"{s}.risky" for s in species_list]
        + ["bij"]
    )
    df_edge = pd.DataFrame(rows, columns=columns)

    if 0.0 < edge_drop_rate < 1.0:
        n_drop = int(len(df_edge) * edge_drop_rate)
        if n_drop > 0:
            drop_idx = rng.sample(range(len(df_edge)), n_drop)
            df_edge = df_edge.drop(index=drop_idx).reset_index(drop=True)
    df_edge = df_edge[df_edge["bij"] >= 1].reset_index(drop=True)

    county_budget: dict = {}
    for c in counties:
        n_inf_in_c = sum(1 for lk in infested_lakes if lake_county[lk] == c)
        if budget_ratio == "random":
            ratio = rng.uniform(0.3, 0.7)
        else:
            ratio = float(budget_ratio)
        county_budget[c] = int(ratio * n_inf_in_c)

    return counties, lakes, lake_county, infestation_status, df_edge, county_budget


def generate_random_ebmc_instance(
    n_counties: int,
    lakes_per_county: int = 30,
    species_probs: dict | None = None,
    edge_drop_rate: float = 0.8,
    weight_low: int = 10,
    weight_high: int = 30,
    budget_ratio: float | str = "random",
    seed: int = 0,
) -> EBMCInstance:
    """Generate a synthetic multi-AIS EBMC instance.

    Per-lake infestation status is sampled independently for four species:
    zebra mussel (zm2019), spiny water flea (ss2019), Eurasian watermilfoil
    (ew2019), and starry stonewort (sf2019). For each county and species,
    exactly ``int(lakes_per_county * species_prob)`` lakes are flagged
    infested. Lake names follow the bundled CSV convention: 2-letter county
    code + 1-based index (``'AA1', 'AA2', ..., 'AB1', ...``), so the same
    instance can later be serialized via :func:`write_random_ebmc_csv` and
    reloaded with :func:`load_random_ebmc_csv` without naming clashes.

    Arc construction: start from all ordered lake pairs, drop
    ``edge_drop_rate`` of them at random, then keep only those with
    ``bij = #species where origin is infested and destination is not >= 1``.

    Parameters
    ----------
    n_counties        : number of counties (players); max 100.
    lakes_per_county  : lakes per county.
    species_probs     : dict ``{species_id -> per-county infestation rate}``;
                        defaults to ``DEFAULT_SPECIES_PROBS``.
    edge_drop_rate    : fraction of candidate ordered pairs dropped uniformly
                        at random before the ``bij >= 1`` filter.
    weight_low/high   : inclusive ``Uniform`` range for arc weights ``n_ij``.
    budget_ratio      : ``'random'`` (per-county ``Uniform(0.3, 0.7)``) or a
                        positive float; the chosen ratio is multiplied by the
                        county's infested-lake count to produce its budget.
    seed              : ``random.Random`` seed for full determinism.
    """
    if species_probs is None:
        species_probs = dict(DEFAULT_SPECIES_PROBS)

    counties, lakes, lake_county, infestation_status, df_edge, county_budget = (
        _build_random_ebmc_data(
            n_counties=n_counties,
            lakes_per_county=lakes_per_county,
            species_probs=species_probs,
            edge_drop_rate=edge_drop_rate,
            weight_low=weight_low,
            weight_high=weight_high,
            budget_ratio=budget_ratio,
            seed=seed,
        )
    )

    I_c = {c: [lk for lk in lakes if lake_county[lk] == c] for c in counties}
    lakes_set = set(lakes)
    I_c_complement = {c: list(lakes_set - set(I_c[c])) for c in counties}

    infested_lakes = sorted(
        lk for lk, vals in infestation_status.items()
        if any(v > 0 for v in vals.values())
    )

    arcs = set(zip(df_edge["dow_origin"], df_edge["dow_destination"]))
    n_dict = dict(
        zip(zip(df_edge["dow_origin"], df_edge["dow_destination"]),
            df_edge["weight"].tolist())
    )
    t_dict = dict(
        zip(zip(df_edge["dow_origin"], df_edge["dow_destination"]),
            df_edge["bij"].astype(float).tolist())
    )

    arcs_c, arcs_plus_c, arcs_minus_c = _derive_arc_partitions(
        arcs, lake_county, counties
    )
    counties_playing, lakes_playing, arcs_playing = _playing_subsets(
        counties, county_budget, I_c, arcs
    )

    return EBMCInstance(
        counties=counties,
        lakes=lakes,
        lake_county=lake_county,
        I_c=I_c,
        I_c_complement=I_c_complement,
        arcs=arcs,
        n=n_dict,
        t=t_dict,
        arcs_c=arcs_c,
        arcs_plus_c=arcs_plus_c,
        arcs_minus_c=arcs_minus_c,
        infested_lakes=infested_lakes,
        county_budget=county_budget,
        counties_playing=counties_playing,
        lakes_playing=lakes_playing,
        arcs_playing=arcs_playing,
        dataset="random",
        meta={
            "n_counties": n_counties,
            "lakes_per_county": lakes_per_county,
            "species_probs": dict(species_probs),
            "edge_drop_rate": edge_drop_rate,
            "weight_low": weight_low,
            "weight_high": weight_high,
            "budget_ratio": budget_ratio,
            "seed": seed,
        },
    )


def write_random_ebmc_csv(
    output_dir: str | Path,
    n_counties: int,
    lakes_per_county: int = 30,
    species_probs: dict | None = None,
    edge_drop_rate: float = 0.8,
    weight_low: int = 10,
    weight_high: int = 30,
    budget_ratio: float | str = "random",
    seed: int = 0,
    update_pickle: bool = True,
) -> Path:
    """Generate a multi-AIS instance and serialize it to disk in the bundled
    CSV format (compatible with :func:`load_random_ebmc_csv`).

    Output filename is ``f"{n_counties}_{lakes_per_county}_{budget_ratio}.csv"``.
    When ``update_pickle=True``, the sibling ``info_data.pickle`` is updated
    (or created) with the key ``(n_counties, lakes_per_county, budget_ratio)``
    mapping to the tuple
    ``(counties, lakes_per_county, infestation_status, county_budget)``.

    Returns the written CSV path.
    """
    import pickle as _pickle

    if species_probs is None:
        species_probs = dict(DEFAULT_SPECIES_PROBS)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    counties, _, _, infestation_status, df_edge, county_budget = (
        _build_random_ebmc_data(
            n_counties=n_counties,
            lakes_per_county=lakes_per_county,
            species_probs=species_probs,
            edge_drop_rate=edge_drop_rate,
            weight_low=weight_low,
            weight_high=weight_high,
            budget_ratio=budget_ratio,
            seed=seed,
        )
    )

    csv_path = output_dir / f"{n_counties}_{lakes_per_county}_{budget_ratio}.csv"
    df_edge.to_csv(csv_path)

    if update_pickle:
        pkl_path = output_dir / "info_data.pickle"
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                info_data = _pickle.load(f)
        else:
            info_data = {}
        info_data[(n_counties, lakes_per_county, budget_ratio)] = (
            counties, lakes_per_county, infestation_status, county_budget
        )
        with open(pkl_path, "wb") as f:
            _pickle.dump(info_data, f)

    return csv_path


# ---------------------------------------------------------------------------
# Random EBMC CSV loader (bundled-format reader)
# ---------------------------------------------------------------------------

# Bundled dataset location (relative to this file)
_BUNDLED_EBMC_RANDOM_DIR = Path(__file__).parent / "datasets" / "ebmc_random"


def _bundled_ebmc_random_path(n_counties: int, lakes_per_county: int = 30) -> Path:
    """Return the bundled random EBMC CSV path inside the package."""
    return _BUNDLED_EBMC_RANDOM_DIR / f"{n_counties}_{lakes_per_county}_random.csv"


def load_random_ebmc(
    n_counties: int,
    lakes_per_county: int = 30,
    budget_ratio: str | float = "random",
    seed: int = 0,
    dataset_dir: str | Path | None = None,
) -> EBMCInstance:
    """High-level loader for bundled random EBMC instances.

    Resolves the CSV path inside ``nash_informed/ebmc/datasets/ebmc_random/``
    (or a user-specified ``dataset_dir``) and forwards to ``load_random_ebmc_csv``.
    Looks for a sibling ``info_data.pickle`` and uses its county-budget dict
    when the requested ``(n_counties, lakes_per_county, budget_ratio)`` key
    is present.

    Available sizes in the bundled dataset: n_counties in {2, 3, 5, 8, 10, 15, 20}.
    """
    if dataset_dir is None:
        dataset_dir = _BUNDLED_EBMC_RANDOM_DIR
    dataset_dir = Path(dataset_dir)
    csv_path = dataset_dir / f"{n_counties}_{lakes_per_county}_random.csv"
    pickle_path = dataset_dir / "info_data.pickle"
    return load_random_ebmc_csv(
        csv_path=csv_path,
        budget_ratio=budget_ratio,
        seed=seed,
        pickle_path=pickle_path if pickle_path.exists() else None,
        pickle_key=(n_counties, lakes_per_county, budget_ratio),
    )


def load_random_ebmc_csv(
    csv_path: str | Path,
    species_columns: tuple = ("zm2019", "ss2019", "ew2019", "sf2019"),
    budget_ratio: float | str = "random",
    seed: int = 0,
    pickle_path: str | Path | None = None,
    pickle_key: tuple | None = None,
) -> EBMCInstance:
    """Load an EBMC random instance from a pre-generated CSV.

    Expected CSV columns:
        dow_origin, dow_destination, weight,
        county_name.origin, {species}.origin, county_name.destination,
        {species}.destination, {species}.risky, bij

    Parameters
    ----------
    csv_path        : path to the CSV.
    species_columns : species name stems present in the CSV.
    budget_ratio    : 'random' -> per-county Uniform{0..|I_c|}; float in (0, 1] ->
                       proportional fixed budget.
    seed            : RNG seed for budget draws when no pickle is available.
    pickle_path     : optional ``info_data.pickle`` providing exact county budgets
                       under the ``pickle_key`` entry. If None, the loader tries a
                       sibling ``info_data.pickle`` next to the CSV.
    pickle_key      : tuple ``(n_counties, lakes_per_county, budget_ratio)`` used
                       to look up the budget dict in the pickle. If None, the loader
                       tries to infer it from the CSV filename.
    """
    import pickle as _pickle
    import re as _re

    csv_path = Path(csv_path)
    rng = np.random.default_rng(seed)

    df = pd.read_csv(csv_path)
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]

    lake_county_map: dict = {}
    for col_pref in ("origin", "destination"):
        dow_col = f"dow_{col_pref}"
        cty_col = f"county_name.{col_pref}"
        sub = df[[dow_col, cty_col]].drop_duplicates()
        for _, row in sub.iterrows():
            lake_county_map[row[dow_col]] = row[cty_col]

    lakes = sorted(lake_county_map.keys())
    counties = sorted(set(lake_county_map.values()))
    lake_county = dict(lake_county_map)
    I_c = {c: [i for i in lakes if lake_county[i] == c] for c in counties}
    lakes_set = set(lakes)
    I_c_complement = {c: list(lakes_set - set(I_c[c])) for c in counties}

    infested: set = set()
    for sp in species_columns:
        for col_pref in ("origin", "destination"):
            sp_col = f"{sp}.{col_pref}"
            dow_col = f"dow_{col_pref}"
            if sp_col in df.columns:
                infested.update(df.loc[df[sp_col] == 1, dow_col].unique().tolist())
    infested_lakes = sorted(infested)

    df = df.loc[df["dow_origin"] != df["dow_destination"]].copy()
    df = df.groupby(["dow_origin", "dow_destination"], as_index=False).agg(
        {"weight": "sum", "bij": "max"}
    )
    df = df.loc[df["bij"] > 0]
    arcs = set(zip(df["dow_origin"], df["dow_destination"]))
    n = dict(zip(zip(df["dow_origin"], df["dow_destination"]), df["weight"].tolist()))
    t = dict(zip(zip(df["dow_origin"], df["dow_destination"]),
                 df["bij"].astype(float).tolist()))

    arcs_c, arcs_plus_c, arcs_minus_c = _derive_arc_partitions(arcs, lake_county, counties)

    lakes_per_county = int(np.median([len(v) for v in I_c.values()]))
    county_budget: dict | None = None
    budget_source = "generated"

    auto_pickle = pickle_path
    if auto_pickle is None:
        sibling = csv_path.parent / "info_data.pickle"
        if sibling.exists():
            auto_pickle = sibling

    if auto_pickle is not None and Path(auto_pickle).exists():
        if pickle_key is None:
            m = _re.match(r"(\d+)_(\d+)_(\w+)\.csv$", csv_path.name)
            if m is not None:
                pickle_key = (int(m.group(1)), int(m.group(2)), m.group(3))
        if pickle_key is not None:
            try:
                with open(auto_pickle, "rb") as f:
                    info_data = _pickle.load(f)
                if pickle_key in info_data:
                    raw_budget = info_data[pickle_key][3]
                    if set(raw_budget.keys()) >= set(counties):
                        county_budget = {c: int(raw_budget[c]) for c in counties}
                        budget_source = f"pickle:{pickle_key}"
            except Exception:
                pass

    if county_budget is None:
        if budget_ratio == "random":
            county_budget = {c: int(rng.integers(0, len(I_c[c]) + 1)) for c in counties}
        else:
            county_budget = {
                c: int(max(1, round(float(budget_ratio) * len(I_c[c]))))
                for c in counties
            }

    counties_playing, lakes_playing, arcs_playing = _playing_subsets(
        counties, county_budget, I_c, arcs
    )

    return EBMCInstance(
        counties=counties,
        lakes=lakes,
        lake_county=lake_county,
        I_c=I_c,
        I_c_complement=I_c_complement,
        arcs=arcs,
        n=n,
        t=t,
        arcs_c=arcs_c,
        arcs_plus_c=arcs_plus_c,
        arcs_minus_c=arcs_minus_c,
        infested_lakes=infested_lakes,
        county_budget=county_budget,
        counties_playing=counties_playing,
        lakes_playing=lakes_playing,
        arcs_playing=arcs_playing,
        dataset="random_csv",
        meta={
            "csv_path": str(csv_path),
            "n_counties": len(counties),
            "lakes_per_county": lakes_per_county,
            "budget_ratio": budget_ratio,
            "seed": seed,
            "budget_source": budget_source,
        },
    )
