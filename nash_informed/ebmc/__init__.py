"""Edge-weighted Budgeted Maximum Coverage (EBMC) Game.

Public exports:
  Instances:
    EBMCInstance, load_minnesota_instance, generate_random_ebmc_instance,
    write_random_ebmc_csv, DEFAULT_SPECIES_PROBS,
    load_random_ebmc_csv, load_random_ebmc

  Model primitives:
    compute_utility, compute_player_utility, flatten_selection
    solve_sw, build_best_response_model, solve_best_response, solve_non_game
    generate_feasible_strategy, check_individual_best, print_solution

  BZR (PNE certification):
    build_master_ebmc, rrr_brd, make_bzr_callback, find_pne_bzr

  Bounded family:
    solve_gc, solve_bgc, solve_bsw, solve_bc, make_first_feasible_callback

  Pipeline:
    run_pipeline_bzr
"""

from .instances import (
    EBMCInstance,
    load_minnesota_instance,
    generate_random_ebmc_instance,
    write_random_ebmc_csv,
    DEFAULT_SPECIES_PROBS,
    load_random_ebmc_csv,
    load_random_ebmc,
)
from .model import (
    compute_utility,
    compute_player_utility,
    flatten_selection,
    solve_sw,
    build_best_response_model,
    solve_best_response,
    solve_non_game,
    generate_feasible_strategy,
    check_individual_best,
    print_solution,
)
from .bzr import (
    build_master_ebmc,
    rrr_brd,
    make_bzr_callback,
    find_pne_bzr,
)
from .bsw import (
    solve_gc,
    solve_bgc,
    solve_bsw,
    solve_bc,
    make_first_feasible_callback,
)
from .pipeline import run_pipeline_bzr
from .datasets.minnesota_groups import (
    MN_NON_COALITIONS,
    MN_BC3_COAL1,
    MN_BC3_COAL2,
    get_bc3_noncoal,
    BC_VARIANTS,
    bc3_coal_kwargs,
)

__all__ = [
    "EBMCInstance",
    "load_minnesota_instance",
    "generate_random_ebmc_instance",
    "write_random_ebmc_csv",
    "DEFAULT_SPECIES_PROBS",
    "load_random_ebmc_csv",
    "load_random_ebmc",
    "compute_utility",
    "compute_player_utility",
    "flatten_selection",
    "solve_sw",
    "build_best_response_model",
    "solve_best_response",
    "solve_non_game",
    "generate_feasible_strategy",
    "check_individual_best",
    "print_solution",
    "build_master_ebmc",
    "rrr_brd",
    "make_bzr_callback",
    "find_pne_bzr",
    "solve_gc",
    "solve_bgc",
    "solve_bsw",
    "solve_bc",
    "make_first_feasible_callback",
    "run_pipeline_bzr",
    "MN_NON_COALITIONS",
    "MN_BC3_COAL1",
    "MN_BC3_COAL2",
    "get_bc3_noncoal",
    "BC_VARIANTS",
    "bc3_coal_kwargs",
]
