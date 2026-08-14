"""Knapsack Potential Game (KPG).

Public exports:
  Instances:
    KPGInstance, load_kpg_instance, load_kpg_bundled, list_bundled_instances,
    generate_kpg_instances

  Model primitives:
    compute_utility, compute_player_utility,
    solve_sw, build_best_response_model, update_best_response_objective,
    generate_feasible_strategy

  BZR (with strategic inequalities for type C):
    build_master_kpg, strategic_inequalities,
    make_separation_callback, make_bzr_callback,
    rrr_brd, find_pne_zr, find_pne_bzr

  Bounded family:
    solve_bsw

  Pipeline:
    run_pipeline_bzr
"""
from .instances import (
    KPGInstance, load_kpg_instance, load_kpg_bundled,
    list_bundled_instances, generate_kpg_instances,
)
from .model import (
    compute_utility, compute_player_utility, solve_sw,
    build_best_response_model, update_best_response_objective,
    generate_feasible_strategy,
)
from .bzr import (
    build_master_kpg, strategic_inequalities,
    make_separation_callback, make_bzr_callback,
    rrr_brd, find_pne_zr, find_pne_bzr,
)
from .bsw import solve_bsw
from .pipeline import run_pipeline_bzr

__all__ = [
    "KPGInstance", "load_kpg_instance", "load_kpg_bundled",
    "list_bundled_instances", "generate_kpg_instances",
    "compute_utility", "compute_player_utility", "solve_sw",
    "build_best_response_model", "update_best_response_objective",
    "generate_feasible_strategy",
    "build_master_kpg", "strategic_inequalities",
    "make_separation_callback", "make_bzr_callback",
    "rrr_brd", "find_pne_zr", "find_pne_bzr",
    "solve_bsw",
    "run_pipeline_bzr",
]
