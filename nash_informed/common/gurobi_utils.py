"""Gurobi parameter helpers shared across game-specific solvers."""

from .config import DEFAULT_CONFIG


def set_gurobi_params(model, config: dict | None = None) -> None:
    """Apply standard Gurobi parameters to a model in place.

    Keys consumed from `config`:
        gurobi_output, gurobi_time_limit, gurobi_mipgap, gurobi_threads.
    Missing keys fall back to DEFAULT_CONFIG.
    """
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    model.setParam("OutputFlag", cfg["gurobi_output"])
    model.setParam("TimeLimit", cfg["gurobi_time_limit"])
    model.setParam("MIPGap", cfg["gurobi_mipgap"])
    if cfg["gurobi_threads"] > 0:
        model.setParam("Threads", cfg["gurobi_threads"])
