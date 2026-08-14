"""Default configuration shared across games.

Each game's pipeline accepts a `config` dict; missing keys fall back to DEFAULT_CONFIG.
"""

DEFAULT_CONFIG: dict = {
    # Randomness
    "seed": 42,

    # Master / BSW Gurobi parameters
    "gurobi_time_limit": 300,   # seconds
    "gurobi_mipgap": 1e-4,
    "gurobi_threads": 16,
    "gurobi_output": 0,         # 0 silent, 1 logs

    # Reference utility numerical tolerance
    "ir_tolerance": 1e-6,
}
