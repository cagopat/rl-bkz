"""
Deterministic BKZ baselines for comparison against RL.

Each function returns:
  (log_norm1_history, total_wall_seconds, actions_taken, tour_times)

tour_times is a list of per-tour elapsed seconds, needed for the
time-budget evaluation in evaluate.py.
"""

from typing import List, Tuple

from fpylll import IntegerMatrix

from env import copy_lattice, gso_log_norms, run_one_tour, BLOCK_SIZES


History   = List[float]
TourTimes = List[float]
Result    = Tuple[History, float, List[int], TourTimes]


def _reduce(B: IntegerMatrix, schedule: List[int],
            use_default: bool = False) -> Result:
    hist: History = [float(gso_log_norms(B)[0])]
    total_time = 0.0
    actions: List[int] = []
    tour_times: TourTimes = []
    for beta in schedule:
        elapsed = run_one_tour(B, beta, use_default_strategy=use_default)
        total_time += elapsed
        actions.append(beta)
        tour_times.append(elapsed)
        hist.append(float(gso_log_norms(B)[0]))
    return hist, total_time, actions, tour_times


def run_fixed(B_orig: IntegerMatrix, block_size: int,
              n_tours: int = 20) -> Result:
    B = copy_lattice(B_orig)
    return _reduce(B, [block_size] * n_tours)


def run_progressive(B_orig: IntegerMatrix, n_tours: int = 20) -> Result:
    base = [10, 15, 20, 25, 30]
    schedule = (base * ((n_tours // len(base)) + 1))[:n_tours]
    B = copy_lattice(B_orig)
    return _reduce(B, schedule)


def run_fplll_default(B_orig: IntegerMatrix, block_size: int = 20,
                      n_tours: int = 20) -> Result:
    B = copy_lattice(B_orig)
    try:
        return _reduce(B, [block_size] * n_tours, use_default=True)
    except RuntimeError:
        return _reduce(B, [block_size] * n_tours, use_default=False)


def run_rl(agent, B_orig: IntegerMatrix, n: int = 0,
           max_steps: int = 20) -> Result:  # n unused; kept for API compatibility
    from env import build_obs, run_one_tour, BLOCK_SIZES

    B = copy_lattice(B_orig)
    hist: History = [float(gso_log_norms(B)[0])]
    total_time = 0.0
    actions: List[int] = []
    tour_times: TourTimes = []

    for step in range(max_steps):
        obs = build_obs(B, step + 1, max_steps)  # match training: step_count is post-increment
        action = agent.select_action(obs, greedy=True)
        beta = BLOCK_SIZES[action]
        elapsed = run_one_tour(B, beta)
        total_time += elapsed
        actions.append(beta)
        tour_times.append(elapsed)
        hist.append(float(gso_log_norms(B)[0]))

    return hist, total_time, actions, tour_times


# fplll Default (BKZ-20 with pruning strategies) is excluded from the main
# comparison: aggressive pruning at small block sizes hurts short-run quality
# and requires the full BKZ 2.0 pipeline (preprocessing + rerandomization) to
# show its advantage. The three baselines below are the standard comparison set.
BASELINES = {
    "Fixed BKZ-20": lambda B, n_tours: run_fixed(B, 20, n_tours),
    "Fixed BKZ-30": lambda B, n_tours: run_fixed(B, 30, n_tours),
    "Progressive":  lambda B, n_tours: run_progressive(B, n_tours),
}