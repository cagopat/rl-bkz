"""
BKZ Gym environment — dimension-invariant state representation.

State  : fixed K=32 interpolated GS log-norms + slope + log-RHF +
         normalised dimension + step fraction  →  K+4 = 36 floats
Action : choose block size from {10, 15, 20, 25, 30}
Reward : improvement in log||b1|| minus time penalty

Using a fixed-size interpolated GS profile means the same network weights
work for any lattice dimension, enabling transfer experiments.
"""

import os
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from fpylll import IntegerMatrix, GSO, LLL, BKZ


_FPLLL_STRATEGIES = None

def _get_strategies():
    """Return loaded fplll strategies, or None if unavailable."""
    global _FPLLL_STRATEGIES
    if _FPLLL_STRATEGIES is None:
        from fpylll import load_strategies_json, BKZ as _BKZ
        path = _BKZ.DEFAULT_STRATEGY.decode()
        if os.path.exists(path):
            _FPLLL_STRATEGIES = load_strategies_json(path)
    return _FPLLL_STRATEGIES

BLOCK_SIZES = [10, 15, 20, 25, 30]

# Number of points to interpolate the GS profile to (fixed across all n)
GS_PROFILE_DIM = 32
# obs_dim = GS_PROFILE_DIM + 4  (slope, log-RHF, n_norm, step_frac)
OBS_DIM = GS_PROFILE_DIM + 4

# Range of n used for normalising the n_norm state feature.
# N_MAX is set conservatively large so n_norm stays in [0,1] for all
# reasonable test dimensions. If you test n > N_MAX the feature
# extrapolates beyond 1.0 — the network handles it but note the OOD regime.
N_MIN, N_MAX = 30, 120



def make_lattice(n: int, bits: int = 30) -> IntegerMatrix:
    B = IntegerMatrix.random(n, "qary", k=n // 2, bits=bits)
    LLL.reduction(B)
    return B


def copy_lattice(B: IntegerMatrix) -> IntegerMatrix:
    n, m = B.nrows, B.ncols
    C = IntegerMatrix(n, m)
    for i in range(n):
        for j in range(m):
            C[i, j] = B[i, j]
    return C


def gso_log_norms(B: IntegerMatrix) -> np.ndarray:
    n = B.nrows
    M = GSO.Mat(B, float_type="double")
    M.update_gso()
    r = np.array([M.get_r(i, i) for i in range(n)], dtype=np.float64)
    return 0.5 * np.log(np.maximum(r, 1e-300))


def build_obs(B: IntegerMatrix, step: int, max_steps: int) -> np.ndarray:
    """
    Fixed-size feature vector (OBS_DIM = 36), independent of lattice dim n.

      [interp_profile[0..31],   interpolated GS log-norms at 32 evenly-spaced
                                 fractional positions along the basis
       slope,                   linear-fit gradient of the GS profile
       log_rhf,                 (log||b1|| - mean(log-norms)) / n
       n_norm,                  (n - N_MIN) / (N_MAX - N_MIN)  ∈ [0, 1]
       step_frac]               step / max_steps
    """
    n = B.nrows
    ln = gso_log_norms(B)

    # Interpolate raw n-dim profile → fixed GS_PROFILE_DIM points
    xs_orig = np.linspace(0.0, 1.0, n)
    xs_new  = np.linspace(0.0, 1.0, GS_PROFILE_DIM)
    profile = np.interp(xs_new, xs_orig, ln).astype(np.float32)

    xs = np.arange(n, dtype=np.float64)
    slope   = float(np.polyfit(xs, ln, 1)[0])
    log_rhf = float((ln[0] - ln.mean()) / n)
    n_norm  = float((n - N_MIN) / (N_MAX - N_MIN))
    step_frac = float(step / max_steps)

    return np.concatenate([profile, [slope, log_rhf, n_norm, step_frac]]).astype(np.float32)


def run_one_tour(B: IntegerMatrix, block_size: int,
                 use_default_strategy: bool = False) -> float:
    if use_default_strategy:
        strats = _get_strategies()
        par = BKZ.Param(
            block_size=block_size,
            strategies=strats if strats is not None else BKZ.DEFAULT_STRATEGY,
            max_loops=1,
            flags=BKZ.MAX_LOOPS,
        )
    else:
        par = BKZ.Param(
            block_size=block_size,
            max_loops=1,
            flags=BKZ.MAX_LOOPS,
        )
    t0 = time.perf_counter()
    BKZ.reduction(B, par)
    return time.perf_counter() - t0



class BKZEnv(gym.Env):
    """
    One episode = one lattice basis reduced for `max_steps` BKZ tours.

    If `n_range` is given (e.g. (40, 65)), a new dimension is sampled
    uniformly from that range at the start of each episode — this is the
    multi-dim training mode needed for transfer experiments.

    Parameters
    ----------
    n        : fixed lattice dimension (used when n_range is None)
    n_range  : (n_lo, n_hi) — if set, n is sampled uniformly each episode
    max_steps, lam, reward_scale : as before
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        n: int = 40,
        n_range: tuple[int, int] | None = None,
        max_steps: int = 20,
        lam: float = 0.05,
        reward_scale: float = 100.0,
    ):
        super().__init__()
        self._n_fixed = n
        self._n_range = n_range
        self.n = n
        self.max_steps = max_steps
        self.lam = lam
        self.reward_scale = reward_scale

        self.action_space = spaces.Discrete(len(BLOCK_SIZES))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )

        self.B = None
        self.step_count = 0
        self._prev_log_norm1 = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self._n_range is not None:
            lo, hi = self._n_range
            self.n = int(self.np_random.integers(lo, hi + 1))
        else:
            self.n = self._n_fixed
        self.B = make_lattice(self.n)
        self.step_count = 0
        obs = build_obs(self.B, 0, self.max_steps)
        self._prev_log_norm1 = float(obs[0])
        return obs, {"n": self.n}

    def step(self, action: int):
        assert self.B is not None, "call reset() first"
        beta = BLOCK_SIZES[int(action)]
        prev = self._prev_log_norm1

        elapsed = run_one_tour(self.B, beta)
        self.step_count += 1
        obs = build_obs(self.B, self.step_count, self.max_steps)
        curr = float(obs[0])

        improvement = prev - curr
        reward = (improvement - self.lam * elapsed) * self.reward_scale
        self._prev_log_norm1 = curr

        terminated = self.step_count >= self.max_steps
        return obs, float(reward), terminated, False, {
            "block_size": beta,
            "improvement": improvement,
            "elapsed": elapsed,
            "log_norm1": curr,
            "n": self.n,
        }