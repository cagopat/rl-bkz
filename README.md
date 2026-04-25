# RL-BKZ: Reinforcement Learning for Adaptive Lattice Reduction

A minimal prototype that uses DQN to learn adaptive BKZ block-size schedules,
benchmarked against fixed and progressive baselines on random lattice instances.

---

## Quick start

```bash
pip install -r requirements.txt

# Train (≈ 400 episodes, n=40, 20 tours/episode)
python train.py

# Evaluate against all baselines
python evaluate.py
```

Both scripts accept `--help` for full option lists.

---

## File overview

| File | Purpose |
|------|---------|
| `env.py` | `BKZEnv` gymnasium environment + lattice utilities |
| `agent.py` | `DQNAgent` (Q-net, replay buffer, target net) |
| `baselines.py` | Fixed BKZ-20/30, progressive, fplll-default schedules |
| `train.py` | Training loop; saves `bkz_dqn.pt` + `train_metrics.json` |
| `evaluate.py` | Comparison table + optional reduction-curve plot |

---

## Design decisions

### State

```
[ log‖b₁*‖, log‖b₂*‖, …, log‖bₙ*‖,   ← GS log-norm profile (n values)
  GS slope,                              ← linear fit over GS indices
  log-RHF estimate,                      ← (log‖b₁‖ − log det^{1/n}) / n
  step / max_steps ]                     ← episode progress
```

The GS log-norm profile is the key signal: BKZ theory predicts that a good
reduction flattens the profile, and the slope tells you how far you are from
that ideal.  RHF is the standard single-number quality proxy.

### Action

Discrete: `β ∈ {10, 15, 20, 25, 30}` — each action runs one BKZ tour at
that block size.

### Reward

```
r_t = reward_scale * ( log‖b₁^(t)‖ − log‖b₁^(t+1)‖ − λ · wall_time )
```

- Positive = shorter first basis vector.
- `λ` (default 0.05) penalises expensive tours; tune up/down to shift the
  agent's compute budget.
- `reward_scale` (default 100) keeps rewards O(1) so the Q-network trains
  stably.

### Agent

Standard DQN:
- 2-hidden-layer MLP (128 units, ReLU)
- Huber loss, Adam optimiser
- Linear ε-greedy decay over 3 000 env steps
- Hard target-network sync every 200 learn steps
- Replay buffer: 20 000 transitions

---

## Baselines

| Name | Description |
|------|-------------|
| Fixed BKZ-20 | Always β = 20 |
| Fixed BKZ-30 | Always β = 30 |
| Progressive | 10 → 15 → 20 → 25 → 30 (cycling) |
| fplll Default | β = 20 with `BKZ.DEFAULT_STRATEGY` (pruning + preprocessing) |
| **RL (DQN)** | Greedy policy from trained agent |

---

## Extending the prototype

### Richer action space
```python
# (block_size, pruning_strength, early_termination)
action_space = spaces.MultiDiscrete([5, 3, 2])
```

### Compressed state for variable n
Replace the raw n-dimensional GS profile with:
- Mean, variance, min, max of log-norms  
- First k PCA components of the profile  
- Slope + curvature of a quadratic fit  

This makes the policy transferable across dimensions.

### Better exploration
- Noisy networks instead of ε-greedy  
- Intrinsic reward for novel GS profiles  

### Algorithm upgrades
- Double DQN / Dueling DQN  
- PPO (better for longer episodes)  
- Multi-step returns (n=5) for faster propagation  

### BKZ-2.0 features
Once block-size control is working, extend to BKZ-2.0 knobs:
- Pruning strength (fpylll `Pruning` object)  
- Preprocessing block size  
- Early termination threshold  

### SVP Challenge instances
Download from https://www.latticechallenge.org/svp-challenge/
and pass them in via `evaluate.py --instances path/to/instances/`.

---

## Key references

- Schnorr & Euchner (1994). Lattice Basis Reduction.
- Chen & Nguyen (2011). BKZ 2.0.
- Ducas (2021). Lattice Reduction – A Toolbox for the Cryptanalyst.
- fpylll: https://github.com/fplll/fpylll
- TU Darmstadt SVP Challenge: https://www.latticechallenge.org/svp-challenge/
