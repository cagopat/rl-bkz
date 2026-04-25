# RL-BKZ: Learning Adaptive BKZ Schedules via Reinforcement Learning

A DQN agent that learns adaptive BKZ block-size schedules for lattice basis reduction and transfers to lattice dimensions it was never trained on.

---

## Key Results

**The agent is trained on n ∈ [40, 65] and evaluated zero-shot on n = 70–110.**

### Equal-tour comparison (RL vs Progressive, paired t-test, n=100 instances)

| n (unseen) | RL improvement | Progressive | Δ | p-value |
|---|---|---|---|---|
| 75 | 0.513 | 0.501 | +0.013 | 0.015 * |
| 80 | 0.559 | 0.539 | +0.020 | 0.0003 *** |
| 85 | 0.617 | 0.588 | +0.029 | <0.0001 *** |
| 90 | 0.658 | 0.623 | +0.035 | <0.0001 *** |
| 95 | 0.713 | 0.662 | +0.051 | <0.0001 *** |
| 100 | 0.759 | 0.713 | +0.046 | <0.0001 *** |
| 105 | 0.798 | 0.745 | +0.053 | <0.0001 *** |
| 110 | 0.842 | 0.777 | +0.065 | <0.0001 *** |

The effect grows with dimension — the further outside the training range, the larger the advantage.

### The agent is cheaper than Fixed BKZ-30, not more expensive

At small-to-mid n the agent mixes cheap and expensive tours intelligently:

| n | Fixed BKZ-30 | RL (DQN) | RL cheaper by |
|---|---|---|---|
| 70 | 0.034 s/tour | 0.018 s/tour | 1.9× |
| 80 | 0.045 s/tour | 0.031 s/tour | 1.5× |
| 90 | 0.061 s/tour | 0.053 s/tour | 1.2× |
| 100 | 0.081 s/tour | 0.080 s/tour | ≈ equal |
| 110 | 0.105 s/tour | 0.105 s/tour | identical |

At large n the agent converges to always selecting BKZ-30, which is optimal and expected. At small n it learns that a mix of BKZ-10 and BKZ-25 gets similar quality for less compute.


---

## How it works

### State (dimension-invariant)

The raw GS log-norm profile has n components and changes size with n. We fix this by interpolating to 32 evenly-spaced points, then appending 4 scalars:

```
[ interp_GS_profile[0..31],   # shape of the basis, fixed 32 points
  GS slope,                    # how far from flat (= fully reduced)
  log-RHF estimate,            # single-number quality proxy
  n_norm,                      # normalised lattice dimension
  step_frac ]                  # episode progress
```

Total: 36 floats, same shape for n=40 or n=110.

### Action

```python
BLOCK_SIZES = [10, 15, 20, 25, 30]
```

Each action runs one BKZ tour at that block size.

### Reward

```
r_t = 100 * ( log‖b₁^(t)‖ − log‖b₁^(t+1)‖ − λ · wall_time )
```

Positive when the first basis vector gets shorter. `λ=0.1` penalises expensive tours.

### Agent

Standard DQN: 2-layer MLP (128 hidden, ReLU), Huber loss, Adam, replay buffer (20k), hard target sync every 200 learn steps, linear ε-greedy decay.

---

## Quick start

```bash
pip install -r requirements.txt

# Multi-dimension training (the main experiment)
python train.py --n_range 40 65 --episodes 800 --max_steps 25 \
    --lam 0.1 --save bkz_dqn_transfer.pt

# Transfer evaluation on unseen dimensions
python evaluate.py --transfer --model bkz_dqn_transfer.pt \
    --test_dims 70 75 80 85 90 95 100 105 110 --n_test 50 --max_steps 25

# Statistical significance (paired t-test vs Progressive)
python ttest.py --test_type improvement \
    --dims 70 75 80 85 90 95 100 105 110 --n_test 100 --max_steps 25

# Single-dimension baseline table
python train.py --n 60 --episodes 600 --max_steps 25 --lam 0.1 --save bkz_dqn_n60.pt
python evaluate.py --n 60 --max_steps 25 --model bkz_dqn_n60.pt --n_test 50
```

---

## Files

| File | Purpose |
|------|---------|
| `env.py` | `BKZEnv` gymnasium environment, dimension-invariant state, lattice utilities |
| `agent.py` | `DQNAgent` — Q-network, replay buffer, target network |
| `baselines.py` | Fixed BKZ-20/30 and Progressive schedule baselines |
| `train.py` | Training loop; supports `--n_range` for multi-dim training |
| `evaluate.py` | Standard eval + transfer eval across dimensions, 4-panel plots |
| `ttest.py` | Paired t-tests: equal-tour and fixed-budget comparisons |
| `generate_strategies.py` | One-time setup for fplll pruning strategies (macOS only) |

---

## Baselines

| Method | Description |
|--------|-------------|
| Fixed BKZ-20 | Always β = 20 |
| Fixed BKZ-30 | Always β = 30, strongest single action |
| Progressive | Cycles 10 → 15 → 20 → 25 → 30, hand-designed |
| **RL (DQN)** | Learned policy, conditioned on current GS profile |

---

## Limitations

- Action space is discrete and small (`β ∈ {10,...,30}`). The agent cannot beat Fixed BKZ-30 on raw quality since BKZ-30 is the strongest available action — extending to `{15, 20, 25, 30, 40}` would allow it.
- Training uses random q-ary lattices. Transfer to structured instances (LWE, SVP Challenge) is untested.
- Only block size is controlled. BKZ 2.0 also exposes pruning strength and preprocessing — extending the action space there is the natural next step.
- Epsilon decays in the first 10% of training (first 80 of 800 episodes), leaving 90% of training near-greedy. More exploration budget may improve the policy.

---

## References

- Aono et al. (2016). *Improved Progressive BKZ Algorithms and Their Precise Cost Estimation by Sharp Simulator.* EUROCRYPT.
- Chen & Nguyen (2011). *BKZ 2.0: Better Lattice Security Estimates.* ASIACRYPT.
- Schnorr & Euchner (1994). *Lattice Basis Reduction.*
- fpylll: https://github.com/fplll/fpylll
