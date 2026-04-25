

import argparse
import json

import numpy as np

from env import BKZEnv, OBS_DIM
from agent import DQNAgent


def train(
    n: int = 40,
    n_range: tuple[int, int] | None = None,
    max_steps: int = 20,
    lam: float = 0.05,
    n_episodes: int = 400,
    lr: float = 3e-4,
    gamma: float = 0.99,
    eps_decay_steps: int = 3000,
    hidden: int = 128,
    batch_size: int = 128,
    target_update_freq: int = 200,
    log_every: int = 25,
    save_path: str = "bkz_dqn.pt",
    metrics_path: str = "train_metrics.json",
) -> DQNAgent:

    env = BKZEnv(n=n, n_range=n_range, max_steps=max_steps, lam=lam)
    n_actions = env.action_space.n

    # obs_dim is now fixed = OBS_DIM regardless of n
    agent = DQNAgent(
        obs_dim=OBS_DIM,
        n_actions=n_actions,
        lr=lr,
        gamma=gamma,
        eps_decay_steps=eps_decay_steps,
        batch_size=batch_size,
        target_update_freq=target_update_freq,
        hidden=hidden,
    )

    mode = f"n_range={n_range}" if n_range else f"n={n}"
    print(f"Training DQN | {mode} | max_steps={max_steps} | lam={lam} | episodes={n_episodes}")
    print(f"obs_dim={OBS_DIM} | n_actions={n_actions} | device={agent.device}")
    print("-" * 60)

    metrics = {"episode_reward": [], "episode_improvement": [],
               "episode_final_log_norm": [], "loss": [], "episode_n": []}

    for ep in range(n_episodes):
        obs, info = env.reset()
        ep_n = info.get("n", n)
        ep_reward = ep_improvement = 0.0
        ep_losses = []

        for _ in range(max_steps):
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated
            agent.observe(obs, action, reward, next_obs, done)
            loss = agent.learn()
            if loss is not None:
                ep_losses.append(loss)
            obs = next_obs
            ep_reward += reward
            ep_improvement += step_info["improvement"]
            if done:
                break

        metrics["episode_reward"].append(ep_reward)
        metrics["episode_improvement"].append(ep_improvement)
        metrics["episode_final_log_norm"].append(float(obs[0]))
        metrics["episode_n"].append(ep_n)
        if ep_losses:
            metrics["loss"].append(float(np.mean(ep_losses)))

        if (ep + 1) % log_every == 0:
            w = log_every
            avg_r  = np.mean(metrics["episode_reward"][-w:])
            avg_im = np.mean(metrics["episode_improvement"][-w:])
            avg_fn = np.mean(metrics["episode_final_log_norm"][-w:])
            avg_l  = np.mean(metrics["loss"][-w:]) if metrics["loss"] else float("nan")
            print(
                f"Ep {ep+1:5d}/{n_episodes} | "
                f"avg_reward={avg_r:7.3f} | "
                f"avg_improvement={avg_im:.4f} | "
                f"avg_final_log_norm={avg_fn:.3f} | "
                f"loss={avg_l:.4f} | "
                f"eps={agent.eps:.3f}"
            )

    agent.save(save_path)
    print(f"\nModel saved → {save_path}")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)
    print(f"Metrics saved → {metrics_path}")

    return agent


def _parse_args():
    p = argparse.ArgumentParser(description="Train RL-BKZ DQN agent")
    p.add_argument("--n",           type=int,   default=40,
                   help="Fixed lattice dimension (ignored if --n_range is set)")
    p.add_argument("--n_range",     type=int,   nargs=2, default=None,
                   metavar=("LO", "HI"),
                   help="Sample n uniformly from [LO, HI] each episode (multi-dim training)")
    p.add_argument("--max_steps",   type=int,   default=20)
    p.add_argument("--lam",         type=float, default=0.05)
    p.add_argument("--episodes",    type=int,   default=400)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--eps_decay",   type=int,   default=3000)
    p.add_argument("--hidden",      type=int,   default=128)
    p.add_argument("--batch_size",  type=int,   default=128)
    p.add_argument("--log_every",   type=int,   default=25)
    p.add_argument("--save",        type=str,   default="bkz_dqn.pt")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        n=args.n,
        n_range=tuple(args.n_range) if args.n_range else None,
        max_steps=args.max_steps,
        lam=args.lam,
        n_episodes=args.episodes,
        lr=args.lr,
        eps_decay_steps=args.eps_decay,
        hidden=args.hidden,
        batch_size=args.batch_size,
        log_every=args.log_every,
        save_path=args.save,
    )