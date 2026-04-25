
import argparse
from pathlib import Path

import numpy as np

from env import BKZEnv, OBS_DIM, make_lattice, BLOCK_SIZES
from agent import DQNAgent
from baselines import BASELINES, run_rl



def _load_agent(model_path: str) -> DQNAgent:
    agent = DQNAgent(OBS_DIM, len(BLOCK_SIZES))
    f = Path(model_path)
    if f.exists():
        agent.load(f)
        print(f"Loaded RL agent from {model_path}")
    else:
        print(f"WARNING: {model_path} not found — using untrained (random) agent")
    return agent


def _run_all(lattices, agent, n, max_steps):
    """Run all baselines + RL on a list of lattices; return results dict."""
    from env import gso_log_norms
    all_results = {name: [] for name in BASELINES}
    all_results["RL (DQN)"] = []

    for i, B in enumerate(lattices, 1):
        init = float(gso_log_norms(B)[0])
        for name, fn in BASELINES.items():
            hist, t, acts, tour_times = fn(B, max_steps)
            all_results[name].append({
                "init": init, "final": hist[-1],
                "improvement": hist[0] - hist[-1],
                "time": t, "hist": hist,
                "actions": acts, "tour_times": tour_times,
            })
        hist, t, acts, tour_times = run_rl(agent, B, n, max_steps)
        all_results["RL (DQN)"].append({
            "init": init, "final": hist[-1],
            "improvement": hist[0] - hist[-1],
            "time": t, "hist": hist,
            "actions": acts, "tour_times": tour_times,
        })
        if i % 5 == 0:
            print(f"  {i}/{len(lattices)}")
    return all_results


def _summary(results):
    improvements = np.array([r["improvement"] for r in results])
    finals       = np.array([r["final"]       for r in results])
    n_tours      = np.array([len(r["tour_times"]) for r in results])
    times        = np.array([r["time"] for r in results])
    return {
        "mean_improvement":   float(improvements.mean()),
        "std_improvement":    float(improvements.std()),
        "mean_final":         float(finals.mean()),
        "std_final":          float(finals.std()),
        "mean_time_per_tour": float((times / n_tours).mean()),
    }


def _budget_final(results, budget):
    finals = []
    for r in results:
        cum, last = 0.0, r["hist"][0]
        for norm, elapsed in zip(r["hist"][1:], r["tour_times"]):
            cum += elapsed
            last = norm
            if cum >= budget:
                break
        finals.append(last)
    return np.array(finals)


def _print_tour_table(all_results):
    print(f"{'Method':<20} {'Δlog‖b₁‖':>10} {'±':>7}  "
          f"{'Final log‖b₁‖':>14} {'±':>7}  {'s/tour':>8}")
    summaries = {}
    for name, results in all_results.items():
        s = _summary(results)
        summaries[name] = s
        marker = " ◄" if name == "RL (DQN)" else ""
        print(f"{name:<20} {s['mean_improvement']:>10.4f} {s['std_improvement']:>7.4f}  "
              f"{s['mean_final']:>14.4f} {s['std_final']:>7.4f}  "
              f"{s['mean_time_per_tour']:>8.3f}{marker}")
    return summaries


def _print_budget_table(all_results, budget, max_steps):
    print(f"\nFixed time budget = {budget:.2f}s  "
          f"(= avg cost of {max_steps} × Fixed BKZ-30 tours)")
    print(f"{'Method':<20} {'Final log‖b₁‖':>14} {'±':>7}  {'Δ vs BKZ-30':>12}")
    budget_summaries = {}
    ref_mean = None
    for name, results in all_results.items():
        finals = _budget_final(results, budget)
        budget_summaries[name] = finals
        if ref_mean is None:
            ref_mean = finals.mean()
        delta = finals.mean() - ref_mean
        sign = "↑" if delta > 0.001 else ("↓" if delta < -0.001 else "≈")
        marker = " ◄" if name == "RL (DQN)" else ""
        print(f"{name:<20} {finals.mean():>14.4f} {finals.std():>7.4f}  "
              f"{delta:>+10.4f} {sign}{marker}")
    print("(↓ = lower = better reduction within the same time budget)")
    return budget_summaries


def _print_action_dist(all_results):
    all_acts = []
    for r in all_results["RL (DQN)"]:
        all_acts.extend(r["actions"])
    print("\nRL action distribution:")
    for bs in BLOCK_SIZES:
        c = all_acts.count(bs)
        pct = c / len(all_acts) * 100 if all_acts else 0
        print(f"  BKZ-{bs:2d}: {c:5d} ({pct:5.1f}%)  {'█' * int(pct / 2)}")



def evaluate(n=40, max_steps=20, lam=0.05, n_test=20,
             model_path="bkz_dqn.pt", plot=True):

    print(f"Generating {n_test} test lattices (n={n})…")
    lattices = [make_lattice(n) for _ in range(n_test)]
    agent = _load_agent(model_path)

    print("\nRunning baselines…")
    all_results = _run_all(lattices, agent, n, max_steps)

    summaries = _print_tour_table(all_results)
    budget = float(np.mean([r["time"] for r in all_results["Fixed BKZ-30"]]))
    budget_summaries = _print_budget_table(all_results, budget, max_steps)
    _print_action_dist(all_results)

    if plot:
        _plot(all_results, budget_summaries, n, max_steps, budget)

    return summaries



def evaluate_transfer(
    train_range: tuple[int, int] = (40, 65),
    test_dims: list[int] = [70, 75, 80],
    max_steps: int = 20,
    n_test: int = 20,
    model_path: str = "bkz_dqn_transfer.pt",
    plot: bool = True,
):
    """
    Compare RL (trained on train_range) vs baselines on each test dimension.
    The key question: does the policy generalise to unseen n?
    """
    agent = _load_agent(model_path)

    print(f"\nTransfer evaluation | train_range={train_range} | test_dims={test_dims}")
    print(f"{'n':>4}  {'Method':<20} {'Δlog‖b₁‖':>10} {'±':>7}  {'s/tour':>8}")
    print("-" * 55)

    transfer_data = {}   # n -> all_results dict (for plotting)

    for n in test_dims:
        print(f"\n── n={n} ──")
        lattices = [make_lattice(n) for _ in range(n_test)]
        all_results = _run_all(lattices, agent, n, max_steps)
        transfer_data[n] = all_results

        for name, results in all_results.items():
            s = _summary(results)
            marker = " ◄" if name == "RL (DQN)" else ""
            print(f"{n:>4}  {name:<20} {s['mean_improvement']:>10.4f} "
                  f"{s['std_improvement']:>7.4f}  {s['mean_time_per_tour']:>8.3f}{marker}")

    if plot:
        _plot_transfer(transfer_data, train_range, max_steps)

    return transfer_data




def _plot(all_results, budget_summaries, n, max_steps, budget):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not available)")
        return

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    method_names = list(all_results.keys())
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"RL-BKZ evaluation  (n={n}, {max_steps} tours)", fontsize=13)

    # Panel 1: reduction curves
    ax = axes[0, 0]
    for (name, results), col in zip(all_results.items(), colors):
        curves = np.array([r["hist"] for r in results])
        mean, std = curves.mean(0), curves.std(0)
        steps = np.arange(len(mean))
        ax.plot(steps, mean, label=name, lw=2.5 if name=="RL (DQN)" else 1.5,
                ls="-" if name=="RL (DQN)" else "--", color=col)
        ax.fill_between(steps, mean-std, mean+std, alpha=0.12, color=col)
    ax.set_xlabel("BKZ tour"); ax.set_ylabel("log‖b₁‖")
    ax.set_title("Reduction curves (mean ± 1σ, equal tours)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Panel 2: box plot
    ax = axes[0, 1]
    bp = ax.boxplot([[r["improvement"] for r in all_results[m]] for m in method_names],
                    patch_artist=True)
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col); patch.set_alpha(0.7)
    ax.set_xticks(range(1, len(method_names)+1))
    ax.set_xticklabels(method_names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Δ log‖b₁‖"); ax.set_title("Total improvement (equal tours)")
    ax.grid(True, axis="y", alpha=0.3)

    # Panel 3: budget bar chart
    ax = axes[1, 0]
    means = [budget_summaries[m].mean() for m in method_names]
    stds  = [budget_summaries[m].std()  for m in method_names]
    bars = ax.bar(np.arange(len(method_names)), means, yerr=stds, capsize=4,
                  color=colors[:len(method_names)], alpha=0.8,
                  error_kw={"elinewidth": 1.5})
    rl_idx = method_names.index("RL (DQN)")
    bars[rl_idx].set_edgecolor("black"); bars[rl_idx].set_linewidth(2)
    ax.set_xticks(np.arange(len(method_names)))
    ax.set_xticklabels(method_names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("log‖b₁‖ reached")
    ax.set_title(f"Quality within fixed time budget ({budget:.1f}s)\n"
                 f"(lower = better; budget = {max_steps}× Fixed BKZ-30 cost)")
    margin = 0.02
    ax.set_ylim(max(means)+margin, min(means)-margin)
    ax.grid(True, axis="y", alpha=0.3)

    # Panel 4: wall-clock curve
    ax = axes[1, 1]
    for (name, results), col in zip(all_results.items(), colors):
        max_t = max(r["time"] for r in results)
        t_grid = np.linspace(0, max_t, 200)
        interps = [np.interp(t_grid,
                             np.concatenate([[0.], np.cumsum(r["tour_times"])]),
                             r["hist"]) for r in results]
        mean = np.mean(interps, axis=0)
        ax.plot(t_grid, mean, label=name, lw=2.5 if name=="RL (DQN)" else 1.5,
                ls="-" if name=="RL (DQN)" else "--", color=col)
    ax.axvline(budget, color="grey", ls=":", lw=1.5, label=f"budget ({budget:.1f}s)")
    ax.set_xlabel("Wall-clock time (s)"); ax.set_ylabel("log‖b₁‖")
    ax.set_title("Reduction vs actual time (mean)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = "rl_bkz_eval.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out}")
    plt.show()


def _plot_transfer(transfer_data, train_range, max_steps):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    dims = sorted(transfer_data.keys())
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    method_names = list(next(iter(transfer_data.values())).keys())

    fig, axes = plt.subplots(1, len(dims), figsize=(5 * len(dims), 5), sharey=False)
    if len(dims) == 1:
        axes = [axes]
    fig.suptitle(f"Transfer evaluation | trained on n∈{train_range}, tested on held-out n",
                 fontsize=12)

    for ax, n in zip(axes, dims):
        all_results = transfer_data[n]
        steps = np.arange(max_steps + 1)
        for (name, results), col in zip(all_results.items(), colors):
            curves = np.array([r["hist"] for r in results])
            mean = curves.mean(0)
            lw = 2.5 if name == "RL (DQN)" else 1.5
            ls = "-"  if name == "RL (DQN)" else "--"
            ax.plot(steps, mean, label=name, lw=lw, ls=ls, color=col)
        ax.set_title(f"n={n} (unseen)")
        ax.set_xlabel("BKZ tour")
        ax.set_ylabel("log‖b₁‖")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = "rl_bkz_transfer.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nTransfer plot saved → {out}")
    plt.show()



def _parse_args():
    p = argparse.ArgumentParser(description="Evaluate RL-BKZ vs baselines")
    p.add_argument("--n",         type=int,   default=40)
    p.add_argument("--max_steps", type=int,   default=20)
    p.add_argument("--lam",       type=float, default=0.05)
    p.add_argument("--n_test",    type=int,   default=20)
    p.add_argument("--model",     type=str,   default="bkz_dqn.pt")
    p.add_argument("--no_plot",   action="store_true")
    p.add_argument("--transfer",  action="store_true",
                   help="Run transfer evaluation instead of standard")
    p.add_argument("--train_range", type=int, nargs=2, default=[40, 65],
                   metavar=("LO", "HI"))
    p.add_argument("--test_dims", type=int, nargs="+", default=[70, 75, 80])
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.transfer:
        evaluate_transfer(
            train_range=tuple(args.train_range),
            test_dims=args.test_dims,
            max_steps=args.max_steps,
            n_test=args.n_test,
            model_path=args.model,
            plot=not args.no_plot,
        )
    else:
        evaluate(
            n=args.n,
            max_steps=args.max_steps,
            lam=args.lam,
            n_test=args.n_test,
            model_path=args.model,
            plot=not args.no_plot,
        )