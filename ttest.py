
import argparse
import numpy as np
from scipy import stats

from env import make_lattice, OBS_DIM, BLOCK_SIZES
from agent import DQNAgent
from baselines import BASELINES, run_rl


def _budget_final(results, budget):
    """Final log‖b₁‖ reached before wall-clock budget is exhausted."""
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


def run_ttest(dims, n_test, max_steps, model_path, test_type, budget):
    agent = DQNAgent(OBS_DIM, len(BLOCK_SIZES))
    agent.load(model_path)
    print(f"Loaded model: {model_path}")
    print(f"test_type={test_type}  n_test={n_test}  max_steps={max_steps}"
          + (f"  budget={budget}s" if test_type == "budget" else "") + "\n")

    if test_type == "improvement":
        metric_label = "Δlog‖b₁‖ (higher=better)"
        better = "+"
    else:
        metric_label = "final log‖b₁‖ within budget (lower=better)"
        better = "-"

    print(f"Metric: {metric_label}")
    print(f"{'n':>4}  {'RL':>8} {'±':>6}  {'Prog':>8} {'±':>6}  {'delta':>7}  {'t':>6}  {'p':>8}  sig")


    for n in dims:
        lattices = [make_lattice(n) for _ in range(n_test)]
        rl_vals, prog_vals = [], []

        for i, B in enumerate(lattices, 1):
            # Progressive
            hist, t, _, tour_times = BASELINES["Progressive"](B, max_steps)
            prog_r = {"hist": hist, "time": t, "tour_times": tour_times}

            # RL
            hist, t, _, tour_times = run_rl(agent, B, n, max_steps)
            rl_r = {"hist": hist, "time": t, "tour_times": tour_times}

            if test_type == "improvement":
                prog_vals.append(prog_r["hist"][0] - prog_r["hist"][-1])
                rl_vals.append(rl_r["hist"][0] - rl_r["hist"][-1])
            else:
                # budget: use each method's own avg BKZ-30 cost if not specified
                b = budget
                prog_vals.append(_budget_final([prog_r], b)[0])
                rl_vals.append(_budget_final([rl_r], b)[0])

            if i % 10 == 0:
                print(f"  n={n}: {i}/{n_test}...", flush=True)

        rl   = np.array(rl_vals)
        prog = np.array(prog_vals)

        # for budget test, lower is better so flip sign for t-stat interpretation
        if test_type == "budget":
            t_stat, p = stats.ttest_rel(-rl, -prog)
        else:
            t_stat, p = stats.ttest_rel(rl, prog)

        delta = rl.mean() - prog.mean()
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        print(f"{n:>4}  {rl.mean():>8.4f} {rl.std():>6.4f}  "
              f"{prog.mean():>8.4f} {prog.std():>6.4f}  "
              f"{delta:>+7.4f}  {t_stat:>6.3f}  {p:>8.4f}  {sig}")

    print("\n* p<0.05  ** p<0.01  *** p<0.001  ns=not significant")
    if test_type == "budget":
        print(f"(delta = RL final - Prog final; negative = RL reached lower norm = better)")


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--dims",       type=int,   nargs="+", default=[70, 75, 80, 85, 90, 95])
    p.add_argument("--n_test",     type=int,   default=50)
    p.add_argument("--max_steps",  type=int,   default=25)
    p.add_argument("--model",      type=str,   default="bkz_dqn_transfer.pt")
    p.add_argument("--test_type",  type=str,   default="improvement",
                   choices=["improvement", "budget"],
                   help="improvement: equal tours | budget: equal wall-clock time")
    p.add_argument("--budget",     type=float, default=0.5,
                   help="Wall-clock budget in seconds (only used with --test_type budget)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    run_ttest(args.dims, args.n_test, args.max_steps, args.model,
              args.test_type, args.budget)