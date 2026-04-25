"""
Generate fplll pruning strategies and write to ~/fplll_strategies/default.json.

Run once before using the fplll Default baseline:
    python generate_strategies.py

The strategy file path compiled into fpylll's macOS binary is read-only, so
we write to ~/fplll_strategies/ and load from there at runtime instead.
"""

import os
from fpylll.fplll.bkz_param import Strategy, dump_strategies_json
from fpylll import Pruning, IntegerMatrix, LLL, GSO


def generate(out_path: str, max_beta: int = 30) -> None:
    strategies = []
    for beta in range(0, max_beta + 1):
        if beta < 10:
            s = Strategy(beta)
        else:
            B = IntegerMatrix.random(beta, "qary", k=beta // 2, bits=20)
            LLL.reduction(B)
            M = GSO.Mat(B, float_type="double")
            M.update_gso()
            r = [M.get_r(i, i) for i in range(beta)]
            try:
                pruning = Pruning.run(
                    r[0], 10000, [r], 0.51,
                    metric="solutions", float_type="double",
                )
                prepro = [beta - 4] if beta > 14 else []
                s = Strategy(beta, prepro, [pruning])
            except Exception:
                s = Strategy(beta)
        strategies.append(s)
        print(f"  beta={beta:2d} done", flush=True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    dump_strategies_json(out_path, strategies)
    print(f"\nWritten {len(strategies)} strategies → {out_path}")


if __name__ == "__main__":
    path = os.path.expanduser("~/fplll_strategies/default.json")
    print(f"Generating fplll strategies → {path}\n")
    generate(path)
    print("\nVerifying…")
    from fpylll import load_strategies_json
    loaded = load_strategies_json(path)
    print(f"Loaded {len(loaded)} strategies OK")
    print("Run evaluate.py — fplll Default is now a real baseline.")
