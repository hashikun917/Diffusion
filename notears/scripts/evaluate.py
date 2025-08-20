# run_notears_eval.py
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from notears.linear import notears_linear
from notears.utils import (
    simulate_dag,
    simulate_parameter,
    simulate_linear_sem,
    simulate_nonlinear_sem,
    count_accuracy,
)

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate NOTEARS on linear/nonlinear SEM")
    p.add_argument("--graph_type", type=str, default="ER",
                   help="Graph type for simulate_dag (e.g., ER, SF)")
    p.add_argument("--s0_scale", type=float, default=2,
                   help="Expected #edges = s0_scale * d (passed to simulate_dag)")
    p.add_argument("--gamma_scale", type=float, default=3.0,
                   help="Nonlinearity scale (gamma = gamma_scale * ones(d)) for tanh case")
    p.add_argument("--lambda1", type=float, default=0.1,
                   help="L1 regularization for notears_linear")
    p.add_argument("--w_threshold", type=float, default=0.3,
                   help="Edge threshold applied inside notears_linear")
    p.add_argument("--out_dir", type=str, required=True,
                   help="Directory to save CSV")
    # （必要なら下の2行をコメントアウト解除してCLIから掃引レンジを変えられます）
    # p.add_argument("--nodes", type=int, nargs="+", default=[10, 20, 50, 100])
    # p.add_argument("--samples", type=int, nargs="+", default=[100, 1000, 10000])
    return p.parse_args()

def main():
    args = parse_args()

    # 固定のレンジ（必要なら parse_args のコメントを外してCLIから渡してください）
    nodes_list   = [10, 20, 50, 100]
    samples_list = [100, 1000, 10000]
    cases        = ["linear", "tanh"]  # tanh=弱/強はgamma_scaleで調整

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for d in nodes_list:
        s0 = int(round(args.s0_scale * d))
        for n in samples_list:
            for case in cases:
                # --- グラフ生成 ---
                B_true = simulate_dag(d, s0, args.graph_type)
                W_true = simulate_parameter(B_true)

                # --- データ生成 ---
                if case == "linear":
                    X = simulate_linear_sem(W_true, n, sem_type="gauss")
                elif case == "tanh":
                    gamma = args.gamma_scale * np.ones(d)
                    X = simulate_nonlinear_sem(W_true, gamma, n, sem_type="tanh")
                else:
                    raise ValueError(f"Unknown case: {case}")

                # --- 因果探索（NOTEARS） ---
                W_est = notears_linear(
                    X, args.lambda1, loss_type="l2", w_threshold=args.w_threshold
                )
                B_est = (W_est != 0)

                # --- 評価 ---
                acc = count_accuracy(B_true, B_est)
                results.append(
                    {
                        "n_nodes": d,
                        "n_samples": n,
                        "case": case,
                        "shd": acc["shd"],
                        "fdr": acc["fdr"],
                    }
                )

    df = pd.DataFrame(results)

    # ファイル名（パラメータを含めて再現性を担保）
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    fname = (
        f"notears_eval_{args.graph_type}_s0{args.s0_scale}_g{args.gamma_scale}"
        f"_lam{args.lambda1}_thr{args.w_threshold}_{ts}.csv"
    )
    out_path = out_dir / fname
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
