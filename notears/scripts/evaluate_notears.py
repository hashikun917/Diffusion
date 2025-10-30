# run_notears_eval.py
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from notears.nonlinear import NotearsMLP, NotearsSobolev
from notears.linear import notears_linear
from notears.utils import (
    simulate_dag,
    simulate_parameter,
    simulate_linear_sem,
    simulate_nonlinear_sem,
    count_accuracy,
)
from notears.nonlinear import notears_nonlinear

from evaluate.utils.save_results import save_params

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
    p.add_argument("--lambda2", type=float, default=0.1,
                   help="Regularization parameter for notears_nonlinear")
    p.add_argument("--nh", type=int, default=10, help="Number of hidden units for notears_nonlinear")
    p.add_argument("--num_expansion", type=int, default=10, help="Number of expansion for notears_nonlinear")
    p.add_argument("--w_threshold", type=float, default=0.3,
                   help="Edge threshold applied inside notears_linear")
    p.add_argument("--n_seeds", type=int, default=1, help="乱数シードの数")
    p.add_argument("--out_dir", type=str, default="results/notears_evaluate",
                   help="Directory to save CSV")
    # （必要なら下の2行をコメントアウト解除してCLIから掃引レンジを変えられます）
    # p.add_argument("--nodes", type=int, nargs="+", default=[10, 20, 50, 100])
    # p.add_argument("--samples", type=int, nargs="+", default=[100, 1000, 10000])
    return p.parse_args()

def main():
    args = parse_args()

    # 固定のレンジ（必要なら parse_args のコメントを外してCLIから渡してください）
    nodes_list   = [10, 20, 50, 100]
    samples_list = [200, 1000]
    cases        = ["tanh"]  # ["linear", "tanh"]
    methods      = ["linear", "mlp", "sob"]

    results = []
    for d in nodes_list:
        s0 = int(round(args.s0_scale * d))
        for n in samples_list:
            for case in cases:
                for method in methods:
                    
                    shd_list = []
                    fdr_list = []
                    for seed in range(args.n_seeds):
                        # --- グラフ生成 ---
                        B_true = simulate_dag(d, s0, args.graph_type)
                        W_true = simulate_parameter(B_true)

                        # --- データ生成 ---
                        if case == "linear":
                            X = simulate_linear_sem(W_true, n, sem_type="gauss").astype(np.float32)
                        elif case == "tanh":
                            gamma = args.gamma_scale * np.ones(d)
                            X = simulate_nonlinear_sem(W_true, gamma, n, sem_type="tanh").astype(np.float32)
                        else:
                            raise ValueError(f"Unknown case: {case}")

                        # --- 因果探索（NOTEARS） ---
                        if method == "linear":
                            W_est = notears_linear(
                                X, args.lambda1, loss_type="l2", w_threshold=args.w_threshold
                            )
                        elif method == "mlp":
                            model = NotearsMLP(dims=[d, args.nh, 1], bias=True)
                            W_est = notears_nonlinear(model, X, lambda1=args.lambda2, lambda2=args.lambda2, w_threshold=args.w_threshold)
                        elif method == "sob":
                            model = NotearsSobolev(d, args.num_expansion)
                            W_est = notears_nonlinear(model, X, lambda1=args.lambda2, lambda2=args.lambda2, w_threshold=args.w_threshold)
                            
                            
                        B_est = (W_est != 0)

                        # --- 評価 ---
                        acc = count_accuracy(B_true, B_est, check_dag=False)
                        shd_list.append(acc["shd"])
                        fdr_list.append(acc["fdr"])
                        
                    results.append(
                        {
                            "n_nodes": d,
                            "n_samples": n,
                            "case": case,
                            "method": method,
                            "shd": np.mean(shd_list),
                            "fdr": np.mean(fdr_list),
                        }
                    )

    df = pd.DataFrame(results)

    # ファイル名（パラメータを含めて再現性を担保）
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir) / now
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "notears_eval.csv", index=False)
    
    params = {
        "graph_type": args.graph_type,
        "s0_scale": args.s0_scale,
        "gamma_scale": args.gamma_scale,
        "lambda1": args.lambda1,
        "lambda2": args.lambda2,
        "nh": args.nh,
        "num_expansion": args.num_expansion,
        "w_threshold": args.w_threshold,
        "n_seeds": args.n_seeds,
    }
    save_params(params, out_dir)
    

if __name__ == "__main__":
    main()

# python実行コマンド例
# python evaluate.py --out_dir results/notears_evaluate --n_seeds 1 --graph_type ER --s0_scale 2 --gamma_scale 3.0 --lambda1 0.1 --lambda2 0.1 --nh 10 --num_expansion 10 --w_threshold 0.3