#!/usr/bin/env python3
# carefl_evaluate_likelihood.py

import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import igraph as ig
from typing import Callable, Optional, Union, Tuple

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.distributions import MultivariateNormal
from tqdm.auto import tqdm

from carefl.nflib.flows import DAGAffineCL, NormalizingFlowModel
from carefl.data.generate_synth_data import CustomSyntheticDataset
from carefl.nflib.nets import MLP4

from notears.utils import (
    simulate_linear_sem,
    simulate_nonlinear_sem,
    simulate_dag,
    simulate_parameter,
)
from notears.linear import notears_linear

import pandas as pd


# ------------------------------
# 既存：真の結合尤度ファクトリ
# ------------------------------
def _normal_logpdf(x: np.ndarray, mu: np.ndarray, sigma: Union[float, np.ndarray]) -> np.ndarray:
    sigma = np.asarray(sigma)
    var = sigma ** 2
    return -0.5 * (np.log(2.0 * np.pi * var) + ((x - mu) ** 2) / var)

def build_sem_logpdf(
    W: np.ndarray,
    gamma: Optional[np.ndarray],
    sem_type: str = "linear",
    noise_scale: Union[float, np.ndarray] = 1.0,
) -> Tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    W = np.asarray(W)
    d = W.shape[0]

    if np.isscalar(noise_scale):
        sigma = float(noise_scale) * np.ones(d)
    else:
        sigma = np.asarray(noise_scale, dtype=float)
        if sigma.shape != (d,):
            raise ValueError("noise_scale must be a scalar or length-d vector.")

    if gamma is None:
        gamma_vec = np.ones(d, dtype=float)
    else:
        gamma_vec = np.asarray(gamma, dtype=float)
        if gamma_vec.shape != (d,):
            raise ValueError("gamma must be length-d vector or None.")

    G = ig.Graph.Weighted_Adjacency(W.tolist())
    order = G.topological_sorting()
    if len(order) != d:
        raise ValueError("W must define a DAG (topological_sorting failed).")
    parents_list = [G.neighbors(j, mode=ig.IN) for j in range(d)]

    def _cond_mean(X: np.ndarray, j: int) -> np.ndarray:
        pa = parents_list[j]
        z = 0.0 if len(pa) == 0 else X[:, pa] @ W[pa, j]
        if sem_type == "linear":
            return z
        elif sem_type == "tanh":
            return gamma_vec[j] * np.tanh(z)
        else:
            raise ValueError(f"Unsupported sem_type: {sem_type}")

    def logpdf_fn(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != d:
            raise ValueError(f"X must have shape [n, {d}]")
        n = X.shape[0]
        logp = np.zeros(n, dtype=float)
        for j in order:
            mu_j = _cond_mean(X, j)
            logp += _normal_logpdf(X[:, j], mu_j, sigma[j])
        return logp

    def pdf_fn(X: np.ndarray) -> np.ndarray:
        return np.exp(logpdf_fn(X))

    return logpdf_fn, pdf_fn


# ------------------------------
# 既存：CAREFL 学習
# ------------------------------

def train_carefl(X, B):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d = X.shape[1]

    dset = CustomSyntheticDataset(X.astype(np.float32), device)
    train_loader = DataLoader(dset, shuffle=True, batch_size=128)

    prior = MultivariateNormal(torch.zeros(d).to(device), torch.eye(d).to(device))
    G = ig.Graph.Adjacency(B.tolist(), mode='directed')
    ordered_vertices = G.topological_sorting()
    flow_list = []

    for v in ordered_vertices[::-1]:
        cond_idx = G.neighbors(v, mode=ig.IN)
        affine = DAGAffineCL(d, cond_idx, [v], MLP4, 100, False, True)
        flow_list.append(affine)

    flow = NormalizingFlowModel(prior, flow_list).to(device)
    flow.train()
    optimizer = optim.Adam(flow.parameters(), lr=1e-3)

    epochs = 200
    for _ in tqdm(range(epochs)):
        for _, x in enumerate(train_loader):
            x = x.to(device)
            _, prior_logprob, log_det = flow(x)
            loss = - torch.sum(prior_logprob + log_det)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return flow


def nll_score_with_various_cases(W_true, use_notears, case, model):
    # 真の分布の logpdf
    d = W_true.shape[0]
    n_eval = 200
    if case == 'linear':
        X = simulate_linear_sem(W_true, n, sem_type='gauss')
        logpdf_fn, _ = build_sem_logpdf(W_true, gamma=None, sem_type="linear", noise_scale=1.0)
        if model == 'true model':
            idx = np.random.choice(X.shape[0], size=n_eval, replace=False)
            X_eval = X[idx]
            ll = logpdf_fn(X_eval)
            nll = -ll.mean() / d
            return nll
    elif case == 'tanh':
        gamma = gamma_scale * np.ones(d)
        X = simulate_nonlinear_sem(W_true, gamma, n, sem_type='tanh')
        logpdf_fn, _ = build_sem_logpdf(W_true, gamma=gamma, sem_type="tanh", noise_scale=1.0)
        if model == 'true model':
            idx = np.random.choice(X.shape[0], size=n_eval, replace=False)
            X_eval = X[idx]
            ll = logpdf_fn(X_eval)
            nll = -ll.mean() / d
            return nll
    else:
        raise ValueError(f"Unknown case: {case}")

    # 構造推定（必要に応じて）
    if use_notears:
        W_est = notears_linear(X, 0.1, loss_type='l2', w_threshold=0.3)
        B = W_est != 0
    else:
        B = W_true != 0

    # CAREFL 学習 → 生成 → 真の分布で評価
    flow = train_carefl(X, B)
    X_gen = flow.sample(n_eval)[-1].detach().cpu().numpy()
    ll_gen = logpdf_fn(X_gen)
    nll_gen = -ll_gen.mean() / d
    return nll_gen


# ------------------------------
# CLI & メイン
# ------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate CAREFL likelihood across settings")
    p.add_argument("--n", type=int, default=1024, help="実験で用いるサンプル数（訓練/生成）")
    p.add_argument("--gamma_scale", type=float, default=3.0, help="tanhケースのスケール")
    p.add_argument("--s0_scale", type=float, default=2, help="平均エッジ数のスケール（s0 = s0_scale * d）")
    p.add_argument("--out_dir", type=str, required=True,
                   help="Directory to save CSV")
    # 任意：対象ノード数やNOTEARS有無などは既存ループに合わせて固定（最小変更）
    return p.parse_args()

def main():
    global n, gamma_scale
    args = parse_args()
    n = args.n
    gamma_scale = args.gamma_scale

    graph_type = 'ER'
    s0_scale = args.s0_scale

    results = []
    for d in [10, 20, 50, 100]:
        for use_notears in [True, False]:
            for case in ['linear', 'tanh']:
                for model in ['true model', 'carefl']:
                    s0 = int(round(s0_scale * d))
                    B_true = simulate_dag(d, s0, graph_type)
                    W_true = simulate_parameter(B_true)
                    nll = nll_score_with_various_cases(W_true, use_notears, case, model)
                    results.append({
                        "n_nodes": d,
                        "use_notears": use_notears,
                        "case": case,
                        "model": model,
                        "neg_log_likelihood_per_dim": nll,
                    })

    df = pd.DataFrame(results)

    # 出力先とファイル名
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    fname = f"carefl_eval_n{n}_gamma{gamma_scale}_s0{s0_scale}_{ts}.csv"
    out_path = out_dir / fname

    df.to_csv(out_path, index=False)
    print(f"Saved CSV: {out_path}")
    print(df.head())

if __name__ == "__main__":
    main()
