#!/usr/bin/env python3
# carefl_eval_nonlinear_notears_variants.py
# 評価対象: 非線形(tanh)のみ。CSV列: n_nodes, notears_type, model, neg_log_likelihood_per_dim

import argparse
from pathlib import Path
from datetime import datetime
from typing import Callable, Optional, Tuple, Union

import numpy as np
import pandas as pd
import igraph as ig

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.distributions import MultivariateNormal
from tqdm.auto import tqdm

from carefl.nflib.flows import DAGAffineCL, NormalizingFlowModel
from carefl.data.generate_synth_data import CustomSyntheticDataset
from carefl.nflib.nets import MLP4

from notears.utils import (
    simulate_dag,
    simulate_parameter,
    simulate_nonlinear_sem,
)
from notears.linear import notears_linear
from notears.nonlinear import NotearsMLP, NotearsSobolev, notears_nonlinear


# =========================
# 真の結合対数尤度ファクトリ（加法的ガウス）
# =========================
def _normal_logpdf(x: np.ndarray, mu: np.ndarray, sigma: Union[float, np.ndarray]) -> np.ndarray:
    sigma = np.asarray(sigma)
    var = sigma ** 2
    return -0.5 * (np.log(2.0 * np.pi * var) + ((x - mu) ** 2) / var)

def build_sem_logpdf(
    W: np.ndarray,
    gamma: Optional[np.ndarray],
    sem_type: str = "tanh",
    noise_scale: Union[float, np.ndarray] = 1.0,
) -> Tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    """X[n,d]→log p(X) を返す関数を生成（tanh/linear対応、ここではtanh使用）"""
    W = np.asarray(W)
    d = W.shape[0]

    sigma = (np.ones(d) * float(noise_scale)) if np.isscalar(noise_scale) else np.asarray(noise_scale, dtype=float)
    if sigma.shape != (d,):
        raise ValueError("noise_scale must be scalar or length-d vector")

    if gamma is None:
        gamma_vec = np.ones(d, dtype=float)
    else:
        gamma_vec = np.asarray(gamma, dtype=float)
        if gamma_vec.shape != (d,):
            raise ValueError("gamma must be length-d vector or None")

    G = ig.Graph.Weighted_Adjacency(W.tolist())
    order = G.topological_sorting()
    parents = [G.neighbors(j, mode=ig.IN) for j in range(d)]

    def _cond_mean(X: np.ndarray, j: int) -> np.ndarray:
        pa = parents[j]
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


# =========================
# CAREFL 学習
# =========================
def train_carefl(X: np.ndarray, B_bool: np.ndarray, epochs: int = 200) -> NormalizingFlowModel:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d = X.shape[1]

    dset = CustomSyntheticDataset(X.astype(np.float64), device)
    train_loader = DataLoader(dset, shuffle=True, batch_size=128)

    prior = MultivariateNormal(torch.zeros(d, device=device), torch.eye(d, device=device))
    G = ig.Graph.Adjacency(B_bool.tolist(), mode='directed')
    order = G.topological_sorting()
    flow_list = []
    for v in order[::-1]:
        cond_idx = G.neighbors(v, mode=ig.IN)
        affine = DAGAffineCL(d, cond_idx, [v], MLP4, 100, False, True)
        flow_list.append(affine)

    flow = NormalizingFlowModel(prior, flow_list).to(device)
    flow.train()
    opt = optim.Adam(flow.parameters(), lr=1e-3)

    for _ in tqdm(range(epochs)):
        for _, xb in enumerate(train_loader):
            xb = xb.to(device)
            _, prior_logprob, log_det = flow(xb)
            loss = - torch.sum(prior_logprob + log_det)
            opt.zero_grad()
            loss.backward()
            opt.step()

    return flow


# =========================
# 因果探索：タイプ別推定
# =========================
def estimate_B(X: np.ndarray, notears_type: str, lamb: float, w_threshold: float, nh: int, num_expansion: int) -> np.ndarray:
    """各 notears_type に基づき B(bool) を返す"""
    d = X.shape[1]

    if notears_type == "unused":
        raise ValueError("estimate_B called with 'unused'. Provide true graph instead.")

    if notears_type == "notears-linear":
        W_est = notears_linear(X, lamb, loss_type='l2', w_threshold=w_threshold)
        return (W_est != 0)

    elif notears_type == "notears-mlp":
        model = NotearsMLP(dims=[d, nh, 1], bias=True)
        W_est = notears_nonlinear(model, X, lambda1=lamb, lambda2=lamb, w_threshold=w_threshold)
        return (W_est != 0)

    elif notears_type == "notears-sob":
        model = NotearsSobolev(d, num_expansion)
        W_est = notears_nonlinear(model, X, lambda1=lamb, lambda2=lamb, w_threshold=w_threshold)
        return (W_est != 0)

    else:
        raise ValueError(f"Unknown notears_type: {notears_type}")


# =========================
# 1条件の評価：tanhのみ
# =========================
def eval_one_setting(
    d: int,
    n: int,
    s0_scale: float,
    gamma_scale: float,
    notears_type: str,
    lamb: float,
    w_threshold: float,
    nh: int,
    num_expansion: int,
    seed: int = 0,
) -> Tuple[float, float]:
    """
    戻り値: (nll_true_per_dim, nll_carefl_per_dim)
        - true は notears_type='unused' のときのみ返す想定。その他タイプ時は true を None に。
    """
    graph_type = "ER"
    s0 = int(round(s0_scale * d))
    B_true = simulate_dag(d, s0, graph_type)
    W_true = simulate_parameter(B_true)
    gamma = gamma_scale * np.ones(d)

    # 真のSEMで学習データ生成（tanh, additive Gaussian）
    X = simulate_nonlinear_sem(W_true, gamma, n, sem_type='tanh')

    # 真の log p(X)（結合）
    logpdf_fn, _ = build_sem_logpdf(W_true, gamma, sem_type="tanh", noise_scale=1.0)
    d_float = float(d)

    # baseline: true model
    nll_true = -logpdf_fn(X).mean() / d_float

    # notears_type に応じて B を準備
    if notears_type == "unused":
        B_use = B_true
    else:
        B_use = estimate_B(X, notears_type, lamb=lamb, w_threshold=w_threshold, nh=nh, num_expansion=num_expansion)

    # CAREFL 学習→生成→NLL
    flow = train_carefl(X, B_use)
    X_gen = flow.sample(n)[-1].detach().cpu().numpy()
    nll_carefl = -logpdf_fn(X_gen).mean() / d_float

    return nll_true if notears_type == "unused" else None, nll_carefl


# =========================
# CLI
# =========================
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate CAREFL NLL (tanh) with notears variants")
    p.add_argument("--n", type=int, default=1024, help="訓練/生成に用いるサンプル数")
    p.add_argument("--gamma_scale", type=float, default=3.0, help="tanh のスケール γ")
    p.add_argument("--s0_scale", type=float, default=2.0, help="平均エッジ数のスケール（s0 = s0_scale * d）")

    p.add_argument("--lambda_", dest="lambda_", type=float, default=0.01, help="notears-mlp/sob の正則化係数（lambda1=lambda2=lambda_）")
    p.add_argument("--w_threshold", type=float, default=0.3, help="notears-mlp/sob の閾値")
    p.add_argument("--nh", type=int, default=10, help="notears-mlp の中間層ユニット数")
    p.add_argument("--num_expansion", type=int, default=10, help="notears-sob の展開次数")

    p.add_argument("--out_dir", type=str, default="results/carefl_evaluate", help="CSV出力先ディレクトリ")
    p.add_argument("--seeds", type=int, nargs="+", default=[0], help="乱数シード（必要なら複数）")
    return p.parse_args()


# =========================
# メイン
# =========================
def main():
    torch.set_default_dtype(torch.double)
    
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nodes_list = [10, 20, 50, 100]
    notears_types = ["unused", "notears-linear", "notears-mlp", "notears-sob"]

    rows = []
    for d in nodes_list:
        for seed in args.seeds:
            # ベースライン（true model）は notears_type='unused' のときだけ記録
            nll_true, _ = eval_one_setting(
                d=d, n=args.n, s0_scale=args.s0_scale, gamma_scale=args.gamma_scale,
                notears_type="unused", lamb=args.lambda_, w_threshold=args.w_threshold,
                nh=args.nh, num_expansion=args.num_expansion, seed=seed,
            )
            rows.append({
                "n_nodes": d,
                "notears_type": "unused",
                "model": "true model",
                "neg_log_likelihood_per_dim": float(nll_true),
                "n": args.n,
                "gamma_scale": args.gamma_scale,
                "s0_scale": args.s0_scale,
                "lambda": args.lambda_,
                "w_threshold": args.w_threshold,
                "nh": args.nh,
                "num_expansion": args.num_expansion,
                "seed": seed,
            })

            # CAREFL（因果探索の違いで比較）
            for nt in notears_types:
                nll_true_tmp, nll_carefl = eval_one_setting(
                    d=d, n=args.n, s0_scale=args.s0_scale, gamma_scale=args.gamma_scale,
                    notears_type=nt, lamb=args.lambda_, w_threshold=args.w_threshold,
                    nh=args.nh, num_expansion=args.num_expansion, seed=seed,
                )
                rows.append({
                    "n_nodes": d,
                    "notears_type": nt,
                    "model": "carefl",
                    "neg_log_likelihood_per_dim": float(nll_carefl),
                    "n": args.n,
                    "gamma_scale": args.gamma_scale,
                    "s0_scale": args.s0_scale,
                    "lambda": args.lambda_,
                    "w_threshold": args.w_threshold,
                    "nh": args.nh,
                    "num_expansion": args.num_expansion,
                    "seed": seed,
                })

    df = pd.DataFrame(rows)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    fname = (
        f"carefl_eval_tanh_n{args.n}_gamma{args.gamma_scale}_s0{args.s0_scale}"
        f"_lam{args.lambda_}_thr{args.w_threshold}_nh{args.nh}_sob{args.num_expansion}_{ts}.csv"
    )
    out_path = Path(args.out_dir) / fname
    df.to_csv(out_path, index=False)

    print(f"Saved CSV: {out_path}")
    print(df.head())


if __name__ == "__main__":
    main()
