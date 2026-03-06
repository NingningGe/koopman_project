#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compare_franka_ur_alignment_report.py

用途：
基于你这次新的训练结果，系统比较：
1. Franka / UR 原始数据质量
2. Franka source latent 健康度
3. UR 对齐前 / 对齐后 latent 质量
4. Franka vs UR(mapped) 分布接近程度
5. Koopman 一步预测与多步 rollout 指标
6. 生成图像 + JSON + 文本报告

默认针对：
- 新 Franka v2 数据
- 新 out_align_koopman_3 训练结果

输出目录：
/home/nng/koopman_project/evaluate/compare_franka_ur_alignment_report
"""

import os
import json
from dataclasses import dataclass, asdict
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# =========================================================
# Config
# =========================================================
@dataclass
class Cfg:
    # datasets
    franka_dataset: str = "/home/nng/koopman_project/data_wang_franka_reach_ablation/dataset_wang_franka_reach_ablation_v2.pt"
    ur_dataset: str = "/home/nng/koopman_project/data_wang_ur_reach_ablation/dataset_wang_ur_reach_ablation.pt"

    # training outputs
    src_ckpt: str = "/home/nng/koopman_project/out_align_koopman_3/src_franka/src_agent.pt"
    tgt_ckpt: str = "/home/nng/koopman_project/out_align_koopman_3/tgt_ur/tgt_agent.pt"
    stage2_bundle: str = "/home/nng/koopman_project/out_align_koopman_3/tgt_ur/stage2_bundle.pt"
    koopman_ckpt: str = "/home/nng/koopman_project/out_align_koopman_3/koopman_fit/aligned_latent_and_koopman.pt"

    out_dir: str = "/home/nng/koopman_project/evaluate/compare_franka_ur_alignment_report"

    # model dims (must match training)
    franka_obs_dim: int = 14
    franka_act_dim: int = 7
    ur_obs_dim: int = 12
    ur_act_dim: int = 6
    lat_obs_dim: int = 8
    lat_act_dim: int = 8
    hidden_dim: int = 256
    n_layers: int = 3

    # analysis
    seed: int = 0
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    pca_points: int = 10000
    latent_points: int = 10000
    hist_bins: int = 80
    raw_hist_dims_show: int = 8
    sat_threshold: float = 0.95
    episode_len: int = 200
    rollout_horizon: int = 50
    encode_bs: int = 4096


cfg = Cfg()


# =========================================================
# Basic utils
# =========================================================
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_dataset(path: str) -> Dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu")
    return obj["data"] if isinstance(obj, dict) and "data" in obj else obj


@torch.no_grad()
def flatten_seq(s: torch.Tensor, a: torch.Tensor, s_next: torch.Tensor):
    return s.reshape(-1, s.shape[-1]), a.reshape(-1, a.shape[-1]), s_next.reshape(-1, s_next.shape[-1])


@torch.no_grad()
def sample_rows(x: torch.Tensor, n: int) -> torch.Tensor:
    idx = torch.randperm(x.shape[0])[:min(n, x.shape[0])]
    return x[idx]


@torch.no_grad()
def pca_2d(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    x = x.float()
    x = x - x.mean(dim=0, keepdim=True)
    _, S, Vh = torch.linalg.svd(x, full_matrices=False)
    y = x @ Vh[:2].T
    eigvals = (S ** 2) / max(x.shape[0] - 1, 1)
    vr = eigvals / eigvals.sum()
    return y, vr


def tensor_stats(x: torch.Tensor) -> Dict:
    return {
        "shape": list(x.shape),
        "mean": x.mean(dim=0).tolist(),
        "std": x.std(dim=0).tolist(),
        "min": x.min(dim=0).values.tolist(),
        "max": x.max(dim=0).values.tolist(),
    }


def scalar_range_stats(x: torch.Tensor) -> Dict:
    return {
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "mean": float(x.mean().item()),
        "std": float(x.std().item()),
        "q01": float(torch.quantile(x, 0.01).item()),
        "q05": float(torch.quantile(x, 0.05).item()),
        "q50": float(torch.quantile(x, 0.50).item()),
        "q95": float(torch.quantile(x, 0.95).item()),
        "q99": float(torch.quantile(x, 0.99).item()),
    }


def compute_cov_rank(x: torch.Tensor, eps_ratio: float = 1e-3) -> Dict:
    x = x.float()
    x = x - x.mean(dim=0, keepdim=True)
    cov = (x.T @ x) / max(x.shape[0] - 1, 1)
    eigvals = torch.linalg.eigvalsh(cov).real
    eigvals = torch.flip(eigvals, dims=[0])
    total = eigvals.sum().item()
    if total <= 0:
        return {
            "eigvals": eigvals.tolist(),
            "explained_ratio": [0.0 for _ in range(len(eigvals))],
            "effective_rank_eps_ratio_1e3": 0,
            "participation_ratio": 0.0,
        }
    explained = (eigvals / eigvals.sum()).tolist()
    thr = eigvals[0].item() * eps_ratio
    eff_rank = int((eigvals > thr).sum().item())
    participation_ratio = float((eigvals.sum() ** 2 / (eigvals.square().sum() + 1e-12)).item())
    return {
        "eigvals": eigvals.tolist(),
        "explained_ratio": explained,
        "effective_rank_eps_ratio_1e3": eff_rank,
        "participation_ratio": participation_ratio,
    }


def latent_quality(z: torch.Tensor, sat_threshold: float) -> Dict:
    sat = (torch.abs(z) > sat_threshold).float().mean(dim=0)
    std = z.std(dim=0)
    return {
        "mean": z.mean(dim=0).tolist(),
        "std": std.tolist(),
        "min": z.min(dim=0).values.tolist(),
        "max": z.max(dim=0).values.tolist(),
        "saturation_ratio": sat.tolist(),
        "avg_std": float(std.mean().item()),
        "avg_saturation": float(sat.mean().item()),
        "collapsed_dims_std_lt_0p03": [i for i, v in enumerate(std.tolist()) if v < 0.03],
        "saturated_dims_ratio_gt_0p5": [i for i, v in enumerate(sat.tolist()) if v > 0.5],
    }


def mean_cov_distance(x: torch.Tensor, y: torch.Tensor) -> Dict:
    """
    一个简单的分布接近度指标：
    - 均值 L2 距离
    - 协方差 Frobenius 距离
    """
    mx = x.mean(dim=0)
    my = y.mean(dim=0)
    xc = x - mx
    yc = y - my
    cx = (xc.T @ xc) / max(x.shape[0] - 1, 1)
    cy = (yc.T @ yc) / max(y.shape[0] - 1, 1)
    mean_l2 = torch.norm(mx - my).item()
    cov_fro = torch.norm(cx - cy).item()
    return {
        "mean_l2": float(mean_l2),
        "cov_fro": float(cov_fro),
    }


# =========================================================
# Model definitions (match training)
# =========================================================
def build_mlp(in_dim: int, out_dim: int, n_layers: int, hidden_dim: int,
              activation: str = "relu", out_act: str = "identity") -> nn.Module:
    acts = {
        "relu": nn.ReLU,
        "leaky_relu": lambda: nn.LeakyReLU(0.2),
        "tanh": nn.Tanh,
        "identity": nn.Identity,
    }
    layers = []
    d = in_dim
    for _ in range(n_layers):
        layers += [nn.Linear(d, hidden_dim), acts[activation]()]
        d = hidden_dim
    layers.append(nn.Linear(d, out_dim))
    if out_act != "identity":
        layers.append(acts[out_act]())
    return nn.Sequential(*layers)


class Agent(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, lat_obs_dim: int, lat_act_dim: int, n_layers: int, hidden_dim: int):
        super().__init__()
        self.obs_enc = build_mlp(obs_dim, lat_obs_dim, n_layers, hidden_dim, activation="relu", out_act="identity")
        self.obs_dec = build_mlp(lat_obs_dim, obs_dim, n_layers, hidden_dim, activation="relu", out_act="identity")
        self.act_enc = build_mlp(obs_dim + act_dim, lat_act_dim, n_layers, hidden_dim, activation="relu", out_act="tanh")
        self.act_dec = build_mlp(obs_dim + lat_act_dim, act_dim, n_layers, hidden_dim, activation="relu", out_act="tanh")
        self.inv_dyn = build_mlp(lat_obs_dim * 2, lat_act_dim, n_layers, hidden_dim, activation="relu", out_act="tanh")
        self.fwd_dyn = build_mlp(lat_obs_dim + lat_act_dim, lat_obs_dim, n_layers, hidden_dim, activation="relu", out_act="identity")
        self.actor = build_mlp(lat_obs_dim, lat_act_dim, n_layers, hidden_dim, activation="relu", out_act="tanh")


class MapNet(nn.Module):
    def __init__(self, dim: int, n_layers: int, hidden_dim: int, out_act: str):
        super().__init__()
        self.net = build_mlp(dim, dim, n_layers, hidden_dim, activation="relu", out_act=out_act)

    def forward(self, x):
        return self.net(x)


# =========================================================
# Encoding / rollout
# =========================================================
@torch.no_grad()
def encode_src(agent, s_all, a_all, sn_all, batch_size, device):
    z_list, u_list, zn_list = [], [], []
    agent.eval()
    for i in range(0, s_all.shape[0], batch_size):
        s = s_all[i:i+batch_size].to(device)
        a = a_all[i:i+batch_size].to(device)
        sn = sn_all[i:i+batch_size].to(device)
        z = agent.obs_enc(s)
        u = agent.act_enc(torch.cat([s, a], dim=-1))
        zn = agent.obs_enc(sn)
        z_list.append(z.cpu())
        u_list.append(u.cpu())
        zn_list.append(zn.cpu())
    return torch.cat(z_list, 0), torch.cat(u_list, 0), torch.cat(zn_list, 0)


@torch.no_grad()
def encode_tgt_before(agent, s_all, a_all, sn_all, batch_size, device):
    z_list, u_list, zn_list = [], [], []
    agent.eval()
    for i in range(0, s_all.shape[0], batch_size):
        s = s_all[i:i+batch_size].to(device)
        a = a_all[i:i+batch_size].to(device)
        sn = sn_all[i:i+batch_size].to(device)
        z = agent.obs_enc(s)
        u = agent.act_enc(torch.cat([s, a], dim=-1))
        zn = agent.obs_enc(sn)
        z_list.append(z.cpu())
        u_list.append(u.cpu())
        zn_list.append(zn.cpu())
    return torch.cat(z_list, 0), torch.cat(u_list, 0), torch.cat(zn_list, 0)


@torch.no_grad()
def encode_tgt_after(agent, z_t2s, u_t2s, z_mean, z_std, s_all, a_all, sn_all, batch_size, device):
    z_list, u_list, zn_list = [], [], []
    agent.eval()
    z_t2s.eval()
    u_t2s.eval()
    z_mean = z_mean.to(device)
    z_std = z_std.to(device)

    for i in range(0, s_all.shape[0], batch_size):
        s = s_all[i:i+batch_size].to(device)
        a = a_all[i:i+batch_size].to(device)
        sn = sn_all[i:i+batch_size].to(device)

        z_t = agent.obs_enc(s)
        u_t = agent.act_enc(torch.cat([s, a], dim=-1))
        zn_t = agent.obs_enc(sn)

        z_n = z_t2s(z_t)
        zn_n = z_t2s(zn_t)

        z = z_n * z_std + z_mean
        zn = zn_n * z_std + z_mean
        u = u_t2s(u_t)

        z_list.append(z.cpu())
        u_list.append(u.cpu())
        zn_list.append(zn.cpu())

    return torch.cat(z_list, 0), torch.cat(u_list, 0), torch.cat(zn_list, 0)


@torch.no_grad()
def rollout_mse(z_flat: torch.Tensor, u_flat: torch.Tensor, A: torch.Tensor, B: torch.Tensor, episode_len: int, horizon: int, device: torch.device):
    z = z_flat.to(device)
    u = u_flat.to(device)
    A = A.to(device)
    B = B.to(device)

    N = z.shape[0]
    assert N % episode_len == 0
    E = N // episode_len
    z_seq = z.view(E, episode_len, -1)
    u_seq = u.view(E, episode_len, -1)

    H = min(horizon, episode_len - 1)
    idx = torch.randperm(E, device=device)[: min(E, 64)]

    z_cur = z_seq[idx, 0]
    gt = z_seq[idx, 1:H+1]
    preds = []
    for t in range(H):
        z_cur = (A @ z_cur.unsqueeze(-1)).squeeze(-1) + (B @ u_seq[idx, t].unsqueeze(-1)).squeeze(-1)
        preds.append(z_cur)
    pred = torch.stack(preds, dim=1)
    return float(torch.mean((pred - gt) ** 2).item())


# =========================================================
# Plotting
# =========================================================
def plot_raw_pca(fr_s: torch.Tensor, ur_s: torch.Tensor, save_path: str):
    ur_pad = torch.cat([ur_s, torch.zeros(ur_s.shape[0], fr_s.shape[1] - ur_s.shape[1])], dim=1)
    z_all = torch.cat([fr_s, ur_pad], dim=0)
    y_all, vr = pca_2d(z_all)
    n = fr_s.shape[0]
    y_fr = y_all[:n]
    y_ur = y_all[n:]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_fr[:, 0], y_fr[:, 1], s=6, alpha=0.35, label="Franka raw")
    ax.scatter(y_ur[:, 0], y_ur[:, 1], s=6, alpha=0.35, label="UR raw")
    ax.set_title("Raw state PCA")
    ax.set_xlabel(f"PC1 ({vr[0].item()*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({vr[1].item()*100:.1f}%)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_delta_action(fr_delta: torch.Tensor, ur_delta: torch.Tensor, fr_act: torch.Tensor, ur_act: torch.Tensor, save_path: str):
    dmax = max(float(torch.quantile(fr_delta, 0.995).item()), float(torch.quantile(ur_delta, 0.995).item()))
    amax = max(float(torch.quantile(fr_act, 0.995).item()), float(torch.quantile(ur_act, 0.995).item()))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(fr_delta.numpy(), bins=cfg.hist_bins, range=(0, dmax), density=True, alpha=0.6, label="Franka")
    axes[0].hist(ur_delta.numpy(), bins=cfg.hist_bins, range=(0, dmax), density=True, alpha=0.6, label="UR")
    axes[0].set_title(r"$\|s_{t+1}-s_t\|$")
    axes[0].legend()

    axes[1].hist(fr_act.numpy(), bins=cfg.hist_bins, range=(0, amax), density=True, alpha=0.6, label="Franka")
    axes[1].hist(ur_act.numpy(), bins=cfg.hist_bins, range=(0, amax), density=True, alpha=0.6, label="UR")
    axes[1].set_title(r"$\|a_t\|$")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_latent_pca(fr_z: torch.Tensor, ur_before: torch.Tensor, ur_after: torch.Tensor, save_path: str):
    z_all = torch.cat([fr_z, ur_before, ur_after], dim=0)
    y, vr = pca_2d(z_all)
    n1 = fr_z.shape[0]
    n2 = ur_before.shape[0]
    y1 = y[:n1]
    y2 = y[n1:n1+n2]
    y3 = y[n1+n2:]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(y1[:, 0], y1[:, 1], s=6, alpha=0.35, label="Franka source")
    axes[0].scatter(y2[:, 0], y2[:, 1], s=6, alpha=0.35, label="UR before")
    axes[0].set_title("Before alignment")
    axes[0].set_xlabel(f"PC1 ({vr[0].item()*100:.1f}%)")
    axes[0].set_ylabel(f"PC2 ({vr[1].item()*100:.1f}%)")
    axes[0].legend()

    axes[1].scatter(y1[:, 0], y1[:, 1], s=6, alpha=0.35, label="Franka source")
    axes[1].scatter(y3[:, 0], y3[:, 1], s=6, alpha=0.35, label="UR after")
    axes[1].set_title("After alignment")
    axes[1].set_xlabel(f"PC1 ({vr[0].item()*100:.1f}%)")
    axes[1].set_ylabel(f"PC2 ({vr[1].item()*100:.1f}%)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_latent_hist(fr_z: torch.Tensor, ur_before: torch.Tensor, ur_after: torch.Tensor, save_path: str):
    dz = fr_z.shape[1]
    fig, axes = plt.subplots(dz, 2, figsize=(11, 2.5 * dz), squeeze=False)
    for d in range(dz):
        xmin = min(float(fr_z[:, d].min()), float(ur_before[:, d].min()), float(ur_after[:, d].min()))
        xmax = max(float(fr_z[:, d].max()), float(ur_before[:, d].max()), float(ur_after[:, d].max()))
        axes[d, 0].hist(fr_z[:, d].numpy(), bins=cfg.hist_bins, range=(xmin, xmax), density=True, alpha=0.6, label="Franka")
        axes[d, 0].hist(ur_before[:, d].numpy(), bins=cfg.hist_bins, range=(xmin, xmax), density=True, alpha=0.6, label="UR before")
        axes[d, 0].set_title(f"latent dim {d} (before)")
        axes[d, 0].legend()

        axes[d, 1].hist(fr_z[:, d].numpy(), bins=cfg.hist_bins, range=(xmin, xmax), density=True, alpha=0.6, label="Franka")
        axes[d, 1].hist(ur_after[:, d].numpy(), bins=cfg.hist_bins, range=(xmin, xmax), density=True, alpha=0.6, label="UR after")
        axes[d, 1].set_title(f"latent dim {d} (after)")
        axes[d, 1].legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_saturation_bar(fr_q: Dict, ur_b_q: Dict, ur_a_q: Dict, save_path: str):
    dz = len(fr_q["saturation_ratio"])
    x = np.arange(dz)
    w = 0.25
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - w, fr_q["saturation_ratio"], width=w, label="Franka source")
    ax.bar(x, ur_b_q["saturation_ratio"], width=w, label="UR before")
    ax.bar(x + w, ur_a_q["saturation_ratio"], width=w, label="UR after")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"dim {i}" for i in range(dz)])
    ax.set_ylabel(f"ratio(|z| > {cfg.sat_threshold})")
    ax.set_title("Latent saturation ratio")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_domain_distance(dist_before: Dict, dist_after: Dict, save_path: str):
    labels = ["mean_l2", "cov_fro"]
    before = [dist_before["mean_l2"], dist_before["cov_fro"]]
    after = [dist_after["mean_l2"], dist_after["cov_fro"]]

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w/2, before, width=w, label="before")
    ax.bar(x + w/2, after, width=w, label="after")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Source vs UR latent distance")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close(fig)


# =========================================================
# Main
# =========================================================
def main():
    set_seed(cfg.seed)
    ensure_dir(cfg.out_dir)
    device = torch.device(cfg.device)
    print(f"[INFO] device={device}")

    # ---------- load datasets ----------
    fr = load_dataset(cfg.franka_dataset)
    ur = load_dataset(cfg.ur_dataset)

    fr_s, fr_a, fr_sn = flatten_seq(fr["s"].float(), fr["a"].float(), fr["s_next"].float())
    ur_s, ur_a, ur_sn = flatten_seq(ur["s"].float(), ur["a"].float(), ur["s_next"].float())

    print(f"[INFO] Franka flat: s={tuple(fr_s.shape)} a={tuple(fr_a.shape)} sn={tuple(fr_sn.shape)}")
    print(f"[INFO] UR flat:     s={tuple(ur_s.shape)} a={tuple(ur_a.shape)} sn={tuple(ur_sn.shape)}")

    fr_delta = torch.norm(fr_sn - fr_s, dim=1)
    ur_delta = torch.norm(ur_sn - ur_s, dim=1)
    fr_act_n = torch.norm(fr_a, dim=1)
    ur_act_n = torch.norm(ur_a, dim=1)

    # ---------- load models ----------
    src = Agent(cfg.franka_obs_dim, cfg.franka_act_dim, cfg.lat_obs_dim, cfg.lat_act_dim, cfg.n_layers, cfg.hidden_dim).to(device)
    tgt = Agent(cfg.ur_obs_dim, cfg.ur_act_dim, cfg.lat_obs_dim, cfg.lat_act_dim, cfg.n_layers, cfg.hidden_dim).to(device)

    src.load_state_dict(torch.load(cfg.src_ckpt, map_location=device))
    tgt.load_state_dict(torch.load(cfg.tgt_ckpt, map_location=device))

    bundle = torch.load(cfg.stage2_bundle, map_location=device)

    z_t2s = MapNet(cfg.lat_obs_dim, cfg.n_layers, cfg.hidden_dim, out_act="tanh").to(device)
    u_t2s = MapNet(cfg.lat_act_dim, cfg.n_layers, cfg.hidden_dim, out_act="tanh").to(device)

    # 训练时保存的是裸 Sequential 的 state_dict，所以这里加载到 .net
    z_t2s.net.load_state_dict(bundle["map_z_t2s"])
    u_t2s.net.load_state_dict(bundle["map_u_t2s"])

    z_mean = bundle["z_mean"]
    z_std = bundle["z_std"]

    koop = torch.load(cfg.koopman_ckpt, map_location="cpu")
    A = koop["data"]["A"]
    B = koop["data"]["B"]

    # ---------- encode ----------
    fr_z, fr_u, fr_zn = encode_src(src, fr_s, fr_a, fr_sn, cfg.encode_bs, device)
    ur_z_before, ur_u_before, ur_zn_before = encode_tgt_before(tgt, ur_s, ur_a, ur_sn, cfg.encode_bs, device)
    ur_z_after, ur_u_after, ur_zn_after = encode_tgt_after(tgt, z_t2s, u_t2s, z_mean, z_std, ur_s, ur_a, ur_sn, cfg.encode_bs, device)

    # small subsets for plots
    fr_s_sub = sample_rows(fr_s, cfg.pca_points)
    ur_s_sub = sample_rows(ur_s, cfg.pca_points)

    fr_z_sub = sample_rows(fr_z, cfg.latent_points)
    ur_z_before_sub = sample_rows(ur_z_before, cfg.latent_points)
    ur_z_after_sub = sample_rows(ur_z_after, cfg.latent_points)

    # ---------- metrics ----------
    summary = {
        "config": asdict(cfg),
        "raw": {
            "franka": {
                "num_transitions": int(fr_s.shape[0]),
                "state_stats": tensor_stats(fr_s),
                "delta_norm": scalar_range_stats(fr_delta),
                "action_norm": scalar_range_stats(fr_act_n),
                "state_cov": compute_cov_rank(sample_rows(fr_s, min(50000, fr_s.shape[0]))),
            },
            "ur": {
                "num_transitions": int(ur_s.shape[0]),
                "state_stats": tensor_stats(ur_s),
                "delta_norm": scalar_range_stats(ur_delta),
                "action_norm": scalar_range_stats(ur_act_n),
                "state_cov": compute_cov_rank(sample_rows(ur_s, min(50000, ur_s.shape[0]))),
            },
        },
        "latent": {
            "franka_source": latent_quality(fr_z_sub, cfg.sat_threshold),
            "ur_before": latent_quality(ur_z_before_sub, cfg.sat_threshold),
            "ur_after": latent_quality(ur_z_after_sub, cfg.sat_threshold),
            "domain_distance_before": mean_cov_distance(fr_z_sub, ur_z_before_sub),
            "domain_distance_after": mean_cov_distance(fr_z_sub, ur_z_after_sub),
        },
        "koopman": {
            "train_1step_mse": float(koop["meta"]["train_1step_mse"]) if "meta" in koop and "train_1step_mse" in koop["meta"] else None,
            "franka_rollout_mse": rollout_mse(fr_z, fr_u, A, B, cfg.episode_len, cfg.rollout_horizon, device),
            "ur_mapped_rollout_mse": rollout_mse(ur_z_after, ur_u_after, A, B, cfg.episode_len, cfg.rollout_horizon, device),
        }
    }

    # ---------- diagnosis ----------
    diagnosis = []

    db = summary["latent"]["domain_distance_before"]
    da = summary["latent"]["domain_distance_after"]
    if da["mean_l2"] < db["mean_l2"]:
        diagnosis.append("UR 对齐后与 Franka 的 latent 均值距离下降。")
    else:
        diagnosis.append("UR 对齐后与 Franka 的 latent 均值距离没有下降。")

    if da["cov_fro"] < db["cov_fro"]:
        diagnosis.append("UR 对齐后与 Franka 的 latent 协方差距离下降，说明分布形状更接近。")
    else:
        diagnosis.append("UR 对齐后与 Franka 的 latent 协方差距离没有下降。")

    if summary["latent"]["ur_after"]["avg_saturation"] <= summary["latent"]["ur_before"]["avg_saturation"]:
        diagnosis.append("UR 对齐后 latent 饱和没有变差。")

    if summary["koopman"]["ur_mapped_rollout_mse"] < summary["koopman"]["franka_rollout_mse"]:
        diagnosis.append("UR(mapped) 的多步 rollout MSE 小于 Franka，说明映射后的 UR latent 较平滑、较线性。")
    else:
        diagnosis.append("UR(mapped) 的多步 rollout MSE 不低于 Franka。")

    summary["diagnosis"] = diagnosis

    # ---------- plots ----------
    plot_raw_pca(fr_s_sub, ur_s_sub, os.path.join(cfg.out_dir, "raw_state_pca.png"))
    plot_delta_action(fr_delta, ur_delta, fr_act_n, ur_act_n, os.path.join(cfg.out_dir, "delta_action_density.png"))
    plot_latent_pca(fr_z_sub, ur_z_before_sub, ur_z_after_sub, os.path.join(cfg.out_dir, "latent_pca_before_after.png"))
    plot_latent_hist(fr_z_sub, ur_z_before_sub, ur_z_after_sub, os.path.join(cfg.out_dir, "latent_hist_before_after.png"))
    plot_saturation_bar(
        summary["latent"]["franka_source"],
        summary["latent"]["ur_before"],
        summary["latent"]["ur_after"],
        os.path.join(cfg.out_dir, "latent_saturation_bar.png"),
    )
    plot_domain_distance(
        summary["latent"]["domain_distance_before"],
        summary["latent"]["domain_distance_after"],
        os.path.join(cfg.out_dir, "domain_distance_before_after.png"),
    )

    # ---------- save ----------
    with open(os.path.join(cfg.out_dir, "compare_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    lines = []
    lines.append("=== Franka vs UR Alignment Report ===")
    lines.append("")
    lines.append("[Raw data]")
    lines.append(f"Franka transitions: {summary['raw']['franka']['num_transitions']}")
    lines.append(f"UR transitions: {summary['raw']['ur']['num_transitions']}")
    lines.append(f"Franka delta mean: {summary['raw']['franka']['delta_norm']['mean']:.6f}")
    lines.append(f"UR delta mean: {summary['raw']['ur']['delta_norm']['mean']:.6f}")
    lines.append(f"Franka action norm mean: {summary['raw']['franka']['action_norm']['mean']:.6f}")
    lines.append(f"UR action norm mean: {summary['raw']['ur']['action_norm']['mean']:.6f}")
    lines.append(f"Franka effective rank: {summary['raw']['franka']['state_cov']['effective_rank_eps_ratio_1e3']}")
    lines.append(f"UR effective rank: {summary['raw']['ur']['state_cov']['effective_rank_eps_ratio_1e3']}")
    lines.append("")
    lines.append("[Latent quality]")
    lines.append(f"Franka source avg std: {summary['latent']['franka_source']['avg_std']:.6f}")
    lines.append(f"UR before avg std: {summary['latent']['ur_before']['avg_std']:.6f}")
    lines.append(f"UR after avg std: {summary['latent']['ur_after']['avg_std']:.6f}")
    lines.append(f"Franka source avg saturation: {summary['latent']['franka_source']['avg_saturation']:.6f}")
    lines.append(f"UR before avg saturation: {summary['latent']['ur_before']['avg_saturation']:.6f}")
    lines.append(f"UR after avg saturation: {summary['latent']['ur_after']['avg_saturation']:.6f}")
    lines.append("")
    lines.append("[Distribution distance]")
    lines.append(f"Before mean_l2: {summary['latent']['domain_distance_before']['mean_l2']:.6f}")
    lines.append(f"After  mean_l2: {summary['latent']['domain_distance_after']['mean_l2']:.6f}")
    lines.append(f"Before cov_fro: {summary['latent']['domain_distance_before']['cov_fro']:.6f}")
    lines.append(f"After  cov_fro: {summary['latent']['domain_distance_after']['cov_fro']:.6f}")
    lines.append("")
    lines.append("[Koopman]")
    lines.append(f"Train 1-step MSE: {summary['koopman']['train_1step_mse']}")
    lines.append(f"Franka {cfg.rollout_horizon}-step rollout MSE: {summary['koopman']['franka_rollout_mse']:.6f}")
    lines.append(f"UR(mapped) {cfg.rollout_horizon}-step rollout MSE: {summary['koopman']['ur_mapped_rollout_mse']:.6f}")
    lines.append("")
    lines.append("[Diagnosis]")
    for s in diagnosis:
        lines.append(f"- {s}")

    report = "\n".join(lines)
    with open(os.path.join(cfg.out_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    print("\n" + report)
    print(f"\n[DONE] saved to: {cfg.out_dir}")


if __name__ == "__main__":
    main()
