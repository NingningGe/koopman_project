#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diagnose_franka_dataset_and_latent.py

用途：
1) 检查 Franka 原始数据是否覆盖不足
2) 检查 Stage1 学到的 latent 是否塌缩 / 饱和
3) 输出若干统计量与图，帮助判断：
   - 是数据采集问题
   - 还是 encoder / latent 学坏了
   - 或者两者都有

输出内容：
- raw_state_hist.png
- raw_state_pca.png
- delta_and_action_norm_hist.png
- latent_hist.png
- latent_pca.png
- saturation_bar.png
- diagnostics_summary.json

运行：
/home/nng/miniconda3/envs/koopman/bin/python /mnt/data/diagnose_franka_dataset_and_latent.py
"""

import os
import json
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


@dataclass
class Cfg:
    franka_dataset: str = "/home/nng/koopman_project/data_wang_franka_reach_ablation/dataset_wang_franka_reach_ablation_v2.pt"
    src_ckpt: str = "/home/nng/koopman_project/out_align_koopman/src_franka/src_agent.pt"
    out_dir: str = "/home/nng/koopman_project/out_align_koopman/diagnose_franka"

    # 必须与 Stage1 训练一致
    obs_dim: int = 14
    act_dim: int = 7
    lat_obs_dim: int = 4
    lat_act_dim: int = 4
    hidden_dim: int = 256
    n_layers: int = 3

    raw_pca_points: int = 10000
    latent_points: int = 10000

    raw_hist_dims: int = 14
    latent_hist_bins: int = 60
    raw_hist_bins: int = 60

    sat_threshold: float = 0.95

    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    seed: int = 0


cfg = Cfg()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


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


def load_dataset(path: str) -> Dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu")
    data = obj["data"] if isinstance(obj, dict) and "data" in obj else obj
    return data


@torch.no_grad()
def pca_2d(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    x = x - x.mean(dim=0, keepdim=True)
    U, S, Vh = torch.linalg.svd(x, full_matrices=False)
    comps = Vh[:2].T
    return x @ comps


class SrcAgent(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, lat_obs_dim: int, lat_act_dim: int,
                 n_layers: int, hidden_dim: int):
        super().__init__()
        self.obs_enc = build_mlp(obs_dim, lat_obs_dim, n_layers, hidden_dim, activation="relu", out_act="tanh")
        self.obs_dec = build_mlp(lat_obs_dim, obs_dim, n_layers, hidden_dim, activation="relu", out_act="identity")
        self.act_enc = build_mlp(obs_dim + act_dim, lat_act_dim, n_layers, hidden_dim, activation="relu", out_act="tanh")
        self.act_dec = build_mlp(obs_dim + lat_act_dim, act_dim, n_layers, hidden_dim, activation="relu", out_act="tanh")
        self.inv_dyn = build_mlp(lat_obs_dim * 2, lat_act_dim, n_layers, hidden_dim, activation="relu", out_act="tanh")
        self.fwd_dyn = build_mlp(lat_obs_dim + lat_act_dim, lat_obs_dim, n_layers, hidden_dim, activation="relu", out_act="tanh")
        self.actor = build_mlp(lat_obs_dim, lat_act_dim, n_layers, hidden_dim, activation="relu", out_act="tanh")


def main():
    set_seed(cfg.seed)
    ensure_dir(cfg.out_dir)
    device = torch.device(cfg.device)
    print(f"[INFO] device={device}")

    data = load_dataset(cfg.franka_dataset)
    s = data["s"].float()
    a = data["a"].float()
    s_next = data["s_next"].float()

    N_eps, T, obs_dim = s.shape
    act_dim = a.shape[-1]
    print(f"[INFO] dataset: s={tuple(s.shape)} a={tuple(a.shape)} s_next={tuple(s_next.shape)}")

    s_flat = s.reshape(-1, obs_dim)
    a_flat = a.reshape(-1, act_dim)
    sn_flat = s_next.reshape(-1, obs_dim)

    delta = torch.norm(sn_flat - s_flat, dim=1)
    action_norm = torch.norm(a_flat, dim=1)

    raw_mean = s_flat.mean(dim=0)
    raw_std = s_flat.std(dim=0)

    src = SrcAgent(
        obs_dim=cfg.obs_dim,
        act_dim=cfg.act_dim,
        lat_obs_dim=cfg.lat_obs_dim,
        lat_act_dim=cfg.lat_act_dim,
        n_layers=cfg.n_layers,
        hidden_dim=cfg.hidden_dim,
    ).to(device)
    src.load_state_dict(torch.load(cfg.src_ckpt, map_location=device))
    src.eval()
    print(f"[INFO] loaded src ckpt: {cfg.src_ckpt}")

    with torch.no_grad():
        z = src.obs_enc(s_flat.to(device)).detach().cpu()

    z_mean = z.mean(dim=0)
    z_std = z.std(dim=0)
    z_min = z.min(dim=0).values
    z_max = z.max(dim=0).values
    sat_ratio = (torch.abs(z) > cfg.sat_threshold).float().mean(dim=0)

    # raw hist
    n_dims = min(cfg.raw_hist_dims, obs_dim)
    ncols = 2
    nrows = int(np.ceil(n_dims / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 2.8 * nrows), squeeze=False)
    for d in range(n_dims):
        r, c = divmod(d, ncols)
        axes[r, c].hist(s_flat[:, d].numpy(), bins=cfg.raw_hist_bins, alpha=0.8, density=True)
        axes[r, c].set_title(f"raw state dim {d}")
    for d in range(n_dims, nrows * ncols):
        r, c = divmod(d, ncols)
        axes[r, c].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.out_dir, "raw_state_hist.png"), dpi=180)
    plt.close(fig)

    # raw pca
    idx_raw = torch.randperm(s_flat.shape[0])[: min(cfg.raw_pca_points, s_flat.shape[0])]
    raw_sub = s_flat[idx_raw]
    raw_pca = pca_2d(raw_sub)
    fig = plt.figure(figsize=(6, 5))
    plt.scatter(raw_pca[:, 0], raw_pca[:, 1], s=6, alpha=0.4)
    plt.title("Franka raw state PCA")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.out_dir, "raw_state_pca.png"), dpi=180)
    plt.close(fig)

    # delta/action hist
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(delta.numpy(), bins=80, alpha=0.8, density=True)
    axes[0].set_title(r"$\|s_{t+1} - s_t\|$")
    axes[1].hist(action_norm.numpy(), bins=80, alpha=0.8, density=True)
    axes[1].set_title(r"$\|a_t\|$")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.out_dir, "delta_and_action_norm_hist.png"), dpi=180)
    plt.close(fig)

    # latent hist
    dz = z.shape[1]
    fig, axes = plt.subplots(dz, 1, figsize=(8, 2.5 * dz), squeeze=False)
    for d in range(dz):
        axes[d, 0].hist(z[:, d].numpy(), bins=cfg.latent_hist_bins, alpha=0.8, density=True)
        axes[d, 0].axvline(cfg.sat_threshold, color="r", linestyle="--", alpha=0.7)
        axes[d, 0].axvline(-cfg.sat_threshold, color="r", linestyle="--", alpha=0.7)
        axes[d, 0].set_title(f"latent dim {d}")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.out_dir, "latent_hist.png"), dpi=180)
    plt.close(fig)

    # latent pca
    idx_lat = torch.randperm(z.shape[0])[: min(cfg.latent_points, z.shape[0])]
    z_sub = z[idx_lat]
    z_pca = pca_2d(z_sub)
    fig = plt.figure(figsize=(6, 5))
    plt.scatter(z_pca[:, 0], z_pca[:, 1], s=6, alpha=0.4)
    plt.title("Franka latent PCA")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.out_dir, "latent_pca.png"), dpi=180)
    plt.close(fig)

    # saturation bar
    fig = plt.figure(figsize=(6, 4))
    x = np.arange(dz)
    plt.bar(x, sat_ratio.numpy())
    plt.ylim(0.0, 1.0)
    plt.xticks(x, [f"dim {i}" for i in range(dz)])
    plt.ylabel(f"ratio(|z| > {cfg.sat_threshold})")
    plt.title("Latent saturation ratio")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.out_dir, "saturation_bar.png"), dpi=180)
    plt.close(fig)

    delta_mean = delta.mean().item()
    delta_std = delta.std().item()
    action_mean = action_norm.mean().item()
    action_std = action_norm.std().item()
    avg_lat_std = z_std.mean().item()
    avg_sat = sat_ratio.mean().item()

    diagnosis = []
    if delta_mean < 1e-3:
        diagnosis.append("原始状态变化量非常小：数据可能运动幅度不足。")
    elif delta_mean < 1e-2:
        diagnosis.append("原始状态变化量偏小：Franka 可能主要在较小局部范围内运动。")
    else:
        diagnosis.append("原始状态变化量不算太小：数据本身未明显静止。")

    if avg_lat_std < 0.05:
        diagnosis.append("latent 方差很小：encoder 可能发生明显塌缩。")
    elif avg_lat_std < 0.15:
        diagnosis.append("latent 方差偏小：存在一定塌缩风险。")
    else:
        diagnosis.append("latent 方差尚可：没有明显整体塌缩。")

    if avg_sat > 0.5:
        diagnosis.append("超过半数 latent 样本接近 ±1：存在严重 tanh 饱和。")
    elif avg_sat > 0.2:
        diagnosis.append("latent 有一定比例接近 ±1：存在中等饱和现象。")
    else:
        diagnosis.append("latent 饱和比例不高。")

    collapsed_dims = [i for i, v in enumerate(z_std.tolist()) if v < 0.03]
    saturated_dims = [i for i, v in enumerate(sat_ratio.tolist()) if v > 0.5]

    summary = {
        "raw": {
            "num_episodes": int(N_eps),
            "episode_len": int(T),
            "obs_dim": int(obs_dim),
            "act_dim": int(act_dim),
            "raw_mean": raw_mean.tolist(),
            "raw_std": raw_std.tolist(),
            "delta_mean": delta_mean,
            "delta_std": delta_std,
            "action_norm_mean": action_mean,
            "action_norm_std": action_std,
        },
        "latent": {
            "lat_obs_dim": int(dz),
            "z_mean": z_mean.tolist(),
            "z_std": z_std.tolist(),
            "z_min": z_min.tolist(),
            "z_max": z_max.tolist(),
            "saturation_threshold": float(cfg.sat_threshold),
            "saturation_ratio": sat_ratio.tolist(),
            "avg_latent_std": avg_lat_std,
            "avg_saturation_ratio": avg_sat,
            "collapsed_dims_std_lt_0p03": collapsed_dims,
            "saturated_dims_ratio_gt_0p5": saturated_dims,
        },
        "diagnosis": diagnosis,
    }

    with open(os.path.join(cfg.out_dir, "diagnostics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n========== Diagnosis ==========")
    for x in diagnosis:
        print("-", x)
    print("collapsed_dims:", collapsed_dims)
    print("saturated_dims:", saturated_dims)
    print(f"[DONE] outputs saved to: {cfg.out_dir}")


if __name__ == "__main__":
    main()
