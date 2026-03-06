#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diagnose_UR_dataset_and_latent.py

用途：
1) 检查 UR 原始数据是否覆盖不足
2) 检查 Stage2 的 target latent（对齐前）是否塌缩 / 饱和
3) 检查 UR -> source latent 映射后（对齐后）是否塌缩 / 饱和
4) 方便与 Franka 的 diagnose 结果做对比

输出内容：
- raw_state_hist.png
- raw_state_pca.png
- delta_and_action_norm_hist.png
- latent_before_hist.png
- latent_before_pca.png
- latent_after_hist.png
- latent_after_pca.png
- saturation_before_bar.png
- saturation_after_bar.png
- diagnostics_summary.json

运行：
/home/nng/miniconda3/envs/koopman/bin/python /home/nng/koopman_project/evaluate/diagnose_UR_dataset_and_latent.py
"""

import os
import json
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# =========================
# Config
# =========================
@dataclass
class Cfg:
    ur_dataset: str = "/home/nng/koopman_project/data_wang_ur_reach_ablation/dataset_wang_ur_reach_ablation.pt"
    tgt_ckpt: str = "/home/nng/koopman_project/out_align_koopman/tgt_ur/tgt_agent.pt"
    stage2_bundle: str = "/home/nng/koopman_project/out_align_koopman/tgt_ur/stage2_bundle.pt"
    out_dir: str = "/home/nng/koopman_project/out_align_koopman/diagnose_ur"

    # 必须与训练时一致
    obs_dim: int = 12
    act_dim: int = 6
    lat_obs_dim: int = 4
    lat_act_dim: int = 4
    hidden_dim: int = 256
    n_layers: int = 3

    raw_pca_points: int = 10000
    latent_points: int = 10000

    raw_hist_dims: int = 12
    latent_hist_bins: int = 60
    raw_hist_bins: int = 60

    sat_threshold: float = 0.95

    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    seed: int = 0


cfg = Cfg()


# =========================
# Utils
# =========================
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


# =========================
# Models
# =========================
class SrcLikeAgent(nn.Module):
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


class DummyAligner(nn.Module):
    def __init__(self, lat_obs_dim: int, lat_act_dim: int, n_layers: int, hidden_dim: int):
        super().__init__()
        self.z_t2s = build_mlp(lat_obs_dim, lat_obs_dim, n_layers, hidden_dim, activation="relu", out_act="identity")
        self.z_s2t = build_mlp(lat_obs_dim, lat_obs_dim, n_layers, hidden_dim, activation="relu", out_act="identity")
        self.u_t2s = build_mlp(lat_act_dim, lat_act_dim, n_layers, hidden_dim, activation="relu", out_act="identity")
        self.u_s2t = build_mlp(lat_act_dim, lat_act_dim, n_layers, hidden_dim, activation="relu", out_act="identity")


# =========================
# Plot helpers
# =========================
def plot_hist_grid(x: torch.Tensor, bins: int, title_prefix: str, save_path: str):
    D = x.shape[1]
    ncols = 2
    nrows = int(np.ceil(D / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 2.8 * nrows), squeeze=False)
    for d in range(D):
        r, c = divmod(d, ncols)
        axes[r, c].hist(x[:, d].numpy(), bins=bins, alpha=0.8, density=True)
        axes[r, c].set_title(f"{title_prefix} dim {d}")
    for d in range(D, nrows * ncols):
        r, c = divmod(d, ncols)
        axes[r, c].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_latent_hist(x: torch.Tensor, bins: int, sat_threshold: float, title_prefix: str, save_path: str):
    D = x.shape[1]
    fig, axes = plt.subplots(D, 1, figsize=(8, 2.5 * D), squeeze=False)
    for d in range(D):
        axes[d, 0].hist(x[:, d].numpy(), bins=bins, alpha=0.8, density=True)
        axes[d, 0].axvline(sat_threshold, color="r", linestyle="--", alpha=0.7)
        axes[d, 0].axvline(-sat_threshold, color="r", linestyle="--", alpha=0.7)
        axes[d, 0].set_title(f"{title_prefix} dim {d}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_pca(x: torch.Tensor, title: str, save_path: str):
    y = pca_2d(x)
    fig = plt.figure(figsize=(6, 5))
    plt.scatter(y[:, 0], y[:, 1], s=6, alpha=0.4)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_saturation_bar(ratios: torch.Tensor, title: str, save_path: str, sat_threshold: float):
    fig = plt.figure(figsize=(6, 4))
    x = np.arange(len(ratios))
    plt.bar(x, ratios.numpy())
    plt.ylim(0.0, 1.0)
    plt.xticks(x, [f"dim {i}" for i in range(len(ratios))])
    plt.ylabel(f"ratio(|z| > {sat_threshold})")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close(fig)


# =========================
# Main
# =========================
def main():
    set_seed(cfg.seed)
    ensure_dir(cfg.out_dir)
    device = torch.device(cfg.device)
    print(f"[INFO] device={device}")

    # ---------- load UR dataset ----------
    data = load_dataset(cfg.ur_dataset)
    s = data["s"].float()          # (N_eps,T,12)
    a = data["a"].float()          # (N_eps,T,6)
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

    # ---------- load target encoder + mapping ----------
    tgt = SrcLikeAgent(
        obs_dim=cfg.obs_dim,
        act_dim=cfg.act_dim,
        lat_obs_dim=cfg.lat_obs_dim,
        lat_act_dim=cfg.lat_act_dim,
        n_layers=cfg.n_layers,
        hidden_dim=cfg.hidden_dim,
    ).to(device)
    tgt.load_state_dict(torch.load(cfg.tgt_ckpt, map_location=device))
    tgt.eval()
    print(f"[INFO] loaded tgt ckpt: {cfg.tgt_ckpt}")

    aligner = DummyAligner(
        lat_obs_dim=cfg.lat_obs_dim,
        lat_act_dim=cfg.lat_act_dim,
        n_layers=cfg.n_layers,
        hidden_dim=cfg.hidden_dim,
    ).to(device)

    bundle = torch.load(cfg.stage2_bundle, map_location=device)
    if "map_z_t2s" in bundle:
        aligner.z_t2s.load_state_dict(bundle["map_z_t2s"])
    elif "z_t2s" in bundle:
        aligner.z_t2s.load_state_dict(bundle["z_t2s"])
    else:
        raise RuntimeError("stage2_bundle 里找不到 z_t2s / map_z_t2s")

    aligner.eval()
    print(f"[INFO] loaded stage2 bundle: {cfg.stage2_bundle}")

    # ---------- latent before / after ----------
    with torch.no_grad():
        z_before = tgt.obs_enc(s_flat.to(device)).detach().cpu()
        z_after = aligner.z_t2s(z_before.to(device)).detach().cpu()

    # ---------- stats ----------
    def latent_stats(z: torch.Tensor):
        z_mean = z.mean(dim=0)
        z_std = z.std(dim=0)
        z_min = z.min(dim=0).values
        z_max = z.max(dim=0).values
        sat_ratio = (torch.abs(z) > cfg.sat_threshold).float().mean(dim=0)
        return z_mean, z_std, z_min, z_max, sat_ratio

    zb_mean, zb_std, zb_min, zb_max, zb_sat = latent_stats(z_before)
    za_mean, za_std, za_min, za_max, za_sat = latent_stats(z_after)

    print("[LATENT BEFORE] std:", zb_std.tolist())
    print("[LATENT BEFORE] sat:", zb_sat.tolist())
    print("[LATENT AFTER ] std:", za_std.tolist())
    print("[LATENT AFTER ] sat:", za_sat.tolist())

    # ---------- plots ----------
    # raw
    plot_hist_grid(s_flat, cfg.raw_hist_bins, "raw state", os.path.join(cfg.out_dir, "raw_state_hist.png"))

    idx_raw = torch.randperm(s_flat.shape[0])[: min(cfg.raw_pca_points, s_flat.shape[0])]
    plot_pca(s_flat[idx_raw], "UR raw state PCA", os.path.join(cfg.out_dir, "raw_state_pca.png"))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(delta.numpy(), bins=80, alpha=0.8, density=True)
    axes[0].set_title(r"$\|s_{t+1} - s_t\|$")
    axes[1].hist(action_norm.numpy(), bins=80, alpha=0.8, density=True)
    axes[1].set_title(r"$\|a_t\|$")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.out_dir, "delta_and_action_norm_hist.png"), dpi=180)
    plt.close(fig)

    # latent before
    plot_latent_hist(z_before, cfg.latent_hist_bins, cfg.sat_threshold, "latent_before", os.path.join(cfg.out_dir, "latent_before_hist.png"))
    idx_before = torch.randperm(z_before.shape[0])[: min(cfg.latent_points, z_before.shape[0])]
    plot_pca(z_before[idx_before], "UR latent before alignment PCA", os.path.join(cfg.out_dir, "latent_before_pca.png"))
    plot_saturation_bar(zb_sat, "Latent saturation ratio (before alignment)", os.path.join(cfg.out_dir, "saturation_before_bar.png"), cfg.sat_threshold)

    # latent after
    plot_latent_hist(z_after, cfg.latent_hist_bins, cfg.sat_threshold, "latent_after", os.path.join(cfg.out_dir, "latent_after_hist.png"))
    idx_after = torch.randperm(z_after.shape[0])[: min(cfg.latent_points, z_after.shape[0])]
    plot_pca(z_after[idx_after], "UR latent after alignment PCA", os.path.join(cfg.out_dir, "latent_after_pca.png"))
    plot_saturation_bar(za_sat, "Latent saturation ratio (after alignment)", os.path.join(cfg.out_dir, "saturation_after_bar.png"), cfg.sat_threshold)

    # ---------- diagnosis ----------
    delta_mean = delta.mean().item()
    delta_std = delta.std().item()
    action_mean = action_norm.mean().item()
    action_std = action_norm.std().item()

    diagnosis = []

    # raw coverage
    if delta_mean < 1e-3:
        diagnosis.append("UR 原始状态变化量非常小：数据可能运动幅度不足。")
    elif delta_mean < 1e-2:
        diagnosis.append("UR 原始状态变化量偏小：可能主要在较小局部范围内运动。")
    else:
        diagnosis.append("UR 原始状态变化量不算太小：数据本身未明显静止。")

    # before alignment
    avg_zb_std = zb_std.mean().item()
    avg_zb_sat = zb_sat.mean().item()
    if avg_zb_std < 0.05:
        diagnosis.append("UR 对齐前 latent 方差很小：target encoder 可能发生明显塌缩。")
    elif avg_zb_std < 0.15:
        diagnosis.append("UR 对齐前 latent 方差偏小：存在一定塌缩风险。")
    else:
        diagnosis.append("UR 对齐前 latent 方差尚可：没有明显整体塌缩。")

    if avg_zb_sat > 0.5:
        diagnosis.append("UR 对齐前 latent 超过半数样本接近 ±1：存在严重 tanh 饱和。")
    elif avg_zb_sat > 0.2:
        diagnosis.append("UR 对齐前 latent 有一定比例接近 ±1：存在中等饱和现象。")
    else:
        diagnosis.append("UR 对齐前 latent 饱和比例不高。")

    # after alignment
    avg_za_std = za_std.mean().item()
    avg_za_sat = za_sat.mean().item()
    if avg_za_std < 0.05:
        diagnosis.append("UR 对齐后 latent 方差很小：映射后可能塌缩到 source latent 的狭小区域。")
    elif avg_za_std < 0.15:
        diagnosis.append("UR 对齐后 latent 方差偏小：映射后分布较集中。")
    else:
        diagnosis.append("UR 对齐后 latent 方差尚可。")

    if avg_za_sat > 0.5:
        diagnosis.append("UR 对齐后 latent 超过半数样本接近 ±1：映射后存在严重饱和。")
    elif avg_za_sat > 0.2:
        diagnosis.append("UR 对齐后 latent 有一定比例接近 ±1：映射后存在中等饱和。")
    else:
        diagnosis.append("UR 对齐后 latent 饱和比例不高。")

    collapsed_before = [i for i, v in enumerate(zb_std.tolist()) if v < 0.03]
    saturated_before = [i for i, v in enumerate(zb_sat.tolist()) if v > 0.5]
    collapsed_after = [i for i, v in enumerate(za_std.tolist()) if v < 0.03]
    saturated_after = [i for i, v in enumerate(za_sat.tolist()) if v > 0.5]

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
        "latent_before": {
            "z_mean": zb_mean.tolist(),
            "z_std": zb_std.tolist(),
            "z_min": zb_min.tolist(),
            "z_max": zb_max.tolist(),
            "saturation_threshold": float(cfg.sat_threshold),
            "saturation_ratio": zb_sat.tolist(),
            "avg_latent_std": avg_zb_std,
            "avg_saturation_ratio": avg_zb_sat,
            "collapsed_dims_std_lt_0p03": collapsed_before,
            "saturated_dims_ratio_gt_0p5": saturated_before,
        },
        "latent_after": {
            "z_mean": za_mean.tolist(),
            "z_std": za_std.tolist(),
            "z_min": za_min.tolist(),
            "z_max": za_max.tolist(),
            "saturation_threshold": float(cfg.sat_threshold),
            "saturation_ratio": za_sat.tolist(),
            "avg_latent_std": avg_za_std,
            "avg_saturation_ratio": avg_za_sat,
            "collapsed_dims_std_lt_0p03": collapsed_after,
            "saturated_dims_ratio_gt_0p5": saturated_after,
        },
        "diagnosis": diagnosis,
    }

    with open(os.path.join(cfg.out_dir, "diagnostics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n========== Diagnosis ==========")
    for x in diagnosis:
        print("-", x)
    print("collapsed_before:", collapsed_before)
    print("saturated_before:", saturated_before)
    print("collapsed_after:", collapsed_after)
    print("saturated_after:", saturated_after)
    print(f"[DONE] outputs saved to: {cfg.out_dir}")


if __name__ == "__main__":
    main()