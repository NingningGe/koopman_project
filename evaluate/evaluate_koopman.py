#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_koopman.py

自动完成：
1) 加载 Stage3 生成的 aligned_latent_and_koopman.pt
2) 计算 one-step / multi-step rollout 指标（Franka 与 UR(mapped)）
3) 可视化若干 episode 的 latent rollout 对比图
4) 加载 Stage1/Stage2 模型与原始数据，画“对齐前 / 对齐后”的潜在分布图
   - source latent:      z_s = src.obs_enc(franka_s)
   - target before map:  z_t = tgt.obs_enc(ur_s)
   - target after map:   z_t2s = z_t2s(z_t)
5) 额外画每个 latent 维度的直方图，对比对齐效果

依赖：
- torch
- matplotlib
- numpy

不依赖 sklearn。PCA 使用 torch 实现。
"""

import os
import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# =========================
# Config（直接在这里改）
# =========================
@dataclass
class Cfg:
    # ---- datasets ----
    franka_dataset: str = "/home/nng/koopman_project/data_wang_franka_reach_ablation/dataset_wang_franka_reach_ablation.pt"
    ur_dataset: str = "/home/nng/koopman_project/data_wang_ur_reach_ablation/dataset_wang_ur_reach_ablation.pt"

    # ---- trained artifacts ----
    root_dir: str = "/home/nng/koopman_project/out_align_koopman"
    src_ckpt: str = "/home/nng/koopman_project/out_align_koopman/src_franka/src_agent.pt"
    tgt_ckpt: str = "/home/nng/koopman_project/out_align_koopman/tgt_ur/tgt_agent.pt"
    stage2_bundle: str = "/home/nng/koopman_project/out_align_koopman/tgt_ur/stage2_bundle.pt"
    koopman_pack: str = "/home/nng/koopman_project/out_align_koopman/koopman_fit/aligned_latent_and_koopman.pt"

    # ---- model dims (需与训练一致) ----
    src_obs_dim: int = 14
    src_act_dim: int = 7
    tgt_obs_dim: int = 12
    tgt_act_dim: int = 6
    lat_obs_dim: int = 4
    lat_act_dim: int = 4
    hidden_dim: int = 256
    n_layers: int = 3

    # ---- eval/plot ----
    seed: int = 0
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    output_dir: str = "/home/nng/koopman_project/out_align_koopman/eval_koopman"

    encode_batch_size: int = 8192
    rollout_horizon: int = 50
    rollout_plot_episodes: int = 4
    distribution_sample_points: int = 5000  # 从 Franka/UR 各抽多少个 state 做分布图
    latent_hist_bins: int = 50


cfg = Cfg()


# =========================
# Utils
# =========================
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def load_dataset(path: str):
    obj = torch.load(path, map_location="cpu")
    data = obj["data"] if isinstance(obj, dict) and "data" in obj else obj
    return data


def flatten_transitions(s: torch.Tensor, a: torch.Tensor, s_next: torch.Tensor):
    N, T, ds = s.shape
    da = a.shape[-1]
    return s.view(N * T, ds), a.view(N * T, da), s_next.view(N * T, ds)


@torch.no_grad()
def pca_2d(x: torch.Tensor) -> torch.Tensor:
    """
    x: (N, D), CPU tensor
    return: (N, 2)
    """
    x = x.float()
    x = x - x.mean(dim=0, keepdim=True)
    # D 很小（4），直接 SVD 非常快
    U, S, Vh = torch.linalg.svd(x, full_matrices=False)
    comps = Vh[:2].T   # (D,2)
    y = x @ comps      # (N,2)
    return y


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# =========================
# Models (必须与训练脚本结构一致)
# =========================
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


class TgtAgent(SrcAgent):
    pass


class DummyAligner(nn.Module):
    """
    只承载 mapping/disc，用于加载 stage2_bundle.pt
    """
    def __init__(self, lat_obs_dim: int, lat_act_dim: int, n_layers: int, hidden_dim: int):
        super().__init__()
        self.z_t2s = build_mlp(lat_obs_dim, lat_obs_dim, n_layers, hidden_dim, activation="relu", out_act="identity")
        self.z_s2t = build_mlp(lat_obs_dim, lat_obs_dim, n_layers, hidden_dim, activation="relu", out_act="identity")
        self.u_t2s = build_mlp(lat_act_dim, lat_act_dim, n_layers, hidden_dim, activation="relu", out_act="identity")
        self.u_s2t = build_mlp(lat_act_dim, lat_act_dim, n_layers, hidden_dim, activation="relu", out_act="identity")
        self.lat_obs_disc = build_mlp(lat_obs_dim, 1, n_layers, hidden_dim, activation="leaky_relu", out_act="identity")
        self.lat_act_disc = build_mlp(lat_act_dim, 1, n_layers, hidden_dim, activation="leaky_relu", out_act="identity")


# =========================
# Koopman eval
# =========================
@torch.no_grad()
def koopman_step(z: torch.Tensor, u: torch.Tensor, A: torch.Tensor, B: torch.Tensor):
    return z @ A.T + u @ B.T


@torch.no_grad()
def one_step_metrics(z: torch.Tensor, u: torch.Tensor, z_next: torch.Tensor, A: torch.Tensor, B: torch.Tensor):
    pred = koopman_step(z, u, A, B)
    mse = torch.mean((pred - z_next) ** 2).item()
    mae = torch.mean(torch.abs(pred - z_next)).item()
    return mse, mae


@torch.no_grad()
def rollout_metrics(z_seq: torch.Tensor, u_seq: torch.Tensor, A: torch.Tensor, B: torch.Tensor, H: int):
    E, T, dz = z_seq.shape
    H = min(H, T - 1)
    z_hat = z_seq[:, 0].clone()
    mses = []
    for t in range(H):
        z_hat = koopman_step(z_hat, u_seq[:, t], A, B)
        z_gt = z_seq[:, t + 1]
        mses.append(torch.mean((z_hat - z_gt) ** 2).item())
    return float(np.mean(mses)), mses


@torch.no_grad()
def reshape_to_episodes(z_flat: torch.Tensor, u_flat: torch.Tensor, T: int):
    N = z_flat.shape[0]
    assert N % T == 0, f"N={N} 不能被 T={T} 整除"
    E = N // T
    return z_flat.view(E, T, -1), u_flat.view(E, T, -1)


# =========================
# Latent encoding for distribution plots
# =========================
@torch.no_grad()
def encode_obs_batched(agent, s_all: torch.Tensor, batch_size: int, device: torch.device):
    agent.eval()
    outs = []
    for i in range(0, s_all.shape[0], batch_size):
        s = s_all[i:i+batch_size].to(device)
        z = agent.obs_enc(s).detach().cpu()
        outs.append(z)
    return torch.cat(outs, dim=0)


@torch.no_grad()
def map_obs_batched(aligner, z_all: torch.Tensor, batch_size: int, device: torch.device):
    aligner.eval()
    outs = []
    for i in range(0, z_all.shape[0], batch_size):
        z = z_all[i:i+batch_size].to(device)
        y = aligner.z_t2s(z).detach().cpu()
        outs.append(y)
    return torch.cat(outs, dim=0)


# =========================
# Plot helpers
# =========================
def plot_rollout_examples(z_seq: torch.Tensor, u_seq: torch.Tensor, A: torch.Tensor, B: torch.Tensor,
                          H: int, save_path: str, title_prefix: str):
    """
    画若干 episode 的 latent rollout 对比
    """
    E, T, dz = z_seq.shape
    H = min(H, T - 1)
    K = min(cfg.rollout_plot_episodes, E)

    idx = torch.randperm(E)[:K]
    t_axis = np.arange(H + 1)

    fig, axes = plt.subplots(K, dz, figsize=(4 * dz, 2.5 * K), squeeze=False)

    for row, e in enumerate(idx.tolist()):
        z0 = z_seq[e, 0].clone()
        gt = z_seq[e, :H+1].cpu().numpy()

        pred_list = [z0.cpu()]
        z = z0.unsqueeze(0)  # (1,dz)
        for t in range(H):
            u_t = u_seq[e, t].unsqueeze(0)
            z = koopman_step(z, u_t, A, B)
            pred_list.append(z.squeeze(0).cpu())
        pred = torch.stack(pred_list, dim=0).numpy()

        for d in range(dz):
            ax = axes[row, d]
            ax.plot(t_axis, gt[:, d], label="GT")
            ax.plot(t_axis, pred[:, d], linestyle="--", label="Koopman")
            if row == 0:
                ax.set_title(f"{title_prefix} latent dim {d}")
            if d == 0:
                ax.set_ylabel(f"ep {e}")
            if row == 0 and d == 0:
                ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_alignment_scatter(z_src: torch.Tensor, z_tgt_before: torch.Tensor, z_tgt_after: torch.Tensor, save_path: str):
    """
    PCA 2D 散点图：对齐前/后分布
    """
    # 统一投影到同一个 2D PCA 子空间：用三者拼接后做 PCA
    z_all = torch.cat([z_src, z_tgt_before, z_tgt_after], dim=0)
    y_all = pca_2d(z_all)

    n1 = z_src.shape[0]
    n2 = z_tgt_before.shape[0]

    y_src = y_all[:n1]
    y_tgt_before = y_all[n1:n1+n2]
    y_tgt_after = y_all[n1+n2:]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(y_src[:, 0], y_src[:, 1], s=6, alpha=0.4, label="Franka source latent")
    axes[0].scatter(y_tgt_before[:, 0], y_tgt_before[:, 1], s=6, alpha=0.4, label="UR before align")
    axes[0].set_title("Before alignment")
    axes[0].legend()

    axes[1].scatter(y_src[:, 0], y_src[:, 1], s=6, alpha=0.4, label="Franka source latent")
    axes[1].scatter(y_tgt_after[:, 0], y_tgt_after[:, 1], s=6, alpha=0.4, label="UR after align")
    axes[1].set_title("After alignment")
    axes[1].legend()

    for ax in axes:
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_alignment_hist(z_src: torch.Tensor, z_tgt_before: torch.Tensor, z_tgt_after: torch.Tensor, save_path: str):
    """
    每个 latent 维度的直方图，对比对齐前后分布
    """
    dz = z_src.shape[1]
    fig, axes = plt.subplots(dz, 2, figsize=(10, 2.6 * dz), squeeze=False)

    for d in range(dz):
        axes[d, 0].hist(z_src[:, d].numpy(), bins=cfg.latent_hist_bins, alpha=0.6, density=True, label="Franka")
        axes[d, 0].hist(z_tgt_before[:, d].numpy(), bins=cfg.latent_hist_bins, alpha=0.6, density=True, label="UR before")
        axes[d, 0].set_title(f"latent dim {d} - before")
        axes[d, 0].legend()

        axes[d, 1].hist(z_src[:, d].numpy(), bins=cfg.latent_hist_bins, alpha=0.6, density=True, label="Franka")
        axes[d, 1].hist(z_tgt_after[:, d].numpy(), bins=cfg.latent_hist_bins, alpha=0.6, density=True, label="UR after")
        axes[d, 1].set_title(f"latent dim {d} - after")
        axes[d, 1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close(fig)


# =========================
# Main
# =========================
def main():
    set_seed(cfg.seed)
    ensure_dir(cfg.output_dir)
    device = torch.device(cfg.device)
    print(f"[INFO] device={device}")

    # ---------- load koopman pack ----------
    pack = torch.load(cfg.koopman_pack, map_location="cpu")
    meta = pack["meta"]
    data = pack["data"]

    A = data["A"].float().to(device)
    B = data["B"].float().to(device)

    fr_z = data["fr_z"].float()
    fr_u = data["fr_u"].float()
    fr_zn = data["fr_zn"].float()

    ur_z = data["ur_z"].float()
    ur_u = data["ur_u"].float()
    ur_zn = data["ur_zn"].float()

    T = int(meta.get("episode_len", 200))
    H = int(meta.get("rollout_horizon", cfg.rollout_horizon))

    print(f"[INFO] loaded Koopman pack: fr_z={tuple(fr_z.shape)} ur_z={tuple(ur_z.shape)} A={tuple(A.shape)} B={tuple(B.shape)}")
    print(f"[INFO] episode_len={T} rollout_horizon={H}")

    # ---------- one-step metrics ----------
    # simple random val split on CPU -> move slice to GPU
    def sample_val(z, u, zn, n=50000):
        idx = torch.randperm(z.shape[0])[: min(n, z.shape[0])]
        return z[idx].to(device), u[idx].to(device), zn[idx].to(device)

    fr_zv, fr_uv, fr_znv = sample_val(fr_z, fr_u, fr_zn)
    ur_zv, ur_uv, ur_znv = sample_val(ur_z, ur_u, ur_zn)

    fr_mse, fr_mae = one_step_metrics(fr_zv, fr_uv, fr_znv, A, B)
    ur_mse, ur_mae = one_step_metrics(ur_zv, ur_uv, ur_znv, A, B)

    print(f"[ONE-STEP] Franka   MSE={fr_mse:.6e} MAE={fr_mae:.6e}")
    print(f"[ONE-STEP] UR(mapped) MSE={ur_mse:.6e} MAE={ur_mae:.6e}")

    # ---------- rollout metrics ----------
    fr_z_seq, fr_u_seq = reshape_to_episodes(fr_z.to(device), fr_u.to(device), T)
    ur_z_seq, ur_u_seq = reshape_to_episodes(ur_z.to(device), ur_u.to(device), T)

    fr_roll_mse, fr_roll_curve = rollout_metrics(fr_z_seq, fr_u_seq, A, B, H)
    ur_roll_mse, ur_roll_curve = rollout_metrics(ur_z_seq, ur_u_seq, A, B, H)

    print(f"[ROLLOUT] Franka   avg-{H}step MSE={fr_roll_mse:.6e} first={fr_roll_curve[0]:.6e} last={fr_roll_curve[-1]:.6e}")
    print(f"[ROLLOUT] UR(mapped) avg-{H}step MSE={ur_roll_mse:.6e} first={ur_roll_curve[0]:.6e} last={ur_roll_curve[-1]:.6e}")

    # ---------- plot rollout examples ----------
    plot_rollout_examples(fr_z_seq, fr_u_seq, A, B, H, os.path.join(cfg.output_dir, "rollout_franka.png"), "Franka")
    plot_rollout_examples(ur_z_seq, ur_u_seq, A, B, H, os.path.join(cfg.output_dir, "rollout_ur_mapped.png"), "UR(mapped)")
    print("[PLOT] saved rollout_franka.png / rollout_ur_mapped.png")

    # ---------- alignment distribution before/after ----------
    # load raw datasets
    fr_data = load_dataset(cfg.franka_dataset)
    ur_data = load_dataset(cfg.ur_dataset)

    fr_s = fr_data["s"].float().reshape(-1, cfg.src_obs_dim)
    ur_s = ur_data["s"].float().reshape(-1, cfg.tgt_obs_dim)

    # random sample
    n_fr = min(cfg.distribution_sample_points, fr_s.shape[0])
    n_ur = min(cfg.distribution_sample_points, ur_s.shape[0])

    fr_idx = torch.randperm(fr_s.shape[0])[:n_fr]
    ur_idx = torch.randperm(ur_s.shape[0])[:n_ur]

    fr_s_sub = fr_s[fr_idx]
    ur_s_sub = ur_s[ur_idx]

    # rebuild models
    src = SrcAgent(cfg.src_obs_dim, cfg.src_act_dim, cfg.lat_obs_dim, cfg.lat_act_dim, cfg.n_layers, cfg.hidden_dim).to(device)
    tgt = TgtAgent(cfg.tgt_obs_dim, cfg.tgt_act_dim, cfg.lat_obs_dim, cfg.lat_act_dim, cfg.n_layers, cfg.hidden_dim).to(device)
    aligner = DummyAligner(cfg.lat_obs_dim, cfg.lat_act_dim, cfg.n_layers, cfg.hidden_dim).to(device)

    # load checkpoints
    src.load_state_dict(torch.load(cfg.src_ckpt, map_location=device))
    tgt.load_state_dict(torch.load(cfg.tgt_ckpt, map_location=device))
    bundle = torch.load(cfg.stage2_bundle, map_location=device)

    # 兼容 stage2_bundle.pt 键名
    if "tgt_agent" in bundle:
        tgt.load_state_dict(bundle["tgt_agent"], strict=False)
    if "map_z_t2s" in bundle:
        aligner.z_t2s.load_state_dict(bundle["map_z_t2s"])
    elif "z_t2s" in bundle:
        aligner.z_t2s.load_state_dict(bundle["z_t2s"])
    else:
        raise RuntimeError("stage2_bundle 里找不到 z_t2s")
    if "map_z_s2t" in bundle:
        aligner.z_s2t.load_state_dict(bundle["map_z_s2t"])
    elif "z_s2t" in bundle:
        aligner.z_s2t.load_state_dict(bundle["z_s2t"])
    if "map_u_t2s" in bundle:
        aligner.u_t2s.load_state_dict(bundle["map_u_t2s"])
    elif "u_t2s" in bundle:
        aligner.u_t2s.load_state_dict(bundle["u_t2s"])
    if "map_u_s2t" in bundle:
        aligner.u_s2t.load_state_dict(bundle["map_u_s2t"])
    elif "u_s2t" in bundle:
        aligner.u_s2t.load_state_dict(bundle["u_s2t"])

    src.eval()
    tgt.eval()
    aligner.eval()

    # source latent / target before / target after
    z_src = encode_obs_batched(src, fr_s_sub, cfg.encode_batch_size, device)          # Franka in source latent
    z_tgt_before = encode_obs_batched(tgt, ur_s_sub, cfg.encode_batch_size, device)   # UR in target latent
    z_tgt_after = map_obs_batched(aligner, z_tgt_before, cfg.encode_batch_size, device)  # UR mapped to source latent

    plot_alignment_scatter(
        z_src, z_tgt_before, z_tgt_after,
        os.path.join(cfg.output_dir, "alignment_scatter_before_after.png")
    )
    plot_alignment_hist(
        z_src, z_tgt_before, z_tgt_after,
        os.path.join(cfg.output_dir, "alignment_hist_before_after.png")
    )
    print("[PLOT] saved alignment_scatter_before_after.png / alignment_hist_before_after.png")

    # ---------- save metrics summary ----------
    summary = {
        "one_step": {
            "franka_mse": fr_mse,
            "franka_mae": fr_mae,
            "ur_mse": ur_mse,
            "ur_mae": ur_mae,
        },
        "rollout": {
            "franka_avg_mse": fr_roll_mse,
            "ur_avg_mse": ur_roll_mse,
            "horizon": H,
        },
        "meta": {
            "koopman_pack": cfg.koopman_pack,
            "src_ckpt": cfg.src_ckpt,
            "tgt_ckpt": cfg.tgt_ckpt,
            "stage2_bundle": cfg.stage2_bundle,
            "distribution_sample_points": cfg.distribution_sample_points,
        }
    }

    import json
    with open(os.path.join(cfg.output_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[DONE] all evaluation files saved to: {cfg.output_dir}")


if __name__ == "__main__":
    main()
