#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_franka_ur_data_quality.py

对比 Franka 与 UR 两套数据的原始数据质量和 latent 质量。

输出：
- franka_vs_ur_raw_pca.png
- franka_vs_ur_delta_action_hist_density.png
- franka_vs_ur_delta_action_hist_count.png
- franka_vs_ur_raw_hist_overlay.png
- franka_vs_ur_latent_pca.png
- franka_vs_ur_latent_hist_overlay.png
- franka_vs_ur_saturation_bar.png
- franka_vs_ur_summary.json
"""

import os
import json
from dataclasses import dataclass, asdict
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


@dataclass
class Cfg:
    franka_dataset: str = "/home/nng/koopman_project/data_wang_franka_reach_ablation/dataset_wang_franka_reach_ablation_v2.pt"
    ur_dataset: str = "/home/nng/koopman_project/data_wang_ur_reach_ablation/dataset_wang_ur_reach_ablation.pt"

    src_ckpt: str = "/home/nng/koopman_project/out_align_koopman_3/src_franka/src_agent.pt"
    tgt_ckpt: str = "/home/nng/koopman_project/out_align_koopman_3/tgt_ur/tgt_agent.pt"
    stage2_bundle: str = "/home/nng/koopman_project/out_align_koopman_3/tgt_ur/stage2_bundle.pt"

    out_dir: str = "/home/nng/koopman_project/out_align_koopman_3/compare_franka_ur"

    franka_obs_dim: int = 14
    franka_act_dim: int = 7
    ur_obs_dim: int = 12
    ur_act_dim: int = 6
    lat_obs_dim: int = 4
    lat_act_dim: int = 4
    hidden_dim: int = 256
    n_layers: int = 3

    seed: int = 0
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    pca_points: int = 10000
    latent_points: int = 10000
    hist_bins: int = 80
    raw_hist_dims_show: int = 8
    sat_threshold: float = 0.95

    delta_hist_max: float = 0.02
    action_hist_max: float = 4.0


cfg = Cfg()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_dataset(path: str) -> Dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu")
    return obj["data"] if isinstance(obj, dict) and "data" in obj else obj


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


@torch.no_grad()
def pca_2d(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    x = x - x.mean(dim=0, keepdim=True)
    _, _, Vh = torch.linalg.svd(x, full_matrices=False)
    comps = Vh[:2].T
    return x @ comps


def flatten_seq(s: torch.Tensor, a: torch.Tensor, s_next: torch.Tensor):
    return s.reshape(-1, s.shape[-1]), a.reshape(-1, a.shape[-1]), s_next.reshape(-1, s_next.shape[-1])


def sample_rows(x: torch.Tensor, n: int) -> torch.Tensor:
    idx = torch.randperm(x.shape[0])[:min(n, x.shape[0])]
    return x[idx]


def stats_tensor(x: torch.Tensor) -> Dict:
    return {
        "shape": list(x.shape),
        "mean": x.mean(dim=0).tolist(),
        "std": x.std(dim=0).tolist(),
        "min": x.min(dim=0).values.tolist(),
        "max": x.max(dim=0).values.tolist(),
    }


def range_scalar(x: torch.Tensor) -> Dict:
    return {
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "mean": float(x.mean().item()),
        "std": float(x.std().item()),
        "q01": float(torch.quantile(x, 0.01).item()),
        "q50": float(torch.quantile(x, 0.50).item()),
        "q99": float(torch.quantile(x, 0.99).item()),
    }


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
    def __init__(self, lat_obs_dim: int, lat_act_dim: int, n_layers: int, hidden_dim: int):
        super().__init__()
        self.z_t2s = build_mlp(lat_obs_dim, lat_obs_dim, n_layers, hidden_dim, activation="relu", out_act="identity")
        self.z_s2t = build_mlp(lat_obs_dim, lat_obs_dim, n_layers, hidden_dim, activation="relu", out_act="identity")
        self.u_t2s = build_mlp(lat_act_dim, lat_act_dim, n_layers, hidden_dim, activation="relu", out_act="identity")
        self.u_s2t = build_mlp(lat_act_dim, lat_act_dim, n_layers, hidden_dim, activation="relu", out_act="identity")


@torch.no_grad()
def encode_obs_batched(agent, s_all: torch.Tensor, batch_size: int, device: torch.device):
    outs = []
    agent.eval()
    for i in range(0, s_all.shape[0], batch_size):
        s = s_all[i:i+batch_size].to(device)
        z = agent.obs_enc(s).detach().cpu()
        outs.append(z)
    return torch.cat(outs, dim=0)


@torch.no_grad()
def map_obs_batched(aligner, z_all: torch.Tensor, batch_size: int, device: torch.device):
    outs = []
    aligner.eval()
    for i in range(0, z_all.shape[0], batch_size):
        z = z_all[i:i+batch_size].to(device)
        y = aligner.z_t2s(z).detach().cpu()
        outs.append(y)
    return torch.cat(outs, dim=0)


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


def plot_raw_pca(fr_s: torch.Tensor, ur_s: torch.Tensor, save_path: str):
    ur_pad = torch.cat([ur_s, torch.zeros(ur_s.shape[0], fr_s.shape[1] - ur_s.shape[1])], dim=1)
    z_all = torch.cat([fr_s, ur_pad], dim=0)
    y_all = pca_2d(z_all)
    n = fr_s.shape[0]
    y_fr = y_all[:n]
    y_ur = y_all[n:]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_fr[:, 0], y_fr[:, 1], s=6, alpha=0.35, label="Franka raw")
    ax.scatter(y_ur[:, 0], y_ur[:, 1], s=6, alpha=0.35, label="UR raw")
    ax.set_title("Franka vs UR raw state PCA")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_delta_action_hist(fr_delta: torch.Tensor, ur_delta: torch.Tensor,
                           fr_act: torch.Tensor, ur_act: torch.Tensor,
                           save_density: str, save_count: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(fr_delta.numpy(), bins=cfg.hist_bins, range=(0, cfg.delta_hist_max), density=True, alpha=0.6, label="Franka")
    axes[0].hist(ur_delta.numpy(), bins=cfg.hist_bins, range=(0, cfg.delta_hist_max), density=True, alpha=0.6, label="UR")
    axes[0].set_title(r"$\|s_{t+1}-s_t\|$ (density, same range)")
    axes[0].legend()

    axes[1].hist(fr_act.numpy(), bins=cfg.hist_bins, range=(0, cfg.action_hist_max), density=True, alpha=0.6, label="Franka")
    axes[1].hist(ur_act.numpy(), bins=cfg.hist_bins, range=(0, cfg.action_hist_max), density=True, alpha=0.6, label="UR")
    axes[1].set_title(r"$\|a_t\|$ (density, same range)")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(save_density, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(fr_delta.numpy(), bins=cfg.hist_bins, range=(0, cfg.delta_hist_max), density=False, alpha=0.6, label="Franka")
    axes[0].hist(ur_delta.numpy(), bins=cfg.hist_bins, range=(0, cfg.delta_hist_max), density=False, alpha=0.6, label="UR")
    axes[0].set_title(r"$\|s_{t+1}-s_t\|$ (count, same range)")
    axes[0].legend()

    axes[1].hist(fr_act.numpy(), bins=cfg.hist_bins, range=(0, cfg.action_hist_max), density=False, alpha=0.6, label="Franka")
    axes[1].hist(ur_act.numpy(), bins=cfg.hist_bins, range=(0, cfg.action_hist_max), density=False, alpha=0.6, label="UR")
    axes[1].set_title(r"$\|a_t\|$ (count, same range)")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(save_count, dpi=180)
    plt.close(fig)


def plot_raw_hist_overlay(fr_s: torch.Tensor, ur_s: torch.Tensor, save_path: str):
    show_dims = min(cfg.raw_hist_dims_show, min(fr_s.shape[1], ur_s.shape[1]))
    ncols = 2
    nrows = int(np.ceil(show_dims / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 2.8 * nrows), squeeze=False)
    for d in range(show_dims):
        r, c = divmod(d, ncols)
        ax = axes[r, c]
        xmin = min(float(fr_s[:, d].min().item()), float(ur_s[:, d].min().item()))
        xmax = max(float(fr_s[:, d].max().item()), float(ur_s[:, d].max().item()))
        ax.hist(fr_s[:, d].numpy(), bins=cfg.hist_bins, range=(xmin, xmax), density=True, alpha=0.6, label="Franka")
        ax.hist(ur_s[:, d].numpy(), bins=cfg.hist_bins, range=(xmin, xmax), density=True, alpha=0.6, label="UR")
        ax.set_title(f"raw state dim {d}")
        ax.legend()
    for d in range(show_dims, nrows * ncols):
        r, c = divmod(d, ncols)
        axes[r, c].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_latent_pca(fr_z: torch.Tensor, ur_before: torch.Tensor, ur_after: torch.Tensor, save_path: str):
    z_all = torch.cat([fr_z, ur_before, ur_after], dim=0)
    y = pca_2d(z_all)
    n1 = fr_z.shape[0]
    n2 = ur_before.shape[0]
    y1 = y[:n1]
    y2 = y[n1:n1+n2]
    y3 = y[n1+n2:]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(y1[:, 0], y1[:, 1], s=6, alpha=0.35, label="Franka source latent")
    axes[0].scatter(y2[:, 0], y2[:, 1], s=6, alpha=0.35, label="UR before align")
    axes[0].set_title("Before alignment")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].legend()

    axes[1].scatter(y1[:, 0], y1[:, 1], s=6, alpha=0.35, label="Franka source latent")
    axes[1].scatter(y3[:, 0], y3[:, 1], s=6, alpha=0.35, label="UR after align")
    axes[1].set_title("After alignment")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_latent_hist_overlay(fr_z: torch.Tensor, ur_before: torch.Tensor, ur_after: torch.Tensor, save_path: str):
    dz = fr_z.shape[1]
    fig, axes = plt.subplots(dz, 2, figsize=(11, 2.6 * dz), squeeze=False)
    for d in range(dz):
        xmin = min(float(fr_z[:, d].min()), float(ur_before[:, d].min()), float(ur_after[:, d].min()))
        xmax = max(float(fr_z[:, d].max()), float(ur_before[:, d].max()), float(ur_after[:, d].max()))
        axes[d, 0].hist(fr_z[:, d].numpy(), bins=cfg.hist_bins, range=(xmin, xmax), density=True, alpha=0.6, label="Franka")
        axes[d, 0].hist(ur_before[:, d].numpy(), bins=cfg.hist_bins, range=(xmin, xmax), density=True, alpha=0.6, label="UR before")
        axes[d, 0].set_title(f"latent dim {d} - before")
        axes[d, 0].legend()
        axes[d, 1].hist(fr_z[:, d].numpy(), bins=cfg.hist_bins, range=(xmin, xmax), density=True, alpha=0.6, label="Franka")
        axes[d, 1].hist(ur_after[:, d].numpy(), bins=cfg.hist_bins, range=(xmin, xmax), density=True, alpha=0.6, label="UR after")
        axes[d, 1].set_title(f"latent dim {d} - after")
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
    ax.set_title("Latent saturation ratio comparison")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close(fig)


def main():
    set_seed(cfg.seed)
    ensure_dir(cfg.out_dir)
    device = torch.device(cfg.device)
    print(f"[INFO] device={device}")

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

    src = SrcAgent(cfg.franka_obs_dim, cfg.franka_act_dim, cfg.lat_obs_dim, cfg.lat_act_dim, cfg.n_layers, cfg.hidden_dim).to(device)
    tgt = TgtAgent(cfg.ur_obs_dim, cfg.ur_act_dim, cfg.lat_obs_dim, cfg.lat_act_dim, cfg.n_layers, cfg.hidden_dim).to(device)
    aligner = DummyAligner(cfg.lat_obs_dim, cfg.lat_act_dim, cfg.n_layers, cfg.hidden_dim).to(device)

    src.load_state_dict(torch.load(cfg.src_ckpt, map_location=device))
    tgt.load_state_dict(torch.load(cfg.tgt_ckpt, map_location=device))
    bundle = torch.load(cfg.stage2_bundle, map_location=device)

    if "map_z_t2s" in bundle:
        aligner.z_t2s.load_state_dict(bundle["map_z_t2s"])
    elif "z_t2s" in bundle:
        aligner.z_t2s.load_state_dict(bundle["z_t2s"])
    else:
        raise RuntimeError("stage2_bundle 中未找到 z_t2s / map_z_t2s")

    src.eval(); tgt.eval(); aligner.eval()

    fr_s_sub = sample_rows(fr_s, cfg.pca_points)
    ur_s_sub = sample_rows(ur_s, cfg.pca_points)

    fr_z = encode_obs_batched(src, sample_rows(fr_s, cfg.latent_points), 8192, device)
    ur_z_before = encode_obs_batched(tgt, sample_rows(ur_s, cfg.latent_points), 8192, device)
    ur_z_after = map_obs_batched(aligner, ur_z_before, 8192, device)

    summary = {
        "config": asdict(cfg),
        "franka": {
            "num_transitions": int(fr_s.shape[0]),
            "raw_state": stats_tensor(fr_s),
            "delta_norm": range_scalar(fr_delta),
            "action_norm": range_scalar(fr_act_n),
            "latent_source": latent_quality(fr_z, cfg.sat_threshold),
        },
        "ur": {
            "num_transitions": int(ur_s.shape[0]),
            "raw_state": stats_tensor(ur_s),
            "delta_norm": range_scalar(ur_delta),
            "action_norm": range_scalar(ur_act_n),
            "latent_before_align": latent_quality(ur_z_before, cfg.sat_threshold),
            "latent_after_align": latent_quality(ur_z_after, cfg.sat_threshold),
        },
        "diagnosis": []
    }

    diag = []
    if summary["franka"]["delta_norm"]["mean"] < summary["ur"]["delta_norm"]["mean"]:
        diag.append("Franka 的每步状态变化均值小于 UR，说明 Franka 轨迹更局部、更平滑。")
    else:
        diag.append("Franka 的每步状态变化均值不小于 UR。")

    fr_lat = summary["franka"]["latent_source"]
    ur_b = summary["ur"]["latent_before_align"]
    ur_a = summary["ur"]["latent_after_align"]

    if fr_lat["avg_saturation"] > 0.5 and ur_b["avg_saturation"] < 0.5:
        diag.append("Franka source latent 存在显著饱和，而 UR 对齐前没有明显整体饱和：说明问题主要在 Franka 源域表征。")
    if len(fr_lat["collapsed_dims_std_lt_0p03"]) > 0 and len(ur_b["collapsed_dims_std_lt_0p03"]) == 0:
        diag.append("Franka source latent 存在塌缩维度，而 UR 对齐前没有塌缩维度：说明不是方法整体坏掉，而是源域 Stage1 更可能有问题。")
    if ur_a["avg_saturation"] < ur_b["avg_saturation"]:
        diag.append("UR 映射到 source latent 后饱和比例下降，说明 Stage2 映射网络本身是健康的。")

    summary["diagnosis"] = diag

    plot_raw_pca(fr_s_sub, ur_s_sub, os.path.join(cfg.out_dir, "franka_vs_ur_raw_pca.png"))
    plot_delta_action_hist(fr_delta, ur_delta, fr_act_n, ur_act_n,
                           os.path.join(cfg.out_dir, "franka_vs_ur_delta_action_hist_density.png"),
                           os.path.join(cfg.out_dir, "franka_vs_ur_delta_action_hist_count.png"))
    plot_raw_hist_overlay(fr_s_sub, ur_s_sub, os.path.join(cfg.out_dir, "franka_vs_ur_raw_hist_overlay.png"))
    plot_latent_pca(fr_z, ur_z_before, ur_z_after, os.path.join(cfg.out_dir, "franka_vs_ur_latent_pca.png"))
    plot_latent_hist_overlay(fr_z, ur_z_before, ur_z_after, os.path.join(cfg.out_dir, "franka_vs_ur_latent_hist_overlay.png"))
    plot_saturation_bar(fr_lat, ur_b, ur_a, os.path.join(cfg.out_dir, "franka_vs_ur_saturation_bar.png"))

    with open(os.path.join(cfg.out_dir, "franka_vs_ur_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n========== Quick Summary ==========")
    for s in diag:
        print("-", s)
    print(f"[DONE] all outputs saved to: {cfg.out_dir}")


if __name__ == "__main__":
    main()
