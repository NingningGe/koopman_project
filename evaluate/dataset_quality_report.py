#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dataset_quality_report.py

用途：
对单个机器人数据集做“可训练性 / 覆盖度 / 多样性 / latent健康度”诊断，
适合在重采数据后快速判断这份数据是否适合做 Koopman / BC / 对齐。

支持格式：
{
  "meta": ...,
  "data": {
    "s": (N_eps, T, obs_dim),
    "a": (N_eps, T, act_dim),
    "s_next": (N_eps, T, obs_dim),
    ...
  }
}

输出：
- quality_summary.json
- raw_state_pca.png
- delta_action_density.png
- delta_action_count.png
- raw_state_hist.png
- episode_diversity_hist.png
- state_cov_eigs.png
- report.txt

使用：
1) 只评估原始数据：
   python dataset_quality_report.py

2) 如需同时评估 latent 健康度：
   把 eval_latent=True，并填写 ckpt_path 与模型维度
"""

import os
import json
from dataclasses import dataclass, asdict
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


@dataclass
class Config:
    # 默认先写成 Franka v2 数据
    dataset_path: str = "/home/nng/koopman_project/data_wang_franka_reach_ablation/dataset_wang_franka_reach_ablation_v2.pt"
    out_dir: str = "/home/nng/koopman_project/evaluate/dataset_quality_report"

    # 数据维度
    obs_dim: int = 14
    act_dim: int = 7

    # latent 检查
    eval_latent: bool = False
    ckpt_path: str = "/home/nng/koopman_project/out_align_koopman/src_franka/src_agent.pt"
    lat_obs_dim: int = 4
    lat_act_dim: int = 4
    hidden_dim: int = 256
    n_layers: int = 3
    sat_threshold: float = 0.95

    # 分析参数
    seed: int = 0
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    pca_points: int = 10000
    hist_bins: int = 80
    raw_hist_dims_show: int = 14
    action_hist_max: float = 4.0
    delta_hist_auto_q99: float = 0.995
    episode_sample_for_diversity: int = 1000


cfg = Config()


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
def pca_2d(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    x = x.float()
    x = x - x.mean(dim=0, keepdim=True)
    _, S, Vh = torch.linalg.svd(x, full_matrices=False)
    comps = Vh[:2].T
    y = x @ comps
    eigvals = (S ** 2) / max(x.shape[0] - 1, 1)
    var_ratio = eigvals / eigvals.sum()
    return y, var_ratio


def sample_rows(x: torch.Tensor, n: int) -> torch.Tensor:
    idx = torch.randperm(x.shape[0])[:min(n, x.shape[0])]
    return x[idx]


def flatten_seq(s: torch.Tensor, a: torch.Tensor, s_next: torch.Tensor):
    return s.reshape(-1, s.shape[-1]), a.reshape(-1, a.shape[-1]), s_next.reshape(-1, s_next.shape[-1])


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


# ========= 可选 latent 检查 =========
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


@torch.no_grad()
def encode_obs_batched(agent, s_all: torch.Tensor, batch_size: int, device: torch.device):
    outs = []
    agent.eval()
    for i in range(0, s_all.shape[0], batch_size):
        s = s_all[i:i + batch_size].to(device)
        z = agent.obs_enc(s).detach().cpu()
        outs.append(z)
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


# ========= 指标 =========
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


def compute_episode_diversity(s_eps: torch.Tensor, max_eps: int) -> Dict:
    n = min(max_eps, s_eps.shape[0])
    idx = torch.randperm(s_eps.shape[0])[:n]
    x = s_eps[idx].mean(dim=1)  # (n, D)

    c = x.mean(dim=0, keepdim=True)
    dist_to_centroid = torch.norm(x - c, dim=1)

    ref = x[: min(128, n)]
    dmat = torch.cdist(x, ref)
    nn_dist = dmat.min(dim=1).values

    return {
        "num_sampled_episodes": int(n),
        "dist_to_centroid": scalar_range_stats(dist_to_centroid),
        "nearest_ref_dist": scalar_range_stats(nn_dist),
        "dist_to_centroid_values": dist_to_centroid.tolist(),
    }


def make_score(summary: Dict) -> Dict:
    rank = summary["state_cov"]["effective_rank_eps_ratio_1e3"]
    pr = summary["state_cov"]["participation_ratio"]
    obs_dim = summary["dataset"]["obs_dim"]

    delta_mean = summary["delta_norm"]["mean"]
    div_mean = summary["episode_diversity"]["dist_to_centroid"]["mean"]

    raw_score = 0.0
    raw_score += min(rank / max(obs_dim, 1), 1.0) * 35.0
    raw_score += min(pr / max(obs_dim, 1), 1.0) * 25.0
    raw_score += min(delta_mean / 0.01, 1.0) * 20.0
    raw_score += min(div_mean / 0.5, 1.0) * 20.0

    latent_score = None
    if "latent" in summary:
        lat = summary["latent"]
        avg_std = lat["avg_std"]
        avg_sat = lat["avg_saturation"]
        collapsed = len(lat["collapsed_dims_std_lt_0p03"])
        saturated = len(lat["saturated_dims_ratio_gt_0p5"])
        lat_dim = len(lat["std"])

        score = 100.0
        score -= min(avg_sat, 1.0) * 50.0
        score -= min(collapsed / max(lat_dim, 1), 1.0) * 30.0
        score -= min(saturated / max(lat_dim, 1), 1.0) * 20.0
        score += min(avg_std / 0.3, 1.0) * 10.0
        latent_score = max(0.0, min(score, 100.0))

    overall = raw_score if latent_score is None else 0.6 * raw_score + 0.4 * latent_score

    return {
        "raw_quality_score_0_100": round(float(max(0.0, min(raw_score, 100.0))), 2),
        "latent_quality_score_0_100": None if latent_score is None else round(float(latent_score), 2),
        "overall_score_0_100": round(float(max(0.0, min(overall, 100.0))), 2),
    }


# ========= 绘图 =========
def plot_raw_state_pca(s_flat: torch.Tensor, save_path: str):
    y, vr = pca_2d(sample_rows(s_flat, cfg.pca_points))
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y[:, 0], y[:, 1], s=7, alpha=0.35)
    ax.set_title("Raw state PCA")
    ax.set_xlabel(f"PC1 ({vr[0].item()*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({vr[1].item()*100:.1f}%)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_delta_action(delta: torch.Tensor, action_norm: torch.Tensor, save_density: str, save_count: str):
    dmax = float(torch.quantile(delta, cfg.delta_hist_auto_q99).item())
    dmax = max(dmax, 1e-6)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(delta.numpy(), bins=cfg.hist_bins, range=(0, dmax), density=True, alpha=0.8)
    axes[0].set_title(r"$\|s_{t+1} - s_t\|$ (density)")
    axes[1].hist(action_norm.numpy(), bins=cfg.hist_bins, range=(0, cfg.action_hist_max), density=True, alpha=0.8)
    axes[1].set_title(r"$\|a_t\|$ (density)")
    plt.tight_layout()
    plt.savefig(save_density, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(delta.numpy(), bins=cfg.hist_bins, range=(0, dmax), density=False, alpha=0.8)
    axes[0].set_title(r"$\|s_{t+1} - s_t\|$ (count)")
    axes[1].hist(action_norm.numpy(), bins=cfg.hist_bins, range=(0, cfg.action_hist_max), density=False, alpha=0.8)
    axes[1].set_title(r"$\|a_t\|$ (count)")
    plt.tight_layout()
    plt.savefig(save_count, dpi=180)
    plt.close(fig)


def plot_raw_state_hist(s_flat: torch.Tensor, save_path: str):
    show_dims = min(cfg.raw_hist_dims_show, s_flat.shape[1])
    ncols = 2
    nrows = int(np.ceil(show_dims / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 2.7 * nrows), squeeze=False)
    for d in range(show_dims):
        r, c = divmod(d, ncols)
        axes[r, c].hist(s_flat[:, d].numpy(), bins=cfg.hist_bins, density=True, alpha=0.8)
        axes[r, c].set_title(f"raw state dim {d}")
    for d in range(show_dims, nrows * ncols):
        r, c = divmod(d, ncols)
        axes[r, c].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_episode_diversity_hist(values, save_path: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(values, bins=cfg.hist_bins, alpha=0.8, density=True)
    ax.set_title("Episode diversity: distance to centroid")
    ax.set_xlabel("distance")
    ax.set_ylabel("density")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_state_cov_eigs(eigvals, explained_ratio, save_path: str):
    eigvals = np.asarray(eigvals)
    explained_ratio = np.asarray(explained_ratio)
    xs = np.arange(len(eigvals))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(xs, eigvals)
    axes[0].set_title("State covariance eigenvalues")
    axes[0].set_xlabel("index")
    axes[0].set_ylabel("eigval")

    axes[1].bar(xs, explained_ratio)
    axes[1].set_title("Explained variance ratio")
    axes[1].set_xlabel("index")
    axes[1].set_ylabel("ratio")

    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close(fig)


# ========= 文本报告 =========
def build_report(summary: Dict, scores: Dict) -> str:
    lines = []
    lines.append("=== Dataset Quality Report ===")
    lines.append("")
    lines.append(f"dataset_path: {summary['dataset']['dataset_path']}")
    lines.append(f"num_episodes: {summary['dataset']['num_episodes']}")
    lines.append(f"episode_len: {summary['dataset']['episode_len']}")
    lines.append(f"num_transitions: {summary['dataset']['num_transitions']}")
    lines.append(f"obs_dim: {summary['dataset']['obs_dim']}")
    lines.append(f"act_dim: {summary['dataset']['act_dim']}")
    lines.append("")

    lines.append("--- Raw coverage ---")
    lines.append(f"delta_norm mean/std: {summary['delta_norm']['mean']:.6f} / {summary['delta_norm']['std']:.6f}")
    lines.append(f"action_norm mean/std: {summary['action_norm']['mean']:.6f} / {summary['action_norm']['std']:.6f}")
    lines.append(f"effective rank: {summary['state_cov']['effective_rank_eps_ratio_1e3']}")
    lines.append(f"participation ratio: {summary['state_cov']['participation_ratio']:.3f}")
    lines.append(f"episode diversity mean: {summary['episode_diversity']['dist_to_centroid']['mean']:.6f}")
    lines.append("")

    if "latent" in summary:
        lat = summary["latent"]
        lines.append("--- Latent health ---")
        lines.append(f"avg latent std: {lat['avg_std']:.6f}")
        lines.append(f"avg saturation ratio: {lat['avg_saturation']:.6f}")
        lines.append(f"collapsed dims: {lat['collapsed_dims_std_lt_0p03']}")
        lines.append(f"saturated dims: {lat['saturated_dims_ratio_gt_0p5']}")
        lines.append("")

    lines.append("--- Scores ---")
    lines.append(f"raw_quality_score_0_100: {scores['raw_quality_score_0_100']}")
    if scores["latent_quality_score_0_100"] is not None:
        lines.append(f"latent_quality_score_0_100: {scores['latent_quality_score_0_100']}")
    lines.append(f"overall_score_0_100: {scores['overall_score_0_100']}")
    lines.append("")

    lines.append("--- Diagnosis ---")
    rank = summary["state_cov"]["effective_rank_eps_ratio_1e3"]
    obs_dim = summary["dataset"]["obs_dim"]
    delta_mean = summary["delta_norm"]["mean"]
    div_mean = summary["episode_diversity"]["dist_to_centroid"]["mean"]

    if rank < max(3, obs_dim // 4):
        lines.append("- 原始状态有效秩偏低：数据可能集中在很窄的流形上。")
    else:
        lines.append("- 原始状态有效秩尚可。")

    if delta_mean < 1e-3:
        lines.append("- 每步状态变化非常小：轨迹可能过于平滑或动作未有效激发系统。")
    elif delta_mean < 3e-3:
        lines.append("- 每步状态变化偏小。")
    else:
        lines.append("- 每步状态变化幅度尚可。")

    if div_mean < 0.05:
        lines.append("- episode 间多样性偏低：很多轨迹可能来自相近初始条件或相似控制分支。")
    else:
        lines.append("- episode 间多样性尚可。")

    if "latent" in summary:
        lat = summary["latent"]
        if lat["avg_saturation"] > 0.5:
            lines.append("- latent 严重饱和：encoder 很可能发生了 collapse。")
        elif lat["avg_saturation"] > 0.2:
            lines.append("- latent 有一定饱和现象。")
        else:
            lines.append("- latent 饱和不明显。")

        if len(lat["collapsed_dims_std_lt_0p03"]) > 0:
            lines.append("- 存在塌缩维度。")
        else:
            lines.append("- 未发现明显塌缩维度。")

    return "\n".join(lines)


def main():
    set_seed(cfg.seed)
    ensure_dir(cfg.out_dir)

    print(f"[INFO] device={cfg.device}")
    data = load_dataset(cfg.dataset_path)

    s = data["s"].float()
    a = data["a"].float()
    s_next = data["s_next"].float()

    n_eps, T, obs_dim = s.shape
    act_dim = a.shape[-1]

    s_flat, a_flat, sn_flat = flatten_seq(s, a, s_next)
    delta = torch.norm(sn_flat - s_flat, dim=1)
    action_norm = torch.norm(a_flat, dim=1)

    print(f"[INFO] dataset: s={tuple(s.shape)} a={tuple(a.shape)} s_next={tuple(s_next.shape)}")

    state_cov = compute_cov_rank(sample_rows(s_flat, min(50000, s_flat.shape[0])))
    ep_div = compute_episode_diversity(s, cfg.episode_sample_for_diversity)

    summary = {
        "config": asdict(cfg),
        "dataset": {
            "dataset_path": cfg.dataset_path,
            "num_episodes": int(n_eps),
            "episode_len": int(T),
            "num_transitions": int(s_flat.shape[0]),
            "obs_dim": int(obs_dim),
            "act_dim": int(act_dim),
        },
        "raw_state": tensor_stats(s_flat),
        "delta_norm": scalar_range_stats(delta),
        "action_norm": scalar_range_stats(action_norm),
        "state_cov": state_cov,
        "episode_diversity": ep_div,
    }

    if cfg.eval_latent:
        device = torch.device(cfg.device)
        agent = Agent(
            obs_dim=cfg.obs_dim,
            act_dim=cfg.act_dim,
            lat_obs_dim=cfg.lat_obs_dim,
            lat_act_dim=cfg.lat_act_dim,
            n_layers=cfg.n_layers,
            hidden_dim=cfg.hidden_dim,
        ).to(device)
        agent.load_state_dict(torch.load(cfg.ckpt_path, map_location=device))
        z = encode_obs_batched(agent, sample_rows(s_flat, min(200000, s_flat.shape[0])), 8192, device)
        summary["latent"] = latent_quality(z, cfg.sat_threshold)

    scores = make_score(summary)
    summary["scores"] = scores

    plot_raw_state_pca(s_flat, os.path.join(cfg.out_dir, "raw_state_pca.png"))
    plot_delta_action(delta, action_norm,
                      os.path.join(cfg.out_dir, "delta_action_density.png"),
                      os.path.join(cfg.out_dir, "delta_action_count.png"))
    plot_raw_state_hist(s_flat, os.path.join(cfg.out_dir, "raw_state_hist.png"))
    plot_episode_diversity_hist(ep_div["dist_to_centroid_values"],
                                os.path.join(cfg.out_dir, "episode_diversity_hist.png"))
    plot_state_cov_eigs(state_cov["eigvals"], state_cov["explained_ratio"],
                        os.path.join(cfg.out_dir, "state_cov_eigs.png"))

    with open(os.path.join(cfg.out_dir, "quality_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    report = build_report(summary, scores)
    with open(os.path.join(cfg.out_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    print("\n" + report)
    print(f"\n[DONE] saved to: {cfg.out_dir}")


if __name__ == "__main__":
    main()
