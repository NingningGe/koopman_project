#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standalone evaluator for Stage-3 Shi-style Koopman checkpoints.

It reads the saved file produced by train_bc_franka_wang_rewrite_shi_stage3.py,
computes several diagnostics, and saves all figures with labeled axes.

Default usage:
python test_koopman_shi_model.py \
  --ckpt /home/nng/koopman_project/out_align_koopman_3/koopman_fit/aligned_latent_and_koopman_shi.pt \
  --out_dir /home/nng/koopman_project/out_align_koopman_3/koopman_fit/test_d12
"""

import argparse
import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def load_payload(path: Path):
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict) or "data" not in obj:
        raise ValueError(f"Unexpected checkpoint format: {path}")
    return obj


def reshape_episode(flat: torch.Tensor, episode_len: int):
    n = flat.shape[0]
    if n % episode_len != 0:
        raise ValueError(f"Flat tensor length {n} is not divisible by episode_len={episode_len}")
    return flat.reshape(n // episode_len, episode_len, flat.shape[-1])


def one_step_predict(z_seq: np.ndarray, u_seq: np.ndarray, A: np.ndarray, B: np.ndarray):
    z_t = z_seq[:, :-1, :]
    u_t = u_seq[:, :-1, :]
    z_next_gt = z_seq[:, 1:, :]
    z_next_pred = z_t @ A.T + u_t @ B.T
    return z_next_pred, z_next_gt


def rollout_predict(z0: np.ndarray, u_seq: np.ndarray, A: np.ndarray, B: np.ndarray):
    h = u_seq.shape[1]
    z = z0.copy()
    preds = []
    for k in range(h):
        z = z @ A.T + u_seq[:, k, :] @ B.T
        preds.append(z.copy())
    return np.stack(preds, axis=1)


def mse_over_horizon(z_seq: np.ndarray, u_seq: np.ndarray, A: np.ndarray, B: np.ndarray, C: np.ndarray, max_h: int):
    max_h = min(max_h, z_seq.shape[1] - 1)
    z0 = z_seq[:, 0, :]
    z_roll = rollout_predict(z0, u_seq[:, :max_h, :], A, B)
    z_gt = z_seq[:, 1:max_h + 1, :]
    y_roll = np.einsum('ij,btj->bti', C, z_roll)
    y_gt = np.einsum('ij,btj->bti', C, z_gt)

    mse_z = ((z_roll - z_gt) ** 2).mean(axis=(0, 2))
    mse_y = ((y_roll - y_gt) ** 2).mean(axis=(0, 2))
    return np.arange(1, max_h + 1), mse_z, mse_y, z_roll, z_gt, y_roll, y_gt


def plot_line(x, y, save_path: Path, title: str, xlabel: str, ylabel: str):
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker='o', markersize=3)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def plot_two_lines(x, y1, y2, save_path: Path, title: str, xlabel: str, ylabel: str, label1: str, label2: str):
    plt.figure(figsize=(8, 5))
    plt.plot(x, y1, marker='o', markersize=3, label=label1)
    plt.plot(x, y2, marker='s', markersize=3, label=label2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def plot_matrix(mat: np.ndarray, save_path: Path, title: str, xlabel: str, ylabel: str):
    plt.figure(figsize=(8, 6))
    im = plt.imshow(mat, aspect='auto')
    plt.colorbar(im)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def plot_eigs(A: np.ndarray, save_path: Path, title: str):
    eigvals = np.linalg.eigvals(A)
    theta = np.linspace(0, 2 * np.pi, 400)
    plt.figure(figsize=(6, 6))
    plt.plot(np.cos(theta), np.sin(theta), linestyle='--', label='Unit circle')
    plt.scatter(eigvals.real, eigvals.imag, s=24, label='Eigenvalues')
    plt.axhline(0.0, linewidth=1)
    plt.axvline(0.0, linewidth=1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(title)
    plt.xlabel('Real part')
    plt.ylabel('Imaginary part')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()
    return eigvals


def plot_singular_values(vals: np.ndarray, save_path: Path, title: str):
    x = np.arange(1, len(vals) + 1)
    plt.figure(figsize=(8, 5))
    plt.semilogy(x, vals, marker='o', markersize=3)
    plt.title(title)
    plt.xlabel('Index')
    plt.ylabel('Singular value (log scale)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def plot_hist(data: np.ndarray, save_path: Path, title: str, xlabel: str, ylabel: str):
    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=50)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def plot_episode_dims(gt_seq: np.ndarray, pred_seq: np.ndarray, save_dir: Path, prefix: str, title_prefix: str, max_dims: int = 8):
    d = min(gt_seq.shape[1], max_dims)
    t = np.arange(gt_seq.shape[0])
    for i in range(d):
        plt.figure(figsize=(9, 4))
        plt.plot(t, gt_seq[:, i], label='Ground truth')
        plt.plot(t, pred_seq[:, i], label='Prediction')
        plt.title(f'{title_prefix} dimension {i}')
        plt.xlabel('Time step')
        plt.ylabel(f'Value of dimension {i}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / f'{prefix}_dim_{i}.png', dpi=160)
        plt.close()


def controllability_matrix(A: np.ndarray, B: np.ndarray):
    n = A.shape[0]
    blocks = [B]
    cur = B.copy()
    for _ in range(1, n):
        cur = A @ cur
        blocks.append(cur)
    return np.concatenate(blocks, axis=1)


def evaluate_domain(name: str, z_seq: np.ndarray, u_seq: np.ndarray, A: np.ndarray, B: np.ndarray, C: np.ndarray,
                    out_dir: Path, max_h: int, episode_index: int):
    mkdir(out_dir)

    # One-step prediction
    z1_pred, z1_gt = one_step_predict(z_seq, u_seq, A, B)
    y1_pred = np.einsum('ij,btj->bti', C, z1_pred)
    y1_gt = np.einsum('ij,btj->bti', C, z1_gt)
    one_step_mse_z = float(np.mean((z1_pred - z1_gt) ** 2))
    one_step_mse_y = float(np.mean((y1_pred - y1_gt) ** 2))
    per_step_y = np.mean((y1_pred - y1_gt) ** 2, axis=(0, 2))
    plot_line(
        np.arange(1, len(per_step_y) + 1),
        per_step_y,
        out_dir / f'{name}_one_step_mse_vs_time.png',
        f'{name} one-step prediction MSE at each time index',
        'Time index inside episode',
        'Mean squared error in Y space',
    )

    # Multi-step rollout from episode start
    horizons, mse_z_h, mse_y_h, z_roll, z_gt, y_roll, y_gt = mse_over_horizon(z_seq, u_seq, A, B, C, max_h)
    plot_two_lines(
        horizons,
        mse_z_h,
        mse_y_h,
        out_dir / f'{name}_rollout_mse_vs_horizon.png',
        f'{name} rollout MSE versus horizon',
        'Rollout horizon',
        'Mean squared error',
        'Z-space MSE',
        'Y-space MSE',
    )

    # Final-horizon error histogram
    final_err_y = np.mean((y_roll[:, -1, :] - y_gt[:, -1, :]) ** 2, axis=1)
    plot_hist(
        final_err_y,
        out_dir / f'{name}_final_horizon_error_histogram.png',
        f'{name} distribution of final-horizon rollout error in Y space',
        'Episode-wise mean squared error at final horizon',
        'Count',
    )

    # Random/selected episode time series
    ep = min(max(episode_index, 0), z_seq.shape[0] - 1)
    ep_h = min(max_h, z_seq.shape[1] - 1)
    z_ep_roll = rollout_predict(z_seq[[ep], 0, :], u_seq[[ep], :ep_h, :], A, B)[0]
    z_ep_gt = z_seq[ep, 1:ep_h + 1, :]
    y_ep_roll = (C @ z_ep_roll.T).T
    y_ep_gt = (C @ z_ep_gt.T).T
    plot_episode_dims(
        y_ep_gt,
        y_ep_roll,
        out_dir,
        f'{name}_episode_{ep}_y',
        f'{name} episode {ep} Y-space rollout',
        max_dims=min(8, y_ep_gt.shape[1]),
    )
    plot_episode_dims(
        z_ep_gt,
        z_ep_roll,
        out_dir,
        f'{name}_episode_{ep}_z',
        f'{name} episode {ep} Z-space rollout',
        max_dims=min(8, z_ep_gt.shape[1]),
    )

    return {
        'one_step_mse_z': one_step_mse_z,
        'one_step_mse_y': one_step_mse_y,
        'rollout_mse_z_last_horizon': float(mse_z_h[-1]),
        'rollout_mse_y_last_horizon': float(mse_y_h[-1]),
        'rollout_mse_z_curve': mse_z_h.tolist(),
        'rollout_mse_y_curve': mse_y_h.tolist(),
        'selected_episode_index': int(ep),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help='Path to aligned_latent_and_koopman_shi.pt')
    parser.add_argument('--out_dir', type=str, required=True, help='Directory to save plots and metrics')
    parser.add_argument('--max_horizon', type=int, default=50, help='Maximum rollout horizon to evaluate')
    parser.add_argument('--episode_index', type=int, default=0, help='Which episode to visualize')
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    out_dir = Path(args.out_dir)
    mkdir(out_dir)

    payload = load_payload(ckpt_path)
    meta = payload['meta']
    data = payload['data']

    episode_len = int(meta['episode_len'])
    A = to_numpy(data['A']).astype(np.float64)
    B = to_numpy(data['B']).astype(np.float64)
    C = to_numpy(data['C']).astype(np.float64)

    fr_z = reshape_episode(data['fr_z'], episode_len)
    fr_u = reshape_episode(data['fr_u'], episode_len)
    ur_z = reshape_episode(data['ur_z'], episode_len)
    ur_u = reshape_episode(data['ur_u'], episode_len)

    fr_z = to_numpy(fr_z).astype(np.float64)
    fr_u = to_numpy(fr_u).astype(np.float64)
    ur_z = to_numpy(ur_z).astype(np.float64)
    ur_u = to_numpy(ur_u).astype(np.float64)

    # Global matrix diagnostics
    plot_matrix(A, out_dir / 'A_matrix_heatmap.png', 'Koopman state matrix A', 'Column index', 'Row index')
    plot_matrix(B, out_dir / 'B_matrix_heatmap.png', 'Koopman control matrix B', 'Control input index', 'State row index')
    eigvals = plot_eigs(A, out_dir / 'A_eigenvalues.png', 'Eigenvalues of Koopman matrix A')
    spectral_radius = float(np.max(np.abs(eigvals)))

    ctrl = controllability_matrix(A, B)
    ctrl_rank = int(np.linalg.matrix_rank(ctrl))
    ctrl_sv = np.linalg.svd(ctrl, compute_uv=False)
    plot_singular_values(ctrl_sv, out_dir / 'controllability_singular_values.png', 'Singular values of controllability matrix')

    # Domain-wise tests
    fr_dir = out_dir / 'franka'
    ur_dir = out_dir / 'ur_mapped'
    fr_metrics = evaluate_domain('franka', fr_z, fr_u, A, B, C, fr_dir, args.max_horizon, args.episode_index)
    ur_metrics = evaluate_domain('ur_mapped', ur_z, ur_u, A, B, C, ur_dir, args.max_horizon, args.episode_index)

    # Cross-domain comparison plots
    horizons = np.arange(1, min(args.max_horizon, episode_len - 1) + 1)
    plot_two_lines(
        horizons,
        np.asarray(fr_metrics['rollout_mse_y_curve']),
        np.asarray(ur_metrics['rollout_mse_y_curve']),
        out_dir / 'compare_rollout_mse_y_vs_horizon.png',
        'Cross-domain rollout MSE in Y space',
        'Rollout horizon',
        'Mean squared error in Y space',
        'Franka',
        'UR mapped',
    )
    plot_two_lines(
        horizons,
        np.asarray(fr_metrics['rollout_mse_z_curve']),
        np.asarray(ur_metrics['rollout_mse_z_curve']),
        out_dir / 'compare_rollout_mse_z_vs_horizon.png',
        'Cross-domain rollout MSE in Z space',
        'Rollout horizon',
        'Mean squared error in Z space',
        'Franka',
        'UR mapped',
    )

    summary = {
        'checkpoint': str(ckpt_path),
        'episode_len': episode_len,
        'A_shape': list(A.shape),
        'B_shape': list(B.shape),
        'C_shape': list(C.shape),
        'spectral_radius_of_A': spectral_radius,
        'controllability_rank': ctrl_rank,
        'controllability_shape': list(ctrl.shape),
        'franka_metrics': fr_metrics,
        'ur_mapped_metrics': ur_metrics,
        'saved_figures': sorted(str(p.relative_to(out_dir)) for p in out_dir.rglob('*.png')),
    }

    with open(out_dir / 'summary_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(out_dir / 'README.txt', 'w', encoding='utf-8') as f:
        f.write(
            'Saved diagnostics for Shi-style Koopman model\n\n'
            f'Checkpoint: {ckpt_path}\n'
            f'Episode length: {episode_len}\n'
            f'A shape: {A.shape}\n'
            f'B shape: {B.shape}\n'
            f'Spectral radius of A: {spectral_radius:.6f}\n'
            f'Controllability rank: {ctrl_rank}\n\n'
            'Main outputs:\n'
            '- A_matrix_heatmap.png: heatmap of Koopman matrix A\n'
            '- B_matrix_heatmap.png: heatmap of control matrix B\n'
            '- A_eigenvalues.png: eigenvalues of A with unit circle\n'
            '- controllability_singular_values.png: singular values of controllability matrix\n'
            '- franka/: one-step, rollout, histogram, and per-dimension time-series plots\n'
            '- ur_mapped/: one-step, rollout, histogram, and per-dimension time-series plots\n'
            '- compare_rollout_mse_y_vs_horizon.png: Franka vs UR rollout error in Y space\n'
            '- compare_rollout_mse_z_vs_horizon.png: Franka vs UR rollout error in Z space\n'
        )

    print(f'[DONE] Saved figures and metrics to: {out_dir}')
    print(f'[INFO] Spectral radius of A: {spectral_radius:.6f}')
    print(f'[INFO] Controllability rank: {ctrl_rank} / {A.shape[0]}')
    print(f'[INFO] Franka rollout Y MSE @ horizon {min(args.max_horizon, episode_len - 1)}: {fr_metrics["rollout_mse_y_last_horizon"]:.6f}')
    print(f'[INFO] UR(mapped) rollout Y MSE @ horizon {min(args.max_horizon, episode_len - 1)}: {ur_metrics["rollout_mse_y_last_horizon"]:.6f}')


if __name__ == '__main__':
    main()
