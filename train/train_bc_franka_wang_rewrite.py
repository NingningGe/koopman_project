#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import json
from dataclasses import dataclass, asdict
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Cfg:
    franka_dataset: str = "/home/nng/koopman_project/data_wang_franka_reach_ablation/dataset_wang_franka_reach_ablation_v2.pt"
    ur_dataset: str = "/home/nng/koopman_project/data_wang_ur_reach_ablation/dataset_wang_ur_reach_ablation.pt"

    out_dir: str = "/home/nng/koopman_project/out_align_koopman_3"
    seed: int = 0

    total_steps_src: int = 50000
    batch_size: int = 256
    src_lr: float = 3e-4
    hidden_dim: int = 256
    n_layers: int = 3
    dyn_cons_update_freq: int = 2
    src_dyn_weight: float = 0.1

    lat_obs_dim: int = 8
    lat_act_dim: int = 8

    action_scale_mode: str = "robust_q995"
    franka_action_scale: float = 1.5
    ur_action_scale: float = 1.0
    action_clip_after_norm: float = 1.5

    total_steps_align: int = 50000
    align_lr: float = 5e-5
    lmbd_cyc: float = 1.0
    lmbd_dyn: float = 0.1
    lmbd_gp: float = 10.0
    disc_steps_per_gen: int = 1
    grad_clip_align: float = 10.0
    z_clip_for_disc: float = 5.0

    episode_len: int = 200
    koopman_ridge: float = 1e-4
    koopman_use_both_domains: bool = True
    koopman_encode_bs: int = 4096
    koopman_rollout_horizon: int = 50

    skip_stage1: bool = False
    skip_stage2: bool = False
    load_stage1_ckpt: str = ""
    load_stage2_ckpt: str = ""

    save_debug_json: bool = True


cfg = Cfg()


def set_seed(seed: int):
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


def action_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(pred, target)


@torch.no_grad()
def flatten_transitions(s: torch.Tensor, a: torch.Tensor, s_next: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_eps, T, ds = s.shape
    da = a.shape[-1]
    return s.reshape(n_eps * T, ds), a.reshape(n_eps * T, da), s_next.reshape(n_eps * T, ds)


def load_dataset(path: str) -> Dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu")
    return obj["data"] if isinstance(obj, dict) and "data" in obj else obj


@torch.no_grad()
def robust_action_scale(a: torch.Tensor, mode: str, fallback: float) -> float:
    a_abs = a.abs().reshape(-1)
    if mode == "fixed":
        scale = float(fallback)
    elif mode == "max_abs":
        scale = float(a_abs.max().item())
    elif mode == "robust_q995":
        scale = float(torch.quantile(a_abs, 0.995).item())
    else:
        raise ValueError(f"Unknown action_scale_mode: {mode}")
    return max(scale, 1e-6)


class TransitionBuffer:
    def __init__(self, s: torch.Tensor, a: torch.Tensor, s_next: torch.Tensor, device: torch.device):
        self.s = s.contiguous()
        self.a = a.contiguous()
        self.s_next = s_next.contiguous()
        self.N = self.s.shape[0]
        self.device = device

    def sample(self, batch_size: int):
        idx = torch.randint(0, self.N, (batch_size,), device=self.device)
        return self.s[idx], self.a[idx], self.s_next[idx]


class SrcAgent(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, lat_obs_dim: int, lat_act_dim: int, n_layers: int, hidden_dim: int):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.lat_obs_dim = lat_obs_dim
        self.lat_act_dim = lat_act_dim

        self.obs_enc = build_mlp(obs_dim, lat_obs_dim, n_layers, hidden_dim, activation="relu", out_act="identity")
        self.obs_dec = build_mlp(lat_obs_dim, obs_dim, n_layers, hidden_dim, activation="relu", out_act="identity")

        self.act_enc = build_mlp(obs_dim + act_dim, lat_act_dim, n_layers, hidden_dim, activation="relu", out_act="tanh")
        self.act_dec = build_mlp(obs_dim + lat_act_dim, act_dim, n_layers, hidden_dim, activation="relu", out_act="tanh")

        self.inv_dyn = build_mlp(lat_obs_dim * 2, lat_act_dim, n_layers, hidden_dim, activation="relu", out_act="tanh")
        self.fwd_dyn = build_mlp(lat_obs_dim + lat_act_dim, lat_obs_dim, n_layers, hidden_dim, activation="relu", out_act="identity")

        self.actor = build_mlp(lat_obs_dim, lat_act_dim, n_layers, hidden_dim, activation="relu", out_act="tanh")

    def bc_loss(self, s, a):
        z = self.obs_enc(s)
        u_hat = self.actor(z)
        a_hat = self.act_dec(torch.cat([s, u_hat], dim=-1))
        return action_loss(a_hat, a)

    def dyn_cons_loss(self, s, a, s_next):
        z = self.obs_enc(s)
        z_next = self.obs_enc(s_next)

        s_rec = self.obs_dec(z)
        loss_rec_s = F.mse_loss(s_rec, s)

        u = self.act_enc(torch.cat([s, a], dim=-1))
        a_rec = self.act_dec(torch.cat([s, u], dim=-1))
        loss_rec_a = action_loss(a_rec, a)

        u_inv = self.inv_dyn(torch.cat([z, z_next], dim=-1))
        loss_inv = F.mse_loss(u_inv, u.detach())

        z_pred = self.fwd_dyn(torch.cat([z, u], dim=-1))
        loss_fwd = F.mse_loss(z_pred, z_next.detach())

        loss = loss_rec_s + loss_rec_a + loss_inv + loss_fwd
        logs = {
            "rec_s": float(loss_rec_s.item()),
            "rec_a": float(loss_rec_a.item()),
            "inv": float(loss_inv.item()),
            "fwd": float(loss_fwd.item()),
        }
        return loss, logs


class TgtAgent(SrcAgent):
    pass


class ObsActAligner:
    def __init__(
        self,
        src: SrcAgent,
        tgt: TgtAgent,
        device: torch.device,
        n_layers: int,
        hidden_dim: int,
        lr: float,
        lmbd_gp: float,
        lmbd_cyc: float,
        lmbd_dyn: float,
        z_mean: torch.Tensor,
        z_std: torch.Tensor,
        z_clip_for_disc: float = 5.0,
        grad_clip_align: float = 10.0,
    ):
        self.src = src
        self.tgt = tgt
        self.device = device
        self.lmbd_gp = lmbd_gp
        self.lmbd_cyc = lmbd_cyc
        self.lmbd_dyn = lmbd_dyn
        self.z_clip_for_disc = z_clip_for_disc
        self.grad_clip_align = grad_clip_align

        self.z_mean = z_mean.detach()
        self.z_std = z_std.detach().clamp_min(1e-6)

        for p in self.src.parameters():
            p.requires_grad = False
        self.src.eval()

        self.lat_obs_disc = build_mlp(
            src.lat_obs_dim, 1, n_layers, hidden_dim, activation="leaky_relu", out_act="identity"
        ).to(device)
        self.lat_act_disc = build_mlp(
            src.lat_act_dim, 1, n_layers, hidden_dim, activation="leaky_relu", out_act="identity"
        ).to(device)

        self.lat_obs_disc_opt = torch.optim.Adam(self.lat_obs_disc.parameters(), lr=lr, betas=(0.5, 0.9))
        self.lat_act_disc_opt = torch.optim.Adam(self.lat_act_disc.parameters(), lr=lr, betas=(0.5, 0.9))

        self.z_t2s = build_mlp(
            src.lat_obs_dim, src.lat_obs_dim, n_layers, hidden_dim, activation="relu", out_act="tanh"
        ).to(device)
        self.z_s2t = build_mlp(
            src.lat_obs_dim, src.lat_obs_dim, n_layers, hidden_dim, activation="relu", out_act="identity"
        ).to(device)

        self.u_t2s = build_mlp(
            src.lat_act_dim, src.lat_act_dim, n_layers, hidden_dim, activation="relu", out_act="tanh"
        ).to(device)
        self.u_s2t = build_mlp(
            src.lat_act_dim, src.lat_act_dim, n_layers, hidden_dim, activation="relu", out_act="identity"
        ).to(device)

        self.gen_opt = torch.optim.Adam(
            list(self.tgt.obs_enc.parameters()) +
            list(self.tgt.obs_dec.parameters()) +
            list(self.tgt.act_enc.parameters()) +
            list(self.tgt.act_dec.parameters()) +
            list(self.z_t2s.parameters()) +
            list(self.z_s2t.parameters()) +
            list(self.u_t2s.parameters()) +
            list(self.u_s2t.parameters()),
            lr=lr,
        )

    def norm_z(self, z):
        return (z - self.z_mean) / self.z_std

    def denorm_z(self, z_n):
        return z_n * self.z_std + self.z_mean

    def _grad_penalty(self, disc: nn.Module, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        B = real.shape[0]
        eps = torch.rand(B, 1, device=self.device).expand_as(real)
        x = eps * real + (1.0 - eps) * fake
        x.requires_grad_(True)
        y = disc(x)
        grad = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=torch.ones_like(y),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return ((grad.norm(2, dim=1) - 1.0) ** 2).mean()

    def update_disc(self, src_s, src_a, tgt_s, tgt_a):
        self.lat_obs_disc_opt.zero_grad()
        self.lat_act_disc_opt.zero_grad()

        with torch.no_grad():
            real_z = self.src.obs_enc(src_s)
            real_z_n = self.norm_z(real_z).clamp(-self.z_clip_for_disc, self.z_clip_for_disc)

            real_u = self.src.act_enc(torch.cat([src_s, src_a], dim=-1))

            z_t = self.tgt.obs_enc(tgt_s)
            fake_z_n = self.z_t2s(z_t).clamp(-self.z_clip_for_disc, self.z_clip_for_disc)

            u_t = self.tgt.act_enc(torch.cat([tgt_s, tgt_a], dim=-1))
            fake_u = self.u_t2s(u_t).clamp(-1.5, 1.5)

        loss_obs = self.lat_obs_disc(fake_z_n).mean() - self.lat_obs_disc(real_z_n).mean()
        gp_obs = self._grad_penalty(self.lat_obs_disc, real_z_n, fake_z_n)
        loss_obs_total = loss_obs + self.lmbd_gp * gp_obs

        loss_act = self.lat_act_disc(fake_u).mean() - self.lat_act_disc(real_u).mean()
        gp_act = self._grad_penalty(self.lat_act_disc, real_u, fake_u)
        loss_act_total = loss_act + self.lmbd_gp * gp_act

        loss = loss_obs_total + loss_act_total
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.lat_obs_disc.parameters(), self.grad_clip_align)
        torch.nn.utils.clip_grad_norm_(self.lat_act_disc.parameters(), self.grad_clip_align)

        self.lat_obs_disc_opt.step()
        self.lat_act_disc_opt.step()

        return {
            "disc_obs": float(loss_obs.item()),
            "gp_obs": float(gp_obs.item()),
            "disc_act": float(loss_act.item()),
            "gp_act": float(gp_act.item()),
        }

    def update_gen(self, src_s, src_a, src_sn, tgt_s, tgt_a, tgt_sn):
        self.gen_opt.zero_grad()

        z_t = self.tgt.obs_enc(tgt_s)
        z_t_next = self.tgt.obs_enc(tgt_sn)
        u_t = self.tgt.act_enc(torch.cat([tgt_s, tgt_a], dim=-1))

        z_s_hat_n = self.z_t2s(z_t).clamp(-self.z_clip_for_disc, self.z_clip_for_disc)
        z_s_next_hat_n = self.z_t2s(z_t_next).clamp(-self.z_clip_for_disc, self.z_clip_for_disc)

        z_s_hat = self.denorm_z(z_s_hat_n)
        z_s_next_hat = self.denorm_z(z_s_next_hat_n)

        u_s_hat = self.u_t2s(u_t).clamp(-1.5, 1.5)

        adv_z = -self.lat_obs_disc(z_s_hat_n).mean()
        adv_u = -self.lat_act_disc(u_s_hat).mean()

        z_t_cyc = self.z_s2t(z_s_hat_n)
        u_t_cyc = self.u_s2t(u_s_hat)

        loss_cyc_z = F.l1_loss(z_t_cyc, z_t.detach())
        loss_cyc_u = F.l1_loss(u_t_cyc, u_t.detach())
        cycle_loss = loss_cyc_z + loss_cyc_u

        a_hat_tgt = self.tgt.act_dec(torch.cat([tgt_s, u_t], dim=-1))
        loss_ae_a = action_loss(a_hat_tgt, tgt_a)

        u_inv = self.src.inv_dyn(torch.cat([z_s_hat, z_s_next_hat], dim=-1))
        loss_inv = F.mse_loss(u_inv, u_s_hat.detach())

        z_pred = self.src.fwd_dyn(torch.cat([z_s_hat, u_s_hat], dim=-1))
        loss_fwd = F.mse_loss(z_pred, z_s_next_hat.detach())

        dyn_loss = loss_inv + loss_fwd

        loss = (
            (adv_z + adv_u)
            + self.lmbd_cyc * cycle_loss
            + self.lmbd_dyn * dyn_loss
            + loss_ae_a
        )
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            list(self.tgt.obs_enc.parameters()) +
            list(self.tgt.obs_dec.parameters()) +
            list(self.tgt.act_enc.parameters()) +
            list(self.tgt.act_dec.parameters()) +
            list(self.z_t2s.parameters()) +
            list(self.z_s2t.parameters()) +
            list(self.u_t2s.parameters()) +
            list(self.u_s2t.parameters()),
            max_norm=self.grad_clip_align,
        )

        self.gen_opt.step()

        return {
            "adv_z": float(adv_z.item()),
            "adv_u": float(adv_u.item()),
            "cyc": float(cycle_loss.item()),
            "ae_a": float(loss_ae_a.item()),
            "inv": float(loss_inv.item()),
            "fwd": float(loss_fwd.item()),
        }


@torch.no_grad()
def fit_koopman_ridge_cpu(z: torch.Tensor, u: torch.Tensor, z_next: torch.Tensor, ridge: float = 1e-4):
    assert z.device.type == "cpu" and u.device.type == "cpu" and z_next.device.type == "cpu"
    N, dz = z.shape
    du = u.shape[1]
    X = torch.cat([z, u], dim=1)
    Y = z_next
    XtX = X.T @ X
    reg = ridge * torch.eye(XtX.shape[0], dtype=X.dtype)
    XtY = X.T @ Y
    W = torch.linalg.solve(XtX + reg, XtY)
    A = W[:dz, :].T.contiguous()
    B = W[dz:, :].T.contiguous()
    pred = X @ W
    mse = torch.mean((pred - Y) ** 2).item()
    return A, B, mse


@torch.no_grad()
def encode_latent_batched_src(agent, s_all, a_all, sn_all, batch_size, device):
    agent.eval()
    z_list, u_list, zn_list = [], [], []
    N = s_all.shape[0]
    for i in range(0, N, batch_size):
        s = s_all[i:i+batch_size]
        a = a_all[i:i+batch_size]
        sn = sn_all[i:i+batch_size]
        z = agent.obs_enc(s)
        u = agent.act_enc(torch.cat([s, a], dim=-1))
        zn = agent.obs_enc(sn)
        z_list.append(z.detach().cpu())
        u_list.append(u.detach().cpu())
        zn_list.append(zn.detach().cpu())
    return torch.cat(z_list, dim=0), torch.cat(u_list, dim=0), torch.cat(zn_list, dim=0)


@torch.no_grad()
def encode_latent_batched_tgt_mapped(tgt_agent, aligner, s_all, a_all, sn_all, batch_size, device):
    tgt_agent.eval()
    aligner.z_t2s.eval()
    aligner.u_t2s.eval()
    z_list, u_list, zn_list = [], [], []
    N = s_all.shape[0]
    for i in range(0, N, batch_size):
        s = s_all[i:i+batch_size]
        a = a_all[i:i+batch_size]
        sn = sn_all[i:i+batch_size]
        z_t = tgt_agent.obs_enc(s)
        u_t = tgt_agent.act_enc(torch.cat([s, a], dim=-1))
        zn_t = tgt_agent.obs_enc(sn)

        z_n = aligner.z_t2s(z_t)
        zn_n = aligner.z_t2s(zn_t)
        z = aligner.denorm_z(z_n)
        zn = aligner.denorm_z(zn_n)
        u = aligner.u_t2s(u_t)

        z_list.append(z.detach().cpu())
        u_list.append(u.detach().cpu())
        zn_list.append(zn.detach().cpu())
    return torch.cat(z_list, dim=0), torch.cat(u_list, dim=0), torch.cat(zn_list, dim=0)


def save_stage2_bundle(tgt_dir: str, tgt: TgtAgent, aligner: ObsActAligner):
    os.makedirs(tgt_dir, exist_ok=True)
    torch.save(tgt.state_dict(), os.path.join(tgt_dir, "tgt_agent.pt"))
    bundle = {
        "tgt_agent": tgt.state_dict(),
        "lat_obs_disc": aligner.lat_obs_disc.state_dict(),
        "lat_act_disc": aligner.lat_act_disc.state_dict(),
        "map_z_t2s": aligner.z_t2s.state_dict(),
        "map_z_s2t": aligner.z_s2t.state_dict(),
        "map_u_t2s": aligner.u_t2s.state_dict(),
        "map_u_s2t": aligner.u_s2t.state_dict(),
        "z_mean": aligner.z_mean,
        "z_std": aligner.z_std,
    }
    torch.save(bundle, os.path.join(tgt_dir, "stage2_bundle.pt"))


def load_stage1(src: SrcAgent, src_ckpt: str, device):
    if not os.path.exists(src_ckpt):
        raise FileNotFoundError(f"Stage1 ckpt not found: {src_ckpt}")
    src.load_state_dict(torch.load(src_ckpt, map_location=device))


def main():
    os.makedirs(cfg.out_dir, exist_ok=True)
    set_seed(cfg.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    fr_data = load_dataset(cfg.franka_dataset)
    ur_data = load_dataset(cfg.ur_dataset)

    fr_s = fr_data["s"].float()
    fr_a = fr_data["a"].float()
    fr_sn = fr_data["s_next"].float()

    ur_s = ur_data["s"].float()
    ur_a = ur_data["a"].float()
    ur_sn = ur_data["s_next"].float()

    print(f"[INFO] Franka raw: s={tuple(fr_s.shape)}, a={tuple(fr_a.shape)}, s_next={tuple(fr_sn.shape)}")
    print(f"[INFO] UR raw:     s={tuple(ur_s.shape)}, a={tuple(ur_a.shape)}, s_next={tuple(ur_sn.shape)}")

    fr_s2, fr_a2_raw, fr_sn2 = flatten_transitions(fr_s, fr_a, fr_sn)
    ur_s2, ur_a2_raw, ur_sn2 = flatten_transitions(ur_s, ur_a, ur_sn)

    if cfg.action_scale_mode == "fixed":
        fr_scale = cfg.franka_action_scale
        ur_scale = cfg.ur_action_scale
    else:
        fr_scale = robust_action_scale(fr_a2_raw, cfg.action_scale_mode, cfg.franka_action_scale)
        ur_scale = robust_action_scale(ur_a2_raw, cfg.action_scale_mode, cfg.ur_action_scale)

    fr_a2 = (fr_a2_raw / fr_scale).clamp(-cfg.action_clip_after_norm, cfg.action_clip_after_norm)
    ur_a2 = (ur_a2_raw / ur_scale).clamp(-cfg.action_clip_after_norm, cfg.action_clip_after_norm)

    print(f"[INFO] Franka flat: N={fr_s2.shape[0]}, obs_dim={fr_s2.shape[1]}, act_dim={fr_a2.shape[1]}, action_scale={fr_scale:.6f}")
    print(f"[INFO] UR flat:     N={ur_s2.shape[0]}, obs_dim={ur_s2.shape[1]}, act_dim={ur_a2.shape[1]}, action_scale={ur_scale:.6f}")

    fr_buf = TransitionBuffer(fr_s2.to(device), fr_a2.to(device), fr_sn2.to(device), device=device)
    ur_buf = TransitionBuffer(ur_s2.to(device), ur_a2.to(device), ur_sn2.to(device), device=device)

    src = SrcAgent(
        obs_dim=fr_s2.shape[1],
        act_dim=fr_a2.shape[1],
        lat_obs_dim=cfg.lat_obs_dim,
        lat_act_dim=cfg.lat_act_dim,
        n_layers=cfg.n_layers,
        hidden_dim=cfg.hidden_dim,
    ).to(device)

    src_dir = os.path.join(cfg.out_dir, "src_franka")
    tgt_dir = os.path.join(cfg.out_dir, "tgt_ur")
    koop_dir = os.path.join(cfg.out_dir, "koopman_fit")
    os.makedirs(src_dir, exist_ok=True)
    os.makedirs(tgt_dir, exist_ok=True)
    os.makedirs(koop_dir, exist_ok=True)

    src_ckpt = cfg.load_stage1_ckpt if cfg.load_stage1_ckpt else os.path.join(src_dir, "src_agent.pt")

    if cfg.skip_stage1:
        print(f"[STAGE 1] skip training, loading src ckpt: {src_ckpt}")
        load_stage1(src, src_ckpt, device)
    else:
        print("[STAGE 1] Training source (Franka) BC + 0.1*dyn-cons ...")
        src_opt = torch.optim.Adam(src.parameters(), lr=cfg.src_lr)
        t0 = time.time()

        for step in range(1, cfg.total_steps_src + 1):
            s, a, sn = fr_buf.sample(cfg.batch_size)
            loss_bc = src.bc_loss(s, a)
            loss = loss_bc

            if step % cfg.dyn_cons_update_freq == 0:
                loss_dyn, _ = src.dyn_cons_loss(s, a, sn)
                loss = loss + cfg.src_dyn_weight * loss_dyn

            src_opt.zero_grad(set_to_none=True)
            loss.backward()
            src_opt.step()

            if step % 2000 == 0:
                with torch.no_grad():
                    s2, a2, sn2 = fr_buf.sample(cfg.batch_size)
                    _, dyn_logs = src.dyn_cons_loss(s2, a2, sn2)
                dt = time.time() - t0
                print(
                    f"[SRC] step={step}/{cfg.total_steps_src} "
                    f"loss={loss.item():.6f} (bc={loss_bc.item():.6f}) | "
                    f"rec_s={dyn_logs['rec_s']:.4f} rec_a={dyn_logs['rec_a']:.4f} "
                    f"inv={dyn_logs['inv']:.4f} fwd={dyn_logs['fwd']:.4f} dt={dt:.1f}s"
                )
                t0 = time.time()

        torch.save(src.state_dict(), src_ckpt)
        print(f"[STAGE 1 DONE] saved src -> {src_ckpt}")

    with torch.no_grad():
        n_stat = min(200000, fr_buf.s.shape[0])
        src_z_sample = src.obs_enc(fr_buf.s[:n_stat])
        z_mean = src_z_sample.mean(dim=0, keepdim=True)
        z_std = src_z_sample.std(dim=0, keepdim=True).clamp_min(1e-6)

    tgt = TgtAgent(
        obs_dim=ur_s2.shape[1],
        act_dim=ur_a2.shape[1],
        lat_obs_dim=cfg.lat_obs_dim,
        lat_act_dim=cfg.lat_act_dim,
        n_layers=cfg.n_layers,
        hidden_dim=cfg.hidden_dim,
    ).to(device)

    aligner = ObsActAligner(
        src=src,
        tgt=tgt,
        device=device,
        n_layers=cfg.n_layers,
        hidden_dim=cfg.hidden_dim,
        lr=cfg.align_lr,
        lmbd_gp=cfg.lmbd_gp,
        lmbd_cyc=cfg.lmbd_cyc,
        lmbd_dyn=cfg.lmbd_dyn,
        z_mean=z_mean,
        z_std=z_std,
        z_clip_for_disc=cfg.z_clip_for_disc,
        grad_clip_align=cfg.grad_clip_align,
    )

    if cfg.skip_stage2:
        print("[STAGE 2] skipped by config")
    else:
        print("[STAGE 2] Aligning target (UR) to source latent space ...")
        t0 = time.time()
        for step in range(1, cfg.total_steps_align + 1):
            for _ in range(cfg.disc_steps_per_gen):
                src_s, src_a, _ = fr_buf.sample(cfg.batch_size)
                tgt_s, tgt_a, _ = ur_buf.sample(cfg.batch_size)
                dlogs = aligner.update_disc(src_s, src_a, tgt_s, tgt_a)

            src_s, src_a, src_sn = fr_buf.sample(cfg.batch_size)
            tgt_s, tgt_a, tgt_sn = ur_buf.sample(cfg.batch_size)
            glogs = aligner.update_gen(src_s, src_a, src_sn, tgt_s, tgt_a, tgt_sn)

            if step % 2000 == 0:
                dt = time.time() - t0
                print(
                    f"[ALIGN] step={step}/{cfg.total_steps_align} "
                    f"d_obs={dlogs['disc_obs']:.3f} gp_obs={dlogs['gp_obs']:.3f} "
                    f"d_act={dlogs['disc_act']:.3f} gp_act={dlogs['gp_act']:.3f} | "
                    f"adv_z={glogs['adv_z']:.3f} adv_u={glogs['adv_u']:.3f} "
                    f"cyc={glogs['cyc']:.3f} ae_a={glogs['ae_a']:.3f} "
                    f"inv={glogs['inv']:.3f} fwd={glogs['fwd']:.3f} dt={dt:.1f}s"
                )
                t0 = time.time()

        save_stage2_bundle(tgt_dir, tgt, aligner)
        print(f"[STAGE 2 DONE] saved tgt+bundle -> {tgt_dir}")

    print("[STAGE 3] Building aligned latent transitions and fitting Koopman operator ...")

    with torch.no_grad():
        fr_z, fr_u, fr_zn = encode_latent_batched_src(
            src, fr_buf.s, fr_buf.a, fr_buf.s_next,
            batch_size=cfg.koopman_encode_bs, device=device,
        )
        ur_z, ur_u, ur_zn = encode_latent_batched_tgt_mapped(
            tgt, aligner, ur_buf.s, ur_buf.a, ur_buf.s_next,
            batch_size=cfg.koopman_encode_bs, device=device,
        )

    if cfg.koopman_use_both_domains:
        z_all = torch.cat([fr_z, ur_z], dim=0)
        u_all = torch.cat([fr_u, ur_u], dim=0)
        zn_all = torch.cat([fr_zn, ur_zn], dim=0)
        domain_note = "franka+ur(mapped)"
    else:
        z_all, u_all, zn_all = fr_z, fr_u, fr_zn
        domain_note = "franka_only"

    A_cpu, B_cpu, train_mse = fit_koopman_ridge_cpu(z_all, u_all, zn_all, ridge=cfg.koopman_ridge)
    print(f"[STAGE 3] train 1-step MSE = {train_mse:.6f} | A={tuple(A_cpu.shape)} B={tuple(B_cpu.shape)}")

    A = A_cpu.to(device)
    B = B_cpu.to(device)

    @torch.no_grad()
    def reshape_to_episodes(z_flat: torch.Tensor, u_flat: torch.Tensor, T: int):
        z_flat = z_flat.to(device)
        u_flat = u_flat.to(device)
        N = z_flat.shape[0]
        assert N % T == 0
        E = N // T
        return z_flat.view(E, T, -1), u_flat.view(E, T, -1)

    @torch.no_grad()
    def eval_multi_step_rollout(z_seq, u_seq, A, B, horizon=50, name=""):
        E, T, dz = z_seq.shape
        H = min(horizon, T - 1)
        idx = torch.randperm(E, device=z_seq.device)[: min(E, 64)]
        z = z_seq[idx, 0]
        gt = z_seq[idx, 1:H+1]
        u = u_seq[idx, :H]
        preds = []
        for t in range(H):
            z = (A @ z.unsqueeze(-1)).squeeze(-1) + (B @ u[:, t].unsqueeze(-1)).squeeze(-1)
            preds.append(z)
        pred = torch.stack(preds, dim=1)
        mse = torch.mean((pred - gt) ** 2).item()
        print(f"[EVAL] {name} {H}-step rollout MSE = {mse:.6f}")

    fr_z_seq, fr_u_seq = reshape_to_episodes(fr_z, fr_u, cfg.episode_len)
    ur_z_seq, ur_u_seq = reshape_to_episodes(ur_z, ur_u, cfg.episode_len)
    eval_multi_step_rollout(fr_z_seq, fr_u_seq, A, B, horizon=cfg.koopman_rollout_horizon, name="franka")
    eval_multi_step_rollout(ur_z_seq, ur_u_seq, A, B, horizon=cfg.koopman_rollout_horizon, name="ur(mapped)")

    save_path = os.path.join(koop_dir, "aligned_latent_and_koopman.pt")
    payload = {
        "meta": {
            "note": "Stage3 aligned latent transitions + Koopman operator",
            "episode_len": cfg.episode_len,
            "ridge": cfg.koopman_ridge,
            "domain_note": domain_note,
            "src_obs_dim": fr_buf.s.shape[1],
            "src_act_dim": fr_buf.a.shape[1],
            "tgt_obs_dim": ur_buf.s.shape[1],
            "tgt_act_dim": ur_buf.a.shape[1],
            "franka_action_scale": fr_scale,
            "ur_action_scale": ur_scale,
            "dz": int(fr_z.shape[1]),
            "du": int(fr_u.shape[1]),
            "train_1step_mse": float(train_mse),
        },
        "data": {
            "fr_z": fr_z,
            "fr_u": fr_u,
            "fr_zn": fr_zn,
            "ur_z": ur_z,
            "ur_u": ur_u,
            "ur_zn": ur_zn,
            "A": A_cpu,
            "B": B_cpu,
        }
    }
    torch.save(payload, save_path)

    if cfg.save_debug_json:
        debug = {
            "cfg": asdict(cfg),
            "franka_action_scale": float(fr_scale),
            "ur_action_scale": float(ur_scale),
            "franka_raw_action_abs_q995": float(torch.quantile(fr_a2_raw.abs().reshape(-1).cpu(), 0.995).item()),
            "ur_raw_action_abs_q995": float(torch.quantile(ur_a2_raw.abs().reshape(-1).cpu(), 0.995).item()),
            "train_koopman_mse": float(train_mse),
            "z_mean": z_mean.detach().cpu().tolist(),
            "z_std": z_std.detach().cpu().tolist(),
        }
        with open(os.path.join(cfg.out_dir, "debug_train_meta.json"), "w", encoding="utf-8") as f:
            json.dump(debug, f, indent=2)

    print(f"[STAGE 3 DONE] saved -> {save_path}")
    print(f"[STAGE 3 DONE] latent shapes: fr_z={tuple(fr_z.shape)} ur_z={tuple(ur_z.shape)} A={tuple(A_cpu.shape)} B={tuple(B_cpu.shape)}")


if __name__ == "__main__":
    main()
