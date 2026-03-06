import os
import math
import time
from dataclasses import dataclass
from typing import Tuple, List

import isaacgym
from isaacgym import gymapi, gymutil, gymtorch
import torch


# ===================== 配置（直接改这里） =====================
@dataclass
class Cfg:
    # UR5 资产（你当前的位置）
    asset_root: str = "/home/nng/software/isaacgym_preview4/isaacgym/assets/MWE_UR5_IK/assets"
    asset_file: str = "ur5.urdf"
    asset_name_in_sim: str = "ur5"       # acquire_jacobian_tensor 需要用 actor 名

    # 数据规模：32 env * 310 eps = 9920 ~ 1w
    num_envs: int = 32
    episodes_per_env: int = 310
    episode_len: int = 200

    # 仿真
    dt: float = 1.0 / 60.0
    substeps: int = 2
    sim_device_id: int = 0
    graphics_device_id: int = 0
    use_gpu_physx: bool = True
    use_gpu_pipeline: bool = True

    # Wang-style：每个 episode 的目标点采样 box
    box_min: Tuple[float, float, float] = (-0.2, -0.25, 0.8)
    box_max: Tuple[float, float, float] = ( 0.2,  0.25, 1.2)
    init_target: Tuple[float, float, float] = (-0.2, 0.0, 1.05)

    # 末端 link 与 tip offset（UR5_IK demo 用 0.18m）
    ee_link_name: str = "tool0"
    tip_offset_z: float = 0.18  # 设 0.0 就是 tool0 位置

    # 控制（任务空间“直线”）参数：保守稳定
    kp_pos: float = 6.0            # 位置误差比例
    max_ee_step: float = 0.02      # 每步 EE 最大位移（m）
    ik_damping: float = 0.05       # DLS 阻尼
    max_qd: float = 1.0            # 关节速度限幅（rad/s）
    smooth_alpha: float = 0.90     # qdot 平滑 (越大越平滑)

    # 把 qdot_cmd 转成 position target 的伺服（稳定+兼容 GPU pipeline）
    max_dq_per_step: float = 0.08  # 每步关节最大位移（rad）

    # Viewer
    render: bool = False
    render_every: int = 2

    # 保存
    out_dir: str = "/home/nng/koopman_project/data_wang_ur_reach_ablation"
    out_file: str = "dataset_wang_ur_reach_ablation.pt"


cfg = Cfg()
# ============================================================


# ----------------- quaternion utils (pure torch) -----------------
def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    out = q.clone()
    out[..., 0:3] = -out[..., 0:3]
    return out

def quat_mul(q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    # xyzw
    x1, y1, z1, w1 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    x2, y2, z2, w2 = r[..., 0], r[..., 1], r[..., 2], r[..., 3]
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return torch.stack([x, y, z, w], dim=-1)

def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # rotate v by q, xyzw
    q_xyz = q[..., 0:3]
    qw = q[..., 3:4]
    t = 2.0 * torch.cross(q_xyz, v, dim=-1)
    return v + qw * t + torch.cross(q_xyz, t, dim=-1)

# ----------------- IK DLS -----------------
def control_ik_dls(j_eef: torch.Tensor, dpose: torch.Tensor, damping: float) -> torch.Tensor:
    """
    j_eef: (N,6,DOF)
    dpose: (N,6,1)
    return dq: (N,DOF)
    """
    jT = j_eef.transpose(1, 2)  # (N,DOF,6)
    eye6 = torch.eye(6, device=j_eef.device, dtype=j_eef.dtype).unsqueeze(0)
    lmbda = eye6 * (damping ** 2)
    inv = torch.linalg.inv(j_eef @ jT + lmbda)
    dq = (jT @ inv @ dpose).squeeze(-1)
    return dq

# ----------------- state -----------------
def build_state_sincos(q_arm: torch.Tensor) -> torch.Tensor:
    # q_arm: (N,6)
    return torch.cat([torch.sin(q_arm), torch.cos(q_arm)], dim=-1)  # (N,12)


def create_sim():
    gym = gymapi.acquire_gym()
    _ = gymutil.parse_arguments(description="Collect UR5 dataset (Wang-style straight-to-random targets)")

    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0, 0, -9.81)
    sim_params.dt = cfg.dt
    sim_params.substeps = cfg.substeps

    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 8
    sim_params.physx.num_velocity_iterations = 2
    sim_params.physx.use_gpu = cfg.use_gpu_physx
    sim_params.use_gpu_pipeline = cfg.use_gpu_pipeline

    sim = gym.create_sim(cfg.sim_device_id, cfg.graphics_device_id, gymapi.SIM_PHYSX, sim_params)
    if sim is None:
        raise RuntimeError("create_sim 失败：检查 IsaacGym/CUDA/驱动")

    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane_params)

    viewer = None
    if cfg.render:
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        if viewer is None:
            raise RuntimeError("create_viewer 失败：检查图形环境")
        cam_pos = gymapi.Vec3(2.5, 2.0, 2.0)
        cam_tgt = gymapi.Vec3(0.0, 0.0, 1.0)
        gym.viewer_camera_look_at(viewer, None, cam_pos, cam_tgt)
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_V, "toggle_render")

    return gym, sim, viewer


def load_ur5_asset(gym, sim):
    opt = gymapi.AssetOptions()
    opt.fix_base_link = True
    opt.flip_visual_attachments = True
    opt.armature = 0.01
    opt.disable_gravity = False
    opt.default_dof_drive_mode = int(gymapi.DOF_MODE_POS)

    asset = gym.load_asset(sim, cfg.asset_root, cfg.asset_file, opt)
    if asset is None:
        raise RuntimeError(f"load_asset 失败：root={cfg.asset_root}, file={cfg.asset_file}")

    dof_count = gym.get_asset_dof_count(asset)
    dof_names = gym.get_asset_dof_names(asset)
    rb_dict = gym.get_asset_rigid_body_dict(asset)

    print(f"[INFO] UR5 dof_count={dof_count}")
    print(f"[INFO] dof_names={dof_names}")
    print(f"[INFO] RigidBody keys sample: {list(rb_dict.keys())[:10]}")
    if cfg.ee_link_name not in rb_dict:
        raise RuntimeError(f"找不到 ee_link_name='{cfg.ee_link_name}'，请检查 rb_dict keys")

    return asset, dof_count, dof_names, rb_dict


def infer_ur5_arm_dof_indices(dof_names: List[str]) -> List[int]:
    """
    解决 7 DOF vs 6 DOF：从 dof_names 中找出 UR5 的 6 个转动关节。
    常见名字：shoulder_pan_joint, shoulder_lift_joint, elbow_joint, wrist_1_joint, wrist_2_joint, wrist_3_joint
    """
    canonical = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]
    name_to_idx = {n: i for i, n in enumerate(dof_names)}
    if all(n in name_to_idx for n in canonical):
        return [name_to_idx[n] for n in canonical]

    # fallback：取前 6 个（很多 urdf 也是这样排的）
    print("[WARN] 未找到 canonical UR5 joint names，fallback 使用前 6 个 DOF")
    return list(range(min(6, len(dof_names))))


def create_envs(gym, sim, asset):
    envs, actors = [], []
    env_lower = gymapi.Vec3(-1.2, -1.2, 0.0)
    env_upper = gymapi.Vec3( 1.2,  1.2, 1.6)
    num_per_row = int(math.sqrt(cfg.num_envs))

    # UR5 base pose（参考 UR5_IK.py）
    ur_pose = gymapi.Transform()
    ur_pose.p = gymapi.Vec3(0.0, 0.0, 1.453)
    ur_pose.r = gymapi.Quat(0.0, 0.707107, 0.0, 0.707107)

    for i in range(cfg.num_envs):
        env = gym.create_env(sim, env_lower, env_upper, num_per_row)
        actor = gym.create_actor(env, asset, ur_pose, cfg.asset_name_in_sim, i, 1)

        dof_props = gym.get_actor_dof_properties(env, actor)
        dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
        dof_props["stiffness"].fill(400.0)
        dof_props["damping"].fill(40.0)
        gym.set_actor_dof_properties(env, actor, dof_props)

        envs.append(env)
        actors.append(actor)

    return envs, actors


def main():
    os.makedirs(cfg.out_dir, exist_ok=True)
    out_path = os.path.join(cfg.out_dir, cfg.out_file)

    gym, sim, viewer = create_sim()
    asset, dof_count, dof_names, rb_dict = load_ur5_asset(gym, sim)

    arm_dof_ids = infer_ur5_arm_dof_indices(dof_names)  # length 6
    arm_dofs = len(arm_dof_ids)
    assert arm_dofs == 6, f"UR5 arm_dofs 应为6，但得到 {arm_dofs}, ids={arm_dof_ids}"

    envs, actors = create_envs(gym, sim, asset)

    # tensor API
    gym.prepare_sim(sim)

    # tensors
    dof_state_tensor = gym.acquire_dof_state_tensor(sim)
    dof_states = gymtorch.wrap_tensor(dof_state_tensor)

    rb_state_tensor = gym.acquire_rigid_body_state_tensor(sim)
    rb_states = gymtorch.wrap_tensor(rb_state_tensor)

    jac_tensor = gym.acquire_jacobian_tensor(sim, cfg.asset_name_in_sim)
    jac = gymtorch.wrap_tensor(jac_tensor)

    # refresh
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_jacobian_tensors(sim)

    device = dof_states.device
    print(f"[INFO] tensor device={device}, gpu_pipeline={cfg.use_gpu_pipeline}")

    dofs_per_env = dof_states.shape[0] // cfg.num_envs
    num_rb = rb_states.shape[0] // cfg.num_envs

    # ---- Jacobian body count inference + reshape (稳定，不硬写 num_bodies) ----
    num_bodies_jac = jac.numel() // (cfg.num_envs * 6 * dofs_per_env)
    dof_view = dof_states.view(cfg.num_envs, dofs_per_env, 2)
    rb_view = rb_states.view(cfg.num_envs, num_rb, 13)
    jac_view = jac.view(cfg.num_envs, num_bodies_jac, 6, dofs_per_env)

    # ---- EE index：必须用 DOMAIN_ENV（局部 index）----
    ee_env_index = int(gym.find_actor_rigid_body_index(envs[0], actors[0], cfg.ee_link_name, gymapi.DOMAIN_ENV))
    if not (0 <= ee_env_index < num_rb):
        raise RuntimeError(f"ee_env_index 非法: {ee_env_index}, num_rb={num_rb}")

    ee_rb_index_asset = rb_dict[cfg.ee_link_name]

    # ---- Jacobian body index 偏移：根据 num_bodies_jac 自动判定 ----
    ee_jac_index = ee_rb_index_asset
    if num_bodies_jac == num_rb - 1:
        ee_jac_index = ee_rb_index_asset - 1
    if not (0 <= ee_jac_index < num_bodies_jac):
        raise RuntimeError(f"ee_jac_index 非法: {ee_jac_index}, bodies_jac={num_bodies_jac}, ee_rb_asset={ee_rb_index_asset}")

    print(f"[INFO] ee_link='{cfg.ee_link_name}', ee_env_index={ee_env_index}, ee_rb_asset={ee_rb_index_asset}, ee_jac_index={ee_jac_index}")
    print(f"[INFO] dofs_per_env={dofs_per_env}, num_rb={num_rb}, num_bodies_jac={num_bodies_jac}, arm_dof_ids={arm_dof_ids}")

    # ---- dataset buffers (CPU, 单文件) ----
    total_eps = cfg.num_envs * cfg.episodes_per_env
    T = cfg.episode_len
    state_dim = 2 * arm_dofs   # 12
    action_dim = arm_dofs      # 6

    S = torch.zeros((total_eps, T, state_dim), dtype=torch.float32)
    A = torch.zeros((total_eps, T, action_dim), dtype=torch.float32)
    SN = torch.zeros((total_eps, T, state_dim), dtype=torch.float32)
    TARGET = torch.zeros((total_eps, 3), dtype=torch.float32)
    ENV_ID = torch.zeros((total_eps,), dtype=torch.long)
    EP_ID = torch.zeros((total_eps,), dtype=torch.long)

    # counters on device
    ep_step = torch.zeros(cfg.num_envs, dtype=torch.long, device=device)
    ep_count = torch.zeros(cfg.num_envs, dtype=torch.long, device=device)
    global_ep_base = torch.arange(cfg.num_envs, device=device, dtype=torch.long) * cfg.episodes_per_env

    # targets on device
    target_pos = torch.zeros((cfg.num_envs, 3), device=device, dtype=torch.float32)
    box_min = torch.tensor(cfg.box_min, device=device, dtype=torch.float32)
    box_max = torch.tensor(cfg.box_max, device=device, dtype=torch.float32)
    init_target = torch.tensor(cfg.init_target, device=device, dtype=torch.float32)

    # tip offset
    tip_offset = torch.tensor([0.0, 0.0, cfg.tip_offset_z], device=device, dtype=torch.float32).repeat(cfg.num_envs, 1)

    # action smoothing qdot_cmd (only for 6 arm dofs)
    qd_cmd = torch.zeros((cfg.num_envs, arm_dofs), device=device, dtype=torch.float32)

    # position target tensor: full dofs_per_env
    pos_tgt = torch.zeros((cfg.num_envs, dofs_per_env), device=device, dtype=torch.float32)
    pos_tgt_flat = pos_tgt.view(-1)

    render_on = cfg.render
    sim_step = 0
    last_print = time.time()

    print(f"[INFO] Collecting UR5 dataset -> {out_path}")
    print(f"[INFO] total_eps={total_eps}, T={T}, state_dim={state_dim}, action_dim={action_dim}")

    while True:
        if torch.all(ep_count >= cfg.episodes_per_env):
            break

        # viewer
        if viewer is not None:
            if gym.query_viewer_has_closed(viewer):
                print("[WARN] viewer closed, exit early")
                break
            for evt in gym.query_viewer_action_events(viewer):
                if evt.action == "toggle_render" and evt.value > 0:
                    render_on = not render_on
                    print(f"[INFO] render_on={render_on}")

        # refresh tensors
        gym.refresh_dof_state_tensor(sim)
        gym.refresh_rigid_body_state_tensor(sim)
        gym.refresh_jacobian_tensors(sim)

        # ---- current q for arm (N,6) ----
        q_all = dof_view[:, :, 0]  # (N, dofs_per_env)
        q_arm = q_all[:, arm_dof_ids]  # (N,6)

        # ---- EE pose from rb_view (local index) ----
        hand_pos = rb_view[:, ee_env_index, 0:3]
        hand_rot = rb_view[:, ee_env_index, 3:7]
        curr_pos = hand_pos + quat_apply(hand_rot, tip_offset)

        # ---- new episode: sample target (Wang-style) ----
        need_new = (ep_step == 0) & (ep_count < cfg.episodes_per_env)
        if torch.any(need_new):
            ids = torch.nonzero(need_new, as_tuple=False).squeeze(-1)
            ep_local = ep_count[ids]
            ep_global = global_ep_base[ids] + ep_local

            first_mask = (ep_local == 0)
            if torch.any(first_mask):
                first_ids = ids[first_mask]
                target_pos[first_ids] = init_target

            other_mask = ~first_mask
            if torch.any(other_mask):
                other_ids = ids[other_mask]
                u = torch.rand((other_ids.numel(), 3), device=device)
                tp = box_min + u * (box_max - box_min)
                target_pos[other_ids] = tp

            # write meta to CPU
            for k in range(ids.numel()):
                env_i = int(ids[k].item())
                g = int((global_ep_base[env_i] + ep_count[env_i]).item())
                TARGET[g] = target_pos[env_i].detach().cpu()
                ENV_ID[g] = env_i
                EP_ID[g] = g

        # ---- task-space straight motion: clamp step ----
        ds = target_pos - curr_pos
        dx = torch.clamp(cfg.kp_pos * ds, -cfg.max_ee_step, cfg.max_ee_step)  # (N,3)
        dpose = torch.cat([dx, torch.zeros_like(dx)], dim=-1).unsqueeze(-1)   # (N,6,1), orientation term = 0

        # ---- Jacobian for EE: pick correct body index, and only arm joints columns ----
        J_full = jac_view[:, ee_jac_index, :, :]           # (N,6,dofs_per_env)
        J_arm = J_full[:, :, arm_dof_ids]                  # (N,6,6)

        dq = control_ik_dls(J_arm, dpose, damping=cfg.ik_damping)  # (N,6)
        qd_raw = torch.clamp(dq / cfg.dt, -cfg.max_qd, cfg.max_qd)
        qd_cmd = cfg.smooth_alpha * qd_cmd + (1.0 - cfg.smooth_alpha) * qd_raw

        # ---- state s_t (N,12) ----
        s_t = build_state_sincos(q_arm)

        # ---- apply control via position targets (stable) ----
        dq_step = torch.clamp(qd_cmd * cfg.dt, -cfg.max_dq_per_step, cfg.max_dq_per_step)  # (N,6)
        q_tgt_arm = q_arm + dq_step

        pos_tgt.zero_()
        pos_tgt[:, arm_dof_ids] = q_tgt_arm
        gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos_tgt_flat))

        # ---- simulate ----
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        # render
        if viewer is not None and render_on and (sim_step % cfg.render_every == 0):
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, True)
        elif viewer is not None:
            gym.poll_viewer_events(viewer)

        # ---- next state ----
        gym.refresh_dof_state_tensor(sim)
        q_all_next = dof_view[:, :, 0]
        q_arm_next = q_all_next[:, arm_dof_ids]
        s_next = build_state_sincos(q_arm_next)

        # ---- record ----
        active = (ep_count < cfg.episodes_per_env)
        if torch.any(active):
            ids = torch.nonzero(active, as_tuple=False).squeeze(-1)
            for k in range(ids.numel()):
                env_i = int(ids[k].item())
                t = int(ep_step[env_i].item())
                g = int((global_ep_base[env_i] + ep_count[env_i]).item())
                if t < T:
                    S[g, t] = s_t[env_i].detach().cpu()
                    A[g, t] = qd_cmd[env_i].detach().cpu()
                    SN[g, t] = s_next[env_i].detach().cpu()

        # step update
        ep_step += 1

        done = (ep_step >= cfg.episode_len) & (ep_count < cfg.episodes_per_env)
        if torch.any(done):
            ep_count[done] += 1
            ep_step[done] = 0

        sim_step += 1
        if time.time() - last_print > 2.0:
            total_done = int(torch.sum(ep_count).item())
            total_all = cfg.num_envs * cfg.episodes_per_env
            pct = 100.0 * total_done / float(total_all)
            print(f"[PROGRESS] episodes_done={total_done}/{total_all} ({pct:.2f}%) | render_on={render_on}")
            last_print = time.time()

    payload = {
        "meta": {
            "robot": "UR5",
            "asset_root": cfg.asset_root,
            "asset_file": cfg.asset_file,
            "num_envs": cfg.num_envs,
            "episodes_per_env": cfg.episodes_per_env,
            "episode_len": cfg.episode_len,
            "dt": cfg.dt,
            "state_def": "s=[sin(q1..q6), cos(q1..q6)] (12-dim, UR5 6 joints)",
            "action_def": "a=qdot_cmd(6-dim) joint velocity command (saved), applied via position-target per step",
            "arm_dof_ids": arm_dof_ids,
            "ee_link_name": cfg.ee_link_name,
            "tip_offset_z": cfg.tip_offset_z,
            "ik_damping": cfg.ik_damping,
            "max_qd": cfg.max_qd,
            "target_strategy": "Wang-style straight-to-random target in box, no reset to init",
            "box_min": cfg.box_min,
            "box_max": cfg.box_max,
            "init_target": cfg.init_target,
            "jacobian_body_count": num_bodies_jac,
            "rigid_body_count": num_rb,
            "dofs_per_env": dofs_per_env,
            "ee_env_index": ee_env_index,
            "ee_rb_index_asset": ee_rb_index_asset,
            "ee_jac_index": ee_jac_index,
        },
        "data": {
            "s": S,
            "a": A,
            "s_next": SN,
            "target_pos": TARGET,
            "env_id": ENV_ID,
            "episode_id": EP_ID,
        }
    }
    torch.save(payload, out_path)
    print(f"[DONE] saved: {out_path}")
    print(f"[DONE] shapes: s={tuple(S.shape)}, a={tuple(A.shape)}, s_next={tuple(SN.shape)}, target_pos={tuple(TARGET.shape)}")

    if viewer is not None:
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


if __name__ == "__main__":
    main()