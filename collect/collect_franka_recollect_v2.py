import os
import math
import time
from dataclasses import dataclass

import isaacgym
from isaacgym import gymapi, gymutil, gymtorch

import torch


# =========================================================
# 配置
# =========================================================
@dataclass
class Cfg:
    # Isaac Gym asset
    asset_root: str = "/home/nng/software/isaacgym_preview4/isaacgym/assets"
    asset_file: str = "urdf/franka_description/robots/franka_panda.urdf"

    # 输出
    out_path: str = "/home/nng/koopman_project/data_wang_franka_reach_ablation/dataset_wang_franka_reach_ablation_v2.pt"

    # 数据规模
    num_envs: int = 32
    episodes_per_env: int = 320  # 32 * 320 = 10240 episodes
    episode_len: int = 200

    # 仿真
    dt: float = 1.0 / 60.0
    substeps: int = 2
    sim_device_id: int = 0
    graphics_device_id: int = 0
    use_gpu_pipeline: bool = True

    # 渲染
    render: bool = False
    render_every: int = 2

    # 任务定义：box 与 UR 保持一致
    target_box_low = (-0.2, -0.25, 0.8)
    target_box_high = (0.2, 0.25, 1.2)

    # Franka arm
    arm_dofs: int = 7
    total_dofs: int = 9

    # 控制
    ik_damping: float = 0.05
    ee_gain: float = 4.0              # 子目标位置误差 -> dpose 的比例
    max_qd: float = 1.5               # 关节速度上限
    vel_smooth: float = 0.85          # 动作平滑，越大越平滑
    null_kp: float = 0.25             # nullspace posture 拉回随机 posture bias 的强度

    # posture bias（扩展关节覆盖）
    # 在 home pose 周围采样一个随机姿态偏置，促使不同 episode 使用不同冗余解分支
    posture_noise = (0.35, 0.45, 0.35, 0.55, 0.35, 0.45, 0.35)

    # 初始 home pose（Franka 常用）
    # 这里只控制 7 个 arm joint；两个手指固定张开
    home_q = (0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785)
    finger_open = (0.04, 0.04)

    # 收集前预热
    warmup_steps: int = 30


cfg = Cfg()


# =========================================================
# 工具函数
# =========================================================
def build_mlp():
    pass


def control_ik(j_eef, dpose, damping=0.05):
    """
    Damped Least Squares IK
    j_eef: (N, 6, 7)
    dpose: (N, 6, 1)
    return: dq (N, 7)
    """
    j_eef_T = torch.transpose(j_eef, 1, 2)  # (N, 7, 6)
    lmbda = torch.eye(6, device=j_eef.device).unsqueeze(0) * (damping ** 2)
    inv = torch.inverse(j_eef @ j_eef_T + lmbda)
    u = (j_eef_T @ inv @ dpose).squeeze(-1)
    return u


def build_state_sincos(q_arm: torch.Tensor) -> torch.Tensor:
    # q_arm: (N,7)
    return torch.cat([torch.sin(q_arm), torch.cos(q_arm)], dim=-1)


def sample_target_pos(n: int, device: torch.device) -> torch.Tensor:
    low = torch.tensor(cfg.target_box_low, device=device, dtype=torch.float32)
    high = torch.tensor(cfg.target_box_high, device=device, dtype=torch.float32)
    return low.unsqueeze(0) + torch.rand((n, 3), device=device) * (high - low).unsqueeze(0)


def clamp_q(q: torch.Tensor, q_lo: torch.Tensor, q_hi: torch.Tensor) -> torch.Tensor:
    return torch.max(torch.min(q, q_hi), q_lo)


def sample_posture_bias(home_q: torch.Tensor, q_lo: torch.Tensor, q_hi: torch.Tensor, n: int, device: torch.device):
    noise_scale = torch.tensor(cfg.posture_noise, device=device, dtype=torch.float32).unsqueeze(0)
    noise = (torch.rand((n, cfg.arm_dofs), device=device) * 2.0 - 1.0) * noise_scale
    q_bias = home_q.unsqueeze(0) + noise
    q_bias = clamp_q(q_bias, q_lo.unsqueeze(0), q_hi.unsqueeze(0))
    return q_bias


# =========================================================
# 创建仿真
# =========================================================
def create_sim():
    gym = gymapi.acquire_gym()
    _ = gymutil.parse_arguments(description="Franka recollect v2")

    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.dt = cfg.dt
    sim_params.substeps = cfg.substeps
    sim_params.use_gpu_pipeline = cfg.use_gpu_pipeline

    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 8
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.use_gpu = True
    sim_params.physx.contact_offset = 0.001
    sim_params.physx.rest_offset = 0.0

    sim = gym.create_sim(
        cfg.sim_device_id,
        cfg.graphics_device_id,
        gymapi.SIM_PHYSX,
        sim_params,
    )
    if sim is None:
        raise RuntimeError("create_sim 失败")

    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane_params)

    viewer = None
    if cfg.render:
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        if viewer is None:
            raise RuntimeError("create_viewer 失败")
        cam_pos = gymapi.Vec3(2.2, 2.0, 1.8)
        cam_tgt = gymapi.Vec3(0.0, 0.0, 0.9)
        gym.viewer_camera_look_at(viewer, None, cam_pos, cam_tgt)

    return gym, sim, viewer


# =========================================================
# 加载 Franka
# =========================================================
def load_franka_asset(gym, sim):
    opt = gymapi.AssetOptions()
    opt.fix_base_link = True
    opt.disable_gravity = False
    opt.flip_visual_attachments = True
    opt.armature = 0.01
    opt.default_dof_drive_mode = int(gymapi.DOF_MODE_POS)

    asset = gym.load_asset(sim, cfg.asset_root, cfg.asset_file, opt)
    if asset is None:
        raise RuntimeError("load_asset 失败")

    dof_count = gym.get_asset_dof_count(asset)
    rb_count = gym.get_asset_rigid_body_count(asset)
    rb_dict = gym.get_asset_rigid_body_dict(asset)

    print(f"[INFO] Franka dof_count={dof_count}, rigid_body_count={rb_count}")
    print(f"[INFO] rigid body keys={list(rb_dict.keys())}")

    return asset, dof_count, rb_count, rb_dict


# =========================================================
# 创建环境
# =========================================================
def create_envs(gym, sim, asset):
    envs = []
    actors = []

    spacing = 1.6
    num_per_row = int(math.sqrt(cfg.num_envs))
    env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
    env_upper = gymapi.Vec3(spacing, spacing, spacing)

    for i in range(cfg.num_envs):
        env = gym.create_env(sim, env_lower, env_upper, num_per_row)

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

        actor = gym.create_actor(env, asset, pose, "franka", i, 1)
        envs.append(env)
        actors.append(actor)

        dof_props = gym.get_actor_dof_properties(env, actor)

        # 位置伺服控制：用 q_target = q + qdot_cmd*dt
        dof_props["driveMode"].fill(int(gymapi.DOF_MODE_POS))
        dof_props["stiffness"][:cfg.arm_dofs].fill(700.0)
        dof_props["damping"][:cfg.arm_dofs].fill(50.0)

        # 手指更稳一点
        dof_props["stiffness"][cfg.arm_dofs:].fill(200.0)
        dof_props["damping"][cfg.arm_dofs:].fill(20.0)

        gym.set_actor_dof_properties(env, actor, dof_props)

    return envs, actors


# =========================================================
# 主函数
# =========================================================
def main():
    os.makedirs(os.path.dirname(cfg.out_path), exist_ok=True)

    gym, sim, viewer = create_sim()
    asset, dof_count, rb_count, rb_dict = load_franka_asset(gym, sim)
    envs, actors = create_envs(gym, sim, asset)

    # hand rigid body index
    if "panda_hand" not in rb_dict:
        raise RuntimeError("asset rigid body dict 中找不到 panda_hand")
    hand_rb_asset_idx = rb_dict["panda_hand"]

    gym.prepare_sim(sim)

    device = torch.device(f"cuda:{cfg.sim_device_id}")

    # tensors
    _dof_state = gym.acquire_dof_state_tensor(sim)
    dof_state = gymtorch.wrap_tensor(_dof_state)              # (num_envs*9, 2)

    _rb_state = gym.acquire_rigid_body_state_tensor(sim)
    rb_state = gymtorch.wrap_tensor(_rb_state)                # (num_envs*rb_per_env, 13)

    _jac = gym.acquire_jacobian_tensor(sim, "franka")
    jac = gymtorch.wrap_tensor(_jac)

    gym.refresh_dof_state_tensor(sim)
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_jacobian_tensors(sim)

    dof_state = dof_state.view(cfg.num_envs, cfg.total_dofs, 2)
    rb_per_env = rb_state.shape[0] // cfg.num_envs
    rb_state = rb_state.view(cfg.num_envs, rb_per_env, 13)

    # Jacobian 对 fixed-base asset 通常少 base body，一般 hand index 要 -1
    if jac.shape[1] == rb_count - 1:
        hand_jac_idx = hand_rb_asset_idx - 1
    elif jac.shape[1] == rb_count:
        hand_jac_idx = hand_rb_asset_idx
    else:
        raise RuntimeError(f"Unexpected jac shape={tuple(jac.shape)}, rb_count={rb_count}")

    print(f"[INFO] hand_rb_asset_idx={hand_rb_asset_idx}, hand_jac_idx={hand_jac_idx}, jac_shape={tuple(jac.shape)}")

    # dof limits
    q_lo = []
    q_hi = []
    for env, actor in zip(envs[:1], actors[:1]):
        props = gym.get_actor_dof_properties(env, actor)
        q_lo = torch.tensor(props["lower"][:cfg.arm_dofs], dtype=torch.float32, device=device)
        q_hi = torch.tensor(props["upper"][:cfg.arm_dofs], dtype=torch.float32, device=device)

    home_q = torch.tensor(cfg.home_q, dtype=torch.float32, device=device)
    finger_q = torch.tensor(cfg.finger_open, dtype=torch.float32, device=device)

    # 初始化到 home pose
    init_q = torch.zeros((cfg.num_envs, cfg.total_dofs), device=device, dtype=torch.float32)
    init_q[:, :cfg.arm_dofs] = home_q.unsqueeze(0)
    init_q[:, cfg.arm_dofs:] = finger_q.unsqueeze(0)

    init_qd = torch.zeros((cfg.num_envs, cfg.total_dofs), device=device, dtype=torch.float32)
    dof_state[:, :, 0] = init_q
    dof_state[:, :, 1] = init_qd
    gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof_state))

    # 目标 position tensor
    pos_target = init_q.clone()
    gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos_target))

    # 预热
    for _ in range(cfg.warmup_steps):
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        if viewer is not None:
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, True)

    # 总 episode 数
    total_eps = cfg.num_envs * cfg.episodes_per_env
    T = cfg.episode_len

    # 预分配数据集（单文件）
    dataset_s = torch.zeros((total_eps, T, 14), dtype=torch.float32)
    dataset_a = torch.zeros((total_eps, T, cfg.arm_dofs), dtype=torch.float32)
    dataset_sn = torch.zeros((total_eps, T, 14), dtype=torch.float32)
    dataset_target = torch.zeros((total_eps, 3), dtype=torch.float32)
    dataset_env_id = torch.zeros((total_eps,), dtype=torch.long)
    dataset_episode_id = torch.zeros((total_eps,), dtype=torch.long)

    # 当前 episode buffer（每个 env 一条）
    buf_s = torch.zeros((cfg.num_envs, T, 14), device=device)
    buf_a = torch.zeros((cfg.num_envs, T, cfg.arm_dofs), device=device)
    buf_sn = torch.zeros((cfg.num_envs, T, 14), device=device)

    # 每个 env 的 episode 状态
    ep_step = torch.zeros((cfg.num_envs,), dtype=torch.long, device=device)
    ep_count = torch.zeros((cfg.num_envs,), dtype=torch.long, device=device)

    # 每个 env 的 episode 目标
    target_pos = sample_target_pos(cfg.num_envs, device)
    start_pos = torch.zeros((cfg.num_envs, 3), device=device)
    posture_bias = sample_posture_bias(home_q, q_lo, q_hi, cfg.num_envs, device)
    prev_qd_cmd = torch.zeros((cfg.num_envs, cfg.arm_dofs), device=device)

    # 读取初始 hand pos
    gym.refresh_rigid_body_state_tensor(sim)
    hand_pos = rb_state[:, hand_rb_asset_idx, 0:3]
    start_pos[:] = hand_pos

    render_on = cfg.render
    sim_step = 0
    t0 = time.time()

    print(f"[INFO] collecting Franka recollect v2 ... total_eps={total_eps}, T={T}")
    print(f"[INFO] box_low={cfg.target_box_low}, box_high={cfg.target_box_high}")

    eye7 = torch.eye(cfg.arm_dofs, device=device).unsqueeze(0).repeat(cfg.num_envs, 1, 1)

    while True:
        done_all = torch.all(ep_count >= cfg.episodes_per_env)
        if done_all:
            break

        if viewer is not None:
            if gym.query_viewer_has_closed(viewer):
                print("[WARN] viewer closed, early stop")
                break

        # refresh tensors
        gym.refresh_dof_state_tensor(sim)
        gym.refresh_rigid_body_state_tensor(sim)
        gym.refresh_jacobian_tensors(sim)

        q = dof_state[:, :cfg.arm_dofs, 0]      # (N,7)
        hand_pos = rb_state[:, hand_rb_asset_idx, 0:3]   # (N,3)
        j_eef = jac[:, hand_jac_idx, :, :cfg.arm_dofs]   # (N,6,7)

        # 对还在采集的 env 生效
        active = (ep_count < cfg.episodes_per_env)

        # step 0 时，为每个 env 开一条新 episode
        new_ep = active & (ep_step == 0)
        if torch.any(new_ep):
            # 新 target
            target_pos[new_ep] = sample_target_pos(int(new_ep.sum().item()), device)
            # 记录当前 hand pos 作为直线起点
            start_pos[new_ep] = hand_pos[new_ep]
            # 新 posture bias
            posture_bias[new_ep] = sample_posture_bias(home_q, q_lo, q_hi, int(new_ep.sum().item()), device)
            # 重置平滑动作缓存
            prev_qd_cmd[new_ep] = 0.0

        # 当前状态
        s_t = build_state_sincos(q)

        # 直线子目标：从 start_pos 到 target_pos
        alpha = ((ep_step.float() + 1.0) / float(T)).clamp(0.0, 1.0).unsqueeze(-1)  # (N,1)
        subgoal = (1.0 - alpha) * start_pos + alpha * target_pos

        # 末端位置误差
        pos_err = subgoal - hand_pos
        dpose = torch.zeros((cfg.num_envs, 6, 1), device=device)
        dpose[:, 0:3, 0] = cfg.ee_gain * pos_err

        # 主任务 IK
        dq_task = control_ik(j_eef, dpose, damping=cfg.ik_damping)   # (N,7)

        # nullspace：朝随机 posture bias 拉一点，增强冗余探索
        jT = torch.transpose(j_eef, 1, 2)    # (N,7,6)
        lambda_eye = torch.eye(6, device=device).unsqueeze(0) * (cfg.ik_damping ** 2)
        j_pinv = jT @ torch.inverse(j_eef @ jT + lambda_eye)         # (N,7,6)
        null_proj = eye7 - j_pinv @ j_eef                            # (N,7,7)

        q_err_null = (posture_bias - q).unsqueeze(-1)                # (N,7,1)
        dq_null = (null_proj @ (cfg.null_kp * q_err_null)).squeeze(-1)

        dq = dq_task + dq_null

        # 转成 joint velocity 命令，并做平滑
        qd_raw = torch.clamp(dq / cfg.dt, -cfg.max_qd, cfg.max_qd)
        qd_cmd = cfg.vel_smooth * prev_qd_cmd + (1.0 - cfg.vel_smooth) * qd_raw
        qd_cmd = torch.clamp(qd_cmd, -cfg.max_qd, cfg.max_qd)

        # inactive env 不再更新
        qd_cmd[~active] = 0.0

        # 用位置伺服执行 qdot_cmd：q_target = q + qdot*dt
        q_next_target = q + qd_cmd * cfg.dt
        q_next_target = clamp_q(q_next_target, q_lo.unsqueeze(0), q_hi.unsqueeze(0))

        pos_target[:, :cfg.arm_dofs] = q_next_target
        pos_target[:, cfg.arm_dofs:] = finger_q.unsqueeze(0)
        gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos_target))

        # step sim
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        if viewer is not None and render_on and (sim_step % cfg.render_every == 0):
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, True)
        elif viewer is not None:
            gym.poll_viewer_events(viewer)

        # refresh next state
        gym.refresh_dof_state_tensor(sim)
        q_next = dof_state[:, :cfg.arm_dofs, 0]
        s_next = build_state_sincos(q_next)

        # 写入当前 step buffer
        active_ids = torch.nonzero(active, as_tuple=False).squeeze(-1)
        for idx in active_ids.tolist():
            t = int(ep_step[idx].item())
            buf_s[idx, t] = s_t[idx]
            buf_a[idx, t] = qd_cmd[idx]
            buf_sn[idx, t] = s_next[idx]

        # episode 步数推进
        ep_step[active] += 1
        prev_qd_cmd[active] = qd_cmd[active]

        # episode 结束，归档
        done = active & (ep_step >= T)
        if torch.any(done):
            done_ids = torch.nonzero(done, as_tuple=False).squeeze(-1)
            for idx in done_ids.tolist():
                e_local = int(ep_count[idx].item())
                g = idx * cfg.episodes_per_env + e_local

                dataset_s[g] = buf_s[idx].detach().cpu()
                dataset_a[g] = buf_a[idx].detach().cpu()
                dataset_sn[g] = buf_sn[idx].detach().cpu()
                dataset_target[g] = target_pos[idx].detach().cpu()
                dataset_env_id[g] = idx
                dataset_episode_id[g] = e_local

                # 清 buffer
                buf_s[idx].zero_()
                buf_a[idx].zero_()
                buf_sn[idx].zero_()

            ep_count[done] += 1
            ep_step[done] = 0

        sim_step += 1

        if sim_step % 200 == 0:
            done_eps = int(ep_count.sum().item())
            elapsed = time.time() - t0
            print(f"[PROGRESS] episodes_done={done_eps}/{total_eps} ({100.0*done_eps/total_eps:.2f}%) elapsed={elapsed:.1f}s")

    # 最终保存
    payload = {
        "meta": {
            "name": "franka_recollect_v2_same_box_as_ur",
            "num_envs": cfg.num_envs,
            "episodes_per_env": cfg.episodes_per_env,
            "episode_len": cfg.episode_len,
            "dt": cfg.dt,
            "state_def": "s=[sin(q1..q7), cos(q1..q7)] (14-dim)",
            "action_def": "a=qdot_cmd(7-dim), generated by straight-line EE subgoal + DLS IK + nullspace posture bias",
            "target_box_low": cfg.target_box_low,
            "target_box_high": cfg.target_box_high,
            "note": "recollected Franka dataset with broader joint coverage, same EE box as UR",
        },
        "data": {
            "s": dataset_s,               # (N_eps, T, 14)
            "a": dataset_a,               # (N_eps, T, 7)
            "s_next": dataset_sn,         # (N_eps, T, 14)
            "target_pos": dataset_target, # (N_eps, 3)
            "env_id": dataset_env_id,     # (N_eps,)
            "episode_id": dataset_episode_id,  # (N_eps,)
        }
    }

    torch.save(payload, cfg.out_path)
    print(f"[DONE] saved -> {cfg.out_path}")
    print(f"[DONE] shapes: s={tuple(dataset_s.shape)}, a={tuple(dataset_a.shape)}, s_next={tuple(dataset_sn.shape)}, target_pos={tuple(dataset_target.shape)}")

    if viewer is not None:
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


if __name__ == "__main__":
    main()