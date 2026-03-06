import os
import math
import time
from dataclasses import dataclass

import isaacgym
from isaacgym import gymapi, gymutil, gymtorch

# 重要：IsaacGym 要求 torch 在 isaacgym 之后 import
import torch


@dataclass
class Cfg:
    # IsaacGym 安装路径（你给的）
    asset_root: str = "/home/nng/software/isaacgym_preview4/isaacgym/assets"
    asset_file: str = "urdf/franka_description/robots/franka_panda.urdf"

    # 数据规模（你可先小跑验证，再拉满到 10000/ENV）
    episodes_per_env: int = 310
    episode_len: int = 200
    num_envs: int = 32  # 建议后面开到 64/128（先确保运动正常）

    # 仿真
    dt: float = 1.0 / 60.0
    substeps: int = 2
    sim_device_id: int = 0
    graphics_device_id: int = 0

    # GPU pipeline（想尽量用 GPU 必须开它）
    use_gpu_physx: bool = True
    use_gpu_pipeline: bool = True

    # Franka DOF
    arm_dofs: int = 7
    total_dofs: int = 9  # 7 arm + 2 fingers

    # 末端直线到随机点采样区域（Wang 的 box）
    box_min = torch.tensor([-0.2, -0.25, 0.8], dtype=torch.float32)
    box_max = torch.tensor([ 0.2,  0.25, 1.2], dtype=torch.float32)

    # 论文里提的“初始夹爪位置”
    # 注：IsaacGym 坐标/桌面不同，Franka home 不一定就在这里。
    # 我们会：所有 env 关节统一 home pose；第 0 个 episode 会自动把末端拉向 init_eef_pos（不算 reset）
    init_eef_pos = torch.tensor([-0.2, 0.0, 1.05], dtype=torch.float32)

    # 控制（对标 Wang：末端 ds*kp clip + JV 的 P 控制思想）
    kp_ee: float = 4.0          # 对标 ReachPolicy 里 move_line 的 kp=4
    max_ee_step: float = 0.03   # 每步末端最大位移（m），越小越稳但更慢
    ik_damping: float = 0.05    # DLS 阻尼，越大越稳但更保守

    # 关节速度限制 + 平滑
    max_qd: float = 1.2         # rad/s
    action_smooth_alpha: float = 0.85  # 0~1，越大越平滑（更少抖动）

    # viewer
    render: bool = False
    render_every: int = 2


    # 保存
    out_dir: str = "./data_wang_franka_reach_ablation"
    chunk_episodes: int = 200   # 每个 env 累积多少 episodes 落盘一次

    # 最终合并为单文件
    merge_at_end: bool = True
    merged_filename: str = "dataset_wang_franka_reach_ablation.pt"
    shard_glob: str = ".pt"  # 当前输出目录下所有 shard 的后缀


cfg = Cfg()


def build_state_sincos(q_arm: torch.Tensor) -> torch.Tensor:
    """q_arm: (N,7) -> s: (N,14)"""
    return torch.cat([torch.sin(q_arm), torch.cos(q_arm)], dim=-1)


def franka_home_q(device):
    # 常见 Franka home pose（近似）
    q = torch.tensor([0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398],
                     dtype=torch.float32, device=device)
    return q


def create_sim():
    gym = gymapi.acquire_gym()
    _ = gymutil.parse_arguments(description="Collect Franka reach ablation dataset (Wang-style sampling)")

    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0, 0, -9.81)
    sim_params.dt = cfg.dt
    sim_params.substeps = cfg.substeps

    # PhysX
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 6
    sim_params.physx.num_velocity_iterations = 2
    sim_params.physx.use_gpu = cfg.use_gpu_physx

    # GPU pipeline（关键，否则很多 tensor API 只能 CPU）
    sim_params.use_gpu_pipeline = cfg.use_gpu_pipeline

    sim = gym.create_sim(cfg.sim_device_id, cfg.graphics_device_id, gymapi.SIM_PHYSX, sim_params)
    if sim is None:
        raise RuntimeError("create_sim 失败：检查 IsaacGym / CUDA / 驱动")

    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane_params)

    viewer = None
    if cfg.render:
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        if viewer is None:
            raise RuntimeError("create_viewer 失败：检查图形环境/显示器")
        cam_pos = gymapi.Vec3(2.0, 2.0, 2.0)
        cam_tgt = gymapi.Vec3(0.0, 0.0, 1.0)
        gym.viewer_camera_look_at(viewer, None, cam_pos, cam_tgt)
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_V, "toggle_render")

    return gym, sim, viewer


def load_franka_asset(gym, sim):
    opt = gymapi.AssetOptions()
    opt.fix_base_link = True
    opt.flip_visual_attachments = True
    opt.armature = 0.01
    # 用关节速度控制（JV ablation）
    opt.default_dof_drive_mode = gymapi.DOF_MODE_VEL

    asset = gym.load_asset(sim, cfg.asset_root, cfg.asset_file, opt)
    if asset is None:
        raise RuntimeError("load_asset 失败：检查 asset_root / asset_file")

    return asset


def create_envs(gym, sim, asset):
    envs, actors = [], []
    env_lower = gymapi.Vec3(-1.0, -1.0, 0.0)
    env_upper = gymapi.Vec3( 1.0,  1.0, 1.5)
    num_per_row = int(math.sqrt(cfg.num_envs))

    for i in range(cfg.num_envs):
        env = gym.create_env(sim, env_lower, env_upper, num_per_row)

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

        actor = gym.create_actor(env, asset, pose, "franka", i, 1)

        dof_props = gym.get_actor_dof_properties(env, actor)
        dof_props["driveMode"].fill(gymapi.DOF_MODE_VEL)
        # 速度伺服的阻尼（让它像真实伺服，不乱抖）
        dof_props["stiffness"].fill(0.0)
        dof_props["damping"].fill(80.0)
        # 限速（可选）
        # dof_props["velocity"].fill(cfg.max_qd)

        gym.set_actor_dof_properties(env, actor, dof_props)

        envs.append(env)
        actors.append(actor)

    return envs, actors


def merge_shards_to_one(out_dir: str, merged_path: str):
    """
    把 out_dir 下的所有 shard(pt) 合并为一个单文件：
    data:
      s        : (N_eps, T, 14)
      a        : (N_eps, T, 7)
      s_next   : (N_eps, T, 14)
      target_pos: (N_eps, 3)   (如果 shard 里叫 target 或 target_pos，都会兼容)
      env_id   : (N_eps,)
      episode_id: (N_eps,)
    """
    files = [f for f in os.listdir(out_dir) if f.endswith(".pt")]
    files = sorted(files)
    if len(files) == 0:
        raise RuntimeError(f"目录里没有找到 shard: {out_dir}")

    all_s, all_a, all_sn, all_tgt = [], [], [], []
    all_env, all_epid = [], []
    src_files = []

    global_ep = 0
    T = None

    for fn in files:
        path = os.path.join(out_dir, fn)
        obj = torch.load(path, map_location="cpu")
        if not isinstance(obj, dict) or "data" not in obj:
            # 跳过非 shard 文件（比如你之后可能已经生成 merged 文件）
            continue

        data = obj["data"]
        s = data["s"]            # (K,T,14)
        a = data["a"]            # (K,T,7)
        sn = data["s_next"]      # (K,T,14)

        # target 兼容两种命名
        if "target_pos" in data:
            tgt = data["target_pos"]   # (K,3) 或 (K,3)
        elif "target" in data:
            tgt = data["target"]
        else:
            tgt = torch.zeros((s.shape[0], 3), dtype=torch.float32)

        # 基本检查
        if T is None:
            T = s.shape[1]
        else:
            if s.shape[1] != T:
                raise RuntimeError(f"T 不一致：{fn} 的 T={s.shape[1]}，期望 T={T}")

        K = s.shape[0]
        env_id = obj.get("meta", {}).get("env_id", -1)

        all_s.append(s.contiguous())
        all_a.append(a.contiguous())
        all_sn.append(sn.contiguous())
        all_tgt.append(tgt.contiguous())

        all_env.append(torch.full((K,), int(env_id), dtype=torch.long))
        all_epid.append(torch.arange(global_ep, global_ep + K, dtype=torch.long))
        global_ep += K

        src_files.append(fn)

    if len(all_s) == 0:
        raise RuntimeError("没有读取到有效 shard（可能目录里只有 merged 文件或格式不对）")

    S = torch.cat(all_s, dim=0)       # (N_eps,T,14)
    A = torch.cat(all_a, dim=0)       # (N_eps,T,7)
    SN = torch.cat(all_sn, dim=0)     # (N_eps,T,14)
    TG = torch.cat(all_tgt, dim=0)    # (N_eps,3)
    ENV = torch.cat(all_env, dim=0)   # (N_eps,)
    EPID = torch.cat(all_epid, dim=0) # (N_eps,)

    merged = {
        "meta": {
            "format": "episode_major",
            "state_dim": int(S.shape[-1]),
            "action_dim": int(A.shape[-1]),
            "episode_len": int(S.shape[1]),
            "num_episodes": int(S.shape[0]),
            "note": "Single-file merged dataset. Do NOT treat different episodes as one continuous trajectory.",
            "source_shards": src_files,
        },
        "data": {
            "s": S.float(),
            "a": A.float(),
            "s_next": SN.float(),
            "target_pos": TG.float(),
            "env_id": ENV,
            "episode_id": EPID,
        }
    }

    torch.save(merged, merged_path)
    print(f"[MERGE] 合并完成：{merged_path}")
    print(f"[MERGE] s={tuple(S.shape)} a={tuple(A.shape)} s_next={tuple(SN.shape)} target_pos={tuple(TG.shape)}")



def main():
    os.makedirs(cfg.out_dir, exist_ok=True)

    gym, sim, viewer = create_sim()
    asset = load_franka_asset(gym, sim)
    envs, actors = create_envs(gym, sim, asset)

    gym.prepare_sim(sim)

    # ---- tensors ----
    dof_state_tensor = gym.acquire_dof_state_tensor(sim)
    dof_states = gymtorch.wrap_tensor(dof_state_tensor)  # (num_envs*total_dofs, 2)

    rb_state_tensor = gym.acquire_rigid_body_state_tensor(sim)
    rb_states = gymtorch.wrap_tensor(rb_state_tensor)    # (num_envs*num_bodies, 13)

    # Jacobian tensor（GPU pipeline 开启时会在 GPU 上）
    jac_tensor = gym.acquire_jacobian_tensor(sim, "franka")
    jac = gymtorch.wrap_tensor(jac_tensor)

    gym.refresh_dof_state_tensor(sim)
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_jacobian_tensors(sim)

    device = dof_states.device  # 跟随 IsaacGym pipeline（cpu 或 cuda）

    # ---- 查末端刚体索引：panda_hand 是“末端刚体名字”
    # 这是 URDF 里的 link 名，很多地方都会用它当 EE
    hand_body_name = "panda_hand"
    hand_rb_index = gym.find_actor_rigid_body_handle(envs[0], actors[0], hand_body_name)
    if hand_rb_index < 0:
        raise RuntimeError(f"找不到 rigid body: {hand_body_name}")
    # reshape views
    dof_states_view = dof_states.view(cfg.num_envs, cfg.total_dofs, 2)
    # rb_states: (num_envs*num_bodies, 13) -> (num_envs, num_bodies, 13)
    num_bodies = rb_states.shape[0] // cfg.num_envs
    rb_states_view = rb_states.view(cfg.num_envs, num_bodies, 13)
    # jac: (num_envs, num_bodies, 6, total_dofs)  (IsaacGym 的标准布局)

    # rb_states 的 body 数（仿真里包含的 rigid bodies）
    num_bodies_rb = rb_states.shape[0] // cfg.num_envs
    rb_states_view = rb_states.view(cfg.num_envs, num_bodies_rb, 13)

    # jacobian tensor 的 body 数（常见：不包含 base body，所以比 rb 少 1）
    num_bodies_jac = jac.numel() // (cfg.num_envs * 6 * cfg.total_dofs)
    jac_view = jac.view(cfg.num_envs, num_bodies_jac, 6, cfg.total_dofs)

    # === 处理 body 索引对齐：如果 jac 比 rb 少 1，则认为 jac 去掉了最前面的 base body ===
    if num_bodies_jac == num_bodies_rb - 1:
        hand_jac_index = hand_rb_index - 1
    else:
        hand_jac_index = hand_rb_index

    if not (0 <= hand_jac_index < num_bodies_jac):
        raise RuntimeError(
            f"hand_jac_index 越界：hand_rb_index={hand_rb_index}, "
            f"hand_jac_index={hand_jac_index}, num_bodies_rb={num_bodies_rb}, num_bodies_jac={num_bodies_jac}"
        )

    # ---- 初始化：所有 env 统一 home joint pose（与你要求一致）----
    home_q = franka_home_q(device).unsqueeze(0).repeat(cfg.num_envs, 1)  # (N,7)
    home_qd = torch.zeros(cfg.num_envs, cfg.arm_dofs, device=device)

    # fingers open
    finger_open = torch.tensor([0.04, 0.04], device=device).unsqueeze(0).repeat(cfg.num_envs, 1)

    # 写入 dof state（位置/速度）
    dof_states_view[:, :, 0] = 0.0
    dof_states_view[:, :, 1] = 0.0
    dof_states_view[:, :cfg.arm_dofs, 0] = home_q
    dof_states_view[:, :cfg.arm_dofs, 1] = home_qd
    dof_states_view[:, cfg.arm_dofs:cfg.total_dofs, 0] = finger_open
    dof_states_view[:, cfg.arm_dofs:cfg.total_dofs, 1] = 0.0

    gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof_states))

    # 速度 target tensor（JV 控制）
    vel_tgt_view = torch.zeros((cfg.num_envs, cfg.total_dofs), device=device, dtype=torch.float32)
    vel_tgt = vel_tgt_view.contiguous().view(-1)

    # ---- 数据 chunk 缓冲 ----
    chunk_eps = cfg.chunk_episodes
    T = cfg.episode_len

    def alloc_chunk():
        return {
            "s": torch.zeros(cfg.num_envs, chunk_eps, T, 14, dtype=torch.float32),
            "a": torch.zeros(cfg.num_envs, chunk_eps, T, cfg.arm_dofs, dtype=torch.float32),
            "s_next": torch.zeros(cfg.num_envs, chunk_eps, T, 14, dtype=torch.float32),
            "target": torch.zeros(cfg.num_envs, chunk_eps, 3, dtype=torch.float32),
        }

    chunk = alloc_chunk()
    chunk_fill = torch.zeros(cfg.num_envs, dtype=torch.long, device=device)

    ep_step = torch.zeros(cfg.num_envs, dtype=torch.long, device=device)
    ep_count = torch.zeros(cfg.num_envs, dtype=torch.long, device=device)

    # 每个 env 当前 episode 的 target
    target_pos = torch.zeros(cfg.num_envs, 3, device=device)

    # JV 动作（平滑）
    qd_cmd = torch.zeros(cfg.num_envs, cfg.arm_dofs, device=device)

    render_on = cfg.render
    total_episodes = cfg.episodes_per_env * cfg.num_envs
    last_print = time.time()
    sim_step = 0

    print(f"[INFO] device={device} | gpu_pipeline={cfg.use_gpu_pipeline} | num_envs={cfg.num_envs}")
    print(f"[INFO] total_episodes = episodes_per_env({cfg.episodes_per_env}) * num_envs({cfg.num_envs}) = {total_episodes}")
    print(f"[INFO] state: 14=[sin(q1..q7),cos(q1..q7)] | action: 7=joint_vel")
    print("[INFO] 采样策略：每回合采样一个 target，末端沿直线闭环逼近该点；回合间不 reset。")

    while True:
        if torch.all(ep_count >= cfg.episodes_per_env):
            break

        # viewer
        if viewer is not None:
            if gym.query_viewer_has_closed(viewer):
                print("[WARN] Viewer 关闭，提前退出")
                break
            for evt in gym.query_viewer_action_events(viewer):
                if evt.action == "toggle_render" and evt.value > 0:
                    render_on = not render_on
                    print(f"[INFO] render_on={render_on}")

        # refresh tensors
        gym.refresh_dof_state_tensor(sim)
        gym.refresh_rigid_body_state_tensor(sim)
        gym.refresh_jacobian_tensors(sim)

        # 当前关节
        q = dof_states_view[:, :cfg.arm_dofs, 0]   # (N,7)
        # 末端位置（panda_hand）
        eef_pos = rb_states_view[:, hand_rb_index, 0:3] # (N,3)

        # episode 开始：采样目标点（第 0 个 episode 先把 target 设置为 init_eef_pos，让它“自然拉过去”）
        need_new = (ep_step == 0) & (ep_count < cfg.episodes_per_env)
        if torch.any(need_new):
            ids = torch.nonzero(need_new, as_tuple=False).squeeze(-1)

            # 第 0 个 episode：先去 init_eef_pos（满足你提到的论文设置，不算 reset）
            is_first_ep = (ep_count[ids] == 0)
            if torch.any(is_first_ep):
                first_ids = ids[is_first_ep]
                target_pos[first_ids] = cfg.init_eef_pos.to(device)

            # 其它 episode：随机采样 box 里的点
            other_ids = ids[~is_first_ep]
            if other_ids.numel() > 0:
                u = torch.rand(other_ids.numel(), 3, device=device)
                t = cfg.box_min.to(device) + u * (cfg.box_max.to(device) - cfg.box_min.to(device))
                target_pos[other_ids] = t

            # 把本 episode 的 target 写进 chunk（按 env 各自的 chunk_fill 索引）
            for idx in ids.tolist():
                ep_i = int(chunk_fill[idx].item())
                if ep_i < chunk_eps:
                    chunk["target"][idx, ep_i] = target_pos[idx].detach().cpu()

        # ---- 末端直线闭环控制（对标 Wang 的 ds*kp clip）----
        ds = (target_pos - eef_pos)  # (N,3)
        dx = ds * cfg.kp_ee
        dx = torch.clamp(dx, -cfg.max_ee_step, cfg.max_ee_step)  # 每步末端最大位移

        # ---- resolved-rate IK (DLS) : dq = J^T (J J^T + λI)^-1 dx ----
        J = jac_view[:, hand_jac_index, 0:3, :cfg.arm_dofs] # (N,3,7)

        # A = J J^T + λI : (N,3,3)
        JJt = J @ J.transpose(1, 2)
        lamI = (cfg.ik_damping ** 2) * torch.eye(3, device=device).unsqueeze(0)
        A = JJt + lamI

        # solve A y = dx  => y: (N,3,1)
        y = torch.linalg.solve(A, dx.unsqueeze(-1))
        dq = (J.transpose(1, 2) @ y).squeeze(-1)  # (N,7)

        # dq 是“这一仿真步的关节增量”，转成关节速度命令 qdot
        qd_raw = dq / cfg.dt

        # 限速
        qd_raw = torch.clamp(qd_raw, -cfg.max_qd, cfg.max_qd)

        # 平滑（减少抽搐）
        alpha = cfg.action_smooth_alpha
        qd_cmd = alpha * qd_cmd + (1.0 - alpha) * qd_raw

        # 记录 state/action（当前）
        s_t = build_state_sincos(q)

        # 下发速度 target（9 dof，后两个手指 0）
        vel_tgt_view.zero_()
        vel_tgt_view[:, :cfg.arm_dofs] = qd_cmd
        gym.set_dof_velocity_target_tensor(sim, gymtorch.unwrap_tensor(vel_tgt))

        # step sim
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        # render
        if viewer is not None and render_on and (sim_step % cfg.render_every == 0):
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, True)
        elif viewer is not None:
            gym.poll_viewer_events(viewer)

        # refresh next
        gym.refresh_dof_state_tensor(sim)
        q_next = dof_states_view[:, :cfg.arm_dofs, 0]
        s_next = build_state_sincos(q_next)

        # 写入 chunk（只写 active）
        active = (ep_count < cfg.episodes_per_env)
        if torch.any(active):
            ids = torch.nonzero(active, as_tuple=False).squeeze(-1).tolist()
            for idx in ids:
                ep_i = int(chunk_fill[idx].item())
                t_i = int(ep_step[idx].item())
                if ep_i < chunk_eps and t_i < T:
                    chunk["s"][idx, ep_i, t_i] = s_t[idx].detach().cpu()
                    chunk["a"][idx, ep_i, t_i] = qd_cmd[idx].detach().cpu()
                    chunk["s_next"][idx, ep_i, t_i] = s_next[idx].detach().cpu()

        # step++
        ep_step += 1

        # episode done
        done = (ep_step >= cfg.episode_len) & (ep_count < cfg.episodes_per_env)
        if torch.any(done):
            ep_count[done] += 1
            ep_step[done] = 0
            chunk_fill[done] += 1

            # chunk 满了落盘
            full = (chunk_fill >= cfg.chunk_episodes)
            if torch.any(full):
                full_ids = torch.nonzero(full, as_tuple=False).squeeze(-1).tolist()
                for idx in full_ids:
                    done_eps = int(ep_count[idx].item())
                    out_path = os.path.join(cfg.out_dir, f"franka_env{idx:03d}_eps{done_eps:05d}.pt")
                    payload = {
                        "meta": {
                            "env_id": idx,
                            "episodes_in_file": cfg.chunk_episodes,
                            "episode_len": cfg.episode_len,
                            "dt": cfg.dt,
                            "state_def": "s=[sin(q1..q7), cos(q1..q7)] (14-dim)",
                            "action_def": "a=qdot_cmd(7-dim) joint velocity control",
                            "sampling_def": "per-episode target sampled in box; eef moves straight (closed-loop) to target; no reset between episodes",
                            "box_min": cfg.box_min.tolist(),
                            "box_max": cfg.box_max.tolist(),
                            "init_eef_pos": cfg.init_eef_pos.tolist(),
                            "kp_ee": cfg.kp_ee,
                            "max_ee_step": cfg.max_ee_step,
                            "ik_damping": cfg.ik_damping,
                            "max_qd": cfg.max_qd,
                            "smooth_alpha": cfg.action_smooth_alpha,
                            "eef_body_name": "panda_hand",
                        },
                        "data": {
                            "s": chunk["s"][idx].clone(),           # (chunk_eps, T, 14)
                            "a": chunk["a"][idx].clone(),           # (chunk_eps, T, 7)
                            "s_next": chunk["s_next"][idx].clone(), # (chunk_eps, T, 14)
                            "target": chunk["target"][idx].clone(), # (chunk_eps, 3)
                        }
                    }
                    torch.save(payload, out_path)

                    # clear
                    chunk["s"][idx].zero_()
                    chunk["a"][idx].zero_()
                    chunk["s_next"][idx].zero_()
                    chunk["target"][idx].zero_()
                    chunk_fill[idx] = 0

        # progress
        sim_step += 1
        if time.time() - last_print > 2.0:
            total_done = int(torch.sum(ep_count).item())
            pct = 100.0 * total_done / float(total_episodes)
            print(f"[PROGRESS] episodes_done={total_done}/{total_episodes} ({pct:.2f}%) | render_on={render_on}")
            last_print = time.time()

    # 保存尾部未满 chunk
    print("[INFO] 保存尾部未满 chunk...")
    for idx in range(cfg.num_envs):
        k = int(chunk_fill[idx].item())
        if k > 0:
            done_eps = int(ep_count[idx].item())
            out_path = os.path.join(cfg.out_dir, f"franka_env{idx:03d}_eps{done_eps:05d}_tail.pt")
            payload = {
                "meta": {
                    "env_id": idx,
                    "episodes_in_file": k,
                    "episode_len": cfg.episode_len,
                    "dt": cfg.dt,
                    "state_def": "s=[sin(q1..q7), cos(q1..q7)] (14-dim)",
                    "action_def": "a=qdot_cmd(7-dim) joint velocity control",
                },
                "data": {
                    "s": chunk["s"][idx, :k].clone(),
                    "a": chunk["a"][idx, :k].clone(),
                    "s_next": chunk["s_next"][idx, :k].clone(),
                    "target": chunk["target"][idx, :k].clone(),
                }
            }
            torch.save(payload, out_path)

    if viewer is not None:
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
    # === 合并所有 shard 为一个最终数据包 ===
    if cfg.merge_at_end:
        merged_path = os.path.join(cfg.out_dir, cfg.merged_filename)
        merge_shards_to_one(cfg.out_dir, merged_path)
    print(f"[DONE] 数据已保存到：{os.path.abspath(cfg.out_dir)}")


if __name__ == "__main__":
    main()