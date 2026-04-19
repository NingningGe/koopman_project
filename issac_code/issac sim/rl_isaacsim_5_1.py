# Isaac Sim 5.1.0 compatible version of rl.py
#
# 主要改动：
# 1) SimulationApp 改为从 isaacsim 导入
# 2) ArticulationAction 改为从 isaacsim.core.utils.types 导入
# 3) 不再使用已弃用的 omni.isaac.franka.tasks / omni.isaac.universal_robots.tasks
# 4) 直接通过 add_reference_to_stage + Articulation 加载机器人 USD
# 5) 统一了保存路径，修复了原代码里部分 /home/cxf/franka 路径不一致的问题
#
# 运行方式：
#   ~/isaacsim/python.sh "/home/nng/koopman_project/issac_code/issac sim/rl_isaacsim_5_1.py"

from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch

from isaacsim import SimulationApp

control = 1
#simulation_app = SimulationApp({"headless": False})
simulation_app = SimulationApp({"headless": True})
import carb
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.storage.native import get_assets_root_path

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent
TRANSFER_MODEL_DIR = PROJECT_ROOT / "cr_transferlearning" / "transfer_learning" / "control_transfer" / "A_to_B" / "Data"

if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

FRANKA_SAVE_DIR = THIS_DIR / "franka"
UR_SAVE_DIR = THIS_DIR / "ur"
FRANKA_SAVE_DIR.mkdir(parents=True, exist_ok=True)
UR_SAVE_DIR.mkdir(parents=True, exist_ok=True)

def create_world() -> World:
    world = World(stage_units_in_meters=1.0, physics_dt=1 / 1000)
    world.scene.add_default_ground_plane()
    return world

def get_robot_asset_path(robot: str) -> str:
    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        raise RuntimeError("Could not find Isaac Sim assets folder")
    if robot.lower() == "franka":
        return assets_root_path + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
    elif robot.lower() == "ur5":
        return assets_root_path + "/Isaac/Robots/UniversalRobots/ur5/ur5.usd"
    else:
        raise ValueError(f"Unsupported robot type: {robot}")

def load_articulation(world: World, robot: str, prim_path: str, name: str) -> Articulation:
    usd_path = get_robot_asset_path(robot)
    add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)

    articulation = Articulation(
        prim_paths_expr=prim_path,
        name=name,
    )
    world.scene.add(articulation)
    world.reset()
    return articulation

def set_velocity_mode(robot: Articulation):
    articulation_controller = robot.get_articulation_controller()
    articulation_controller.switch_control_mode("velocity")
    return articulation_controller

def rollout_robot(
    robot: Articulation,
    world: World,
    articulation_controller,
    actor,
    state_dim: int,
    action_dim: int,
    steps: int,
    substeps: int,
    save_prefix: Path,
    use_latent: bool = False,
    latent_models: dict | None = None,
    seed: int = 0,
    low: torch.Tensor | None = None,
    high: torch.Tensor | None = None,
):
    np.random.seed(seed)

    if low is None or high is None:
        raise ValueError("low/high must be provided for initialization")

    p = np.array([np.random.uniform(low[i], high[i]) for i in range(len(low))], dtype=np.float32)

    if action_dim == 7:
        p0 = np.concatenate([p[:7], [0.0, 0.0]])
        v0 = np.concatenate([p[7:], [0.0, 0.0]])
    else:
        p0 = np.concatenate([p[:6]])
        v0 = np.concatenate([p[6:]])

    robot.set_joint_positions(p0)
    robot.set_joint_velocities(v0)

    s_data = []
    u_data = []

    for _ in range(steps):
        positions = robot.get_joint_positions()
        velocities = robot.get_joint_velocities()

        if action_dim == 7:
            state = np.concatenate([positions[:7], velocities[:7]]).astype(np.float32)
        else:
            state = np.concatenate([positions[:6], velocities[:6]]).astype(np.float32)

        s_data.append(state)
        state_tensor = torch.tensor(state, dtype=torch.float32)

        if use_latent:
            enc_net = latent_models["enc_net"]
            dec_net = latent_models["dec_net"]
            with torch.no_grad():
                Y = enc_net.ENC(state_tensor)
                control_latent = actor(Y).squeeze(0) if Y.ndim == 2 else actor(Y)
                control = dec_net.DEC(torch.cat((control_latent, state_tensor), dim=-1)).cpu().numpy()
        else:
            with torch.no_grad():
                control = actor(state_tensor).squeeze(0).cpu().numpy()

        u_data.append(control)

        if action_dim == 7:
            control_apply = np.concatenate([control, [0.0, 0.0]])
        else:
            control_apply = np.array(control, dtype=np.float32)

        actions = ArticulationAction(
            joint_positions=None,
            joint_velocities=control_apply,
            joint_efforts=None,
        )
        articulation_controller.apply_action(actions)

        for _ in range(substeps):
            world.step(render=False)

    s_data = np.asarray(s_data)
    u_data = np.asarray(u_data)

    np.save(save_prefix.parent / f"u_data_{save_prefix.name}.npy", u_data)
    np.save(save_prefix.parent / f"s_data_{save_prefix.name}.npy", s_data)
    
    
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(state_dim, 1, figsize=(10, max(8, state_dim)), sharex=True)
    if state_dim == 1:
        axes = [axes]
    for i in range(state_dim):
        axes[i].plot(s_data[:, i], label="state", color="blue")
        axes[i].set_title(f"Dimension {i + 1}")
        axes[i].set_ylabel("value")
    axes[-1].set_xlabel("time step")
    axes[0].legend()
    plt.tight_layout()
    plt.savefig(save_prefix.parent / f"{save_prefix.name}_plot.png", dpi=150)
    plt.close(fig)

    return s_data, u_data

def load_td3_actor(model_path: str):
    import td3_continuous_action as td3
    actor_state_dict, qf1_state_dict, qf2_state_dict = torch.load(model_path, map_location="cpu")
    actor = td3.Actor(8)
    actor.load_state_dict(actor_state_dict)
    actor.eval()
    return actor

def load_franka_latent_models():
    import three_models_speed as lka

    ckpt_path = str(TRANSFER_MODEL_DIR / "franka" / "unifiedur_transferlayer3_edim100_eloss1.pth")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    net_state_dict = ckpt["net_state_dict"]
    enc_net1_state_dict = ckpt["enc_net1_state_dict"]
    enc_net4_state_dict = ckpt["enc_net4_state_dict"]
    dec_net1_state_dict = ckpt["dec_net1_state_dict"]
    dec_net4_state_dict = ckpt["dec_net4_state_dict"]

    primary_udim1 = 7
    primary_sdim1 = 14
    common_sdim = 20
    common_udim = 10
    in_dim = common_sdim
    u_dim = common_udim
    layer_width = 128
    layer_width2 = 256
    layer_depth = 3
    encode_dim = 100

    XENC_layers1 = [primary_sdim1] + [layer_width] * layer_depth + [common_sdim]
    enc_net1 = lka.ENC_net(XENC_layers1)

    UENC_layers1 = [primary_udim1 + primary_sdim1] + [layer_width] * layer_depth + [common_udim]
    enc_net4 = lka.ENC_net(UENC_layers1)

    DEC_layers1 = [common_sdim] + [layer_width] * layer_depth + [primary_sdim1]
    dec_net1 = lka.DEC_net(DEC_layers1)

    DEC_layers4 = [common_udim + primary_sdim1] + [layer_width] * layer_depth + [primary_udim1]
    dec_net4 = lka.DEC_net(DEC_layers4)

    layers = [in_dim] + [layer_width2] * layer_depth + [encode_dim]
    Nkoopman = in_dim + encode_dim
    net = lka.Network(layers, Nkoopman, u_dim)

    net.load_state_dict(net_state_dict)
    enc_net1.load_state_dict(enc_net1_state_dict)
    enc_net4.load_state_dict(enc_net4_state_dict)
    dec_net1.load_state_dict(dec_net1_state_dict)
    dec_net4.load_state_dict(dec_net4_state_dict)

    net.eval()
    enc_net1.eval()
    enc_net4.eval()
    dec_net1.eval()
    dec_net4.eval()

    return {
        "net": net,
        "enc_net": enc_net1,
        "enc_net_action": enc_net4,
        "dec_net_state": dec_net1,
        "dec_net": dec_net4,
    }

if control == 1:
    world = create_world()
    franka = load_articulation(world, robot="franka", prim_path="/World/Franka", name="franka")
    franka.disable_gravity()
    articulation_controller = set_velocity_mode(franka)

    actor = load_td3_actor("/home/nng/koopman_project/issac_code/tranfer_control/1/td3_continuous_action_20250620-163601.cleanrl_model")

    low = torch.tensor([-0.3, -1.1, -0.3, -1.9, -0.3, 1.3, -0.3, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4], dtype=torch.float32)
    high = torch.tensor([0.3, -0.5, 0.3, -1.3, 0.3, 1.9, 0.3, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4], dtype=torch.float32)

    rollout_robot(franka, world, articulation_controller, actor, 14, 7, 100, 20, FRANKA_SAVE_DIR / "franka", False, None, 7, low, high)

elif control == 2:
    world = create_world()
    ur5 = load_articulation(world, robot="ur5", prim_path="/World/UR5", name="ur5")
    ur5.disable_gravity()
    articulation_controller = set_velocity_mode(ur5)

    actor = load_td3_actor("/home/nng/koopman_project/issac_code/tranfer_control/ur2.cleanrl_model")

    low = torch.tensor([-0.3, -1.9, 0.5, -1.9, -0.3, -0.3, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4], dtype=torch.float32)
    high = torch.tensor([0.3, -1.3, 1.1, -1.3, 0.3, 0.3, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4], dtype=torch.float32)

    rollout_robot(ur5, world, articulation_controller, actor, 12, 6, 100, 20, UR_SAVE_DIR / "ur", False, None, 8, low, high)

elif control == 3:
    world = create_world()
    franka = load_articulation(world, robot="franka", prim_path="/World/Franka", name="franka")
    franka.disable_gravity()
    articulation_controller = set_velocity_mode(franka)

    actor = load_td3_actor("/home/nng/koopman_project/issac_code/tranfer_control/2/td3_continuous_action_20250620-165810.cleanrl_model")
    latent_models = load_franka_latent_models()

    low = torch.tensor([-0.3, -1.1, -0.3, -1.9, -0.3, 1.3, -0.3, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4], dtype=torch.float32)
    high = torch.tensor([0.3, -0.5, 0.3, -1.3, 0.3, 1.9, 0.3, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4], dtype=torch.float32)

    s_data, u_data = rollout_robot(franka, world, articulation_controller, actor, 14, 7, 100, 20, FRANKA_SAVE_DIR / "franka_latent", True, latent_models, 1, low, high)
    print("Final state:", s_data[-1, :])

else:
    raise ValueError(f"Unsupported control mode: {control}")

simulation_app.close()
