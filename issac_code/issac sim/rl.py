#   ~/isaacsim/python.sh "/home/nng/koopman_project/issac_code/issac sim/rl.py"

control=3
from pathlib import Path
if control==1:
    from omni.isaac.kit import SimulationApp

    simulation_app = SimulationApp({"headless": False})

    from omni.isaac.franka.tasks import PickPlace
    from omni.isaac.franka.controllers import PickPlaceController
    from omni.isaac.core import World
    from omni.isaac.core.utils.types import ArticulationAction
    from omni.isaac.core import SimulationContext
    import numpy as np
    import time

    my_world = World(stage_units_in_meters=1.0, physics_dt=1/1000)
    my_task = PickPlace()
    my_world.add_task(my_task)
    simulation_context = SimulationContext()

    my_world.reset()
    task_params = my_task.get_params()
    my_franka = my_world.scene.get_object(task_params["robot_name"]["value"])
    my_franka.disable_gravity()

    articulation_controller = my_franka.get_articulation_controller()
    articulation_controller.switch_control_mode("velocity")

    import td3_continuous_action as td3
    model_path = "/home/nng/koopman_project/issac_code/tranfer_control/1/td3_continuous_action_20250620-163601.cleanrl_model"
    import torch
    actor_state_dict, qf1_state_dict, qf2_state_dict = torch.load(model_path)
    net1 = td3.Actor(8)
    net1.load_state_dict(actor_state_dict)

    np.random.seed(7)
    low2 = torch.tensor([-0.3, -1.1, -0.3, -1.9, -0.3, 1.3, -0.3, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4],dtype=torch.float32)

    high2 = torch.tensor([0.3, -0.5, 0.3, -1.3, 0.3, 1.9, 0.3, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],dtype=torch.float32)
    p = np.array([np.random.uniform(low2[i], high2[i]) for i in range(len(low2))]).astype(np.float32)
    p0 = np.concatenate([p[:7], [0, 0]])
    my_franka.set_joint_positions(p0)
    v0 = np.concatenate([p[7:], [0, 0]])
    my_franka.set_joint_velocities(v0)
    s_data = []
    u_data = []

    for i in range(100):
        positions = my_franka.get_joint_positions()
        velocities = my_franka.get_joint_velocities()
        state = np.concatenate([positions[:7], velocities[:7]]).flatten()
        s_data.append(state)
        control = net1(torch.Tensor(state)).squeeze(0).detach().numpy()
        u_data.append(control)
        control = np.concatenate([control, [0, 0]])
        actions = ArticulationAction(
                joint_positions=None,
                joint_velocities=control,
                joint_efforts=None,
            )
        articulation_controller.apply_action(actions)
        for j in range(20):
            my_world.step(render=False)
        

    u_data = np.array(u_data)
    s_data = np.array(s_data)
    save_dir = Path("/home/nng/koopman_project/issac_code/franka")
    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / "u_data_franka.npy", u_data)
    np.save(save_dir / "s_data_franka.npy", s_data)
    import numpy as np
    import matplotlib.pyplot as plt
    s_data = np.load('/home/nng/koopman_project/issac_code/issac sim/franka/s_data_franka.npy')
    s_data=s_data.reshape(-1,14)
    fig, axes = plt.subplots(14, 1, figsize=(10, 14), sharex=True)
    for i in range(14):
        axes[i].plot(s_data[:, i], label='angles', color='blue')
        axes[i].set_title(f'Dimension {i+1}')
        axes[i].set_ylabel('Angle')
    axes[-1].set_xlabel('Time step')
    axes[0].legend()
    plt.tight_layout()
    plt.show()



elif control==2:
    #ur5
    from omni.isaac.kit import SimulationApp

    simulation_app = SimulationApp({"headless": False})

    from omni.isaac.universal_robots.tasks import PickPlace
    from omni.isaac.universal_robots.controllers import PickPlaceController
    from omni.isaac.core import World
    from omni.isaac.core.utils.types import ArticulationAction
    from omni.isaac.core import SimulationContext
    import numpy as np
    import time

    my_world = World(stage_units_in_meters=1.0, physics_dt=1/1000)
    my_task = PickPlace()
    my_world.add_task(my_task)
    simulation_context = SimulationContext()

    my_world.reset()
    task_params = my_task.get_params()
    my_ur = my_world.scene.get_object(task_params["robot_name"]["value"])
    my_ur.disable_gravity()

    articulation_controller = my_ur.get_articulation_controller()
    articulation_controller.switch_control_mode("velocity")

    import td3_continuous_action as td3
    model_path = "/home/nng/koopman_project/issac_code/tranfer_control/ur2.cleanrl_model"
    import torch
    actor_state_dict, qf1_state_dict, qf2_state_dict = torch.load(model_path)
    net1 = td3.Actor(8)
    net1.load_state_dict(actor_state_dict)

    np.random.seed(8)
    low2 = torch.tensor([-0.3, -1.9, 0.5, -1.9, -0.3, -0.3, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4],dtype=torch.float32)

    high2 = torch.tensor([0.3, -1.3, 1.1, -1.3, 0.3,  0.3, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],dtype=torch.float32)
    p = np.array([np.random.uniform(low2[i], high2[i]) for i in range(len(low2))]).astype(np.float32)
    p0 = np.concatenate([p[:6]])
    my_ur.set_joint_positions(p0)
    v0 = np.concatenate([p[6:]])
    my_ur.set_joint_velocities(v0)
    s_data = []
    u_data = []

    for i in range(100):
        positions = my_ur.get_joint_positions()
        velocities = my_ur.get_joint_velocities()
        state = np.concatenate([positions[:6], velocities[:6]]).flatten()
        s_data.append(state)
        control = net1(torch.Tensor(state)).squeeze(0).detach().numpy()
        u_data.append(control)
        control = np.concatenate([control])
        actions = ArticulationAction(
                joint_positions=None,
                joint_velocities=control,
                joint_efforts=None,
            )
        articulation_controller.apply_action(actions)
        for j in range(20):
            my_world.step(render=False)
        

    u_data = np.array(u_data)
    s_data = np.array(s_data)
    save_dir = Path("/home/nng/koopman_project/issac_code/franka")
    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / "u_data_ur.npy", u_data)
    np.save(save_dir / "s_data_ur.npy", s_data)

    import numpy as np
    import matplotlib.pyplot as plt
    s_data = np.load('/home/nng/koopman_project/issac_code/franka/s_data_ur.npy')
    s_data=s_data.reshape(-1,12)
    fig, axes = plt.subplots(12, 1, figsize=(10, 12), sharex=True)
    for i in range(12):
        axes[i].plot(s_data[:, i], label='angles', color='blue')
        axes[i].set_title(f'Dimension {i+1}')
        axes[i].set_ylabel('Angle')
    axes[-1].set_xlabel('Time step')
    axes[0].legend()
    plt.tight_layout()
    plt.show()

elif control==3:
    from omni.isaac.kit import SimulationApp

    simulation_app = SimulationApp({"headless": False})

    from omni.isaac.franka.tasks import PickPlace
    from omni.isaac.franka.controllers import PickPlaceController
    from omni.isaac.core import World
    from omni.isaac.core.utils.types import ArticulationAction
    from omni.isaac.core import SimulationContext
    import numpy as np
    import time

    my_world = World(stage_units_in_meters=1.0, physics_dt=1/1000)
    my_task = PickPlace()
    my_world.add_task(my_task)
    simulation_context = SimulationContext()

    my_world.reset()
    task_params = my_task.get_params()
    my_franka = my_world.scene.get_object(task_params["robot_name"]["value"])
    my_franka.disable_gravity()

    articulation_controller = my_franka.get_articulation_controller()
    articulation_controller.switch_control_mode("velocity")

    import td3_continuous_action as td3
    model_path = "/home/nng/koopman_project/issac_code/tranfer_control/2/td3_continuous_action_20250620-165810.cleanrl_model"
    import torch
    actor_state_dict, qf1_state_dict, qf2_state_dict = torch.load(model_path)
    net1 = td3.Actor(8)
    net1.load_state_dict(actor_state_dict)

    import three_models_speed as lka
    dicts = torch.load("/home/nng/koopman_project/cr_transferlearning/transfer_learning/control_transfer/A_to_B/Data/franka/unifiedur_transferlayer3_edim100_eloss1.pth",map_location=torch.device('cpu'))
    net_state_dict = dicts["net_state_dict"]
    dnet_state_dict = dicts["dnet_state_dict"]
    enc_net1_state_dict = dicts["enc_net1_state_dict"]
    enc_net4_state_dict = dicts["enc_net4_state_dict"]
    dec_net1_state_dict = dicts["dec_net1_state_dict"]
    dec_net4_state_dict = dicts["dec_net4_state_dict"]
    primary_udim1 = 7
    primary_sdim1 = 14
    common_sdim = 20
    common_udim = 10
    in_dim = common_sdim
    u_dim = common_udim
    Nstate = in_dim
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
    DEC7 = [common_sdim + common_sdim] + [layer_width] * layer_depth + [common_udim]
    dnet = lka.DEC_net(DEC7)
    layers = [in_dim] + [layer_width2] * layer_depth + [encode_dim]
    Nkoopman = in_dim + encode_dim
    net = lka.Network(layers, Nkoopman, u_dim)
    net.load_state_dict(net_state_dict)
    enc_net1.load_state_dict(enc_net1_state_dict)
    enc_net4.load_state_dict(enc_net4_state_dict)
    dec_net1.load_state_dict(dec_net1_state_dict)
    dec_net4.load_state_dict(dec_net4_state_dict)

    np.random.seed(1)
    low2 = torch.tensor([-0.3, -1.1, -0.3, -1.9, -0.3, 1.3, -0.3, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4],dtype=torch.float32)

    high2 = torch.tensor([0.3, -0.5, 0.3, -1.3, 0.3, 1.9, 0.3, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],dtype=torch.float32)
    p = np.array([np.random.uniform(low2[i], high2[i]) for i in range(len(low2))]).astype(np.float32)
    p0 = np.concatenate([p[:7], [0, 0]])
    my_franka.set_joint_positions(p0)
    v0 = np.concatenate([p[7:], [0, 0]])
    my_franka.set_joint_velocities(v0)
    s_data = []
    u_data = []

    for i in range(100):
        positions = my_franka.get_joint_positions()
        velocities = my_franka.get_joint_velocities()
        state = np.concatenate([positions[:7], velocities[:7]]).flatten()
        s_data.append(state)
        Y = enc_net1.ENC(torch.tensor(state))
        Control = net1(Y).squeeze(0)
        control = dec_net4.DEC(torch.cat((Control,torch.tensor(state)),dim=-1)).detach().numpy()
        u_data.append(control)
        control = np.concatenate([control, [0, 0]])
        actions = ArticulationAction(
                joint_positions=None,
                joint_velocities=control,
                joint_efforts=None,
            )
        articulation_controller.apply_action(actions)
        for j in range(20):
            my_world.step(render=False)
            #my_world.step(render=(j == 19))
        

    u_data = np.array(u_data)
    s_data = np.array(s_data)
    save_dir = Path("/home/nng/koopman_project/issac_code/franka")
    save_dir.mkdir(parents=True, exist_ok=True)

    np.save(save_dir / "u_data_franka3.npy", u_data)
    np.save(save_dir / "s_data_franka3.npy", s_data)


    import numpy as np
    import matplotlib.pyplot as plt
    #s_data = np.load('/home/nng/koopman_project/issac_code/franka/s_data_franka.npy')
    s_data = np.load(save_dir / "s_data_franka3.npy")
    s_data=s_data.reshape(-1,14)
    print(s_data[-1,:])


    save_dir = Path("/home/nng/koopman_project/issac_code/franka")
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(14, 1, figsize=(10, 14), sharex=True)
    for i in range(14):
        axes[i].plot(s_data[:, i], label='angles', color='blue')
        axes[i].set_title(f'Dimension {i+1}')
        axes[i].set_ylabel('Angle')
    axes[-1].set_xlabel('Time step')
    axes[0].legend()
    plt.tight_layout()
    plt.savefig(save_dir / "s_data_franka_plot.png", dpi=150)
    plt.close(fig)