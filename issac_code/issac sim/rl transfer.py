#      ~/isaacsim/python.sh "/home/nng/koopman_project/issac_code/issac sim/rl transfer.py"

#2/3.5
control=3.5

if control==1:
    # ur
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
    model_path = "/home/nng/koopman_project/issac_code/tranfer_control/1/td3_continuous_action_20250620-164253.cleanrl_model"
    import torch
    actor_state_dict, qf1_state_dict, qf2_state_dict = torch.load(model_path)
    net1 = td3.Actor(8)
    net1.load_state_dict(actor_state_dict)
    import three_models_speed as lka
    #  #
    dicts = torch.load("/home/nng/koopman_project/issac_code/tranfer_control/unifiedur_transferlayer3_edim100_eloss1.pth",map_location=torch.device('cpu'))
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
    dicts = torch.load("/home/nng/koopman_project/issac_code/tranfer_control/2unifiedur_transferlayer3_edim100_eloss1.pth", map_location=torch.device('cpu'))
    enc_net2_state_dict = dicts["enc_net2_state_dict"]
    enc_net5_state_dict = dicts["enc_net5_state_dict"]
    dec_net2_state_dict = dicts["dec_net2_state_dict"]
    dec_net5_state_dict = dicts["dec_net5_state_dict"]
    primary_udim2 = 6
    primary_sdim2 = 12
    XENC_layers2 = [primary_sdim2] + [layer_width] * layer_depth + [common_sdim]
    enc_net2 = lka.ENC_net(XENC_layers2)
    UENC_layers2 = [primary_udim2 + primary_sdim2] + [layer_width] * layer_depth + [common_udim]
    enc_net5 = lka.ENC_net(UENC_layers2)
    DEC_layers2 = [common_sdim] + [layer_width] * layer_depth + [primary_sdim2]
    dec_net2 = lka.DEC_net(DEC_layers2)
    DEC_layers5 = [common_udim + primary_sdim2] + [layer_width] * layer_depth + [primary_udim2]
    dec_net5 = lka.DEC_net(DEC_layers5)
    net.cpu().double().load_state_dict(net_state_dict)
    dnet.cpu().double().load_state_dict(dnet_state_dict)
    enc_net1.cpu().double().load_state_dict(enc_net1_state_dict)
    enc_net4.cpu().double().load_state_dict(enc_net4_state_dict)
    dec_net1.cpu().double().load_state_dict(dec_net1_state_dict)
    dec_net4.cpu().double().load_state_dict(dec_net4_state_dict)
    enc_net2.cpu().double().load_state_dict(enc_net2_state_dict)
    enc_net5.cpu().double().load_state_dict(enc_net5_state_dict)
    dec_net2.cpu().double().load_state_dict(dec_net2_state_dict)
    dec_net5.cpu().double().load_state_dict(dec_net5_state_dict)

    np.random.seed(14)
    low2 = torch.tensor([-0.3, -1.1, -0.3, -1.9, -0.3, 1.3, -0.3, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4],dtype=torch.float32)

    high2 = torch.tensor([0.3, -0.5, 0.3, -1.3, 0.3, 1.9, 0.3, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],dtype=torch.float32)
    p1 = np.array([np.random.uniform(low2[i], high2[i]) for i in range(len(low2))]).astype(np.float32)
    p = dec_net2.DEC(enc_net1.ENC(torch.DoubleTensor(p1))).detach().numpy()
    p0 = np.concatenate([p[:6]])
    v0 = np.concatenate([p[6:]])
    my_ur.set_joint_positions(p0)
    my_ur.set_joint_velocities(v0)
    s_data = []
    u_data = []
    Y_data = []
    for i in range(100):
        positions = my_ur.get_joint_positions()
        velocities = my_ur.get_joint_velocities()
        state = np.concatenate([positions[:6], velocities[:6]]).flatten()
        s_data.append(state)
        Y = enc_net2.ENC(torch.DoubleTensor(state))
        Y_data.append(Y.detach().numpy())
        franka_state=dec_net1.DEC(Y)
        control_franka = net1(franka_state.float()).squeeze(0)
        control = dec_net5.DEC(torch.cat((enc_net4.ENC(torch.cat((control_franka,franka_state),dim=-1)),torch.DoubleTensor(state)),dim=-1)).detach().numpy().flatten()
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
    Y_data = np.array(Y_data)
    np.save('/home/nng/koopman_project/issac_code/issac sim/ur/u_data_ur.npy', u_data)
    np.save('/home/nng/koopman_project/issac_code/issac sim/ur/s_data_ur.npy', s_data)

    import numpy as np
    import matplotlib.pyplot as plt
    s_data = np.load('/home/nng/koopman_project/issac_code/issac sim/ur/s_data_ur.npy')
    s_data=s_data.reshape(-1,12)
    print(s_data[-1,:])
    fig, axes = plt.subplots(12, 1, figsize=(10, 12), sharex=True)
    for i in range(12):
        axes[i].plot(s_data[:, i], label='angles', color='blue')
        axes[i].set_title(f'Dimension {i+1}')
        axes[i].set_ylabel('Angle')
    axes[-1].set_xlabel('Time step')
    axes[0].legend()
    plt.tight_layout()

    fig, axes = plt.subplots(20, 1, figsize=(10, 20), sharex=True)
    for i in range(20):
        axes[i].plot(Y_data[:, i], label='angles', color='blue')
        axes[i].set_title(f'Dimension {i+1}')
        axes[i].set_ylabel('Angle')
    axes[-1].set_xlabel('Time step')
    axes[0].legend()
    plt.tight_layout()
    plt.show()

elif control==2:
    # franka
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
    model_path = "/home/nng/koopman_project/issac_code/issac sim/traj_speed_file/td3_continuous_action2.cleanrl_model"
    import torch
    actor_state_dict, qf1_state_dict, qf2_state_dict = torch.load(model_path)
    net1 = td3.Actor(8)
    net1.load_state_dict(actor_state_dict)
    import three_models_speed as lka
    dicts = torch.load("/home/nng/koopman_project/issac_code/issac sim/traj_speed_file/unifiedsingle_trainlayer3_edim100_eloss1.pth",map_location=torch.device('cpu'))
    net_state_dict = dicts["net_state_dict"]
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
    layers = [in_dim] + [layer_width] * layer_depth + [encode_dim]
    Nkoopman = in_dim + encode_dim
    net = lka.Network(layers, Nkoopman, u_dim)
    dicts = torch.load("/home/nng/koopman_project/issac_code/issac sim/traj_speed_file/unifiedtwo_transferlayer3_edim100_eloss1.pth",map_location=torch.device('cpu'))
    enc_net2_state_dict = dicts["enc_net2_state_dict"]
    enc_net5_state_dict = dicts["enc_net5_state_dict"]
    dec_net2_state_dict = dicts["dec_net2_state_dict"]
    dec_net5_state_dict = dicts["dec_net5_state_dict"]
    primary_udim2 = 6
    primary_sdim2 = 12
    XENC_layers2 = [primary_sdim2] + [layer_width] * layer_depth + [common_sdim]
    enc_net2 = lka.ENC_net(XENC_layers2)
    UENC_layers2 = [primary_udim2 + primary_sdim2] + [layer_width] * layer_depth + [common_udim]
    enc_net5 = lka.ENC_net(UENC_layers2)
    DEC_layers2 = [common_sdim] + [layer_width] * layer_depth + [primary_sdim2]
    dec_net2 = lka.DEC_net(DEC_layers2)
    DEC_layers5 = [common_udim + primary_sdim2] + [layer_width] * layer_depth + [primary_udim2]
    dec_net5 = lka.DEC_net(DEC_layers5)
    net = lka.Network(layers, Nkoopman, u_dim)
    net.cpu().double().load_state_dict(net_state_dict)
    enc_net1.cpu().double().load_state_dict(enc_net1_state_dict)
    enc_net4.cpu().double().load_state_dict(enc_net4_state_dict)
    dec_net1.cpu().double().load_state_dict(dec_net1_state_dict)
    dec_net4.cpu().double().load_state_dict(dec_net4_state_dict)
    enc_net2.cpu().double().load_state_dict(enc_net2_state_dict)
    enc_net5.cpu().double().load_state_dict(enc_net5_state_dict)
    dec_net2.cpu().double().load_state_dict(dec_net2_state_dict)
    dec_net5.cpu().double().load_state_dict(dec_net5_state_dict)

    p1 = np.array([0, -1.6, 0.8, -1.6, 0,0,0,0,0,0,0,0])
    point = dec_net1.DEC(enc_net2.ENC(torch.DoubleTensor(p1))).detach().numpy()
    point = np.tile(point, (100, 1))
#SPEED8 24
#FORCE6 12
    np.random.seed(8)#3 6 8!! 12 24
    low2 = torch.tensor([-0.3, -1.9, 0.5, -1.9, -0.3, -0.3, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4],dtype=torch.float32)

    high2 = torch.tensor([0.3, -1.3, 1.1, -1.3, 0.3,  0.3, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],dtype=torch.float32)
    p1 = np.array([np.random.uniform(low2[i], high2[i]) for i in range(len(low2))]).astype(np.float32)
    p = dec_net1.DEC(enc_net2.ENC(torch.DoubleTensor(p1))).detach().numpy()
    p0 = np.concatenate([p[:7], [0, 0]])
    v0 = np.concatenate([p[7:], [0, 0]])
    my_franka.set_joint_positions(p0)
    my_franka.set_joint_velocities(v0)
    s_data = []
    u_data = []
    for i in range(100):
        positions = my_franka.get_joint_positions()
        velocities = my_franka.get_joint_velocities()
        state = np.concatenate([positions[:7], velocities[:7]]).flatten()
        s_data.append(state)
        franka_state=dec_net2.DEC(enc_net1.ENC(torch.DoubleTensor(state)))
        control_franka = net1(franka_state.float()).squeeze(0)
        control = dec_net4.DEC(torch.cat((enc_net5.ENC(torch.cat((control_franka,franka_state),dim=-1)),torch.DoubleTensor(state)),dim=-1)).detach().numpy().flatten()
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
    np.save('/home/nng/koopman_project/issac_code/issac sim/ur/u_data_franka.npy', u_data)
    np.save('/home/nng/koopman_project/issac_code/issac sim/ur/s_data_franka.npy', s_data)

    import numpy as np
    import matplotlib.pyplot as plt
    s_data = np.load('/home/nng/koopman_project/issac_code/issac sim/ur/s_data_franka.npy')
    s_data=s_data.reshape(-1,14)
    plt.figure(figsize=(15, 12))

    tick_label_size = 18      # 坐标轴刻度标签大小
    axis_label_size = 16      # 坐标轴标签大小
    legend_fontsize = 28      # 图例字体大小

    for i in range(7):
        plt.subplot(7, 1, i + 1)

        # 绘制实际轨迹
        plt.plot(s_data[:, i], color='blue', linewidth=1.5, label='       ')
        
        # 绘制目标直线
        plt.plot(point[:, i], color='red', linewidth=1.5, linestyle='--', label='       ')

        plt.grid(True, alpha=0.3)

        # 获取当前坐标轴
        ax = plt.gca()
        
        # 设置刻度字体大小
        plt.tick_params(axis='y', labelsize=tick_label_size)
        plt.tick_params(axis='x', labelsize=tick_label_size)

        if i == 0:
            ax.legend(fontsize=legend_fontsize, loc='upper right')

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()


elif control==3:
    # ur
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
    #model_path = "/home/nng/koopman_project/issac_code/tranfer_control/Franka_and_ur.cleanrl_model"
    model_path = "/home/nng/koopman_project/issac_code/tranfer_control/Franka_and_Ur.cleanrl_model"
    import torch
    actor_state_dict, qf1_state_dict, qf2_state_dict = torch.load(model_path)
    net1 = td3.Actor(8)
    net1.load_state_dict(actor_state_dict)
    import three_models_speed as lka
    dicts = torch.load("/home/nng/koopman_project/issac_code/tranfer_control/unifiedur_transferlayer3_edim100_eloss1.pth",map_location=torch.device('cpu'))
    
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
    dicts = torch.load("/home/nng/koopman_project/issac_code/tranfer_control/2unifiedur_transferlayer3_edim100_eloss1.pth", map_location=torch.device('cpu'))
    enc_net2_state_dict = dicts["enc_net2_state_dict"]
    enc_net5_state_dict = dicts["enc_net5_state_dict"]
    dec_net2_state_dict = dicts["dec_net2_state_dict"]
    dec_net5_state_dict = dicts["dec_net5_state_dict"]
    primary_udim2 = 6
    primary_sdim2 = 12
    XENC_layers2 = [primary_sdim2] + [layer_width] * layer_depth + [common_sdim]
    enc_net2 = lka.ENC_net(XENC_layers2)
    UENC_layers2 = [primary_udim2 + primary_sdim2] + [layer_width] * layer_depth + [common_udim]
    enc_net5 = lka.ENC_net(UENC_layers2)
    DEC_layers2 = [common_sdim] + [layer_width] * layer_depth + [primary_sdim2]
    dec_net2 = lka.DEC_net(DEC_layers2)
    DEC_layers5 = [common_udim + primary_sdim2] + [layer_width] * layer_depth + [primary_udim2]
    dec_net5 = lka.DEC_net(DEC_layers5)
    net.cpu().double().load_state_dict(net_state_dict)
    dnet.cpu().double().load_state_dict(dnet_state_dict)
    enc_net1.cpu().double().load_state_dict(enc_net1_state_dict)
    enc_net4.cpu().double().load_state_dict(enc_net4_state_dict)
    dec_net1.cpu().double().load_state_dict(dec_net1_state_dict)
    dec_net4.cpu().double().load_state_dict(dec_net4_state_dict)
    enc_net2.cpu().double().load_state_dict(enc_net2_state_dict)
    enc_net5.cpu().double().load_state_dict(enc_net5_state_dict)
    dec_net2.cpu().double().load_state_dict(dec_net2_state_dict)
    dec_net5.cpu().double().load_state_dict(dec_net5_state_dict)

    np.random.seed(18)#11
    # low2 = torch.tensor([-0.3, -1.1, -0.3, -1.9, -0.3, 1.3, -0.3, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4],dtype=torch.float32)

    # high2 = torch.tensor([0.3, -0.5, 0.3, -1.3, 0.3, 1.9, 0.3, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],dtype=torch.float32)
    # p1 = np.array([np.random.uniform(low2[i], high2[i]) for i in range(len(low2))]).astype(np.float32)
    # p = dec_net2.DEC(enc_net1.ENC(torch.DoubleTensor(p1))).detach().numpy()
    
    low1 = torch.tensor([-0.3, -1.9, 0.5, -1.9, -0.3, -0.3, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4],
                   dtype=torch.float32)
    high1 = torch.tensor([0.3, -1.3, 1.1, -1.3, 0.3,  0.3, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
                    dtype=torch.float32)
    p = np.array([np.random.uniform(low1[i], high1[i]) for i in range(len(low1))]).astype(np.float32)
    p0 = np.concatenate([p[:6]])
    v0 = np.concatenate([p[6:]])
    my_ur.set_joint_positions(p0)
    my_ur.set_joint_velocities(v0)
    s_data = []
    u_data = []
    Y_data = []
    for i in range(100):
        positions = my_ur.get_joint_positions()
        velocities = my_ur.get_joint_velocities()
        state = np.concatenate([positions[:6], velocities[:6]]).flatten()
        s_data.append(state)
        Y = enc_net2.ENC(torch.DoubleTensor(state))
        Y_data.append(Y.detach().numpy())
        Control_franka = net1(Y.float()).squeeze(0)
        control = dec_net5.DEC(torch.cat((Control_franka,torch.DoubleTensor(state)),dim=-1)).detach().numpy().flatten()
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
    Y_data = np.array(Y_data)
    np.save('/home/nng/koopman_project/issac_code/issac sim/ur/u_data_ur.npy', u_data)
    np.save('/home/nng/koopman_project/issac_code/issac sim/ur/s_data_ur.npy', s_data)

    import numpy as np
    import matplotlib.pyplot as plt
    s_data = np.load('/home/nng/koopman_project/issac_code/issac sim/ur/s_data_ur.npy')
    s_data=s_data.reshape(-1,12)
    print(s_data[-1,:])
    fig, axes = plt.subplots(12, 1, figsize=(10, 12), sharex=True)
    for i in range(12):
        axes[i].plot(s_data[:, i], label='angles', color='blue')
        axes[i].set_title(f'Dimension {i+1}')
        axes[i].set_ylabel('Angle')
    axes[-1].set_xlabel('Time step')
    axes[0].legend()
    plt.tight_layout()

    fig, axes = plt.subplots(20, 1, figsize=(10, 20), sharex=True)
    for i in range(20):
        axes[i].plot(Y_data[:, i], label='angles', color='blue')
        axes[i].set_title(f'Dimension {i+1}')
        axes[i].set_ylabel('Angle')
    axes[-1].set_xlabel('Time step')
    axes[0].legend()
    plt.tight_layout()
    plt.show()


elif control==3.5:
    # ur
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
    model_path = "/home/nng/koopman_project/issac_code/tranfer_control/2/td3_continuous_action_20250620-165810.cleanrl_model"
    import torch
    actor_state_dict, qf1_state_dict, qf2_state_dict = torch.load(model_path)
    net1 = td3.Actor(8)
    net1.load_state_dict(actor_state_dict)
    import three_models_speed as lka
    dicts = torch.load("/home/nng/koopman_project/issac_code/tranfer_control/unifiedur_transferlayer3_edim100_eloss1.pth",map_location=torch.device('cpu'))
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
    dicts = torch.load("/home/nng/koopman_project/issac_code/tranfer_control/2unifiedur_transferlayer3_edim100_eloss1.pth", map_location=torch.device('cpu'))
    enc_net2_state_dict = dicts["enc_net2_state_dict"]
    enc_net5_state_dict = dicts["enc_net5_state_dict"]
    dec_net2_state_dict = dicts["dec_net2_state_dict"]
    dec_net5_state_dict = dicts["dec_net5_state_dict"]
    primary_udim2 = 6
    primary_sdim2 = 12
    XENC_layers2 = [primary_sdim2] + [layer_width] * layer_depth + [common_sdim]
    enc_net2 = lka.ENC_net(XENC_layers2)
    UENC_layers2 = [primary_udim2 + primary_sdim2] + [layer_width] * layer_depth + [common_udim]
    enc_net5 = lka.ENC_net(UENC_layers2)
    DEC_layers2 = [common_sdim] + [layer_width] * layer_depth + [primary_sdim2]
    dec_net2 = lka.DEC_net(DEC_layers2)
    DEC_layers5 = [common_udim + primary_sdim2] + [layer_width] * layer_depth + [primary_udim2]
    dec_net5 = lka.DEC_net(DEC_layers5)
    net.cpu().double().load_state_dict(net_state_dict)
    dnet.cpu().double().load_state_dict(dnet_state_dict)
    enc_net1.cpu().double().load_state_dict(enc_net1_state_dict)
    enc_net4.cpu().double().load_state_dict(enc_net4_state_dict)
    dec_net1.cpu().double().load_state_dict(dec_net1_state_dict)
    dec_net4.cpu().double().load_state_dict(dec_net4_state_dict)
    enc_net2.cpu().double().load_state_dict(enc_net2_state_dict)
    enc_net5.cpu().double().load_state_dict(enc_net5_state_dict)
    dec_net2.cpu().double().load_state_dict(dec_net2_state_dict)
    dec_net5.cpu().double().load_state_dict(dec_net5_state_dict)


    np.random.seed(53)#效果最好
    #np.random.seed(11)#beginning11 35
    #np.random.seed(60)#second53 60
    # p1 = np.array([[0, -0.8, 0, -1.6, 0, 1.6, 0, 0, 0, 0, 0, 0, 0, 0]])
    # point = dec_net2.DEC(enc_net1.ENC(torch.DoubleTensor(p1))).detach().numpy()
    # print(point)
    # point = np.tile(point, (300, 1))

    point = np.array([-0.04857543, -1.63611, 0.7875574, -1.621743, 0.02078001, -0.03624132])
    # point = np.array([-0.04357543, -1.61896137,  0.7875574, -1.61896377,  0.03078001, -0.03624132])
    point = np.tile(point, (300, 1))
    low1 = torch.tensor([-0.3, -1.9, 0.5, -1.9, -0.3, -0.3, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4],
                   dtype=torch.float32)
    high1 = torch.tensor([0.3, -1.3, 1.1, -1.3, 0.3,  0.3, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
                    dtype=torch.float32)
    p = np.array([np.random.uniform(low1[i], high1[i]) for i in range(len(low1))]).astype(np.float32)
    p0 = np.concatenate([p[:6]])
    v0 = np.concatenate([p[6:]])
    my_ur.set_joint_positions(p0)
    my_ur.set_joint_velocities(v0)
    s_data = []
    u_data = []
    Y_data = []
    for i in range(300):
        positions = my_ur.get_joint_positions()
        velocities = my_ur.get_joint_velocities()
        state = np.concatenate([positions[:6], velocities[:6]]).flatten()
        s_data.append(state)
        Y = enc_net2.ENC(torch.DoubleTensor(state))
        Y_data.append(Y.detach().numpy())
        Control_franka = net1(Y.float()).squeeze(0)
        control = dec_net5.DEC(torch.cat((Control_franka,torch.DoubleTensor(state)),dim=-1)).detach().numpy().flatten()
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
    Y_data = np.array(Y_data)
    np.save('/home/nng/koopman_project/issac_code/issac sim/ur/u_data_ur.npy', u_data)
    np.save('/home/nng/koopman_project/issac_code/issac sim/ur/s_data_ur.npy', s_data)

    import numpy as np
    import matplotlib.pyplot as plt

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif'] # 学术标准字体
    plt.rcParams['axes.unicode_minus'] = False # 正常显示负号

    s_data = np.load('/home/nng/koopman_project/issac_code/issac sim/ur/s_data_ur.npy')
    s_data = s_data.reshape(-1, 12)
    
    # 设置画布大小
    plt.figure(figsize=(14, 12))

    tick_label_size = 16      # 坐标轴刻度数字大小
    axis_label_size = 18      # 坐标轴标签大小
    legend_fontsize = 20      # 图例字体大小

    for i in range(6):
        plt.subplot(6, 1, i + 1)

        # 绘制实际轨迹
        plt.plot(s_data[:, i], color='#1f77b4', linewidth=2.5, label='Actual Trajectory')
        
        # 绘制目标直线
        plt.plot(point[:, i], color='#d62728', linewidth=2.5, linestyle='--', label='Target Position')

        # 添加网格线
        plt.grid(True, linestyle='--', alpha=0.6)

        # 设置纵坐标标签 (英文 + 物理单位)
        plt.ylabel(f'Joint {i+1}\n(rad)', fontsize=axis_label_size)

        # 获取当前坐标轴
        ax = plt.gca()
        
        # 设置刻度字体大小
        plt.tick_params(axis='y', labelsize=tick_label_size)
        plt.tick_params(axis='x', labelsize=tick_label_size)

        # 设置横坐标：只在最底部显示 Time Steps
        if i == 5:
            plt.xlabel('Time Steps', fontsize=axis_label_size)
        else:
            plt.setp(ax.get_xticklabels(), visible=False) # 隐藏上方子图的横坐标数字

        # 只在第一张图的最上方显示一次图例
        if i == 0:
            ax.legend(fontsize=legend_fontsize, loc='upper right', bbox_to_anchor=(1.0, 1.4), ncol=2)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.15) 
    
    # 保存高清 PNG 与 PDF
    save_path = '/home/nng/koopman_project/issac_code/issac sim/ur/Zero_Shot_Transfer_Pro.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
    
    print(f"[SUCCESS] 高清英文版轨迹追踪图已成功保存至: {save_path}")
    plt.close() # 释放内存