import numpy as np
import three_models_robots as lka
import torch

dicts = torch.load("/home/ccr/project-koopman/unified_model/unified_model_robots/Data/unified_three_robots/unifiedFRANKAlayer3_edim100_eloss1.pth", map_location=torch.device('cpu'))
net_state_dict = dicts["net_state_dict"]
enc_net1_state_dict = dicts["enc_net1_state_dict"]
enc_net2_state_dict = dicts["enc_net2_state_dict"]
enc_net3_state_dict = dicts["enc_net3_state_dict"]
enc_net4_state_dict = dicts["enc_net4_state_dict"]
enc_net5_state_dict = dicts["enc_net5_state_dict"]
enc_net6_state_dict = dicts["enc_net6_state_dict"]
dec_net1_state_dict = dicts["dec_net1_state_dict"]
dec_net2_state_dict = dicts["dec_net2_state_dict"]
dec_net3_state_dict = dicts["dec_net3_state_dict"]
dec_net4_state_dict = dicts["dec_net4_state_dict"]
dec_net5_state_dict = dicts["dec_net5_state_dict"]
dec_net6_state_dict = dicts["dec_net6_state_dict"]
primary_udim1 = 7
primary_udim2 = 6
primary_udim3 = 5
primary_sdim1 = 14
primary_sdim2 = 12
primary_sdim3 = 10
common_sdim = 20
common_udim = 10
weight_sdim = 14
in_dim = common_sdim
u_dim = common_udim
Nstate = in_dim
layer_width = 128
layer_width2 = 256
layer_depth = 3
encode_dim = 100
XENC_layers1 = [primary_sdim1] + [layer_width] * layer_depth + [common_sdim]
enc_net1 = lka.ENC_net(XENC_layers1)
XENC_layers2 = [primary_sdim2] + [layer_width] * layer_depth + [common_sdim]
enc_net2 = lka.ENC_net(XENC_layers2)
XENC_layers3 = [primary_sdim3] + [layer_width] * layer_depth + [common_sdim]
enc_net3 = lka.ENC_net(XENC_layers3)
UENC_layers1 = [primary_udim1 + primary_sdim1] + [layer_width] * layer_depth + [common_udim]
enc_net4 = lka.ENC_net(UENC_layers1)
UENC_layers2 = [primary_udim2 + primary_sdim2] + [layer_width] * layer_depth + [common_udim]
enc_net5 = lka.ENC_net(UENC_layers2)
UENC_layers3 = [primary_udim3 + primary_sdim3] + [layer_width] * layer_depth + [common_udim]
enc_net6 = lka.ENC_net(UENC_layers3)
DEC_layers1 = [common_sdim] + [layer_width] * layer_depth + [primary_sdim1]
dec_net1 = lka.DEC_net(DEC_layers1)
DEC_layers2 = [common_sdim] + [layer_width] * layer_depth + [primary_sdim2]
dec_net2 = lka.DEC_net(DEC_layers2)
DEC_layers3 = [common_sdim] + [layer_width] * layer_depth + [primary_sdim3]
dec_net3 = lka.DEC_net(DEC_layers3)
DEC_layers4 = [common_udim + primary_sdim1] + [layer_width] * layer_depth + [primary_udim1]
dec_net4 = lka.DEC_net(DEC_layers4)
DEC_layers5 = [common_udim + primary_sdim2] + [layer_width] * layer_depth + [primary_udim2]
dec_net5 = lka.DEC_net(DEC_layers5)
DEC_layers6 = [common_udim + primary_sdim3] + [layer_width] * layer_depth + [primary_udim3]
dec_net6 = lka.DEC_net(DEC_layers6)
layers = [in_dim] + [layer_width2] * layer_depth + [encode_dim]
Nkoopman = in_dim + encode_dim
net = lka.Network(layers, Nkoopman, u_dim)
net.cpu().double().load_state_dict(net_state_dict)
enc_net1.cpu().double().load_state_dict(enc_net1_state_dict)
enc_net2.cpu().double().load_state_dict(enc_net2_state_dict)
enc_net3.cpu().double().load_state_dict(enc_net3_state_dict)
enc_net4.cpu().double().load_state_dict(enc_net4_state_dict)
enc_net5.cpu().double().load_state_dict(enc_net5_state_dict)
enc_net6.cpu().double().load_state_dict(enc_net6_state_dict)
dec_net1.cpu().double().load_state_dict(dec_net1_state_dict)
dec_net2.cpu().double().load_state_dict(dec_net2_state_dict)
dec_net3.cpu().double().load_state_dict(dec_net3_state_dict)
dec_net4.cpu().double().load_state_dict(dec_net4_state_dict)
dec_net5.cpu().double().load_state_dict(dec_net5_state_dict)
dec_net6.cpu().double().load_state_dict(dec_net6_state_dict)

traj_data1 = np.load("/home/ccr/project-koopman/robot_data/speed_traj_data/traj_train_file/Ktest_data_franka.npy")
traj_data2 = np.load("/home/ccr/project-koopman/robot_data/speed_traj_data/traj_train_file/Ktest_data_ur.npy")
traj_data3 = np.load("/home/ccr/project-koopman/robot_data/speed_traj_data/traj_train_file/Ktest_data_dofbot.npy")
traj_data1 = torch.DoubleTensor(traj_data1.reshape(-1,21))
X1_enc = enc_net1.ENC(traj_data1[:, 7:])
X1_current = net.encode(X1_enc)
U1_enc = enc_net4.ENC(traj_data1)
traj_data2 = torch.DoubleTensor(traj_data2.reshape(-1,18))
X2_enc = enc_net2.ENC(traj_data2[:, 6:])
X2_current = net.encode(X2_enc)
U2_enc = enc_net5.ENC(traj_data2)
traj_data3 = torch.DoubleTensor(traj_data3.reshape(-1,15))
X3_enc = enc_net3.ENC(traj_data3[:, 5:])
X3_current = net.encode(X3_enc)
U3_enc = enc_net6.ENC(traj_data3)

import matplotlib.pyplot as plt

def plot_histograms(data, ax=None,title=''):
    num_dims = data.shape[1]
    if ax is None:
        ax = plt.gca()  # Get current axis if not passed

    for i in range(num_dims):
        ax.hist(data[:, i], bins=50, alpha=0.7, label=f'Dimension {i+1}')

    ax.set_title(f'{title}')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')

# 画图的函数，按组展示
def plot_grouped_histograms():
    fig, axes = plt.subplots(4, 3, figsize=(15, 15))

    plot_histograms(traj_data1[:, 7:14], ax=axes[0, 0], title='Franka Orign State')
    plot_histograms(traj_data2[:, 6:12], ax=axes[0, 1], title='UR5 Orign State')
    plot_histograms(traj_data3[:, 5:10], ax=axes[0, 2], title='Dofbot Orign State')

    plot_histograms(X1_enc.detach().numpy(), ax=axes[1, 0], title='Franka Latent State')
    plot_histograms(X2_enc.detach().numpy(), ax=axes[1, 1], title='UR5 Latent State')
    plot_histograms(X3_enc.detach().numpy(), ax=axes[1, 2], title='Dofbot Latent State')


    plot_histograms(traj_data1[:, 14:], ax=axes[2, 0], title='Franka Orign Action')
    plot_histograms(traj_data2[:, 12:], ax=axes[2, 1], title='UR5 Orign Action')
    plot_histograms(traj_data3[:, 10:], ax=axes[2, 2], title='Dofbot Orign Action')

    plot_histograms(U1_enc.detach().numpy(), ax=axes[3, 0], title='Franka Latent Action')
    plot_histograms(U2_enc.detach().numpy(), ax=axes[3, 1], title='UR5 Latent Action')
    plot_histograms(U3_enc.detach().numpy(), ax=axes[3, 2], title='Dofbot Latent Action')

    # 调整布局
    plt.tight_layout()
    plt.show()

# 调用函数
plot_grouped_histograms()


