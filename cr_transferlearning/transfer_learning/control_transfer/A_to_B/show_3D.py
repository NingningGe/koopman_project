from bokeh.plotting import figure, output_file, show
from bokeh.layouts import gridplot
import numpy as np
import franka as lka
import torch

dicts = torch.load("/home/ccr/project-koopman/control_transfer/A_to_B/Data/franka/unifiedur_transferlayer3_edim100_eloss1.pth",map_location=torch.device('cpu'))
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
dicts = torch.load("/home/ccr/project-koopman/control_transfer/A_to_B/Data/franka_to_ur2/unifiedur_transferlayer3_edim100_eloss1.pth", map_location=torch.device('cpu'))
enc_net2_state_dict = dicts["enc_net2_state_dict"]
enc_net5_state_dict = dicts["enc_net5_state_dict"]
dec_net2_state_dict = dicts["dec_net2_state_dict"]
dec_net5_state_dict = dicts["dec_net5_state_dict"]
primary_udim2 = 6
primary_sdim2 = 12
XENC_layers2 = [primary_sdim2] + [layer_width2] * layer_depth + [common_sdim]
enc_net2 = lka.ENC_net(XENC_layers2)
UENC_layers2 = [primary_udim2 + primary_sdim2] + [layer_width2] * layer_depth + [common_udim]
enc_net5 = lka.ENC_net(UENC_layers2)
DEC_layers2 = [common_sdim] + [layer_width2] * layer_depth + [primary_sdim2]
dec_net2 = lka.DEC_net(DEC_layers2)
DEC_layers5 = [common_udim + primary_sdim2] + [layer_width2] * layer_depth + [primary_udim2]
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

Ktest_data1 = np.load("/home/ccr/project-koopman/robot_data/speed_traj_data/traj_train_file/Ktest_data_franka.npy")
Ktest_data2 = np.load("/home/ccr/project-koopman/robot_data/speed_traj_data/traj_train_file/Ktest_data_ur.npy")

traj1 = Ktest_data1.reshape(-1,3*primary_udim1)
traj2 = Ktest_data2.reshape(-1,3*primary_udim2)
X1 = torch.DoubleTensor(traj1)
X2 = torch.DoubleTensor(traj2)
X1_enc = enc_net1.ENC(X1[:, primary_udim1:])
X2_enc = enc_net2.ENC(X2[:, primary_udim2:])
U1_enc = enc_net4.ENC(X1)
U2_enc = enc_net5.ENC(X2)
dataset1 = torch.cat((U1_enc, X1_enc), dim=-1).reshape(-1, 30).detach().numpy()
dataset2 = torch.cat((U2_enc, X2_enc), dim=-1).reshape(-1, 30).detach().numpy()
X1_enc_dec = dec_net2.DEC(X1_enc)
u1_all = torch.cat([U1_enc, X1_enc_dec], dim=-1)
U1_enc_dec = dec_net5.DEC(u1_all)
dataset3 = torch.cat((U1_enc_dec, X1_enc_dec), dim=-1).reshape(-1, 3*primary_udim2).detach().numpy()
dataset4 = X2.reshape(-1, 3*primary_udim2).detach().numpy()
X2_enc_dec = dec_net1.DEC(X2_enc)
u2_all = torch.cat([U2_enc, X2_enc_dec], dim=-1)
U2_enc_dec = dec_net4.DEC(u2_all)
dataset5 = torch.cat((U2_enc_dec, X2_enc_dec), dim=-1).reshape(-1, 3*primary_udim1).detach().numpy()
dataset6 = X1.reshape(-1, 3*primary_udim1).detach().numpy()

# # 打印每个维度的最大值和最小值
# for i in range(dataset1.shape[1]):  # 遍历每个维度
#     dim_min = np.min(dataset1[:, i])  # 获取当前维度的最小值
#     dim_max = np.max(dataset1[:, i])  # 获取当前维度的最大值
#     print(f"维度 {i + 1}: 最小值 = {dim_min:.4f}, 最大值 = {dim_max:.4f}")

# p1 = np.array([0, -0.8, 0, -1.6, 0, 1.6,0,0,0,0,0,0,0,0])
# p = dec_net2.DEC(enc_net1.ENC(torch.DoubleTensor(p1)))
# print(p)
# p1 = np.array([0, -1.6, 0.8, -1.6, 0,0,0,0,0,0,0,0])
# p = dec_net1.DEC(enc_net2.ENC(torch.DoubleTensor(p1)))
# print(p)

import matplotlib.pyplot as plt

def plot_histograms(data):
    num_dims = data.shape[1]
    fig, axes = plt.subplots(num_dims, 1, figsize=(10, num_dims * 5))
    if num_dims == 1:
        axes = [axes]

    for i in range(num_dims):
        axes[i].hist(data[:, i], bins=50, color='blue', alpha=0.7)
        # axes[i].set_title(f'Distribution of Dimension {i + 1}')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

def plot_histograms1(data, ax=None):
    num_dims = data.shape[1]
    if ax is None:
        ax = plt.gca()  # Get current axis if not passed

    for i in range(num_dims):
        ax.hist(data[:, i], bins=50, alpha=0.7, label=f'Dimension {i+1}')

    ax.set_title('Franka Latent Data Distribution', fontsize=26)
    ax.set_xlabel('Value', fontsize=26)
    ax.set_ylabel('Frequency', fontsize=26)
    ax.tick_params(axis='both', labelsize=20)

def plot_histograms2(data, ax=None):
    num_dims = data.shape[1]
    if ax is None:
        ax = plt.gca()  # Get current axis if not passed

    for i in range(num_dims):
        ax.hist(data[:, i], bins=50, alpha=0.7, label=f'Dimension {i+1}')

    ax.set_title('UR5 Latent Data Distribution', fontsize=26)
    ax.set_xlabel('Value', fontsize=26)
    ax.set_ylabel('Frequency', fontsize=26)
    ax.tick_params(axis='both', labelsize=20)

def plot_histograms3(data, ax=None):
    num_dims = data.shape[1]
    if ax is None:
        ax = plt.gca()  # Get current axis if not passed

    for i in range(num_dims):
        ax.hist(data[:, i], bins=50, alpha=0.7, label=f'Dimension {i+1}')

    ax.set_title('UR5 Decode Data Distribution', fontsize=26)
    ax.set_xlabel('Value', fontsize=26)
    ax.set_ylabel('Frequency', fontsize=26)
    ax.tick_params(axis='both', labelsize=20)

def plot_histograms4(data, ax=None):
    num_dims = data.shape[1]
    if ax is None:
        ax = plt.gca()  # Get current axis if not passed

    for i in range(num_dims):
        ax.hist(data[:, i], bins=50, alpha=0.7, label=f'Dimension {i+1}')

    ax.set_title('UR5 Original Data Distribution', fontsize=26)
    ax.set_xlabel('Value', fontsize=26)
    ax.set_ylabel('Frequency', fontsize=26)
    ax.tick_params(axis='both', labelsize=20)

def plot_histograms5(data, ax=None):
    num_dims = data.shape[1]
    if ax is None:
        ax = plt.gca()  # Get current axis if not passed

    for i in range(num_dims):
        ax.hist(data[:, i], bins=50, alpha=0.7, label=f'Dimension {i+1}')

    ax.set_title('Franka Decode Data Distribution', fontsize=26)
    ax.set_xlabel('Value', fontsize=26)
    ax.set_ylabel('Frequency', fontsize=26)
    ax.tick_params(axis='both', labelsize=20)

def plot_histograms6(data, ax=None):
    num_dims = data.shape[1]
    if ax is None:
        ax = plt.gca()  # Get current axis if not passed

    for i in range(num_dims):
        ax.hist(data[:, i], bins=50, alpha=0.7, label=f'Dimension {i + 1}')

    ax.set_title('Franka Original Data Distribution', fontsize=26)
    ax.set_xlabel('Value', fontsize=26)
    ax.set_ylabel('Frequency', fontsize=26)
    ax.tick_params(axis='both', labelsize=20)

def plot_grouped_histograms():
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))  # 创建一个3x3的子图

    # 第一组：traj_data1, traj_data2, traj_data3
    plot_histograms1(dataset1, ax=axes[0, 0])
    plot_histograms2(dataset2, ax=axes[0, 1])

    plot_histograms3(dataset3, ax=axes[1, 0])
    plot_histograms4(dataset4, ax=axes[1, 1])

    plot_histograms5(dataset5, ax=axes[2, 0])
    plot_histograms6(dataset6, ax=axes[2, 1])

    # 调整布局
    plt.tight_layout()
    plt.show()



# 调用函数
plot_grouped_histograms()



# a=np.load('/home/ccr/下载/cross_embodiment_transfer-main/human_demonstrations/Reach/UR5e/JOINT_VELOCITY/20250618T201327-95bd91485e1843e4b560298c4f430253-100.npz')
# print(a['robot0_joint_pos_cos'].shape)
# a=np.load('/home/ccr/下载/cross_embodiment_transfer-main/human_demonstrations/Reach/Sawyer/JOINT_VELOCITY/20250613T172102-06268234c212459fb6f9de2e72d6a877-100.npz')
# print(a.files)
# print(a['action'].shape)
# print(a['target_to_robot0_eef_pos'].shape)