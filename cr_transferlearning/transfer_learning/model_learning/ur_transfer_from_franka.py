import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from copy import copy
import argparse
import os
from torch.utils.tensorboard import SummaryWriter
import sys
sys.path.append("../utility/")
from scipy.integrate import odeint
import time
import franka as lka

dicts = torch.load("/control_transfer/A_to_B/Data/franka/unifiedur_transferlayer3_edim100_eloss1.pth", map_location=torch.device('cpu'))
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
dicts = torch.load("/control_transfer/A_to_B/Data/franka_to_ur/2unifiedur_transferlayer3_edim100_eloss1.pth", map_location=torch.device('cpu'))
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


Ktest_data1 = np.load("/robot_data/speed_traj_data/traj_train_file/Ktest_data_franka.npy")
Ktrain_data1 = np.load("/robot_data/speed_traj_data/traj_train_file/Ktrain_data_franka.npy")
Ktest_data2 = np.load("/robot_data/speed_traj_data/traj_train_file/Ktest_data_ur.npy")
Ktrain_data2 = np.load("/robot_data/speed_traj_data/traj_train_file/Ktrain_data_ur.npy")

def gaussian_init_(n_units, std=1):
    sampler = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([std / n_units]))
    Omega = sampler.sample((n_units, n_units))[..., 0]
    return Omega


class Network(nn.Module):
    def __init__(self, encode_layers, Nkoopman, u_dim):
        super(Network, self).__init__()
        Layers = OrderedDict()
        for layer_i in range(len(encode_layers) - 1):
            Layers["linear_{}".format(layer_i)] = nn.Linear(encode_layers[layer_i], encode_layers[layer_i + 1])
            if layer_i != len(encode_layers) - 2:
                Layers["relu_{}".format(layer_i)] = nn.ReLU()
        self.encode_net = nn.Sequential(Layers)
        self.Nkoopman = Nkoopman
        self.u_dim = u_dim
        self.lA = nn.Linear(Nkoopman, Nkoopman, bias=False)
        self.lA.weight.data = gaussian_init_(Nkoopman, std=1)
        U, _, V = torch.svd(self.lA.weight.data)
        self.lA.weight.data = torch.mm(U, V.t()) * 0.9
        self.lB = nn.Linear(u_dim, Nkoopman, bias=False)

    def encode(self, x):
        return torch.cat([x, self.encode_net(x)], dim=-1)

    def forward(self, x, u):
        return self.lA(x) + self.lB(u)


class Discriminator(nn.Module):
    def __init__(self, latent_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))


class ENC_net(nn.Module):
    def __init__(self, ENC_layers):
        super(ENC_net, self).__init__()
        ENCLayers = OrderedDict()
        for layer_i in range(len(ENC_layers) - 1):
            ENCLayers["linear_{}".format(layer_i)] = nn.Linear(ENC_layers[layer_i], ENC_layers[layer_i + 1])
            if layer_i != len(ENC_layers) - 2:
                ENCLayers["relu_{}".format(layer_i)] = nn.ReLU()
        self.ENC_net = nn.Sequential(ENCLayers)

    def ENC(self, x):
        return self.ENC_net(x)


class DEC_net(nn.Module):
    def __init__(self, DEC_layers):
        super(DEC_net, self).__init__()
        DECLayers = OrderedDict()
        for layer_i in range(len(DEC_layers) - 1):
            DECLayers["linear_{}".format(layer_i)] = nn.Linear(DEC_layers[layer_i], DEC_layers[layer_i + 1])
            if layer_i != len(DEC_layers) - 2:
                DECLayers["relu_{}".format(layer_i)] = nn.ReLU()
        self.DEC_net = nn.Sequential(DECLayers)

    def DEC(self, x):
        return self.DEC_net(x)

def Cyc_loss(X1,X2,enc_net1,enc_net2,enc_net4,enc_net5,dec_net1,dec_net2,dec_net4,dec_net5,mse_loss,primary_udim1,primary_udim2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X1 = torch.DoubleTensor(X1).to(device)
    X2 = torch.DoubleTensor(X2).to(device)
    x1_x2 = dec_net2.DEC(enc_net1.ENC(X1[:, :, primary_udim1:]))
    x2_x1 = dec_net1.DEC(enc_net2.ENC(X2[:, :, primary_udim2:]))
    u1_u2 = dec_net5.DEC(torch.cat([enc_net4.ENC(X1),x1_x2], dim=-1))
    u2_u1 = dec_net4.DEC(torch.cat([enc_net5.ENC(X2),x2_x1], dim=-1))
    x1_x2_x1 = dec_net1.DEC(enc_net2.ENC(x1_x2))
    x2_x1_x2 = dec_net2.DEC(enc_net1.ENC(x2_x1))
    u1_u2_u1 = dec_net4.DEC(torch.cat([enc_net5.ENC(torch.cat([u1_u2, x1_x2], dim=-1)), x1_x2_x1], dim=-1))
    u2_u1_u2 = dec_net5.DEC(torch.cat([enc_net4.ENC(torch.cat([u2_u1, x2_x1], dim=-1)), x2_x1_x2], dim=-1))
    X1_X1 = torch.cat([u1_u2_u1, x1_x2_x1], dim=-1)
    X2_X2 = torch.cat([u2_u1_u2, x2_x1_x2], dim=-1)
    cyc_loss = mse_loss(X1_X1, X1) + mse_loss(X2_X2, X2)
    return cyc_loss


def Klinear_loss(X2,net,dnet,enc_net2,enc_net5,dec_net2,dec_net5,mse_loss,gamma,Nstate,primary_udim2):
    steps, train_traj_num, NKoopman = X2.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X2 = torch.DoubleTensor(X2).to(device)
    X2_enc = enc_net2.ENC(X2[:, :, primary_udim2:])
    X2_current = net.encode(X2_enc[0, :, :])
    beta = 1.0
    beta_sum = 0.0
    Y_loss = torch.zeros(1, dtype=torch.float64).to(device)
    dec_Y_loss = torch.zeros(1, dtype=torch.float64).to(device)
    x_loss = torch.zeros(1, dtype=torch.float64).to(device)
    x_enc_loss = torch.zeros(1, dtype=torch.float64).to(device)
    u_rec_loss = torch.zeros(1, dtype=torch.float64).to(device)
    U_rec_loss = torch.zeros(1, dtype=torch.float64).to(device)
    x_enc_dec_loss = torch.zeros(1, dtype=torch.float64).to(device)
    Y_enc_loss = torch.zeros(1, dtype=torch.float64).to(device)
    for i in range(steps - 1):
        X2_current_old = X2_current
        X2_now = dec_net2.DEC(X2_current[:, :Nstate])
        U2_all = torch.cat((X2[i, :, :primary_udim2], X2_now), dim=-1)
        U2 = enc_net5.ENC(U2_all)
        beta_sum += beta
        u2 = torch.cat((U2, X2_now), dim=-1)
        U2_dec = dec_net5.DEC(u2)
        u_rec_loss += beta * mse_loss(U2_dec, X2[i, :, :primary_udim2])
        U2_dec_all = torch.cat((U2_dec, X2_now), dim=-1)
        U2_dec_enc = enc_net5.ENC(U2_dec_all)
        U_rec_loss += beta * mse_loss(U2, U2_dec_enc)

        X2_current = net.forward(X2_current, U2)
        Y_loss += beta * mse_loss(X2_current[:, :Nstate], X2_enc[i + 1, :, :])
        X2_current_old = torch.cat((X2_current_old[:,:20], X2_current[:,:20]), dim=-1)
        dec_U2 = dnet.DEC(X2_current_old)
        dec_Y_loss += beta * (mse_loss(dec_U2, U2))
        X2_next = dec_net2.DEC(X2_current[:, :Nstate])
        x_loss += beta * mse_loss(X2[i + 1, :, primary_udim2:], X2_next)
        X2_next_enc = enc_net2.ENC(X2_next)
        x_enc_loss += beta * mse_loss(X2_current[:, :Nstate], X2_next_enc)
        X2_next_enc_dec = dec_net2.DEC(X2_next_enc)
        x_enc_dec_loss += beta * mse_loss(X2_next_enc_dec, X2_next)
        X2_current_encoded = net.encode(X2_current[:, :Nstate])
        Y_enc_loss += beta * mse_loss(X2_current_encoded, X2_current)
        beta *= gamma
    Y_loss /= beta_sum
    dec_Y_loss /= beta_sum
    x_loss /= beta_sum
    x_enc_loss /= beta_sum
    u_rec_loss /= beta_sum
    U_rec_loss /= beta_sum
    x_enc_dec_loss /= beta_sum
    Y_enc_loss /= beta_sum
    x_loss *= 100
    loss = Y_loss + x_loss + x_enc_loss + u_rec_loss + U_rec_loss + x_enc_dec_loss + Y_enc_loss + dec_Y_loss
    return loss, Y_loss, x_loss, x_enc_loss, u_rec_loss, U_rec_loss ,x_enc_dec_loss, Y_enc_loss,dec_Y_loss


def gen_data(X1,X2,enc_net1,enc_net2,enc_net4,enc_net5,dec_net1,dec_net2,dec_net4,dec_net5,primary_udim1,primary_udim2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X1 = torch.DoubleTensor(X1).to(device)
    X2 = torch.DoubleTensor(X2).to(device)
    X1_enc = enc_net1.ENC(X1[:, :, primary_udim1:])
    X2_enc = enc_net2.ENC(X2[:, :, primary_udim2:])
    U1_enc = enc_net4.ENC(X1)
    U2_enc = enc_net5.ENC(X2)
    dataset1 = torch.cat((U1_enc,X1_enc), dim=-1).reshape(-1,30)
    dataset2 = torch.cat((U2_enc,X2_enc), dim=-1).reshape(-1,30)
    X1_enc_dec = dec_net2.DEC(X1_enc)
    u1_all = torch.cat([U1_enc, X1_enc_dec], dim=-1)
    U1_enc_dec = dec_net5.DEC(u1_all)
    dataset3 = torch.cat((U1_enc_dec, X1_enc_dec), dim=-1).reshape(-1,3*primary_udim2)
    dataset4 = X2.reshape(-1,3*primary_udim2)
    X2_enc_dec = dec_net1.DEC(X2_enc)
    u2_all = torch.cat([U2_enc, X2_enc_dec], dim=-1)
    U2_enc_dec = dec_net4.DEC(u2_all)
    dataset5 = torch.cat((U2_enc_dec, X2_enc_dec), dim=-1).reshape(-1,3*primary_udim1)
    dataset6 = X1.reshape(-1, 3*primary_udim1)
    return dataset1,dataset2,dataset3,dataset4,dataset5,dataset6

def discriminator_loss(X1,X2,enc_net1,enc_net2,enc_net4,enc_net5,dec_net1,dec_net2,dec_net4,dec_net5,discriminator1,discriminator2,discriminator3,criterion,primary_udim1,primary_udim2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset1,dataset2,dataset3,dataset4,dataset5,dataset6 = gen_data(X1,X2,enc_net1,enc_net2,enc_net4,enc_net5,dec_net1,dec_net2,dec_net4,dec_net5,primary_udim1,primary_udim2)
    y_system1 = torch.zeros((dataset1.shape[0],1), device=device).double()    # 系统1的标签，0表示来自系统1
    y_system2 = torch.ones((dataset1.shape[0],1), device=device).double()  # 系统2的标签，1表示来自系统2
    pred_system1 = discriminator1(dataset1)
    pred_system2 = discriminator1(dataset2)
    loss_d1 = criterion(pred_system1, y_system1)
    loss_d2 = criterion(pred_system2, y_system2)
    disloss1 = (loss_d1 + loss_d2)/2
    y_system3 = torch.zeros((dataset3.shape[0],1), device=device).double()    # 系统1的标签，0表示来自系统1
    y_system4 = torch.ones((dataset4.shape[0],1), device=device).double()  # 系统2的标签，1表示来自系统2
    pred_system3 = discriminator2(dataset3)
    pred_system4 = discriminator2(dataset4)
    loss_d1 = criterion(pred_system3, y_system3)
    loss_d2 = criterion(pred_system4, y_system4)
    disloss2 = (loss_d1 + loss_d2)/2
    y_system5 = torch.zeros((dataset5.shape[0],1), device=device).double()    # 系统1的标签，0表示来自系统1
    y_system6 = torch.ones((dataset6.shape[0],1), device=device).double()  # 系统2的标签，1表示来自系统2
    pred_system5 = discriminator3(dataset5)
    pred_system6 = discriminator3(dataset6)
    loss_d1 = criterion(pred_system5, y_system5)
    loss_d2 = criterion(pred_system6, y_system6)
    disloss3 = (loss_d1 + loss_d2)/2
    disloss = (disloss1 + disloss2 + disloss3) / 3
    return disloss

def generator_loss(X1, X2, enc_net1, enc_net2, enc_net4, enc_net5,dec_net1,dec_net2,dec_net4,dec_net5,discriminator1,discriminator2,discriminator3,criterion,primary_udim1,primary_udim2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset1,dataset2,dataset3,dataset4,dataset5,dataset6 = gen_data(X1,X2,enc_net1,enc_net2,enc_net4,enc_net5,dec_net1,dec_net2,dec_net4,dec_net5,primary_udim1,primary_udim2)
    y_system = torch.ones((dataset1.shape[0], 1), device=device).double()  # 系统2的标签，1表示来自系统2
    pred_system = discriminator1(dataset1)
    genloss1 = criterion(pred_system, y_system)
    y_system = torch.ones((dataset3.shape[0], 1), device=device).double()  # 系统2的标签，1表示来自系统2
    pred_system = discriminator2(dataset3)
    genloss2 = criterion(pred_system, y_system)
    y_system = torch.ones((dataset5.shape[0], 1), device=device).double()  # 系统2的标签，1表示来自系统2
    pred_system = discriminator3(dataset5)
    genloss3 = criterion(pred_system, y_system)
    genloss = (genloss1 + genloss2 + genloss3) / 3
    return genloss

def Eig_loss(net):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = net.lA.weight
    c = torch.linalg.eigvals(A).abs() - torch.ones(1, dtype=torch.float64).to(device)
    mask = c > 0
    loss = c[mask].sum()
    return loss


def train(env_name, train_steps=200000, suffix="", all_loss=0, \
          encode_dim=12, layer_depth=3, e_loss=1, gamma=0.5, Ktrain_samples=50000):
    np.random.seed(98)
    Kbatch_size = 128
    primary_udim2 = 6
    primary_sdim2 = 12
    layer_width = 256
    XENC_layers2 = [primary_sdim2] + [layer_width] * layer_depth + [common_sdim]
    enc_net2 = ENC_net(XENC_layers2)
    UENC_layers2 = [primary_udim2 + primary_sdim2] + [layer_width] * layer_depth + [common_udim]
    enc_net5 = ENC_net(UENC_layers2)
    DEC_layers2 = [common_sdim] + [layer_width] * layer_depth + [primary_sdim2]
    dec_net2 = DEC_net(DEC_layers2)
    DEC_layers5 = [common_udim + primary_sdim2] + [layer_width] * layer_depth + [primary_udim2]
    dec_net5 = DEC_net(DEC_layers5)
    discriminator1 = Discriminator(common_sdim + common_udim)
    discriminator2 = Discriminator(primary_sdim2 + primary_udim2)
    discriminator3 = Discriminator(primary_sdim1 + primary_udim1)
    learning_rate = 1e-3
    models = [net,dnet,enc_net1, enc_net2, enc_net4, enc_net5,dec_net1, dec_net2, dec_net4, dec_net5,discriminator1, discriminator2, discriminator3]
    for model in models:
        model.cuda().double()
    criterion = nn.BCELoss()
    mse_loss = nn.MSELoss()
    optimizer = torch.optim.Adam([
        {'params': enc_net2.parameters()},
        {'params': enc_net5.parameters()},
        {'params': dec_net2.parameters()},
        {'params': dec_net5.parameters()},
    ], lr=learning_rate)
    optimizer2 = torch.optim.Adam([
        {'params': discriminator1.parameters()},
        {'params': discriminator2.parameters()},
        {'params': discriminator3.parameters()},
    ], lr=5e-7)
    optimizer3 = torch.optim.Adam([
        {'params': enc_net2.parameters()},
        {'params': enc_net5.parameters()},
        {'params': dec_net2.parameters()},
        {'params': dec_net5.parameters()},
    ], lr=5e-5)
    for name, param in net.named_parameters():
        print("model:", name, param.requires_grad)
    # train
    eval_step = 1000
    best_loss = 1000.0
    logdir = "/home/ccr/project-koopman/control_transfer/A_to_B/Data/" + suffix + "/unified" + env_name + "layer{}_edim{}_eloss{}".format(
        layer_depth, encode_dim, e_loss)
    if not os.path.exists("/home/ccr/project-koopman/control_transfer/A_to_B/Data/" + suffix):
        os.makedirs("/home/ccr/project-koopman/control_transfer/A_to_B/Data/" + suffix)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    writer = SummaryWriter(log_dir=logdir)
    for i in range(train_steps):
        Ktrain_samples1 = Ktrain_data1.shape[1]
        Franka_indices = np.random.choice(Ktrain_samples1, size=Kbatch_size, replace=True)  # replace是否放回采样
        Ktrain_samples2 = Ktrain_data2.shape[1]
        UR_indices = np.random.choice(Ktrain_samples2, size=Kbatch_size, replace=True)  # replace是否放回采样
        X1 = Ktrain_data1[:, Franka_indices, :]
        X2 = Ktrain_data2[:, UR_indices, :]
        disloss = discriminator_loss(X1, X2, enc_net1, enc_net2, enc_net4, enc_net5, dec_net1, dec_net2, dec_net4,dec_net5, discriminator1, discriminator2, discriminator3, criterion, primary_udim1,primary_udim2)
        optimizer2.zero_grad()
        disloss.backward()
        optimizer2.step()
        genloss = generator_loss(X1, X2, enc_net1, enc_net2, enc_net4, enc_net5, dec_net1, dec_net2, dec_net4, dec_net5,discriminator1, discriminator2, discriminator3, criterion, primary_udim1,primary_udim2)
        optimizer3.zero_grad()
        genloss.backward()
        optimizer3.step()
        Kloss,Y_loss, x_loss, x_enc_loss, u_rec_loss, U_rec_loss ,x_enc_dec_loss, Y_enc_loss,dec_Y_loss = Klinear_loss(X2,net,dnet,enc_net2,enc_net5,dec_net2,dec_net5,mse_loss,gamma,Nstate,primary_udim2)
        Eloss = Eig_loss(net)
        cyc_loss = Cyc_loss(X1,X2,enc_net1,enc_net2,enc_net4,enc_net5,dec_net1,dec_net2,dec_net4,dec_net5,mse_loss,primary_udim1,primary_udim2)
        loss = Kloss + Eloss + cyc_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar('Train/Kloss', Kloss, i)
        writer.add_scalar('Train/x_loss', x_loss, i)
        writer.add_scalar('Train/dec_Y_loss', dec_Y_loss, i)
        writer.add_scalar('Train/cyc_loss', cyc_loss, i)
        writer.add_scalar('Train/disloss', disloss, i)
        writer.add_scalar('Train/genloss', genloss, i)
        if (i + 1) % eval_step == 0:
            # K loss
            with torch.no_grad():
                Kloss,Y_loss, x_loss, x_enc_loss, u_rec_loss, U_rec_loss ,x_enc_dec_loss, Y_enc_loss ,dec_Y_loss= Klinear_loss(Ktest_data2,net,dnet,enc_net2,enc_net5,dec_net2,dec_net5,mse_loss,gamma,Nstate,primary_udim2)
                Eloss = Eig_loss(net)
                loss = Kloss
                Kloss = Kloss.detach().cpu().numpy()
                Eloss = Eloss.detach().cpu().numpy()
                loss = loss.detach().cpu().numpy()
                writer.add_scalar('Eval/Kloss', Kloss, i)
                writer.add_scalar('Eval/cyc_loss', cyc_loss, i)
                writer.add_scalar('Eval/Eloss', Eloss, i)
                writer.add_scalar('Eval/best_loss', best_loss, i)
                if loss < best_loss:
                    best_loss = copy(Kloss)
                    torch.save({
                        'enc_net2_state_dict': enc_net2.state_dict(),
                        'enc_net5_state_dict': enc_net5.state_dict(),
                        'dec_net2_state_dict': dec_net2.state_dict(),
                        'dec_net5_state_dict': dec_net5.state_dict(),
                    }, logdir + ".pth")
                print("Step:{} Eval-loss{} K-loss:{}".format(i, loss, Kloss))
                # print("-------------END-------------")
        writer.add_scalar('Eval/best_loss', best_loss, i)
    print("END-best_loss{}".format(best_loss))


def main():
    train(args.env, suffix=args.suffix, encode_dim=args.encode_dim, \
          layer_depth=args.layer_depth, \
          e_loss=args.e_loss, gamma=args.gamma, \
          Ktrain_samples=args.K_train_samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="ur_transfer")
    parser.add_argument("--suffix", type=str, default="franka_to_ur3")
    parser.add_argument("--K_train_samples", type=int, default=50000)
    parser.add_argument("--e_loss", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.8)
    parser.add_argument("--encode_dim", type=int, default=100)
    parser.add_argument("--layer_depth", type=int, default=3)
    args = parser.parse_args()
    main()

