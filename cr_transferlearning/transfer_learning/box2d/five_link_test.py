import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import gym
import matplotlib.pyplot as plt
import random
from collections import OrderedDict
from copy import copy
import argparse
import os
from torch.utils.tensorboard import SummaryWriter
import sys

sys.path.append("../utility/")
from scipy.integrate import odeint
import time

Five_train_data = np.load('/home/ccr/robot/train-koopman/npy_file/five_link/Ktrain_data_primary.npy')
Five_test_data = np.load('/home/ccr/robot/train-koopman/npy_file/five_link/Ktest_data_primary.npy')
Three_train_data = np.load('/home/ccr/robot/train-koopman/npy_file/three_link/Ktrain_data_primary.npy')
Three_test_data = np.load('/home/ccr/robot/train-koopman/npy_file/three_link/Ktest_data_primary.npy')
Two_train_data = np.load('/home/ccr/robot/train-koopman/npy_file/two_link/Ktest_data_primary.npy')
Two_test_data = np.load('/home/ccr/robot/train-koopman/npy_file/two_link/Ktrain_data_primary.npy')


# define network
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

    def encode(self, x, weight):
        x_all = torch.cat([x, weight], axis=-1)
        return torch.cat([x, self.encode_net(x_all)], axis=-1)

    def forward(self, x, u):
        return self.lA(x) + self.lB(u)


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

def l1_regularization(model, lambda_l1):
    l1_norm = sum(p.abs().sum() for p in model.parameters())  # L1 正则化
    return lambda_l1 * l1_norm

def off_diagonal_l1_loss(A):
    # 创建一个掩码，掩盖对角线元素
    mask = 1 - torch.eye(A.size(0), device=A.device)
    off_diag_elements = A * mask  # 选择非对角线元素

    # 计算非对角线元素的 L1 norm
    l1_loss = off_diag_elements.abs().sum()  # L1 loss: 求绝对值的和
    return l1_loss

# loss function
def Klinear_loss(X1, net, enc_net1, enc_net4, dec_net1, v1, v4, mse_loss, gamma, Nstate):
    steps, train_traj_num, NKoopman = X1.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X1 = torch.DoubleTensor(X1).to(device)
    X1_enc = enc_net1.ENC(X1[:, :, 5:])
    X1_dec = dec_net1.DEC(X1_enc)
    v1_expanded = v1.to(device).unsqueeze(0).expand(X1.shape[1], 10).double()
    X1_current = net.encode(X1_enc[0, :, :], v1_expanded)
    beta = 1.0
    beta_sum = 0.0
    loss = torch.zeros(1, dtype=torch.float64).to(device)
    loss1 = torch.zeros(1, dtype=torch.float64).to(device)
    reg_loss = torch.zeros(1, dtype=torch.float64).to(device)
    loss1 += (mse_loss(X1[:, :, 5:], X1_dec[:, :, :]))
    for i in range(steps - 1):
        U1_all = torch.cat((X1[i, :, :5],X1_current[:, :Nstate],v4.to(device).unsqueeze(0).expand(X1.shape[1], 5)),dim = -1)
        U1_enc = enc_net4.ENC(U1_all)
        X1_current = net.forward(X1_current, U1_enc)
        X1_current_dec = dec_net1.DEC(X1_current[:, :Nstate])
        X1_current_enc = enc_net1.ENC(X1_current_dec)
        beta_sum += beta
        loss += beta * (mse_loss(X1_current[:, :Nstate], X1_current_enc))
        loss += beta * (mse_loss(X1_current[:, :Nstate], X1_enc[i + 1, :, :]))
        X1_next = dec_net1.DEC(X1_current[:, :Nstate])
        loss += beta * (mse_loss(X1[i + 1, :, 5:10], X1_next[:, :5]))
        loss += 10000 * beta * (mse_loss(X1[i + 1, :, 10:], X1_next[:, 5:]))
        X1_current_encoded = net.encode(X1_current[:, :Nstate], v1_expanded)
        loss += (mse_loss(X1_current_encoded, X1_current))
        beta *= gamma
    loss = loss / beta_sum
    lambda_l1 = 0.00001
    reg_loss += l1_regularization(net, lambda_l1)
    reg_loss += l1_regularization(enc_net1, lambda_l1)
    reg_loss += l1_regularization(enc_net4, lambda_l1)
    A_loss = off_diagonal_l1_loss(net.lA.weight)
    loss = loss + loss1 + reg_loss + A_loss
    return loss


def Eig_loss(net):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = net.lA.weight
    c = torch.linalg.eigvals(A).abs() - torch.ones(1, dtype=torch.float64).to(device)
    mask = c > 0
    loss = c[mask].sum()
    return loss


def train(env_name, train_steps=250000, suffix="", all_loss=0, \
          encode_dim=12, layer_depth=3, e_loss=1, gamma=0.5, Ktrain_samples=50000):
    v1 = torch.randn(10)
    v4 = torch.randn(5)
    v1 = nn.Parameter(v1)
    v4 = nn.Parameter(v4)
    primary_udim1 = 5
    primary_sdim1 = 10
    common_sdim = 10
    common_udim = 5
    weight_sdim = 10
    weight_udim = 5
    in_dim = common_sdim
    u_dim = common_udim
    Nstate = in_dim
    layer_width = 128
    XENC_layers1 = [primary_sdim1] + [layer_width] * layer_depth + [common_sdim]
    enc_net1 = ENC_net(XENC_layers1)
    UENC_layers1 = [primary_udim1 + common_sdim + weight_udim] + [layer_width] * layer_depth + [common_udim]
    enc_net4 = ENC_net(UENC_layers1)
    DEC_layers1 = [common_sdim] + [layer_width] * layer_depth + [primary_sdim1]
    dec_net1 = DEC_net(DEC_layers1)
    layers = [in_dim + weight_sdim] + [layer_width] * layer_depth + [encode_dim]
    Nkoopman = in_dim + encode_dim
    print("layers:", layers)
    net = Network(layers, Nkoopman, u_dim)
    eval_step = 1000
    learning_rate = 1e-3
    net.cuda().double()
    enc_net1.cuda().double()
    enc_net4.cuda().double()
    dec_net1.cuda().double()
    mse_loss = nn.MSELoss()
    optimizer = torch.optim.Adam([
        {'params': net.parameters()},
        {'params': enc_net1.parameters()},
        {'params': enc_net4.parameters()},
        {'params': dec_net1.parameters()},
        {'params': v1},
        {'params': v4},
    ], lr=learning_rate)
    for name, param in net.named_parameters():
        print("model:", name, param.requires_grad)
    # train
    eval_step = 1000
    best_loss = 1000.0
    logdir = "/home/ccr/robot/train-koopman/Data/" + suffix + "/DUC_" + env_name + "layer{}_edim{}_eloss{}".format(
        layer_depth, encode_dim, e_loss)
    if not os.path.exists("/home/ccr/robot/train-koopman/Data/" + suffix):
        os.makedirs("/home/ccr/robot/train-koopman/Data/" + suffix)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    writer = SummaryWriter(log_dir=logdir)
    for i in range(train_steps):
        Five_train_samples = Five_train_data.shape[1]
        Five_indices = np.random.choice(Five_train_samples, size=100, replace=True)  # replace是否放回采样
        X1 = Five_train_data[:, Five_indices, :]
        Kloss = Klinear_loss(X1, net, enc_net1, enc_net4, dec_net1, v1, v4, mse_loss,gamma, Nstate)
        Eloss = Eig_loss(net)
        loss = Kloss + Eloss if e_loss else Kloss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar('Train/Kloss', Kloss, i)
        writer.add_scalar('Train/Eloss', Eloss, i)
        # writer.add_scalar('Train/Dloss',Dloss,i)
        writer.add_scalar('Train/loss', loss, i)
        # print("Step:{} Loss:{}".format(i,loss.detach().cpu().numpy()))
        if (i + 1) % eval_step == 0:
            # K loss
            with torch.no_grad():
                Kloss = Klinear_loss(Five_test_data, net, enc_net1, enc_net4, dec_net1, v1, v4,  mse_loss, gamma, Nstate)
                Eloss = Eig_loss(net)
                loss = Kloss
                Kloss = Kloss.detach().cpu().numpy()
                Eloss = Eloss.detach().cpu().numpy()
                # Dloss = Dloss.detach().cpu().numpy()
                loss = loss.detach().cpu().numpy()
                writer.add_scalar('Eval/Kloss', Kloss, i)
                writer.add_scalar('Eval/Eloss', Eloss, i)
                writer.add_scalar('Eval/best_loss', best_loss, i)
                writer.add_scalar('Eval/loss', loss, i)
                if loss < best_loss:
                    best_loss = copy(Kloss)
                    torch.save({
                        'net_state_dict': net.state_dict(),
                        'enc_net1_state_dict': enc_net1.state_dict(),
                        'enc_net4_state_dict': enc_net4.state_dict(),
                        'dec_net1_state_dict': dec_net1.state_dict(),
                        'v1': v1.data,
                        'v4': v4.data,
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
    parser.add_argument("--env", type=str, default="DampingPendulum")
    parser.add_argument("--suffix", type=str, default="enc440")
    parser.add_argument("--K_train_samples", type=int, default=50000)
    parser.add_argument("--e_loss", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.8)
    parser.add_argument("--encode_dim", type=int, default=60)
    parser.add_argument("--layer_depth", type=int, default=3)
    args = parser.parse_args()
    main()






# import torch
# import numpy as np
# import torch.nn as nn
# import torch.nn.functional as F
# import gym
# import matplotlib.pyplot as plt
# import random
# from collections import OrderedDict
# from copy import copy
# import argparse
# import os
# from torch.utils.tensorboard import SummaryWriter
# import sys
#
# sys.path.append("../utility/")
# from scipy.integrate import odeint
# import time
#
# Five_train_data = np.load('/home/ccr/robot/train-koopman/npy_file/five_link/Ktrain_data_primary.npy')
# Five_test_data = np.load('/home/ccr/robot/train-koopman/npy_file/five_link/Ktest_data_primary.npy')
#
# # define network
# def gaussian_init_(n_units, std=1):
#     sampler = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([std / n_units]))
#     Omega = sampler.sample((n_units, n_units))[..., 0]
#     return Omega
#
#
# class Network(nn.Module):
#     def __init__(self, encode_layers, bilinear_layers, Nkoopman, u_dim):
#         super(Network, self).__init__()
#         ELayers = OrderedDict()
#         for layer_i in range(len(encode_layers) - 1):
#             ELayers["linear_{}".format(layer_i)] = nn.Linear(encode_layers[layer_i], encode_layers[layer_i + 1])
#             if layer_i != len(encode_layers) - 2:
#                 ELayers["relu_{}".format(layer_i)] = nn.ReLU()
#         self.encode_net = nn.Sequential(ELayers)
#         BLayers = OrderedDict()
#         for layer_i in range(len(bilinear_layers) - 1):
#             BLayers["linear_{}".format(layer_i)] = nn.Linear(bilinear_layers[layer_i], bilinear_layers[layer_i + 1])
#             if layer_i != len(bilinear_layers) - 2:
#                 BLayers["relu_{}".format(layer_i)] = nn.ReLU()
#         self.bilinear_net = nn.Sequential(BLayers)
#         self.Nkoopman = Nkoopman
#         self.u_dim = u_dim
#         self.lA = nn.Linear(Nkoopman, Nkoopman, bias=False)
#         self.lA.weight.data = gaussian_init_(Nkoopman, std=1)
#         U, _, V = torch.svd(self.lA.weight.data)
#         self.lA.weight.data = torch.mm(U, V.t()) * 0.9
#         self.lB = nn.Linear(bilinear_layers[-1], Nkoopman, bias=False)
#
#     def encode(self, x):
#         return torch.cat([x, self.encode_net(x)], axis=-1)
#
#     def bicode(self, x, u):
#         x_all = torch.cat([x, u], axis=-1)
#         return self.bilinear_net(x_all)
#
#     def forward(self, x, b):
#         return self.lA(x) + self.lB(b)
#
#
# class ENC_net(nn.Module):
#     def __init__(self, ENC_layers):
#         super(ENC_net, self).__init__()
#         ENCLayers = OrderedDict ()
#         for layer_i in range(len(ENC_layers) - 1):
#             ENCLayers["linear_{}".format(layer_i)] = nn.Linear(ENC_layers[layer_i], ENC_layers[layer_i + 1])
#             if layer_i != len(ENC_layers) - 2:
#                 ENCLayers["relu_{}".format(layer_i)] = nn.ReLU()
#         self.ENC_net = nn.Sequential(ENCLayers)
#
#     def ENC(self, x):
#         return self.ENC_net(x)
#
# class DEC_net(nn.Module):
#     def __init__(self, DEC_layers):
#         super(DEC_net, self).__init__()
#         DECLayers = OrderedDict ()
#         for layer_i in range(len(DEC_layers) - 1):
#             DECLayers["linear_{}".format(layer_i)] = nn.Linear(DEC_layers[layer_i], DEC_layers[layer_i + 1])
#             if layer_i != len(DEC_layers) - 2:
#                 DECLayers["relu_{}".format(layer_i)] = nn.ReLU()
#         self.DEC_net = nn.Sequential(DECLayers)
#
#     def DEC(self, x):
#         return self.DEC_net(x)
#
# def l1_regularization(model, lambda_l1):
#     l1_norm = sum(p.abs().sum() for p in model.parameters())  # L1 正则化
#     return lambda_l1 * l1_norm
#
# def off_diagonal_l1_loss(A):
#     # 创建一个掩码，掩盖对角线元素
#     mask = 1 - torch.eye(A.size(0), device=A.device)
#     off_diag_elements = A * mask  # 选择非对角线元素
#
#     # 计算非对角线元素的 L1 norm
#     l1_loss = off_diag_elements.abs().sum()  # L1 loss: 求绝对值的和
#     return l1_loss
#
#
# # loss function
# def Klinear_loss(X1, net, enc_net1,dec_net1, mse_loss, u_dim, gamma, Nstate):
#     steps, train_traj_num, NKoopman = X1.shape
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     X1 = torch.DoubleTensor(X1).to(device)
#     X1_enc = enc_net1.ENC(X1[:,:,5:])
#     X1_dec = dec_net1.DEC(X1_enc)
#     X1_current = net.encode(X1_enc[0, :, :])
#     beta = 1.0
#     beta_sum = 0.0
#     loss = torch.zeros(1, dtype=torch.float64).to(device)
#     reg_loss = torch.zeros(1, dtype=torch.float64).to(device)
#     loss1 = torch.zeros(1, dtype=torch.float64).to(device)
#     loss1 += mse_loss(X1[:,:,5:], X1_dec[:,:,:])
#     for i in range(steps - 1):
#         bilinear = net.bicode(X1_current[:, :Nstate].detach(), X1[i, :, :5])  # detach's problem
#         X1_current = net.forward(X1_current, bilinear)
#         X1_current_dec = dec_net1.DEC(X1_current[:, :Nstate])
#         X1_current_enc = enc_net1.ENC(X1_current_dec)
#         beta_sum += beta
#         loss += beta * (mse_loss(X1_current[:, :Nstate], X1_current_enc))
#         loss += beta * (mse_loss(X1_current[:, :Nstate], X1_enc[i + 1, :, :]))
#         X1_next = dec_net1.DEC(X1_current[:, :Nstate])
#         loss += beta * (mse_loss(X1[i + 1, :, 5:10],X1_next[:,:5]))
#         loss += beta * 10000 * (mse_loss(X1[i + 1, :, 10:15],X1_next[:,5:10]))
#         X1_current_encoded = net.encode(X1_current[:, :Nstate])
#         loss += beta * (mse_loss(X1_current_encoded, X1_current))
#         beta *= gamma
#     lambda_l1 = 0.000001
#     # reg_loss += l1_regularization(net, lambda_l1)
#     # reg_loss += l1_regularization(enc_net1, lambda_l1)
#     A_loss = off_diagonal_l1_loss(net.lA.weight)
#     loss = loss / beta_sum
#     loss = loss + loss1 + reg_loss + A_loss
#     return loss
#
#
# def Eig_loss(net):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     A = net.lA.weight
#     c = torch.linalg.eigvals(A).abs() - torch.ones(1, dtype=torch.float64).to(device)
#     mask = c > 0
#     loss = c[mask].sum()
#     return loss
#
#
# def train(env_name, train_steps=250000, suffix="", all_loss=0, \
#           encode_dim=12, layer_depth=3, e_loss=1, gamma=0.5, Ktrain_samples=50000):
#     Ksteps = 10
#     Kbatch_size = 100
#     primary_udim1 = 5
#     primary_sdim1 = 10
#     common_sdim = 10
#     common_udim = 5
#     in_dim = common_sdim
#     u_dim = primary_udim1
#     Nstate = in_dim
#     # layer_depth = 4
#     layer_width = 128
#     XENC_layers1 = [primary_sdim1] + [layer_width] * layer_depth + [common_sdim]
#     enc_net1 = ENC_net(XENC_layers1)
#     DEC_layers1 = [common_sdim] + [layer_width] * layer_depth + [primary_sdim1]
#     dec_net1 = DEC_net(DEC_layers1)
#     layers = [in_dim] + [layer_width] * layer_depth + [encode_dim]
#     blayers = [in_dim + primary_udim1] + [layer_width] * layer_depth + [common_udim]
#     Nkoopman = in_dim + encode_dim
#     print("layers:", layers)
#     net = Network(layers,blayers,Nkoopman, u_dim)
#     eval_step = 1000
#     learning_rate = 1e-3
#     net.cuda().double()
#     enc_net1.cuda().double()
#     dec_net1.cuda().double()
#     mse_loss = nn.MSELoss()
#     optimizer = torch.optim.Adam([
#         {'params': net.parameters()},
#         {'params': enc_net1.parameters()},
#         {'params': dec_net1.parameters()},
#     ], lr=learning_rate)
#     for name, param in net.named_parameters():
#         print("model:", name, param.requires_grad)
#     # train
#     eval_step = 1000
#     best_loss = 1000.0
#     best_state_dict = {}
#     logdir = "/home/ccr/robot/train-koopman/Data/" + suffix + "/DUC_" + env_name + "layer{}_edim{}_eloss{}".format(
#         layer_depth, encode_dim, e_loss)
#     if not os.path.exists("/home/ccr/robot/train-koopman/Data/" + suffix):
#         os.makedirs("/home/ccr/robot/train-koopman/Data/" + suffix)
#     if not os.path.exists(logdir):
#         os.makedirs(logdir)
#     writer = SummaryWriter(log_dir=logdir)
#     start_time = time.process_time()
#     for i in range(train_steps):
#         Five_train_samples = Five_train_data.shape[1]
#         Five_indices = np.random.choice(Five_train_samples, size=100, replace=True) #replace是否放回采样
#         X1 = Five_train_data[:, Five_indices, :]
#         Kloss = Klinear_loss(X1, net, enc_net1, dec_net1, mse_loss, u_dim, gamma, Nstate)
#         Eloss = Eig_loss(net)
#         loss = Kloss + Eloss if e_loss else Kloss
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         writer.add_scalar('Train/Kloss', Kloss, i)
#         writer.add_scalar('Train/Eloss', Eloss, i)
#         # writer.add_scalar('Train/Dloss',Dloss,i)
#         writer.add_scalar('Train/loss', loss, i)
#         # print("Step:{} Loss:{}".format(i,loss.detach().cpu().numpy()))
#         if (i + 1) % eval_step == 0:
#             # K loss
#             with torch.no_grad():
#                 Kloss = Klinear_loss(Five_test_data, net, enc_net1, dec_net1,mse_loss, u_dim, gamma, Nstate)
#                 Eloss = Eig_loss(net)
#                 loss = Kloss
#                 Kloss = Kloss.detach().cpu().numpy()
#                 Eloss = Eloss.detach().cpu().numpy()
#                 # Dloss = Dloss.detach().cpu().numpy()
#                 loss = loss.detach().cpu().numpy()
#                 writer.add_scalar('Eval/Kloss', Kloss, i)
#                 writer.add_scalar('Eval/Eloss', Eloss, i)
#                 writer.add_scalar('Eval/best_loss', best_loss, i)
#                 writer.add_scalar('Eval/loss', loss, i)
#                 if loss < best_loss:
#                     best_loss = copy(Kloss)
#                     # best_state_dict = copy(net.state_dict())
#                     # Saved_dict = {'model': best_state_dict, 'layer': layers, 'blayer': blayers}
#                     # torch.save(Saved_dict, logdir + ".pth")
#                     torch.save({
#                         'net_state_dict': net.state_dict(),
#                         'enc_net1_state_dict': enc_net1.state_dict(),
#                         'dec_net1_state_dict': dec_net1.state_dict(),
#                     }, logdir + ".pth")
#                 print("Step:{} Eval-loss{} K-loss:{}".format(i, loss, Kloss))
#                 # print("-------------END-------------")
#         writer.add_scalar('Eval/best_loss', best_loss, i)
#         # if (time.process_time()-start_time)>=210*3600:
#         #     print("time out!:{}".format(time.clock()-start_time))
#         #     break
#     print("END-best_loss{}".format(best_loss))
#
#
# def main():
#     train(args.env, suffix=args.suffix,encode_dim=args.encode_dim,\
#            layer_depth=args.layer_depth, \
#           e_loss=args.e_loss, gamma=args.gamma, \
#           Ktrain_samples=args.K_train_samples)
#
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--env", type=str, default="DampingPendulum")
#     parser.add_argument("--suffix", type=str, default="enc432")
#     parser.add_argument("--K_train_samples", type=int, default=50000)
#     parser.add_argument("--e_loss", type=int, default=0)
#     parser.add_argument("--gamma", type=float, default=0.8)
#     parser.add_argument("--encode_dim", type=int, default=60)
#     parser.add_argument("--layer_depth", type=int, default=3)
#     args = parser.parse_args()
#     main()
