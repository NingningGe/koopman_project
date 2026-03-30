import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import gym
import matplotlib.pyplot as plt
import random
import argparse
from collections import OrderedDict
from copy import copy
import franka as lka
import scipy
import scipy.linalg
from scipy.integrate import odeint
import pybullet as pb
import pybullet_data
import math
import sys
sys.path.append("../utility")
# Franka simulator
from bokeh.plotting import figure, output_file, show
from bokeh.layouts import gridplot

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
net.load_state_dict(net_state_dict)
dnet.load_state_dict(dnet_state_dict)
enc_net1.load_state_dict(enc_net1_state_dict)
enc_net4.load_state_dict(enc_net4_state_dict)
dec_net1.load_state_dict(dec_net1_state_dict)
dec_net4.load_state_dict(dec_net4_state_dict)

import os
suffix = "enc_test4"
if not os.path.exists("/home/ccr/unified_represention/franka/traj/photo/" + suffix):
    os.makedirs("/home/ccr/unified_represention/franka/traj/photo/" + suffix)

output_file("/home/ccr/unified_represention/franka/traj/photo/" + suffix + "/1.html")

def plot(test_data,udim):
    state1 = test_data.reshape(-1, 3*udim)
    x = np.arange(len(state1[:, 10]))
    y_data = [state1[:, i] for i in range(udim, 3*udim)]
    plots = []
    for i, y in enumerate(y_data):
        p = figure(title=f"Plot {i + 1}", width=800, height=400)
        p.line(x, y, line_width=2)
        plots.append(p)
    grid = gridplot([plots[i:i + 3] for i in range(0, len(plots), 3)])
    state1 = torch.Tensor(state1)
    A = state1.shape[0]
    if udim==7:
        for i in range(A - 10):
            X1_enc = enc_net1.ENC(state1[i, udim:])
            X1_current = net.encode(X1_enc)
            if i % 10 == 0:
                position_data = [state1[i, udim:].detach().numpy()]
                for j in range(10):
                    X1_current_dec1 = dec_net1.DEC(X1_current[:Nstate])
                    U1_enc = enc_net4.ENC(torch.cat((state1[i+j, :udim], X1_current_dec1), dim=-1))
                    X1_current = net.forward(X1_current, U1_enc)
                    X1_current_dec = dec_net1.DEC(X1_current[:Nstate])
                    position_data.append(X1_current_dec.detach().numpy())
                position_data = np.array(position_data).reshape(11, 2*udim)
                x = np.arange(i, i + 11)
                for j in range(2*udim):  # 绘制12条线
                    plots[j].line(x, position_data[:, j], line_color="red", line_width=1)
        show(grid)
def main():
    test_data = np.load('/model_transfer_robot/speed_traj_data/traj_train_file/test_data1_franka.npy')
    plot(test_data,7)
    test_data = np.load('/model_transfer_robot/speed_traj_data/traj_train_file/test_data2_franka.npy')
    plot(test_data,7)
    test_data = np.load('/model_transfer_robot/speed_traj_data/traj_train_file/test_data3_franka.npy')
    plot(test_data,7)
    test_data = np.load('/model_transfer_robot/speed_traj_data/traj_train_file/test_data4_franka.npy')
    plot(test_data,7)

if __name__ == "__main__":
    main()