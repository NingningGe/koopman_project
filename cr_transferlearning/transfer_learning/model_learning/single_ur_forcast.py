from bokeh.plotting import figure, output_file, show
from bokeh.layouts import gridplot
import numpy as np
import franka as lka
import torch

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
dicts = torch.load("/control_transfer/A_to_B/Data/franka_to_ur2/unifiedur_transferlayer3_edim100_eloss1.pth", map_location=torch.device('cpu'))
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

import os
suffix = "enc_test2"
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
    state1 = torch.DoubleTensor(state1)
    A = state1.shape[0]
    if udim==6:
        for i in range(A - 10):
            X2_enc = enc_net2.ENC(state1[i, udim:])
            X2_current = net.encode(X2_enc)
            if i % 10 == 0:
                position_data = [state1[i, udim:].detach().numpy()]
                for j in range(10):
                    X2_current_dec1 = dec_net2.DEC(X2_current[:Nstate])
                    U2_enc = enc_net5.ENC(torch.cat((state1[i+j, :udim], X2_current_dec1), dim=-1))
                    X2_current = net.forward(X2_current, U2_enc)
                    X2_current_dec = dec_net2.DEC(X2_current[:Nstate])
                    position_data.append(X2_current_dec.detach().numpy())
                position_data = np.array(position_data).reshape(11, 2*udim)
                x = np.arange(i, i + 11)
                for j in range(2*udim):  # 绘制12条线
                    plots[j].line(x, position_data[:, j], line_color="red", line_width=1)
        show(grid)

def main():
    test_data = np.load('/robot_data/speed_traj_data/traj_train_file/test_data1_ur.npy')
    plot(test_data,6)
    test_data = np.load('/robot_data/speed_traj_data/traj_train_file/test_data2_ur.npy')
    plot(test_data,6)
    test_data = np.load('/robot_data/speed_traj_data/traj_train_file/test_data3_ur.npy')
    plot(test_data,6)
    test_data = np.load('/robot_data/speed_traj_data/traj_train_file/test_data4_ur.npy')
    plot(test_data,6)



if __name__ == "__main__":
    main()