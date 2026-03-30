import torch
import numpy as np
import three_models_robots as lka
import sys
sys.path.append("../utility")
from bokeh.plotting import figure, output_file, show
from bokeh.layouts import gridplot

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
    elif udim==6:
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

    elif udim==5:
        for i in range(A - 10):
            X3_enc = enc_net3.ENC(state1[i, udim:])
            X3_current = net.encode(X3_enc)
            if i % 10 == 0:
                position_data = [state1[i, udim:].detach().numpy()]
                for j in range(10):
                    X3_current_dec1 = dec_net3.DEC(X3_current[:Nstate])
                    U3_enc = enc_net6.ENC(torch.cat((state1[i+j, :udim], X3_current_dec1), dim=-1))
                    X3_current = net.forward(X3_current, U3_enc)
                    X3_current_dec = dec_net3.DEC(X3_current[:Nstate])
                    position_data.append(X3_current_dec.detach().numpy())
                position_data = np.array(position_data).reshape(11, 2*udim)
                x = np.arange(i, i + 11)
                for j in range(2*udim):  # 绘制12条线
                    plots[j].line(x, position_data[:, j], line_color="red", line_width=1)
        show(grid)

def main():
    test_data = np.load('/home/ccr/project-koopman/robot_data/speed_traj_data/traj_train_file/test_data1_franka.npy')
    plot(test_data,7)
    test_data = np.load('/home/ccr/project-koopman/robot_data/speed_traj_data/traj_train_file/test_data2_franka.npy')
    plot(test_data,7)
    test_data = np.load('/home/ccr/project-koopman/robot_data/speed_traj_data/traj_train_file/test_data3_franka.npy')
    plot(test_data,7)
    test_data = np.load('/home/ccr/project-koopman/robot_data/speed_traj_data/traj_train_file/test_data4_franka.npy')
    plot(test_data,7)
    test_data = np.load('/home/ccr/project-koopman/robot_data/speed_traj_data/traj_train_file/test_data1_ur.npy')
    plot(test_data,6)
    test_data = np.load('/home/ccr/project-koopman/robot_data/speed_traj_data/traj_train_file/test_data2_ur.npy')
    plot(test_data,6)
    test_data = np.load('/home/ccr/project-koopman/robot_data/speed_traj_data/traj_train_file/test_data3_ur.npy')
    plot(test_data,6)
    test_data = np.load('/home/ccr/project-koopman/robot_data/speed_traj_data/traj_train_file/test_data4_ur.npy')
    plot(test_data,6)
    test_data = np.load('/home/ccr/project-koopman/robot_data/speed_traj_data/traj_train_file/test_data1_dofbot.npy')
    plot(test_data,5)
    test_data = np.load('/home/ccr/project-koopman/robot_data/speed_traj_data/traj_train_file/test_data2_dofbot.npy')
    plot(test_data,5)
    test_data = np.load('/home/ccr/project-koopman/robot_data/speed_traj_data/traj_train_file/test_data3_dofbot.npy')
    plot(test_data,5)
    test_data = np.load('/home/ccr/project-koopman/robot_data/speed_traj_data/traj_train_file/test_data4_dofbot.npy')
    plot(test_data,5)


if __name__ == "__main__":
    main()

# def calculate_prediction_error(test_data, udim):
#     state1 = test_data.reshape(-1, 3 * udim)
#     state1 = torch.DoubleTensor(state1)
#     A = state1.shape[0]
#
#     total_errors = []
#
#     if udim == 7:
#         for i in range(A - 10):
#             X1_enc = enc_net1.ENC(state1[i, udim:])
#             X1_current = net.encode(X1_enc)
#
#             position_data = [state1[i, udim:].detach().numpy()]
#             predicted_data = []
#
#             for j in range(10):
#                 X1_current_dec1 = dec_net1.DEC(X1_current[:Nstate])
#                 U1_enc = enc_net4.ENC(torch.cat((state1[i + j, :udim], X1_current_dec1), dim=-1))
#                 X1_current = net.forward(X1_current, U1_enc)
#                 X1_current_dec = dec_net1.DEC(X1_current[:Nstate])
#                 predicted_data.append(X1_current_dec.detach().numpy())
#
#             # 计算预测误差
#             predicted_data = np.array(predicted_data)
#             actual_data = state1[i + 1:i + 11, udim:].detach().numpy()
#
#             # 计算每个时间步的均方根误差
#             step_errors = []
#             for j in range(10):
#                 mse = np.mean((predicted_data[j] - actual_data[j]) ** 2)
#                 rmse = np.sqrt(mse)
#                 step_errors.append(rmse)
#
#             total_errors.extend(step_errors)
#
#     elif udim == 6:
#         for i in range(A - 10):
#             X2_enc = enc_net2.ENC(state1[i, udim:])
#             X2_current = net.encode(X2_enc)
#
#             position_data = [state1[i, udim:].detach().numpy()]
#             predicted_data = []
#
#             for j in range(10):
#                 X2_current_dec1 = dec_net2.DEC(X2_current[:Nstate])
#                 U2_enc = enc_net5.ENC(torch.cat((state1[i + j, :udim], X2_current_dec1), dim=-1))
#                 X2_current = net.forward(X2_current, U2_enc)
#                 X2_current_dec = dec_net2.DEC(X2_current[:Nstate])
#                 predicted_data.append(X2_current_dec.detach().numpy())
#
#             # 计算预测误差
#             predicted_data = np.array(predicted_data)
#             actual_data = state1[i + 1:i + 11, udim:].detach().numpy()
#
#             # 计算每个时间步的均方根误差
#             step_errors = []
#             for j in range(10):
#                 mse = np.mean((predicted_data[j] - actual_data[j]) ** 2)
#                 rmse = np.sqrt(mse)
#                 step_errors.append(rmse)
#
#             total_errors.extend(step_errors)
#
#     elif udim == 5:
#         for i in range(A - 10):
#             X3_enc = enc_net3.ENC(state1[i, udim:])
#             X3_current = net.encode(X3_enc)
#
#             position_data = [state1[i, udim:].detach().numpy()]
#             predicted_data = []
#
#             for j in range(10):
#                 X3_current_dec1 = dec_net3.DEC(X3_current[:Nstate])
#                 U3_enc = enc_net6.ENC(torch.cat((state1[i + j, :udim], X3_current_dec1), dim=-1))
#                 X3_current = net.forward(X3_current, U3_enc)
#                 X3_current_dec = dec_net3.DEC(X3_current[:Nstate])
#                 predicted_data.append(X3_current_dec.detach().numpy())
#
#             # 计算预测误差
#             predicted_data = np.array(predicted_data)
#             actual_data = state1[i + 1:i + 11, udim:].detach().numpy()
#
#             # 计算每个时间步的均方根误差
#             step_errors = []
#             for j in range(10):
#                 mse = np.mean((predicted_data[j] - actual_data[j]) ** 2)
#                 rmse = np.sqrt(mse)
#                 step_errors.append(rmse)
#
#             total_errors.extend(step_errors)
#
#     # 计算整体误差统计
#     if total_errors:
#         avg_error = np.mean(total_errors)
#         std_error = np.std(total_errors)
#         max_error = np.max(total_errors)
#         min_error = np.min(total_errors)
#
#         return {
#             'average_rmse': avg_error,
#             'std_rmse': std_error,
#             'max_rmse': max_error,
#             'min_rmse': min_error,
#             'all_errors': total_errors
#         }
#     else:
#         return None
#
#
# def plot(test_data, udim):
#     state1 = test_data.reshape(-1, 3 * udim)
#     x = np.arange(len(state1[:, 10]))
#     y_data = [state1[:, i] for i in range(udim, 3 * udim)]
#     plots = []
#     for i, y in enumerate(y_data):
#         p = figure(title=f"Plot {i + 1}", width=800, height=400)
#         p.line(x, y, line_width=2)
#         plots.append(p)
#     grid = gridplot([plots[i:i + 3] for i in range(0, len(plots), 3)])
#     state1 = torch.DoubleTensor(state1)
#     A = state1.shape[0]
#
#     if udim == 7:
#         for i in range(A - 10):
#             X1_enc = enc_net1.ENC(state1[i, udim:])
#             X1_current = net.encode(X1_enc)
#             if i % 10 == 0:
#                 position_data = [state1[i, udim:].detach().numpy()]
#                 for j in range(10):
#                     X1_current_dec1 = dec_net1.DEC(X1_current[:Nstate])
#                     U1_enc = enc_net4.ENC(torch.cat((state1[i + j, :udim], X1_current_dec1), dim=-1))
#                     X1_current = net.forward(X1_current, U1_enc)
#                     X1_current_dec = dec_net1.DEC(X1_current[:Nstate])
#                     position_data.append(X1_current_dec.detach().numpy())
#                 position_data = np.array(position_data).reshape(11, 2 * udim)
#                 x = np.arange(i, i + 11)
#                 for j in range(2 * udim):  # 绘制12条线
#                     plots[j].line(x, position_data[:, j], line_color="red", line_width=1)
#         show(grid)
#     elif udim == 6:
#         for i in range(A - 10):
#             X2_enc = enc_net2.ENC(state1[i, udim:])
#             X2_current = net.encode(X2_enc)
#             if i % 10 == 0:
#                 position_data = [state1[i, udim:].detach().numpy()]
#                 for j in range(10):
#                     X2_current_dec1 = dec_net2.DEC(X2_current[:Nstate])
#                     U2_enc = enc_net5.ENC(torch.cat((state1[i + j, :udim], X2_current_dec1), dim=-1))
#                     X2_current = net.forward(X2_current, U2_enc)
#                     X2_current_dec = dec_net2.DEC(X2_current[:Nstate])
#                     position_data.append(X2_current_dec.detach().numpy())
#                 position_data = np.array(position_data).reshape(11, 2 * udim)
#                 x = np.arange(i, i + 11)
#                 for j in range(2 * udim):  # 绘制12条线
#                     plots[j].line(x, position_data[:, j], line_color="red", line_width=1)
#         show(grid)
#
#     elif udim == 5:
#         for i in range(A - 10):
#             X3_enc = enc_net3.ENC(state1[i, udim:])
#             X3_current = net.encode(X3_enc)
#             if i % 10 == 0:
#                 position_data = [state1[i, udim:].detach().numpy()]
#                 for j in range(10):
#                     X3_current_dec1 = dec_net3.DEC(X3_current[:Nstate])
#                     U3_enc = enc_net6.ENC(torch.cat((state1[i + j, :udim], X3_current_dec1), dim=-1))
#                     X3_current = net.forward(X3_current, U3_enc)
#                     X3_current_dec = dec_net3.DEC(X3_current[:Nstate])
#                     position_data.append(X3_current_dec.detach().numpy())
#                 position_data = np.array(position_data).reshape(11, 2 * udim)
#                 x = np.arange(i, i + 11)
#                 for j in range(2 * udim):  # 绘制12条线
#                     plots[j].line(x, position_data[:, j], line_color="red", line_width=1)
#         show(grid)
#
#
# def main():
#     # 计算并打印每个系统的预测误差
#     test_data = np.load('/home/ccr/project-koopman/robot_data/speed_traj_data/traj_train_file/test_data1_franka.npy')
#     franka_error = calculate_prediction_error(test_data, 7)
#     print("Franka系统预测误差:")
#     print(f"  平均RMSE: {franka_error['average_rmse']:.6f}")
#     print(f"  标准差: {franka_error['std_rmse']:.6f}")
#     print(f"  最大误差: {franka_error['max_rmse']:.6f}")
#     print(f"  最小误差: {franka_error['min_rmse']:.6f}")
#
#     # 绘制Franka系统图像
#     plot(test_data, 7)
#
#     test_data = np.load('/home/ccr/project-koopman/robot_data/speed_traj_data/traj_train_file/test_data1_ur.npy')
#     ur_error = calculate_prediction_error(test_data, 6)
#     print("\nUR系统预测误差:")
#     print(f"  平均RMSE: {ur_error['average_rmse']:.6f}")
#     print(f"  标准差: {ur_error['std_rmse']:.6f}")
#     print(f"  最大误差: {ur_error['max_rmse']:.6f}")
#     print(f"  最小误差: {ur_error['min_rmse']:.6f}")
#
#     # 绘制UR系统图像
#     plot(test_data, 6)
#
#     test_data = np.load('/home/ccr/project-koopman/robot_data/speed_traj_data/traj_train_file/test_data2_dofbot.npy')
#     dofbot_error = calculate_prediction_error(test_data, 5)
#     print("\nDofbot系统预测误差:")
#     print(f"  平均RMSE: {dofbot_error['average_rmse']:.6f}")
#     print(f"  标准差: {dofbot_error['std_rmse']:.6f}")
#     print(f"  最大误差: {dofbot_error['max_rmse']:.6f}")
#     print(f"  最小误差: {dofbot_error['min_rmse']:.6f}")
#
#     # 绘制Dofbot系统图像
#     plot(test_data, 5)
#
#     # 比较三个系统的性能
#     print("\n系统性能比较:")
#     print(f"  Franka平均RMSE: {franka_error['average_rmse']:.6f}")
#     print(f"  UR平均RMSE: {ur_error['average_rmse']:.6f}")
#     print(f"  Dofbot平均RMSE: {dofbot_error['average_rmse']:.6f}")
#
#
# if __name__ == "__main__":
#     main()