import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import franka as lka 

# ================= 1. 配置与论文风格设置 =================
save_dir = '/home/nng/koopman_project/lunwen/koopman'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'legend.fontsize': 10,
    'figure.dpi': 300
})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型架构参数
primary_udim1, primary_sdim1 = 7, 14
common_sdim, common_udim = 20, 10
encode_dim, layer_depth = 100, 3
layer_width, layer_width2 = 128, 256
Nstate, Nkoopman = common_sdim, common_sdim + encode_dim

# ================= 2. 模型加载逻辑 =================
def load_models():
    enc_net1 = lka.ENC_net([primary_sdim1] + [layer_width] * layer_depth + [common_sdim]).double().to(device)
    enc_net4 = lka.ENC_net([primary_udim1 + primary_sdim1] + [layer_width] * layer_depth + [common_udim]).double().to(device)
    dec_net1 = lka.DEC_net([common_sdim] + [layer_width] * layer_depth + [primary_sdim1]).double().to(device)
    net = lka.Network([common_sdim] + [layer_width2] * layer_depth + [encode_dim], Nkoopman, common_udim).double().to(device)

    pth_path = "/home/nng/koopman_project/cr_transferlearning/transfer_learning/control_transfer/A_to_B/Data/franka/unifiedur_transferlayer3_edim100_eloss1.pth"
    dicts = torch.load(pth_path, map_location=device, weights_only=False)
    net.load_state_dict(dicts["net_state_dict"])
    enc_net1.load_state_dict(dicts["enc_net1_state_dict"])
    enc_net4.load_state_dict(dicts["enc_net4_state_dict"])
    dec_net1.load_state_dict(dicts["dec_net1_state_dict"])
    return net.eval(), enc_net1.eval(), enc_net4.eval(), dec_net1.eval()

net, enc_net1, enc_net4, dec_net1 = load_models()

# ================= 3. 核心评估函数 =================
def run_comprehensive_eval(path, tag, num_samples=3):
    raw_data = np.load(path, allow_pickle=True)
    state_all = torch.DoubleTensor(raw_data.reshape(-1, 21)).to(device)
    total_steps = state_all.shape[0]
    forecast_steps = min(400, total_steps - 10) # 动态适配步数

    for s in range(num_samples):
        start_idx = random.randint(0, total_steps - forecast_steps - 1)
        current_tag = f"{tag}_S{s+1}"
        
        u_true = state_all[start_idx : start_idx + forecast_steps, :primary_udim1]
        x_true = state_all[start_idx : start_idx + forecast_steps, primary_udim1:]
        
        x_pred_list = []
        with torch.no_grad():
            curr_g = net.encode(enc_net1.ENC(x_true[0:1, :]))
            
            for i in range(forecast_steps - 1):
                # 递归预测：使用前一步预测的状态反馈给动作编码器
                curr_x_dec = dec_net1.DEC(curr_g[:, :Nstate]).reshape(1, 14)
                u_curr = u_true[i:i+1, :].reshape(1, 7)
                
                u_input = torch.cat([u_curr, curr_x_dec], dim=1) # 维度修复
                curr_g = net.forward(curr_g, enc_net4.ENC(u_input))
                
                x_next_raw = dec_net1.DEC(curr_g[:, :Nstate])
                x_pred_list.append(x_next_raw.cpu().numpy())
                
        x_pred = np.vstack(x_pred_list)
        x_true_np = x_true[1:forecast_steps, :].cpu().numpy()

        # --- 绘图 1: 多步前向预测追踪图 (选取第一关节) ---
        fig, ax = plt.subplots(2, 1, figsize=(12, 10))
        ax[0].plot(x_true_np[:, 0], 'k-', label='True Position', linewidth=2)
        ax[0].plot(x_pred[:, 0], 'r--', label='Koopman Forecast', linewidth=1.5)
        ax[0].set_ylabel('Pos $q_1$ (rad)')
        ax[0].legend()
        ax[0].set_title(f'Multi-step State Tracking ({current_tag})')

        ax[1].plot(x_true_np[:, 7], 'k-', label='True Velocity', linewidth=2)
        ax[1].plot(x_pred[:, 7], 'b--', label='Koopman Forecast', linewidth=1.5)
        ax[1].set_ylabel('Vel $\dot{q}_1$ (rad/s)')
        ax[1].set_xlabel('Time Steps')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/tracking_{current_tag}.png')

        # --- 绘图 2: 预测误差随步数演化图 (MAE) ---
        mae_step = np.mean(np.abs(x_true_np - x_pred), axis=1)
        plt.figure(figsize=(10, 5))
        plt.plot(mae_step, color='darkorange', linewidth=2)
        plt.fill_between(range(len(mae_step)), 0, mae_step, alpha=0.2, color='orange')
        plt.title('Error Evolution over Prediction Horizon')
        plt.xlabel('Prediction Steps')
        plt.ylabel('Mean Absolute Error')
        plt.savefig(f'{save_dir}/error_steps_{current_tag}.png')

        # --- 绘图 3: 百分比误差分析 ---
        rel_error = np.abs(x_true_np - x_pred) / (np.abs(x_true_np) + 1e-3) * 100
        avg_rel_err = np.mean(rel_error, axis=1)
        plt.figure(figsize=(10, 5))
        plt.plot(avg_rel_err, color='crimson', label='Relative Error %')
        plt.axhline(y=5, color='gray', linestyle='--', label='5% Bound')
        plt.ylim(0, 30)
        plt.title('Prediction Relative Error Percentage')
        plt.ylabel('Error (%)')
        plt.legend()
        plt.savefig(f'{save_dir}/percent_err_{current_tag}.png')

        # --- 绘图 4: 相空间预测精度对比 ---
        plt.figure(figsize=(8, 8))
        plt.plot(x_true_np[:, 0], x_true_np[:, 7], 'k', alpha=0.3, label='Ground Truth')
        plt.plot(x_pred[:, 0], x_pred[:, 7], 'r--', label='Koopman Prediction')
        plt.xlabel('Position $q_1$')
        plt.ylabel('Velocity $\dot{q}_1$')
        plt.title('Phase Portrait Comparison')
        plt.legend()
        plt.savefig(f'{save_dir}/phase_{current_tag}.png')
        
        plt.close('all')

    print(f"已生成 {tag} 的全维度评估报告图。")

def main():
    test_files = [
        ('/home/nng/koopman_project/cr_transferlearning/robot_data/speed_traj_data/traj_train_file/test_data1_franka.npy', "Traj1"),
        ('/home/nng/koopman_project/cr_transferlearning/robot_data/speed_traj_data/traj_train_file/test_data3_franka.npy', "Traj2")
    ]
    for path, tag in test_files:
        if os.path.exists(path):
            run_comprehensive_eval(path, tag, num_samples=3)

if __name__ == "__main__":
    main()