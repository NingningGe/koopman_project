#新模型效果更好
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import franka as lka 

# ================= 1. 配置与论文风格设置 =================
save_dir = '/home/nng/koopman_project/lunwen/第三章/ur_test_jpg2'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 顶刊专业绘图参数
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 16,
    'axes.labelsize': 20,
    'axes.titlesize': 22,
    'legend.fontsize': 14,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'lines.linewidth': 2.5,
    'figure.dpi': 300
})

device = torch.device("cpu")

# ================= 2. UR 模型架构参数 =================
primary_udim_ur = 6
primary_sdim_ur = 12
common_sdim = 20
common_udim = 10
layer_depth = 3
layer_width = 128       # UR 编解码器的宽度
layer_width2 = 256      # Franka Koopman 核心的宽度
encode_dim = 100
Nstate = common_sdim
Nkoopman = common_sdim + encode_dim

# ================= 3. 模型加载逻辑 (应用双源网络拼接) =================
def load_ur_models():
    print("[INFO] 开始加载 Franka 先验法则与 UR 域迁移模型...")

    # 1. 提取 Franka 的先验 Koopman 矩阵
    path_franka = "/home/nng/koopman_project/cr_transferlearning/transfer_learning/control_transfer/A_to_B/Data/franka/unifiedur_transferlayer3_edim100_eloss1.pth"
    dicts_f = torch.load(path_franka, map_location=device)
    net_state_dict = dicts_f["net_state_dict"]
    
    # 2. 提取目标域 UR 最新训练的对齐模型
    path_ur = "/home/nng/koopman_project/cr_transferlearning/transfer_learning/control_transfer/A_to_B/Data/franka_to_ur/2unifiedur_transferlayer3_edim100_eloss1.pth"
    dicts_u = torch.load(path_ur, map_location=device)
    enc_net2_state_dict = dicts_u["enc_net2_state_dict"]
    enc_net5_state_dict = dicts_u["enc_net5_state_dict"]
    dec_net2_state_dict = dicts_u["dec_net2_state_dict"]
    dec_net5_state_dict = dicts_u["dec_net5_state_dict"]

    # 3. 初始化 UR 编解码网络 (宽度 layer_width = 128)
    XENC_layers2 = [primary_sdim_ur] + [layer_width] * layer_depth + [common_sdim]
    enc_net2 = lka.ENC_net(XENC_layers2)
    
    UENC_layers2 = [primary_udim_ur + primary_sdim_ur] + [layer_width] * layer_depth + [common_udim]
    enc_net5 = lka.ENC_net(UENC_layers2)
    
    DEC_layers2 = [common_sdim] + [layer_width] * layer_depth + [primary_sdim_ur]
    dec_net2 = lka.DEC_net(DEC_layers2)
    
    DEC_layers5 = [common_udim + primary_sdim_ur] + [layer_width] * layer_depth + [primary_udim_ur]
    dec_net5 = lka.DEC_net(DEC_layers5)

    # 4. 初始化 Franka 核心网络 (宽度 layer_width2 = 256)
    layers = [common_sdim] + [layer_width2] * layer_depth + [encode_dim]
    net = lka.Network(layers, Nkoopman, common_udim)

    # 5. 加载权重并开启评估模式
    net.to(device).double().load_state_dict(net_state_dict)
    enc_net2.to(device).double().load_state_dict(enc_net2_state_dict)
    enc_net5.to(device).double().load_state_dict(enc_net5_state_dict)
    dec_net2.to(device).double().load_state_dict(dec_net2_state_dict)
    dec_net5.to(device).double().load_state_dict(dec_net5_state_dict)

    return net.eval(), enc_net2.eval(), enc_net5.eval(), dec_net2.eval()

# ================= 4. 测试与特定关节绘图 =================
def evaluate_specific_joint(data_path, tag, joint_idx, models):
    """
    针对指定的关节进行测试与绘图。
    joint_idx: 0 代表 Joint 1; 5 代表 Joint 6。
    """
    net, enc_net2, enc_net5, dec_net2 = models
    joint_num = joint_idx + 1 # 物理关节编号 (1~6)
    
    print(f"正在处理数据集: [{tag}] ---> 锁定关节: [Joint {joint_num}] (测试步数: 20步)")
    test_data = np.load(data_path)
    state_all = test_data.reshape(-1, 18) 
    
    # 预测步数: 20步
    test_steps = 50
    
    u_phys = torch.DoubleTensor(state_all[:test_steps, :primary_udim_ur]).to(device)
    s_phys = torch.DoubleTensor(state_all[:test_steps, primary_udim_ur:]).to(device)
    
    # 闭环推演
    preds = []
    with torch.no_grad():
        X2_enc = enc_net2.ENC(s_phys[0:1])
        X2_current = net.encode(X2_enc)
        
        for t in range(test_steps):
            X2_current_dec = dec_net2.DEC(X2_current[:, :Nstate])
            preds.append(X2_current_dec.squeeze().cpu().numpy())
            
            U2_enc = enc_net5.ENC(torch.cat((u_phys[t:t+1], X2_current_dec), dim=-1))
            X2_current = net.forward(X2_current, U2_enc)
            
    x_pred_all = np.array(preds)
    x_true_all = s_phys.cpu().numpy()

    # 提取特定关节的 [位置, 速度] 数据
    pos_idx = joint_idx
    vel_idx = joint_idx + 6
    
    gt_pos = x_true_all[:, pos_idx]
    gt_vel = x_true_all[:, vel_idx]
    pred_pos = x_pred_all[:, pos_idx]
    pred_vel = x_pred_all[:, vel_idx]

    # 为了计算误差，将位置和速度组合成 (20, 2) 的矩阵
    gt_joint = np.column_stack((gt_pos, gt_vel))
    pred_joint = np.column_stack((pred_pos, pred_vel))

    # ================= 独立生成 4 张专注该关节的图 =================
    
    # --- 绘图 1: 该关节的状态追踪 (分上下两图：位置 & 速度) ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    # 位置图
    ax1.plot(gt_pos, 'k-', alpha=0.8, label='Ground Truth')
    ax1.plot(pred_pos, 'r--', label='Koopman Prediction')
    ax1.set_ylabel(f'Position $q_{joint_num}$')
    ax1.set_title(f'Tracking Performance of Joint {joint_num} ({tag})')
    ax1.legend()
    ax1.grid(linestyle=':', alpha=0.5)
    # 速度图
    ax2.plot(gt_vel, 'k-', alpha=0.8, label='Ground Truth')
    ax2.plot(pred_vel, 'r--', label='Koopman Prediction')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel(f'Velocity $\\dot{{q}}_{joint_num}$')
    ax2.legend()
    ax2.grid(linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/tracking_{tag}_J{joint_num}.png')

    # --- 绘图 2: 该关节专属 L2 误差 ---
    l2_error = np.linalg.norm(gt_joint - pred_joint, axis=1)
    plt.figure(figsize=(10, 6))
    plt.plot(l2_error, 'b-', lw=3)
    plt.xlabel('Steps')
    plt.ylabel('L2 Norm Error')
    plt.title(f'Prediction Error Accumulation for Joint {joint_num}')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.savefig(f'{save_dir}/l2_err_{tag}_J{joint_num}.png')

    # --- 绘图 3: 该关节专属相对误差百分比 ---
    rel_error = np.abs(gt_joint - pred_joint) / (np.abs(gt_joint) + 1e-3) * 100
    avg_rel_err = np.mean(rel_error, axis=1)
    plt.figure(figsize=(10, 6))
    plt.plot(avg_rel_err, color='crimson', label='Relative Error %', lw=3)
    plt.axhline(y=5, color='gray', linestyle='--', label='5% Bound')
    plt.ylim(0, max(20, np.max(avg_rel_err)*1.2))
    plt.title(f'Relative Error Percentage for Joint {joint_num}')
    plt.ylabel('Error (%)')
    plt.xlabel('Steps')
    plt.legend()
    plt.grid(axis='y', linestyle=':', alpha=0.5)
    plt.savefig(f'{save_dir}/percent_err_{tag}_J{joint_num}.png')

    # --- 绘图 4: 该关节专属相空间预测精度对比 ---
    plt.figure(figsize=(8, 8))
    plt.plot(gt_pos, gt_vel, 'k-', alpha=0.4, lw=3, label='Ground Truth')
    plt.plot(pred_pos, pred_vel, 'r--', lw=2.5, label='Koopman Prediction')
    plt.xlabel(f'Position $q_{joint_num}$')
    plt.ylabel(f'Velocity $\\dot{{q}}_{joint_num}$')
    plt.title(f'Phase Portrait Comparison (Joint {joint_num})')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.savefig(f'{save_dir}/phase_{tag}_J{joint_num}.png')
    
    plt.close('all')
    print(f"√ 已生成 {tag} (J{joint_num}) 的全套评估图。")

def main():
    models = load_ur_models()
    
    # 指定测试数据集路径
    data_path = "/home/nng/koopman_project/cr_transferlearning/robot_data/speed_traj_data/traj_train_file/test_data4_ur.npy"
    tag = "data4"
    
    if not os.path.exists(data_path):
        print(f"警告: 找不到数据集 {data_path}")
        return
        
    print(f"\n========== 开始全面评估 {tag} 数据集 ==========")
    
    # 循环遍历 6 个关节 (索引 0 到 5)
    for joint_idx in range(6):
        evaluate_specific_joint(data_path, tag, joint_idx, models)
            
    print(f"\n[SUCCESS] 所有 6 个关节的专项测试完毕！")
    print(f"共生成 {6 * 4} 张图片，已全部保存在: {save_dir}")

if __name__ == "__main__":
    main()