import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ================= 1. 论文级绘图参数设置 =================
# 这里的字号和线宽经过专门调整，确保图片缩放到 Word 宽度的一半时依然清晰
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 18,               # 基础字号
    'axes.titlesize': 24,           # 标题字号
    'axes.labelsize': 22,           # 坐标轴标签字号
    'xtick.labelsize': 18,          # x轴刻度字号
    'ytick.labelsize': 18,          # y轴刻度字号
    'legend.fontsize': 16,          # 图例字号
    'lines.linewidth': 2.5,         # 线宽
    'figure.facecolor': 'white'     # 背景白色
})

# 设定保存目录
save_dir = '/home/nng/koopman_project/lunwen'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# ================= 2. 数据加载与预处理 =================
franka_path = '/home/nng/koopman_project/cr_transferlearning/robot_data/speed_traj_data/traj_train_file/Ktrain_data_franka.npy'
ur_path = '/home/nng/koopman_project/cr_transferlearning/robot_data/speed_traj_data/traj_train_file/Ktrain_data_ur.npy'

print("正在加载数据集...")
# 原始数据形状分别为 (15, 20000, 21) 和 (15, 20000, 18)
raw_franka = np.load(franka_path, allow_pickle=True)
raw_ur = np.load(ur_path, allow_pickle=True)

# 展平数据用于统计分布图
data_franka_flat = raw_franka.reshape(-1, raw_franka.shape[-1])
data_ur_flat = raw_ur.reshape(-1, raw_ur.shape[-1])

# ================= 3. 绘图任务 =================

# --- 图 1: 关节状态分布 (Joint Position Distribution) ---
# 证明采样覆盖了足够的工作空间
plt.figure(figsize=(18, 8))
plt.subplot(1, 2, 1)
for i in range(7):
    # 根据你的拼接逻辑：noisy_speeds(7) + positions(7) + velocities(7)
    # 位置索引应为 7 到 13
    sns.kdeplot(data_franka_flat[:, 7+i], label=f'Joint {i+1}', fill=True, alpha=0.1)
plt.title('Franka Joint Position Distribution')
plt.xlabel('Angle (rad)')
plt.ylabel('Density')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.subplot(1, 2, 2)
for i in range(6):
    # UR 位置索引应为 6 到 11
    sns.kdeplot(data_ur_flat[:, 6+i], label=f'Joint {i+1}', fill=True, alpha=0.1)
plt.title('UR Joint Position Distribution')
plt.xlabel('Angle (rad)')
plt.ylabel('Density')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, '2.3_distribution_comparison.png'), dpi=300)
print(f"已保存: {save_dir}/2.3_distribution_comparison.png")

# --- 图 2: 动力学相空间轨迹 (Phase Portrait) ---
# 展示非线性动力学演化。良好的 Koopman 数据应展现出连续的轨迹
plt.figure(figsize=(18, 8))
plt.subplot(1, 2, 1)
# 绘制 Franka J1: 位置(索引7) vs 速度(索引14)
plt.scatter(data_franka_flat[:10000, 7], data_franka_flat[:10000, 14], s=2, alpha=0.5, c='#1f77b4')
plt.title('Franka Phase Portrait (Joint 1)')
plt.xlabel('Position $q_1$ (rad)')
plt.ylabel('Velocity $\dot{q}_1$ (rad/s)')

plt.subplot(1, 2, 2)
# 绘制 UR J1: 位置(索引6) vs 速度(索引12)
plt.scatter(data_ur_flat[:10000, 6], data_ur_flat[:10000, 12], s=2, alpha=0.5, c='#d62728')
plt.title('UR Phase Portrait (Joint 1)')
plt.xlabel('Position $q_1$ (rad)')
plt.ylabel('Velocity $\dot{q}_1$ (rad/s)')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, '2.3_phase_portrait.png'), dpi=300)
print(f"已保存: {save_dir}/2.3_phase_portrait.png")

# --- 图 3: 激励信号与响应 (Control-Response Time Series) ---
# 证明控制输入 u 确实驱动了状态变化，且数据连贯
plt.figure(figsize=(16, 7))
# 关键：从 15 条轨迹中选第 1 条，取前 800 个点
single_traj = raw_franka[0]
time_steps = 800
time_axis = np.arange(time_steps)

plt.plot(time_axis, single_traj[:time_steps, 0], 'k--', label='Control Input ($u_1$)')
plt.plot(time_axis, single_traj[:time_steps, 14], 'r-', label='Actual Velocity ($\dot{q}_1$)', alpha=0.8)
plt.title('Control Input vs. System Response (Franka J1)')
plt.xlabel('Time Steps')
plt.ylabel('Velocity (rad/s)')
plt.legend(loc='upper right')
plt.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, '2.3_excitation_signal.png'), dpi=300)
print(f"已保存: {save_dir}/2.3_excitation_signal.png")

plt.show()