import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.manifold import TSNE
import franka as lka 

# ================= 1. 环境与极清大字号设置 =================
save_dir = '/home/nng/koopman_project/lunwen/第三章/test_jpg'
if not os.path.exists(save_dir): os.makedirs(save_dir)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 22,                
    'axes.labelsize': 28,           # 增加坐标轴标签字号
    'axes.titlesize': 32,           # 增加标题字号
    'legend.fontsize': 18,          
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'figure.dpi': 300,
    'axes.linewidth': 2             # 坐标轴边框加粗
})

device = torch.device("cpu")

# ================= 2. 加载权重与数据 =================
path_f = "/home/nng/koopman_project/cr_transferlearning/transfer_learning/control_transfer/A_to_B/Data/franka/unifiedur_transferlayer3_edim100_eloss1.pth"
path_u = "/home/nng/koopman_project/cr_transferlearning/transfer_learning/control_transfer/A_to_B/Data/franka_to_ur2/unifiedur_transferlayer3_edim100_eloss1.pth"

# 提取测试数据状态量
data_f_raw = np.load('/home/nng/koopman_project/cr_transferlearning/robot_data/speed_traj_data/traj_train_file/Ktest_data_franka.npy').reshape(-1, 21)[:, 7:21]
data_u_raw = np.load('/home/nng/koopman_project/cr_transferlearning/robot_data/speed_traj_data/traj_train_file/Ktest_data_ur.npy').reshape(-1, 18)[:, 6:18]

def get_latent_data():
    enc_f = lka.ENC_net([14] + [128]*3 + [20]).double()
    enc_u = lka.ENC_net([12] + [256]*3 + [20]).double()
    enc_f.load_state_dict(torch.load(path_f, map_location=device)["enc_net1_state_dict"])
    enc_u.load_state_dict(torch.load(path_u, map_location=device)["enc_net2_state_dict"])
    
    with torch.no_grad():
        z_f = enc_f.ENC(torch.DoubleTensor(data_f_raw)).numpy()
        z_u = enc_u.ENC(torch.DoubleTensor(data_u_raw)).numpy()
    return z_f, z_u

z_f, z_u = get_latent_data()

# ================= 3. 绘制 4 合 1 分布对比图 (全称坐标轴版) =================
def plot_combined_distribution_professional():
    fig, axes = plt.subplots(4, 1, figsize=(16, 28))
    
    def draw_kde_final(ax, data, title, palette, x_label):
        num_dims = data.shape[1]
        colors = sns.color_palette(palette, num_dims) 
        for i in range(num_dims):
            sns.kdeplot(data[:, i], ax=ax, color=colors[i], lw=3.5, alpha=0.8)
        ax.set_title(title, pad=25, fontweight='bold')
        ax.set_xlabel(x_label, labelpad=15)
        ax.set_ylabel("Probability Density", labelpad=15)
        ax.grid(axis='y', linestyle='--', alpha=0.4)

    # A. Franka 原始物理空间
    draw_kde_final(axes[0], data_f_raw, "A. Franka Raw Physical State Distribution (14-Dim)", 
                   "husl", "Physical State Magnitude (Joint Position/Velocity)")
    
    # B. UR 原始物理空间
    draw_kde_final(axes[1], data_u_raw, "B. UR Raw Physical State Distribution (12-Dim)", 
                   "husl", "Physical State Magnitude (Joint Position/Velocity)")
    
    # C. Franka 潜在空间
    draw_kde_final(axes[2], z_f, "C. Franka Aligned Latent Feature Distribution (20-Dim)", 
                   "turbo", "Latent Space Activation Value")
    
    # D. UR 潜在空间
    draw_kde_final(axes[3], z_u, "D. UR Aligned Latent Feature Distribution (20-Dim)", 
                   "turbo", "Latent Space Activation Value")

    plt.tight_layout(pad=5.0)
    plt.savefig(os.path.join(save_dir, "3-7_distribution_final_4in1.png"), bbox_inches='tight')
    plt.close()
    print("已生成专业分布对比图: 3-7_distribution_final_4in1.png")

# ================= 4. 绘制独立 t-SNE 拓扑图 (全称坐标轴版) =================
def plot_tsne_professional():
    print("正在计算 t-SNE 拓扑映射...")
    samples = 1500
    z_combined = np.vstack([z_f[:samples], z_u[:samples]])
    tsne = TSNE(n_components=2, perplexity=35, n_iter=1000, random_state=42)
    z_2d = tsne.fit_transform(z_combined)

    plt.figure(figsize=(12, 11))
    plt.scatter(z_2d[:samples, 0], z_2d[:samples, 1], c='#0044FF', alpha=0.6, s=70, 
                label='Source Robot (Franka) Latent', edgecolors='white', linewidth=0.5)
    plt.scatter(z_2d[samples:, 0], z_2d[samples:, 1], c='#FF0000', alpha=0.6, s=70, 
                label='Target Robot (UR) Latent', marker='x', linewidths=2.5)
    
    plt.title("Fig 3-8. Manifold Topology Alignment Verification", pad=25, fontweight='bold')
    plt.xlabel("t-SNE Reduced Dimension 1", labelpad=15)
    plt.ylabel("t-SNE Reduced Dimension 2", labelpad=15)
    plt.legend(frameon=True, shadow=True, facecolor='white', loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "3-8_tsne_final_alignment.png"))
    plt.close()
    print("已生成专业 t-SNE 图: 3-8_tsne_final_alignment.png")

if __name__ == "__main__":
    plot_combined_distribution_professional()
    plot_tsne_professional()
    print(f"所有 3.4 节图片已生成。请检查目录: {save_dir}")