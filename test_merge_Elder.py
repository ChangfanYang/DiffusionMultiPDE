import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os

# === 路径配置 ===
merged_file = "/data/yangchangfan/DiffusionPDE/data/Elder-merged/merge_1.npy"
gt_base_path = "/data/yangchangfan/DiffusionPDE/data/training/Elder/"
save_path = "test_merge_elder.png"

# === 加载归一化后数据 ===
combined_data = np.load(merged_file)
# pred_fields = combined_data[:, :, 0:4]   # S_c, u_u_0, u_v_0, c_flow_0
pred_fields = combined_data[:, :, [0, 4, 14, 24]] # S_c_0, u_u_1, u_v_1, c_flow_1


# === 加载归一化范围 ===
ranges = {
    "S_c": loadmat(os.path.join(gt_base_path, "S_c", "range_allS_c.mat"))["range_allS_c"][0],
    "u_u": loadmat(os.path.join(gt_base_path, "u_u", "range_allu_u.mat"))["range_allu_u"][0],
    "u_v": loadmat(os.path.join(gt_base_path, "u_v", "range_allu_v.mat"))["range_allu_v"][0],
    "c_flow": loadmat(os.path.join(gt_base_path, "c_flow", "range_allc_flow.mat"))["range_allc_flow"][0],
}

# === 反归一化函数 ===
def minmax_denormalize(x, min_val, max_val):
    return (x + 0.9) * (max_val - min_val) / 1.8 + min_val

# === 执行反归一化 ===
denorm_fields = [
    minmax_denormalize(pred_fields[:, :, 0], *ranges["S_c"]),
    minmax_denormalize(pred_fields[:, :, 1], *ranges["u_u"]),
    minmax_denormalize(pred_fields[:, :, 2], *ranges["u_v"]),
    minmax_denormalize(pred_fields[:, :, 3], *ranges["c_flow"]),
]

# === 加载 Ground Truth 原始数据 ===
gt_fields = [
    list(loadmat(os.path.join(gt_base_path, "S_c", "1", "0.mat")).values())[-1],
    list(loadmat(os.path.join(gt_base_path, "u_u", "1", "1.mat")).values())[-1],
    list(loadmat(os.path.join(gt_base_path, "u_v", "1", "1.mat")).values())[-1],
    list(loadmat(os.path.join(gt_base_path, "c_flow", "1", "1.mat")).values())[-1],
]

# === 可视化对比 ===
titles = ["S_c", "u_u_0", "u_v_0", "c_flow_0"]
fig, axs = plt.subplots(2, 4, figsize=(16, 8))

for i in range(4):
    im_pred = axs[0, i].imshow(denorm_fields[i], cmap='viridis')
    axs[0, i].set_title(f"Predicted {titles[i]}")
    axs[0, i].axis('off')
    plt.colorbar(im_pred, ax=axs[0, i], fraction=0.046, pad=0.04)

    im_gt = axs[1, i].imshow(gt_fields[i], cmap='viridis')
    axs[1, i].set_title(f"Ground Truth {titles[i]}")
    axs[1, i].axis('off')
    plt.colorbar(im_gt, ax=axs[1, i], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(save_path)
plt.close()

print(f"图像已保存至: {save_path}")
