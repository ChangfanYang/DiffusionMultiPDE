import matplotlib.pyplot as plt
from scipy.io import loadmat

# 读取mat文件
data = loadmat('darcy_results.mat')

# 提取a和u数据
a_data = data['a']
u_data = data['u']

# 获取数据的维度信息
num_slices = a_data.shape[2]  # 假设第三维是切片维度

# 创建一个包含多个子图的图形
fig, axes = plt.subplots(2, num_slices, figsize=(15, 6))

# 绘制a数据的不同切片
for i in range(num_slices):
    axes[0, i].pcolormesh(a_data[:, :, i])
    axes[0, i].set_title(f'a data - slice {i}')
    axes[0, i].axis('square')

# 绘制u数据的不同切片
for i in range(num_slices):
    axes[1, i].pcolormesh(u_data[:, :, i])
    axes[1, i].set_title(f'u data - slice {i}')
    axes[1, i].axis('square')

# 调整布局
plt.tight_layout()

# 设置保存的图片文件名和格式，这里以保存为PNG格式为例，你也可以根据需求换成如 'results.jpg'（保存为JPEG格式）等
save_filename = '/home/yangchangfan/CODE/DiffusionPDE/darcy_results.png'
# 保存图片
plt.savefig(save_filename)
print(f"图片已成功保存为 {save_filename}")