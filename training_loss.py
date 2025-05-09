import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# 读取日志文件
logs = []
with open("/home/yangchangfan/CODE/DiffusionPDE/pretrained-TE_heat_3w/00000--uncond-ddpmpp-edm-gpus8-batch96-fp32/stats.jsonl", "r") as file:
    for line in file:
        line = line.strip()  # 去除首尾空格
        if line:  # 确保行不为空
            try:
                log = json.loads(line)
                logs.append(log)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line: {line}. Error: {e}")

# 提取数据
ticks = [log["Progress/tick"]["mean"] for log in logs]  # 提取进度（tick）
losses = [log["Loss/loss"]["mean"] for log in logs]  # 提取损失值

plt.figure(figsize=(10, 6))
plt.plot(ticks, losses, label="Training Loss", color="blue", marker="o", linestyle="-")
plt.title("Training Loss", fontsize=16)
plt.xlabel("Training Tick", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend(fontsize=12)
plt.savefig("training_loss.png", dpi=300)



# # 读取日志文件
# import json
# import matplotlib.pyplot as plt

# # 读取日志文件
# logs = []
# with open("/home/yangchangfan/CODE/DiffusionPDE/inference_losses.jsonl", "r") as file:
#     for line in file:
#         line = line.strip()
#         if line:
#             try:
#                 log = json.loads(line)
#                 logs.append(log)
#             except json.JSONDecodeError as e:
#                 print(f"Warning: Failed to parse line: {line}. Error: {e}")

# # 提取数据
# steps = [log["step"] for log in logs]  # 提取步数
# L_pde = [log["L_pde"] for log in logs]  # 提取 L_pde
# L_obs_mater = [log["L_obs_mater"] for log in logs]  # 提取 L_obs_mater
# L_obs_Ez = [log["L_obs_Ez"] for log in logs]  # 提取 L_obs_Ez
# L_obs_T = [log["L_obs_T"] for log in logs]  # 提取 L_obs_T


# plt.figure(figsize=(10, 6))
# plt.plot(steps, L_pde, label="L_pde", color="blue", marker="o", linestyle="-")
# plt.plot(steps, L_obs_mater, label="L_obs_mater", color="green", marker="s", linestyle="-")
# plt.plot(steps, L_obs_Ez, label="L_obs_Ez", color="red", marker="^", linestyle="-")
# plt.plot(steps, L_obs_T, label="L_obs_T", color="purple", marker="x", linestyle="-")
# plt.title("Losses Over Steps", fontsize=16)
# plt.xlabel("Step", fontsize=14)
# plt.ylabel("Loss", fontsize=14)
# plt.legend(fontsize=12)
# plt.grid(True, linestyle="--", alpha=0.7)
# plt.savefig("inference loss.png", dpi=300)
# plt.show()