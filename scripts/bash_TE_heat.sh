#!/bin/bash

# 检查 GNU Parallel 是否安装
if ! command -v parallel &> /dev/null; then
    echo "GNU Parallel 未安装，请先运行: sudo apt-get install parallel (Linux) 或 brew install parallel (Mac)"
    exit 1
fi

# Python 脚本路径（不含 CUDA_VISIBLE_DEVICES）
PYTHON_SCRIPT="/home/yangchangfan/anaconda3/envs/DiffusionPDE/bin/python3 /home/yangchangfan/CODE/DiffusionPDE/generate_pde.py --config /home/yangchangfan/CODE/DiffusionPDE/configs/TE_heat.yaml"

# 配置参数（你可以根据需要修改这些值）
START_ID=30001
END_ID=31001
BATCH_SIZE=144      # 每个子任务处理多少个样本
# GPUS=(1 3 2 4)      # 使用的 GPU 编号（可以是任意顺序）
GPUS=(1 2 3 4 5 6 7)       # 使用的 GPU 编号（可以是任意顺序）
gpu_count=7


# 计算总样本数和任务数
TOTAL_SAMPLES=$((END_ID - START_ID + 1))
TOTAL_TASKS=$(( (TOTAL_SAMPLES + BATCH_SIZE - 1) / BATCH_SIZE ))

# 存储所有任务参数
JOBS=()

current_id=$START_ID
for ((t = 0; t < TOTAL_TASKS; t++)); do
    start=$current_id
    end=$((start + BATCH_SIZE - 1))
    if ((end > END_ID)); then end=$END_ID; fi

    JOBS+=("$start $end")
    current_id=$((end + 1))
done

# 打印任务总数
echo "总共生成了 $TOTAL_TASKS 个任务，每个任务处理最多 $BATCH_SIZE 个样本。"

# 构建任务分配表
TASK_ASSIGNMENTS=()

gpu_index=0
job_per_gpu=0

for task in "${JOBS[@]}"; do
    gpu_idx=$((gpu_index % gpu_count))
    gpu_id=${GPUS[$gpu_idx]}

    gpu_index=$(($gpu_index+1))

    TASK_ASSIGNMENTS+=("$gpu_id $task")
done



# 输出调试信息（可注释掉）
printf "%s\n" "${TASK_ASSIGNMENTS[@]}" | nl

# 使用 GNU Parallel 并行执行
printf "%s\n" "${TASK_ASSIGNMENTS[@]}" | parallel -j+0 --colsep ' ' '
    CUDA_VISIBLE_DEVICES={1} '"$PYTHON_SCRIPT"' --start {2} --end {3}
'