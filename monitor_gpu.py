import torch
import time
import os
import subprocess

# ======== 配置区域 ========
GPU_INDICES = [1]  # 要抢占的 GPU 编号
TARGET_MB = 80000     # 每张卡的目标显存（MiB）
ALLOC_MB = 4096        # 每次分配大小（MiB）

SHELL_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "monitor_gpu.sh"
)
# =========================

tensors = {gpu: [] for gpu in GPU_INDICES}
devices = {gpu: torch.device(f'cuda:{gpu}') for gpu in GPU_INDICES}

# 每次分配的张量大小（元素数量）
elements_per_tensor = (ALLOC_MB * 1024 * 1024) // 4  # float32 占 4 bytes

print(f"开始在 GPU {GPU_INDICES} 上分配显存，每次分配 {ALLOC_MB} MiB，直到任一 GPU 本程序占用超过 {TARGET_MB} MiB")

try:
    while True:
        triggered = False
        for gpu in GPU_INDICES:
            device = devices[gpu]
            allocated = torch.cuda.memory_allocated(device) / (1024 * 1024)
            print(f"[{time.strftime('%H:%M:%S')}] GPU {gpu} 本程序显存占用：{allocated:.2f} MiB")

            if allocated >= TARGET_MB:
                print(f"GPU {gpu} 已达到 {allocated:.2f} MiB，占用完成")
                triggered = True
                break

            # 分配 256MiB 张量
            try:
                tensor = torch.empty(elements_per_tensor, dtype=torch.float32, device=device)
                tensors[gpu].append(tensor)
            except RuntimeError:
                # print(f"GPU {gpu} 分配失败，跳过一次")
                time.sleep(1)

        if triggered:
            while True:
                time.sleep(1)
        time.sleep(0.1)
except KeyboardInterrupt:
    print("被手动中断，释放显存")


