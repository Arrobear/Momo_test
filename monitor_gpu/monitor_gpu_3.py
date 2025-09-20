import torch
import time
import subprocess
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import stage_1_function
import generate_prompt

# ======== 配置区域 ========
print(config.gpu_ids)

def monitor_gpu():
    GPU_INDICES = [config.gpu_ids[0]]  # 要抢占的 GPU 编号
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
    last_log_time = 0  # 上一次打印时间
    try:
        while True:
            triggered = False
            for gpu in GPU_INDICES:
                device = devices[gpu]
                allocated = torch.cuda.memory_allocated(device) / (1024 * 1024)

                now = time.time()
                if now - last_log_time >= 2:  # 每隔 2 秒打印一次
                    print(f"[{time.strftime('%H:%M:%S')}] GPU {gpu} 本程序显存占用：{allocated:.2f} MiB")
                    last_log_time = now

                if allocated >= TARGET_MB:
                    print(f"GPU {gpu} 已达到 {allocated:.2f} MiB，占用完成")
                    triggered = True
                    break

                try:
                    tensor = torch.empty(elements_per_tensor, dtype=torch.float32, device=device)
                    tensors[gpu].append(tensor)
                except RuntimeError:
                    time.sleep(15)
                if os.path.exists("stop_3.txt"):
                    print("检测到 stop.txt，程序退出。")
                    return

            if triggered:
                print("占满后等待中，按 Ctrl+C 或 touch stop.txt 可退出")
                while True:
                    if os.path.exists("stop_3.txt"):
                        print("检测到 stop.txt，程序退出。")
                        return
                    time.sleep(1)
                break  # 结束整个程序

            time.sleep(0.1)
    except KeyboardInterrupt:
        print("被手动中断，释放显存")
monitor_gpu()