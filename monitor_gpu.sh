#!/bin/bash

# 设置你要运行的 Python 程序路径
PYTHON_SCRIPT="/nasdata/haoyahui/Momo_test/main.py"

PYTHON_SCRIPT_M="/nasdata/haoyahui/Momo_test/monitor_gpu.py"

# 显存使用阈值（单位：MiB）
THRESHOLD=5000

# 是否已经执行过脚本
EXECUTED=0

while true; do
    echo "====== $(date '+%Y-%m-%d %H:%M:%S') - 当前显存使用情况 ======"

    # 获取每张 GPU 的显存使用情况（单位：MiB）
    USAGES=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)

    INDEX=0
    AVAILABLE_GPUS=""
    while IFS= read -r usage; do
        echo "GPU $INDEX: ${usage} MiB"
        if [ "$usage" -lt "$THRESHOLD" ]; then
            AVAILABLE_GPUS="$AVAILABLE_GPUS $INDEX"
        fi
        INDEX=$((INDEX + 1))
    done <<< "$USAGES"

    # 去掉前后空格
    AVAILABLE_GPUS=$(echo "$AVAILABLE_GPUS" | xargs)

    if [ -n "$AVAILABLE_GPUS" ] && [ "$EXECUTED" -eq 0 ]; then
        echo "可用 GPU 编号：$AVAILABLE_GPUS"
        echo "运行程序：$PYTHON_SCRIPT"

        # 通过环境变量传给 Python
        FREE_GPUS="$AVAILABLE_GPUS" python "$PYTHON_SCRIPT" &
        PID=$!
        EXECUTED=1

        # 等待 Python 程序运行完成
        wait $PID
        echo "程序执行完毕"

        python "$PYTHON_SCRIPT_M" &
        MONITOR_PID=$!
        echo "监控脚本已启动，PID: $MONITOR_PID"
        # 等待监控脚本运行完成
        wait $MONITOR_PID
        echo "监控脚本执行完毕，退出脚本"
        exit 0
    fi

    sleep 1
done