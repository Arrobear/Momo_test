#!/bin/bash

# 设置你要运行的 Python 程序路径
PYTHON_SCRIPT="/nasdata/haoyahui/Momo_test/main.py"
PYTHON_SCRIPT_M="/nasdata/haoyahui/Momo_test/monitor_gpu_4.py"

# 显存使用阈值（单位：MiB）
THRESHOLD=5000

# 是否已经执行过脚本
EXECUTED=0

# 输出间隔（秒）
PRINT_INTERVAL=15
LAST_PRINT_TIME=0

while true; do
    CURRENT_TIME=$(date +%s)
    
    # 每 PRINT_INTERVAL 秒输出一次显存情况
    if (( CURRENT_TIME - LAST_PRINT_TIME >= PRINT_INTERVAL )); then
        echo "====== $(date '+%Y-%m-%d %H:%M:%S') - 当前显存使用情况 ======"
        LAST_PRINT_TIME=$CURRENT_TIME
        PRINT_INFO=1
    else
        PRINT_INFO=0
    fi

    # 获取每张 GPU 的显存使用情况（单位：MiB）
    USAGES=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)

    INDEX=0
    AVAILABLE_GPUS=""
    while IFS= read -r usage; do
        if [ "$PRINT_INFO" -eq 1 ]; then
            echo "GPU $INDEX: ${usage} MiB"
        fi
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
        echo "模型程序已启动，PID: $PID"
        EXECUTED=1

        # 等待 Python 程序运行完成
        wait $PID
        echo "程序执行完毕"

        FREE_GPUS="$AVAILABLE_GPUS" python "$PYTHON_SCRIPT_M" &
        MONITOR_PID=$!
        echo "监控脚本已启动，PID: $MONITOR_PID"
        # 等待监控脚本运行完成
        wait $MONITOR_PID
        echo "监控脚本执行完毕，退出脚本"
        exit 0
    fi

    # 检测间隔 0.1 秒
    sleep 0.1
done
