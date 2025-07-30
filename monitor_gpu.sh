#!/bin/bash

# 设置你要运行的 Python 程序路径
PYTHON_SCRIPT="/nasdata/haoyahui/Momo_test/main.py"

# 显存使用阈值（单位：MiB）
THRESHOLD=1000

# 是否已经执行过脚本
EXECUTED=0

while true; do
    echo "====== $(date '+%Y-%m-%d %H:%M:%S') - 当前显存使用情况 ======"

    # 获取每张 GPU 的显存使用情况（单位：MiB）
    USAGES=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)

    INDEX=0
    while IFS= read -r usage; do
        echo "GPU $INDEX: ${usage} MiB"
        if [ "$usage" -lt "$THRESHOLD" ]; then
            echo "GPU $INDEX 使用显存为 ${usage}MiB，低于 ${THRESHOLD}MiB，触发条件"

            if [ "$EXECUTED" -eq 0 ]; then
                echo "运行程序：$PYTHON_SCRIPT"
                python "$PYTHON_SCRIPT" &
                PID=$!
                EXECUTED=1

                # 等待 Python 程序运行完成
                wait $PID
                echo "程序执行完毕，退出监控脚本"
                exit 0
            fi

            break
        fi
        INDEX=$((INDEX + 1))
    done <<< "$USAGES"

    # 每 30 秒检测一次
    sleep 30
done

