#!/bin/bash
INTERVAL=${1:-10}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/monitor_cpu.sh"

if ! [[ "$INTERVAL" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
   ! awk -v interval="$INTERVAL" 'BEGIN {exit !(interval > 0)}'; then
    echo "Usage: bash scripts/monitor.sh [positive interval in seconds]" >&2
    exit 1
fi

init_cpu_monitor
clear
previous=$(cpu_snapshot 2>/dev/null) || previous=''

while true; do
    sleep "$INTERVAL"
    current=$(cpu_snapshot 2>/dev/null) || current=''
    capacity=$(cpu_capacity 2>/dev/null) || capacity=''
    cpu_line="CPU 사용량 (cgroup): 측정 불가"
    if [ -n "$current" ] && [ -n "$capacity" ]; then
        cpu_line="CPU 사용량 (cgroup): 측정 대기"
        if [ -n "$previous" ]; then
            read -r prev_time prev_usage <<< "$previous"
            read -r current_time current_usage <<< "$current"
            cpu_line=$(format_cpu_usage "$prev_time" "$prev_usage" "$current_time" "$current_usage" "$capacity")
        fi
    fi
    previous=$current

    echo "=== 시스템 모니터링 ($(date +%H:%M:%S)) ==="
    echo "$cpu_line"

    # RAM 사용량
    ram_info=$(free -m | awk '/Mem:/ {print $3 "/" $2 " MB"}')
    echo "RAM 사용량 (free 기준, 호스트 포함 가능): $ram_info"

    # GPU 정보
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader | \
        awk -F', ' '{print "VRAM 사용량: "$1" / "$2" | GPU 사용률: "$3}'
    else
        echo "GPU 정보: nvidia-smi 없음"
    fi

    echo
done
