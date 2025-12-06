#!/bin/bash
# GPU Monitoring Script for NYC Simulation
# Monitors GPU utilization in real-time during simulation

CONTAINER_NAME="fairness-dev"
INTERVAL=1  # Check every 1 second
DURATION=${1:-600}  # Default 10 minutes

echo "=== GPU Monitoring Started ==="
echo "Container: $CONTAINER_NAME"
echo "Interval: ${INTERVAL}s"
echo "Duration: ${DURATION}s"
echo "Press Ctrl+C to stop"
echo ""

start_time=$(date +%s)
end_time=$((start_time + DURATION))

while [ $(date +%s) -lt $end_time ]; do
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Get GPU stats
    gpu_stats=$(docker exec $CONTAINER_NAME nvidia-smi \
        --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu \
        --format=csv,noheader,nounits 2>/dev/null)
    
    if [ $? -eq 0 ]; then
        echo "[$timestamp] GPU Status:"
        echo "$gpu_stats" | while IFS=',' read -r idx util_gpu util_mem mem_used mem_total temp; do
            echo "  GPU $idx: ${util_gpu}% | Mem: ${mem_used}/${mem_total}MB (${util_mem}%) | Temp: ${temp}°C"
        done
    else
        echo "[$timestamp] Error: Cannot access GPU stats"
    fi
    
    # Check if simulation process is running
    sim_running=$(docker exec $CONTAINER_NAME ps aux | grep "python3 -m src.main" | grep -v grep | wc -l)
    if [ "$sim_running" -gt 0 ]; then
        echo "  Simulation: RUNNING"
    else
        echo "  Simulation: NOT RUNNING"
    fi
    
    echo ""
    sleep $INTERVAL
done

echo "=== Monitoring Complete ==="

