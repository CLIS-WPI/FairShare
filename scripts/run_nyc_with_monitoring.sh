#!/bin/bash
# Run NYC simulation with GPU monitoring

CONTAINER_NAME="fairness-dev"
CONFIG_FILE="paper_scenario_nyc"
POLICY=${1:-static}
OUTPUT_DIR="results/nyc_${POLICY}"
GPU_ID=${2:-0}

echo "=== Starting NYC Simulation with GPU Monitoring ==="
echo "Policy: $POLICY"
echo "GPU: $GPU_ID"
echo "Output: $OUTPUT_DIR"
echo ""

# Start monitoring in background
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$SCRIPT_DIR/monitor_gpu.sh" 600 > "/tmp/gpu_monitor_${POLICY}.log" 2>&1 &
MONITOR_PID=$!
echo "GPU Monitor started (PID: $MONITOR_PID)"
echo "Monitor log: /tmp/gpu_monitor_${POLICY}.log"
echo ""

# Run simulation
echo "Starting simulation..."
docker exec $CONTAINER_NAME bash -c "cd /workspace && CUDA_VISIBLE_DEVICES=$GPU_ID python3 -m src.main --scenario $CONFIG_FILE --policy $POLICY --output $OUTPUT_DIR --gpu-id $GPU_ID" 2>&1 | tee "/tmp/nyc_${POLICY}_run.log"

SIM_EXIT_CODE=$?

# Stop monitoring
kill $MONITOR_PID 2>/dev/null
wait $MONITOR_PID 2>/dev/null

echo ""
echo "=== Simulation Complete ==="
echo "Exit code: $SIM_EXIT_CODE"
echo "Results: $OUTPUT_DIR"
echo "Log: /tmp/nyc_${POLICY}_run.log"
echo "GPU Monitor log: /tmp/gpu_monitor_${POLICY}.log"

# Show final GPU stats
echo ""
echo "Final GPU Status:"
docker exec $CONTAINER_NAME nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null

# Check results
if [ -d "/home/tanglab/Desktop/fuzzy-fairness-dss-leo/results/nyc_${POLICY}" ]; then
    echo ""
    echo "Output files:"
    ls -lh "/home/tanglab/Desktop/fuzzy-fairness-dss-leo/results/nyc_${POLICY}/" 2>/dev/null | tail -5
fi

exit $SIM_EXIT_CODE

