#!/bin/bash
# Run all policies for NYC scenario comparison

CONTAINER_NAME="fairness-dev"
CONFIG_FILE="paper_scenario_nyc"
OUTPUT_BASE="results/nyc_comparison"

echo "=== Running All Policies for NYC Scenario ==="
echo "Scenario: $CONFIG_FILE"
echo "Output: $OUTPUT_BASE"
echo ""

# Create output directory
docker exec $CONTAINER_NAME bash -c "mkdir -p /workspace/$OUTPUT_BASE"

# 1. Static Policy (already done, but we can verify)
echo "1. Static Policy (GPU 0)..."
if [ ! -f "/home/tanglab/Desktop/fuzzy-fairness-dss-leo/results/nyc_static/nyc_manhattan_scale_static.csv" ]; then
    docker exec -d $CONTAINER_NAME bash -c "cd /workspace && CUDA_VISIBLE_DEVICES=0 python3 -m src.main --scenario $CONFIG_FILE --policy static --output $OUTPUT_BASE/static --gpu-id 0 > $OUTPUT_BASE/static_log.txt 2>&1" &
    STATIC_PID=$!
    echo "   Started (PID: $STATIC_PID)"
else
    echo "   ✓ Already completed"
fi

# 2. Priority Policy
echo "2. Priority Policy (GPU 0)..."
if [ ! -f "/home/tanglab/Desktop/fuzzy-fairness-dss-leo/results/nyc_priority/nyc_manhattan_scale_priority.csv" ]; then
    docker exec -d $CONTAINER_NAME bash -c "cd /workspace && CUDA_VISIBLE_DEVICES=0 python3 -m src.main --scenario $CONFIG_FILE --policy priority --output $OUTPUT_BASE/priority --gpu-id 0 > $OUTPUT_BASE/priority_log.txt 2>&1" &
    PRIORITY_PID=$!
    echo "   Started (PID: $PRIORITY_PID)"
else
    echo "   ✓ Already completed"
fi

# 3. DQN Policy (GPU 1 for training)
echo "3. DQN Policy (GPU 1)..."
echo "   Note: DQN requires training first, then evaluation"
echo "   This will take longer due to training phase"

# Check if DQN model exists
DQN_MODEL="/workspace/models/dqn/nyc_dqn_model.h5"
MODEL_EXISTS=$(docker exec $CONTAINER_NAME bash -c "test -f $DQN_MODEL && echo 'yes' || echo 'no'")

if [ "$MODEL_EXISTS" = "no" ]; then
    echo "   Training DQN model (this may take a while)..."
    docker exec -d $CONTAINER_NAME bash -c "cd /workspace && CUDA_VISIBLE_DEVICES=1 python3 -m src.train_rl --scenario $CONFIG_FILE --algo dqn --episodes 500 --gpu-id 1 --output $OUTPUT_BASE/dqn_training --model-save-path $DQN_MODEL > $OUTPUT_BASE/dqn_training_log.txt 2>&1" &
    DQN_TRAIN_PID=$!
    echo "   Training started (PID: $DQN_TRAIN_PID)"
    echo "   Waiting for training to complete..."
    wait $DQN_TRAIN_PID
    echo "   ✓ Training completed"
fi

# Evaluate DQN
echo "   Evaluating DQN model..."
docker exec -d $CONTAINER_NAME bash -c "cd /workspace && CUDA_VISIBLE_DEVICES=1 python3 -m src.main --scenario $CONFIG_FILE --policy dqn --output $OUTPUT_BASE/dqn --gpu-id 1 --dqn-model-path $DQN_MODEL > $OUTPUT_BASE/dqn_eval_log.txt 2>&1" &
DQN_EVAL_PID=$!
echo "   Evaluation started (PID: $DQN_EVAL_PID)"

echo ""
echo "=== All Policies Started ==="
echo "Monitor progress:"
echo "  docker exec $CONTAINER_NAME tail -f $OUTPUT_BASE/*_log.txt"
echo ""
echo "Check GPU usage:"
echo "  docker exec $CONTAINER_NAME nvidia-smi"
echo ""
echo "Results will be in: $OUTPUT_BASE/"

