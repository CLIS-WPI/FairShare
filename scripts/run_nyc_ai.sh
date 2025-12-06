#!/bin/bash
# Train and run AI policy on GPU 1 for NYC scenario

set -e

SCENARIO="paper_scenario_nyc"
OUTPUT_BASE="results/nyc_ai"
GPU_ID=1
EPISODES=500

echo "========================================="
echo "NYC AI Training & Evaluation (GPU 1)"
echo "========================================="
echo "Scenario: $SCENARIO"
echo "Output: $OUTPUT_BASE"
echo "Episodes: $EPISODES"
echo ""

# Create output directories
mkdir -p "$OUTPUT_BASE/training"
mkdir -p "$OUTPUT_BASE/evaluation"

# Training Phase
echo "Training DQN agent..."
CUDA_VISIBLE_DEVICES=$GPU_ID python3 -m scripts.train_dqn_baseline \
    --scenario "$SCENARIO" \
    --episodes $EPISODES \
    --gpu-id $GPU_ID \
    --output "$OUTPUT_BASE/training"

echo "✓ Training complete"
echo ""

# Evaluation Phase
echo "Running trained DQN agent..."
CUDA_VISIBLE_DEVICES=$GPU_ID python3 -m src.main \
    --scenario "$SCENARIO" \
    --policy dqn_trained \
    --output "$OUTPUT_BASE/evaluation" \
    --gpu-id $GPU_ID \
    --dqn-model "$OUTPUT_BASE/training/models/dqn_baseline_final.h5"

echo "✓ Evaluation complete"
echo ""

echo "========================================="
echo "AI training & evaluation complete!"
echo "Results saved to: $OUTPUT_BASE"
echo "========================================="

