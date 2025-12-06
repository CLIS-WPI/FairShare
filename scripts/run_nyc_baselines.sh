#!/bin/bash
# Run baseline policies (Static, Priority) on GPU 0 for NYC scenario

set -e

SCENARIO="paper_scenario_nyc"
OUTPUT_BASE="results/nyc_baselines"
GPU_ID=0

echo "========================================="
echo "NYC Baseline Simulations (GPU 0)"
echo "========================================="
echo "Scenario: $SCENARIO"
echo "Output: $OUTPUT_BASE"
echo ""

# Create output directories
mkdir -p "$OUTPUT_BASE/static"
mkdir -p "$OUTPUT_BASE/priority"

# Run Static Policy
echo "Running Static Policy..."
CUDA_VISIBLE_DEVICES=$GPU_ID python3 -m src.main \
    --scenario "$SCENARIO" \
    --policy static \
    --output "$OUTPUT_BASE/static" \
    --gpu-id $GPU_ID

echo "✓ Static Policy complete"
echo ""

# Run Priority Policy
echo "Running Priority Policy..."
CUDA_VISIBLE_DEVICES=$GPU_ID python3 -m src.main \
    --scenario "$SCENARIO" \
    --policy priority \
    --output "$OUTPUT_BASE/priority" \
    --gpu-id $GPU_ID

echo "✓ Priority Policy complete"
echo ""

echo "========================================="
echo "Baseline simulations complete!"
echo "Results saved to: $OUTPUT_BASE"
echo "========================================="

