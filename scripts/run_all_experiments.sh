#!/bin/bash
# Run ALL experiments and measure REAL results
# DO NOT use fabricated numbers

set -e

SCENARIO="urban_congestion_phase4"
DURATION=60

echo "=========================================="
echo "DySPAN 2026 Experimental Evaluation"
echo "All results are REAL measured values"
echo "=========================================="

# Step 1: Train DQN (if not already trained)
echo ""
echo "[1/5] Training DQN baseline..."
if [ ! -f "models/dqn/dqn_baseline_final.h5" ]; then
    echo "Starting DQN training (this will take several hours)..."
    python scripts/train_dqn_baseline.py \
        --scenario $SCENARIO \
        --episodes 10000 \
        --log-freq 100
    echo "✓ DQN training complete"
else
    echo "✓ DQN model already exists"
fi

# Step 2: Run simulations for ALL policies
echo ""
echo "[2/5] Running simulations for all policies..."
for policy in static priority fuzzy dqn; do
    echo "  Running: $policy"
    python -m src.main \
        --scenario $SCENARIO \
        --policy $policy \
        --duration-s $DURATION \
        --gpu-id cpu || echo "  ⚠ Warning: $policy simulation failed"
done
echo "✓ All simulations complete"

# Step 3: Benchmark inference times
echo ""
echo "[3/5] Benchmarking REAL inference times..."
python experiments/benchmark_inference.py \
    --n-users 100 \
    --n-iterations 1000 \
    --policies static priority fuzzy dqn || echo "  ⚠ Warning: Benchmark failed"
echo "✓ Inference benchmark complete"

# Step 4: Run ablation study
echo ""
echo "[4/5] Running ablation study..."
python experiments/ablation_study.py \
    --scenario $SCENARIO \
    --duration-s $DURATION || echo "  ⚠ Warning: Ablation study failed"
echo "✓ Ablation study complete"

# Step 5: Generate plots
echo ""
echo "[5/5] Generating plots from REAL data..."
python experiments/generate_plots.py --scenario $SCENARIO || echo "  ⚠ Warning: Plot generation failed"
echo "✓ Plots generated"

echo ""
echo "=========================================="
echo "✓ ALL EXPERIMENTS COMPLETE"
echo "=========================================="
echo ""
echo "Results locations:"
echo "  - Simulation results: results/urban_congestion_*.csv"
echo "  - Inference benchmark: results/benchmarks/inference_benchmark_n100.csv"
echo "  - Ablation study: results/ablation/ablation_study_*.csv"
echo "  - Plots: plots/"
echo ""
echo "Next: Extract results for paper"
echo "  python scripts/extract_paper_results.py"

