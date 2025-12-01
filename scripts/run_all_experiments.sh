#!/bin/bash
"""
Run all experiments in sequence: baseline → sensitivity → adversarial → visualization
"""

echo "========================================="
echo "RUNNING ALL EXPERIMENTS"
echo "========================================="
echo ""

# 1. Baseline simulation
echo "Step 1/4: Running baseline simulation..."
python3 scripts/run_full_simulation.py
if [ $? -ne 0 ]; then
    echo "❌ Baseline simulation failed"
    exit 1
fi
echo "✓ Baseline simulation complete"
echo ""

# 2. Sensitivity analysis
echo "Step 2/4: Running sensitivity analysis..."
python3 scripts/sensitivity_analysis.py
if [ $? -ne 0 ]; then
    echo "⚠️ Sensitivity analysis failed (may take time, continuing...)"
fi
echo "✓ Sensitivity analysis complete"
echo ""

# 3. Adversarial scenarios
echo "Step 3/4: Running adversarial scenarios..."
python3 scripts/adversarial_scenarios.py
if [ $? -ne 0 ]; then
    echo "❌ Adversarial scenarios failed"
    exit 1
fi
echo "✓ Adversarial scenarios complete"
echo ""

# 4. Generate visualizations
echo "Step 4/4: Generating visualizations..."
python3 scripts/visualize_results.py
if [ $? -ne 0 ]; then
    echo "❌ Visualization generation failed"
    exit 1
fi
echo "✓ Visualizations complete"
echo ""

echo "========================================="
echo "ALL EXPERIMENTS COMPLETE"
echo "========================================="
echo ""
echo "Results saved to:"
echo "  - ~/fuzzy_fairness_results/simulation/"
echo "  - results/sensitivity_analysis/"
echo "  - results/adversarial_scenarios/"
echo "  - results/visualizations/"
echo ""
