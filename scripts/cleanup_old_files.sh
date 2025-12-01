#!/bin/bash
# Cleanup script for old project files
# Run with caution - lists files before removing

set -e

echo "========================================="
echo "Cleaning Up Old Project Files"
echo "========================================="
echo ""

# Files to remove (old fuzzy-specific and documentation)
FILES_TO_REMOVE=(
    # Old fuzzy-specific
    "src/fairness/fuzzy_core.py"
    "src/fairness/fuzzy_core_gpu.py"
    "src/fairness/membership.py"
    "src/fairness/membership_phase3.py"
    "src/fairness/rule_base.py"
    "src/fairness/rule_base_phase3.py"
    "src/dss/policies/fuzzy_adaptive.py"
    "src/dss/policies/fuzzy_adaptive_gpu.py"
    "src/dss/policies/load_adaptive.py"
    
    # Old documentation
    "PHASE1_IMPLEMENTATION.md"
    "PHASE2_IMPLEMENTATION.md"
    "PHASE3_IMPLEMENTATION.md"
    "PHASE4_IMPLEMENTATION.md"
    "PHASE5_IMPLEMENTATION.md"
    "PHASE6_IMPLEMENTATION.md"
    "BUG_FIXES_FINAL.md"
    "CRITICAL_ISSUES_ANALYSIS.md"
    "DEBUGGING_ACTION_PLAN.md"
    "DIAGNOSIS_RESULTS_ANALYSIS.md"
    "DIAGNOSIS_SUMMARY.md"
    "FINAL_FIXES_SUMMARY.md"
    "FIXES_APPLIED.md"
    "FIXES_STATUS.md"
    "PRIORITY_FIX_FINAL.md"
    "PRIORITY_POLICY_FIX.md"
    "PRIORITY_RESULTS_ANALYSIS.md"
    "CONSTRAINT_ADJUSTMENT.md"
    "STATUS_REPORT.md"
    "RESULTS_VERIFICATION.md"
    "STRUCTURE_CHECK.md"
    "PAPER_ARTIFACTS.md"
    
    # Old logs and temp files
    "*.log"
    "*.pid"
    "dqn_training.log"
    "priority_*.log"
    "simulation.log"
    
    # Old DQN models (will recreate)
    "models/dqn/"
)

# Directories to clean (keep structure, remove contents)
DIRS_TO_CLEAN=(
    "results/urban_congestion_*.csv"
    "results/paper_tables/"
)

echo "Files to be removed:"
for file in "${FILES_TO_REMOVE[@]}"; do
    if [ -e "$file" ] || [ -n "$(find . -name "$file" 2>/dev/null)" ]; then
        echo "  - $file"
    fi
done

echo ""
read -p "Continue with cleanup? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cleanup cancelled."
    exit 0
fi

echo ""
echo "Removing files..."
for file in "${FILES_TO_REMOVE[@]}"; do
    if [[ "$file" == *"*"* ]]; then
        # Handle glob patterns
        find . -name "$file" -type f -delete 2>/dev/null || true
    elif [ -f "$file" ]; then
        rm -f "$file"
        echo "  ✓ Removed: $file"
    elif [ -d "$file" ]; then
        rm -rf "$file"
        echo "  ✓ Removed directory: $file"
    fi
done

echo ""
echo "Cleaning result directories..."
for pattern in "${DIRS_TO_CLEAN[@]}"; do
    find results -name "$pattern" -type f -delete 2>/dev/null || true
done

echo ""
echo "========================================="
echo "Cleanup complete!"
echo "========================================="

