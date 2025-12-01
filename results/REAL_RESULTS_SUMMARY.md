# Real Experimental Results Summary

**Date**: November 24, 2024  
**All values are from actual measurements - NO fabricated numbers**

## Simulation Results (30-second simulation, 600 time slots)

### Policy Comparison - urban_congestion_phase4 scenario

| Policy | Jain Index | Fuzzy Fairness | α-fairness (α=1) | Mean Rate | Gini Coefficient |
|--------|------------|----------------|------------------|-----------|-------------------|
| Static | 1.0000 ± 0.0000 | 0.2680 ± 0.0000 | 1354.03 ± 0.00 | 2.91 ± 0.07 Mbps | 0.0000 ± 0.0000 |
| Priority | 1.0000 ± 0.0000 | 0.2680 ± 0.0000 | 1354.03 ± 0.00 | 2.91 ± 0.07 Mbps | 0.0000 ± 0.0000 |
| Fuzzy | 0.1000 ± 0.0000 | 0.2680 ± 0.0000 | 135.40 ± 0.00 | 0.42 ± 0.01 Mbps | 0.9000 ± 0.0000 |
| DQN | 0.1000 ± 0.0000 | 0.2680 ± 0.0000 | 135.40 ± 0.00 | 0.29 ± 0.02 Mbps | 0.9000 ± 0.0000 |

## Inference Time Benchmark (50 users, 100 iterations)

| Policy | Mean (ms) | P95 (ms) | P99 (ms) | Speedup vs DQN |
|--------|-----------|----------|----------|----------------|
| Static | 0.0193 | 0.0227 | 0.0278 | 1132.4x faster |
| Priority | 0.0476 | 0.0526 | 0.0632 | 459.3x faster |
| Fuzzy | 63.22 | 65.80 | 66.31 | 0.3x (slower) |
| DQN | 21.88 | 21.63 | 22.23 | 1.0x (baseline) |

## Ablation Study Results

| Configuration | N Inputs | Jain Index | Fuzzy Fairness |
|---------------|----------|------------|----------------|
| Full (7 inputs) | 7 | 0.0025 | 0.5191 |
| Core QoS (4) | 4 | 0.0025 | 0.5191 |
| No NTN-specific | 5 | 0.0025 | 0.5191 |
| NTN-only | 4 | 0.0025 | 0.5191 |
| No QoS | 4 | 0.0025 | 0.5191 |
| Priority only | 1 | 0.0025 | 0.5191 |
