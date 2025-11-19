# Running Experiments Inside Docker Container

## Quick Start

Once you're inside the Docker container (`/workspace`), you can run experiments with these commands:

### 1. Quick Test (30 seconds, CPU mode)

```bash
python -m src.main \
  --scenario urban_congestion_phase4 \
  --policy fuzzy \
  --gpu-id cpu \
  --duration-s 30
```

### 2. Full Simulation (10 minutes, GPU)

```bash
python -m src.main \
  --scenario urban_congestion_phase4 \
  --policy fuzzy \
  --gpu-id 0 \
  --duration-s 600
```

### 3. Compare All Policies

```bash
# Static policy
python -m src.main --scenario urban_congestion_phase4 --policy static --duration-s 600

# Priority policy
python -m src.main --scenario urban_congestion_phase4 --policy priority --duration-s 600

# Fuzzy policy
python -m src.main --scenario urban_congestion_phase4 --policy fuzzy --duration-s 600
```

## Available Scenarios

- `urban_congestion_phase4` - Dense urban (500 users, 3 operators)
- `rural_coverage_phase4` - Sparse rural (100 users, 2 operators)
- `emergency_response_phase4` - Emergency (200 users, bursty traffic)

## Available Policies

- `static` - Equal allocation
- `priority` - Priority-based allocation
- `fuzzy` - Fuzzy adaptive allocation (recommended)

## Command-Line Options

```bash
python -m src.main \
  --scenario SCENARIO_NAME    # Scenario name or YAML path
  --policy POLICY_NAME         # static, priority, or fuzzy
  --gpu-id GPU_ID              # 0, 1, ... or "cpu"
  --duration-s SECONDS         # Override simulation duration
  --output OUTPUT_DIR          # Output directory (default: results)
```

## Generate Plots

After running simulations, generate plots:

```bash
python experiments/generate_plots.py --scenario urban_congestion_phase4
```

This creates plots in `plots/` directory:
- `fairness_time_*.pdf` - Fairness over time
- `policy_comparison_*.pdf` - Policy comparison
- `rate_cdf_*.pdf` - Rate CDF
- `operator_imbalance_heat_*.pdf` - Operator imbalance
- `doppler_fairness_scatter_*.pdf` - Doppler vs Fairness

## Check Results

Results are saved to `results/` directory as CSV files:

```bash
# List results
ls -lh results/

# View a result file
head results/urban_congestion_phase4_fuzzy.csv
```

## Interactive Notebook

Start Jupyter Lab:

```bash
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

Then open `notebooks/interactive_demo.ipynb` in the browser.

## Example Workflow

```bash
# 1. Run quick test
python -m src.main --scenario urban_congestion_phase4 --policy fuzzy --gpu-id cpu --duration-s 30

# 2. Check results
ls -lh results/

# 3. Generate plots
python experiments/generate_plots.py --scenario urban_congestion_phase4

# 4. View plots
ls -lh plots/
```

## Troubleshooting

### No GPUs found
Use CPU mode:
```bash
python -m src.main --scenario urban_congestion_phase4 --policy fuzzy --gpu-id cpu --duration-s 30
```

### Scenario not found
Check available scenarios:
```bash
ls experiments/scenarios/
```

### Import errors
Make sure you're in `/workspace`:
```bash
cd /workspace
python -m src.main --scenario urban_congestion_phase4 --policy fuzzy --gpu-id cpu --duration-s 30
```

