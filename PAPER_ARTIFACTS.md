# Paper Artifacts and Reproducibility Guide

## 🏆 Artifact Status

- ✅ **Functional**: All components work as described
- ✅ **Available**: Code and data are publicly available
- ✅ **Reproducible**: Complete instructions provided

## 📦 Artifact Contents

### 1. Source Code
- Complete implementation (Phases 1-4)
- All modules and dependencies
- Test suite

### 2. Docker Environment
- Pre-configured Docker image
- All dependencies included
- GPU support enabled

### 3. Example Datasets
- TLE files for Starlink and OneWeb
- Scenario YAML files (urban, rural, emergency)
- Pre-generated results (optional)

### 4. Documentation
- Complete README
- Phase-by-phase implementation guides
- API documentation

### 5. Example Outputs
- Sample CSV results
- Generated plots (PDF)
- Interactive notebook

## 🚀 Quick Start (5 minutes)

### Step 1: Clone Repository
```bash
git clone https://github.com/your-org/fuzzy-fairness-dss-leo.git
cd fuzzy-fairness-dss-leo
```

### Step 2: Build Docker Image
```bash
docker build -f docker/Dockerfile.final -t fuzzy-fairness-dss:latest .
```

### Step 3: Run Simulation
```bash
docker run --gpus all -v $(pwd):/workspace fuzzy-fairness-dss:latest \
  python -m src.main --scenario urban_congestion_phase4 --policy fuzzy --duration-s 30
```

### Step 4: Generate Plots
```bash
docker run --gpus all -v $(pwd):/workspace fuzzy-fairness-dss:latest \
  python experiments/generate_plots.py --scenario urban_congestion_phase4
```

## 📊 Reproducing Paper Figures

### Figure 1: Fairness Over Time
```bash
python experiments/generate_plots.py --scenario urban_congestion_phase4
# Output: plots/fairness_time_urban_congestion_phase4.pdf
```

### Figure 2: Policy Comparison
```bash
# Run all three policies
python -m src.main --scenario urban_congestion_phase4 --policy static --duration-s 600
python -m src.main --scenario urban_congestion_phase4 --policy priority --duration-s 600
python -m src.main --scenario urban_congestion_phase4 --policy fuzzy --duration-s 600

# Generate comparison plot
python experiments/generate_plots.py --scenario urban_congestion_phase4
# Output: plots/policy_comparison_urban_congestion_phase4.pdf
```

### Figure 3: Interactive Analysis
```bash
jupyter lab notebooks/interactive_demo.ipynb
```

## 🔍 Verification Checklist

- [ ] Docker image builds successfully
- [ ] All tests pass (`pytest tests/`)
- [ ] Simulation runs without errors
- [ ] CSV files are generated in `results/`
- [ ] Plots are generated in `plots/`
- [ ] Notebook executes all cells successfully

## 📝 Known Issues and Limitations

1. **OpenNTN Installation**: May require authentication for private repositories. Fallback models are used if OpenNTN is unavailable.

2. **GPU Requirements**: GPU tests require NVIDIA GPU with CUDA support. CPU mode is available as fallback.

3. **TLE Files**: Some TLE files may be outdated. Users can update from Celestrak or other sources.

## 🐛 Troubleshooting

### Issue: "Sionna not found"
**Solution**: Install Sionna separately:
```bash
pip install sionna==1.2.1
```

### Issue: "OpenNTN import failed"
**Solution**: This is expected if repository is private. System uses fallback channel models.

### Issue: "No GPUs found"
**Solution**: Use `--gpu-id cpu` flag or run in CPU mode.

## 📧 Contact

For questions or issues:
- Open an issue on GitHub
- Contact: your.email@example.com

## 📄 License

MIT License - see `LICENSE` file.
