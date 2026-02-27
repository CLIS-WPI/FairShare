# FairShare: Auditable Geographic Fairness for Multi-Operator LEO Spectrum Sharing

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16+-orange.svg)](https://www.tensorflow.org/)

> **A simulation framework for evaluating geographic fairness in LEO satellite spectrum allocation. Incorporates Keplerian orbital dynamics, inter-beam co-channel interference, and three real-world constellation geometries to demonstrate that conventional allocation policies systematically disadvantage rural users.**

## Overview

This repository contains the simulation framework and experimental results for the FairShare paper. Key findings from multi-snapshot simulation with interference:

- **SNR-Priority scheduling** induces a **1.84x** mean urban-rural access disparity, with temporal fluctuations reaching **3.9x**
- **FairShare** achieves affirmative fairness (Delta_geo = **0.68**) with **zero variance** across all orbital snapshots, interference conditions, and constellation geometries
- The structural bias is **policy-inherent** and persists across Starlink, OneWeb, and Kuiper constellations
- Increasing bandwidth **amplifies** rather than alleviates the disparity

## Key Results

### Main Results (Starlink Shell 1, W=300 MHz, 20x50=1,000 samples, with interference)

| Policy | Urban Rate | Rural Rate | Delta_geo |
|--------|-----------|-----------|-----------|
| Equal Static | 35.2 +/- 0.2% | 35.2 +/- 0.3% | 1.01 +/- 0.01 |
| SNR Priority | 39.3 +/- 3.9% | 25.7 +/- 8.3% | **1.84 +/- 0.93** |
| Demand Prop. | 38.7 +/- 0.2% | 29.8 +/- 0.3% | 1.31 +/- 0.02 |
| **FairShare** | **28.0 +/- 0.0%** | **41.0 +/- 0.0%** | **0.68 +/- 0.00** |

Standard deviations reflect temporal variability due to changing orbital geometry and interference patterns.

### Cross-Constellation Validation

| Constellation | Alt (km) | Sats | Priority Delta_geo | FairShare Delta_geo |
|---------------|----------|------|--------------------|---------------------|
| Starlink Shell 1 | 550 | 1,584 | 1.84 +/- 0.93 | **0.68 +/- 0.00** |
| OneWeb Phase 1 | 1,200 | 648 | 1.77 +/- 0.59 | **0.68 +/- 0.00** |
| Kuiper Shell 1 | 630 | 1,156 | 1.60 +/- 0.49 | **0.68 +/- 0.00** |

### Interference Impact (Starlink Shell 1)

| Policy | SINR no intf. (dB) | SINR with intf. (dB) | Delta_geo no intf. | Delta_geo with intf. |
|--------|-------------------|---------------------|-------------------|---------------------|
| Equal Static | 42.3 | 20.8 | 1.00 | 1.01 |
| SNR Priority | 47.2 | 32.7 | 2.55 | 1.84 |
| Demand Prop. | 42.9 | 22.0 | 1.36 | 1.31 |
| **FairShare** | 46.6 | 32.1 | **0.68** | **0.68** |

### Geographic Disparity Ratio (Delta_geo)
- **Delta_geo = 1.0**: Perfect geographic fairness
- **Delta_geo > 1.0**: Urban bias (unfair to rural)
- **Delta_geo < 1.0**: Rural-compensating (FairShare achieves this)

## Simulation Features

### Orbital Dynamics (Multi-Snapshot)
- Keplerian orbital propagator (ECI to ECEF with Earth rotation)
- 20 snapshots at 30-second intervals (10 minutes of satellite motion)
- Satellite positions re-propagated at each snapshot

### Three Constellation Configurations
- **Starlink Shell 1**: 72 planes x 22 sats = 1,584 satellites, 550 km, 53.0 deg
- **OneWeb Phase 1**: 18 planes x 36 sats = 648 satellites, 1,200 km, 87.9 deg
- **Kuiper Shell 1**: 34 planes x 34 sats = 1,156 satellites, 630 km, 51.9 deg

### Interference Model
- 7-beam hexagonal spot-beam layout per satellite
- ITU-R S.1528 parabolic beam gain pattern (G_max = 30 dBi, theta_3dB = 1.5 deg)
- 4-color frequency reuse pattern
- Co-channel inter-beam interference computed for full SINR

### Channel Model
- 3GPP TR 38.811 compliant
- Elevation-dependent path loss and clutter loss
- Location-dependent shadow fading (sigma_SF = 8 dB urban, 4 dB rural)
- Ka-band (20 GHz), EIRP = 45 dBW, user terminal gain = 30 dBi

## Quick Start

### Installation

```bash
git clone https://github.com/CLIS-WPI/FairShare.git
cd FairShare
pip install -r requirements.txt
```

### Run Workshop Revision Simulation

```bash
# Full simulation (3 constellations x 20 snapshots x 50 MC runs)
python scripts/run_workshop_revision.py

# Quick test (5 snapshots x 5 MC runs)
python scripts/run_workshop_revision.py --quick

# Single constellation
python scripts/run_workshop_revision.py --constellations starlink_shell1
```

### Generate Figures

```bash
python scripts/plot_workshop_results.py
```

### Docker

```bash
cd docker
docker compose -f docker-compose.dev.yaml up -d
docker exec -it fairshare-dev bash
python scripts/run_workshop_revision.py
```

## Project Structure

```
FairShare/
├── src/
│   ├── main.py                          # Original simulation entry point
│   ├── channel/
│   │   ├── beam_model.py                # Beam pattern, hexagonal layout, SINR
│   │   ├── constellation_config.py      # Starlink/OneWeb/Kuiper configs
│   │   └── openntn_channel.py           # 3GPP TR 38.811 channel model
│   ├── dss/
│   │   └── policies/
│   │       └── fairshare.py             # FairShare allocation policy
│   ├── allocation/                      # Resource allocation engine
│   └── fairness/                        # Fairness metrics
├── scripts/
│   ├── run_workshop_revision.py         # Multi-snapshot simulation engine
│   └── plot_workshop_results.py         # Result visualization
├── experiments/
│   └── scenarios/
│       └── workshop_revision.yaml       # Scenario configuration
├── results/
│   └── workshop_revision/               # Simulation outputs (JSON + figures)
├── Images/                              # Paper figures
├── docker/
│   ├── docker-compose.dev.yaml
│   └── Dockerfile.dev
└── tests/
```

## Methodology

### User Distribution (NYC Metropolitan Area, 40.7 N, 74.0 W)
- **Urban** (50%): Gaussian (sigma ~ 5.5 km) centered at metropolitan core
- **Suburban** (20%): Uniform annular ring at 22-55 km
- **Rural** (30%): Outer ring at 55-165 km

### Allocation Policies
1. **Equal Static**: Random allocation independent of channel quality (fairness baseline)
2. **SNR Priority**: Top-SINR users allocated first (efficiency baseline)
3. **Demand Proportional**: Weighted by demand x channel quality (commercial proxy)
4. **FairShare**: Geographic quota-based allocation (40% urban, 25% suburban, 35% rural)

### Statistical Methodology
- 20 orbital snapshots x 50 Monte Carlo channel realizations = 1,000 samples per constellation
- Standard deviations reflect temporal variability, not statistical uncertainty
- Full-buffer downlink traffic model (worst-case contention)

## Citation

```bibtex
@inproceedings{fairshare2026,
  title={FairShare: Auditable Geographic Fairness for Multi-Operator LEO Spectrum Sharing},
  author={Hashemi Natanzi, Seyed Bagher and Mohammadi, Hossein and Marojevic, Vuk and Tang, Bo},
  booktitle={IEEE DySPAN 2026 Workshop},
  year={2026}
}
```

## License

MIT License - see [LICENSE](LICENSE) file.

## Acknowledgments

- Supported in part by NTIA Award No. 51-60-IF007 and NSF Award 2332661
- [3GPP TR 38.811](https://www.3gpp.org/ftp/Specs/archive/38_series/38.811/) - NTN channel models
- [ITU-R S.1528](https://www.itu.int/rec/R-REC-S.1528/) - Satellite antenna patterns
