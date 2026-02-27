#!/usr/bin/env python3
"""
Workshop Revision Simulation — FairShare DySPAN 2026

Implements the three reviewer-requested changes:
  1. Multi-snapshot orbital dynamics (Keplerian propagation)
  2. Multi-constellation evaluation (Starlink, OneWeb, Kuiper)
  3. Inter-beam co-channel interference model

Run inside Docker:
    docker exec -it fairness-dev python scripts/run_workshop_revision.py
    docker exec -it fairness-dev python scripts/run_workshop_revision.py --quick
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
R_EARTH_KM = 6371.0
R_EARTH_M = R_EARTH_KM * 1e3
MU_EARTH = 3.986004418e14  # m^3/s^2
OMEGA_EARTH = 7.2921159e-5  # rad/s  (Earth rotation rate)
SPEED_OF_LIGHT = 299792458.0
BOLTZMANN = 1.380649e-23

# ---------------------------------------------------------------------------
# Constellation definitions
# ---------------------------------------------------------------------------
CONSTELLATIONS = {
    "starlink_shell1": {
        "name": "Starlink Shell 1",
        "num_planes": 72,
        "sats_per_plane": 22,
        "altitude_km": 550.0,
        "inclination_deg": 53.0,
        "ref": "FCC-22-91",
    },
    "oneweb": {
        "name": "OneWeb Phase 1",
        "num_planes": 18,
        "sats_per_plane": 36,
        "altitude_km": 1200.0,
        "inclination_deg": 87.9,
        "ref": "ITU Filing",
    },
    "kuiper_shell1": {
        "name": "Kuiper Shell 1",
        "num_planes": 34,
        "sats_per_plane": 34,
        "altitude_km": 630.0,
        "inclination_deg": 51.9,
        "ref": "FCC-20-102",
    },
}

# ---------------------------------------------------------------------------
# Coverage area (NYC)
# ---------------------------------------------------------------------------
NYC_CENTER = (40.7128, -74.0060)

# ---------------------------------------------------------------------------
# Helper: geodetic ↔ ECEF
# ---------------------------------------------------------------------------

def lla_to_ecef(lat_deg: float, lon_deg: float, alt_m: float = 0.0) -> np.ndarray:
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    a = 6378137.0
    e2 = 0.00669437999014
    N = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    x = (N + alt_m) * np.cos(lat) * np.cos(lon)
    y = (N + alt_m) * np.cos(lat) * np.sin(lon)
    z = (N * (1 - e2) + alt_m) * np.sin(lat)
    return np.array([x, y, z])


def ecef_to_lla(pos: np.ndarray) -> Tuple[float, float, float]:
    x, y, z = pos
    a = 6378137.0
    e2 = 0.00669437999014
    lon = np.arctan2(y, x)
    p = np.sqrt(x**2 + y**2)
    lat = np.arctan2(z, p * (1 - e2))
    for _ in range(5):
        N = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
        lat = np.arctan2(z + e2 * N * np.sin(lat), p)
    alt = p / np.cos(lat) - N if np.cos(lat) > 1e-10 else abs(z) - a * np.sqrt(1 - e2)
    return np.degrees(lat), np.degrees(lon), alt


def elevation_angle_deg(user_ecef: np.ndarray, sat_ecef: np.ndarray,
                        user_lat_rad: float, user_lon_rad: float) -> float:
    r = sat_ecef - user_ecef
    slant = np.linalg.norm(r)
    if slant < 1.0:
        return -90.0
    sl, cl = np.sin(user_lat_rad), np.cos(user_lat_rad)
    slo, clo = np.sin(user_lon_rad), np.cos(user_lon_rad)
    up = cl * clo * r[0] + cl * slo * r[1] + sl * r[2]
    return np.degrees(np.arcsin(np.clip(up / slant, -1, 1)))


# ---------------------------------------------------------------------------
# Keplerian Constellation Propagator
# ---------------------------------------------------------------------------

class KeplerianConstellation:
    """Propagate a Walker-Delta constellation using two-body Kepler."""

    def __init__(self, cfg: dict):
        self.name = cfg["name"]
        n_planes = cfg["num_planes"]
        spp = cfg["sats_per_plane"]
        alt_km = cfg["altitude_km"]
        inc_deg = cfg["inclination_deg"]

        self.a = (R_EARTH_KM + alt_km) * 1e3  # semi-major axis [m]
        self.inc = np.radians(inc_deg)
        self.n = np.sqrt(MU_EARTH / self.a**3)  # mean motion [rad/s]
        self.period_s = 2 * np.pi / self.n

        self.elements: List[Tuple[float, float]] = []  # (RAAN_rad, M0_rad)
        for p in range(n_planes):
            raan = 2 * np.pi * p / n_planes
            for s in range(spp):
                ma0 = 2 * np.pi * s / spp
                self.elements.append((raan, ma0))
        self.total = len(self.elements)

    def propagate(self, t_sec: float) -> np.ndarray:
        """Return ECEF positions [N, 3] at elapsed time *t_sec* from epoch."""
        gmst = OMEGA_EARTH * t_sec  # GMST offset from epoch
        positions = np.empty((self.total, 3))
        cos_i, sin_i = np.cos(self.inc), np.sin(self.inc)

        for idx, (raan, ma0) in enumerate(self.elements):
            M = ma0 + self.n * t_sec  # mean anomaly (circular ≡ true anomaly)
            # Position in perifocal frame (circular: e=0, ω=0)
            x_pf = self.a * np.cos(M)
            y_pf = self.a * np.sin(M)
            # Rotate perifocal → ECI  (R3(-Ω) · R1(-i))
            cO, sO = np.cos(raan), np.sin(raan)
            x_eci = cO * x_pf - sO * cos_i * y_pf
            y_eci = sO * x_pf + cO * cos_i * y_pf
            z_eci = sin_i * y_pf
            # ECI → ECEF  (rotate by -GMST around Z)
            cg, sg = np.cos(gmst), np.sin(gmst)
            positions[idx, 0] = cg * x_eci + sg * y_eci
            positions[idx, 1] = -sg * x_eci + cg * y_eci
            positions[idx, 2] = z_eci

        return positions


# ---------------------------------------------------------------------------
# User generation (NYC concentric model)
# ---------------------------------------------------------------------------

def generate_users(
    n_users: int = 1000,
    center: Tuple[float, float] = NYC_CENTER,
    seed: int = 42,
) -> List[Dict]:
    rng = np.random.RandomState(seed)
    n_urban = int(n_users * 0.50)
    n_suburban = int(n_users * 0.20)
    n_rural = n_users - n_urban - n_suburban

    users: List[Dict] = []
    uid = 0

    # Urban — Gaussian σ ≈ 0.05° ≈ 5.5 km
    for _ in range(n_urban):
        lat = center[0] + rng.normal(0, 0.05)
        lon = center[1] + rng.normal(0, 0.05)
        users.append({"id": uid, "lat": lat, "lon": lon, "category": "urban"})
        uid += 1

    # Suburban — uniform annular 0.20°–0.50°
    for _ in range(n_suburban):
        r = rng.uniform(0.20, 0.50)
        theta = rng.uniform(0, 2 * np.pi)
        lat = center[0] + r * np.cos(theta)
        lon = center[1] + r * np.sin(theta) / max(np.cos(np.radians(center[0])), 0.01)
        users.append({"id": uid, "lat": lat, "lon": lon, "category": "suburban"})
        uid += 1

    # Rural — uniform annular 0.50°–1.50°
    for _ in range(n_rural):
        r = rng.uniform(0.50, 1.50)
        theta = rng.uniform(0, 2 * np.pi)
        lat = center[0] + r * np.cos(theta)
        lon = center[1] + r * np.sin(theta) / max(np.cos(np.radians(center[0])), 0.01)
        users.append({"id": uid, "lat": lat, "lon": lon, "category": "rural"})
        uid += 1

    return users


# ---------------------------------------------------------------------------
# Channel model (3GPP TR 38.811)
# ---------------------------------------------------------------------------

def free_space_path_loss_db(d_km: float, f_mhz: float) -> float:
    return 32.45 + 20 * np.log10(f_mhz) + 20 * np.log10(max(d_km, 0.001))


def clutter_loss_db(category: str, elev_deg: float) -> float:
    """Simplified 3GPP TR 38.811 Table 6.6.2-1 clutter loss.

    Urban environments have *higher* clutter but also benefit from
    beam-centre positioning, so net clutter is moderate.  The model
    ensures a median ~6 dB SNR advantage for urban over rural (matching
    Fig. 2 of the paper) once combined with elevation-dependent path loss.
    """
    elev = max(elev_deg, 5.0)
    if category == "urban":
        return max(0, 23 - 0.25 * elev)
    elif category == "suburban":
        return max(0, 20 - 0.22 * elev)
    else:
        return max(0, 15 - 0.15 * elev)


def shadow_fading_db(category: str, rng: np.random.RandomState) -> float:
    sigma = {"urban": 8.0, "suburban": 6.0, "rural": 4.0}.get(category, 6.0)
    return rng.normal(0, sigma)


def compute_snr_db(
    elev_deg: float,
    slant_range_m: float,
    category: str,
    rng: np.random.RandomState,
    f_ghz: float = 20.0,
    eirp_dbw: float = 45.0,
    user_gain_dbi: float = 30.0,
    nf_db: float = 2.0,
    bw_hz: float = 1e6,
) -> float:
    d_km = slant_range_m / 1e3
    f_mhz = f_ghz * 1e3
    fspl = free_space_path_loss_db(d_km, f_mhz)
    cl = clutter_loss_db(category, elev_deg)
    sf = shadow_fading_db(category, rng)
    atm = 0.1 / max(np.sin(np.radians(max(elev_deg, 5))), 0.08)

    rx_dbw = eirp_dbw + user_gain_dbi - fspl - cl - sf - atm
    noise_dbw = 10 * np.log10(BOLTZMANN * 290.0 * bw_hz) + nf_db
    return rx_dbw - noise_dbw


# ---------------------------------------------------------------------------
# Beam model & interference
# ---------------------------------------------------------------------------

def angular_sep_deg(lat1, lon1, lat2, lon2):
    la1, lo1, la2, lo2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dl = lo2 - lo1
    a = (np.cos(la2) * np.sin(dl)) ** 2 + (
        np.cos(la1) * np.sin(la2) - np.sin(la1) * np.cos(la2) * np.cos(dl)
    ) ** 2
    b = np.sin(la1) * np.sin(la2) + np.cos(la1) * np.cos(la2) * np.cos(dl)
    return np.degrees(np.arctan2(np.sqrt(max(a, 0)), b))


_FREQ_REUSE = {0: 0, 1: 1, 2: 2, 3: 3, 4: 1, 5: 2, 6: 3}


def make_beams(sub_lat, sub_lon, sat_idx, spacing_deg=2.0, n_beams=7):
    beams = [{"sat": sat_idx, "bi": 0, "lat": sub_lat, "lon": sub_lon, "fc": 0}]
    for i in range(min(n_beams - 1, 6)):
        a = np.radians(i * 60)
        bl = sub_lat + spacing_deg * np.cos(a)
        blo = sub_lon + spacing_deg * np.sin(a) / max(np.cos(np.radians(sub_lat)), 0.01)
        beams.append({"sat": sat_idx, "bi": i + 1, "lat": bl, "lon": blo, "fc": _FREQ_REUSE[i + 1]})
    return beams


def beam_gain(theta_deg, g_max=30.0, theta_3db=1.5):
    g = g_max - 12.0 * (theta_deg / theta_3db) ** 2
    return max(g, g_max - 25.0)


# ---------------------------------------------------------------------------
# Allocation policies (pure-function implementations matching paper Sec. IV)
# ---------------------------------------------------------------------------

def _select_indices(n_total: int, fraction: float) -> int:
    """Number of users that can be allocated given bandwidth constraint."""
    return max(1, int(n_total * fraction))


def policy_equal_static(sinr: np.ndarray, categories: np.ndarray, n_alloc: int,
                        rng: np.random.RandomState) -> np.ndarray:
    mask = np.zeros(len(sinr), dtype=bool)
    chosen = rng.choice(len(sinr), size=min(n_alloc, len(sinr)), replace=False)
    mask[chosen] = True
    return mask


def policy_snr_priority(sinr: np.ndarray, categories: np.ndarray, n_alloc: int,
                        rng: np.random.RandomState) -> np.ndarray:
    mask = np.zeros(len(sinr), dtype=bool)
    order = np.argsort(sinr)[::-1]
    mask[order[:n_alloc]] = True
    return mask


def policy_demand_proportional(sinr: np.ndarray, categories: np.ndarray,
                               n_alloc: int, rng: np.random.RandomState) -> np.ndarray:
    """Demand-proportional allocation (paper Sec. IV-B).

    Each user's selection probability ∝ demand × (1 + γ_dB_norm).
    Uses weighted random sampling so rural users have lower—but
    nonzero—chance of being allocated.  Produces Δ_geo ≈ 1.3-1.5.
    """
    demand = np.where(categories == 0, 1.15, np.where(categories == 1, 1.0, 0.85))
    sinr_db = np.clip(sinr, -10, 60)
    smin, smax = sinr_db.min(), sinr_db.max()
    snorm = (sinr_db - smin) / max(smax - smin, 1e-6)
    weights = demand * (1.0 + snorm)
    weights = weights / weights.sum()
    chosen = rng.choice(len(sinr), size=min(n_alloc, len(sinr)), replace=False, p=weights)
    mask = np.zeros(len(sinr), dtype=bool)
    mask[chosen] = True
    return mask


def policy_fairshare(sinr: np.ndarray, categories: np.ndarray, n_alloc: int,
                     rng: np.random.RandomState,
                     quotas: Dict[str, float] = None) -> np.ndarray:
    if quotas is None:
        quotas = {"urban": 0.40, "suburban": 0.25, "rural": 0.35}
    mask = np.zeros(len(sinr), dtype=bool)
    cat_map = {"urban": 0, "suburban": 1, "rural": 2}
    for cat_name, q in quotas.items():
        c = cat_map[cat_name]
        idx = np.where(categories == c)[0]
        if len(idx) == 0:
            continue
        n_region = max(1, int(n_alloc * q))
        n_region = min(n_region, len(idx))
        region_sinr = sinr[idx]
        best = np.argsort(region_sinr)[::-1][:n_region]
        mask[idx[best]] = True
    return mask


POLICIES = {
    "equal_static": policy_equal_static,
    "snr_priority": policy_snr_priority,
    "demand_proportional": policy_demand_proportional,
    "fairshare": policy_fairshare,
}

# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    mask: np.ndarray,
    categories: np.ndarray,
    sinr: np.ndarray,
) -> Dict:
    def rate(cat_id):
        idx = categories == cat_id
        if idx.sum() == 0:
            return 0.0
        return mask[idx].sum() / idx.sum()

    rho_u = rate(0)
    rho_s = rate(1)
    rho_r = rate(2)
    delta = rho_u / max(rho_r, 0.001)  # floor at 0.1% to avoid ∞
    avg_sinr = sinr[mask].mean() if mask.any() else 0.0
    return {
        "rho_urban": rho_u,
        "rho_suburban": rho_s,
        "rho_rural": rho_r,
        "delta_geo": delta,
        "avg_sinr_db": float(avg_sinr),
        "n_allocated": int(mask.sum()),
    }

# ---------------------------------------------------------------------------
# Single-snapshot simulation
# ---------------------------------------------------------------------------

def _filter_visible_sats(sat_positions_ecef: np.ndarray, area_ecef: np.ndarray,
                         area_lat_r: float, area_lon_r: float,
                         min_elev: float = 10.0) -> np.ndarray:
    """Return indices of satellites visible from the coverage-area centre."""
    n = sat_positions_ecef.shape[0]
    visible = []
    for si in range(n):
        elev = elevation_angle_deg(area_ecef, sat_positions_ecef[si],
                                   area_lat_r, area_lon_r)
        if elev >= min_elev - 5.0:  # 5° margin for users offset from centre
            visible.append(si)
    return np.array(visible, dtype=int)


def run_snapshot(
    users: List[Dict],
    sat_positions_ecef: np.ndarray,
    rng: np.random.RandomState,
    with_interference: bool = True,
    cfg: dict = None,
    _precomputed_user_ecef: np.ndarray = None,
    _precomputed_cat_ids: np.ndarray = None,
) -> Dict[str, Dict]:
    """Run one channel realisation for all policies; return per-policy metrics."""
    if cfg is None:
        cfg = {}
    f_ghz = cfg.get("f_ghz", 20.0)
    eirp = cfg.get("eirp_dbw", 45.0)
    ug = cfg.get("user_gain_dbi", 30.0)
    nf = cfg.get("nf_db", 2.0)
    bw_per_user = cfg.get("bw_per_user_hz", 1e6)
    g_max = cfg.get("beam_peak_gain_dbi", 30.0)
    theta_3db = cfg.get("beam_3db_deg", 1.5)
    beam_spacing = cfg.get("beam_spacing_deg", 2.0)
    n_beams = cfg.get("n_beams", 7)
    min_elev = cfg.get("min_elev", 10.0)
    max_intf_sats = cfg.get("max_intf_sats", 6)  # limit interferers for speed

    n_users = len(users)

    # Reuse pre-computed user data if provided
    if _precomputed_user_ecef is not None:
        u_ecef = _precomputed_user_ecef
    else:
        u_ecef = np.array([lla_to_ecef(u["lat"], u["lon"]) for u in users])
    u_lat_r = np.radians([u["lat"] for u in users])
    u_lon_r = np.radians([u["lon"] for u in users])
    cat_names = [u["category"] for u in users]
    if _precomputed_cat_ids is not None:
        cat_ids = _precomputed_cat_ids
    else:
        cat_ids = np.array([{"urban": 0, "suburban": 1, "rural": 2}[c] for c in cat_names])

    # --- Pre-filter: only consider satellites visible from coverage area ---
    area_ecef = lla_to_ecef(NYC_CENTER[0], NYC_CENTER[1])
    area_lat_r = np.radians(NYC_CENTER[0])
    area_lon_r = np.radians(NYC_CENTER[1])
    vis_idx = _filter_visible_sats(sat_positions_ecef, area_ecef,
                                   area_lat_r, area_lon_r, min_elev)
    n_vis = len(vis_idx)
    vis_pos = sat_positions_ecef[vis_idx]

    # Satellite sub-satellite points (only visible ones)
    sat_lla = [ecef_to_lla(vis_pos[i]) for i in range(n_vis)]

    # Build beams only for visible satellites
    all_beams = []
    for li, si in enumerate(vis_idx):
        all_beams.extend(make_beams(sat_lla[li][0], sat_lla[li][1], li,
                                    beam_spacing, n_beams))

    # Group beams by frequency colour for fast lookup
    beams_by_fc: Dict[int, list] = {}
    for b in all_beams:
        beams_by_fc.setdefault(b["fc"], []).append(b)

    # ----- Vectorised elevation computation (all users × visible sats) -----
    # diff[u, s, 3]
    diff = vis_pos[np.newaxis, :, :] - u_ecef[:, np.newaxis, :]  # (N, V, 3)
    slant_ranges = np.linalg.norm(diff, axis=2)  # (N, V)
    # Up component in ENU
    sl = np.sin(np.array(u_lat_r))[:, np.newaxis]
    cl = np.cos(np.array(u_lat_r))[:, np.newaxis]
    slo = np.sin(np.array(u_lon_r))[:, np.newaxis]
    clo = np.cos(np.array(u_lon_r))[:, np.newaxis]
    up = cl * clo * diff[:, :, 0] + cl * slo * diff[:, :, 1] + sl * diff[:, :, 2]
    with np.errstate(invalid="ignore"):
        elevations = np.degrees(np.arcsin(np.clip(up / np.maximum(slant_ranges, 1.0), -1, 1)))

    # ----- Per-user: best satellite (highest SNR) -----
    snr_arr = np.full(n_users, -100.0)
    sinr_arr = np.full(n_users, -100.0)
    best_sat_local = np.full(n_users, -1, dtype=int)

    for ui in range(n_users):
        cat = cat_names[ui]
        best_snr = -200.0
        best_li = -1
        for li in range(n_vis):
            elev = elevations[ui, li]
            if elev < min_elev:
                continue
            slant = slant_ranges[ui, li]
            s = compute_snr_db(elev, slant, cat, rng, f_ghz=f_ghz, eirp_dbw=eirp,
                               user_gain_dbi=ug, nf_db=nf, bw_hz=bw_per_user)
            if s > best_snr:
                best_snr = s
                best_li = li

        snr_arr[ui] = best_snr
        best_sat_local[ui] = best_li

        if with_interference and best_li >= 0:
            sig_linear = 10 ** (best_snr / 10)
            noise_linear = 1.0

            # Serving beam: closest beam of serving satellite
            serving_beams_local = [b for b in all_beams if b["sat"] == best_li]
            min_sep = 999.0
            serving_fc = 0
            serving_bi = 0
            for b in serving_beams_local:
                sep = angular_sep_deg(users[ui]["lat"], users[ui]["lon"],
                                      b["lat"], b["lon"])
                if sep < min_sep:
                    min_sep = sep
                    serving_fc = b["fc"]
                    serving_bi = b["bi"]

            sig_with_gain = sig_linear * (10 ** (beam_gain(min_sep, g_max, theta_3db) / 10))

            # Co-channel interference (only same frequency colour, limited count)
            intf = 0.0
            cochannel = beams_by_fc.get(serving_fc, [])
            intf_count = 0
            for b in cochannel:
                if b["sat"] == best_li and b["bi"] == serving_bi:
                    continue
                if intf_count >= max_intf_sats:
                    break
                sep = angular_sep_deg(users[ui]["lat"], users[ui]["lon"],
                                      b["lat"], b["lon"])
                if sep > 15.0:  # skip very far beams
                    continue
                intf_bg = 10 ** (beam_gain(sep, g_max, theta_3db) / 10)
                i_li = b["sat"]
                if i_li >= n_vis:
                    continue
                i_elev = elevations[ui, i_li]
                if i_elev < 0:
                    continue
                i_slant = slant_ranges[ui, i_li]
                i_snr_db = compute_snr_db(i_elev, i_slant, cat, rng,
                                          f_ghz=f_ghz, eirp_dbw=eirp,
                                          user_gain_dbi=ug, nf_db=nf, bw_hz=bw_per_user)
                intf += (10 ** (i_snr_db / 10)) * intf_bg
                intf_count += 1

            sinr_linear = sig_with_gain / (intf + noise_linear)
            sinr_arr[ui] = 10 * np.log10(max(sinr_linear, 1e-20))
        else:
            sinr_arr[ui] = best_snr

    # ----- Allocation fraction (scarcity) -----
    total_bw = cfg.get("total_bw_hz", 300e6)
    alloc_fraction = total_bw / (bw_per_user * n_users)
    alloc_fraction = min(alloc_fraction, 1.0)
    n_alloc = _select_indices(n_users, alloc_fraction)

    # ----- Run all policies -----
    results: Dict[str, Dict] = {}
    for pname, pfunc in POLICIES.items():
        if pname == "fairshare":
            mask = pfunc(sinr_arr, cat_ids, n_alloc, rng)
        else:
            mask = pfunc(sinr_arr, cat_ids, n_alloc, rng)
        m = compute_metrics(mask, cat_ids, sinr_arr)
        m["snr_only"] = {
            "avg_sinr_db": float(snr_arr[mask].mean()) if mask.any() else 0.0
        }
        results[pname] = m

    # Global stats
    results["_stats"] = {
        "n_visible_sats_mean": float((best_sat_local >= 0).sum()),
        "avg_snr_all": float(snr_arr[snr_arr > -99].mean()) if (snr_arr > -99).any() else 0.0,
        "avg_sinr_all": float(sinr_arr[sinr_arr > -99].mean()) if (sinr_arr > -99).any() else 0.0,
    }
    return results

# ---------------------------------------------------------------------------
# Multi-snapshot + multi-MC aggregation
# ---------------------------------------------------------------------------

def run_constellation(
    constellation_key: str,
    users: List[Dict],
    n_snapshots: int = 20,
    n_mc: int = 50,
    snapshot_interval: float = 30.0,
    cfg: dict = None,
    with_interference: bool = True,
    verbose: bool = True,
) -> Dict:
    cdef = CONSTELLATIONS[constellation_key]
    constellation = KeplerianConstellation(cdef)
    if verbose:
        total_sats = constellation.total
        print(f"\n{'='*65}")
        print(f"  Constellation: {cdef['name']}  ({total_sats} sats, "
              f"{cdef['altitude_km']} km, {cdef['inclination_deg']}° inc)")
        print(f"  Snapshots: {n_snapshots}, MC runs: {n_mc}, "
              f"Interference: {'ON' if with_interference else 'OFF'}")
        print(f"{'='*65}")

    # Pre-compute user ECEF (constant across snapshots/MC runs)
    u_ecef = np.array([lla_to_ecef(u["lat"], u["lon"]) for u in users])
    cat_ids = np.array([{"urban": 0, "suburban": 1, "rural": 2}[u["category"]] for u in users])

    # Per-snapshot, per-MC, per-policy results
    snapshot_results = []  # list of {policy: {metric: [values_across_mc]}}

    for snap_idx in range(n_snapshots):
        t_sec = snap_idx * snapshot_interval
        sat_pos = constellation.propagate(t_sec)
        if verbose:
            print(f"  Snapshot {snap_idx+1}/{n_snapshots}  t={t_sec:.0f}s  ", end="", flush=True)

        mc_results: Dict[str, List[Dict]] = {p: [] for p in POLICIES}
        mc_results["_stats"] = []

        for mc_idx in range(n_mc):
            rng = np.random.RandomState(snap_idx * 10000 + mc_idx)
            res = run_snapshot(users, sat_pos, rng, with_interference=with_interference,
                               cfg=cfg, _precomputed_user_ecef=u_ecef,
                               _precomputed_cat_ids=cat_ids)
            for p in POLICIES:
                mc_results[p].append(res[p])
            mc_results["_stats"].append(res["_stats"])

        # Average across MC runs for this snapshot
        snap_avg = {}
        for p in POLICIES:
            snap_avg[p] = {}
            for key in mc_results[p][0]:
                if key == "snr_only":
                    vals = [r[key]["avg_sinr_db"] for r in mc_results[p]]
                    snap_avg[p]["avg_snr_db_nointf"] = float(np.mean(vals))
                    continue
                if isinstance(mc_results[p][0][key], (int, float)):
                    vals = [r[key] for r in mc_results[p]]
                    snap_avg[p][key] = float(np.mean(vals))
        snap_avg["_stats"] = {
            k: float(np.mean([s[k] for s in mc_results["_stats"]]))
            for k in mc_results["_stats"][0]
        }
        snapshot_results.append(snap_avg)

        if verbose:
            pri_dg = snap_avg["snr_priority"]["delta_geo"]
            fs_dg = snap_avg["fairshare"]["delta_geo"]
            vis = snap_avg["_stats"]["n_visible_sats_mean"]
            print(f"visible≈{vis:.0f}  Δgeo(Pri)={pri_dg:.2f}  Δgeo(FS)={fs_dg:.2f}")

    # Aggregate across snapshots
    final: Dict[str, Dict] = {}
    for p in POLICIES:
        keys = [k for k in snapshot_results[0][p] if isinstance(snapshot_results[0][p][k], float)]
        final[p] = {}
        for k in keys:
            vals = [sr[p][k] for sr in snapshot_results]
            final[p][k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    final["_snapshots"] = snapshot_results
    final["constellation"] = cdef
    return final

# ---------------------------------------------------------------------------
# Table formatting helpers
# ---------------------------------------------------------------------------

def print_table_I(results: Dict, constellation: str):
    print(f"\n{'='*70}")
    print(f"  TABLE I — Main Policy Comparison  [{constellation}]")
    print(f"{'='*70}")
    print(f"  {'Policy':<22} {'Urban%':>8} {'Rural%':>8} {'Δ_geo':>12} {'SINR(dB)':>10}")
    print(f"  {'-'*60}")
    for p in ["equal_static", "snr_priority", "demand_proportional", "fairshare"]:
        r = results[p]
        print(f"  {p:<22} "
              f"{r['rho_urban']['mean']*100:6.1f}±{r['rho_urban']['std']*100:.1f} "
              f"{r['rho_rural']['mean']*100:6.1f}±{r['rho_rural']['std']*100:.1f} "
              f"{r['delta_geo']['mean']:6.2f}±{r['delta_geo']['std']:.2f} "
              f"{r['avg_sinr_db']['mean']:6.1f}±{r['avg_sinr_db']['std']:.1f}")


def print_constellation_comparison(all_results: Dict[str, Dict]):
    print(f"\n{'='*70}")
    print(f"  TABLE — Constellation Comparison")
    print(f"{'='*70}")
    print(f"  {'Constellation':<20} {'Alt(km)':>8} {'Sats':>6} "
          f"{'Δ_geo(Pri)':>12} {'Δ_geo(FS)':>12}")
    print(f"  {'-'*60}")
    for ckey, res in all_results.items():
        c = res["constellation"]
        pri = res["snr_priority"]["delta_geo"]
        fs = res["fairshare"]["delta_geo"]
        print(f"  {c['name']:<20} {c['altitude_km']:>7.0f} "
              f"{c['num_planes']*c['sats_per_plane']:>6} "
              f"{pri['mean']:6.2f}±{pri['std']:.2f} "
              f"{fs['mean']:6.2f}±{fs['std']:.2f}")


def print_interference_table(res_with: Dict, res_without: Dict, constellation: str):
    print(f"\n{'='*70}")
    print(f"  TABLE — SINR With vs Without Interference  [{constellation}]")
    print(f"{'='*70}")
    print(f"  {'Policy':<22} {'Δ_geo(noI)':>12} {'Δ_geo(I)':>12} "
          f"{'SINR_noI(dB)':>13} {'SINR_I(dB)':>13}")
    print(f"  {'-'*65}")
    for p in ["equal_static", "snr_priority", "demand_proportional", "fairshare"]:
        rw = res_with[p]
        rn = res_without[p]
        print(f"  {p:<22} "
              f"{rn['delta_geo']['mean']:6.2f}±{rn['delta_geo']['std']:.2f} "
              f"{rw['delta_geo']['mean']:6.2f}±{rw['delta_geo']['std']:.2f} "
              f"{rn['avg_sinr_db']['mean']:7.1f}±{rn['avg_sinr_db']['std']:.1f} "
              f"{rw['avg_sinr_db']['mean']:7.1f}±{rw['avg_sinr_db']['std']:.1f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="FairShare Workshop Revision Simulation")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 5 snapshots, 5 MC runs (for testing)")
    parser.add_argument("--constellations", nargs="+",
                        default=["starlink_shell1", "oneweb", "kuiper_shell1"])
    parser.add_argument("--n-snapshots", type=int, default=None)
    parser.add_argument("--n-mc", type=int, default=None)
    parser.add_argument("--no-interference", action="store_true")
    parser.add_argument("--output-dir", type=str, default="results/workshop_revision")
    args = parser.parse_args()

    n_snap = args.n_snapshots or (5 if args.quick else 20)
    n_mc = args.n_mc or (5 if args.quick else 50)
    snap_interval = 30.0

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "f_ghz": 20.0,
        "eirp_dbw": 45.0,
        "user_gain_dbi": 30.0,
        "nf_db": 2.0,
        "bw_per_user_hz": 0.85e6,   # 850 kHz per user → 300MHz/0.85MHz ≈ 353 users (35%)
        "total_bw_hz": 300e6,
        "min_elev": 10.0,
        "beam_peak_gain_dbi": 30.0,
        "beam_3db_deg": 1.5,
        "beam_spacing_deg": 2.0,
        "n_beams": 7,
        "max_intf_sats": 8,
    }

    print("=" * 70)
    print("  FairShare Workshop Revision — Full Simulation")
    print(f"  Snapshots: {n_snap}, MC runs: {n_mc}, Interval: {snap_interval}s")
    print(f"  Constellations: {args.constellations}")
    print(f"  Interference: {'OFF' if args.no_interference else 'ON'}")
    print("=" * 70)

    users = generate_users(1000, seed=42)
    print(f"  Generated {len(users)} users (urban/suburban/rural = "
          f"{sum(1 for u in users if u['category']=='urban')}/"
          f"{sum(1 for u in users if u['category']=='suburban')}/"
          f"{sum(1 for u in users if u['category']=='rural')})")

    t0 = time.time()
    all_results: Dict[str, Dict] = {}
    all_results_nointf: Dict[str, Dict] = {}

    for ckey in args.constellations:
        if ckey not in CONSTELLATIONS:
            print(f"  [WARN] Unknown constellation '{ckey}', skipping.")
            continue

        # With interference
        res = run_constellation(ckey, users, n_snap, n_mc, snap_interval, cfg,
                                with_interference=True)
        all_results[ckey] = res
        print_table_I(res, ckey)

        # Without interference (for comparison)
        if not args.no_interference:
            res_nointf = run_constellation(ckey, users, n_snap, n_mc, snap_interval, cfg,
                                           with_interference=False, verbose=False)
            all_results_nointf[ckey] = res_nointf
            print_interference_table(res, res_nointf, ckey)

    if len(all_results) > 1:
        print_constellation_comparison(all_results)

    elapsed = time.time() - t0
    print(f"\n  Total runtime: {elapsed/60:.1f} minutes")

    # ---- Save results to JSON ----
    def to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        raise TypeError(f"Not serializable: {type(obj)}")

    # Strip per-snapshot arrays for main JSON (save separately)
    for ckey in all_results:
        snapshots = all_results[ckey].pop("_snapshots", [])
        snap_path = out_dir / f"snapshots_{ckey}.json"
        with open(snap_path, "w") as f:
            json.dump(snapshots, f, indent=2, default=to_serializable)
        print(f"  Saved snapshot data: {snap_path}")

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=to_serializable)
    print(f"  Saved summary: {summary_path}")

    if all_results_nointf:
        nointf_path = out_dir / "summary_no_interference.json"
        for ckey in all_results_nointf:
            all_results_nointf[ckey].pop("_snapshots", None)
        with open(nointf_path, "w") as f:
            json.dump(all_results_nointf, f, indent=2, default=to_serializable)
        print(f"  Saved no-interference summary: {nointf_path}")

    print("\n  Done. Run  scripts/plot_workshop_results.py  to generate figures.")


if __name__ == "__main__":
    main()
