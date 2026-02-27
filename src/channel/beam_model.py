"""
Beam model for Ka-band LEO satellite spot beams.

Implements:
- Parabolic beam gain rolloff (ITU-R S.1528 approximation)
- Hexagonal 7-beam layout per satellite
- 4-color frequency reuse pattern
- Inter-beam co-channel interference computation
"""

import numpy as np
from typing import List, Tuple, Dict


FREQ_REUSE_7BEAM = {0: 0, 1: 1, 2: 2, 3: 3, 4: 1, 5: 2, 6: 3}


def beam_gain_db(
    theta_deg: float,
    g_max_dbi: float = 30.0,
    theta_3db_deg: float = 1.5,
) -> float:
    """ITU-R S.1528 parabolic beam pattern with sidelobe floor."""
    g = g_max_dbi - 12.0 * (theta_deg / theta_3db_deg) ** 2
    return max(g, g_max_dbi - 25.0)


def angular_separation_deg(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Great-circle angular separation in degrees (Vincenty formula)."""
    lat1r, lon1r = np.radians(lat1), np.radians(lon1)
    lat2r, lon2r = np.radians(lat2), np.radians(lon2)
    dlon = lon2r - lon1r
    a = (np.cos(lat2r) * np.sin(dlon)) ** 2 + (
        np.cos(lat1r) * np.sin(lat2r)
        - np.sin(lat1r) * np.cos(lat2r) * np.cos(dlon)
    ) ** 2
    b = np.sin(lat1r) * np.sin(lat2r) + np.cos(lat1r) * np.cos(lat2r) * np.cos(dlon)
    return np.degrees(np.arctan2(np.sqrt(a), b))


def generate_beam_layout(
    sub_lat: float,
    sub_lon: float,
    num_beams: int = 7,
    beam_spacing_deg: float = 2.0,
) -> List[Dict]:
    """
    Hexagonal beam layout centred on the satellite sub-point.

    Returns list of dicts with center_lat, center_lon, beam_idx, freq_color.
    """
    beams = [
        {
            "beam_idx": 0,
            "center_lat": sub_lat,
            "center_lon": sub_lon,
            "freq_color": FREQ_REUSE_7BEAM[0],
        }
    ]
    if num_beams <= 1:
        return beams

    for i in range(min(num_beams - 1, 6)):
        angle_rad = np.radians(i * 60.0)
        blat = sub_lat + beam_spacing_deg * np.cos(angle_rad)
        blon = sub_lon + beam_spacing_deg * np.sin(angle_rad) / max(
            np.cos(np.radians(sub_lat)), 0.01
        )
        beams.append(
            {
                "beam_idx": i + 1,
                "center_lat": blat,
                "center_lon": blon,
                "freq_color": FREQ_REUSE_7BEAM[i + 1],
            }
        )
    return beams


def compute_interference_sinr(
    user_lat: float,
    user_lon: float,
    serving_sat_idx: int,
    serving_beam_idx: int,
    all_beams: List[Dict],
    signal_power_linear: float,
    noise_power_linear: float,
    path_loss_to_sat: Dict[int, float],
    g_max_dbi: float = 30.0,
    theta_3db_deg: float = 1.5,
) -> float:
    """
    Compute SINR at a user including co-channel interference.

    Args:
        user_lat, user_lon: user position
        serving_sat_idx: index of the serving satellite
        serving_beam_idx: beam index within the serving satellite
        all_beams: flat list of beam dicts, each with keys
            sat_idx, beam_idx, center_lat, center_lon, freq_color
        signal_power_linear: received signal power (linear, not dB)
        noise_power_linear: noise power (linear)
        path_loss_to_sat: {sat_idx: path_loss_linear} for each satellite
        g_max_dbi, theta_3db_deg: beam pattern parameters

    Returns:
        SINR in dB
    """
    serving_freq = None
    for b in all_beams:
        if b["sat_idx"] == serving_sat_idx and b["beam_idx"] == serving_beam_idx:
            serving_freq = b["freq_color"]
            break
    if serving_freq is None:
        return 10.0 * np.log10(signal_power_linear / noise_power_linear)

    interference = 0.0
    for b in all_beams:
        if b["sat_idx"] == serving_sat_idx and b["beam_idx"] == serving_beam_idx:
            continue
        if b["freq_color"] != serving_freq:
            continue

        theta = angular_separation_deg(
            user_lat, user_lon, b["center_lat"], b["center_lon"]
        )
        gain_db = beam_gain_db(theta, g_max_dbi, theta_3db_deg)
        gain_linear = 10.0 ** (gain_db / 10.0)

        sat_pl = path_loss_to_sat.get(b["sat_idx"], 1e-20)
        interference += sat_pl * gain_linear

    sinr_linear = signal_power_linear / (interference + noise_power_linear)
    return 10.0 * np.log10(max(sinr_linear, 1e-20))
