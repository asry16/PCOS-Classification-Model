"""
Geometric Metrics & Biomarker Profiling Module for PCOS-BioQuant.
Implements the novel Peripheral Dispersion Index (PDI), Peripheral Ring Concentration (PRC),
Central Stromal Sparing, and Ovarian Morphometric Biomarkers.
"""

import math
import numpy as np


class GeometricMetricsCalculator:
    def __init__(self, peripheral_cutoff: float = 0.65, central_cutoff: float = 0.40):
        self.peripheral_cutoff = peripheral_cutoff
        self.central_cutoff = central_cutoff

    def compute(self, follicles: list, ovary_data: dict):
        """
        Computes the complete quantitative biomarker profile for an ovarian ultrasound scan.
        """
        cx, cy = ovary_data["centroid"]
        hull = ovary_data["hull"]
        ovary_area = ovary_data["area_px"]
        hull_pts = hull.reshape(-1, 2).astype(np.float32)

        N = len(follicles)
        if N == 0:
            return {
                "fnpo": 0,
                "pdi_mean": 0.0,
                "pdi_median": 0.0,
                "pdi_std": 0.0,
                "pdi_iqr": 0.0,
                "prc_65": 0.0,
                "prc_75": 0.0,
                "central_fraction": 0.0,
                "stroma_to_ovary_ratio": 1.0,
                "mean_diameter_mm": 0.0,
                "median_diameter_mm": 0.0,
                "follicles_enriched": [],
                "pdi_values": [],
            }

        pdi_values = []
        enriched_follicles = []
        total_follicle_area = 0.0

        for f in follicles:
            fx, fy = f["x"], f["y"]
            r_px = f["radius_px"]
            total_follicle_area += math.pi * (r_px ** 2)

            # Euclidean distance from ovarian center to follicle center
            d_center = math.hypot(fx - cx, fy - cy)
            theta = math.atan2(fy - cy, fx - cx)

            # Ray-Capsule intersection to find boundary distance R(theta)
            ray_dx = math.cos(theta)
            ray_dy = math.sin(theta)
            max_dist = 800.0

            p1 = (cx, cy)
            p2 = (cx + ray_dx * max_dist, cy + ray_dy * max_dist)

            closest_b_dist = max_dist
            bx = cx + ray_dx * 50
            by = cy + ray_dy * 50

            for i in range(len(hull_pts)):
                q1 = hull_pts[i]
                q2 = hull_pts[(i + 1) % len(hull_pts)]

                denom = (p2[0] - p1[0]) * (q2[1] - q1[1]) - (p2[1] - p1[1]) * (q2[0] - q1[0])
                if abs(denom) > 1e-6:
                    t = ((q1[0] - p1[0]) * (q2[1] - q1[1]) - (q1[1] - p1[1]) * (q2[0] - q1[0])) / denom
                    u = ((q1[0] - p1[0]) * (p2[1] - p1[1]) - (q1[1] - p1[1]) * (p2[0] - p1[0])) / denom
                    if t > 0 and 0.0 <= u <= 1.0:
                        inter_d = math.hypot(p1[0] + t * (p2[0] - p1[0]) - cx, p1[1] + t * (p2[1] - p1[1]) - cy)
                        if inter_d < closest_b_dist:
                            closest_b_dist = inter_d
                            bx = p1[0] + t * (p2[0] - p1[0])
                            by = p1[1] + t * (p2[1] - p1[1])

            # Normalized Peripheral Dispersion rho_i in [0, 1]
            rho = float(np.clip(d_center / max(1.0, closest_b_dist), 0.05, 0.98))
            pdi_values.append(rho)

            f_enriched = dict(f)
            f_enriched["pdi"] = round(rho, 3)
            f_enriched["center_dist_px"] = round(d_center, 1)
            f_enriched["boundary_dist_px"] = round(closest_b_dist, 1)
            f_enriched["boundary_pt"] = (round(float(bx), 1), round(float(by), 1))
            f_enriched["is_peripheral"] = bool(rho >= self.peripheral_cutoff)
            f_enriched["is_central"] = bool(rho <= self.central_cutoff)
            enriched_follicles.append(f_enriched)

        pdi_arr = np.array(pdi_values)
        pdi_mean = float(np.mean(pdi_arr))
        pdi_median = float(np.median(pdi_arr))
        pdi_std = float(np.std(pdi_arr))
        q75, q25 = np.percentile(pdi_arr, [75, 25])
        pdi_iqr = float(q75 - q25)

        # Peripheral Ring Concentration (PRC)
        prc_65 = float(np.mean(pdi_arr >= self.peripheral_cutoff))
        prc_75 = float(np.mean(pdi_arr >= 0.75))
        central_fraction = float(np.mean(pdi_arr <= self.central_cutoff))

        # Stroma-to-Ovary Area Ratio
        stroma_area = max(0.0, ovary_area - total_follicle_area)
        stroma_ratio = float(stroma_area / max(1.0, ovary_area))

        # Follicle Diameters
        diameters = [f["diameter_mm"] for f in follicles]
        mean_diam = float(np.mean(diameters))
        median_diam = float(np.median(diameters))

        return {
            "fnpo": N,
            "pdi_mean": round(pdi_mean, 3),
            "pdi_median": round(pdi_median, 3),
            "pdi_std": round(pdi_std, 3),
            "pdi_iqr": round(pdi_iqr, 3),
            "prc_65": round(prc_65, 3),
            "prc_75": round(prc_75, 3),
            "central_fraction": round(central_fraction, 3),
            "stroma_to_ovary_ratio": round(stroma_ratio, 3),
            "mean_diameter_mm": round(mean_diam, 2),
            "median_diameter_mm": round(median_diam, 2),
            "follicles_enriched": enriched_follicles,
            "pdi_values": [round(v, 3) for v in pdi_values],
        }
