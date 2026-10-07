"""
Clinical Visualizer Module for PCOS-BioQuant.
Generates publication-quality clinical overlays, multi-panel diagnostic figures,
and automated ultrasound report cards.
"""

import math
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class ClinicalVisualizer:
    def __init__(self, dpi: int = 150):
        self.dpi = dpi

    def render_overlay(self, canonical_bgr: np.ndarray, ovary_data: dict, metrics: dict, report: dict):
        """
        Creates an annotated visual overlay image using OpenCV.
        """
        overlay = canonical_bgr.copy()
        h, w = canonical_bgr.shape[:2]
        cx, cy = int(ovary_data["centroid"][0]), int(ovary_data["centroid"][1])
        hull = ovary_data["hull"]
        ellipse = ovary_data["ellipse"]

        # 1. Draw Ovarian Capsule Boundary (Cyan / Teal, thickness 2)
        cv2.polylines(overlay, [hull], isClosed=True, color=(255, 235, 0), thickness=2, lineType=cv2.LINE_AA)

        # 2. Draw Central Stroma Core Boundary (Subtle Dashed Orange)
        (e_cx, e_cy), (d1, d2), angle = ellipse
        sr_x = int(max(d1, d2) * 0.5 * 0.35)
        sr_y = int(min(d1, d2) * 0.5 * 0.35)
        if sr_x > 0 and sr_y > 0:
            cv2.ellipse(overlay, (cx, cy), (sr_x, sr_y), int(angle), 0, 360, (0, 140, 255), 1, lineType=cv2.LINE_AA)

        # 3. Draw Centroid (Yellow Crosshair)
        cv2.circle(overlay, (cx, cy), 3, (0, 255, 255), -1, lineType=cv2.LINE_AA)
        cv2.drawMarker(overlay, (cx, cy), (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2)

        # 4. Draw Follicles & Vectors
        for f in metrics.get("follicles_enriched", []):
            fx, fy = int(f["x"]), int(f["y"])
            r = max(3, int(f["radius_px"]))
            rho = f["pdi"]
            is_peri = f["is_peripheral"]

            # Color coding: Lime for peripheral (String-of-Pearls), Amber for central
            color = (0, 255, 128) if is_peri else (0, 165, 255)

            # Follicle ring
            cv2.circle(overlay, (fx, fy), r, color, 2, lineType=cv2.LINE_AA)
            cv2.circle(overlay, (fx, fy), 2, color, -1, lineType=cv2.LINE_AA)

            # Center-to-follicle radial vector
            cv2.line(overlay, (cx, cy), (fx, fy), (180, 180, 180), 1, lineType=cv2.LINE_AA)

            # Follicle-to-capsule ray
            bx, by = int(f["boundary_pt"][0]), int(f["boundary_pt"][1])
            cv2.line(overlay, (fx, fy), (bx, by), (50, 50, 240), 1, lineType=cv2.LINE_AA)

            # PDI value badge
            pdi_str = f"{rho:.2f}"
            cv2.putText(overlay, pdi_str, (fx + r + 3, fy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

        # 5. Executive Diagnostic Header Banner
        banner_h = 52
        banner = np.zeros((banner_h, w, 3), dtype=np.uint8)
        
        # Banner color: Crimson for PCOM positive, Emerald for Normal, Slate for Borderline
        if "POSITIVE" in report["morphology_match"]:
            header_color = (25, 25, 160)
            tag_color = (80, 80, 255)
        elif "NEGATIVE" in report["morphology_match"]:
            header_color = (20, 100, 30)
            tag_color = (50, 220, 80)
        else:
            header_color = (40, 70, 120)
            tag_color = (80, 180, 255)

        banner[:] = header_color
        cv2.line(banner, (0, banner_h - 1), (w, banner_h - 1), tag_color, 2)

        # Text on banner
        cv2.putText(banner, f"PCOS-BioQuant: {report['morphology_match'].split('(')[0].strip()}", 
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(banner, f"FNPO: {report['fnpo']} | Mean PDI: {report['pdi_mean']:.3f} | PRC(>=0.65): {report['prc_65_pct']}% | Conf: {report['confidence_pct']}%", 
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 240, 255), 1, cv2.LINE_AA)

        combined = np.vstack([banner, overlay])
        return combined

    def render_paper_figure(self, canonical_bgr: np.ndarray, enhanced: np.ndarray, ovary_data: dict, metrics: dict, report: dict, save_path: str):
        """
        Creates a high-resolution 4-panel publication-grade figure for scientific papers.
        """
        fig, axes = plt.subplots(1, 4, figsize=(18, 5), dpi=self.dpi)
        
        # Panel A: Original B-Mode Ultrasound
        axes[0].imshow(cv2.cvtColor(canonical_bgr, cv2.COLOR_BGR2RGB))
        axes[0].set_title("(a) B-Mode Transvaginal Ultrasound", fontsize=11, fontweight="bold", pad=8)
        axes[0].axis("off")

        # Panel B: Contrast-Enhanced & Despeckled
        axes[1].imshow(enhanced, cmap="gray")
        axes[1].set_title("(b) Despeckled & CLAHE Enhanced", fontsize=11, fontweight="bold", pad=8)
        axes[1].axis("off")

        # Panel C: Ovarian Capsule & Stroma Segmentation
        axes[2].imshow(cv2.cvtColor(canonical_bgr, cv2.COLOR_BGR2RGB))
        hull_pts = ovary_data["hull"].reshape(-1, 2)
        hull_plot = np.vstack([hull_pts, hull_pts[0]])
        axes[2].plot(hull_plot[:, 0], hull_plot[:, 1], color="#00e5ff", linewidth=2.0, label="Ovarian Capsule")
        cx, cy = ovary_data["centroid"]
        axes[2].scatter([cx], [cy], color="#ffea00", s=90, marker="+", linewidths=2.5, label="Ovarian Centroid")

        # Central Stroma Ellipse
        (e_cx, e_cy), (d1, d2), angle = ovary_data["ellipse"]
        stroma_patch = patches.Ellipse(
            (cx, cy), max(d1, d2) * 0.35, min(d1, d2) * 0.35,
            angle=angle, fill=True, color="#ff9100", alpha=0.25, label="Central Stroma Zone"
        )
        axes[2].add_patch(stroma_patch)
        axes[2].set_title("(c) Capsule Boundary & Stroma Zone", fontsize=11, fontweight="bold", pad=8)
        axes[2].legend(loc="lower right", fontsize=8, framealpha=0.7)
        axes[2].axis("off")

        # Panel D: Follicle Localization & PDI Radial Ray-Casting
        axes[3].imshow(cv2.cvtColor(canonical_bgr, cv2.COLOR_BGR2RGB))
        axes[3].plot(hull_plot[:, 0], hull_plot[:, 1], color="#00e5ff", linewidth=1.5)
        axes[3].scatter([cx], [cy], color="#ffea00", s=60, marker="+", linewidths=2.0)

        for f in metrics.get("follicles_enriched", []):
            fx, fy = f["x"], f["y"]
            r = f["radius_px"]
            rho = f["pdi"]
            is_peri = f["is_peripheral"]

            color = "#00e676" if is_peri else "#ff9100"
            circ = plt.Circle((fx, fy), r, color=color, fill=False, linewidth=1.8)
            axes[3].add_patch(circ)

            # Center-to-follicle vector
            axes[3].plot([cx, fx], [cy, fy], color="#ffffff", linestyle=":", linewidth=0.8, alpha=0.7)
            # Follicle-to-capsule ray
            bx, by = f["boundary_pt"]
            axes[3].plot([fx, bx], [fy, by], color="#ff1744", linestyle="--", linewidth=0.9, alpha=0.8)

            axes[3].text(fx + r + 2, fy, f"{rho:.2f}", color="#ffffff", fontsize=7,
                         bbox=dict(boxstyle="square,pad=0.1", fc="black", ec="none", alpha=0.6))

        pdi_mean = metrics["pdi_mean"]
        fnpo = metrics["fnpo"]
        diag = "PCOM Positive" if "POSITIVE" in report["morphology_match"] else ("Borderline" if "BORDERLINE" in report["morphology_match"] else "Normal")
        axes[3].set_title(f"(d) BioQuant PDI Map (FNPO: {fnpo} | PDI: {pdi_mean:.3f})\nDiagnosis: {diag}", fontsize=11, fontweight="bold", pad=8)
        axes[3].axis("off")

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
