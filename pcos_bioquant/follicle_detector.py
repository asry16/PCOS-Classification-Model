"""
Antral Follicle Detection Module for PCOS-BioQuant.
Localizes hypoechoic (fluid-filled) antral follicles using multi-scale morphological
top-hat filtering, radial gradient wall detection, and geometric ellipse fitting.
"""

import math
import cv2
import numpy as np


class AntralFollicleDetector:
    def __init__(
        self,
        pixel_per_mm: float = 7.5,
        min_follicle_mm: float = 2.0,
        max_follicle_mm: float = 9.5,
        min_contrast: float = 0.12,
        nms_iou_threshold: float = 0.35,
    ):
        self.pixel_per_mm = pixel_per_mm
        self.min_follicle_mm = min_follicle_mm
        self.max_follicle_mm = max_follicle_mm
        self.min_contrast = min_contrast
        self.nms_iou_threshold = nms_iou_threshold

        # Pixel radii corresponding to 2mm to 9.5mm at canonical scale
        self.min_radius_px = max(5, int((min_follicle_mm / 2.0) * pixel_per_mm))
        self.max_radius_px = int((max_follicle_mm / 2.0) * pixel_per_mm)

    def detect(self, canonical_gray: np.ndarray, enhanced: np.ndarray, ovary_mask: np.ndarray, ovary_centroid: tuple):
        """
        Detects, validates, and fits ellipses to all antral follicles within the ovarian stroma.
        """
        h, w = canonical_gray.shape
        cx_ov, cy_ov = ovary_centroid

        # Scales covering the antral follicle spectrum (2-9 mm)
        scale_radii = [6, 9, 13, 17, 22, 28]
        candidates = []

        ovary_mean_intensity = float(np.mean(canonical_gray[ovary_mask > 0])) if np.any(ovary_mask > 0) else 100.0

        for r in scale_radii:
            se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
            # Black Top-Hat isolates dark features of size <= r
            blackhat = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, se)
            masked_bh = np.where(ovary_mask > 0, blackhat, 0)

            # Adaptive regional percentile threshold
            if np.any(ovary_mask > 0):
                thresh_val = float(np.percentile(masked_bh[ovary_mask > 0], 84))
            else:
                thresh_val = 22.0
            thresh_val = max(16.0, thresh_val)

            _, b_bin = cv2.threshold(masked_bh, thresh_val, 255, cv2.THRESH_BINARY)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(b_bin)

            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                bx, by, bw, bh = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
                fx, fy = centroids[i]

                # Check if centroid is within ovary
                if (0 <= int(fy) < h) and (0 <= int(fx) < w):
                    if ovary_mask[int(fy), int(fx)] == 0:
                        continue

                expected_area = math.pi * (r ** 2)
                if (0.30 * expected_area <= area <= 2.5 * expected_area):
                    aspect = float(bw) / max(1, bh)
                    if 0.40 <= aspect <= 2.4:
                        equiv_r = math.sqrt(area / math.pi)
                        if self.min_radius_px * 0.75 <= equiv_r <= self.max_radius_px * 1.35:
                            # Peak response in blackhat
                            strength = float(masked_bh[int(fy), int(fx)])
                            candidates.append({
                                "x": float(fx),
                                "y": float(fy),
                                "radius": float(equiv_r),
                                "area": float(area),
                                "bw": int(bw),
                                "bh": int(bh),
                                "strength": strength
                            })

        if not candidates:
            return []

        # Sort candidates by morphological strength
        candidates.sort(key=lambda c: -c["strength"])

        # Non-Maximum Suppression (NMS) and Acoustic Validation
        validated = []
        y_grid, x_grid = np.ogrid[:h, :w]

        for cand in candidates:
            fx, fy = cand["x"], cand["y"]
            r = cand["radius"]

            # 1. Spatial suppression against already validated follicles
            overlap = False
            for v in validated:
                dist = math.hypot(fx - v["x"], fy - v["y"])
                if dist < 0.70 * (r + v["radius_px"]):
                    overlap = True
                    break
            if overlap:
                continue

            # 2. Local Acoustic Contrast Validation:
            # Fluid cavity (hypoechoic core) vs surrounding echogenic stroma rim
            dist_mat = np.hypot(x_grid - fx, y_grid - fy)
            core_mask = (dist_mat <= 0.65 * r) & (ovary_mask > 0)
            rim_mask = (dist_mat >= 1.15 * r) & (dist_mat <= 1.85 * r) & (ovary_mask > 0)

            if np.sum(core_mask) < 5 or np.sum(rim_mask) < 10:
                continue

            core_mean = float(np.mean(canonical_gray[core_mask]))
            rim_mean = float(np.mean(canonical_gray[rim_mask]))

            # Hypoechoic contrast formula: C = (I_rim - I_core) / I_rim
            contrast = (rim_mean - core_mean) / max(1.0, rim_mean)

            # In pelvic ultrasound:
            # - Antral follicles are dark fluid pockets: contrast >= min_contrast
            # - Core intensity must not be brighter than overall ovarian stroma
            if contrast >= self.min_contrast and core_mean < ovary_mean_intensity * 1.15:
                # 3. Geometric Ellipse Fitting to Follicle Wall
                # Extract local patch around follicle
                pad = int(r * 1.4)
                px1, py1 = max(0, int(fx - pad)), max(0, int(fy - pad))
                px2, py2 = min(w, int(fx + pad)), min(h, int(fy + pad))
                patch = canonical_gray[py1:py2, px1:px2]

                # Threshold local patch to find follicle contour
                _, local_bin = cv2.threshold(patch, int(rim_mean * 0.82), 255, cv2.THRESH_BINARY_INV)
                patch_contours, _ = cv2.findContours(local_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                major_ax = r * 2.0
                minor_ax = r * 2.0
                angle = 0.0

                if patch_contours:
                    # Select contour closest to patch center
                    p_cx = fx - px1
                    p_cy = fy - py1
                    best_f_c = None
                    best_d = 9999.0
                    for pc in patch_contours:
                        pm = cv2.moments(pc)
                        if pm["m00"] > 0:
                            mcx = pm["m10"] / pm["m00"]
                            mcy = pm["m01"] / pm["m00"]
                            d = math.hypot(mcx - p_cx, mcy - p_cy)
                            if d < best_d and cv2.contourArea(pc) > 10:
                                best_d = d
                                best_f_c = pc
                    if best_f_c is not None and len(best_f_c) >= 5:
                        (f_cx_p, f_cy_p), (d1, d2), angle = cv2.fitEllipse(best_f_c)
                        major_ax = max(d1, d2)
                        minor_ax = min(d1, d2)

                diam_px = float(2.0 * r)
                diam_mm = round(diam_px / self.pixel_per_mm, 2)

                validated.append({
                    "id": len(validated) + 1,
                    "x": round(fx, 2),
                    "y": round(fy, 2),
                    "radius_px": round(r, 2),
                    "diameter_px": round(diam_px, 2),
                    "diameter_mm": diam_mm,
                    "major_axis_px": round(major_ax, 2),
                    "minor_axis_px": round(minor_ax, 2),
                    "ellipse_angle": round(angle, 1),
                    "contrast": round(contrast, 3),
                    "core_mean": round(core_mean, 1),
                    "rim_mean": round(rim_mean, 1),
                })

        return validated
