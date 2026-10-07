"""
Ovarian Boundary Segmentation Module for PCOS-BioQuant.
Extracts the ovarian capsule contour, centroid (center of mass),
fitted bounding ellipse, and central stromal zone.
"""

import math
import cv2
import numpy as np


class OvarySegmenter:
    def __init__(self, stroma_core_radius_ratio: float = 0.35):
        self.stroma_core_radius_ratio = stroma_core_radius_ratio

    def segment(self, canonical_gray: np.ndarray, enhanced: np.ndarray, orig_shape: tuple = None):
        """
        Segments the ovarian capsule boundary and computes ovarian geometry.
        Returns a dictionary containing centroid, contour, hull, ellipse, mask, and area.
        """
        h, w = canonical_gray.shape
        img_center = np.array([w / 2.0, h / 2.0])

        # Step 1: Low-pass Gaussian filtering to isolate macroscopic organ parenchyma
        blurred = cv2.GaussianBlur(canonical_gray, (23, 23), 0)
        
        # Step 2: Otsu adaptive thresholding
        _, otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Step 3: Morphological closing to unify ovarian stroma across intra-follicular gaps
        k_size = max(21, int(min(h, w) * 0.08))
        if k_size % 2 == 0:
            k_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        coalesced = cv2.morphologyEx(otsu_thresh, cv2.MORPH_CLOSE, kernel)

        # Step 4: Extract candidates and select the anatomically plausible ovarian contour
        contours, _ = cv2.findContours(coalesced, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_contour = None
        if contours:
            valid = []
            for c in contours:
                area = cv2.contourArea(c)
                if area > 0.10 * (w * h):
                    M = cv2.moments(c)
                    if M["m00"] > 0:
                        c_center = np.array([M["m10"] / M["m00"], M["m01"] / M["m00"]])
                        dist = np.linalg.norm(c_center - img_center)
                        valid.append((area, dist, c))
            if valid:
                # Rank by area penalized by distance from image center
                valid.sort(key=lambda item: -item[0] + item[1] * 250)
                best_contour = valid[0][2]
            else:
                best_contour = max(contours, key=cv2.contourArea)

        # Step 5: Fallback if contour is degenerated
        if best_contour is None or cv2.contourArea(best_contour) < 0.08 * (w * h):
            cx, cy = w / 2.0, h / 2.0
            rx, ry = w * 0.42, h * 0.42
            hull = cv2.ellipse2Poly((int(cx), int(cy)), (int(rx), int(ry)), 0, 0, 360, 5)
            ellipse = ((cx, cy), (2 * rx, 2 * ry), 0.0)
            area = math.pi * rx * ry
        else:
            # Smooth capsule via convex hull
            hull = cv2.convexHull(best_contour)
            area = float(cv2.contourArea(hull))
            
            # Robust ellipse fit
            if len(hull) >= 5:
                ellipse = cv2.fitEllipse(hull)
                (cx, cy), (d1, d2), angle = ellipse
                # Keep centroid well within organ boundaries
                cx = float(np.clip(cx, w * 0.25, w * 0.75))
                cy = float(np.clip(cy, h * 0.25, h * 0.75))
                ellipse = ((cx, cy), (d1, d2), angle)
            else:
                M = cv2.moments(hull)
                cx = float(M["m10"] / M["m00"]) if M["m00"] > 0 else w / 2.0
                cy = float(M["m01"] / M["m00"]) if M["m00"] > 0 else h / 2.0
                rx, ry = w * 0.4, h * 0.4
                ellipse = ((cx, cy), (2 * rx, 2 * ry), 0.0)

        # Binary mask
        ovary_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(ovary_mask, [hull], -1, 255, -1)

        # Ellipse geometric properties
        (_, _), (axis_1, axis_2), angle = ellipse
        major_axis = max(axis_1, axis_2)
        minor_axis = min(axis_1, axis_2)
        eccentricity = math.sqrt(max(0.0, 1.0 - (minor_axis / max(1e-4, major_axis)) ** 2))

        # Central Stroma Core Mask (inner core where follicles are absent in classic PCOS)
        stroma_mask = np.zeros((h, w), dtype=np.uint8)
        stroma_radius_x = int(major_axis * 0.5 * self.stroma_core_radius_ratio)
        stroma_radius_y = int(minor_axis * 0.5 * self.stroma_core_radius_ratio)
        if stroma_radius_x > 0 and stroma_radius_y > 0:
            cv2.ellipse(stroma_mask, (int(cx), int(cy)), (stroma_radius_x, stroma_radius_y), int(angle), 0, 360, 255, -1)
        stroma_mask = cv2.bitwise_and(stroma_mask, ovary_mask)

        # Mean stroma echogenicity
        stroma_echogenicity = float(np.mean(canonical_gray[stroma_mask > 0])) if np.any(stroma_mask > 0) else float(np.mean(canonical_gray))

        # Anatomical Organ Validity Check:
        # Detect if image contains uterine characteristics (transvaginal probe sector view with myometrial striation)
        orig_aspect = (orig_shape[1] / max(1, orig_shape[0])) if orig_shape else 1.0
        sob_y = cv2.Sobel(canonical_gray, cv2.CV_32F, 0, 1, ksize=3)
        mean_horiz_edge = float(np.mean(np.abs(sob_y)))

        is_uterine = bool(orig_aspect > 1.35 and mean_horiz_edge > 21.0)
        is_ovary = not is_uterine
        organ_type = "Ovary" if is_ovary else "Uterine Corpus / Non-Ovarian Scan"

        return {
            "centroid": (round(cx, 2), round(cy, 2)),
            "hull": hull,
            "ellipse": ellipse,
            "mask": ovary_mask,
            "stroma_mask": stroma_mask,
            "area_px": round(area, 1),
            "major_axis_px": round(major_axis, 1),
            "minor_axis_px": round(minor_axis, 1),
            "eccentricity": round(eccentricity, 3),
            "stroma_echogenicity": round(stroma_echogenicity, 1),
            "is_ovary": is_ovary,
            "organ_type": organ_type
        }
