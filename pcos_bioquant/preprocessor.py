"""
Ultrasound Preprocessing Module for PCOS-BioQuant.
Handles ultrasound scan sector detection, UI/header artifact removal,
multiplicative speckle noise suppression, and contrast enhancement.
"""

import math
import cv2
import numpy as np


class UltrasoundPreprocessor:
    def __init__(self, target_dim: int = 384, clahe_clip: float = 2.2, clahe_grid: tuple = (8, 8)):
        self.target_dim = target_dim
        self.clahe_clip = clahe_clip
        self.clahe_grid = clahe_grid

    def process(self, img_bgr: np.ndarray):
        """
        Runs full preprocessing on an input ultrasound image (BGR format).
        Returns a dictionary containing standardized gray, enhanced, and denoised images
        along with spatial scaling metadata.
        """
        if img_bgr is None:
            raise ValueError("Input image is None")

        orig_h, orig_w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape) == 3 else img_bgr.copy()

        # Step 1: Detect and crop active acoustic sector if image has broad black borders
        crop_box = self._extract_acoustic_sector(gray)
        x1, y1, x2, y2 = crop_box
        cropped_bgr = img_bgr[y1:y2, x1:x2]
        cropped_gray = gray[y1:y2, x1:x2]

        crop_h, crop_w = cropped_gray.shape

        # Step 2: Canonical isotropic resizing for uniform scale processing
        scale_x = self.target_dim / float(crop_w)
        scale_y = self.target_dim / float(crop_h)
        canonical_gray = cv2.resize(cropped_gray, (self.target_dim, self.target_dim), interpolation=cv2.INTER_AREA)
        canonical_bgr = cv2.resize(cropped_bgr, (self.target_dim, self.target_dim), interpolation=cv2.INTER_AREA)

        # Step 3: Speckle suppression via edge-preserving Bilateral Filter
        # Bilateral filter reduces Rayleigh speckle without blurring the thin hyperechoic follicle boundary
        denoised = cv2.bilateralFilter(canonical_gray, d=7, sigmaColor=35, sigmaSpace=35)

        # Step 4: Local acoustic contrast enhancement via CLAHE
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=self.clahe_grid)
        enhanced = clahe.apply(denoised)

        # Step 5: Texture and noise metrics
        speckle_snr = float(np.mean(denoised)) / (float(np.std(denoised)) + 1e-5)

        return {
            "canonical_bgr": canonical_bgr,
            "canonical_gray": canonical_gray,
            "denoised": denoised,
            "enhanced": enhanced,
            "orig_shape": (orig_h, orig_w),
            "crop_box": crop_box,
            "scale_factors": (scale_x, scale_y),
            "target_dim": self.target_dim,
            "speckle_snr": round(speckle_snr, 2)
        }

    def _extract_acoustic_sector(self, gray: np.ndarray):
        """
        Delineates active tissue sector from dark probe background and screen headers.
        """
        h, w = gray.shape
        # If image is already a cropped ROI (approx square and smaller than 450px), keep as is
        aspect = float(w) / max(1, h)
        if 0.75 <= aspect <= 1.35 and max(h, w) <= 450:
            return (0, 0, w, h)

        # Mask tissue (> 18 intensity)
        _, thresh = cv2.threshold(gray, 18, 255, cv2.THRESH_BINARY)
        # Remove small UI dots/text via opening
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_clean)
        
        # Merge scan lines via closing
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)

        # Find largest connected tissue area
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return (0, 0, w, h)

        best_c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(best_c)

        if area < 0.10 * (w * h):
            return (0, 0, w, h)

        bx, by, bw, bh = cv2.boundingRect(best_c)
        pad = 8
        x1 = max(0, bx - pad)
        y1 = max(0, by - pad)
        x2 = min(w, bx + bw + pad)
        y2 = min(h, by + bh + pad)

        # Ensure valid non-empty box
        if (x2 - x1 < 60) or (y2 - y1 < 60):
            return (0, 0, w, h)

        return (x1, y1, x2, y2)
