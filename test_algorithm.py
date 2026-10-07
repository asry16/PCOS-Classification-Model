import os
import glob
import math
import cv2
import numpy as np

def preprocess_ultrasound(img_bgr):
    """
    Extracts the active ultrasound ROI and applies speckle reduction.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Check if there is a dominant black border (ultrasound cone/fan scan)
    # Threshold for dark background
    _, bg_mask = cv2.threshold(gray, 18, 255, cv2.THRESH_BINARY)
    
    # Morphological closing to fill acoustic holes in tissue
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tissue_closed = cv2.morphologyEx(bg_mask, cv2.MORPH_CLOSE, kernel)
    
    # Find largest tissue component
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(tissue_closed)
    
    if num_labels > 1:
        # Find largest non-background component
        largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        x, y, bw, bh, area = stats[largest_idx]
        
        # If the tissue region occupies a significant part but has borders, crop ROI
        if area > 0.15 * (w * h) and (bw < 0.95 * w or bh < 0.95 * h):
            # Crop to tissue bounding box with small margin
            pad = 8
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(w, x + bw + pad), min(h, y + bh + pad)
            gray = gray[y1:y2, x1:x2]
            img_bgr = img_bgr[y1:y2, x1:x2]
    
    # Speckle reduction: Bilateral filter preserves sharp follicle edges while smoothing speckle noise
    denoised = cv2.bilateralFilter(gray, d=7, sigmaColor=35, sigmaSpace=35)
    
    # CLAHE for acoustic contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    
    return img_bgr, gray, enhanced

def segment_ovarian_boundary(gray, enhanced):
    """
    Delineates the ovarian capsule boundary using adaptive thresholding,
    convex hull, and robust ellipse fitting.
    """
    h, w = gray.shape
    
    # Smooth to get macroscopic ovarian tissue mass
    blurred = cv2.GaussianBlur(gray, (19, 19), 0)
    
    # Otsu thresholding with tissue bias
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological closing to merge ovarian tissue into a single connected mass
    k_size = max(15, min(h, w) // 12)
    if k_size % 2 == 0:
        k_size += 1
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    ovary_mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k)
    
    # Find contours
    contours, _ = cv2.findContours(ovary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # Fallback to image-centric ellipse
        cx, cy = w / 2.0, h / 2.0
        rx, ry = w * 0.42, h * 0.42
        box = ((cx, cy), (2 * rx, 2 * ry), 0)
        # Approximate contour
        pts = cv2.ellipse2Poly((int(cx), int(cy)), (int(rx), int(ry)), 0, 0, 360, 5)
        return box, (cx, cy), pts, np.ones_like(gray, dtype=np.uint8) * 255
    
    # Pick contour closest to center with good area
    img_center = np.array([w / 2.0, h / 2.0])
    valid_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        if area > 0.12 * (w * h):
            M = cv2.moments(c)
            if M["m00"] > 0:
                c_center = np.array([M["m10"] / M["m00"], M["m01"] / M["m00"]])
                dist = np.linalg.norm(c_center - img_center)
                valid_contours.append((area, dist, c))
    
    if valid_contours:
        # Sort by combination of area and proximity to center
        valid_contours.sort(key=lambda item: -item[0] + item[1] * 200)
        best_c = valid_contours[0][2]
    else:
        best_c = max(contours, key=cv2.contourArea)
    
    # Compute convex hull for anatomically smooth capsule
    hull = cv2.convexHull(best_c)
    
    # Fit ellipse if enough points
    if len(hull) >= 5:
        ovary_ellipse = cv2.fitEllipse(hull)
        cx, cy = ovary_ellipse[0]
        # Ensure center is within image
        cx = np.clip(cx, w * 0.2, w * 0.8)
        cy = np.clip(cy, h * 0.2, h * 0.8)
        ovary_ellipse = ((cx, cy), ovary_ellipse[1], ovary_ellipse[2])
    else:
        M = cv2.moments(hull)
        cx = M["m10"] / M["m00"] if M["m00"] > 0 else w / 2.0
        cy = M["m01"] / M["m00"] if M["m00"] > 0 else h / 2.0
        rx, ry = w * 0.4, h * 0.4
        ovary_ellipse = ((cx, cy), (2 * rx, 2 * ry), 0)
    
    # Create binary mask of the ovary
    mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.drawContours(mask, [hull], -1, 255, -1)
    
    return ovary_ellipse, (cx, cy), hull, mask

def detect_antral_follicles(gray, enhanced, ovary_mask, ovary_center):
    """
    Localizes hypoechoic (anechoic) antral follicles within the ovarian stroma.
    Uses multi-threshold regional segmentation and morphological filtering.
    """
    h, w = gray.shape
    cx, cy = ovary_center
    
    # Follicles are dark regions surrounded by brighter stroma
    # Local adaptive thresholding or bottom-hat transform
    # Black Top-Hat (Bottom-Hat) emphasizes dark structures smaller than kernel
    follicle_candidates = []
    
    # Multi-scale bottom-hat
    for k_rad in [5, 9, 13, 17]:
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_rad * 2 + 1, k_rad * 2 + 1))
        blackhat = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, se)
        
        # Adaptive threshold on blackhat
        thresh_val = np.percentile(blackhat[ovary_mask > 0], 82) if np.any(ovary_mask > 0) else 25
        _, bin_cand = cv2.threshold(blackhat, max(12, thresh_val), 255, cv2.THRESH_BINARY)
        bin_cand = cv2.bitwise_and(bin_cand, bin_cand, mask=ovary_mask)
        
        # Watershed / connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_cand)
        
        for i in range(1, num_labels):
            x, y, bw, bh, area = stats[i]
            fx, fy = centroids[i]
            
            # Antral follicle size range: 2-9 mm (typically 20 to 1200 px^2 depending on resolution)
            min_area = max(18, int(0.0003 * (w * h)))
            max_area = int(0.045 * (w * h))
            
            if min_area <= area <= max_area:
                # Aspect ratio check: follicles are circular/elliptical (aspect ratio between 0.4 and 2.5)
                aspect = float(bw) / max(1, bh)
                if 0.35 <= aspect <= 2.8:
                    # Hypoechoic check: internal intensity must be darker than local ovarian mean
                    sample_y, sample_x = int(np.clip(fy, 0, h - 1)), int(np.clip(fx, 0, w - 1))
                    f_int = float(gray[sample_y, sample_x])
                    ovary_mean = float(np.mean(gray[ovary_mask > 0])) if np.any(ovary_mask > 0) else 100
                    
                    if f_int < ovary_mean * 1.15:
                        radius = math.sqrt(area / math.pi)
                        follicle_candidates.append({
                            "center": (fx, fy),
                            "area": area,
                            "radius": radius,
                            "aspect": aspect,
                            "intensity": f_int,
                            "bbox": (x, y, bw, bh)
                        })
    
    # Non-maximum suppression / deduplication of overlapping follicle detections
    if not follicle_candidates:
        return []
    
    # Sort by dark intensity and size
    follicle_candidates.sort(key=lambda f: (f["intensity"], -f["area"]))
    
    cleaned_follicles = []
    for cand in follicle_candidates:
        cfx, cfy = cand["center"]
        c_rad = cand["radius"]
        
        # Check distance to already accepted follicles
        too_close = False
        for accepted in cleaned_follicles:
            afx, afy = accepted["center"]
            dist = math.hypot(cfx - afx, cfy - afy)
            # If centers are closer than 0.75 * sum of radii, merge/suppress
            if dist < 0.75 * (c_rad + accepted["radius"]):
                too_close = True
                break
        
        if not too_close:
            cleaned_follicles.append(cand)
            
    return cleaned_follicles

def compute_peripheral_dispersion_index(follicles, ovary_center, ovary_hull):
    """
    Computes the novel Peripheral Dispersion Index (PDI) for each follicle and aggregate ovary.
    PDI = (Distance of Follicle from Ovarian Center) / (Distance from Center to Ovarian Boundary along ray)
    """
    if not follicles:
        return {
            "pdi_mean": 0.0,
            "pdi_median": 0.0,
            "pdi_std": 0.0,
            "pdi_values": [],
            "peripheral_ratio_65": 0.0,
            "follicle_count": 0
        }
    
    cx, cy = ovary_center
    hull_pts = ovary_hull.reshape(-1, 2).astype(np.float32)
    
    pdi_list = []
    enriched_follicles = []
    
    for f in follicles:
        fx, fy = f["center"]
        
        # 1. Distance from center to follicle
        d_follicle = math.hypot(fx - cx, fy - cy)
        
        # 2. Find intersection of ray (cx, cy) -> (fx, fy) with the ovarian boundary contour
        theta = math.atan2(fy - cy, fx - cx)
        
        # Ray casting: probe along ray from center outward
        ray_dx = math.cos(theta)
        ray_dy = math.sin(theta)
        
        # Max search distance
        max_dist = 600.0
        boundary_dist = None
        
        # Fast ray-polygon intersection
        # Test line segment from center (cx, cy) to (cx + ray_dx * 1000, cy + ray_dy * 1000)
        p1 = (cx, cy)
        p2 = (cx + ray_dx * max_dist, cy + ray_dy * max_dist)
        
        closest_b_dist = max_dist
        found_intersection = False
        
        for i in range(len(hull_pts)):
            q1 = hull_pts[i]
            q2 = hull_pts[(i + 1) % len(hull_pts)]
            
            # Line intersection formula
            # p1 + t*(p2 - p1) = q1 + u*(q2 - q1)
            denom = (p2[0] - p1[0]) * (q2[1] - q1[1]) - (p2[1] - p1[1]) * (q2[0] - q1[0])
            if abs(denom) > 1e-6:
                t = ((q1[0] - p1[0]) * (q2[1] - q1[1]) - (q1[1] - p1[1]) * (q2[0] - q1[0])) / denom
                u = ((q1[0] - p1[0]) * (p2[1] - p1[1]) - (q1[1] - p1[1]) * (p2[0] - p1[0])) / denom
                
                if t > 0 and 0.0 <= u <= 1.0:
                    inter_x = p1[0] + t * (p2[0] - p1[0])
                    inter_y = p1[1] + t * (p2[1] - p1[1])
                    b_dist = math.hypot(inter_x - cx, inter_y - cy)
                    if b_dist < closest_b_dist:
                        closest_b_dist = b_dist
                        found_intersection = True
                        bx, by = inter_x, inter_y
        
        if not found_intersection or closest_b_dist < 1e-3:
            closest_b_dist = max(1.0, d_follicle)
            bx, by = cx + ray_dx * closest_b_dist, cy + ray_dy * closest_b_dist
            
        # Normalized dispersion rho in [0, 1]
        rho = np.clip(d_follicle / closest_b_dist, 0.05, 0.98)
        pdi_list.append(rho)
        
        f_copy = dict(f)
        f_copy["pdi"] = float(rho)
        f_copy["center_dist"] = float(d_follicle)
        f_copy["boundary_dist"] = float(closest_b_dist)
        f_copy["boundary_point"] = (float(bx), float(by))
        enriched_follicles.append(f_copy)
    
    pdi_arr = np.array(pdi_list)
    pdi_mean = float(np.mean(pdi_arr))
    pdi_median = float(np.median(pdi_arr))
    pdi_std = float(np.std(pdi_arr))
    
    # Subcapsular peripheral fraction (rho >= 0.65)
    peripheral_count = int(np.sum(pdi_arr >= 0.65))
    peripheral_ratio = float(peripheral_count / len(pdi_arr))
    
    return {
        "pdi_mean": round(pdi_mean, 3),
        "pdi_median": round(pdi_median, 3),
        "pdi_std": round(pdi_std, 3),
        "pdi_values": pdi_list,
        "peripheral_ratio_65": round(peripheral_ratio, 3),
        "follicle_count": len(follicles),
        "follicles": enriched_follicles
    }

def render_clinical_overlay(img_bgr, ovary_ellipse, ovary_center, ovary_hull, metrics):
    """
    Renders a publication-grade quantitative clinical overlay.
    """
    overlay = img_bgr.copy()
    h, w = img_bgr.shape[:2]
    cx, cy = int(ovary_center[0]), int(ovary_center[1])
    
    # 1. Draw ovarian boundary capsule (Cyan / Teal)
    cv2.polylines(overlay, [ovary_hull], isClosed=True, color=(255, 230, 0), thickness=2, lineType=cv2.LINE_AA)
    
    # 2. Draw central stroma zone (35% radial margin - orange dash or subtle fill)
    cv2.circle(overlay, (cx, cy), 4, (0, 255, 255), -1, lineType=cv2.LINE_AA)
    cv2.drawMarker(overlay, (cx, cy), color=(0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=12, thickness=2)
    
    # 3. Draw follicles & dispersion rays
    for f in metrics.get("follicles", []):
        fx, fy = int(f["center"][0]), int(f["center"][1])
        r = max(3, int(f["radius"]))
        rho = f["pdi"]
        
        # Color coding: Green/Lime if peripheral (PCOS string-of-pearls sign), Orange/Yellow if central
        color = (0, 255, 127) if rho >= 0.65 else (0, 165, 255)
        
        # Follicle contour
        cv2.circle(overlay, (fx, fy), r, color, 2, lineType=cv2.LINE_AA)
        cv2.circle(overlay, (fx, fy), 2, color, -1, lineType=cv2.LINE_AA)
        
        # Radial dispersion vector from ovary center to follicle
        cv2.line(overlay, (cx, cy), (fx, fy), (180, 180, 180), 1, lineType=cv2.LINE_AA)
        
        # Follicle to boundary vector
        bx, by = int(f["boundary_point"][0]), int(f["boundary_point"][1])
        cv2.line(overlay, (fx, fy), (bx, by), (80, 80, 240), 1, lineType=cv2.LINE_AA)
        
        # Label PDI value next to follicle
        label = f"{rho:.2f}"
        cv2.putText(overlay, label, (fx + r + 2, fy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        
    return overlay

# Test on 3 infected and 3 noninfected images
dataset_dir = r"C:\Users\ritur\.cache\kagglehub\datasets\ibadeus\pcos-xai-ultrasound-dataset\versions\1\PCOS"
inf_paths = sorted(glob.glob(os.path.join(dataset_dir, "infected", "*.jpg")))[:3]
non_paths = sorted(glob.glob(os.path.join(dataset_dir, "noninfected", "*.jpg")))[:3]

results = []
for label, paths in [("PCOS_Positive", inf_paths), ("Control_Normal", non_paths)]:
    for p in paths:
        fname = os.path.basename(p)
        orig = cv2.imread(p)
        img_bgr, gray, enhanced = preprocess_ultrasound(orig)
        ovary_ellipse, center, hull, mask = segment_ovarian_boundary(gray, enhanced)
        follicles = detect_antral_follicles(gray, enhanced, mask, center)
        metrics = compute_peripheral_dispersion_index(follicles, center, hull)
        
        overlay = render_clinical_overlay(img_bgr, ovary_ellipse, center, hull, metrics)
        out_path = os.path.join("test_outputs", f"{label}_{fname}")
        cv2.imwrite(out_path, overlay)
        
        results.append({
            "group": label,
            "filename": fname,
            "follicles": metrics["follicle_count"],
            "pdi_mean": metrics["pdi_mean"],
            "pdi_median": metrics["pdi_median"],
            "prc_65": metrics["peripheral_ratio_65"]
        })

print("\n===== QUANTITATIVE VALIDATION SAMPLES =====")
for r in results:
    rotterdam = "POSITIVE (PCOM)" if r["follicles"] >= 12 and r["pdi_mean"] >= 0.65 else ("BORDERLINE" if r["follicles"] >= 10 else "NEGATIVE (Normal)")
    print(f"[{r['group']}] {r['filename']} -> FNPO: {r['follicles']:2d} | PDI: {r['pdi_mean']:.3f} | PRC(>=0.65): {r['prc_65']*100:.1f}% | Rotterdam: {rotterdam}")
