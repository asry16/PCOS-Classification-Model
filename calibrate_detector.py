import os
import glob
import math
import cv2
import numpy as np

def calibrate_and_detect(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Standardize image size for consistent spatial scale (canonical size 360x360)
    target_dim = 360
    scale_factor = target_dim / max(h, w)
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)
    resized_gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # 1. Ultrasound Sector / Active ROI Detection
    # Threshold background
    _, bg_mask = cv2.threshold(resized_gray, 20, 255, cv2.THRESH_BINARY)
    # Morphological closing
    k_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    bg_clean = cv2.morphologyEx(bg_mask, cv2.MORPH_CLOSE, k_bg)
    
    # Active tissue contour
    contours, _ = cv2.findContours(bg_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        best_tissue = max(contours, key=cv2.contourArea)
        x_b, y_b, w_b, h_b = cv2.boundingRect(best_tissue)
        # Crop to active tissue region if surrounded by large dark border
        if w_b < 0.85 * new_w or h_b < 0.85 * new_h:
            pad = 5
            x1, y1 = max(0, x_b - pad), max(0, y_b - pad)
            x2, y2 = min(new_w, x_b + w_b + pad), min(new_h, y_b + h_b + pad)
            resized_gray = resized_gray[y1:y2, x1:x2]
            
    # Resize cropped active tissue to canonical 360x360
    proc_img = cv2.resize(resized_gray, (target_dim, target_dim), interpolation=cv2.INTER_AREA)
    
    # 2. Speckle suppression & contrast enhancement
    # Bilateral filter smoothes acoustic speckle while keeping follicle cyst borders sharp
    denoised = cv2.bilateralFilter(proc_img, d=7, sigmaColor=35, sigmaSpace=35)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    
    # 3. Ovarian Boundary Segmentation (Ellipse / Convex Hull)
    # The ovary boundary is defined by the outer margin of ovarian stroma
    blurred_ovary = cv2.GaussianBlur(denoised, (21, 21), 0)
    _, ovary_thresh = cv2.threshold(blurred_ovary, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k_ovary = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    ovary_mask = cv2.morphologyEx(ovary_thresh, cv2.MORPH_CLOSE, k_ovary)
    
    contours_ov, _ = cv2.findContours(ovary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours_ov:
        c_ov = max(contours_ov, key=cv2.contourArea)
        ovary_hull = cv2.convexHull(c_ov)
        M = cv2.moments(ovary_hull)
        if M["m00"] > 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        else:
            cx, cy = target_dim / 2.0, target_dim / 2.0
    else:
        cx, cy = target_dim / 2.0, target_dim / 2.0
        ovary_hull = cv2.ellipse2Poly((int(cx), int(cy)), (int(target_dim * 0.42), int(target_dim * 0.42)), 0, 0, 360, 5)
        
    ovary_binary = np.zeros((target_dim, target_dim), dtype=np.uint8)
    cv2.drawContours(ovary_binary, [ovary_hull], -1, 255, -1)
    
    # 4. Multi-scale Antral Follicle Detection
    # In 360x360 canonical space:
    # 2mm to 9mm antral follicles correspond to radii between ~7 pixels and ~28 pixels.
    # We use multi-scale Black Top-Hat & Regional Adaptive Filtering
    candidates = []
    
    # Scales of radii (radii in pixels: 8, 12, 16, 20, 25)
    radii = [8, 12, 16, 21, 26]
    
    for r in radii:
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        # Top-hat on inverted image or Black-hat on enhanced
        blackhat = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, se)
        
        # Adaptive thresholding on blackhat within ovary
        masked_bh = np.where(ovary_binary > 0, blackhat, 0)
        p75 = np.percentile(masked_bh[ovary_binary > 0], 85) if np.any(ovary_binary > 0) else 25
        thresh_val = max(18, p75)
        
        _, b_bin = cv2.threshold(masked_bh, thresh_val, 255, cv2.THRESH_BINARY)
        
        # Find candidate connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(b_bin)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            bx, by, bw, bh = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            fx, fy = centroids[i]
            
            # Expected area for radius r: pi * r^2
            expected_area = math.pi * (r ** 2)
            if 0.35 * expected_area <= area <= 2.2 * expected_area:
                aspect = float(bw) / max(1, bh)
                if 0.45 <= aspect <= 2.2:
                    candidates.append({
                        "x": float(fx),
                        "y": float(fy),
                        "radius": math.sqrt(area / math.pi),
                        "area": area,
                        "bw": bw,
                        "bh": bh,
                        "strength": float(blackhat[int(fy), int(fx)])
                    })
                    
    # Sort candidates by strength
    candidates.sort(key=lambda c: -c["strength"])
    
    # 5. Strict Follicle Filtering:
    # True antral follicles must:
    # a) Have dark hypoechoic interior (I_core < I_stroma)
    # b) Be surrounded by brighter stroma on at least 3 sides (acoustic rim)
    # c) Non-overlapping with higher-confidence follicles
    validated = []
    y_idx, x_idx = np.ogrid[:target_dim, :target_dim]
    ovary_mean_intensity = float(np.mean(proc_img[ovary_binary > 0])) if np.any(ovary_binary > 0) else 100.0
    
    for cand in candidates:
        fx, fy = cand["x"], cand["y"]
        r = cand["radius"]
        
        # Distance to existing validated follicles
        overlap = False
        for v in validated:
            d = math.hypot(fx - v["x"], fy - v["y"])
            if d < 0.75 * (r + v["radius"]):
                overlap = True
                break
        if overlap:
            continue
            
        # Core & Rim intensity
        dist_mat = np.hypot(x_idx - fx, y_idx - fy)
        core_mask = (dist_mat <= 0.65 * r) & (ovary_binary > 0)
        rim_mask = (dist_mat >= 1.15 * r) & (dist_mat <= 1.85 * r) & (ovary_binary > 0)
        
        if np.sum(core_mask) < 6 or np.sum(rim_mask) < 12:
            continue
            
        core_mean = float(np.mean(proc_img[core_mask]))
        rim_mean = float(np.mean(proc_img[rim_mask]))
        
        # Hypoechoic contrast requirement:
        # Follicle must be darker than surrounding stroma
        contrast = (rim_mean - core_mean) / max(1.0, rim_mean)
        
        # In ultrasound, fluid cysts have core_mean substantially lower than stroma
        if contrast >= 0.12 and core_mean < ovary_mean_intensity * 1.1:
            cand["contrast"] = round(contrast, 3)
            cand["core_mean"] = round(core_mean, 1)
            cand["rim_mean"] = round(rim_mean, 1)
            validated.append(cand)
            
    # 6. Compute Geometric Peripheral Dispersion Index (PDI)
    # PDI = (Dist from Ovary Center) / (Dist to Ovary Boundary along ray)
    hull_pts = ovary_hull.reshape(-1, 2).astype(np.float32)
    pdis = []
    
    for f in validated:
        fx, fy = f["x"], f["y"]
        d_center = math.hypot(fx - cx, fy - cy)
        theta = math.atan2(fy - cy, fx - cx)
        
        # Ray to boundary
        ray_dx, ray_dy = math.cos(theta), math.sin(theta)
        p1 = (cx, cy)
        p2 = (cx + ray_dx * 600, cy + ray_dy * 600)
        
        closest_b = 600.0
        bx, by = cx + ray_dx * 100, cy + ray_dy * 100
        
        for i in range(len(hull_pts)):
            q1 = hull_pts[i]
            q2 = hull_pts[(i + 1) % len(hull_pts)]
            denom = (p2[0] - p1[0]) * (q2[1] - q1[1]) - (p2[1] - p1[1]) * (q2[0] - q1[0])
            if abs(denom) > 1e-6:
                t = ((q1[0] - p1[0]) * (q2[1] - q1[1]) - (q1[1] - p1[1]) * (q2[0] - q1[0])) / denom
                u = ((q1[0] - p1[0]) * (p2[1] - p1[1]) - (q1[1] - p1[1]) * (p2[0] - p1[0])) / denom
                if t > 0 and 0.0 <= u <= 1.0:
                    inter_d = math.hypot(p1[0] + t * (p2[0] - p1[0]) - cx, p1[1] + t * (p2[1] - p1[1]) - cy)
                    if inter_d < closest_b:
                        closest_b = inter_d
                        bx = p1[0] + t * (p2[0] - p1[0])
                        by = p1[1] + t * (p2[1] - p1[1])
                        
        rho = np.clip(d_center / max(1.0, closest_b), 0.05, 0.98)
        f["pdi"] = round(float(rho), 3)
        f["boundary_pt"] = (round(float(bx), 1), round(float(by), 1))
        pdis.append(rho)
        
    fnpo = len(validated)
    pdi_mean = float(np.mean(pdis)) if pdis else 0.0
    pdi_median = float(np.median(pdis)) if pdis else 0.0
    prc_65 = float(np.mean(np.array(pdis) >= 0.65)) if pdis else 0.0
    
    # Rotterdam PCOM criteria classification:
    # 1. FNPO >= 12 (or revised >= 20)
    # 2. PDI >= 0.65 (peripheral "string-of-pearls" displacement)
    if fnpo >= 12 and pdi_mean >= 0.65:
        rotterdam = "POSITIVE (PCOM)"
    elif fnpo >= 10 or (fnpo >= 8 and pdi_mean >= 0.70):
        rotterdam = "BORDERLINE"
    else:
        rotterdam = "NEGATIVE (Normal)"
        
    return {
        "fnpo": fnpo,
        "pdi_mean": round(pdi_mean, 3),
        "pdi_median": round(pdi_median, 3),
        "prc_65": round(prc_65, 3),
        "rotterdam": rotterdam,
        "follicles": validated
    }

# Run on 10 PCOS and 10 Normal images
dataset_dir = r"C:\Users\ritur\.cache\kagglehub\datasets\ibadeus\pcos-xai-ultrasound-dataset\versions\1\PCOS"
pcos_samples = sorted(glob.glob(os.path.join(dataset_dir, "infected", "image1*.jpg")))[:10]
norm_samples = sorted(glob.glob(os.path.join(dataset_dir, "noninfected", "*.jpg")))[:10]

print("\n" + "="*70)
print(f"{'GROUP':<10} | {'FILENAME':<16} | {'FNPO':<5} | {'MEAN PDI':<9} | {'PRC (>=0.65)':<12} | {'ROTTERDAM':<18}")
print("="*70)

pcos_f_list, pcos_pdi_list = [], []
for p in pcos_samples:
    fn = os.path.basename(p)
    res = calibrate_and_detect(p)
    if res:
        pcos_f_list.append(res["fnpo"])
        pcos_pdi_list.append(res["pdi_mean"])
        print(f"{'PCOS':<10} | {fn:<16} | {res['fnpo']:<5d} | {res['pdi_mean']:<9.3f} | {res['prc_65']*100:<11.1f}% | {res['rotterdam']:<18}")

norm_f_list, norm_pdi_list = [], []
for p in norm_samples:
    fn = os.path.basename(p)
    res = calibrate_and_detect(p)
    if res:
        norm_f_list.append(res["fnpo"])
        norm_pdi_list.append(res["pdi_mean"])
        print(f"{'NORMAL':<10} | {fn:<16} | {res['fnpo']:<5d} | {res['pdi_mean']:<9.3f} | {res['prc_65']*100:<11.1f}% | {res['rotterdam']:<18}")

print("="*70)
print(f"PCOS Group Means   -> FNPO: {np.mean(pcos_f_list):.1f} follicles | PDI: {np.mean(pcos_pdi_list):.3f}")
print(f"Normal Group Means -> FNPO: {np.mean(norm_f_list):.1f} follicles | PDI: {np.mean(norm_pdi_list):.3f}")
print("="*70)
