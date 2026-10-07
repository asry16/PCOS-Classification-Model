import os
import glob
import math
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def evaluate_follicles_and_pdi(img_path, save_plot_path=None):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # 1. Preprocessing: Handle non-cropped vs cropped
    # For full ultrasound screens with black backgrounds, find the active scan region
    _, thresh_active = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    active_pts = np.argwhere(thresh_active > 0)
    
    if len(active_pts) > 0:
        min_y, min_x = active_pts.min(axis=0)
        max_y, max_x = active_pts.max(axis=0)
        # If there are prominent black margins (>15% of frame)
        if (max_x - min_x < 0.88 * w) or (max_y - min_y < 0.88 * h):
            crop_pad = 5
            c_y1 = max(0, min_y - crop_pad)
            c_y2 = min(h, max_y + crop_pad)
            c_x1 = max(0, min_x - crop_pad)
            c_x2 = min(w, max_x + crop_pad)
            gray = gray[c_y1:c_y2, c_x1:c_x2]
            img = img[c_y1:c_y2, c_x1:c_x2]
            h, w = gray.shape
            
    # Bilateral smoothing for speckle suppression
    denoised = cv2.bilateralFilter(gray, d=5, sigmaColor=30, sigmaSpace=30)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    
    # 2. Ovary boundary & centroid estimation
    # The ovary occupies the central portion of the ultrasound frame
    # We use adaptive thresholding and morphological closing
    blurred = cv2.GaussianBlur(denoised, (15, 15), 0)
    _, ovary_bin = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Fill small holes
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(11, min(h, w)//15), max(11, min(h, w)//15)))
    ovary_closed = cv2.morphologyEx(ovary_bin, cv2.MORPH_CLOSE, k_close)
    
    contours, _ = cv2.findContours(ovary_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        best_c = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(best_c)
        M = cv2.moments(hull)
        if M["m00"] > 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        else:
            cx, cy = w / 2.0, h / 2.0
    else:
        cx, cy = w / 2.0, h / 2.0
        # Default ellipse
        hull = cv2.ellipse2Poly((int(cx), int(cy)), (int(w * 0.42), int(h * 0.42)), 0, 0, 360, 5)
        
    ovary_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(ovary_mask, [hull], -1, 255, -1)
    
    # 3. Multi-scale Laplacian of Gaussian (LoG) for dark acoustic follicles
    # Dark circular blobs on bright background correspond to POSITIVE response of normalized LoG:
    # LoG = sigma^2 * (d2I/dx2 + d2I/dy2)
    # Scales: sigma from 2.5 to 14.0 (radii approx 3.5px to 20px)
    sigmas = np.linspace(2.5, 12.0, 10)
    log_responses = []
    
    for sigma in sigmas:
        # Gaussian smoothing
        # Using cv2.Laplacian on Gaussian blurred image
        ksize = int(6 * sigma + 1)
        if ksize % 2 == 0:
            ksize += 1
        blurred_s = cv2.GaussianBlur(enhanced.astype(np.float32), (ksize, ksize), sigma)
        # Standard Laplacian: positive for local minima (dark regions)
        lap = cv2.Laplacian(blurred_s, cv2.CV_32F, ksize=3)
        # Normalized LoG response: sigma^2 * Laplacian
        norm_log = (sigma ** 2) * lap
        log_responses.append(norm_log)
        
    log_volume = np.stack(log_responses, axis=-1)
    
    # Find local 3D maxima in (x, y, scale)
    candidates = []
    threshold = 12.0 # Minimum LoG response for fluid cyst contrast
    
    for scale_idx, sigma in enumerate(sigmas):
        resp = log_volume[:, :, scale_idx]
        # Restrict to inside ovary mask
        resp_ovary = np.where(ovary_mask > 0, resp, 0)
        
        # Regional maxima via maximum filter
        from scipy.ndimage import maximum_filter
        local_max = maximum_filter(resp_ovary, size=int(2 * sigma + 1))
        peaks = (resp_ovary == local_max) & (resp_ovary > threshold)
        
        peak_y, peak_x = np.nonzero(peaks)
        for py, px in zip(peak_y, peak_x):
            val = resp_ovary[py, px]
            radius = math.sqrt(2.0) * sigma
            candidates.append({
                "x": float(px),
                "y": float(py),
                "radius": float(radius),
                "sigma": float(sigma),
                "response": float(val)
            })
            
    # Sort candidates by response strength
    candidates.sort(key=lambda c: -c["response"])
    
    # 4. Rigorous Follicle Validation: Non-maximum suppression + Stroma Contrast Check
    validated_follicles = []
    for cand in candidates:
        cx_f, cy_f = cand["x"], cand["y"]
        r = cand["radius"]
        
        # Suppression check
        overlap = False
        for accepted in validated_follicles:
            dist = math.hypot(cx_f - accepted["x"], cy_f - accepted["y"])
            if dist < 0.7 * (r + accepted["radius"]):
                overlap = True
                break
        if overlap:
            continue
            
        # Stroma contrast test:
        # Follicle core (inner disk of radius 0.6 * r)
        # Stroma rim (annulus between 1.1 * r and 1.8 * r)
        y_grid, x_grid = np.ogrid[:h, :w]
        dist_from_c = np.hypot(x_grid - cx_f, y_grid - cy_f)
        
        inner_mask = (dist_from_c <= 0.6 * r) & (ovary_mask > 0)
        rim_mask = (dist_from_c >= 1.1 * r) & (dist_from_c <= 1.8 * r) & (ovary_mask > 0)
        
        if np.sum(inner_mask) < 4 or np.sum(rim_mask) < 8:
            continue
            
        inner_mean = np.mean(gray[inner_mask])
        rim_mean = np.mean(gray[rim_mask])
        
        # Hypoechoic contrast criterion: rim must be brighter than core
        contrast = (rim_mean - inner_mean) / max(1.0, rim_mean)
        
        # True antral follicles have contrast > 0.12 (anechoic fluid with echogenic rim)
        # and core intensity must not be extremely bright
        if contrast >= 0.10 and inner_mean < 140:
            cand["contrast"] = float(contrast)
            cand["inner_mean"] = float(inner_mean)
            cand["rim_mean"] = float(rim_mean)
            validated_follicles.append(cand)
            
    # 5. Compute PDI for each validated follicle
    hull_pts = hull.reshape(-1, 2).astype(np.float32)
    pdi_values = []
    
    for f in validated_follicles:
        fx, fy = f["x"], f["y"]
        d_center = math.hypot(fx - cx, fy - cy)
        theta = math.atan2(fy - cy, fx - cx)
        
        # Ray casting to find boundary intersection
        ray_dx, ray_dy = math.cos(theta), math.sin(theta)
        max_dist = math.hypot(w, h)
        p1 = (cx, cy)
        p2 = (cx + ray_dx * max_dist, cy + ray_dy * max_dist)
        
        closest_b = max_dist
        bx, by = cx + ray_dx * 50, cy + ray_dy * 50
        
        for i in range(len(hull_pts)):
            q1 = hull_pts[i]
            q2 = hull_pts[(i + 1) % len(hull_pts)]
            denom = (p2[0] - p1[0]) * (q2[1] - q1[1]) - (p2[1] - p1[1]) * (q2[0] - q1[0])
            if abs(denom) > 1e-6:
                t = ((q1[0] - p1[0]) * (q2[1] - q1[1]) - (q1[1] - p1[1]) * (q2[0] - q1[0])) / denom
                u = ((q1[0] - p1[0]) * (p2[1] - p1[1]) - (q1[1] - p1[1]) * (p2[0] - p1[0])) / denom
                if t > 0 and 0.0 <= u <= 1.0:
                    inter_dist = math.hypot(p1[0] + t * (p2[0] - p1[0]) - cx, p1[1] + t * (p2[1] - p1[1]) - cy)
                    if inter_dist < closest_b:
                        closest_b = inter_dist
                        bx = p1[0] + t * (p2[0] - p1[0])
                        by = p1[1] + t * (p2[1] - p1[1])
                        
        rho = np.clip(d_center / max(1.0, closest_b), 0.05, 0.98)
        f["pdi"] = float(rho)
        f["center_dist"] = float(d_center)
        f["boundary_dist"] = float(closest_b)
        f["bx"] = float(bx)
        f["by"] = float(by)
        pdi_values.append(rho)
        
    pdi_mean = float(np.mean(pdi_values)) if pdi_values else 0.0
    pdi_median = float(np.median(pdi_values)) if pdi_values else 0.0
    prc_65 = float(np.mean(np.array(pdi_values) >= 0.65)) if pdi_values else 0.0
    fnpo = len(validated_follicles)
    
    # 6. Save visualization if requested
    if save_plot_path:
        fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # Ovary contour
        hull_plot = np.vstack([hull_pts, hull_pts[0]])
        ax.plot(hull_plot[:, 0], hull_plot[:, 1], color='#00e5ff', linewidth=2.0, label='Ovary Capsule Boundary')
        
        # Ovary center
        ax.scatter([cx], [cy], color='#ffea00', s=80, marker='+', linewidths=2.5, label=f'Ovarian Centroid ({cx:.0f},{cy:.0f})')
        
        # Follicles and vectors
        for idx, f in enumerate(validated_follicles):
            fx, fy, r, rho = f["x"], f["y"], f["radius"], f["pdi"]
            # Color: Green if peripheral (PCOS string-of-pearls), Amber if central
            color = '#00e676' if rho >= 0.65 else '#ff9100'
            
            circle = plt.Circle((fx, fy), r, color=color, fill=False, linewidth=1.8)
            ax.add_patch(circle)
            
            # Center to follicle line
            ax.plot([cx, fx], [cy, fy], color='#ffffff', linestyle=':', linewidth=0.8, alpha=0.7)
            # Follicle to boundary ray
            ax.plot([fx, f["bx"]], [fy, f["by"]], color='#ff1744', linestyle='--', linewidth=0.9, alpha=0.8)
            
            # PDI text
            ax.text(fx + r + 2, fy, f"{rho:.2f}", color='#ffffff', fontsize=7, 
                    bbox=dict(boxstyle="square,pad=0.1", fc="black", ec="none", alpha=0.6))
            
        ax.set_title(f"FNPO: {fnpo} Follicles | Mean PDI: {pdi_mean:.3f} | PRC (>=0.65): {prc_65*100:.1f}%\nRotterdam: {'POSITIVE (PCOM)' if fnpo >= 12 and pdi_mean >= 0.65 else ('BORDERLINE' if fnpo >= 10 else 'NEGATIVE (Normal)')}", fontsize=10, pad=10)
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(save_plot_path, bbox_inches='tight')
        plt.close()
        
    return {
        "fnpo": fnpo,
        "pdi_mean": round(pdi_mean, 3),
        "pdi_median": round(pdi_median, 3),
        "prc_65": round(prc_65, 3),
        "follicles": validated_follicles
    }

# Test on 5 infected and 5 noninfected
dataset_dir = r"C:\Users\ritur\.cache\kagglehub\datasets\ibadeus\pcos-xai-ultrasound-dataset\versions\1\PCOS"
inf_files = sorted(glob.glob(os.path.join(dataset_dir, "infected", "*.jpg")))[:5]
non_files = sorted(glob.glob(os.path.join(dataset_dir, "noninfected", "*.jpg")))[:5]

os.makedirs("test_outputs_log", exist_ok=True)

print("\n--- TESTING MULTI-SCALE LOG FOLLICLE DETECTOR & PDI ---")
for p in inf_files:
    fname = os.path.basename(p)
    out_img = os.path.join("test_outputs_log", f"PCOS_{fname}.png")
    res = evaluate_follicles_and_pdi(p, out_img)
    print(f"[PCOS] {fname:15s} -> FNPO: {res['fnpo']:2d} | Mean PDI: {res['pdi_mean']:.3f} | Median PDI: {res['pdi_median']:.3f} | PRC(>=0.65): {res['prc_65']*100:.1f}%")

for p in non_files:
    fname = os.path.basename(p)
    out_img = os.path.join("test_outputs_log", f"NORMAL_{fname}.png")
    res = evaluate_follicles_and_pdi(p, out_img)
    print(f"[NORMAL] {fname:15s} -> FNPO: {res['fnpo']:2d} | Mean PDI: {res['pdi_mean']:.3f} | Median PDI: {res['pdi_median']:.3f} | PRC(>=0.65): {res['prc_65']*100:.1f}%")
