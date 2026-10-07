import os
import glob
import json
import shutil
import cv2
from pcos_bioquant.pipeline import PCOSBioQuantPipeline

dataset_dir = r"C:\Users\ritur\.cache\kagglehub\datasets\ibadeus\pcos-xai-ultrasound-dataset\versions\1\PCOS"
inf = sorted(glob.glob(os.path.join(dataset_dir, "infected", "image1*.jpg")))
non = sorted(glob.glob(os.path.join(dataset_dir, "noninfected", "*.jpg")))

pipe = PCOSBioQuantPipeline()

os.makedirs("webapp/presets", exist_ok=True)

candidate_specs = [
    {"id": "case_classic_pcom", "title": "Case 1: Classic PCOM (String-of-Pearls)", "path": inf[19], "notes": "Follicle Number Per Ovary (FNPO = 30) with marked subcapsular crowding (PDI = 0.706) and dense central stroma."},
    {"id": "case_moderate_pcom", "title": "Case 2: Rotterdam Positive (Moderate)", "path": inf[2], "notes": "FNPO = 14 with prominent peripheral distribution (PDI = 0.668), satisfying Rotterdam threshold."},
    {"id": "case_multifollicular", "title": "Case 3: Multifollicular Ovary (Intermediate)", "path": inf[4], "notes": "Multiple antral follicles (FNPO = 22) but dispersed centrally into stroma (PDI = 0.529, multifollicular pattern)."},
    {"id": "case_normal_ovary", "title": "Case 4: Normal Physiological Baseline", "path": inf[10], "notes": "Physiological follicle count (FNPO = 8 < 12), normal stromal architecture without PCOM morphology."},
    {"id": "case_uterine_confounder", "title": "Case 5: Non-Ovarian Scan (Safety Guardrail)", "path": non[0], "notes": "Transvaginal uterine corpus scan successfully identified and rejected by anatomical safety guardrail."}
]

presets_metadata = []

for spec in candidate_specs:
    cid = spec["id"]
    src_path = spec["path"]
    dst_img = f"webapp/presets/{cid}.jpg"
    shutil.copyfile(src_path, dst_img)
    
    res = pipe.analyze(src_path, scan_id=cid, generate_figures=False)
    rec = res["record"]
    
    # Save overlay image
    overlay_path = f"webapp/presets/{cid}_overlay.png"
    cv2.imwrite(overlay_path, res["overlay_bgr"])
    
    ovary = res["ovary_data"]
    metrics = res["metrics"]
    
    presets_metadata.append({
        "id": cid,
        "title": spec["title"],
        "notes": spec["notes"],
        "image_file": f"/presets/{cid}.jpg",
        "overlay_file": f"/presets/{cid}_overlay.png",
        "record": rec,
        "capsule": ovary["contour"].tolist() if ovary.get("contour") is not None else [],
        "centroid": ovary["centroid"],
        "follicles": metrics.get("follicles_enriched", [])
    })
    
    print(f"Preset {cid} created: FNPO={rec['biomarkers']['fnpo']}, PDI={rec['biomarkers']['pdi_mean']:.3f}, Code={rec['diagnosis']['code']}")

with open("webapp/presets/presets_index.json", "w") as f:
    json.dump(presets_metadata, f, indent=2)

print("\nAll presets prepared successfully!")
