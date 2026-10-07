"""
Demo script to test the end-to-end PCOS-BioQuant pipeline on sample ultrasound scans.
"""

import os
import glob
import cv2
import json

from pcos_bioquant.pipeline import PCOSBioQuantPipeline

def main():
    print("=" * 65)
    print("  PCOS-BioQuant: Clinically-Aligned Quantitative Assistant Demo")
    print("=" * 65)

    pipeline = PCOSBioQuantPipeline()
    dataset_dir = r"C:\Users\ritur\.cache\kagglehub\datasets\ibadeus\pcos-xai-ultrasound-dataset\versions\1\PCOS"

    # Select representative samples
    pcos_samples = sorted(glob.glob(os.path.join(dataset_dir, "infected", "image1*.jpg")))[:3]
    normal_samples = sorted(glob.glob(os.path.join(dataset_dir, "noninfected", "*.jpg")))[:3]

    os.makedirs("demo_outputs", exist_ok=True)

    for group_name, sample_list in [("PCOS", pcos_samples), ("NORMAL", normal_samples)]:
        print(f"\nEvaluating {group_name} Cohort Samples...")
        for p in sample_list:
            fname = os.path.basename(p)
            scan_id = f"{group_name}_{os.path.splitext(fname)[0]}"
            fig_path = os.path.join("demo_outputs", f"{scan_id}_paper_fig.png")
            overlay_path = os.path.join("demo_outputs", f"{scan_id}_overlay.png")

            res = pipeline.analyze(p, scan_id=scan_id, generate_figures=True, save_fig_path=fig_path)
            cv2.imwrite(overlay_path, res["overlay_bgr"])

            rec = res["record"]
            print(f" -> Scan ID: {scan_id}")
            print(f"    Diagnosis: {rec['diagnosis']['morphology_match']}")
            print(f"    Confidence: {rec['diagnosis']['confidence_pct']}% | Severity: {rec['diagnosis']['clinical_severity']}")
            print(f"    String-of-Pearls: {rec['diagnosis']['string_of_pearls_sign']}")
            print(f"    FNPO: {rec['biomarkers']['fnpo']} follicles | Mean PDI: {rec['biomarkers']['pdi_mean']:.3f} | PRC(>=0.65): {rec['biomarkers']['prc_65_pct']}%")
            print(f"    Central Stroma Clearance: {rec['biomarkers']['central_sparing_pct']}%")

    print("\n" + "=" * 65)
    print("Demo completed successfully! Diagnostic figures saved to 'demo_outputs/'")
    print("=" * 65)

if __name__ == "__main__":
    main()
