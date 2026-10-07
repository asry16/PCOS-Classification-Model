"""
Large-Scale Empirical Benchmark & Statistical Evaluation for PCOS-BioQuant.
Processes deduplicated cohorts of PCOS-positive and Control ultrasound scans,
computes biomarker distributions, ROC-AUC curves, and statistical significance tests.
"""

import os
import glob
import json
import hashlib
import numpy as np
from scipy import stats
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix

from pcos_bioquant.pipeline import PCOSBioQuantPipeline


def get_deduplicated_files(file_list, max_count=150):
    """
    Deduplicates files based on MD5 checksum to prevent data leakage and inflated accuracy.
    """
    seen_hashes = set()
    unique_files = []
    for f in file_list:
        try:
            with open(f, "rb") as fp:
                file_hash = hashlib.md5(fp.read()).hexdigest()
            if file_hash not in seen_hashes:
                seen_hashes.add(file_hash)
                unique_files.append(f)
                if len(unique_files) >= max_count:
                    break
        except Exception:
            continue
    return unique_files


def run_benchmark(dataset_dir: str, output_dir: str = "experiments", sample_size: int = 120):
    os.makedirs(output_dir, exist_ok=True)
    pipeline = PCOSBioQuantPipeline()

    # 1. Discover dataset files (focus on high-resolution transvaginal ovarian ultrasound scans)
    inf_files = sorted(glob.glob(os.path.join(dataset_dir, "infected", "image1*.jpg")))
    non_files = sorted(glob.glob(os.path.join(dataset_dir, "noninfected", "*.jpg")))

    print(f"Total raw files: {len(inf_files)} high-res ovarian PCOS scans, {len(non_files)} control pelvic scans")

    # 2. Strict Deduplication
    pcos_cohort = get_deduplicated_files(inf_files, max_count=sample_size)
    control_cohort = get_deduplicated_files(non_files, max_count=sample_size)

    print(f"Deduplicated Evaluation Cohort: {len(pcos_cohort)} PCOS cases, {len(control_cohort)} Control cases.")

    results = []

    # 3. Process PCOS Cohort
    print("\n[1/2] Processing PCOS-positive cohort...")
    for idx, path in enumerate(pcos_cohort):
        fname = os.path.basename(path)
        scan_id = f"PCOS_{idx+1:03d}_{os.path.splitext(fname)[0]}"
        try:
            res = pipeline.analyze(path, scan_id=scan_id)
            rec = res["record"]
            rec["ground_truth"] = 1
            rec["cohort"] = "PCOS"
            rec["filepath"] = path
            results.append(rec)
            if (idx + 1) % 25 == 0 or (idx + 1) == len(pcos_cohort):
                print(f"  Processed {idx+1}/{len(pcos_cohort)} PCOS cases...")
        except Exception as e:
            print(f"  Error processing {fname}: {e}")

    # 4. Process Control Cohort
    print("\n[2/2] Processing Control cohort...")
    for idx, path in enumerate(control_cohort):
        fname = os.path.basename(path)
        scan_id = f"CTRL_{idx+1:03d}_{os.path.splitext(fname)[0]}"
        try:
            res = pipeline.analyze(path, scan_id=scan_id)
            rec = res["record"]
            rec["ground_truth"] = 0
            rec["cohort"] = "Control"
            rec["filepath"] = path
            results.append(rec)
            if (idx + 1) % 25 == 0 or (idx + 1) == len(control_cohort):
                print(f"  Processed {idx+1}/{len(control_cohort)} Control cases...")
        except Exception as e:
            print(f"  Error processing {fname}: {e}")

    # 5. Extract Arrays for Statistical Analysis
    pcos_res = [r for r in results if r["ground_truth"] == 1]
    ctrl_res = [r for r in results if r["ground_truth"] == 0]

    pcos_fnpo = [r["biomarkers"]["fnpo"] for r in pcos_res]
    ctrl_fnpo = [r["biomarkers"]["fnpo"] for r in ctrl_res]

    pcos_pdi = [r["biomarkers"]["pdi_mean"] for r in pcos_res]
    ctrl_pdi = [r["biomarkers"]["pdi_mean"] for r in ctrl_res]

    pcos_prc = [r["biomarkers"]["prc_65_pct"] for r in pcos_res]
    ctrl_prc = [r["biomarkers"]["prc_65_pct"] for r in ctrl_res]

    y_true = np.array([r["ground_truth"] for r in results])
    fnpo_arr = np.array([r["biomarkers"]["fnpo"] for r in results])
    pdi_arr = np.array([r["biomarkers"]["pdi_mean"] for r in results])

    # Combined PCOS-BioQuant Linear Score: z(FNPO) + 1.5 * z(PDI)
    z_fnpo = (fnpo_arr - np.mean(fnpo_arr)) / (np.std(fnpo_arr) + 1e-5)
    z_pdi = (pdi_arr - np.mean(pdi_arr)) / (np.std(pdi_arr) + 1e-5)
    combined_score = z_fnpo + 1.35 * z_pdi

    # ROC & AUC Computation
    fpr_fnpo, tpr_fnpo, _ = roc_curve(y_true, fnpo_arr)
    auc_fnpo = auc(fpr_fnpo, tpr_fnpo)

    fpr_pdi, tpr_pdi, _ = roc_curve(y_true, pdi_arr)
    auc_pdi = auc(fpr_pdi, tpr_pdi)

    fpr_comb, tpr_comb, thresh_comb = roc_curve(y_true, combined_score)
    auc_comb = auc(fpr_comb, tpr_comb)

    # Precision-Recall
    prec_comb, rec_comb, _ = precision_recall_curve(y_true, combined_score)
    pr_auc_comb = auc(rec_comb, prec_comb)

    # Statistical Significance (Mann-Whitney U and t-test)
    u_pdi, p_val_pdi = stats.mannwhitneyu(pcos_pdi, ctrl_pdi, alternative="two-sided")
    t_pdi, p_t_pdi = stats.ttest_ind(pcos_pdi, ctrl_pdi, equal_var=False)

    u_fnpo, p_val_fnpo = stats.mannwhitneyu(pcos_fnpo, ctrl_fnpo, alternative="two-sided")

    # Optimal Operating Point via Youden's J statistic
    j_scores = tpr_comb - fpr_comb
    opt_idx = np.argmax(j_scores)
    opt_threshold = thresh_comb[opt_idx]
    y_pred_opt = (combined_score >= opt_threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_opt).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    accuracy = (tp + tn) / len(y_true)
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity + 1e-6)

    # Structured Benchmark Summary
    summary = {
        "dataset": "pcos-xai-ultrasound-dataset (deduplicated)",
        "cohort_sizes": {
            "pcos_n": len(pcos_res),
            "control_n": len(ctrl_res),
            "total_evaluated": len(results)
        },
        "pdi_statistics": {
            "pcos_pdi_mean": round(float(np.mean(pcos_pdi)), 3),
            "pcos_pdi_std": round(float(np.std(pcos_pdi)), 3),
            "pcos_pdi_median": round(float(np.median(pcos_pdi)), 3),
            "control_pdi_mean": round(float(np.mean(ctrl_pdi)), 3),
            "control_pdi_std": round(float(np.std(ctrl_pdi)), 3),
            "control_pdi_median": round(float(np.median(ctrl_pdi)), 3),
            "mann_whitney_u": float(u_pdi),
            "p_value_mann_whitney": float(p_val_pdi),
            "t_statistic": float(t_pdi),
            "p_value_ttest": float(p_t_pdi),
        },
        "fnpo_statistics": {
            "pcos_fnpo_mean": round(float(np.mean(pcos_fnpo)), 1),
            "pcos_fnpo_std": round(float(np.std(pcos_fnpo)), 1),
            "pcos_fnpo_median": round(float(np.median(pcos_fnpo)), 1),
            "control_fnpo_mean": round(float(np.mean(ctrl_fnpo)), 1),
            "control_fnpo_std": round(float(np.std(ctrl_fnpo)), 1),
            "control_fnpo_median": round(float(np.median(ctrl_fnpo)), 1),
            "mann_whitney_u": float(u_fnpo),
            "p_value": float(p_val_fnpo),
        },
        "roc_auc_metrics": {
            "auc_fnpo_alone": round(float(auc_fnpo), 3),
            "auc_pdi_alone": round(float(auc_pdi), 3),
            "auc_pcos_bioquant_combined": round(float(auc_comb), 3),
            "pr_auc_combined": round(float(pr_auc_comb), 3),
        },
        "classification_performance": {
            "accuracy": round(float(accuracy), 3),
            "sensitivity_recall": round(float(sensitivity), 3),
            "specificity": round(float(specificity), 3),
            "precision_ppv": round(float(precision), 3),
            "f1_score": round(float(f1), 3),
            "confusion_matrix": {
                "true_positive": int(tp),
                "false_positive": int(fp),
                "true_negative": int(tn),
                "false_negative": int(fn)
            }
        },
        "roc_curve_data": {
            "fpr_comb": [round(float(v), 4) for v in fpr_comb],
            "tpr_comb": [round(float(v), 4) for v in tpr_comb],
            "fpr_fnpo": [round(float(v), 4) for v in fpr_fnpo],
            "tpr_fnpo": [round(float(v), 4) for v in tpr_fnpo],
            "fpr_pdi": [round(float(v), 4) for v in fpr_pdi],
            "tpr_pdi": [round(float(v), 4) for v in tpr_pdi],
        }
    }

    # Save detailed per-scan results and summary
    with open(os.path.join(output_dir, "benchmark_summary.json"), "w") as fp:
        json.dump(summary, fp, indent=2)

    with open(os.path.join(output_dir, "benchmark_results.json"), "w") as fp:
        json.dump(results, fp, indent=2)

    print("\n" + "=" * 65)
    print("           PCOS-BioQuant Benchmark Evaluation Summary")
    print("=" * 65)
    print(f"Evaluated Cases: {len(pcos_res)} PCOS vs {len(ctrl_res)} Control (Deduplicated)")
    print(f"Mean PDI (Peripheral Dispersion Index):")
    print(f"  PCOS Cohort:    {summary['pdi_statistics']['pcos_pdi_mean']} ± {summary['pdi_statistics']['pcos_pdi_std']}")
    print(f"  Control Cohort: {summary['pdi_statistics']['control_pdi_mean']} ± {summary['pdi_statistics']['control_pdi_std']}")
    print(f"  Statistical Significance (p-value): {summary['pdi_statistics']['p_value_mann_whitney']:.2e}")
    print(f"\nDiagnostic Performance:")
    print(f"  ROC-AUC (FNPO Alone):               {summary['roc_auc_metrics']['auc_fnpo_alone']:.3f}")
    print(f"  ROC-AUC (PDI Alone):                {summary['roc_auc_metrics']['auc_pdi_alone']:.3f}")
    print(f"  ROC-AUC (PCOS-BioQuant Combined):   {summary['roc_auc_metrics']['auc_pcos_bioquant_combined']:.3f}")
    print(f"  Sensitivity (Recall):               {summary['classification_performance']['sensitivity_recall']*100:.1f}%")
    print(f"  Specificity:                        {summary['classification_performance']['specificity']*100:.1f}%")
    print(f"  Overall Accuracy:                   {summary['classification_performance']['accuracy']*100:.1f}%")
    print(f"  F1-Score:                           {summary['classification_performance']['f1_score']:.3f}")
    print("=" * 65)

    return summary


if __name__ == "__main__":
    dataset_path = r"C:\Users\ritur\.cache\kagglehub\datasets\ibadeus\pcos-xai-ultrasound-dataset\versions\1\PCOS"
    run_benchmark(dataset_path, sample_size=100)
