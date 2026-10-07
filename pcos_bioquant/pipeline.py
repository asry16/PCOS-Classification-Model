"""
End-to-End PCOS-BioQuant Analysis Pipeline.
Integrates preprocessing, ovary capsule segmentation, antral follicle detection,
PDI geometric metric computation, Rotterdam clinical diagnostic reasoning,
and publication-grade visualization.
"""

import os
import cv2
import numpy as np

from .preprocessor import UltrasoundPreprocessor
from .ovary_segmenter import OvarySegmenter
from .follicle_detector import AntralFollicleDetector
from .geometric_metrics import GeometricMetricsCalculator
from .clinical_engine import ClinicalDiagnosticEngine
from .visualizer import ClinicalVisualizer


class PCOSBioQuantPipeline:
    def __init__(
        self,
        target_dim: int = 384,
        pixel_per_mm: float = 7.5,
        fnpo_rotterdam_threshold: int = 12,
        fnpo_revised_threshold: int = 20,
        pdi_peripheral_cutoff: float = 0.65,
        prc_peripheral_cutoff: float = 0.50,
    ):
        self.preprocessor = UltrasoundPreprocessor(target_dim=target_dim)
        self.ovary_segmenter = OvarySegmenter(stroma_core_radius_ratio=0.35)
        self.follicle_detector = AntralFollicleDetector(pixel_per_mm=pixel_per_mm)
        self.metrics_calculator = GeometricMetricsCalculator(
            peripheral_cutoff=pdi_peripheral_cutoff,
            central_cutoff=0.40
        )
        self.clinical_engine = ClinicalDiagnosticEngine(
            fnpo_threshold_rotterdam=fnpo_rotterdam_threshold,
            fnpo_threshold_revised=fnpo_revised_threshold,
            pdi_peripheral_cutoff=pdi_peripheral_cutoff,
            prc_peripheral_cutoff=prc_peripheral_cutoff,
        )
        self.visualizer = ClinicalVisualizer()

    def analyze(self, image_input, scan_id: str = "PCOS-SCAN-001", generate_figures: bool = False, save_fig_path: str = None):
        """
        Executes end-to-end analysis on an image path (str) or a BGR numpy array.
        """
        # Load image
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Ultrasound image not found at: {image_input}")
            img_bgr = cv2.imread(image_input)
            if img_bgr is None:
                raise ValueError(f"Could not read image file: {image_input}")
            if scan_id == "PCOS-SCAN-001":
                scan_id = os.path.splitext(os.path.basename(image_input))[0]
        elif isinstance(image_input, np.ndarray):
            img_bgr = image_input.copy()
        else:
            raise TypeError("image_input must be a file path string or numpy ndarray")

        # Step 1: Preprocess ultrasound frame
        prep_data = self.preprocessor.process(img_bgr)
        canonical_bgr = prep_data["canonical_bgr"]
        canonical_gray = prep_data["canonical_gray"]
        enhanced = prep_data["enhanced"]

        # Step 2: Segment ovarian capsule boundary & centroid
        ovary_data = self.ovary_segmenter.segment(
            canonical_gray, enhanced, orig_shape=prep_data.get("orig_shape")
        )

        # Step 3: Localize antral follicles (only if confirmed ovarian tissue)
        if not ovary_data.get("is_ovary", True):
            follicles = []
        else:
            follicles = self.follicle_detector.detect(
                canonical_gray, enhanced, ovary_data["mask"], ovary_data["centroid"]
            )

        # Step 4: Compute Peripheral Dispersion Index (PDI) & Biomarkers
        metrics = self.metrics_calculator.compute(follicles, ovary_data)

        # Step 5: Clinical Diagnostic Reasoning (Rotterdam/ESHRE Criteria)
        report = self.clinical_engine.diagnose(metrics, ovary_data, scan_id=scan_id)

        # Step 6: Render Visualizations
        overlay = self.visualizer.render_overlay(canonical_bgr, ovary_data, metrics, report)

        paper_figure_saved = None
        if generate_figures and save_fig_path:
            self.visualizer.render_paper_figure(
                canonical_bgr, enhanced, ovary_data, metrics, report, save_fig_path
            )
            paper_figure_saved = save_fig_path

        # JSON-serializable clinical record
        clinical_record = {
            "scan_id": report["scan_id"],
            "timestamp": report["timestamp"],
            "diagnosis": {
                "code": report["diagnosis_code"],
                "morphology_match": report["morphology_match"],
                "clinical_severity": report["clinical_severity"],
                "confidence_pct": report["confidence_pct"],
                "string_of_pearls_sign": report["string_of_pearls_sign"],
                "dispersion_profile": report["dispersion_profile"],
            },
            "biomarkers": {
                "fnpo": metrics["fnpo"],
                "pdi_mean": metrics["pdi_mean"],
                "pdi_median": metrics["pdi_median"],
                "pdi_std": metrics["pdi_std"],
                "pdi_iqr": metrics["pdi_iqr"],
                "prc_65_pct": report["prc_65_pct"],
                "central_sparing_pct": report["central_sparing_pct"],
                "stroma_to_ovary_ratio": metrics["stroma_to_ovary_ratio"],
                "mean_diameter_mm": metrics["mean_diameter_mm"],
                "median_diameter_mm": metrics["median_diameter_mm"],
            },
            "ovary_morphology": {
                "area_px": ovary_data["area_px"],
                "major_axis_px": ovary_data["major_axis_px"],
                "minor_axis_px": ovary_data["minor_axis_px"],
                "eccentricity": ovary_data["eccentricity"],
                "stroma_echogenicity": ovary_data["stroma_echogenicity"],
            },
            "follicles_count": len(metrics["follicles_enriched"]),
            "findings_narrative": report["findings_narrative"],
            "clinical_recommendations": report["clinical_recommendations"],
        }

        return {
            "record": clinical_record,
            "metrics": metrics,
            "report": report,
            "ovary_data": ovary_data,
            "prep_data": prep_data,
            "overlay_bgr": overlay,
            "paper_figure_path": paper_figure_saved,
        }
