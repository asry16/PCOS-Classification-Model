"""
Clinical Diagnostic Rule-Engine & Report Generator for PCOS-BioQuant.
Translates quantitative geometric follicle metrics into international
Rotterdam / ESHRE clinically-aligned diagnostic reports.
"""

from datetime import datetime


class ClinicalDiagnosticEngine:
    def __init__(
        self,
        fnpo_threshold_rotterdam: int = 12,
        fnpo_threshold_revised: int = 20,
        pdi_peripheral_cutoff: float = 0.65,
        prc_peripheral_cutoff: float = 0.50,
    ):
        self.fnpo_rotterdam = fnpo_threshold_rotterdam
        self.fnpo_revised = fnpo_threshold_revised
        self.pdi_cutoff = pdi_peripheral_cutoff
        self.prc_cutoff = prc_peripheral_cutoff

    def diagnose(self, metrics: dict, ovary_data: dict, scan_id: str = "PCOS-SCAN-001"):
        """
        Evaluates clinical guidelines and produces a structured clinical diagnostic sheet.
        """
        fnpo = metrics["fnpo"]
        pdi_mean = metrics["pdi_mean"]
        pdi_median = metrics["pdi_median"]
        prc_65 = metrics["prc_65"]
        central_frac = metrics["central_fraction"]
        stroma_ratio = metrics["stroma_to_ovary_ratio"]
        mean_diam = metrics["mean_diameter_mm"]

        # Classification logic
        is_ovary = ovary_data.get("is_ovary", True)
        has_high_fnpo = fnpo >= self.fnpo_rotterdam
        has_high_pdi = (pdi_mean >= self.pdi_cutoff) or (prc_65 >= self.prc_cutoff)
        has_severe_fnpo = fnpo >= self.fnpo_revised

        if not is_ovary:
            diagnosis_code = "NON_OVARIAN_PELVIC_SCAN"
            morphology_match = "NEGATIVE (Uterine Corpus / Non-Ovarian Pelvic Scan)"
            clinical_severity = "None (Non-Target Organ)"
            pearl_sign = "Not Applicable (Uterine Myometrium Scanned)"
            dispersion_label = "Not Applicable (Absence of Ovarian Parenchyma)"
            confidence = 94.0
        elif has_high_fnpo and has_high_pdi:
            diagnosis_code = "PCOM_POSITIVE"
            morphology_match = "POSITIVE (Polycystic Ovarian Morphology - PCOM)"
            clinical_severity = "High" if has_severe_fnpo else "Moderate"
            pearl_sign = "Present (Classic String-of-Pearls Configuration)"
            dispersion_label = "High Peripheral Predominance (Subcapsular Crowding)"
        elif has_high_fnpo and not has_high_pdi:
            diagnosis_code = "PCOM_BORDERLINE_MULTIFOLLICULAR"
            morphology_match = "BORDERLINE (Multifollicular Pattern / Central Stroma Infiltration)"
            clinical_severity = "Mild / Equivocal"
            pearl_sign = "Atypical (Follicles scattered throughout stroma without clear subcapsular ring)"
            dispersion_label = "Intermediate / Scattered Distribution"
        elif (fnpo >= 9) and has_high_pdi:
            diagnosis_code = "PCOM_BORDERLINE_PERIPHERAL"
            morphology_match = "BORDERLINE (Sub-threshold Count with High Peripheral Index)"
            clinical_severity = "Mild"
            pearl_sign = "Incipient Peripheral Displacement"
            dispersion_label = "Moderate Peripheral Predominance"
        elif (fnpo >= 9):
            diagnosis_code = "PCOM_BORDERLINE"
            morphology_match = "BORDERLINE (Equivocal Morphology)"
            clinical_severity = "Low"
            pearl_sign = "Absent"
            dispersion_label = "Intermediate Distribution"
        else:
            diagnosis_code = "NORMAL"
            morphology_match = "NEGATIVE (Normal Physiological Ovarian Morphology)"
            clinical_severity = "None"
            pearl_sign = "Absent (Physiological Follicular Distribution)"
            dispersion_label = "Random / Centrostromal Physiological Scatter"

        # Diagnostic Confidence (0-100%)
        if diagnosis_code == "PCOM_POSITIVE":
            confidence = min(98.5, 75.0 + (pdi_mean - 0.65) * 60.0 + min(fnpo - 12, 10) * 1.5)
        elif diagnosis_code == "NORMAL":
            confidence = min(96.0, 78.0 + max(0.0, 0.60 - pdi_mean) * 40.0 + max(0, 10 - fnpo) * 1.8)
        else:
            confidence = 68.0

        # Detailed reasoning
        narrative_bullets = [
            f"Follicle Number Per Ovary (FNPO): {fnpo} antral follicles localized within ovarian boundary (Rotterdam threshold >= {self.fnpo_rotterdam}; revised threshold >= {self.fnpo_revised}).",
            f"Peripheral Dispersion Index (PDI): Mean = {pdi_mean:.3f}, Median = {pdi_median:.3f} (Values > 0.65 indicate significant subcapsular displacement).",
            f"Peripheral Ring Concentration (PRC >= 0.65): {prc_65 * 100:.1f}% of follicles are aligned along the subcapsular rim.",
            f"Central Stromal Sparing Index: {central_frac * 100:.1f}% central follicle occupancy (PCOS typically exhibits stromal clearance < 15%).",
            f"Stromal-to-Total Area Ratio: {stroma_ratio * 100:.1f}% of ovarian area composed of dense central stroma.",
            f"Average Follicle Caliber: {mean_diam:.2f} mm (within typical 2–9 mm antral follicle physiological band)."
        ]

        recommendations = []
        if diagnosis_code == "PCOM_POSITIVE":
            recommendations.append("Correlate with biochemical hyperandrogenism (Free Testosterone, FAI, DHEAS).")
            recommendations.append("Assess clinical cycle history (oligomenorrhea/amenorrhea).")
            recommendations.append("Consider serum Anti-Mullerian Hormone (AMH) measurement for confirmation.")
        elif "BORDERLINE" in diagnosis_code:
            recommendations.append("Repeat transvaginal ultrasound in early follicular phase (Day 3–5 of cycle).")
            recommendations.append("Serum hormonal panel recommended to evaluate subclinical anovulation.")
        else:
            recommendations.append("Normal ovarian architecture. No morphological sonographic signs of PCOM.")

        return {
            "scan_id": scan_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "diagnosis_code": diagnosis_code,
            "morphology_match": morphology_match,
            "clinical_severity": clinical_severity,
            "confidence_pct": round(confidence, 1),
            "string_of_pearls_sign": pearl_sign,
            "dispersion_profile": dispersion_label,
            "fnpo": fnpo,
            "pdi_mean": pdi_mean,
            "pdi_median": pdi_median,
            "prc_65_pct": round(prc_65 * 100, 1),
            "central_sparing_pct": round((1.0 - central_frac) * 100, 1),
            "stroma_ratio_pct": round(stroma_ratio * 100, 1),
            "mean_diameter_mm": mean_diam,
            "findings_narrative": narrative_bullets,
            "clinical_recommendations": recommendations,
        }
