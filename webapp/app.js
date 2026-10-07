/**
 * PCOS-BioQuant — Interactive Frontend Engine & Diagnostic Canvas
 */

// Global State
const state = {
  presets: [],
  currentPresetId: "case_classic_pcom",
  currentAnalysis: null,
  activeTab: "diagnostic-suite",
  viewMode: "overlay", // "overlay" | "raw"
  hoveredFollicleId: null,
  layers: {
    capsule: true,
    stroma: true,
    follicles: true,
    rays: true,
    opacity: 0.85
  },
  images: {
    raw: null,
    overlay: null
  }
};

// DOM References
const elements = {
  tabs: document.querySelectorAll(".nav-tab"),
  panes: document.querySelectorAll(".tab-pane"),
  presetContainer: document.getElementById("preset-chips-container"),
  canvas: document.getElementById("scan-canvas"),
  canvasWrapper: document.getElementById("canvas-wrapper"),
  tooltip: document.getElementById("canvas-tooltip"),
  loader: document.getElementById("loader-overlay"),
  
  // Toggles & Controls
  btnToggleOverlay: document.getElementById("btn-toggle-overlay"),
  btnToggleRaw: document.getElementById("btn-toggle-raw"),
  chkCapsule: document.getElementById("chk-capsule"),
  chkStroma: document.getElementById("chk-stroma"),
  chkFollicles: document.getElementById("chk-follicles"),
  chkRays: document.getElementById("chk-rays"),
  sliderOpacity: document.getElementById("slider-opacity"),
  valOpacity: document.getElementById("val-opacity"),
  
  // Table
  follicleCount: document.getElementById("table-follicle-count"),
  follicleTableBody: document.getElementById("follicle-table-body"),
  
  // Report Sheet
  repScanId: document.getElementById("rep-scan-id"),
  repTimestamp: document.getElementById("rep-timestamp"),
  diagBanner: document.getElementById("diag-banner"),
  diagMatch: document.getElementById("diag-match"),
  diagConf: document.getElementById("diag-conf"),
  diagPearls: document.getElementById("diag-pearls"),
  diagProfile: document.getElementById("diag-profile"),
  diagSeverity: document.getElementById("diag-severity"),
  
  // Biomarkers
  valFnpo: document.getElementById("val-fnpo"),
  barFnpo: document.getElementById("bar-fnpo"),
  valPdi: document.getElementById("val-pdi"),
  barPdi: document.getElementById("bar-pdi"),
  valPrc: document.getElementById("val-prc"),
  barPrc: document.getElementById("bar-prc"),
  valClearance: document.getElementById("val-clearance"),
  barClearance: document.getElementById("bar-clearance"),
  
  // Findings & Recommendations
  findingsList: document.getElementById("findings-list"),
  recommendationsBox: document.getElementById("recommendations-box"),
  
  // File Upload & Export
  fileInput: document.getElementById("scan-file-input"),
  btnPrint: document.getElementById("btn-print-report"),
  btnExportJson: document.getElementById("btn-export-json"),
  btnExportOverlay: document.getElementById("btn-export-overlay")
};

// Canvas 2D Context
const ctx = elements.canvas.getContext("2d");

// ==================== INITIALIZATION ====================
document.addEventListener("DOMContentLoaded", () => {
  setupNavigationTabs();
  setupViewportControls();
  setupCanvasInteractions();
  setupUploadAndExports();
  loadPresets();
});

// ==================== NAVIGATION TABS ====================
function setupNavigationTabs() {
  elements.tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      const targetTab = tab.getAttribute("data-tab");
      state.activeTab = targetTab;

      elements.tabs.forEach(t => t.classList.remove("active"));
      elements.panes.forEach(p => p.classList.remove("active"));

      tab.classList.add("active");
      const activePane = document.getElementById(`pane-${targetTab}`);
      if (activePane) activePane.classList.add("active");
    });
  });
}

// ==================== VIEWPORT CONTROLS ====================
function setupViewportControls() {
  elements.btnToggleOverlay.addEventListener("click", () => {
    state.viewMode = "overlay";
    elements.btnToggleOverlay.classList.add("active");
    elements.btnToggleRaw.classList.remove("active");
    renderCanvas();
  });

  elements.btnToggleRaw.addEventListener("click", () => {
    state.viewMode = "raw";
    elements.btnToggleRaw.classList.add("active");
    elements.btnToggleOverlay.classList.remove("active");
    renderCanvas();
  });

  elements.chkCapsule.addEventListener("change", (e) => {
    state.layers.capsule = e.target.checked;
    renderCanvas();
  });

  elements.chkStroma.addEventListener("change", (e) => {
    state.layers.stroma = e.target.checked;
    renderCanvas();
  });

  elements.chkFollicles.addEventListener("change", (e) => {
    state.layers.follicles = e.target.checked;
    renderCanvas();
  });

  elements.chkRays.addEventListener("change", (e) => {
    state.layers.rays = e.target.checked;
    renderCanvas();
  });

  elements.sliderOpacity.addEventListener("input", (e) => {
    const val = parseInt(e.target.value, 10);
    state.layers.opacity = val / 100.0;
    elements.valOpacity.textContent = `${val}%`;
    renderCanvas();
  });
}

// ==================== PRESET DATA LOADING ====================
async function loadPresets() {
  try {
    const res = await fetch("/api/presets");
    if (!res.ok) throw new Error("Failed to load presets");
    const presets = await res.json();
    state.presets = presets;
    renderPresetChips();
    if (presets.length > 0) {
      selectPreset(presets[0].id);
    }
  } catch (err) {
    console.error("Error loading presets:", err);
  }
}

function renderPresetChips() {
  elements.presetContainer.innerHTML = "";
  state.presets.forEach(p => {
    const chip = document.createElement("button");
    chip.className = `preset-chip ${p.id === state.currentPresetId ? "active" : ""}`;
    chip.setAttribute("data-preset", p.id);

    let indicatorClass = "pcom-pos";
    const code = p.record?.diagnosis?.code;
    if (code === "PCOM_BORDERLINE" || code === "PCOM_BORDERLINE_MULTIFOLLICULAR") indicatorClass = "pcom-bord";
    else if (code === "NORMAL") indicatorClass = "pcom-norm";
    else if (code === "NON_OVARIAN_PELVIC_SCAN") indicatorClass = "pcom-guard";

    chip.innerHTML = `<span class="preset-indicator ${indicatorClass}"></span> ${p.title}`;
    chip.addEventListener("click", () => selectPreset(p.id));
    elements.presetContainer.appendChild(chip);
  });
}

function selectPreset(presetId) {
  state.currentPresetId = presetId;
  document.querySelectorAll(".preset-chip").forEach(c => {
    c.classList.toggle("active", c.getAttribute("data-preset") === presetId);
  });

  const preset = state.presets.find(p => p.id === presetId);
  if (!preset) return;

  state.currentAnalysis = {
    scan_id: preset.id,
    record: preset.record,
    capsule: preset.capsule,
    centroid: preset.centroid,
    follicles: preset.follicles,
    is_ovary: preset.record.diagnosis.code !== "NON_OVARIAN_PELVIC_SCAN",
    organ_type: preset.record.diagnosis.code === "NON_OVARIAN_PELVIC_SCAN" ? "Uterine Corpus" : "Ovary"
  };

  updateReportSheet(preset.record);
  updateFollicleTable(preset.follicles);
  loadPresetImages(preset);
}

function loadPresetImages(preset) {
  elements.loader.style.display = "flex";
  let loadedCount = 0;

  const rawImg = new Image();
  rawImg.src = preset.image_file;
  rawImg.onload = () => {
    state.images.raw = rawImg;
    loadedCount++;
    if (loadedCount === 2) onImagesReady();
  };

  const overlayImg = new Image();
  overlayImg.src = preset.overlay_file;
  overlayImg.onload = () => {
    state.images.overlay = overlayImg;
    loadedCount++;
    if (loadedCount === 2) onImagesReady();
  };
}

function onImagesReady() {
  elements.loader.style.display = "none";
  renderCanvas();
}

// ==================== REPORT SHEET UPDATES ====================
function updateReportSheet(rec) {
  if (!rec) return;

  elements.repScanId.textContent = `SCAN: ${rec.scan_id}`;
  elements.repTimestamp.textContent = rec.timestamp || new Date().toLocaleString();

  const diag = rec.diagnosis;
  const bio = rec.biomarkers;

  // Banner status
  elements.diagBanner.className = "diagnosis-banner";
  if (diag.code === "PCOM_POSITIVE") elements.diagBanner.classList.add("pcom-positive");
  else if (diag.code?.includes("BORDERLINE")) elements.diagBanner.classList.add("pcom-borderline");
  else if (diag.code === "NORMAL") elements.diagBanner.classList.add("pcom-normal");
  else elements.diagBanner.classList.add("pcom-guardrail");

  elements.diagMatch.textContent = diag.morphology_match;
  elements.diagConf.textContent = `${diag.confidence_pct?.toFixed(1)}%`;
  elements.diagPearls.textContent = diag.string_of_pearls_sign;
  elements.diagProfile.textContent = diag.dispersion_profile;
  elements.diagSeverity.textContent = diag.clinical_severity;

  // Metrics
  elements.valFnpo.textContent = bio.fnpo;
  const fnpoWidth = Math.min(100, (bio.fnpo / 35.0) * 100);
  elements.barFnpo.style.width = `${fnpoWidth}%`;

  elements.valPdi.textContent = bio.pdi_mean?.toFixed(3);
  const pdiWidth = Math.min(100, (bio.pdi_mean / 1.0) * 100);
  elements.barPdi.style.width = `${pdiWidth}%`;

  elements.valPrc.textContent = `${bio.prc_65_pct?.toFixed(1)}%`;
  elements.barPrc.style.width = `${bio.prc_65_pct}%`;

  elements.valClearance.textContent = `${bio.central_sparing_pct?.toFixed(1)}%`;
  elements.barClearance.style.width = `${bio.central_sparing_pct}%`;

  // Findings
  elements.findingsList.innerHTML = "";
  (rec.findings_narrative || []).forEach(f => {
    const li = document.createElement("li");
    li.textContent = f;
    elements.findingsList.appendChild(li);
  });

  // Recommendations
  elements.recommendationsBox.innerHTML = "";
  const ul = document.createElement("ul");
  (rec.clinical_recommendations || []).forEach(r => {
    const li = document.createElement("li");
    li.textContent = r;
    ul.appendChild(li);
  });
  elements.recommendationsBox.appendChild(ul);
}

// ==================== FOLLICLE TABLE ====================
function updateFollicleTable(follicles = []) {
  elements.follicleCount.textContent = follicles.length;
  elements.follicleTableBody.innerHTML = "";

  if (!follicles || follicles.length === 0) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td colspan="8" style="text-align:center; color: var(--text-muted); padding: 18px;">No antral follicles detected in this scan.</td>`;
    elements.follicleTableBody.appendChild(tr);
    return;
  }

  follicles.forEach(f => {
    const tr = document.createElement("tr");
    tr.setAttribute("data-follicle-id", f.id);
    if (state.hoveredFollicleId === f.id) tr.classList.add("highlighted");

    const isPeriph = f.is_peripheral || f.pdi >= 0.65;
    const badgeHtml = isPeriph 
      ? `<span class="legend-badge badge-periph">Subcapsular Rim</span>` 
      : `<span class="legend-badge badge-centr">Centrostromal</span>`;

    tr.innerHTML = `
      <td><strong>${f.id}</strong></td>
      <td>(${f.x}, ${f.y})</td>
      <td>${f.diameter_mm?.toFixed(1)} mm</td>
      <td>${f.center_dist_px?.toFixed(1)}</td>
      <td>${f.boundary_dist_px?.toFixed(1)}</td>
      <td><strong>${f.pdi?.toFixed(3)}</strong></td>
      <td>${badgeHtml}</td>
      <td>${f.contrast?.toFixed(2)}</td>
    `;

    tr.addEventListener("mouseenter", () => {
      state.hoveredFollicleId = f.id;
      tr.classList.add("highlighted");
      renderCanvas();
    });

    tr.addEventListener("mouseleave", () => {
      state.hoveredFollicleId = null;
      tr.classList.remove("highlighted");
      renderCanvas();
    });

    elements.follicleTableBody.appendChild(tr);
  });
}

// ==================== CANVAS RENDERING ====================
function renderCanvas() {
  const canvas = elements.canvas;
  const w = canvas.width;
  const h = canvas.height;

  ctx.clearRect(0, 0, w, h);

  // 1. Draw Background Image
  const activeImg = (state.viewMode === "overlay" && state.images.overlay) ? state.images.overlay : state.images.raw;
  if (activeImg) {
    ctx.drawImage(activeImg, 0, 0, w, h);
  } else {
    ctx.fillStyle = "#0A0E1A";
    ctx.fillRect(0, 0, w, h);
  }

  const analysis = state.currentAnalysis;
  if (!analysis) return;

  // Scale factor (analysis coordinates are in 384x384 space)
  const scale = w / 384.0;
  const opacity = state.layers.opacity;

  // 2. Draw Capsule Boundary
  if (state.layers.capsule && analysis.capsule && analysis.capsule.length > 0) {
    ctx.save();
    ctx.globalAlpha = opacity;
    ctx.beginPath();
    const pts = analysis.capsule;
    ctx.moveTo(pts[0][0][0] * scale, pts[0][0][1] * scale);
    for (let i = 1; i < pts.length; i++) {
      ctx.lineTo(pts[i][0][0] * scale, pts[i][0][1] * scale);
    }
    ctx.closePath();
    ctx.strokeStyle = "#06B6D4";
    ctx.lineWidth = 2.2;
    ctx.shadowColor = "rgba(6, 182, 212, 0.6)";
    ctx.shadowBlur = 8;
    ctx.stroke();
    ctx.restore();
  }

  // 3. Draw Centroid & Central Stroma Core
  if (analysis.centroid) {
    const cx = analysis.centroid[0] * scale;
    const cy = analysis.centroid[1] * scale;

    if (state.layers.stroma) {
      ctx.save();
      ctx.globalAlpha = opacity * 0.75;
      ctx.beginPath();
      // Stromal core radius approx 35% of major axis
      const rx = (analysis.major_axis_px || 120) * 0.5 * 0.35 * scale;
      const ry = (analysis.minor_axis_px || 90) * 0.5 * 0.35 * scale;
      ctx.ellipse(cx, cy, Math.max(15, rx), Math.max(12, ry), 0, 0, Math.PI * 2);
      ctx.strokeStyle = "#A855F7";
      ctx.lineWidth = 1.8;
      ctx.setLineDash([4, 4]);
      ctx.stroke();
      ctx.restore();
    }

    // Centroid marker
    ctx.save();
    ctx.beginPath();
    ctx.arc(cx, cy, 4, 0, Math.PI * 2);
    ctx.fillStyle = "#06B6D4";
    ctx.fill();
    ctx.strokeStyle = "#FFFFFF";
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.restore();
  }

  // 4. Draw Radial Rays & Follicles
  if (analysis.follicles && analysis.follicles.length > 0) {
    const cx = analysis.centroid ? analysis.centroid[0] * scale : w / 2;
    const cy = analysis.centroid ? analysis.centroid[1] * scale : h / 2;

    analysis.follicles.forEach(f => {
      const fx = f.x * scale;
      const fy = f.y * scale;
      const fr = Math.max(3, f.radius_px * scale);
      const isHovered = state.hoveredFollicleId === f.id;
      const isPeriph = f.is_peripheral || f.pdi >= 0.65;

      // Draw Ray
      if (state.layers.rays) {
        ctx.save();
        ctx.globalAlpha = isHovered ? 0.9 : opacity * 0.45;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(fx, fy);
        ctx.strokeStyle = isPeriph ? "#84CC16" : "#F59E0B";
        ctx.lineWidth = isHovered ? 2.0 : 1.0;
        ctx.stroke();

        // Continuation ray to outer capsule boundary
        if (f.boundary_pt) {
          const bx = f.boundary_pt[0] * scale;
          const by = f.boundary_pt[1] * scale;
          ctx.beginPath();
          ctx.moveTo(fx, fy);
          ctx.lineTo(bx, by);
          ctx.strokeStyle = "rgba(244, 63, 94, 0.4)";
          ctx.setLineDash([2, 3]);
          ctx.lineWidth = 1.0;
          ctx.stroke();
        }
        ctx.restore();
      }

      // Draw Follicle Marker
      if (state.layers.follicles) {
        ctx.save();
        ctx.beginPath();
        ctx.arc(fx, fy, fr, 0, Math.PI * 2);

        // Core fill
        ctx.fillStyle = "rgba(0, 0, 0, 0.45)";
        ctx.fill();

        // Perimeter ring
        ctx.strokeStyle = isPeriph ? "#84CC16" : "#F59E0B";
        ctx.lineWidth = isHovered ? 3.0 : 1.8;
        if (isHovered) {
          ctx.shadowColor = ctx.strokeStyle;
          ctx.shadowBlur = 10;
        }
        ctx.stroke();

        // Follicle ID Tag
        ctx.fillStyle = "#FFFFFF";
        ctx.font = `600 ${Math.max(9, Math.round(9 * scale))}px JetBrains Mono, monospace`;
        ctx.fillText(f.id, fx + fr + 3, fy - 3);

        ctx.restore();
      }
    });
  }
}

// ==================== CANVAS INTERACTION & TOOLTIP ====================
function setupCanvasInteractions() {
  const canvas = elements.canvas;
  const tooltip = elements.tooltip;

  canvas.addEventListener("mousemove", (e) => {
    const rect = canvas.getBoundingClientRect();
    const mouseX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const mouseY = (e.clientY - rect.top) * (canvas.height / rect.height);
    const scale = canvas.width / 384.0;

    const analysis = state.currentAnalysis;
    if (!analysis || !analysis.follicles) return;

    let found = null;
    for (const f of analysis.follicles) {
      const fx = f.x * scale;
      const fy = f.y * scale;
      const fr = Math.max(6, f.radius_px * scale + 4);
      const dist = Math.hypot(mouseX - fx, mouseY - fy);
      if (dist <= fr) {
        found = f;
        break;
      }
    }

    if (found) {
      state.hoveredFollicleId = found.id;
      tooltip.style.display = "block";
      tooltip.style.left = `${e.clientX - rect.left + 15}px`;
      tooltip.style.top = `${e.clientY - rect.top - 20}px`;
      tooltip.innerHTML = `
        <strong>Follicle #${found.id}</strong><br>
        Caliber: ${found.diameter_mm?.toFixed(1)} mm<br>
        PDI (&rho;): <strong>${found.pdi?.toFixed(3)}</strong><br>
        d_center: ${found.center_dist_px?.toFixed(1)} px<br>
        R_boundary: ${found.boundary_dist_px?.toFixed(1)} px<br>
        Zone: ${found.is_peripheral ? "Subcapsular Rim" : "Centrostromal"}
      `;

      // Highlight table row
      document.querySelectorAll("#follicle-table tbody tr").forEach(tr => {
        tr.classList.toggle("highlighted", tr.getAttribute("data-follicle-id") == found.id);
      });
      renderCanvas();
    } else {
      if (state.hoveredFollicleId !== null) {
        state.hoveredFollicleId = null;
        tooltip.style.display = "none";
        document.querySelectorAll("#follicle-table tbody tr").forEach(tr => tr.classList.remove("highlighted"));
        renderCanvas();
      }
    }
  });

  canvas.addEventListener("mouseleave", () => {
    state.hoveredFollicleId = null;
    tooltip.style.display = "none";
    document.querySelectorAll("#follicle-table tbody tr").forEach(tr => tr.classList.remove("highlighted"));
    renderCanvas();
  });
}

// ==================== FILE UPLOAD & LIVE ANALYSIS ====================
function setupUploadAndExports() {
  // File upload
  elements.fileInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = async (evt) => {
      const base64Data = evt.target.result;
      analyzeUploadedImage(base64Data, file.name);
    };
    reader.readAsDataURL(file);
  });

  // Print Report Card
  elements.btnPrint.addEventListener("click", () => {
    window.print();
  });

  // Export JSON
  elements.btnExportJson.addEventListener("click", () => {
    if (!state.currentAnalysis) return;
    const blob = new Blob([JSON.stringify(state.currentAnalysis.record, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${state.currentAnalysis.scan_id}_clinical_report.json`;
    a.click();
    URL.revokeObjectURL(url);
  });

  // Export Annotated Image
  elements.btnExportOverlay.addEventListener("click", () => {
    const link = document.createElement("a");
    link.download = `${state.currentAnalysis?.scan_id || "scan"}_pcos_bioquant.png`;
    link.href = elements.canvas.toDataURL("image/png");
    link.click();
  });
}

async function analyzeUploadedImage(base64Image, fileName) {
  elements.loader.style.display = "flex";
  try {
    const scanId = `UPLOAD_${fileName.replace(/\.[^/.]+$/, "")}`;
    const res = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image: base64Image,
        scan_id: scanId
      })
    });

    if (!res.ok) throw new Error("Analysis failed");
    const data = await res.json();

    state.currentAnalysis = {
      scan_id: data.scan_id,
      record: data.record,
      capsule: data.capsule,
      centroid: data.centroid,
      major_axis_px: data.major_axis_px,
      minor_axis_px: data.minor_axis_px,
      follicles: data.follicles,
      is_ovary: data.is_ovary,
      organ_type: data.organ_type
    };

    updateReportSheet(data.record);
    updateFollicleTable(data.follicles);

    // Load returned base64 images
    const rawImg = new Image();
    rawImg.src = data.raw_b64;
    const overlayImg = new Image();
    overlayImg.src = data.overlay_b64;

    let loaded = 0;
    const checkDone = () => {
      loaded++;
      if (loaded === 2) {
        state.images.raw = rawImg;
        state.images.overlay = overlayImg;
        elements.loader.style.display = "none";
        renderCanvas();
      }
    };
    rawImg.onload = checkDone;
    overlayImg.onload = checkDone;

  } catch (err) {
    elements.loader.style.display = "none";
    alert(`Analysis error: ${err.message}`);
    console.error(err);
  }
}
