import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const workspaceDir = path.resolve(scriptDir, "..");
const SKILL_DIR = process.env.CODEX_PRESENTATIONS_SKILL_DIR;
const RUNTIME_PYTHON = process.env.CODEX_RUNTIME_PYTHON ?? "python3";
if (!SKILL_DIR) throw new Error("Set CODEX_PRESENTATIONS_SKILL_DIR.");

const buildDir = path.join(workspaceDir, ".codex-presentation-build");
const stagingDir = path.join(workspaceDir, ".codex-finalizer");
const finalPath = path.join(workspaceDir, "presentation", "Inventory_Demand_Forecasting_Phase2.pptx");
const coverImage = path.join(workspaceDir, "presentation", "assets", "inventory-forecast-cover.png");

const FONT = "Aptos";
const NAVY = "#14233D";
const NAVY_2 = "#1C3558";
const TEAL = "#12A89D";
const TEAL_LIGHT = "#D9F3F0";
const AMBER = "#F2B43D";
const AMBER_LIGHT = "#FFF0CA";
const BLUE = "#3D73C5";
const BLUE_LIGHT = "#DFE9F8";
const CORAL = "#D95C50";
const CORAL_LIGHT = "#F9E2DF";
const INK = "#17243A";
const MUTED = "#617087";
const GRID = "#D9E2EC";
const BG = "#F4F7FA";
const WHITE = "#FFFFFF";

await fs.mkdir(buildDir, { recursive: true });
await fs.mkdir(stagingDir, { recursive: true });
await fs.mkdir(path.dirname(finalPath), { recursive: true });

const presentation = Presentation.create({ slideSize: { width: 1280, height: 720 } });
presentation.theme.defaultFont = FONT;

function addText(slide, text, position, style = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position,
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = {
    fontSize: style.fontSize ?? 22,
    bold: style.bold ?? false,
    color: style.color ?? INK,
    alignment: style.alignment ?? "left",
    verticalAlignment: style.verticalAlignment ?? "top",
  };
  return shape;
}

function addRect(slide, position, fill, radius = false, lineFill = "none") {
  return slide.shapes.add({
    geometry: radius ? "roundRect" : "rect",
    position,
    fill,
    line: { style: "solid", fill: lineFill, width: lineFill === "none" ? 0 : 1 },
    borderRadius: radius ? "rounded-xl" : undefined,
  });
}

function addHeader(slide, section, title, subtitle = "") {
  addText(slide, section.toUpperCase(), { left: 64, top: 30, width: 300, height: 24 }, {
    fontSize: 13, bold: true, color: TEAL,
  });
  addText(slide, title, { left: 64, top: 58, width: 1140, height: 54 }, {
    fontSize: 36, bold: true, color: NAVY,
  });
  if (subtitle) {
    addText(slide, subtitle, { left: 64, top: 119, width: 1130, height: 38 }, {
      fontSize: 18, color: MUTED,
    });
  }
  addText(slide, String(presentation.slides.items.length), { left: 1184, top: 676, width: 32, height: 18 }, {
    fontSize: 11, color: MUTED, alignment: "right",
  });
}

function addNotes(slide, lines) {
  slide.speakerNotes.text = lines.join("\n");
  slide.speakerNotes.setVisible(true);
}

async function readImageBlob(filePath) {
  const bytes = await fs.readFile(filePath);
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

const { applyPresentationChartFont } = await import(pathToFileURL(
  path.join(SKILL_DIR, "container_tools", "artifact_tool_utils.mjs"),
).href);

function styleChart(chart) {
  applyPresentationChartFont(chart, { fontFamily: FONT });
  chart.xAxis = {
    ...chart.xAxis,
    textStyle: { fill: MUTED, fontSize: 13 },
    line: { style: "solid", fill: GRID, width: 1 },
  };
  chart.yAxis = {
    ...chart.yAxis,
    textStyle: { fill: MUTED, fontSize: 13 },
    line: { style: "solid", fill: GRID, width: 1 },
  };
}

// 1. Cover
{
  const slide = presentation.slides.add();
  slide.background.fill = NAVY;
  addRect(slide, { left: 0, top: 0, width: 18, height: 720 }, TEAL);
  addText(slide, "TEAM PROJECT  |  PHASE 2", { left: 72, top: 60, width: 420, height: 26 }, {
    fontSize: 14, bold: true, color: AMBER,
  });
  addText(slide, "Inventory demand\nforecasting", { left: 72, top: 124, width: 610, height: 150 }, {
    fontSize: 54, bold: true, color: WHITE,
  });
  addText(slide, "From historical sales to probabilistic inventory decisions", { left: 76, top: 300, width: 590, height: 70 }, {
    fontSize: 25, color: "#D9E3EF",
  });
  addText(slide, "Kalman filtering  ·  Gradient boosting  ·  Global DeepAR", { left: 76, top: 392, width: 620, height: 34 }, {
    fontSize: 18, bold: true, color: TEAL_LIGHT,
  });
  const imageBytes = await readImageBlob(coverImage);
  slide.images.add({
    blob: imageBytes,
    contentType: "image/png",
    alt: "Illustration of retail inventory with a probabilistic demand forecast",
    fit: "contain",
    position: { left: 630, top: 58, width: 620, height: 570 },
  });
  addText(slide, "Divy Kumar Jain  ·  Praveen Kumar  ·  Palwai Tejaswini  ·  Jatin  ·  Piyush Raj", { left: 76, top: 604, width: 700, height: 52 }, {
    fontSize: 16, color: WHITE,
  });
  addText(slide, "M5 Forecasting Accuracy dataset  |  November 2026", { left: 76, top: 654, width: 600, height: 24 }, {
    fontSize: 13, color: "#AFC0D4",
  });
  addNotes(slide, [
    "Opening: ask the audience to think about how a retailer orders tomorrow's inventory when demand can be zero or suddenly spike.",
    "This deck presents the corrected and validated Phase 2 experiments.",
  ]);
}

// 2. Decision problem
{
  const slide = presentation.slides.add();
  slide.background.fill = BG;
  addHeader(slide, "The decision", "How much should the store order tomorrow?", "A forecast must balance two different mistakes before demand becomes known.");
  addRect(slide, { left: 70, top: 200, width: 500, height: 300 }, CORAL_LIGHT, true);
  addText(slide, "TOO LITTLE", { left: 108, top: 235, width: 410, height: 32 }, { fontSize: 16, bold: true, color: CORAL });
  addText(slide, "Stockout", { left: 108, top: 278, width: 410, height: 54 }, { fontSize: 38, bold: true, color: NAVY });
  addText(slide, "Lost sales\nLower service level\nUnhappy customers", { left: 108, top: 350, width: 380, height: 110 }, { fontSize: 23, color: INK });
  addRect(slide, { left: 710, top: 200, width: 500, height: 300 }, AMBER_LIGHT, true);
  addText(slide, "TOO MUCH", { left: 748, top: 235, width: 410, height: 32 }, { fontSize: 16, bold: true, color: "#B67800" });
  addText(slide, "Overstock", { left: 748, top: 278, width: 410, height: 54 }, { fontSize: 38, bold: true, color: NAVY });
  addText(slide, "Holding cost\nTied-up cash\nWaste or markdowns", { left: 748, top: 350, width: 380, height: 110 }, { fontSize: 23, color: INK });
  addText(slide, "Audience question", { left: 80, top: 548, width: 220, height: 30 }, { fontSize: 15, bold: true, color: TEAL });
  addText(slide, "Which mistake would you penalize more: one lost sale or one unsold unit?", { left: 80, top: 580, width: 1090, height: 48 }, { fontSize: 27, bold: true, color: NAVY, alignment: "center" });
  addNotes(slide, [
    "Audience prompt: invite one or two answers before explaining that the inventory simulation tests several cost ratios.",
    "The project treats inventory costs as educational assumptions, not Walmart business values.",
  ]);
}

// 3. Demand characteristics
{
  const slide = presentation.slides.add();
  slide.background.fill = WHITE;
  addHeader(slide, "Why forecasting is difficult", "Retail demand is sparse, seasonal, and uneven", "The selected series contain many zero-demand days and occasional spikes.");
  addText(slide, "64.7%", { left: 78, top: 210, width: 300, height: 90 }, { fontSize: 64, bold: true, color: CORAL });
  addText(slide, "zero-sales rows", { left: 82, top: 304, width: 280, height: 40 }, { fontSize: 23, color: INK });
  addText(slide, "1,913", { left: 470, top: 210, width: 250, height: 90 }, { fontSize: 64, bold: true, color: BLUE });
  addText(slide, "daily observations per series", { left: 474, top: 304, width: 290, height: 52 }, { fontSize: 23, color: INK });
  addText(slide, "100", { left: 880, top: 210, width: 240, height: 90 }, { fontSize: 64, bold: true, color: TEAL });
  addText(slide, "item-store series", { left: 884, top: 304, width: 250, height: 40 }, { fontSize: 23, color: INK });
  addRect(slide, { left: 76, top: 408, width: 1128, height: 138 }, NAVY, true);
  addText(slide, "The model must learn recurring calendar patterns without treating every zero as missing demand.", { left: 120, top: 442, width: 1040, height: 62 }, { fontSize: 28, bold: true, color: WHITE, alignment: "center", verticalAlignment: "middle" });
  addText(slide, "Availability uses price presence and a closure flag. It remains a proxy, not verified inventory-on-hand.", { left: 112, top: 576, width: 1050, height: 36 }, { fontSize: 17, color: CORAL, alignment: "center" });
  addNotes(slide, [
    "Source: results/phase2/data_metadata.json.",
    "Explain that zero demand may be genuine no-sale demand or may reflect uncertain availability; the project keeps a separate availability proxy.",
  ]);
}

// 4. Experimental design
{
  const slide = presentation.slides.add();
  slide.background.fill = BG;
  addHeader(slide, "Fair comparison", "One time boundary for every model", "Training, validation, and final testing remain strictly separated.");
  addText(slide, "2011", { left: 78, top: 188, width: 100, height: 30 }, { fontSize: 16, bold: true, color: MUTED });
  addText(slide, "13 Mar 2016", { left: 685, top: 188, width: 150, height: 30 }, { fontSize: 16, bold: true, color: MUTED });
  addText(slide, "27 Mar", { left: 916, top: 188, width: 100, height: 30 }, { fontSize: 16, bold: true, color: MUTED });
  addText(slide, "24 Apr", { left: 1125, top: 188, width: 100, height: 30 }, { fontSize: 16, bold: true, color: MUTED, alignment: "right" });
  addRect(slide, { left: 80, top: 230, width: 700, height: 92 }, BLUE, true);
  addText(slide, "TRAIN", { left: 105, top: 248, width: 130, height: 28 }, { fontSize: 16, bold: true, color: WHITE });
  addText(slide, "Fit model parameters", { left: 105, top: 278, width: 320, height: 30 }, { fontSize: 22, color: WHITE });
  addRect(slide, { left: 790, top: 230, width: 208, height: 92 }, AMBER, true);
  addText(slide, "VALIDATE", { left: 810, top: 248, width: 150, height: 28 }, { fontSize: 16, bold: true, color: NAVY });
  addText(slide, "Early stopping", { left: 810, top: 278, width: 160, height: 28 }, { fontSize: 20, color: NAVY });
  addRect(slide, { left: 1008, top: 230, width: 194, height: 92 }, TEAL, true);
  addText(slide, "TEST", { left: 1028, top: 248, width: 130, height: 28 }, { fontSize: 16, bold: true, color: WHITE });
  addText(slide, "Untouched", { left: 1028, top: 278, width: 150, height: 28 }, { fontSize: 20, color: WHITE });
  addText(slide, "4 rolling origins", { left: 125, top: 390, width: 270, height: 46 }, { fontSize: 30, bold: true, color: NAVY });
  addText(slide, "Each origin forecasts the next 7 days", { left: 125, top: 440, width: 330, height: 54 }, { fontSize: 20, color: MUTED });
  addText(slide, "2,800 final forecasts", { left: 500, top: 390, width: 310, height: 46 }, { fontSize: 30, bold: true, color: NAVY });
  addText(slide, "100 series × 4 origins × 7 days", { left: 500, top: 440, width: 330, height: 54 }, { fontSize: 20, color: MUTED });
  addText(slide, "Same keys for every model", { left: 865, top: 390, width: 320, height: 72 }, { fontSize: 30, bold: true, color: NAVY });
  addText(slide, "Same dates, actuals, and horizon", { left: 865, top: 468, width: 320, height: 54 }, { fontSize: 20, color: MUTED });
  addNotes(slide, [
    "Source: results/phase2/data_metadata.json and configs/phase2_config.json.",
    "Emphasize that no model sees the final 28 test days during training or model selection.",
  ]);
}

// 5. Phase 1 baseline
{
  const slide = presentation.slides.add();
  slide.background.fill = WHITE;
  addHeader(slide, "Phase 1 reference", "Three baselines establish the starting point", "Lower mean absolute error is better.");
  const chart = slide.charts.add("bar", {
    position: { left: 64, top: 178, width: 730, height: 410 },
    categories: ["Seasonal Naive", "Local Kalman", "Gradient Boosting"],
    series: [{ name: "MAE", values: [1.2096, 1.0595, 1.0224], fill: BLUE }],
    barOptions: { direction: "bar", grouping: "clustered", gapWidth: 58 },
    hasLegend: false,
    xAxis: { min: 0, max: 1.35, majorGridlines: { style: "solid", fill: GRID, width: 1 }, title: { text: "Mean absolute error" } },
    yAxis: { majorGridlines: null },
    dataLabels: { showValue: true, position: "outEnd", numberFormatCode: "0.0000", textStyle: { fill: NAVY, fontSize: 14, bold: true } },
  });
  styleChart(chart);
  addText(slide, "What each baseline teaches us", { left: 850, top: 190, width: 340, height: 36 }, { fontSize: 23, bold: true, color: NAVY });
  addText(slide, "Seasonal Naive\nA minimum weekly benchmark", { left: 850, top: 252, width: 340, height: 66 }, { fontSize: 20, color: INK });
  addText(slide, "Local Kalman\nFast adaptation with uncertainty", { left: 850, top: 354, width: 340, height: 66 }, { fontSize: 20, color: INK });
  addText(slide, "Gradient Boosting\nBest Phase 1 MAE and WAPE", { left: 850, top: 456, width: 340, height: 66 }, { fontSize: 20, color: INK });
  addRect(slide, { left: 830, top: 548, width: 380, height: 74 }, TEAL_LIGHT, true);
  addText(slide, "Phase 2 must beat 1.0224 MAE", { left: 856, top: 568, width: 330, height: 32 }, { fontSize: 21, bold: true, color: NAVY, alignment: "center" });
  addNotes(slide, [
    "Audience prompt: ask which baseline they expect a neural model to beat first.",
    "Source: results/phase1_model_comparison.csv.",
  ]);
}

// 6. DeepAR concept
{
  const slide = presentation.slides.add();
  slide.background.fill = BG;
  addHeader(slide, "Global sequence learning", "DeepAR learns one model from many time series", "Each series contributes examples to a shared recurrent model.");
  addText(slide, "100 item-store histories", { left: 74, top: 188, width: 300, height: 36 }, { fontSize: 22, bold: true, color: NAVY });
  for (let i = 0; i < 6; i += 1) {
    addRect(slide, { left: 86, top: 244 + i * 42, width: 255, height: 17 }, i % 2 === 0 ? BLUE : TEAL, true);
  }
  addText(slide, "→", { left: 367, top: 330, width: 80, height: 60 }, { fontSize: 44, bold: true, color: TEAL, alignment: "center" });
  addRect(slide, { left: 450, top: 218, width: 335, height: 290 }, NAVY, true);
  addText(slide, "Shared LSTM", { left: 490, top: 252, width: 255, height: 50 }, { fontSize: 34, bold: true, color: WHITE, alignment: "center" });
  addText(slide, "Previous demand\nKnown calendar features\nStatic item and store identity", { left: 500, top: 330, width: 235, height: 116 }, { fontSize: 20, color: "#DCE6F2", alignment: "center" });
  addText(slide, "→", { left: 802, top: 330, width: 80, height: 60 }, { fontSize: 44, bold: true, color: TEAL, alignment: "center" });
  addText(slide, "Demand distribution", { left: 885, top: 188, width: 320, height: 36 }, { fontSize: 22, bold: true, color: NAVY });
  addRect(slide, { left: 898, top: 242, width: 285, height: 78 }, TEAL_LIGHT, true);
  addText(slide, "Negative Binomial", { left: 920, top: 265, width: 240, height: 35 }, { fontSize: 23, bold: true, color: NAVY, alignment: "center" });
  addRect(slide, { left: 898, top: 342, width: 285, height: 78 }, BLUE_LIGHT, true);
  addText(slide, "100 recursive paths", { left: 920, top: 365, width: 240, height: 35 }, { fontSize: 23, bold: true, color: NAVY, alignment: "center" });
  addRect(slide, { left: 898, top: 442, width: 285, height: 78 }, AMBER_LIGHT, true);
  addText(slide, "q10  ·  q50  ·  q90", { left: 920, top: 465, width: 240, height: 35 }, { fontSize: 23, bold: true, color: NAVY, alignment: "center" });
  addText(slide, "Teacher forcing uses demand from t−1. During forecasting, sampled demand feeds the next step.", { left: 134, top: 575, width: 1010, height: 40 }, { fontSize: 19, bold: true, color: MUTED, alignment: "center" });
  addNotes(slide, [
    "Presenter explanation: one network shares patterns across all 100 series instead of fitting 100 separate neural networks.",
    "DeepAR source: Salinas et al., International Journal of Forecasting, 2020, doi:10.1016/j.ijforecast.2019.07.001.",
    "Architecture source: configs/phase2_config.json and src/phase2/deepar_model.py.",
  ]);
}

// 7. Probabilistic forecast example
{
  const slide = presentation.slides.add();
  slide.background.fill = WHITE;
  addHeader(slide, "Reading a probabilistic forecast", "A range answers more than a single number", "Example: FOODS_1_065 at store WI_1, origin 27 March 2016.");
  const chart = slide.charts.add("line", {
    position: { left: 58, top: 178, width: 820, height: 420 },
    categories: ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5", "Day 6", "Day 7"],
    series: [
      { name: "Actual", values: [1, 0, 0, 1, 0, 1, 2], line: { style: "solid", fill: NAVY, width: 4 }, marker: { symbol: "circle", size: 8, fill: NAVY } },
      { name: "q10", values: [0, 0, 0, 0, 0, 0, 0], line: { style: "dash", fill: BLUE, width: 2 } },
      { name: "q50", values: [0, 0, 0, 1, 0, 1, 1], line: { style: "solid", fill: TEAL, width: 3 }, marker: { symbol: "circle", size: 6, fill: TEAL } },
      { name: "q90", values: [2, 2, 2, 3, 3, 3, 3], line: { style: "dash", fill: AMBER, width: 2 } },
    ],
    hasLegend: true,
    legend: { position: "bottom", overlay: false },
    yAxis: { min: 0, max: 4, majorUnit: 1, title: { text: "Units" }, majorGridlines: { style: "solid", fill: GRID, width: 1 } },
    xAxis: { majorGridlines: null },
  });
  styleChart(chart);
  addText(slide, "q50", { left: 930, top: 210, width: 180, height: 38 }, { fontSize: 29, bold: true, color: TEAL });
  addText(slide, "Typical demand", { left: 930, top: 252, width: 250, height: 32 }, { fontSize: 19, color: INK });
  addText(slide, "q90", { left: 930, top: 330, width: 180, height: 38 }, { fontSize: 29, bold: true, color: AMBER });
  addText(slide, "Higher-service order", { left: 930, top: 372, width: 250, height: 32 }, { fontSize: 19, color: INK });
  addText(slide, "q10", { left: 930, top: 450, width: 180, height: 38 }, { fontSize: 29, bold: true, color: BLUE });
  addText(slide, "Low-demand case", { left: 930, top: 492, width: 250, height: 32 }, { fontSize: 19, color: INK });
  addRect(slide, { left: 902, top: 556, width: 300, height: 68 }, NAVY, true);
  addText(slide, "Would you order 0, 1, or 3?", { left: 920, top: 576, width: 265, height: 30 }, { fontSize: 19, bold: true, color: WHITE, alignment: "center" });
  addNotes(slide, [
    "Audience prompt: ask which quantile they would choose and what business cost assumption drives that choice.",
    "Source: results/phase2/deepar_base_v2_predictions.csv.gz.",
  ]);
}

// 8. Hybrid
{
  const slide = presentation.slides.add();
  slide.background.fill = BG;
  addHeader(slide, "Hybrid architecture", "The Kalman filter supplies state features to DeepAR", "The hybrid combines features inside one model. It does not average two forecasts.");
  addRect(slide, { left: 72, top: 205, width: 300, height: 190 }, BLUE_LIGHT, true);
  addText(slide, "Observed history", { left: 105, top: 235, width: 235, height: 40 }, { fontSize: 27, bold: true, color: NAVY, alignment: "center" });
  addText(slide, "Only demand available\nbefore the forecast origin", { left: 105, top: 302, width: 235, height: 62 }, { fontSize: 20, color: INK, alignment: "center" });
  addText(slide, "→", { left: 390, top: 270, width: 70, height: 60 }, { fontSize: 42, bold: true, color: TEAL, alignment: "center" });
  addRect(slide, { left: 470, top: 185, width: 340, height: 230 }, AMBER_LIGHT, true);
  addText(slide, "Local Kalman filter", { left: 510, top: 220, width: 260, height: 40 }, { fontSize: 27, bold: true, color: NAVY, alignment: "center" });
  addText(slide, "Level\nState variance\nPrevious innovation", { left: 510, top: 285, width: 260, height: 100 }, { fontSize: 21, color: INK, alignment: "center" });
  addText(slide, "→", { left: 825, top: 270, width: 70, height: 60 }, { fontSize: 42, bold: true, color: TEAL, alignment: "center" });
  addRect(slide, { left: 905, top: 205, width: 300, height: 190 }, TEAL_LIGHT, true);
  addText(slide, "Global DeepAR", { left: 940, top: 235, width: 230, height: 40 }, { fontSize: 27, bold: true, color: NAVY, alignment: "center" });
  addText(slide, "Learns whether local state\ninformation improves forecasts", { left: 940, top: 294, width: 230, height: 84 }, { fontSize: 18, color: INK, alignment: "center" });
  addRect(slide, { left: 128, top: 480, width: 1024, height: 118 }, NAVY, true);
  addText(slide, "Causal safeguard", { left: 162, top: 505, width: 210, height: 32 }, { fontSize: 18, bold: true, color: AMBER });
  addText(slide, "Each state is recorded before the current target. Scaling uses only strictly past usable observations.", { left: 380, top: 503, width: 720, height: 58 }, { fontSize: 22, color: WHITE });
  addText(slide, "Changing future sales cannot change earlier Kalman features.", { left: 380, top: 561, width: 720, height: 28 }, { fontSize: 17, bold: true, color: TEAL_LIGHT });
  addNotes(slide, [
    "Kalman source: R. E. Kalman, Journal of Basic Engineering, 1960, doi:10.1115/1.3662552.",
    "Leakage correction evidence: report/PHASE2_REMEDIATION_REPORT.md and tests/phase2/test_kalman_features.py.",
  ]);
}

// 9. Headline results
{
  const slide = presentation.slides.add();
  slide.background.fill = WHITE;
  addHeader(slide, "Overall evaluation", "DeepAR records the lowest MAE on the shared test set", "All five models use the same 2,800 final forecast observations.");
  const chart = slide.charts.add("bar", {
    position: { left: 56, top: 170, width: 815, height: 445 },
    categories: ["Seasonal Naive", "Phase 1 Kalman", "Gradient Boosting", "DeepAR v2", "Causal hybrid v2"],
    series: [{ name: "MAE", values: [1.2096, 1.0595, 1.0224, 0.9221, 0.9559], fill: TEAL }],
    barOptions: { direction: "bar", grouping: "clustered", gapWidth: 52 },
    hasLegend: false,
    xAxis: { min: 0, max: 1.35, title: { text: "Mean absolute error" }, majorGridlines: { style: "solid", fill: GRID, width: 1 } },
    yAxis: { majorGridlines: null },
    dataLabels: { showValue: true, position: "outEnd", numberFormatCode: "0.0000", textStyle: { fill: NAVY, fontSize: 14, bold: true } },
  });
  styleChart(chart);
  addText(slide, "0.9221", { left: 932, top: 190, width: 240, height: 74 }, { fontSize: 54, bold: true, color: TEAL, alignment: "center" });
  addText(slide, "DeepAR MAE", { left: 932, top: 270, width: 240, height: 34 }, { fontSize: 21, bold: true, color: NAVY, alignment: "center" });
  addRect(slide, { left: 910, top: 338, width: 286, height: 102 }, BLUE_LIGHT, true);
  addText(slide, "Gradient Boosting still has\nthe lowest RMSE: 2.0397", { left: 930, top: 365, width: 246, height: 55 }, { fontSize: 19, color: NAVY, alignment: "center" });
  addRect(slide, { left: 910, top: 466, width: 286, height: 126 }, CORAL_LIGHT, true);
  addText(slide, "The corrected hybrid does not\nimprove accuracy over DeepAR", { left: 930, top: 490, width: 246, height: 82 }, { fontSize: 18, bold: true, color: CORAL, alignment: "center" });
  addNotes(slide, [
    "Source: results/phase1_model_comparison.csv, results/phase2/deepar_base_v2_metrics.csv, and results/phase2/deepar_kalman_hybrid_v2_metrics.csv.",
    "Explain the metric trade-off: DeepAR leads typical absolute error, while gradient boosting handles the largest misses better according to RMSE.",
  ]);
}

// 10. Calibration
{
  const slide = presentation.slides.add();
  slide.background.fill = BG;
  addHeader(slide, "Probabilistic evaluation", "The 80% intervals are too conservative", "Observed coverage stays above the nominal target at every forecast step.");
  const chart = slide.charts.add("line", {
    position: { left: 58, top: 176, width: 850, height: 420 },
    categories: ["1", "2", "3", "4", "5", "6", "7"],
    series: [
      { name: "DeepAR", values: [95, 94.75, 93.75, 95, 93, 93.25, 92.5], line: { style: "solid", fill: TEAL, width: 3 }, marker: { symbol: "circle", size: 7, fill: TEAL } },
      { name: "Causal hybrid", values: [94.5, 94, 91.75, 95.25, 92.75, 92.75, 91.75], line: { style: "solid", fill: BLUE, width: 3 }, marker: { symbol: "circle", size: 7, fill: BLUE } },
      { name: "Nominal", values: [80, 80, 80, 80, 80, 80, 80], line: { style: "dash", fill: CORAL, width: 2 } },
    ],
    hasLegend: true,
    legend: { position: "bottom", overlay: false },
    yAxis: { min: 75, max: 100, majorUnit: 5, title: { text: "Coverage (%)" }, majorGridlines: { style: "solid", fill: GRID, width: 1 } },
    xAxis: { title: { text: "Forecast horizon day" }, majorGridlines: null },
  });
  styleChart(chart);
  addText(slide, "93.89%", { left: 970, top: 210, width: 190, height: 58 }, { fontSize: 42, bold: true, color: TEAL, alignment: "center" });
  addText(slide, "DeepAR coverage", { left: 960, top: 276, width: 210, height: 30 }, { fontSize: 19, color: NAVY, alignment: "center" });
  addText(slide, "3.37", { left: 970, top: 354, width: 190, height: 58 }, { fontSize: 42, bold: true, color: BLUE, alignment: "center" });
  addText(slide, "mean interval width", { left: 950, top: 420, width: 230, height: 30 }, { fontSize: 19, color: NAVY, alignment: "center" });
  addRect(slide, { left: 932, top: 492, width: 270, height: 112 }, AMBER_LIGHT, true);
  addText(slide, "Wider intervals catch more actuals,\nbut they provide less precise guidance.", { left: 952, top: 510, width: 230, height: 80 }, { fontSize: 17, color: NAVY, alignment: "center" });
  addNotes(slide, [
    "Audience prompt: ask whether 94% coverage should be called better when the target was 80%.",
    "Source: results/phase2/deepar_base_v2_calibration.csv and deepar_kalman_hybrid_v2_calibration.csv.",
  ]);
}

// 11. Cold start
{
  const slide = presentation.slides.add();
  slide.background.fill = WHITE;
  addHeader(slide, "Generalization", "Unseen items remain harder than familiar items", "The transfer experiment excludes 19 item IDs and all their targets from training.");
  const chart = slide.charts.add("bar", {
    position: { left: 58, top: 178, width: 760, height: 420 },
    categories: ["7 days", "14 days", "28 days", "56 days", "Zero shot"],
    series: [{ name: "MAE", values: [1.1673, 1.1118, 1.1278, 1.1720, 2.3938], fill: AMBER }],
    barOptions: { direction: "bar", grouping: "clustered", gapWidth: 55 },
    hasLegend: false,
    xAxis: { min: 0, max: 2.7, title: { text: "Mean absolute error" }, majorGridlines: { style: "solid", fill: GRID, width: 1 } },
    yAxis: { majorGridlines: null },
    dataLabels: { showValue: true, position: "outEnd", numberFormatCode: "0.0000", textStyle: { fill: NAVY, fontSize: 14, bold: true } },
  });
  styleChart(chart);
  addText(slide, "Limited-history transfer", { left: 866, top: 196, width: 330, height: 36 }, { fontSize: 24, bold: true, color: TEAL });
  addText(slide, "Held-out item identities\nSome permitted target history\n19 items, 532 forecasts", { left: 866, top: 246, width: 320, height: 112 }, { fontSize: 20, color: INK });
  addText(slide, "Zero shot", { left: 866, top: 390, width: 330, height: 36 }, { fontSize: 24, bold: true, color: CORAL });
  addText(slide, "No held-out target history\nUnknown item embedding\nTraining-only fallback scale", { left: 866, top: 440, width: 320, height: 112 }, { fontSize: 20, color: INK });
  addText(slide, "New combinations use only 3 series and 84 points, so conclusions remain narrow.", { left: 120, top: 625, width: 1040, height: 32 }, { fontSize: 17, bold: true, color: CORAL, alignment: "center" });
  addNotes(slide, [
    "Clarify that limited context on already-seen items answers a different question and is not shown in this chart.",
    "Source: results/phase2/cold_start_results.csv.",
  ]);
}

// 12. Ablation
{
  const slide = presentation.slides.add();
  slide.background.fill = BG;
  addHeader(slide, "Model analysis", "Price and availability produce the best pinball loss", "Lower mean pinball loss indicates better probabilistic forecasts.");
  const chart = slide.charts.add("bar", {
    position: { left: 58, top: 174, width: 790, height: 440 },
    categories: ["DeepAR", "Hybrid", "Target only", "Calendar", "Kalman addition", "Calendar + static"],
    series: [{ name: "Mean pinball", values: [0.3113, 0.3161, 0.3184, 0.3226, 0.3278, 0.3671], fill: TEAL }],
    barOptions: { direction: "bar", grouping: "clustered", gapWidth: 42 },
    hasLegend: false,
    xAxis: { min: 0.28, max: 0.39, majorUnit: 0.02, title: { text: "Mean pinball loss" }, majorGridlines: { style: "solid", fill: GRID, width: 1 } },
    yAxis: { majorGridlines: null },
    dataLabels: { showValue: true, position: "outEnd", numberFormatCode: "0.0000", textStyle: { fill: NAVY, fontSize: 14, bold: true } },
  });
  styleChart(chart);
  addRect(slide, { left: 890, top: 205, width: 300, height: 142 }, TEAL_LIGHT, true);
  addText(slide, "Best probabilistic score", { left: 915, top: 235, width: 250, height: 34 }, { fontSize: 20, bold: true, color: TEAL, alignment: "center" });
  addText(slide, "DeepAR\n0.3113", { left: 915, top: 276, width: 250, height: 62 }, { fontSize: 24, bold: true, color: NAVY, alignment: "center" });
  addRect(slide, { left: 890, top: 368, width: 300, height: 186 }, CORAL_LIGHT, true);
  addText(slide, "Hybrid conclusion", { left: 915, top: 394, width: 250, height: 34 }, { fontSize: 20, bold: true, color: CORAL, alignment: "center" });
  addText(slide, "The paired bootstrap interval\nfor DeepAR minus hybrid crosses zero.\nNo clear MAE advantage.", { left: 915, top: 440, width: 250, height: 98 }, { fontSize: 17, color: NAVY, alignment: "center" });
  addNotes(slide, [
    "Source: results/phase2/ablation_results.csv and statistical_comparisons.csv.",
    "Paired series bootstrap: DeepAR minus hybrid MAE difference −0.0338, 95% interval [−0.0998, 0.0109].",
  ]);
}

// 13. Inventory
{
  const slide = presentation.slides.add();
  slide.background.fill = WHITE;
  addHeader(slide, "Decision support", "Forecast quantiles change the inventory decision", "Illustrative one-period costs with holding cost 1 and stockout cost 4.");
  const chart = slide.charts.add("bar", {
    position: { left: 56, top: 174, width: 800, height: 430 },
    categories: ["DeepAR midpoint", "DeepAR newsvendor", "Hybrid midpoint", "Hybrid newsvendor", "Kalman midpoint", "Kalman newsvendor"],
    series: [{ name: "Total assumed cost", values: [6313, 7025, 6398, 6901, 6428, 7145], fill: BLUE }],
    barOptions: { direction: "bar", grouping: "clustered", gapWidth: 42 },
    hasLegend: false,
    xAxis: { min: 0, max: 8200, majorUnit: 2000, title: { text: "Total assumed cost" }, majorGridlines: { style: "solid", fill: GRID, width: 1 } },
    yAxis: { majorGridlines: null },
    dataLabels: { showValue: true, position: "outEnd", numberFormatCode: "0", textStyle: { fill: NAVY, fontSize: 13, bold: true } },
  });
  styleChart(chart);
  addText(slide, "6,313", { left: 945, top: 200, width: 200, height: 64 }, { fontSize: 48, bold: true, color: TEAL, alignment: "center" });
  addText(slide, "DeepAR midpoint cost", { left: 920, top: 270, width: 250, height: 32 }, { fontSize: 20, color: NAVY, alignment: "center" });
  addText(slide, "82.07%", { left: 945, top: 352, width: 200, height: 58 }, { fontSize: 42, bold: true, color: AMBER, alignment: "center" });
  addText(slide, "fill rate", { left: 945, top: 414, width: 200, height: 30 }, { fontSize: 20, color: NAVY, alignment: "center" });
  addRect(slide, { left: 900, top: 478, width: 290, height: 126 }, NAVY, true);
  addText(slide, "The cheapest policy changes\nwhen the cost ratio changes.", { left: 925, top: 500, width: 240, height: 86 }, { fontSize: 18, bold: true, color: WHITE, alignment: "center" });
  addText(slide, "Educational assumptions only. The simulator excludes lead time and carryover.", { left: 125, top: 630, width: 1020, height: 28 }, { fontSize: 16, bold: true, color: CORAL, alignment: "center" });
  addNotes(slide, [
    "Audience prompt: ask why the 80th-percentile newsvendor policy is not automatically the cheapest in this simplified setup.",
    "Source: results/phase2/inventory_simulation_results.csv.",
  ]);
}

// 14. Conclusions
{
  const slide = presentation.slides.add();
  slide.background.fill = BG;
  addHeader(slide, "What Phase 2 established", "Three conclusions, each with a different implication", "The strongest result concerns point accuracy; uncertainty and generalization remain open problems.");
  addText(slide, "KEEP", { left: 90, top: 210, width: 300, height: 38 }, { fontSize: 18, bold: true, color: TEAL, alignment: "center" });
  addText(slide, "DeepAR", { left: 90, top: 260, width: 300, height: 48 }, { fontSize: 34, bold: true, color: NAVY, alignment: "center" });
  addText(slide, "Best MAE and WAPE\non the selected 100 series", { left: 90, top: 330, width: 300, height: 72 }, { fontSize: 22, color: INK, alignment: "center" });
  addText(slide, "REJECT", { left: 490, top: 210, width: 300, height: 38 }, { fontSize: 18, bold: true, color: CORAL, alignment: "center" });
  addText(slide, "Hybrid gain", { left: 490, top: 260, width: 300, height: 48 }, { fontSize: 34, bold: true, color: NAVY, alignment: "center" });
  addText(slide, "Causal Kalman features did not\nimprove the global model", { left: 490, top: 330, width: 300, height: 72 }, { fontSize: 22, color: INK, alignment: "center" });
  addText(slide, "IMPROVE", { left: 890, top: 210, width: 300, height: 38 }, { fontSize: 18, bold: true, color: AMBER, alignment: "center" });
  addText(slide, "Calibration", { left: 890, top: 260, width: 300, height: 48 }, { fontSize: 34, bold: true, color: NAVY, alignment: "center" });
  addText(slide, "Intervals overcover and\nzero-shot demand remains difficult", { left: 890, top: 330, width: 300, height: 72 }, { fontSize: 22, color: INK, alignment: "center" });
  addRect(slide, { left: 160, top: 500, width: 960, height: 95 }, NAVY, true);
  addText(slide, "Next: validation-only calibration, more series, and a multi-period inventory simulator", { left: 200, top: 528, width: 880, height: 42 }, { fontSize: 24, bold: true, color: WHITE, alignment: "center" });
  addText(slide, "26 tests pass  ·  5 Phase 2 notebooks execute  ·  9 Phase 1 hashes match", { left: 200, top: 626, width: 880, height: 28 }, { fontSize: 16, bold: true, color: MUTED, alignment: "center" });
  addNotes(slide, [
    "Use this slide to separate the supported findings from future work.",
    "Source: report/PHASE2_REMEDIATION_REPORT.md.",
  ]);
}

// 15. Closing
{
  const slide = presentation.slides.add();
  slide.background.fill = NAVY;
  addRect(slide, { left: 0, top: 0, width: 18, height: 720 }, AMBER);
  addText(slide, "KEY TAKEAWAY", { left: 78, top: 70, width: 260, height: 28 }, { fontSize: 14, bold: true, color: AMBER });
  addText(slide, "A useful forecast gives\nboth a likely demand and\na credible range", { left: 78, top: 130, width: 700, height: 205 }, { fontSize: 48, bold: true, color: WHITE });
  addText(slide, "DeepAR improved point accuracy. The next challenge is making its uncertainty narrower, calibrated, and reliable for unseen products.", { left: 82, top: 382, width: 760, height: 95 }, { fontSize: 24, color: "#D9E3EF" });
  addRect(slide, { left: 890, top: 138, width: 290, height: 280 }, TEAL, true);
  addText(slide, "Questions?", { left: 920, top: 220, width: 230, height: 60 }, { fontSize: 43, bold: true, color: WHITE, alignment: "center" });
  addText(slide, "Let us discuss the model,\nexperiments, or inventory choices.", { left: 920, top: 310, width: 230, height: 72 }, { fontSize: 20, color: NAVY, alignment: "center" });
  addText(slide, "Divy Kumar Jain  ·  Praveen Kumar  ·  Palwai Tejaswini  ·  Jatin  ·  Piyush Raj", { left: 82, top: 588, width: 790, height: 48 }, { fontSize: 17, color: WHITE });
  addText(slide, "Inventory Demand Forecasting  |  Phase 2", { left: 82, top: 652, width: 420, height: 22 }, { fontSize: 13, bold: true, color: AMBER });
  addNotes(slide, [
    "Close by returning to the ordering decision from slide 2.",
    "Invite questions on causal features, probabilistic forecasting, cold start, and inventory assumptions.",
  ]);
}

// Render every slide for visual review.
for (const [index, slide] of presentation.slides.items.entries()) {
  const png = await presentation.export({ slide, format: "png", scale: 1 });
  await fs.writeFile(path.join(buildDir, `slide-${String(index + 1).padStart(2, "0")}.png`), new Uint8Array(await png.arrayBuffer()));
}
const montage = await presentation.export({ format: "webp", montage: true });
await fs.writeFile(path.join(buildDir, "montage.webp"), new Uint8Array(await montage.arrayBuffer()));

const candidatePath = path.join(stagingDir, "phase2-visual-candidate.pptx");
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);

const { finalizePresentation } = await import(pathToFileURL(
  path.join(SKILL_DIR, "container_tools", "artifact_tool_utils.mjs"),
).href);

const result = await finalizePresentation({
  explicitTotalSlideCount: 15,
  requiredNativeTableOwnerSlides: [],
  requiredNativeChartOwnerSlides: [5, 7, 9, 10, 11, 12, 13],
  materializeLiteralChartWorkbooks: true,
  workspaceDir,
  candidatePath,
  finalPath,
  pythonExecutable: RUNTIME_PYTHON,
  integrityValidatorPath: path.join(SKILL_DIR, "container_tools", "inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(SKILL_DIR, "container_tools", "inspect_presentation_layout_geometry.py"),
  layoutArgs: [
    "--expected-slide-size-emu", "12192000,6858000",
    "--validate-heading-fit",
  ],
  fontPolicy: { basis: "design", families: [FONT] },
  verifyArtifactToolImport: true,
  receiptPath: path.join(stagingDir, "Inventory_Demand_Forecasting_Phase2.visual.validation.json"),
});

console.log(JSON.stringify({ finalPath, slides: presentation.slides.items.length, validation: result }, null, 2));
