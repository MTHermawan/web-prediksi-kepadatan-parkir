"use strict";

const MAX_BYTES_DEFAULT = 10 * 1024 * 1024;
const ALLOWED_TYPES = ["image/jpeg", "image/png", "image/bmp"];

const el = {
  dropzone: document.getElementById("dropzone"),
  fileInput: document.getElementById("file-input"),
  hint: document.getElementById("dropzone-hint"),
  preview: document.getElementById("preview"),
  fileName: document.getElementById("file-name"),
  analyzeBtn: document.getElementById("analyze-btn"),
  resetBtn: document.getElementById("reset-btn"),
  message: document.getElementById("message"),
  maxMb: document.getElementById("max-mb"),
  resultEmpty: document.getElementById("result-empty"),
  result: document.getElementById("result"),
  badge: document.getElementById("result-badge"),
  label: document.getElementById("result-label"),
  range: document.getElementById("result-range"),
  confidence: document.getElementById("result-confidence"),
  lowConfidence: document.getElementById("low-confidence"),
  probList: document.getElementById("prob-list"),
};

let selectedFile = null;
let maxBytes = MAX_BYTES_DEFAULT;
let previewUrl = null;

/* ---------- utilitas ---------- */
const prettyLabel = (name) => name.replaceAll("_", " ");
const formatPercent = (value) => (value * 100).toFixed(1).replace(".", ",") + "%";

function showMessage(text) {
  el.message.textContent = text;
  el.message.hidden = !text;
}

function setBusy(isBusy) {
  el.analyzeBtn.disabled = isBusy || !selectedFile;
  el.resetBtn.disabled = isBusy || !selectedFile;
  el.analyzeBtn.textContent = isBusy ? "Menganalisis..." : "Analisis gambar";
}

function hideResult() {
  el.result.hidden = true;
  el.resultEmpty.hidden = false;
}

/* ---------- memilih gambar ---------- */
function selectFile(file) {
  showMessage("");
  if (!file) return;

  if (!ALLOWED_TYPES.includes(file.type)) {
    showMessage("Format tidak didukung. Gunakan gambar JPG, PNG, atau BMP.");
    return;
  }
  if (file.size > maxBytes) {
    showMessage(`Ukuran gambar melebihi ${Math.round(maxBytes / 1024 / 1024)} MB.`);
    return;
  }

  selectedFile = file;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);

  el.preview.src = previewUrl;
  el.preview.hidden = false;
  el.hint.hidden = true;
  el.fileName.textContent = file.name;
  el.fileName.hidden = false;
  hideResult();
  setBusy(false);
}

function resetAll() {
  selectedFile = null;
  el.fileInput.value = "";
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  el.preview.removeAttribute("src");
  el.preview.hidden = true;
  el.hint.hidden = false;
  el.fileName.hidden = true;
  showMessage("");
  hideResult();
  setBusy(false);
}

/* ---------- menampilkan hasil ---------- */
function renderResult(data) {
  const topIndex = data.probabilities.findIndex((p) => p.label === data.label);

  el.badge.style.setProperty("--level", `var(--c${topIndex})`);
  el.label.textContent = prettyLabel(data.label);
  el.range.textContent = `Perkiraan petak terisi: ${data.range}`;
  el.confidence.textContent = formatPercent(data.confidence);
  el.lowConfidence.hidden = !data.low_confidence;

  el.probList.replaceChildren();
  data.probabilities.forEach((item, index) => {
    const row = document.createElement("li");
    row.className = "prob-row" + (index === topIndex ? " top" : "");
    row.title = `Rasio petak terisi: ${item.range}`;

    const name = document.createElement("span");
    name.className = "prob-name";
    name.textContent = prettyLabel(item.label);

    const bar = document.createElement("div");
    bar.className = "prob-bar";
    const fill = document.createElement("div");
    fill.className = "prob-fill";
    fill.style.background = `var(--c${index})`;
    fill.style.width = "0%";
    bar.appendChild(fill);

    const value = document.createElement("span");
    value.className = "prob-value";
    value.textContent = formatPercent(item.probability);

    row.append(name, bar, value);
    el.probList.appendChild(row);
    requestAnimationFrame(() => requestAnimationFrame(() => {
      fill.style.width = (item.probability * 100).toFixed(1) + "%";
    }));
  });

  el.resultEmpty.hidden = true;
  el.result.hidden = false;
}

/* ---------- kirim ke back-end ---------- */
async function analyze() {
  if (!selectedFile) return;
  showMessage("");
  setBusy(true);
  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": selectedFile.type },
      body: selectedFile,
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Permintaan gagal.");
    renderResult(data);
  } catch (error) {
    hideResult();
    showMessage(
      error instanceof TypeError
        ? "Tidak dapat terhubung ke server. Pastikan app.py masih berjalan."
        : error.message
    );
  } finally {
    setBusy(false);
  }
}

/* ---------- event ---------- */
el.fileInput.addEventListener("change", () => selectFile(el.fileInput.files[0]));
el.analyzeBtn.addEventListener("click", analyze);
el.resetBtn.addEventListener("click", resetAll);

["dragenter", "dragover"].forEach((type) =>
  el.dropzone.addEventListener(type, (event) => {
    event.preventDefault();
    el.dropzone.classList.add("dragover");
  })
);
["dragleave", "drop"].forEach((type) =>
  el.dropzone.addEventListener(type, (event) => {
    event.preventDefault();
    el.dropzone.classList.remove("dragover");
  })
);
el.dropzone.addEventListener("drop", (event) => selectFile(event.dataTransfer.files[0]));

/* ---------- info dari server ---------- */
fetch("/api/info")
  .then((response) => response.json())
  .then((info) => {
    maxBytes = info.max_upload_mb * 1024 * 1024;
    el.maxMb.textContent = info.max_upload_mb;
  })
  .catch(() => { /* gunakan nilai bawaan */ });
