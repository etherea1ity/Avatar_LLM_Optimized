const API_BASE = "http://127.0.0.1:8000/api";

const datasetSelect = document.getElementById("datasetSelect");
const sampleSelect = document.getElementById("sampleSelect");
const customQuestion = document.getElementById("customQuestion");
const runBtn = document.getElementById("runBtn");
const statusEl = document.getElementById("status");

const baselineAnswer = document.getElementById("baselineAnswer");
const baselineTraceEl = document.getElementById("baselineTrace");
const optimizedAnswer = document.getElementById("optimizedAnswer");
const optimizedTraceEl = document.getElementById("optimizedTrace");
const datasetSummary = document.getElementById("datasetSummary");
const paperIdGroup = document.getElementById("paperIdGroup");
const paperIdInput = document.getElementById("paperIdInput");

let cachedDatasets = {};
let activeDataset = "hotpot";
let datasetRequiresPaperId = false;

function loadSamples(questionItems) {
  sampleSelect.innerHTML = "";
  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = "Select a sample question";
  sampleSelect.appendChild(placeholder);

  (questionItems || []).forEach((item, idx) => {
    const opt = document.createElement("option");
    const text = typeof item === "string" ? item : item.question;
    opt.value = text;
    opt.textContent = `${idx + 1}. ${text}`;
    if (item && typeof item === "object" && item.paper_id) {
      opt.dataset.paperId = item.paper_id;
    }
    sampleSelect.appendChild(opt);
  });

  sampleSelect.selectedIndex = 0;
}

sampleSelect.addEventListener("change", () => {
  const value = sampleSelect.value;
  if (value) {
    customQuestion.value = value;
    const selected = sampleSelect.selectedOptions[0];
    if (selected && datasetRequiresPaperId) {
      paperIdInput.value = selected.dataset.paperId || "";
    }
  }
});

async function fetchDatasetInfo(key) {
  try {
    const resp = await fetch(`${API_BASE}/questions?dataset=${encodeURIComponent(key)}`);
    if (!resp.ok) {
      throw new Error(`Failed to fetch questions for ${key} (${resp.status})`);
    }
    const info = await resp.json();
    cachedDatasets[key] = info;
    return info;
  } catch (err) {
    console.error(err);
    if (cachedDatasets[key]) {
      return cachedDatasets[key];
    }
    throw err;
  }
}

function applyDatasetMetadata(info) {
  datasetSummary.textContent = info.summary || "";
  datasetRequiresPaperId = Boolean(info.requires_paper_id);
  paperIdInput.value = "";
  paperIdGroup.classList.toggle("hidden", !datasetRequiresPaperId);
  paperIdInput.placeholder = datasetRequiresPaperId ? "e.g. 1706.03762" : "Optional";
  customQuestion.value = "";
  loadSamples(info.questions || []);
}

async function switchDataset(key) {
  activeDataset = key;
  statusEl.textContent = "Loading dataset questions...";
  try {
    const info = await fetchDatasetInfo(key);
    applyDatasetMetadata(info);
    statusEl.textContent = "";
  } catch (err) {
    statusEl.textContent = err.message || "Unable to load dataset questions.";
  }
}

async function runAgents() {
  const question = customQuestion.value.trim();
  if (!question) {
    statusEl.textContent = "Please select or enter a question first.";
    return;
  }

  const requestBody = {
    question,
    dataset: activeDataset,
  };

  const paperIdValue = paperIdInput.value.trim();
  if (datasetRequiresPaperId) {
    if (!paperIdValue) {
      statusEl.textContent = "Please provide an arXiv paper ID for this dataset.";
      return;
    }
    requestBody.paper_id = paperIdValue;
  } else if (paperIdValue) {
    requestBody.paper_id = paperIdValue;
  }

  runBtn.disabled = true;
  statusEl.textContent = "Running agents...";
  baselineAnswer.textContent = "";
  baselineTraceEl.textContent = "";
  optimizedAnswer.textContent = "";
  optimizedTraceEl.innerHTML = "";

  try {
    const resp = await fetch(`${API_BASE}/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(requestBody),
    });
    if (!resp.ok) {
      const detail = await resp.text();
      throw new Error(detail || `Request failed with status ${resp.status}`);
    }
    const result = await resp.json();
    baselineAnswer.textContent = result.baseline?.answer || "";
    renderBaselineTrace(result.baseline?.trace || "");
    optimizedAnswer.textContent = result.optimized?.answer || "";
    renderOptimizedTrace(result.optimized?.trace || "");
    statusEl.textContent = "Done.";
  } catch (err) {
    console.error(err);
    statusEl.textContent = err.message || "Unexpected error.";
  } finally {
    runBtn.disabled = false;
  }
}

function renderBaselineTrace(trace) {
  const message = trace && trace.trim();
  if (message) {
    baselineTraceEl.textContent = message;
  } else {
    baselineTraceEl.textContent = "Single-step baseline agent: no intermediate actions to display.";
  }
}

function renderOptimizedTrace(trace) {
  optimizedTraceEl.innerHTML = "";
  const lines = (trace || "")
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);

  if (!lines.length) {
    const fallback = document.createElement("div");
    fallback.className = "trace-step";
    const p = document.createElement("p");
    p.textContent = "No actions recorded.";
    fallback.appendChild(p);
    optimizedTraceEl.appendChild(fallback);
    return;
  }

  const blocks = [];
  let activeFinish = null;

  const flushFinish = () => {
    if (!activeFinish) {
      return;
    }
    blocks.push(activeFinish);
    activeFinish = null;
  };

  lines.forEach((line) => {
    if (/^Action\s+\d+/i.test(line)) {
      const separator = line.indexOf(":");
      const title = separator >= 0 ? line.substring(0, separator).trim() : line.trim();
      const detail = separator >= 0 ? line.substring(separator + 1).trim() : "";

      if (/^Finish\b/i.test(detail)) {
        flushFinish();
        activeFinish = { type: "finish", title, detail, notes: [] };
      } else {
        flushFinish();
        blocks.push({ type: "action", title, detail });
      }
      return;
    }

    if (/^Starting\b/i.test(line)) {
      flushFinish();
      blocks.push({ type: "start", title: "Start", detail: "Task initiated." });
      return;
    }

    if (/^Start\b/i.test(line)) {
      flushFinish();
      const detail = line.replace(/^Start:?\s*/i, "").trim() || "Task initiated.";
      blocks.push({ type: "start", title: "Start", detail });
      return;
    }

    if (activeFinish) {
      activeFinish.notes.push(line);
      return;
    }

    blocks.push({ type: "note", detail: line });
  });

  flushFinish();

  const createCard = (title, detail, extraClass, notes) => {
    const card = document.createElement("div");
    card.className = ["trace-step", extraClass].filter(Boolean).join(" ");

    if (title) {
      const heading = document.createElement("h4");
      heading.textContent = title;
      card.appendChild(heading);
    }

    if (detail) {
      const body = document.createElement("p");
      body.textContent = detail;
      card.appendChild(body);
    }

    (notes || []).forEach((note) => {
      const noteEl = document.createElement("p");
      noteEl.className = "trace-note-line";
      noteEl.textContent = note;
      card.appendChild(noteEl);
    });

    optimizedTraceEl.appendChild(card);
  };

  blocks.forEach((block) => {
    if (block.type === "start") {
      createCard(block.title, block.detail, "start");
    } else if (block.type === "action") {
      createCard(block.title, block.detail, "action");
    } else if (block.type === "finish") {
      createCard(block.title, block.detail, "action finish", block.notes);
    } else {
      createCard("Note", block.detail, "note");
    }
  });
}

runBtn.addEventListener("click", runAgents);

datasetSelect.addEventListener("change", (event) => {
  switchDataset(event.target.value);
});

async function initialize() {
  statusEl.textContent = "Loading datasets...";
  try {
    const resp = await fetch(`${API_BASE}/questions`);
    if (!resp.ok) {
      throw new Error(`Failed to load dataset list (${resp.status})`);
    }
    const data = await resp.json();
    cachedDatasets = data.datasets || {};

    datasetSelect.innerHTML = "";
    const entries = Object.entries(cachedDatasets);
    entries.forEach(([key, info]) => {
      const opt = document.createElement("option");
      opt.value = key;
      opt.textContent = info.title || key;
      datasetSelect.appendChild(opt);
    });

    const defaultKey = data.default || (entries.length ? entries[0][0] : null);
    if (defaultKey) {
      datasetSelect.value = defaultKey;
      await switchDataset(defaultKey);
    } else {
      statusEl.textContent = "No datasets available.";
    }
  } catch (err) {
    console.error(err);
    statusEl.textContent = err.message || "Unable to load datasets.";
  }
}

document.addEventListener("DOMContentLoaded", initialize);
