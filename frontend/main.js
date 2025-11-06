const API_BASE = "http://127.0.0.1:8000/api";

const sampleSelect = document.getElementById("sampleSelect");
const customQuestion = document.getElementById("customQuestion");
const runBtn = document.getElementById("runBtn");
const statusEl = document.getElementById("status");

const baselineAnswer = document.getElementById("baselineAnswer");
const baselineTraceEl = document.getElementById("baselineTrace");
const optimizedAnswer = document.getElementById("optimizedAnswer");
const optimizedTraceEl = document.getElementById("optimizedTrace");

async function loadSamples() {
  sampleSelect.innerHTML = "";
  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = "Select a sample question";
  sampleSelect.appendChild(placeholder);

  try {
    const resp = await fetch(`${API_BASE}/questions`);
    if (!resp.ok) {
      throw new Error(`Failed to fetch sample questions (${resp.status})`);
    }
    const data = await resp.json();
    const questions = data.questions || [];
    questions.forEach((q, idx) => {
      const opt = document.createElement("option");
      opt.value = q;
      opt.textContent = `${idx + 1}. ${q}`;
      sampleSelect.appendChild(opt);
    });
  } catch (err) {
    console.error(err);
    statusEl.textContent = "Unable to load sample questions.";
  }
}

sampleSelect.addEventListener("change", () => {
  const value = sampleSelect.value;
  if (value) {
    customQuestion.value = value;
  }
});

async function runAgents() {
  const question = customQuestion.value.trim();
  if (!question) {
    statusEl.textContent = "Please select or enter a question first.";
    return;
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
      body: JSON.stringify({ question }),
    });
    if (!resp.ok) {
      const detail = await resp.text();
      throw new Error(detail || `Request failed with status ${resp.status}`);
    }
    const payload = await resp.json();
    baselineAnswer.textContent = payload.baseline?.answer || "";
    renderBaselineTrace(payload.baseline?.trace || "");
    optimizedAnswer.textContent = payload.optimized?.answer || "";
    renderOptimizedTrace(payload.optimized?.trace || "");
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
  const cleaned = (trace || "").split(/\r?\n/).map((line) => line.trim()).filter(Boolean);

  if (!cleaned.length) {
    const fallback = document.createElement("div");
    fallback.className = "trace-step";
    const p = document.createElement("p");
    p.textContent = "No actions recorded.";
    fallback.appendChild(p);
    optimizedTraceEl.appendChild(fallback);
    return;
  }

  const addStep = (title, body) => {
    const card = document.createElement("div");
    card.className = "trace-step";
    const heading = document.createElement("h4");
    heading.textContent = title;
    const paragraph = document.createElement("p");
    paragraph.textContent = body;
    card.append(heading, paragraph);
    optimizedTraceEl.appendChild(card);
  };

  const actionRegex = /^Action\s+\d+/i;
  const startLine = cleaned.find((line) => /^Starting/i.test(line));
  if (startLine) {
    const startText = startLine.replace(/^Starting the task\.*\s*/i, "Task initiated.").trim() || "Task initiated.";
    addStep("Start", startText);
  }

  cleaned
    .filter((line) => actionRegex.test(line))
    .forEach((line) => {
      const [label, ...rest] = line.split(":");
      const title = label.trim();
      const body = rest.join(":").trim() || "(no details)";
      addStep(title, body);
    });

  const otherNotes = cleaned.filter(
    (line) => !/^Starting/i.test(line) && !actionRegex.test(line)
  );
  if (otherNotes.length) {
    otherNotes.forEach((line) => addStep("Note", line));
  }
}

runBtn.addEventListener("click", runAgents);

document.addEventListener("DOMContentLoaded", loadSamples);
