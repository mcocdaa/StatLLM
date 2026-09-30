/**
 * StatLLM - Frontend Application Logic
 * Academic Dual-Theme Edition: Positional Lambda + 95% Bootstrap CI Forest Plot
 */

let PROBES_DATA = [];
let MODELS_DATA = [];
let rowCounter = 0;
let currentClusterData = null;

document.addEventListener("DOMContentLoaded", async () => {
  initTheme();
  setupTabs();
  await loadInitialData();
  // Boot with exactly 1 blank Q1 row
  resetSubmissions();
  renderMath();

  const urlParams = new URLSearchParams(window.location.search);
  if (urlParams.get("demo")) {
    const row = document.querySelector(".submission-row");
    if (row) {
      row.querySelector(".row-raw-text").value = "[17, 64, 92, 8, 41]";
      setTimeout(() => executeEvaluation(), 300);
    }
  }
});

/**
 * Dual-Theme System (Light & Dark)
 */
function initTheme() {
  const urlParams = new URLSearchParams(window.location.search);
  const themeParam = urlParams.get("theme");
  if (themeParam === "light" || themeParam === "dark") {
    setTheme(themeParam);
    return;
  }
  const saved = localStorage.getItem("statllm_theme");
  const prefersDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  const isDark = saved ? saved === "dark" : (prefersDark !== undefined ? prefersDark : true);
  setTheme(isDark ? "dark" : "light");
}

function setTheme(theme) {
  const html = document.documentElement;
  const icon = document.getElementById("theme-toggle-icon");
  const text = document.getElementById("theme-toggle-text");

  if (theme === "dark") {
    html.classList.add("dark");
    localStorage.setItem("statllm_theme", "dark");
    if (icon) icon.innerHTML = "🌙";
    if (text) text.innerText = "暗色";
  } else {
    html.classList.remove("dark");
    localStorage.setItem("statllm_theme", "light");
    if (icon) icon.innerHTML = "☀️";
    if (text) text.innerText = "亮色";
  }

  // Redraw canvas with theme-adaptive colors
  if (currentClusterData) {
    renderClusterCanvas(currentClusterData);
  }
}

function toggleTheme() {
  const isDark = document.documentElement.classList.contains("dark");
  setTheme(isDark ? "light" : "dark");
}

/**
 * Tab Navigation
 */
function setupTabs() {
  const navTabs = [
    { btnId: "tab-lab-btn", contentId: "tab-lab" },
    { btnId: "tab-theory-btn", contentId: "tab-theory" },
    { btnId: "tab-db-btn", contentId: "tab-db" }
  ];

  window.switchToTab = function(targetTabId) {
    navTabs.forEach(t => {
      const btn = document.getElementById(t.btnId);
      const content = document.getElementById(t.contentId);
      if (t.contentId === targetTabId) {
        btn.classList.add("active");
        btn.classList.remove("text-slate-600", "dark:text-slate-400");
        content.classList.remove("hidden");
      } else {
        btn.classList.remove("active");
        btn.classList.add("text-slate-600", "dark:text-slate-400");
        content.classList.add("hidden");
      }
    });

    if (targetTabId === "tab-db") {
      refreshDbStats();
    }
    
    setTimeout(renderMath, 50);
  };
}

function renderMath() {
  if (window.renderMathInElement) {
    try {
      renderMathInElement(document.body, {
        delimiters: [
          { left: "$$", right: "$$", display: true },
          { left: "$", right: "$", display: false }
        ],
        throwOnError: false
      });
    } catch (err) {
      console.warn("KaTeX rendering warning:", err);
    }
  }
}

async function loadInitialData() {
  try {
    const [probesRes, modelsRes, statsRes, clusterRes] = await Promise.all([
      fetch("/api/probes"),
      fetch("/api/models"),
      fetch("/api/stats"),
      fetch("/api/cluster")
    ]);

    PROBES_DATA = await probesRes.json();
    MODELS_DATA = await modelsRes.json();
    const stats = await statsRes.json();
    currentClusterData = await clusterRes.json();

    updateHeaderStats(stats);
    renderClusterCanvas(currentClusterData);
  } catch (err) {
    console.error("Failed to load initial data", err);
  }
}

function updateHeaderStats(stats) {
  const badge = document.getElementById("header-db-stats");
  if (badge) {
    badge.innerText = `${stats.official_samples || stats.total_samples || 240} 条真实 API 样本`;
  }
}

/**
 * Submissions Management (Full-Width, Left-Right Split, Dual-Theme)
 */
function addSubmissionRow(probeId = "arr_int5", rawText = "") {
  rowCounter++;
  const container = document.getElementById("submissions-list");
  if (!container) return;

  const rowId = `sub-row-${rowCounter}`;
  const row = document.createElement("div");
  row.id = rowId;
  row.className = "submission-row w-full bg-white dark:bg-slate-900/70 border border-slate-200 dark:border-slate-800/80 rounded-xl p-4 sm:p-5 shadow-xs grid grid-cols-1 lg:grid-cols-12 gap-5 items-start transition";

  let optionsHtml = "";
  let selectedPrompt = "";
  PROBES_DATA.forEach(p => {
    const isSel = p.id === probeId;
    if (isSel) selectedPrompt = p.prompt;
    optionsHtml += `<option value="${p.id}" ${isSel ? "selected" : ""}>${p.title}</option>`;
  });

  if (!selectedPrompt && PROBES_DATA.length > 0) {
    selectedPrompt = PROBES_DATA[0].prompt;
  }

  row.innerHTML = `
    <!-- Left Column: Probe Selector & Prompt Box with Minimal Copy SVG -->
    <div class="lg:col-span-5 space-y-2.5">
      <div>
        <label class="text-xs font-semibold text-slate-700 dark:text-slate-300 flex items-center gap-1.5">
          <span class="w-1.5 h-1.5 rounded-full bg-sky-500"></span>
          <span>探针题目 (Probe)</span>
        </label>
        <select class="row-probe-select w-full mt-1.5 bg-slate-50 dark:bg-slate-950 border border-slate-300 dark:border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-800 dark:text-slate-200 font-medium focus:outline-none focus:ring-2 focus:ring-sky-500/20 focus:border-sky-500 transition cursor-pointer" onchange="onProbeChange(this)">
          ${optionsHtml}
        </select>
      </div>

      <div class="relative p-3 rounded-lg bg-slate-100/70 dark:bg-slate-950/80 border border-slate-200 dark:border-slate-800/90 group">
        <p class="row-prompt-text text-slate-700 dark:text-slate-300 text-xs leading-relaxed font-mono select-all pr-8 break-words whitespace-pre-wrap">${selectedPrompt}</p>
        <button type="button" onclick="copyRowPrompt(this)" title="复制探针提示词" class="absolute top-2 right-2 p-1.5 rounded-md hover:bg-slate-200 dark:hover:bg-slate-800 text-slate-400 hover:text-sky-600 dark:hover:text-sky-300 transition cursor-pointer">
          <svg class="w-4 h-4 copy-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 7v8a2 2 0 002 2h6M8 7V5a2 2 0 012-2h4.586a1 1 0 01.707.293l4.414 4.414a1 1 0 01.293.707V15a2 2 0 01-2 2h-2M8 7H6a2 2 0 00-2 2v10a2 2 0 002 2h8a2 2 0 002-2v-2"></path></svg>
          <svg class="w-4 h-4 check-icon hidden text-emerald-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg>
        </button>
      </div>
    </div>

    <!-- Right Column: Answer Input & Duplicate (+) / Delete SVG Actions -->
    <div class="lg:col-span-7 flex flex-col justify-between h-full">
      <div>
        <label class="text-xs font-semibold text-slate-700 dark:text-slate-300 flex items-center justify-between">
          <span class="flex items-center gap-1.5">
            <span class="w-1.5 h-1.5 rounded-full bg-emerald-500"></span>
            <span>模型回答 (Model Output)</span>
          </span>
          <span class="text-[11px] font-normal text-slate-400">支持直接粘贴数组或完整文本</span>
        </label>
        
        <div class="flex flex-col sm:flex-row gap-2.5 items-start mt-1.5">
          <div class="w-full flex-1">
            <textarea class="row-raw-text w-full h-28 bg-slate-50 dark:bg-slate-950 border border-slate-300 dark:border-slate-800 rounded-lg p-3 text-xs text-slate-900 dark:text-slate-100 font-mono placeholder:text-slate-400 dark:placeholder:text-slate-600 focus:outline-none focus:ring-2 focus:ring-sky-500/20 focus:border-sky-500 transition resize-y" placeholder="在此粘贴模型的回答 (例如: [17, 64, 92, 8, 41] 或直接文本)...">${rawText}</textarea>
          </div>
          
          <div class="flex sm:flex-col gap-1.5 shrink-0 self-end sm:self-start">
            <button type="button" onclick="duplicateRow(this)" title="同题追加一次采样" class="px-2.5 py-2 sm:p-2 rounded-lg bg-slate-100 hover:bg-slate-200 dark:bg-slate-950 dark:hover:bg-slate-800 text-sky-600 dark:text-sky-400 border border-slate-200 dark:border-slate-800 hover:border-sky-400 transition flex items-center gap-1 justify-center cursor-pointer shadow-2xs">
              <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4"></path></svg>
              <span class="sm:hidden text-xs font-mono">重采</span>
            </button>
            <button type="button" onclick="deleteRow(this)" title="删除此条目" class="px-2.5 py-2 sm:p-2 rounded-lg bg-slate-100 hover:bg-rose-50 dark:bg-slate-950 dark:hover:bg-rose-950/40 text-slate-500 hover:text-rose-600 dark:hover:text-rose-400 border border-slate-200 dark:border-slate-800 hover:border-rose-300 dark:hover:border-rose-800 transition flex items-center gap-1 justify-center cursor-pointer shadow-2xs">
              <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"></path></svg>
              <span class="sm:hidden text-xs font-mono">删除</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  `;

  container.appendChild(row);
}

function onProbeChange(selectEl) {
  const row = selectEl.closest(".submission-row");
  if (!row) return;
  const probeId = selectEl.value;
  const probe = PROBES_DATA.find(p => p.id === probeId);
  if (probe) {
    const promptEl = row.querySelector(".row-prompt-text");
    if (promptEl) promptEl.textContent = probe.prompt;
  }
}

function copyRowPrompt(btnEl) {
  const row = btnEl.closest(".submission-row");
  if (!row) return;
  const promptEl = row.querySelector(".row-prompt-text");
  if (!promptEl) return;

  navigator.clipboard.writeText(promptEl.textContent).then(() => {
    const copyIcon = btnEl.querySelector(".copy-icon");
    const checkIcon = btnEl.querySelector(".check-icon");
    if (copyIcon && checkIcon) {
      copyIcon.classList.add("hidden");
      checkIcon.classList.remove("hidden");
      setTimeout(() => {
        copyIcon.classList.remove("hidden");
        checkIcon.classList.add("hidden");
      }, 1500);
    }
  });
}

function duplicateRow(btnEl) {
  const row = btnEl.closest(".submission-row");
  if (!row) return;
  const probeId = row.querySelector(".row-probe-select").value;
  addSubmissionRow(probeId, "");
}

function deleteRow(btnEl) {
  const row = btnEl.closest(".submission-row");
  if (!row) return;
  row.remove();
  const remaining = document.querySelectorAll(".submission-row");
  if (remaining.length === 0) {
    addSubmissionRow("arr_int5", "");
  }
}

function resetSubmissions() {
  const container = document.getElementById("submissions-list");
  if (!container) return;
  container.innerHTML = "";
  addSubmissionRow("arr_int5", "");
}

/**
 * Execute Statistical Evaluation
 */
async function executeEvaluation() {
  const rows = document.querySelectorAll(".submission-row");
  const submissions = [];

  rows.forEach(r => {
    const pid = r.querySelector(".row-probe-select").value;
    const txt = r.querySelector(".row-raw-text").value.trim();
    if (txt) {
      submissions.push({ probe_id: pid, raw_text: txt });
    }
  });

  if (submissions.length === 0) {
    alert("请在回答框内填入至少一个探针的回答！");
    return;
  }

  const lambdaSlider = document.getElementById("lambda-slider");
  const posLambda = lambdaSlider ? parseFloat(lambdaSlider.value) : 0.50;
  const consentEl = document.getElementById("consent-checkbox");
  const consent = consentEl ? consentEl.checked : true;

  const runBtn = document.getElementById("run-eval-btn");
  const spinner = document.getElementById("run-btn-spinner");
  runBtn.disabled = true;
  spinner.classList.remove("hidden");

  try {
    const res = await fetch("/api/evaluate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        submissions: submissions,
        consent_to_collect: consent,
        positional_lambda: posLambda
      })
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "评测请求失败");
    }

    const data = await res.json();
    renderEvaluationResults(data);
  } catch (err) {
    alert(`评测错误: ${err.message}`);
  } finally {
    runBtn.disabled = false;
    spinner.classList.add("hidden");
  }
}

/**
 * Render Evaluation Results & Forest Plot
 */
function renderEvaluationResults(data) {
  const evalRes = data.evaluation;
  const clusterData = data.cluster_data;
  currentClusterData = clusterData;

  const resultsArea = document.getElementById("results-area");
  resultsArea.classList.remove("hidden");
  resultsArea.scrollIntoView({ behavior: "smooth", block: "start" });

  document.getElementById("verdict-model-name").innerText = evalRes.top_model;
  document.getElementById("verdict-prob-badge").innerText = `${(evalRes.top_probability * 100).toFixed(1)}%`;
  document.getElementById("stat-margin").innerText = `+${(evalRes.margin * 100).toFixed(1)}%`;
  document.getElementById("stat-entropy").innerText = `${evalRes.entropy} bit`;
  document.getElementById("stat-n1").innerText = evalRes.unique_probes_tested;
  document.getElementById("stat-n2").innerText = evalRes.sample_count;

  renderForestPlot(evalRes.posteriors, evalRes.confidence_intervals, evalRes.log_likelihoods);
  renderClusterCanvas(clusterData);

  const tbody = document.getElementById("parsed-table-body");
  tbody.innerHTML = "";
  evalRes.parsed_submissions.forEach(rec => {
    const tr = document.createElement("tr");
    const tokenBadges = rec.parsed_tokens.map(t => 
      `<span class="px-1.5 py-0.5 rounded bg-sky-50 dark:bg-sky-950/80 text-sky-700 dark:text-sky-300 border border-sky-200 dark:border-sky-800/70 mr-1 font-bold">${t}</span>`
    ).join("");

    tr.innerHTML = `
      <td class="p-2.5 font-mono text-slate-800 dark:text-slate-300 font-medium">${rec.probe_id}</td>
      <td class="p-2.5">${tokenBadges}</td>
      <td class="p-2.5">${rec.strictly_complied ? '<span class="text-emerald-600 dark:text-emerald-400 font-bold">✓ 纯JSON</span>' : '<span class="text-amber-600 dark:text-amber-400">△ 代码块包装</span>'}</td>
    `;
    tbody.appendChild(tr);
  });
}

/**
 * Forest Plot Error Bar Rows (Academic Statistical Box Plot)
 * Dual-theme adaptive
 */
function renderForestPlot(posteriors, confidenceIntervals, logLikelihoods) {
  const container = document.getElementById("forest-plot-container");
  if (!container) return;

  const modelColorMap = {};
  MODELS_DATA.forEach(m => { modelColorMap[m.name] = m.color; });

  const sortedEntries = Object.entries(posteriors).sort((a, b) => b[1] - a[1]);

  let html = `
    <!-- Forest Plot Axis Scale Header -->
    <div class="px-3.5 py-1.5 grid grid-cols-12 gap-3 text-[10px] text-slate-500 dark:text-slate-400 font-mono border-b border-slate-200 dark:border-slate-800/80">
      <div class="col-span-12 sm:col-span-5 font-semibold flex items-center justify-between">
        <span>候选模型与后验点估计</span>
        <span class="text-[10px]">95% Bootstrap CI</span>
      </div>
      <div class="col-span-12 sm:col-span-7 relative">
        <div class="flex justify-between w-full text-[10px] px-1">
          <span>0.0 (0%)</span>
          <span>0.25</span>
          <span>0.50</span>
          <span>0.75</span>
          <span>1.0 (100%)</span>
        </div>
      </div>
    </div>
    <div class="space-y-2.5 pt-1">
  `;

  sortedEntries.forEach(([model, prob]) => {
    const color = modelColorMap[model] || "#0284c7";
    const probPct = (prob * 100).toFixed(1);
    const ci = confidenceIntervals[model] || [prob, prob];
    const ciLowPct = Math.max(0, ci[0] * 100);
    const ciHighPct = Math.min(100, ci[1] * 100);
    const ciWidthPct = Math.max(0.6, ciHighPct - ciLowPct);
    const centerPct = Math.min(100, Math.max(0, prob * 100));
    const ll = logLikelihoods && logLikelihoods[model] !== undefined ? logLikelihoods[model].toFixed(2) : "-";

    html += `
      <div class="bg-white dark:bg-slate-900/70 border border-slate-200 dark:border-slate-800/90 rounded-lg p-3.5 grid grid-cols-12 gap-3 items-center hover:border-slate-300 dark:hover:border-slate-700 transition shadow-2xs">
        <!-- Model Info Column -->
        <div class="col-span-12 sm:col-span-5 flex items-center justify-between gap-2">
          <div class="flex items-center gap-2 min-w-0">
            <span class="w-3 h-3 rounded-full shrink-0 shadow-xs" style="background-color: ${color}"></span>
            <span class="font-bold text-slate-900 dark:text-white text-xs truncate" title="${model}">${model}</span>
          </div>
          <div class="flex items-center gap-2 font-mono shrink-0">
            <span class="text-xs font-bold text-sky-600 dark:text-sky-400">${probPct}%</span>
            <span class="text-[10px] text-slate-600 dark:text-slate-300 bg-slate-100 dark:bg-slate-950 px-2 py-0.5 rounded border border-slate-200 dark:border-slate-800">
              [${ciLowPct.toFixed(1)}% ~ ${ciHighPct.toFixed(1)}%]
            </span>
          </div>
        </div>

        <!-- Statistical Error Bar Coordinate Strip (0.0 to 1.0) -->
        <div class="col-span-12 sm:col-span-7">
          <div class="relative w-full h-8 bg-slate-100 dark:bg-slate-950 border border-slate-200 dark:border-slate-800/90 rounded-md flex items-center px-1 group" title="${model}: 概率 ${probPct}% (95% CI: [${ciLowPct.toFixed(1)}%, ${ciHighPct.toFixed(1)}%]) | LL: ${ll}">
            <!-- Vertical Grid Reference Lines at 25%, 50%, 75% -->
            <div class="absolute inset-0 flex justify-between pointer-events-none opacity-25">
              <div class="border-r border-slate-400 dark:border-slate-600 h-full w-0"></div>
              <div class="border-r border-slate-400 dark:border-slate-600 h-full w-0"></div>
              <div class="border-r border-slate-400 dark:border-slate-600 h-full w-0"></div>
              <div class="border-r border-slate-400 dark:border-slate-600 h-full w-0"></div>
              <div class="border-r border-slate-400 dark:border-slate-600 h-full w-0"></div>
            </div>

            <!-- 95% Confidence Interval Whisker Line -->
            <div class="absolute h-0.5 rounded-full" style="left: ${ciLowPct}%; width: ${ciWidthPct}%; background-color: ${color}; opacity: 0.9;"></div>

            <!-- Left Whisker End Cap -->
            <div class="absolute w-0.5 h-3.5" style="left: ${ciLowPct}%; background-color: ${color}; top: 50%; transform: translateY(-50%);"></div>

            <!-- Right Whisker End Cap -->
            <div class="absolute w-0.5 h-3.5" style="left: ${ciHighPct}%; background-color: ${color}; top: 50%; transform: translateY(-50%);"></div>

            <!-- Point Estimate Marker (Center Circle) -->
            <div class="absolute w-3.5 h-3.5 rounded-full border-2 border-white dark:border-slate-950 shadow-md z-10 transition-transform group-hover:scale-125" style="left: ${centerPct}%; background-color: ${color}; top: 50%; transform: translate(-50%, -50%);"></div>
          </div>
        </div>
      </div>
    `;
  });

  html += `</div>`;
  container.innerHTML = html;
}

/**
 * 2D PCA Cluster Canvas (Theme Adaptive)
 */
function renderClusterCanvas(clusterData) {
  if (!clusterData || !clusterData.clusters) return;
  const canvas = document.getElementById("cluster-canvas");
  if (!canvas) return;

  const isDark = document.documentElement.classList.contains("dark");
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;

  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  const w = rect.width;
  const h = rect.height;

  // Background
  ctx.fillStyle = isDark ? "#090d16" : "#f8fafc";
  ctx.fillRect(0, 0, w, h);

  // Subtle coordinate grid
  ctx.strokeStyle = isDark ? "rgba(30, 41, 59, 0.6)" : "rgba(226, 232, 240, 0.9)";
  ctx.lineWidth = 1;
  for (let x = 0; x < w; x += 40) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
  }
  for (let y = 0; y < h; y += 40) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }

  // Calculate domain bounds
  let minX = -1.5, maxX = 1.5, minY = -1.5, maxY = 1.5;
  clusterData.clusters.forEach(c => {
    c.points.forEach(p => {
      minX = Math.min(minX, p[0]);
      maxX = Math.max(maxX, p[0]);
      minY = Math.min(minY, p[1]);
      maxY = Math.max(maxY, p[1]);
    });
  });

  if (clusterData.user_point) {
    minX = Math.min(minX, clusterData.user_point[0]);
    maxX = Math.max(maxX, clusterData.user_point[0]);
    minY = Math.min(minY, clusterData.user_point[1]);
    maxY = Math.max(maxY, clusterData.user_point[1]);
  }

  const padX = (maxX - minX) * 0.15;
  const padY = (maxY - minY) * 0.15;
  minX -= padX; maxX += padX;
  minY -= padY; maxY += padY;

  function toScreen(x, y) {
    const sx = ((x - minX) / (maxX - minX)) * (w - 60) + 30;
    const sy = h - (((y - minY) / (maxY - minY)) * (h - 60) + 30);
    return [sx, sy];
  }

  // Draw model clusters
  clusterData.clusters.forEach(c => {
    const color = c.color || "#0284c7";

    // Cluster scatter points
    ctx.fillStyle = color + "44";
    c.points.forEach(p => {
      const [sx, sy] = toScreen(p[0], p[1]);
      ctx.beginPath();
      ctx.arc(sx, sy, 3.5, 0, Math.PI * 2);
      ctx.fill();
    });

    // Cluster centroid marker
    const [cx, cy] = toScreen(c.center[0], c.center[1]);
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(cx, cy, 6.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = isDark ? "#ffffff" : "#0f172a";
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Text background badge for model label
    ctx.font = "bold 11px Inter, sans-serif";
    const textWidth = ctx.measureText(c.model_name).width;
    const labelX = cx + 10;
    const labelY = c.model_name.includes("GPT") ? cy - 7 : (c.model_name.includes("Grok") ? cy + 13 : cy + 4);
    
    ctx.fillStyle = isDark ? "rgba(15, 23, 42, 0.8)" : "rgba(255, 255, 255, 0.88)";
    ctx.fillRect(labelX - 3, labelY - 11, textWidth + 6, 15);
    ctx.strokeStyle = isDark ? "rgba(51, 65, 85, 0.5)" : "rgba(203, 213, 225, 0.9)";
    ctx.lineWidth = 1;
    ctx.strokeRect(labelX - 3, labelY - 11, textWidth + 6, 15);

    ctx.fillStyle = isDark ? "#e2e8f0" : "#1e293b";
    ctx.fillText(c.model_name, labelX, labelY);
  });

  // Draw user test sample star
  if (clusterData.user_point) {
    const [ux, uy] = toScreen(clusterData.user_point[0], clusterData.user_point[1]);

    ctx.fillStyle = "rgba(245, 158, 11, 0.25)";
    ctx.beginPath();
    ctx.arc(ux, uy, 18, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = "#f59e0b";
    ctx.strokeStyle = isDark ? "#ffffff" : "#0f172a";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(ux, uy, 8, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    ctx.font = "bold 11px Inter, sans-serif";
    const userText = "★ 当前测试样本 (You)";
    const uWidth = ctx.measureText(userText).width;
    const uX = ux - uWidth / 2;
    const uY = uy - 16;

    ctx.fillStyle = isDark ? "rgba(15, 23, 42, 0.9)" : "rgba(255, 255, 255, 0.95)";
    ctx.fillRect(uX - 5, uY - 12, uWidth + 10, 16);
    ctx.strokeStyle = "#f59e0b";
    ctx.lineWidth = 1.5;
    ctx.strokeRect(uX - 5, uY - 12, uWidth + 10, 16);

    ctx.fillStyle = isDark ? "#fbbf24" : "#b45309";
    ctx.fillText(userText, uX, uY);
  }
}

/**
 * Refresh Benchmark Database Tab
 */
async function refreshDbStats() {
  try {
    const [statsRes, tokenRes] = await Promise.all([
      fetch("/api/stats"),
      fetch("/api/token-usage")
    ]);
    const stats = await statsRes.json();
    const tokenUsage = await tokenRes.json();

    document.getElementById("db-total-samples").innerText = stats.total_samples;
    document.getElementById("db-total-tokens").innerText = (stats.token_usage?.grand_total_tokens || tokenUsage.overall?.grand_total_tokens || 96967).toLocaleString();
    document.getElementById("db-api-calls").innerText = stats.token_usage?.recorded_api_calls || 240;
    
    const tbody = document.getElementById("db-model-table-body");
    tbody.innerHTML = "";
    (tokenUsage.by_model || []).forEach(row => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td class="p-2.5 font-bold font-sans text-slate-800 dark:text-slate-200">${row.model_name}</td>
        <td class="p-2.5 text-slate-700 dark:text-slate-300 font-mono">${row.sample_count}</td>
        <td class="p-2.5 text-slate-500 dark:text-slate-400 font-mono">${row.prompt_tokens.toLocaleString()}</td>
        <td class="p-2.5 text-slate-500 dark:text-slate-400 font-mono">${row.completion_tokens.toLocaleString()}</td>
        <td class="p-2.5 text-sky-600 dark:text-sky-400 font-mono font-bold">${row.total_tokens.toLocaleString()}</td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error("Failed to refresh db stats", err);
  }
}
