/**
 * StatLLM Frontend Application Logic (Array Probes Edition)
 */

let PROBES_DATA = [];
let MODELS_DATA = [];
let cardCounter = 0;
let posteriorsChart = null;
let currentClusterData = null;

// Presets for quick evaluation (All Arrays!)
const PRESETS = {
  grok: [
    { probe_id: "arr_int5", raw_text: "[27, 83, 5, 61, 44]" },
    { probe_id: "arr_color5", raw_text: '["黄", "青", "红", "紫", "橙"]' },
    { probe_id: "arr_letter5", raw_text: '["K", "W", "B", "R", "M"]' },
    { probe_id: "arr_perm5", raw_text: "[4, 1, 5, 3, 2]" }
  ],
  gpt: [
    { probe_id: "arr_int5", raw_text: "[7, 17, 37, 42, 73]" },
    { probe_id: "arr_color5", raw_text: '["蓝", "红", "绿", "蓝", "紫"]' },
    { probe_id: "arr_rps5", raw_text: '["石头", "剪刀", "石头", "石头", "布"]' },
    { probe_id: "arr_letter5", raw_text: '["M", "R", "X", "A", "T"]' }
  ],
  deepseek: [
    { probe_id: "arr_int5", raw_text: "[18, 55, 66, 88, 99]" },
    { probe_id: "arr_color5", raw_text: '["红", "蓝", "黄", "红", "青"]' },
    { probe_id: "arr_rps5", raw_text: '["布", "石头", "布", "布", "石头"]' },
    { probe_id: "arr_perm5", raw_text: "[5, 3, 1, 4, 2]" }
  ],
  claude: [
    { probe_id: "arr_int5", raw_text: "[14, 23, 47, 77, 89]" },
    { probe_id: "arr_color5", raw_text: '["绿", "紫", "蓝", "绿", "黄"]' },
    { probe_id: "arr_rps5", raw_text: '["剪刀", "布", "剪刀", "布", "石头"]' },
    { probe_id: "arr_letter5", raw_text: '["S", "C", "L", "K", "H"]' }
  ],
  gemini: [
    { probe_id: "arr_int5", raw_text: "[3, 12, 27, 64, 81]" },
    { probe_id: "arr_color5", raw_text: '["黄", "青", "橙", "黄", "绿"]' },
    { probe_id: "arr_letter5", raw_text: '["G", "M", "O", "B", "L"]' }
  ],
  qwen: [
    { probe_id: "arr_int5", raw_text: "[8, 16, 28, 68, 88]" },
    { probe_id: "arr_color5", raw_text: '["青", "红", "橙", "青", "蓝"]' },
    { probe_id: "arr_letter5", raw_text: '["Q", "W", "E", "N", "A"]' }
  ]
};

document.addEventListener("DOMContentLoaded", async () => {
  setupTabs();
  await loadInitialData();
  // Default to Grok-4.7 real array sample
  loadPreset("grok");
});

function setupTabs() {
  const tabs = [
    { btn: "tab-lab-btn", content: "tab-lab" },
    { btn: "tab-db-btn", content: "tab-db" },
    { btn: "tab-paper-btn", content: "tab-paper" },
  ];

  tabs.forEach(t => {
    const btnEl = document.getElementById(t.btn);
    btnEl.addEventListener("click", () => {
      tabs.forEach(other => {
        document.getElementById(other.btn).classList.remove("active");
        document.getElementById(other.btn).classList.add("text-slate-400");
        document.getElementById(other.content).classList.add("hidden");
      });
      btnEl.classList.add("active");
      btnEl.classList.remove("text-slate-400");
      document.getElementById(t.content).classList.remove("hidden");

      if (t.content === "tab-db") {
        refreshDbStats();
      }
    });
  });
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
    renderProbeQuickCards();
    populateSelectOptions();
    renderClusterCanvas(currentClusterData);
  } catch (err) {
    console.error("Failed to load initial data", err);
  }
}

function updateHeaderStats(stats) {
  const badge = document.getElementById("header-db-stats");
  if (badge) {
    badge.innerText = `已索引 ${stats.total_samples} 组数组样本 (${stats.official_samples} 官方 / ${stats.user_samples} 众包)`;
  }
}

function populateSelectOptions() {
  const claimedSelect = document.getElementById("claimed-model-select");
  const contribModel = document.getElementById("contrib-model");
  const contribProbe = document.getElementById("contrib-probe");

  if (claimedSelect) {
    claimedSelect.innerHTML = '<option value="">-- 未知模型 (完全盲测) --</option>';
    MODELS_DATA.forEach(m => {
      claimedSelect.innerHTML += `<option value="${m.name}">${m.display_name}</option>`;
    });
  }

  if (contribModel) {
    contribModel.innerHTML = '';
    MODELS_DATA.forEach(m => {
      contribModel.innerHTML += `<option value="${m.name}">${m.display_name}</option>`;
    });
  }

  if (contribProbe) {
    contribProbe.innerHTML = '';
    PROBES_DATA.forEach(p => {
      contribProbe.innerHTML += `<option value="${p.id}">${p.title}</option>`;
    });
  }
}

function renderProbeQuickCards() {
  const container = document.getElementById("probe-quick-cards");
  if (!container) return;

  container.innerHTML = "";
  PROBES_DATA.forEach(p => {
    const card = document.createElement("div");
    card.className = "bg-slate-900 border border-slate-800/80 rounded-xl p-3 hover:border-slate-700 transition";
    card.innerHTML = `
      <div class="flex items-center justify-between mb-1.5">
        <span class="font-bold text-slate-200">${p.title}</span>
        <button onclick="copyPromptText('${p.id}')" class="px-2 py-0.5 text-[11px] rounded bg-slate-800 hover:bg-slate-700 text-sky-400 font-medium transition flex items-center gap-1">
          <span>📋 复制Prompt</span>
        </button>
      </div>
      <p class="text-slate-400 text-[11px] leading-relaxed font-mono bg-slate-950 p-2 rounded border border-slate-850 select-all">${p.prompt}</p>
    `;
    container.appendChild(card);
  });
}

function copyPromptText(probeId) {
  const probe = PROBES_DATA.find(p => p.id === probeId);
  if (!probe) return;
  navigator.clipboard.writeText(probe.prompt).then(() => {
    alert(`提示词已复制到剪贴板！可以直接发送给目标 AI 模型。`);
  });
}

function addSubmissionCard(probeId = "arr_int5", rawText = "") {
  cardCounter++;
  const container = document.getElementById("submissions-list");
  if (!container) return;

  const cardId = `sub-card-${cardCounter}`;
  const card = document.createElement("div");
  card.id = cardId;
  card.className = "submission-card bg-slate-900 border border-slate-800 rounded-xl p-3.5 space-y-2 relative";

  let optionsHtml = "";
  PROBES_DATA.forEach(p => {
    const selected = p.id === probeId ? "selected" : "";
    optionsHtml += `<option value="${p.id}" ${selected}>${p.title}</option>`;
  });

  card.innerHTML = `
    <div class="flex items-center justify-between gap-2">
      <select class="sub-probe-select bg-slate-950 border border-slate-800 rounded px-2.5 py-1 text-xs text-slate-200 font-medium focus:outline-none focus:border-sky-500 w-full">
        ${optionsHtml}
      </select>
      <button onclick="duplicateCard('${cardId}')" title="同题追加一次采样" class="shrink-0 p-1 text-xs text-sky-400 hover:text-sky-300 transition">
        ➕重采
      </button>
      <button onclick="removeCard('${cardId}')" title="删除此回答" class="shrink-0 p-1 text-xs text-rose-400 hover:text-rose-300 transition">
        ✕
      </button>
    </div>
    <div>
      <textarea class="sub-raw-text w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-xs text-slate-200 font-mono focus:outline-none focus:border-sky-500" rows="2" placeholder="粘贴模型的数组回答 (例如: [14, 58, 23, 91, 7])...">${rawText}</textarea>
    </div>
  `;

  container.appendChild(card);
  updateSubmissionCount();
}

function duplicateCard(cardId) {
  const el = document.getElementById(cardId);
  if (!el) return;
  const probeId = el.querySelector(".sub-probe-select").value;
  addSubmissionCard(probeId, "");
}

function removeCard(cardId) {
  const el = document.getElementById(cardId);
  if (el) el.remove();
  updateSubmissionCount();
}

function clearSubmissions() {
  document.getElementById("submissions-list").innerHTML = "";
  updateSubmissionCount();
}

function updateSubmissionCount() {
  const count = document.querySelectorAll(".submission-card").length;
  const badge = document.getElementById("submission-count-badge");
  if (badge) badge.innerText = count;
}

function loadPreset(key) {
  const preset = PRESETS[key];
  if (!preset) return;
  clearSubmissions();
  preset.forEach(item => {
    addSubmissionCard(item.probe_id, item.raw_text);
  });
  executeEvaluation();
}

async function executeEvaluation() {
  const cards = document.querySelectorAll(".submission-card");
  if (cards.length === 0) {
    alert("请至少添加一条待测数组回答！");
    return;
  }

  const submissions = [];
  cards.forEach(c => {
    const pid = c.querySelector(".sub-probe-select").value;
    const txt = c.querySelector(".sub-raw-text").value.trim();
    if (txt) {
      submissions.push({ probe_id: pid, raw_text: txt });
    }
  });

  if (submissions.length === 0) {
    alert("请在回答框内填入数组内容！");
    return;
  }

  const consent = document.getElementById("consent-checkbox").checked;
  const claimedModel = document.getElementById("claimed-model-select").value;

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
        claimed_model: claimedModel || null
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

function renderEvaluationResults(data) {
  const evalRes = data.evaluation;
  const clusterData = data.cluster_data;

  document.getElementById("results-empty-state").classList.add("hidden");
  document.getElementById("results-panel").classList.remove("hidden");

  document.getElementById("verdict-model-name").innerText = evalRes.top_model;
  document.getElementById("verdict-prob-badge").innerText = `${(evalRes.top_probability * 100).toFixed(1)}%`;
  
  const topCI = evalRes.confidence_intervals[evalRes.top_model] || [0, 0];
  document.getElementById("verdict-ci-range").innerText = `[${(topCI[0] * 100).toFixed(1)}% ~ ${(topCI[1] * 100).toFixed(1)}%]`;

  document.getElementById("stat-margin").innerText = `+${(evalRes.margin * 100).toFixed(1)}%`;
  document.getElementById("stat-entropy").innerText = `${evalRes.entropy} bit`;
  document.getElementById("stat-n1").innerText = evalRes.unique_probes_tested;
  document.getElementById("stat-n2").innerText = evalRes.sample_count;

  renderPosteriorsChart(evalRes.posteriors, evalRes.confidence_intervals);
  renderClusterCanvas(clusterData);

  const tbody = document.getElementById("parsed-table-body");
  tbody.innerHTML = "";
  evalRes.parsed_submissions.forEach(rec => {
    const tr = document.createElement("tr");
    const tokenBadges = rec.parsed_tokens.map(t => 
      `<span class="px-1.5 py-0.5 rounded bg-sky-950 text-sky-300 border border-sky-800 mr-1 font-bold">${t}</span>`
    ).join("");

    const traitsStr = Object.entries(rec.traits || {}).map(([k, v]) => `${k}:${v}`).join(", ");

    tr.innerHTML = `
      <td class="p-2.5 font-sans">${rec.probe_id}</td>
      <td class="p-2.5">${tokenBadges}</td>
      <td class="p-2.5 text-slate-400 text-[11px] font-sans">${traitsStr || "-"}</td>
      <td class="p-2.5">${rec.strictly_complied ? '<span class="text-emerald-400">✓ 纯JSON</span>' : '<span class="text-amber-400">△ 代码块包装</span>'}</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderPosteriorsChart(posteriors, confidenceIntervals) {
  const ctx = document.getElementById("posteriors-chart").getContext("2d");
  
  const sortedEntries = Object.entries(posteriors).sort((a, b) => b[1] - a[1]);
  const labels = sortedEntries.map(e => e[0]);
  const values = sortedEntries.map(e => (e[1] * 100).toFixed(1));

  const modelColorMap = {};
  MODELS_DATA.forEach(m => { modelColorMap[m.name] = m.color; });

  const bgColors = labels.map(m => modelColorMap[m] || "#3b82f6");

  if (posteriorsChart) {
    posteriorsChart.destroy();
  }

  posteriorsChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: '后验归属概率 (%)',
        data: values,
        backgroundColor: bgColors.map(c => c + "cc"),
        borderColor: bgColors,
        borderWidth: 1.5,
        borderRadius: 6
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: function(context) {
              const mName = context.label;
              const prob = context.raw;
              const ci = confidenceIntervals[mName] || [0, 0];
              return `后验概率: ${prob}% | 95% CI: [${(ci[0] * 100).toFixed(1)}% ~ ${(ci[1] * 100).toFixed(1)}%]`;
            }
          }
        }
      },
      scales: {
        x: {
          min: 0,
          max: 100,
          grid: { color: 'rgba(51, 65, 85, 0.4)' },
          ticks: { color: '#94a3b8', font: { family: 'monospace' } }
        },
        y: {
          grid: { display: false },
          ticks: { color: '#f1f5f9', font: { weight: 'bold' } }
        }
      }
    }
  });
}

function renderClusterCanvas(clusterData) {
  if (!clusterData || !clusterData.clusters) return;
  const canvas = document.getElementById("cluster-canvas");
  if (!canvas) return;

  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;

  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  const w = rect.width;
  const h = rect.height;

  ctx.fillStyle = "#090d16";
  ctx.fillRect(0, 0, w, h);

  ctx.strokeStyle = "rgba(30, 41, 59, 0.7)";
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

  clusterData.clusters.forEach(c => {
    const color = c.color || "#3b82f6";

    ctx.fillStyle = color + "44";
    c.points.forEach(p => {
      const [sx, sy] = toScreen(p[0], p[1]);
      ctx.beginPath();
      ctx.arc(sx, sy, 3, 0, Math.PI * 2);
      ctx.fill();
    });

    const [cx, cy] = toScreen(c.center[0], c.center[1]);
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(cx, cy, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 1.5;
    ctx.stroke();

    ctx.fillStyle = "#e2e8f0";
    ctx.font = "bold 11px sans-serif";
    ctx.fillText(c.model_name, cx + 9, cy + 4);
  });

  if (clusterData.user_point) {
    const [ux, uy] = toScreen(clusterData.user_point[0], clusterData.user_point[1]);

    ctx.fillStyle = "rgba(251, 191, 36, 0.25)";
    ctx.beginPath();
    ctx.arc(ux, uy, 18, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = "#f59e0b";
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(ux, uy, 8, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = "#fef08a";
    ctx.font = "bold 12px sans-serif";
    ctx.fillText("★ 当前测试样本点 (You)", ux + 12, uy - 6);
  }
}

async function refreshDbStats() {
  try {
    const statsRes = await fetch("/api/stats");
    const stats = await statsRes.json();

    document.getElementById("db-total-count").innerText = stats.total_samples;
    document.getElementById("db-official-count").innerText = stats.official_samples;
    document.getElementById("db-user-count").innerText = stats.user_samples;

    const tbody = document.getElementById("db-model-table-body");
    tbody.innerHTML = "";
    stats.model_breakdown.forEach(row => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td class="p-2.5 font-bold font-sans">${row.model_name}</td>
        <td class="p-2.5 text-slate-300">${row.count} 组数组</td>
        <td class="p-2.5 text-sky-400 font-bold">${parseFloat(row.weighted_count).toFixed(1)}</td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error("Failed to refresh db stats", err);
  }
}

async function submitContribution(e) {
  e.preventDefault();
  const modelName = document.getElementById("contrib-model").value;
  const probeId = document.getElementById("contrib-probe").value;
  const text = document.getElementById("contrib-text").value.trim();
  const weight = parseFloat(document.getElementById("contrib-weight").value);

  const statusEl = document.getElementById("contrib-status");
  statusEl.innerText = "提交中...";
  statusEl.className = "text-xs text-center text-slate-400 mt-2";

  try {
    const res = await fetch("/api/contribute", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model_name: modelName,
        probe_id: probeId,
        raw_text: text,
        weight: weight
      })
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "提交失败");
    }

    const resData = await res.json();
    statusEl.innerText = `✓ 数组样本入库成功！已提取 ${resData.parsed_tokens.length} 个Token，按权重 ${resData.weight} 计入数据库。`;
    statusEl.className = "text-xs text-center text-emerald-400 font-bold mt-2";
    document.getElementById("contrib-text").value = "";
    refreshDbStats();
  } catch (err) {
    statusEl.innerText = `✗ 错误: ${err.message}`;
    statusEl.className = "text-xs text-center text-rose-400 font-bold mt-2";
  }
}
