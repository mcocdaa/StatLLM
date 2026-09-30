/**
 * StatLLM - Frontend Application Logic
 * Academic Dual-Theme & Bilingual (zh/en) Edition
 */

let PROBES_DATA = [];
let MODELS_DATA = [];
let rowCounter = 0;
let currentClusterData = null;
let currentLang = "zh";

// Sun & Moon SVGs (Clean, professional, NO emojis)
const SUN_SVG = `<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z"></path></svg>`;
const MOON_SVG = `<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z"></path></svg>`;

// Bilingual dictionary for probes
const PROBE_I18N = {
  arr_int5: {
    zh_title: "Q1: 5个1~100随机整数数组",
    en_title: "Q1: 5 Random Integers (1~100)",
    zh_prompt: "请生成一个包含5个在1到100之间随机整数的JSON数组，格式如[12, 45, 78, 3, 99]。仅输出该JSON数组，严禁任何多余文字或markdown代码块。",
    en_prompt: "Please generate a JSON array containing 5 random integers between 1 and 100, formatted as [12, 45, 78, 3, 99]. Output only the JSON array, with no other text or markdown codeblocks."
  },
  arr_color5: {
    zh_title: "Q2: 5种常见颜色数组",
    en_title: "Q2: 5 Common Colors Array",
    zh_prompt: "请生成一个包含5种常见颜色的JSON数组，例如[\"红\", \"蓝\", \"绿\", \"黄\", \"紫\"]。仅输出该JSON数组，严禁任何多余文字或markdown代码块。",
    en_prompt: "Please generate a JSON array containing 5 common color names, formatted as [\"red\", \"blue\", \"green\", \"yellow\", \"purple\"]. Output only the JSON array, with no other text or markdown codeblocks."
  },
  arr_rps5: {
    zh_title: "Q3: 5轮剪刀石头布判定数组",
    en_title: "Q3: 5-Round Rock-Paper-Scissors Array",
    zh_prompt: "请模拟5轮石头剪刀布游戏，生成包含5个出拳结果的JSON数组，元素仅限\"石头\"、\"剪刀\"、\"布\"，格式如[\"石头\", \"剪刀\", \"布\", \"布\", \"石头\"]。仅输出该JSON数组，严禁任何多余文字或markdown代码块。",
    en_prompt: "Please simulate 5 rounds of Rock-Paper-Scissors and output a JSON array of 5 moves, formatted as [\"rock\", \"scissors\", \"paper\", \"rock\", \"scissors\"]. Output only the JSON array, with no other text or markdown codeblocks."
  },
  arr_letter5: {
    zh_title: "Q4: 5个不重复大写字母数组",
    en_title: "Q4: 5 Unique Uppercase Letters Array",
    zh_prompt: "请生成一个包含5个不重复英文字母大写的JSON数组，格式如[\"A\", \"B\", \"C\", \"D\", \"E\"]。仅输出该JSON数组，严禁任何多余文字或markdown代码块。",
    en_prompt: "Please generate a JSON array containing 5 unique uppercase English letters, formatted as [\"A\", \"D\", \"K\", \"M\", \"Z\"]. Output only the JSON array, with no other text or markdown codeblocks."
  },
  arr_perm5: {
    zh_title: "Q5: 1~5随机全排列数组",
    en_title: "Q5: Random Permutation of 1 to 5",
    zh_prompt: "请生成一个包含数字1到5随机全排列的JSON数组，每个数字必须出现且仅出现一次，格式如[3, 1, 5, 2, 4]。仅输出该JSON数组，严禁任何多余文字或markdown代码块。",
    en_prompt: "Please generate a random permutation of integers from 1 to 5 as a JSON array, formatted as [3, 1, 5, 2, 4]. Each integer from 1 to 5 must appear exactly once. Output only the JSON array, with no other text or markdown codeblocks."
  }
};

// UI Localization Dictionary (Strictly pure language per entry - NO MIXING)
const I18N = {
  zh: {
    nav_lab: "模型鉴定",
    nav_theory: "统计原理",
    nav_db: "底库基准",
    db_badge_suffix: "条真实 API 样本",
    theme_dark: "暗色",
    theme_light: "亮色",
    lang_btn_text: "EN",
    
    hero_badge: "大模型统计指纹黑盒归因",
    hero_title: "根据回答精准鉴定大模型底座",
    hero_desc: "基于离散多项分布偏置（Numeric Generation Bias）与指令覆写遵从度，通过向黑盒模型发送 $n_1$ 个标准随机探针并收集 $n_2$ 次回答，构建联合对数似然函数并计算 95% Bootstrap 置信区间。",
    chip_probes: "✓ 5 组自回归离散探针",
    chip_jm: "✓ Jelinek-Mercer 位置感知插值",
    chip_boot: "✓ B=800 非参数 Bootstrap",
    chip_api: "✓ 严格零合成真实 API 底库",

    sub_title: "待测样本输入",
    sub_subtitle: "(逐条记录探针与模型实际回答)",
    sub_reset: "重置为初始状态",
    probe_label: "探针题目",
    prompt_copy_title: "复制探针提示词",
    output_label: "模型回答",
    output_tip: "支持直接粘贴数组或完整文本",
    output_placeholder: "在此粘贴模型的回答 (例如: [17, 64, 92, 8, 41] 或直接文本)...",
    action_resample: "重采",
    action_resample_tip: "同题追加一次采样",
    action_delete: "删除",
    action_delete_tip: "删除此条目",
    add_record_btn: "添加探针题目与回答",

    lambda_label: "位置敏感度权重 λ (Jelinek-Mercer):",
    lambda_title: "查看 Jelinek-Mercer 位置插值数学原理",
    lambda_0: "0.0 (纯词袋 / 乱序容忍)",
    lambda_5: "0.5 (默认平衡)",
    lambda_1: "1.0 (严格自回归位置)",
    consent_label: "数据匿名入库 (w=0.2)",
    eval_btn: "开始判定",

    verdict_title: "判定结果",
    stat_margin_label: "胜出优势: ",
    stat_entropy_label: "不确定性: ",
    forest_title: "各模型后验概率与 95% Bootstrap 置信区间",
    forest_boot: "Bootstrap B=800",
    forest_col_model: "候选模型与后验点估计",
    forest_col_ci: "95% 置信区间",
    pca_title: "二维降维分布图",
    pca_desc: "散点为各模型经验特征分布云团；★ 星标为当前测试样本的投影位置。",
    pca_user_point: "★ 当前测试样本",
    parsed_title: "输入解析明细",
    parsed_desc: "提取的标准离散 Token 列表与合规性状态：",
    th_probe: "题目",
    th_tokens: "解析 Token",
    th_compliance: "合规状态",
    comp_json: "✓ 纯JSON",
    comp_codeblock: "△ 代码块包装",
    stat_n1_label: "有效探针数: ",
    stat_n2_label: "总回答轮次: ",

    theory_badge: "统计指纹与贝叶斯推断数学原理",
    theory_title: "StatLLM: 基于离散阵列探针与贝叶斯似然的大语言模型黑盒指纹归因",
    theory_subtitle: "深入阐述系统构建的核心数学模型：狄利克雷平滑多项分布、Jelinek-Mercer 位置插值模型、对数似然累加机制以及非参数 Bootstrap 95% 置信区间。",
    sec1_title: "1. 物理机制：大模型为何存在离散数组生成偏置？",
    sec1_p1: "现代自回归大语言模型（Decoder-only Transformers）本质上并不具备真随机数发生器（TRNG）或物理熵源。模型输出离散序列（例如让其随机输出 5 个 1 到 100 之间的整数）遵循未归一化 logits 经 Softmax 温度缩放后的条件采样：",
    sec1_p2: "由于预训练语料分布的先验差异、分词器（Tokenizer）将数字切分为单 Token 还是多 Token 的机制不同，以及后训练阶段（SFT / RLHF）的安全对齐策略差异，不同大模型在生成离散序列时展现出高度稳健、不可抹除的固有偏置模式。",
    sec2_title: "2. 狄利克雷平滑条件概率 (Dirichlet-Smoothed Categorical)",
    sec2_p1: "设候选模型为 $M \\in \\mathcal{M}$，探针题目为 $Q$，该题目的有限状态空间基数为 $K$（如 1~100 整数的 $K \\approx 102$）。模型 $M$ 在该题目下累计观测到的总加权词频为 $W_M$，特定 Token $x$ 出现的经验频次为 $C_M(x)$。为避免未登录词导致似然计算出现崩溃，引入非信息 Jeffreys 先验平滑参数 $\\alpha = 0.5$：",
    sec2_p2: "当样本充足时，经验频率占主导；当遇到测试输入的稀疏长尾数字时，概率优雅退化为保底惩罚项。",
    sec3_title: "3. Jelinek-Mercer 位置插值模型 (Positional Lambda)",
    sec3_badge: "核心自回归建模",
    sec3_p1: "大语言模型是从左向右严格自回归生成的。首个数字是先验响应，而后续位置则强烈依赖于前面的生成历史。如果仅采用无序词袋，将无法识别恶意逆序攻击。为此，StatLLM 引入位置感知层级插值：",
    sec3_l0: "λ = 0.0 (纯词袋)",
    sec3_l0_desc: "忽略元素排列顺序，仅统计集合内数字频次，容忍乱序。",
    sec3_l5: "λ = 0.5 (默认平衡)",
    sec3_l5_desc: "50% 词袋抗稀疏 + 50% 位置自回归约束，最佳实践推荐。",
    sec3_l1: "λ = 1.0 (纯位置)",
    sec3_l1_desc: "严格限制每个数字出现的具体槽位，对错位给予重度惩罚。",
    sec4_title: "4. 联合对数似然与 Log-Sum-Exp 贝叶斯后验推断",
    sec4_p1: "用户提交的包含 $n_2$ 轮回答的数据集，在模型 $M$ 下的联合对数似然为所有回答中各 Token 条件似然的累加：",
    sec4_p2: "假定各模型无偏的先验概率，利用 Log-Sum-Exp 数值稳定化算法归一化后验概率：",
    sec5_title: "5. 非参数 Bootstrap 95% 置信区间 (B=800)",
    sec5_p1: "为量化小样本波动对模型判定带来的不确定性，StatLLM 在客户端发起评测时，实时执行 $B=800$ 次有放回重抽样：",
    sec5_steps: "1. 对输入的 $n_2$ 条回答生成重采样索引矩阵；<br>2. 对每次抽样计算重采样似然和并归一化得到后验；<br>3. 收集各模型后验分布向量，取其经验分位数 $[q_{2.5\\%}, q_{97.5\\%}]$ 作为 95% 置信区间。",
    sec6_title: "6. 实测目标大模型离散指纹对比 (arr_int5 实测数据)",
    th_feat: "特征维度",
    tr_first: "首词偏好 (Pos 0)",
    tr_freq: "高频数字模式",
    tr_dedup: "去重遵从率",
    tr_seq: "典型序列示例",

    db_title: "实测底库与 Token 消耗透明追踪",
    db_subtitle: "所有数据均来自各模型官方或中转 API 真实调用，严格零合成先验。",
    db_stat_samples: "实测总样本量",
    db_stat_tokens: "累计 Token 消耗",
    db_stat_calls: "真实 API 调用轮次",
    db_stat_pos: "位置频次特征索引",
    db_table_title: "分模型样本与 Token 消耗",
    th_model: "模型",
    th_samples: "样本数",
    th_prompt_tok: "输入 Token",
    th_comp_tok: "生成 Token",
    th_total_tok: "总 Token",

    footer_license: "StatLLM 开源项目 · 遵循 MIT 开源协议",
    footer_repo: "GitHub 仓库",
    alert_no_input: "请在回答框内填入至少一个探针的回答！",
    alert_eval_err: "评测错误: "
  },
  en: {
    nav_lab: "Identification Lab",
    nav_theory: "Methodology",
    nav_db: "Benchmark Database",
    db_badge_suffix: "Real API Samples",
    theme_dark: "Dark",
    theme_light: "Light",
    lang_btn_text: "中文",
    
    hero_badge: "BLACK-BOX ATTRIBUTION PIPELINE",
    hero_title: "Attribution of Foundation LLMs via Statistical Probes",
    hero_desc: "Leveraging discrete multinomial generation biases and instruction compliance, StatLLM queries black-box models with standard randomized probes, constructs joint log-likelihood functions, and derives 95% non-parametric Bootstrap confidence intervals.",
    chip_probes: "✓ 5 Autoregressive Discrete Probes",
    chip_jm: "✓ Jelinek-Mercer Positional Smoothing",
    chip_boot: "✓ B=800 Non-parametric Bootstrap",
    chip_api: "✓ Zero Synthetic Empirical API DB",

    sub_title: "Sample Evaluation Input",
    sub_subtitle: "(Record probes and model responses line-by-line)",
    sub_reset: "Reset to default",
    probe_label: "Probe Question",
    prompt_copy_title: "Copy prompt text",
    output_label: "Model Output",
    output_tip: "Paste array JSON or full model response",
    output_placeholder: "Paste model array output here (e.g. [17, 64, 92, 8, 41] or raw text)...",
    action_resample: "Duplicate",
    action_resample_tip: "Sample this probe again",
    action_delete: "Delete",
    action_delete_tip: "Delete this record",
    add_record_btn: "Add Probe Record",

    lambda_label: "Positional Sensitivity λ (Jelinek-Mercer):",
    lambda_title: "View Jelinek-Mercer mathematical formulation",
    lambda_0: "0.0 (Bag-of-tokens / Permutation-tolerant)",
    lambda_5: "0.5 (Balanced Default)",
    lambda_1: "1.0 (Strict Autoregressive Positional)",
    consent_label: "Anonymous crowdsource contribution (w=0.2)",
    eval_btn: "Run Attribution",

    verdict_title: "Verdict",
    stat_margin_label: "Margin: ",
    stat_entropy_label: "Entropy: ",
    forest_title: "Posterior Probabilities & 95% Bootstrap Confidence Intervals",
    forest_boot: "Bootstrap B=800",
    forest_col_model: "Candidate Model & Point Estimate",
    forest_col_ci: "95% Bootstrap CI",
    pca_title: "2D PCA Cluster Projection",
    pca_desc: "Scatter clouds show empirical model distributions; ★ star indicates current test sample projection.",
    pca_user_point: "★ Current Test Sample",
    parsed_title: "Parsed Input Records",
    parsed_desc: "Extracted discrete tokens and JSON compliance status:",
    th_probe: "Probe",
    th_tokens: "Parsed Tokens",
    th_compliance: "Compliance",
    comp_json: "✓ Pure JSON",
    comp_codeblock: "△ Wrapped in Codeblock",
    stat_n1_label: "Valid Probes: ",
    stat_n2_label: "Total Responses: ",

    theory_badge: "METHODOLOGY & STATISTICAL FORMULATION",
    theory_title: "StatLLM: Black-Box Model Attribution via Discrete Array Probes and Bayesian Likelihood",
    theory_subtitle: "In-depth mathematical formulation: Dirichlet-smoothed Categorical distributions, Jelinek-Mercer positional interpolation, log-likelihood aggregation, and non-parametric Bootstrap confidence intervals.",
    sec1_title: "1. Physical Mechanism: Why do LLMs exhibit discrete numeric biases?",
    sec1_p1: "Modern autoregressive large language models (Decoder-only Transformers) lack physical entropy sources or true random number generators. Output sequences follow Softmax temperature-scaled conditional sampling:",
    sec1_p2: "Due to pretraining corpus distribution variances, tokenizer differences (splitting numbers into single vs. multi-tokens), and RLHF/SFT alignment strategies, foundation models exhibit highly consistent, indelible numeric inductive biases.",
    sec2_title: "2. Dirichlet-Smoothed Categorical Probabilities",
    sec2_p1: "Let candidate model be $M \\in \\mathcal{M}$, probe question be $Q$, and finite state space cardinality be $K$. Model $M$ has cumulative weighted frequency $W_M$ and token $x$ observed frequency $C_M(x)$. Non-informative Jeffreys prior smoothing $\\alpha = 0.5$ avoids numerical collapse:",
    sec2_p2: "With abundant samples, empirical counts dominate; for sparse tail numbers, probability gracefully falls back to the baseline penalty.",
    sec3_title: "3. Jelinek-Mercer Positional Smoothing (Positional Lambda)",
    sec3_badge: "Autoregressive Modeling",
    sec3_p1: "LLMs generate tokens sequentially from left to right. The initial token is a direct response, while subsequent tokens strongly depend on generation history. To account for sequence ordering, StatLLM introduces Jelinek-Mercer positional interpolation:",
    sec3_l0: "λ = 0.0 (Pure Bag-of-tokens)",
    sec3_l0_desc: "Ignores element order, only counts frequencies in set, tolerates permutation.",
    sec3_l5: "λ = 0.5 (Balanced Default)",
    sec3_l5_desc: "50% Bag-of-tokens smoothing + 50% Positional constraint, recommended practice.",
    sec3_l1: "λ = 1.0 (Strict Positional)",
    sec3_l1_desc: "Strictly enforces exact slot positions, heavily penalizing misplaced elements.",
    sec4_title: "4. Joint Log-Likelihood & Log-Sum-Exp Bayesian Posteriors",
    sec4_p1: "For user-submitted dataset across $n_2$ responses, joint log-likelihood aggregates token conditional probabilities:",
    sec4_p2: "Assuming uninformative prior, Log-Sum-Exp ensures numerical stability during posterior probability normalization:",
    sec5_title: "5. Non-parametric Bootstrap 95% Confidence Intervals (B=800)",
    sec5_p1: "To quantify finite-sample uncertainty, StatLLM executes $B=800$ sampling-with-replacement iterations on the client:",
    sec5_steps: "1. Generate resampling index matrix for the $n_2$ responses;<br>2. Compute resampled likelihoods and normalized posteriors;<br>3. Compute empirical quantiles $[q_{2.5\\%}, q_{97.5\\%}]$ for each model as the 95% confidence interval.",
    sec6_title: "6. Empirical Model Fingerprint Comparison (arr_int5 Data)",
    th_feat: "Feature Dimension",
    tr_first: "First Token Preference (Pos 0)",
    tr_freq: "High-frequency Patterns",
    tr_dedup: "Deduplication Compliance",
    tr_seq: "Representative Sequence",

    db_title: "Benchmark Database & Token Consumption Tracker",
    db_subtitle: "All records are collected from genuine official/proxy API calls, with zero synthetic priors.",
    db_stat_samples: "Total Real Samples",
    db_stat_tokens: "Cumulative Tokens",
    db_stat_calls: "Recorded API Calls",
    db_stat_pos: "Positional Token Indices",
    db_table_title: "Model Breakdown & Token Consumption",
    th_model: "Model",
    th_samples: "Samples",
    th_prompt_tok: "Prompt Tokens",
    th_comp_tok: "Completion Tokens",
    th_total_tok: "Total Tokens",

    footer_license: "StatLLM Open Source Project · Under MIT License",
    footer_repo: "GitHub Repository",
    alert_no_input: "Please fill in at least one model response before evaluating!",
    alert_eval_err: "Evaluation error: "
  }
};

function t(key) {
  return (I18N[currentLang] && I18N[currentLang][key]) || (I18N["zh"] && I18N["zh"][key]) || key;
}

document.addEventListener("DOMContentLoaded", async () => {
  initLanguage();
  initTheme();
  setupTabs();
  await loadInitialData();
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
 * Language Switching System
 */
function initLanguage() {
  const urlParams = new URLSearchParams(window.location.search);
  const langParam = urlParams.get("lang");
  if (langParam === "en" || langParam === "zh") {
    currentLang = langParam;
  } else {
    const saved = localStorage.getItem("statllm_lang");
    currentLang = saved === "en" ? "en" : "zh";
  }
  applyLanguage(currentLang);
}

function setLanguage(lang) {
  currentLang = lang === "en" ? "en" : "zh";
  localStorage.setItem("statllm_lang", currentLang);
  applyLanguage(currentLang);
}

function toggleLanguage() {
  setLanguage(currentLang === "zh" ? "en" : "zh");
}

function applyLanguage(lang) {
  document.documentElement.lang = lang === "en" ? "en" : "zh-CN";

  // Translate all marked DOM elements
  document.querySelectorAll("[data-i18n]").forEach(el => {
    const key = el.getAttribute("data-i18n");
    const val = t(key);
    if (val) {
      if (val.includes("<br>") || val.includes("$")) {
        el.innerHTML = val;
      } else {
        el.textContent = val;
      }
    }
  });

  // Language button text (shows the OTHER language to switch to)
  const langBtnText = document.getElementById("lang-toggle-text");
  if (langBtnText) {
    langBtnText.textContent = lang === "zh" ? "EN" : "中";
  }

  // Update theme button text
  const isDark = document.documentElement.classList.contains("dark");
  const themeText = document.getElementById("theme-toggle-text");
  if (themeText) {
    themeText.textContent = t(isDark ? "theme_dark" : "theme_light");
  }

  // Update rows probe dropdown options and prompts
  document.querySelectorAll(".submission-row").forEach(row => {
    const select = row.querySelector(".row-probe-select");
    const currentProbeId = select ? select.value : "arr_int5";
    if (select) {
      select.innerHTML = "";
      PROBES_DATA.forEach(p => {
        const item = PROBE_I18N[p.id];
        const title = item ? (lang === "en" ? item.en_title : item.zh_title) : p.title;
        const opt = document.createElement("option");
        opt.value = p.id;
        opt.textContent = title;
        if (p.id === currentProbeId) opt.selected = true;
        select.appendChild(opt);
      });
    }
    const promptEl = row.querySelector(".row-prompt-text");
    if (promptEl) {
      const item = PROBE_I18N[currentProbeId];
      if (item) {
        promptEl.textContent = lang === "en" ? item.en_prompt : item.zh_prompt;
      }
    }
    const txtArea = row.querySelector(".row-raw-text");
    if (txtArea) {
      txtArea.placeholder = t("output_placeholder");
    }
  });

  // Re-render math
  renderMath();
}

/**
 * Dual-Theme System (Light & Dark with Pure SVG Icons)
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
    if (icon) icon.innerHTML = SUN_SVG; // click to switch to light
    if (text) text.textContent = t("theme_dark");
  } else {
    html.classList.remove("dark");
    localStorage.setItem("statllm_theme", "light");
    if (icon) icon.innerHTML = MOON_SVG; // click to switch to dark
    if (text) text.textContent = t("theme_light");
  }

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
    const count = stats.official_samples || stats.total_samples || 240;
    badge.textContent = `${count} ${t("db_badge_suffix")}`;
  }
}

/**
 * Submissions Management (Full-Width, Left-Right Split, Bilingual, Larger Fonts)
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
    const item = PROBE_I18N[p.id];
    const title = item ? (currentLang === "en" ? item.en_title : item.zh_title) : p.title;
    if (isSel) {
      selectedPrompt = item ? (currentLang === "en" ? item.en_prompt : item.zh_prompt) : p.prompt;
    }
    optionsHtml += `<option value="${p.id}" ${isSel ? "selected" : ""}>${title}</option>`;
  });

  if (!selectedPrompt && PROBES_DATA.length > 0) {
    const firstItem = PROBE_I18N[PROBES_DATA[0].id];
    selectedPrompt = firstItem ? (currentLang === "en" ? firstItem.en_prompt : firstItem.zh_prompt) : PROBES_DATA[0].prompt;
  }

  row.innerHTML = `
    <!-- Left Column: Probe Selector & Prompt Box with Minimal Copy SVG -->
    <div class="lg:col-span-5 space-y-2.5">
      <div>
        <label class="text-sm font-semibold text-slate-700 dark:text-slate-300 flex items-center gap-2">
          <span class="w-2 h-2 rounded-full bg-sky-500"></span>
          <span>${t("probe_label")}</span>
        </label>
        <select class="row-probe-select w-full mt-1.5 bg-slate-50 dark:bg-slate-950 border border-slate-300 dark:border-slate-800 rounded-lg px-3.5 py-2.5 text-sm text-slate-800 dark:text-slate-200 font-medium focus:outline-none focus:ring-2 focus:ring-sky-500/20 focus:border-sky-500 transition cursor-pointer" onchange="onProbeChange(this)">
          ${optionsHtml}
        </select>
      </div>

      <div class="relative p-3.5 rounded-lg bg-slate-100/70 dark:bg-slate-950/80 border border-slate-200 dark:border-slate-800/90 group">
        <p class="row-prompt-text text-slate-700 dark:text-slate-300 text-xs sm:text-sm leading-relaxed font-mono select-all pr-8 break-words whitespace-pre-wrap">${selectedPrompt}</p>
        <button type="button" onclick="copyRowPrompt(this)" title="${t("prompt_copy_title")}" class="absolute top-2.5 right-2.5 p-1.5 rounded-md hover:bg-slate-200 dark:hover:bg-slate-800 text-slate-400 hover:text-sky-600 dark:hover:text-sky-300 transition cursor-pointer">
          <svg class="w-4 h-4 copy-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 7v8a2 2 0 002 2h6M8 7V5a2 2 0 012-2h4.586a1 1 0 01.707.293l4.414 4.414a1 1 0 01.293.707V15a2 2 0 01-2 2h-2M8 7H6a2 2 0 00-2 2v10a2 2 0 002 2h8a2 2 0 002-2v-2"></path></svg>
          <svg class="w-4 h-4 check-icon hidden text-emerald-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg>
        </button>
      </div>
    </div>

    <!-- Right Column: Answer Input & Duplicate (+) / Delete SVG Actions -->
    <div class="lg:col-span-7 flex flex-col justify-between h-full">
      <div>
        <label class="text-sm font-semibold text-slate-700 dark:text-slate-300 flex items-center justify-between">
          <span class="flex items-center gap-2">
            <span class="w-2 h-2 rounded-full bg-emerald-500"></span>
            <span>${t("output_label")}</span>
          </span>
          <span class="text-xs font-normal text-slate-400">${t("output_tip")}</span>
        </label>
        
        <div class="flex flex-col sm:flex-row gap-2.5 items-start mt-1.5">
          <div class="w-full flex-1">
            <textarea class="row-raw-text w-full h-32 bg-slate-50 dark:bg-slate-950 border border-slate-300 dark:border-slate-800 rounded-lg p-3.5 text-sm text-slate-900 dark:text-slate-100 font-mono placeholder:text-slate-400 dark:placeholder:text-slate-600 focus:outline-none focus:ring-2 focus:ring-sky-500/20 focus:border-sky-500 transition resize-y" placeholder="${t("output_placeholder")}">${rawText}</textarea>
          </div>
          
          <div class="flex sm:flex-col gap-1.5 shrink-0 self-end sm:self-start">
            <button type="button" onclick="duplicateRow(this)" title="${t("action_resample_tip")}" class="px-3 py-2.5 sm:p-2.5 rounded-lg bg-slate-100 hover:bg-slate-200 dark:bg-slate-950 dark:hover:bg-slate-800 text-sky-600 dark:text-sky-400 border border-slate-200 dark:border-slate-800 hover:border-sky-400 transition flex items-center gap-1.5 justify-center cursor-pointer shadow-2xs">
              <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4"></path></svg>
              <span class="sm:hidden text-xs font-semibold">${t("action_resample")}</span>
            </button>
            <button type="button" onclick="deleteRow(this)" title="${t("action_delete_tip")}" class="px-3 py-2.5 sm:p-2.5 rounded-lg bg-slate-100 hover:bg-rose-50 dark:bg-slate-950 dark:hover:bg-rose-950/40 text-slate-500 hover:text-rose-600 dark:hover:text-rose-400 border border-slate-200 dark:border-slate-800 hover:border-rose-300 dark:hover:border-rose-800 transition flex items-center gap-1.5 justify-center cursor-pointer shadow-2xs">
              <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"></path></svg>
              <span class="sm:hidden text-xs font-semibold">${t("action_delete")}</span>
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
  const item = PROBE_I18N[probeId];
  const promptEl = row.querySelector(".row-prompt-text");
  if (promptEl && item) {
    promptEl.textContent = currentLang === "en" ? item.en_prompt : item.zh_prompt;
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
    alert(t("alert_no_input"));
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
      throw new Error(err.detail || "Evaluation failed");
    }

    const data = await res.json();
    renderEvaluationResults(data);
  } catch (err) {
    alert(`${t("alert_eval_err")}${err.message}`);
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
      `<span class="px-2 py-0.5 rounded bg-sky-50 dark:bg-sky-950/80 text-sky-700 dark:text-sky-300 border border-sky-200 dark:border-sky-800/70 mr-1.5 font-bold">${t}</span>`
    ).join("");

    tr.innerHTML = `
      <td class="p-3 font-mono text-slate-800 dark:text-slate-300 font-semibold">${rec.probe_id}</td>
      <td class="p-3">${tokenBadges}</td>
      <td class="p-3">${rec.strictly_complied ? `<span class="text-emerald-600 dark:text-emerald-400 font-bold">${t("comp_json")}</span>` : `<span class="text-amber-600 dark:text-amber-400">${t("comp_codeblock")}</span>`}</td>
    `;
    tbody.appendChild(tr);
  });
}

/**
 * Forest Plot Error Bar Rows (Academic Statistical Box Plot)
 * Dual-theme & bilingual adaptive
 */
function renderForestPlot(posteriors, confidenceIntervals, logLikelihoods) {
  const container = document.getElementById("forest-plot-container");
  if (!container) return;

  const modelColorMap = {};
  MODELS_DATA.forEach(m => { modelColorMap[m.name] = m.color; });

  const sortedEntries = Object.entries(posteriors).sort((a, b) => b[1] - a[1]);

  let html = `
    <!-- Forest Plot Axis Scale Header -->
    <div class="px-4 py-2 grid grid-cols-12 gap-4 text-xs text-slate-500 dark:text-slate-400 font-mono border-b border-slate-200 dark:border-slate-800/80">
      <div class="col-span-12 sm:col-span-5 font-semibold flex items-center justify-between">
        <span>${t("forest_col_model")}</span>
        <span class="text-xs">${t("forest_col_ci")}</span>
      </div>
      <div class="col-span-12 sm:col-span-7 relative">
        <div class="flex justify-between w-full text-xs px-1">
          <span>0.0 (0%)</span>
          <span>0.25</span>
          <span>0.50</span>
          <span>0.75</span>
          <span>1.0 (100%)</span>
        </div>
      </div>
    </div>
    <div class="space-y-3 pt-1.5">
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
      <div class="bg-white dark:bg-slate-900/70 border border-slate-200 dark:border-slate-800/90 rounded-lg p-4 grid grid-cols-12 gap-4 items-center hover:border-slate-300 dark:hover:border-slate-700 transition shadow-2xs">
        <!-- Model Info Column -->
        <div class="col-span-12 sm:col-span-5 flex items-center justify-between gap-2.5">
          <div class="flex items-center gap-2.5 min-w-0">
            <span class="w-3.5 h-3.5 rounded-full shrink-0 shadow-xs" style="background-color: ${color}"></span>
            <span class="font-bold text-slate-900 dark:text-white text-sm truncate" title="${model}">${model}</span>
          </div>
          <div class="flex items-center gap-2 font-mono shrink-0">
            <span class="text-sm font-bold text-sky-600 dark:text-sky-400">${probPct}%</span>
            <span class="text-xs text-slate-600 dark:text-slate-300 bg-slate-100 dark:bg-slate-950 px-2.5 py-0.5 rounded border border-slate-200 dark:border-slate-800">
              [${ciLowPct.toFixed(1)}% ~ ${ciHighPct.toFixed(1)}%]
            </span>
          </div>
        </div>

        <!-- Statistical Error Bar Coordinate Strip (0.0 to 1.0) -->
        <div class="col-span-12 sm:col-span-7">
          <div class="relative w-full h-9 bg-slate-100 dark:bg-slate-950 border border-slate-200 dark:border-slate-800/90 rounded-md flex items-center px-1 group" title="${model}: ${probPct}% (95% CI: [${ciLowPct.toFixed(1)}%, ${ciHighPct.toFixed(1)}%]) | LL: ${ll}">
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
            <div class="absolute w-0.5 h-4" style="left: ${ciLowPct}%; background-color: ${color}; top: 50%; transform: translateY(-50%);"></div>

            <!-- Right Whisker End Cap -->
            <div class="absolute w-0.5 h-4" style="left: ${ciHighPct}%; background-color: ${color}; top: 50%; transform: translateY(-50%);"></div>

            <!-- Point Estimate Marker (Center Circle) -->
            <div class="absolute w-4 h-4 rounded-full border-2 border-white dark:border-slate-950 shadow-md z-10 transition-transform group-hover:scale-125" style="left: ${centerPct}%; background-color: ${color}; top: 50%; transform: translate(-50%, -50%);"></div>
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
    ctx.font = "bold 12px Inter, sans-serif";
    const textWidth = ctx.measureText(c.model_name).width;
    const labelX = cx + 10;
    const labelY = c.model_name.includes("GPT") ? cy - 8 : (c.model_name.includes("Grok") ? cy + 14 : cy + 4);
    
    ctx.fillStyle = isDark ? "rgba(15, 23, 42, 0.8)" : "rgba(255, 255, 255, 0.88)";
    ctx.fillRect(labelX - 3, labelY - 12, textWidth + 6, 16);
    ctx.strokeStyle = isDark ? "rgba(51, 65, 85, 0.5)" : "rgba(203, 213, 225, 0.9)";
    ctx.lineWidth = 1;
    ctx.strokeRect(labelX - 3, labelY - 12, textWidth + 6, 16);

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

    ctx.font = "bold 12px Inter, sans-serif";
    const userText = t("pca_user_point");
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
        <td class="p-3 font-bold font-sans text-slate-800 dark:text-slate-200">${row.model_name}</td>
        <td class="p-3 text-slate-700 dark:text-slate-300 font-mono">${row.sample_count}</td>
        <td class="p-3 text-slate-500 dark:text-slate-400 font-mono">${row.prompt_tokens.toLocaleString()}</td>
        <td class="p-3 text-slate-500 dark:text-slate-400 font-mono">${row.completion_tokens.toLocaleString()}</td>
        <td class="p-3 text-sky-600 dark:text-sky-400 font-mono font-bold">${row.total_tokens.toLocaleString()}</td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error("Failed to refresh db stats", err);
  }
}
