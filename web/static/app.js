/**
 * StatLLM - Frontend Application Logic
 * Academic Dual-Theme & Bilingual (zh/en) Edition
 */

let PROBES_DATA = [];
let MODELS_DATA = [];
let rowCounter = 0;
let currentClusterData = null;
let currentLang = "zh";
let cachedStats = null;
let cachedEvaluationData = null;

// Token display mapping for strict language purity (e.g. discrete colors & RPS in EN)
const TOKEN_DISPLAY = {
  en: {
    "红": "Red", "橙": "Orange", "黄": "Yellow", "绿": "Green",
    "青": "Cyan", "蓝": "Blue", "紫": "Purple",
    "石头": "Rock", "剪刀": "Scissors", "布": "Paper"
  }
};

function formatDisplayToken(tok) {
  if (currentLang === "en" && TOKEN_DISPLAY.en[tok]) {
    return TOKEN_DISPLAY.en[tok];
  }
  return tok;
}

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
    zh_title: "Q2: 5个离散颜色序列数组",
    en_title: "Q2: 5 Discrete Colors Array",
    zh_prompt: "在[红, 橙, 黄, 绿, 青, 蓝, 紫]中随机挑选5次，组成JSON数组，例如[\"红\", \"蓝\", \"绿\", \"红\", \"紫\"]。仅输出该JSON数组，严禁任何多余文字或markdown代码块。",
    en_prompt: "Please randomly choose 5 times from [Red, Orange, Yellow, Green, Cyan, Blue, Purple] to form a JSON array, e.g. [\"Red\", \"Blue\", \"Green\", \"Red\", \"Purple\"]. Output only the JSON array, with no other text or markdown codeblocks."
  },
  arr_rps5: {
    zh_title: "Q3: 5局石头剪刀布出拳序列",
    en_title: "Q3: 5-Round Rock-Paper-Scissors Array",
    zh_prompt: "进行5次完全独立的石头剪刀布随机选择，输出一个JSON数组，例如[\"石头\", \"剪刀\", \"石头\", \"布\", \"剪刀\"]。仅输出该JSON数组，严禁任何多余文字或markdown代码块。",
    en_prompt: "Please simulate 5 independent rounds of Rock-Paper-Scissors and output a JSON array of 5 moves from [\"Rock\", \"Scissors\", \"Paper\"], e.g. [\"Rock\", \"Scissors\", \"Paper\", \"Rock\", \"Scissors\"]. Output only the JSON array, with no other text or markdown codeblocks."
  },
  arr_letter5: {
    zh_title: "Q4: 5个随机大写英文字母数组",
    en_title: "Q4: 5 Random Uppercase Letters Array",
    zh_prompt: "请生成一个包含5个随机大写英文字母（A-Z）的JSON数组，例如[\"M\", \"X\", \"R\", \"A\", \"K\"]。仅输出该JSON数组，严禁任何多余文字或markdown代码块。",
    en_prompt: "Please generate a JSON array of 5 random uppercase English letters (A-Z), formatted as [\"M\", \"X\", \"R\", \"A\", \"K\"]. Output only the JSON array, with no other text or markdown codeblocks."
  },
  arr_perm5: {
    zh_title: "Q5: [1,2,3,4,5] 随机置乱排列",
    en_title: "Q5: [1,2,3,4,5] Random Permutation Array",
    zh_prompt: "将数字[1, 2, 3, 4, 5]完全随机打乱，输出一个打乱后的JSON数组，例如[3, 1, 5, 2, 4]。仅输出该JSON数组，严禁任何多余文字或markdown代码块。",
    en_prompt: "Please randomly shuffle the numbers [1, 2, 3, 4, 5] and output a JSON array, e.g. [3, 1, 5, 2, 4]. Each integer from 1 to 5 must appear exactly once. Output only the JSON array, with no other text or markdown codeblocks."
  }
};

// UI Localization Dictionary (Strictly pure language per entry - NO MIXING)
const I18N = {
  zh: {
    doc_title: "StatLLM - 大模型统计指纹与黑盒归因平台",
    brand_subtitle: "统计指纹归因",
    nav_lab: "模型鉴定",
    nav_theory: "统计原理",
    nav_db: "底库基准",
    db_badge_suffix: "条真实 API 样本",
    theme_dark: "暗色",
    theme_light: "亮色",
    lang_btn_text: "EN",
    title_lang_toggle: "切换为英文",
    title_theme_toggle: "切换暗色/亮色主题",
    title_reset_btn: "重置为初始状态",
    title_lambda_help: "查看 Jelinek-Mercer 位置插值数学原理",
    
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
    stat_margin_label: "吻合度优势: ",
    stat_entropy_label: "不确定性: ",
    advisory_title: "统计学提示：当前样本量与置信区间范围",
    advisory_badge: "建议增加测试轮次",
    advisory_current: "当前题目数: ",
    advisory_ci_span: "95% 区间半宽: ",
    advisory_recom: "建议补充至: ",
    advisory_shrink: "预计收窄",
    advisory_unit_items: "题",
    advisory_span_label: "全幅跨度",
    advisory_sufficient: "评测样本量充足（已提供 {count} 道题目），95% 置信区间高度收敛，统计鉴别具备高信度。",
    advisory_desc_tpl: "由于当前仅提供了 {count} 道题目的回答，95% 置信区间跨度较宽（±{radius}%）。大模型在单次离散抽样中存在固有随机性；建议增加测试轮次（如每道题测试 2~3 轮，达到 15 条以上样本），置信区间将依 {math_rate} 显著收窄约 {shrink}%（收窄至 ±{newRadius}%），大幅消除不确定度并精准锁定基座模型。",
    forest_title: "各模型独立吻合度与置信区间",
    forest_boot: "双层区间: 68% / 95%",
    forest_col_model: "候选模型与独立吻合度",
    forest_col_ci: "95% 置信区间",
    forest_tooltip_fit: "独立吻合度",
    forest_tooltip_post: "归一化排他后验",
    forest_tooltip_ci: "95% 置信区间",
    ci_legend_68: "68% 核心区间",
    ci_legend_95: "95% 置信区间",
    ci_point_tip: "中心估计点",
    stat_details_toggle: "查看统计量明细与方差分解 (SE, ΔLL, N_ref)",
    th_stat_model: "候选模型",
    th_stat_fitness: "独立吻合度",
    th_stat_ci68: "68% 核心区间",
    th_stat_ci95: "95% 置信区间",
    th_stat_ll: "累计似然 ΣlnP",
    th_stat_delta_ll: "净对数增益 ΔLL",
    th_stat_se: "标准误 SE",
    th_stat_nref: "底库样本数 N_ref",
    pca_title: "二维降维分布图",
    pca_desc: "散点为各模型经验特征分布云团；★ 星标为当前测试样本。点击或悬浮图例可单选高亮对比。",
    pca_user_point: "★ 当前测试样本",
    pca_legend_all: "全部显示",
    pca_coord_label: "坐标: ",
    pca_samples_label: "样本数: ",
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
    tr_dedup_deepseek: "局部重复",
    tr_seq: "典型序列示例",

    db_title: "实测底库与 Token 消耗透明追踪",
    db_subtitle: "所有数据均来自各模型官方或中转 API 真实调用，严格零合成先验。",
    db_stat_samples: "实测总样本量",
    db_stat_tokens: "累计 Token 消耗",
    db_stat_calls: "真实 API 调用轮次",
    db_stat_pos: "位置频次特征索引",
    db_pos_unit: "条",
    db_table_title: "分模型样本与 Token 消耗",
    th_model: "模型",
    th_samples: "样本数",
    th_prompt_tok: "输入 Token",
    th_comp_tok: "生成 Token",
    th_total_tok: "总 Token",

    db_archive_title: "全量底库归档与热迁移 (ZIP Bundle)",
    db_archive_desc: "支持一键导出自包含 ZIP 压缩包（包含 manifest 描述、JSONL 样本流与 SQLite 二进制快照），可直接在其他环境或新服务器上一键导入恢复。",
    db_btn_export: "导出归档包 (ZIP)",
    db_btn_import: "导入归档包 (ZIP)",
    modal_import_title: "导入底库归档 ZIP",
    modal_import_desc: "请选择导入策略：增量合并会保留现有样本并跳过重复项；全量覆盖会重置当前库为压缩包内的数据快照。",
    modal_import_mode_label: "导入策略：",
    modal_import_opt_merge: "增量合并 (保留现有，去重导入)",
    modal_import_opt_replace: "全量覆盖 (重置底库为新快照)",
    modal_cancel: "取消",
    modal_choose_file: "选择 ZIP 文件",
    toast_exporting: "正在生成 ZIP 归档包...",
    toast_importing: "正在解析并导入归档包...",
    toast_import_success: "成功导入 {n} 条样本 (跳过 {skip} 条重复项)！",
    toast_import_fail: "导入失败: ",
    toast_export_fail: "导出失败: ",

    footer_license: "StatLLM 开源项目 · 遵循 MIT 开源协议",
    footer_repo: "GitHub 仓库",
    alert_no_input: "请在回答框内填入至少一个探针的回答！",
    alert_eval_err: "评测错误: "
  },
  en: {
    doc_title: "StatLLM - LLM Statistical Fingerprinting & Attribution Engine",
    brand_subtitle: "Statistical Attribution",
    nav_lab: "Identification Lab",
    nav_theory: "Methodology",
    nav_db: "Benchmark Database",
    db_badge_suffix: "Real API Samples",
    theme_dark: "Dark",
    theme_light: "Light",
    lang_btn_text: "ZH",
    title_lang_toggle: "Switch to Chinese",
    title_theme_toggle: "Toggle Dark / Light Theme",
    title_reset_btn: "Reset to default",
    title_lambda_help: "View Jelinek-Mercer mathematical formulation",
    
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
    stat_margin_label: "Fit Margin: ",
    stat_entropy_label: "Entropy: ",
    advisory_title: "Statistical Advisory: Sample Size & Confidence Interval Span",
    advisory_badge: "Additional Rounds Recommended",
    advisory_current: "Current Samples: ",
    advisory_ci_span: "95% CI Radius: ",
    advisory_recom: "Recommended: ",
    advisory_shrink: "Estimated Reduction",
    advisory_unit_items: "items",
    advisory_span_label: "Full Span",
    advisory_sufficient: "Sufficient sample size ({count} items evaluated); 95% confidence interval is highly converged with strong attribution certainty.",
    advisory_desc_tpl: "With only {count} sample(s) provided, the 95% confidence interval is relatively wide (span ±{radius}%). LLMs exhibit intrinsic stochasticity in single-shot outputs; submitting additional rounds (e.g. 2–3 runs per probe, ≥15 samples) will shrink the confidence interval by {math_rate} by ~{shrink}% (down to ±{newRadius}%), significantly reducing uncertainty and sharpening model attribution.",
    forest_title: "Candidate Model Fitness & Confidence Intervals",
    forest_boot: "Dual CI: 68% / 95%",
    forest_col_model: "Candidate Model & Independent Fit",
    forest_col_ci: "95% Confidence Interval",
    forest_tooltip_fit: "Independent Fitness",
    forest_tooltip_post: "Normalized Posterior",
    forest_tooltip_ci: "95% Confidence Interval",
    ci_legend_68: "68% Core Interval",
    ci_legend_95: "95% Confidence Interval",
    ci_point_tip: "Point Estimate",
    stat_details_toggle: "Statistical Diagnostics & Variance Decomposition (SE, ΔLL, N_ref)",
    th_stat_model: "Candidate Model",
    th_stat_fitness: "Fitness",
    th_stat_ci68: "68% Core CI",
    th_stat_ci95: "95% Confidence CI",
    th_stat_ll: "Log-Likelihood ΣlnP",
    th_stat_delta_ll: "Net Log Gain ΔLL",
    th_stat_se: "Std Error (SE)",
    th_stat_nref: "Empirical Base (N_ref)",
    pca_title: "2D PCA Cluster Projection",
    pca_desc: "Scatter clouds depict empirical distributions; ★ star marks test sample. Click/hover legend to isolate models.",
    pca_user_point: "★ Current Test Sample",
    pca_legend_all: "Show All",
    pca_coord_label: "Coord: ",
    pca_samples_label: "Samples: ",
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
    tr_dedup_deepseek: "Local Duplicates",
    tr_seq: "Representative Sequence",

    db_title: "Benchmark Database & Token Consumption Tracker",
    db_subtitle: "All records are collected from genuine official/proxy API calls, with zero synthetic priors.",
    db_stat_samples: "Total Real Samples",
    db_stat_tokens: "Cumulative Tokens",
    db_stat_calls: "Recorded API Calls",
    db_stat_pos: "Positional Token Indices",
    db_pos_unit: "items",
    db_table_title: "Model Breakdown & Token Consumption",
    th_model: "Model",
    th_samples: "Samples",
    th_prompt_tok: "Prompt Tokens",
    th_comp_tok: "Completion Tokens",
    th_total_tok: "Total Tokens",

    db_archive_title: "Database Archive & Rapid Migration (ZIP Bundle)",
    db_archive_desc: "Export a self-contained ZIP archive bundle (including manifest, JSONL sample streams, and SQLite binary snapshot) for instant 1-click restore across machines.",
    db_btn_export: "Export Archive (ZIP)",
    db_btn_import: "Import Archive (ZIP)",
    modal_import_title: "Import Database Archive ZIP",
    modal_import_desc: "Select import strategy: Merge appends new samples and skips duplicates; Replace resets the database to the archive snapshot.",
    modal_import_mode_label: "Import Strategy:",
    modal_import_opt_merge: "Incremental Merge (Keep existing, skip duplicates)",
    modal_import_opt_replace: "Full Replace (Reset database to snapshot)",
    modal_cancel: "Cancel",
    modal_choose_file: "Select ZIP File",
    toast_exporting: "Generating ZIP archive...",
    toast_importing: "Parsing and importing archive bundle...",
    toast_import_success: "Successfully imported {n} samples (skipped {skip} duplicates)!",
    toast_import_fail: "Import failed: ",
    toast_export_fail: "Export failed: ",

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
  document.title = t("doc_title");

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
    langBtnText.textContent = lang === "zh" ? "EN" : "ZH";
  }
  const langToggleBtn = document.getElementById("lang-toggle-btn");
  if (langToggleBtn) {
    langToggleBtn.title = t("title_lang_toggle");
  }

  // Update theme button text & tooltip
  const isDark = document.documentElement.classList.contains("dark");
  const themeText = document.getElementById("theme-toggle-text");
  if (themeText) {
    themeText.textContent = t(isDark ? "theme_dark" : "theme_light");
  }
  const themeToggleBtn = document.getElementById("theme-toggle-btn");
  if (themeToggleBtn) {
    themeToggleBtn.title = t("title_theme_toggle");
  }

  // Update reset and lambda button tooltips
  const resetBtn = document.getElementById("reset-submissions-btn");
  if (resetBtn) resetBtn.title = t("title_reset_btn");
  const lambdaHelpBtn = document.getElementById("lambda-help-btn");
  if (lambdaHelpBtn) lambdaHelpBtn.title = t("title_lambda_help");

  // Update cached header stats and database tab units
  if (cachedStats) {
    updateHeaderStats(cachedStats);
  }
  const posEl = document.getElementById("db-pos-tokens");
  if (posEl) {
    const posCount = (cachedStats && cachedStats.positional_token_count) || 345;
    posEl.innerText = `${posCount} ${t("db_pos_unit")}`;
  }

  // Update rows probe dropdown options, prompts, placeholders, and tooltips
  document.querySelectorAll(".submission-row").forEach(row => {
    const probeLabelEl = row.querySelector(".row-probe-label");
    if (probeLabelEl) probeLabelEl.textContent = t("probe_label");
    const outputLabelEl = row.querySelector(".row-output-label");
    if (outputLabelEl) outputLabelEl.textContent = t("output_label");
    const outputTipEl = row.querySelector(".row-output-tip");
    if (outputTipEl) outputTipEl.textContent = t("output_tip");

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
    const copyBtn = row.querySelector("button[onclick='copyRowPrompt(this)']");
    if (copyBtn) copyBtn.title = t("prompt_copy_title");
    const dupBtn = row.querySelector("button[onclick='duplicateRow(this)']");
    if (dupBtn) {
      dupBtn.title = t("action_resample_tip");
      const span = dupBtn.querySelector("span");
      if (span) span.textContent = t("action_resample");
    }
    const delBtn = row.querySelector("button[onclick='deleteRow(this)']");
    if (delBtn) {
      delBtn.title = t("action_delete_tip");
      const span = delBtn.querySelector("span");
      if (span) span.textContent = t("action_delete");
    }
  });

  // Re-render evaluation results if active, else re-render cluster canvas
  if (cachedEvaluationData) {
    renderEvaluationResults(cachedEvaluationData);
  } else if (currentClusterData) {
    renderClusterCanvas(currentClusterData);
  }

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

  const hash = window.location.hash.replace("#", "");
  if (hash && ["tab-lab", "tab-theory", "tab-db"].includes(hash)) {
    window.switchToTab(hash);
  }
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
    cachedStats = stats;
    currentClusterData = await clusterRes.json();

    updateHeaderStats(stats);
    const posCount = stats.positional_token_count || 345;
    const posEl = document.getElementById("db-pos-tokens");
    if (posEl) posEl.innerText = `${posCount} ${t("db_pos_unit")}`;

    renderClusterCanvas(currentClusterData);
  } catch (err) {
    console.error("Failed to load initial data", err);
  }
}

function updateHeaderStats(stats) {
  const badge = document.getElementById("header-db-stats");
  if (badge) {
    const count = stats.official_samples || stats.total_samples || 600;
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
  row.className = "submission-row w-full bg-white dark:bg-slate-900/70 border border-slate-200 dark:border-slate-800/80 rounded-xl p-4 sm:p-5 shadow-xs grid grid-cols-1 lg:grid-cols-12 gap-5 items-stretch transition";

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
    <div class="lg:col-span-5 flex flex-col justify-between min-h-0">
      <div>
        <label class="text-sm font-semibold text-slate-700 dark:text-slate-300 flex items-center gap-2 h-6">
          <span class="w-2 h-2 rounded-full bg-sky-500"></span>
          <span class="row-probe-label">${t("probe_label")}</span>
        </label>
        <select class="row-probe-select w-full mt-1.5 bg-slate-50 dark:bg-slate-950 border border-slate-300 dark:border-slate-800 rounded-lg px-3.5 py-2.5 text-sm text-slate-800 dark:text-slate-200 font-medium focus:outline-none focus:ring-2 focus:ring-sky-500/20 focus:border-sky-500 transition cursor-pointer" onchange="onProbeChange(this)">
          ${optionsHtml}
        </select>
      </div>

      <div class="relative mt-2.5 p-3.5 rounded-lg bg-slate-100/70 dark:bg-slate-950/80 border border-slate-200 dark:border-slate-800/90 group flex-1 min-h-[76px] flex flex-col justify-center">
        <p class="row-prompt-text text-slate-700 dark:text-slate-300 text-xs sm:text-sm leading-relaxed font-mono select-all pr-8 break-words whitespace-pre-wrap">${selectedPrompt}</p>
        <button type="button" onclick="copyRowPrompt(this)" title="${t("prompt_copy_title")}" class="absolute top-2.5 right-2.5 p-1.5 rounded-md hover:bg-slate-200 dark:hover:bg-slate-800 text-slate-400 hover:text-sky-600 dark:hover:text-sky-300 transition cursor-pointer">
          <svg class="w-4 h-4 copy-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 7v8a2 2 0 002 2h6M8 7V5a2 2 0 012-2h4.586a1 1 0 01.707.293l4.414 4.414a1 1 0 01.293.707V15a2 2 0 01-2 2h-2M8 7H6a2 2 0 00-2 2v10a2 2 0 002 2h8a2 2 0 002-2v-2"></path></svg>
          <svg class="w-4 h-4 check-icon hidden text-emerald-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg>
        </button>
      </div>
    </div>

    <!-- Right Column: Answer Input & Duplicate (+) / Delete SVG Actions -->
    <div class="lg:col-span-7 flex flex-col justify-between min-h-0">
      <label class="text-sm font-semibold text-slate-700 dark:text-slate-300 flex items-center justify-between h-6">
        <span class="flex items-center gap-2">
          <span class="w-2 h-2 rounded-full bg-emerald-500"></span>
          <span class="row-output-label">${t("output_label")}</span>
        </span>
        <span class="row-output-tip text-xs font-normal text-slate-400">${t("output_tip")}</span>
      </label>
      
      <div class="flex flex-col sm:flex-row gap-2.5 items-stretch mt-1.5 flex-1 min-h-0">
        <div class="w-full flex-1 flex flex-col min-h-0">
          <textarea class="row-raw-text w-full h-full flex-1 min-h-[120px] bg-slate-50 dark:bg-slate-950 border border-slate-300 dark:border-slate-800 rounded-lg p-3.5 text-sm text-slate-900 dark:text-slate-100 font-mono placeholder:text-slate-400 dark:placeholder:text-slate-600 focus:outline-none focus:ring-2 focus:ring-sky-500/20 focus:border-sky-500 transition resize-none" placeholder="${t("output_placeholder")}">${rawText}</textarea>
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
  const consentEl = document.getElementById("consent-checkbox");
  if (consentEl) consentEl.checked = false;
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
  const consent = consentEl ? consentEl.checked : false;

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
  cachedEvaluationData = data;
  const evalRes = data.evaluation;
  const clusterData = data.cluster_data;
  currentClusterData = clusterData;

  const legendContainer = document.getElementById("pca-legend-container");
  if (legendContainer) legendContainer.innerHTML = "";

  const resultsArea = document.getElementById("results-area");
  resultsArea.classList.remove("hidden");
  resultsArea.scrollIntoView({ behavior: "smooth", block: "start" });

  document.getElementById("verdict-model-name").innerText = evalRes.top_model;
  const bestFit = evalRes.top_fitness !== undefined ? evalRes.top_fitness : evalRes.top_probability;
  document.getElementById("verdict-prob-badge").innerText = `${(bestFit * 100).toFixed(1)}%`;
  document.getElementById("stat-margin").innerText = `+${(evalRes.margin * 100).toFixed(1)}%`;
  document.getElementById("stat-entropy").innerText = `${evalRes.entropy} bit`;
  document.getElementById("stat-n1").innerText = evalRes.unique_probes_tested;
  document.getElementById("stat-n2").innerText = evalRes.sample_count;

  // Render Sample Size & Uncertainty Advisory Banner
  renderSampleAdvisory(evalRes);

  const displayScores = evalRes.independent_fitness || evalRes.posteriors;
  renderForestPlot(
    displayScores,
    evalRes.confidence_intervals_95 || evalRes.confidence_intervals,
    evalRes.log_likelihoods,
    evalRes.posteriors,
    evalRes.confidence_intervals_68 || {},
    evalRes.model_statistics || {},
    evalRes.summary_statistics || {}
  );
  renderClusterCanvas(clusterData);

  const tbody = document.getElementById("parsed-table-body");
  tbody.innerHTML = "";
  evalRes.parsed_submissions.forEach(rec => {
    const tr = document.createElement("tr");
    const tokenBadges = rec.parsed_tokens.map(tok => 
      `<span class="px-2 py-0.5 rounded bg-sky-50 dark:bg-sky-950/80 text-sky-700 dark:text-sky-300 border border-sky-200 dark:border-sky-800/70 mr-1.5 font-bold">${formatDisplayToken(tok)}</span>`
    ).join("");

    const probeItem = PROBE_I18N[rec.probe_id];
    const probeTitle = probeItem ? (currentLang === "en" ? probeItem.en_title : probeItem.zh_title) : rec.probe_id;

    tr.innerHTML = `
      <td class="p-3 font-mono text-slate-800 dark:text-slate-300 font-semibold" title="${probeTitle}">${rec.probe_id}</td>
      <td class="p-3">${tokenBadges}</td>
      <td class="p-3">${rec.strictly_complied ? `<span class="text-emerald-600 dark:text-emerald-400 font-bold">${t("comp_json")}</span>` : `<span class="text-amber-600 dark:text-amber-400">${t("comp_codeblock")}</span>`}</td>
    `;
    tbody.appendChild(tr);
  });
}

function formatMathSqrtN() {
  if (window.katex && typeof window.katex.renderToString === "function") {
    try {
      return window.katex.renderToString("1/\\sqrt{n}", { displayMode: false, throwOnError: false });
    } catch (e) {
      console.warn("KaTeX renderToString error:", e);
    }
  }
  return '<span class="inline-flex items-center font-mono font-bold">1/&radic;<span style="text-decoration:overline; padding-left:1px;">n</span></span>';
}

/**
 * Render Sample Size & Uncertainty Advisory Banner
 */
function renderSampleAdvisory(evalRes) {
  const container = document.getElementById("sample-size-advisory-container");
  if (!container) return;

  const advisory = evalRes.sample_advisory || {};
  const isLow = advisory.is_low_sample !== undefined ? advisory.is_low_sample : (evalRes.sample_count <= 8);
  const count = evalRes.sample_count || (evalRes.parsed_submissions ? evalRes.parsed_submissions.length : 1);

  container.classList.remove("hidden");

  if (isLow) {
    const ciRadius = advisory.ci_radius_pct !== undefined ? advisory.ci_radius_pct : 23.8;
    const ciSpan = advisory.ci_span_pct !== undefined ? advisory.ci_span_pct : 47.6;
    const shrink = advisory.reduction_pct !== undefined ? advisory.reduction_pct : 42.3;
    const newRadius = advisory.expected_radius_pct !== undefined ? advisory.expected_radius_pct : 13.8;
    const recom = advisory.recommended_samples || 15;

    const mathHtml = `<span class="math-rate-badge inline-flex items-center align-middle mx-1 font-mono font-bold bg-amber-500/20 dark:bg-amber-400/20 text-amber-950 dark:text-amber-100 px-1.5 py-0.5 rounded shadow-xs">${formatMathSqrtN()}</span>`;

    let desc = t("advisory_desc_tpl")
      .replace("{count}", count)
      .replace("{radius}", ciRadius)
      .replace("{shrink}", shrink)
      .replace("{newRadius}", newRadius)
      .replace("{math_rate}", mathHtml)
      .replace("1/√n", mathHtml);

    container.innerHTML = `
      <div class="rounded-xl border border-amber-500/40 bg-amber-500/10 dark:border-amber-400/30 dark:bg-amber-950/40 p-4 transition-all duration-200">
        <div class="flex items-start gap-3">
          <div class="p-1.5 rounded-lg bg-amber-500/20 text-amber-600 dark:text-amber-400 shrink-0 mt-0.5">
            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
            </svg>
          </div>
          <div class="flex-1 space-y-2">
            <div class="flex flex-wrap items-center justify-between gap-2">
              <h4 class="text-sm font-bold text-amber-900 dark:text-amber-200">${t("advisory_title")}</h4>
              <span class="text-xs font-mono px-2.5 py-0.5 rounded-full bg-amber-200/70 dark:bg-amber-900/80 text-amber-900 dark:text-amber-200 font-semibold border border-amber-300 dark:border-amber-700/60">${t("advisory_badge")}</span>
            </div>
            <p class="text-xs sm:text-sm text-amber-950/85 dark:text-amber-200/90 leading-relaxed font-sans">${desc}</p>
            <div class="flex flex-wrap items-center gap-x-4 gap-y-1.5 pt-1 text-xs text-amber-800 dark:text-amber-300 font-mono">
              <span class="inline-flex items-center gap-1">• <span>${t("advisory_current")}</span><strong>${count} ${t("advisory_unit_items")}</strong></span>
              <span class="inline-flex items-center gap-1">• <span>${t("advisory_ci_span")}</span><strong>±${ciRadius}%</strong> <span class="opacity-75">(${t("advisory_span_label")}: ${ciSpan}%)</span></span>
              <span class="inline-flex items-center gap-1">• <span>${t("advisory_recom")}</span><strong>≥${recom} ${t("advisory_unit_items")}</strong> <span class="opacity-80">(${t("advisory_shrink")}: ~${shrink}%)</span></span>
            </div>
          </div>
        </div>
      </div>
    `;
  } else {
    let msg = t("advisory_sufficient").replace("{count}", count);
    container.innerHTML = `
      <div class="rounded-xl border border-emerald-500/30 bg-emerald-500/10 dark:border-emerald-400/20 dark:bg-emerald-950/30 px-4 py-3 transition-all duration-200">
        <div class="flex items-center gap-2.5 text-xs sm:text-sm text-emerald-800 dark:text-emerald-300 font-sans">
          <svg class="w-4 h-4 text-emerald-600 dark:text-emerald-400 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path>
          </svg>
          <span>${msg}</span>
        </div>
      </div>
    `;
  }
}

/**
 * Forest Plot Error Bar Rows (Academic Statistical Box Plot)
 * Dual-theme & bilingual adaptive
 */
function renderForestPlot(
  fitnessScores,
  confidenceIntervals,
  logLikelihoods,
  posteriors = {},
  ci68Map = {},
  modelStats = {},
  summaryStats = {}
) {
  const container = document.getElementById("forest-plot-container");
  if (!container) return;

  const modelColorMap = {};
  MODELS_DATA.forEach(m => { modelColorMap[m.name] = m.color; });

  const sortedEntries = Object.entries(fitnessScores).sort((a, b) => b[1] - a[1]);

  let html = `
    <!-- Forest Plot Axis Scale Header & Interval Legend -->
    <div class="px-4 py-2.5 grid grid-cols-12 gap-4 text-xs text-slate-500 dark:text-slate-400 font-mono border-b border-slate-200 dark:border-slate-800/80 items-center">
      <div class="col-span-12 sm:col-span-5 font-semibold flex items-center justify-between">
        <span>${t("forest_col_model")}</span>
        <div class="flex items-center gap-2.5 text-[11px] font-normal">
          <span class="inline-flex items-center gap-1.5 text-slate-600 dark:text-slate-300">
            <span class="w-3 h-2 rounded-xs bg-sky-500/40 border border-sky-500 inline-block"></span>
            <span>${t("ci_legend_68")}</span>
          </span>
          <span class="inline-flex items-center gap-1.5 text-slate-400">
            <span class="w-3 h-0.5 bg-slate-400 inline-block"></span>
            <span>${t("ci_legend_95")}</span>
          </span>
        </div>
      </div>
      <div class="col-span-12 sm:col-span-7 relative">
        <div class="flex justify-between w-full text-xs px-1">
          <span>0%</span>
          <span>25%</span>
          <span>50%</span>
          <span>75%</span>
          <span>100%</span>
        </div>
      </div>
    </div>
    <div class="space-y-3 pt-1.5">
  `;

  sortedEntries.forEach(([model, prob]) => {
    const color = modelColorMap[model] || "#0284c7";
    const probPct = (prob * 100).toFixed(1);
    
    // 95% Confidence Interval (full outer bounds)
    const ci95 = confidenceIntervals[model] || [prob, prob];
    const ci95LowPct = Math.max(0, ci95[0] * 100);
    const ci95HighPct = Math.min(100, ci95[1] * 100);
    const ci95WidthPct = Math.max(0.6, ci95HighPct - ci95LowPct);

    // 68% Confidence Interval (core central probability mass)
    const ci68 = ci68Map[model] || [prob, prob];
    const ci68LowPct = Math.max(0, ci68[0] * 100);
    const ci68HighPct = Math.min(100, ci68[1] * 100);
    const ci68WidthPct = Math.max(0.8, ci68HighPct - ci68LowPct);

    const centerPct = Math.min(100, Math.max(0, prob * 100));
    const ll = logLikelihoods && logLikelihoods[model] !== undefined ? logLikelihoods[model].toFixed(2) : "-";
    const normP = posteriors[model] !== undefined ? (posteriors[model] * 100).toFixed(1) : null;
    const mStat = modelStats[model] || {};
    const seVal = mStat.standard_error !== undefined ? mStat.standard_error : "-";
    const deltaVal = mStat.delta_ll !== undefined ? (mStat.delta_ll > 0 ? `+${mStat.delta_ll}` : `${mStat.delta_ll}`) : "-";
    const refN = mStat.ref_samples || "-";

    const tooltipText = `${model}: ${t("forest_tooltip_fit")} ${probPct}% | 68% CI: [${ci68LowPct.toFixed(1)}%, ${ci68HighPct.toFixed(1)}%] | 95% CI: [${ci95LowPct.toFixed(1)}%, ${ci95HighPct.toFixed(1)}%] | ΔLL: ${deltaVal} | SE: ${seVal} | N_ref: ${refN}`;

    html += `
      <div class="bg-white dark:bg-slate-900/70 border border-slate-200 dark:border-slate-800/90 rounded-lg p-3.5 sm:p-4 grid grid-cols-12 gap-3 sm:gap-4 items-center hover:border-slate-300 dark:hover:border-slate-700 transition shadow-2xs">
        <!-- Model Info Column: Only display model name, point estimate, and 95% CI -->
        <div class="col-span-12 sm:col-span-5 flex items-center justify-between gap-2 min-w-0">
          <div class="flex items-center gap-2 min-w-0">
            <span class="w-3.5 h-3.5 rounded-full shrink-0 shadow-xs" style="background-color: ${color}"></span>
            <span class="font-bold text-slate-900 dark:text-white text-sm truncate" title="${model}">${model}</span>
          </div>
          <div class="flex items-center gap-2 font-mono shrink-0">
            <span class="text-sm font-bold text-sky-600 dark:text-sky-400">${probPct}%</span>
            <!-- Clean single 95% interval pill badge with hover explanation -->
            <span class="text-xs font-semibold text-slate-700 dark:text-slate-300 bg-slate-100 dark:bg-slate-950 px-2 py-0.5 rounded border border-slate-200 dark:border-slate-800 shadow-2xs cursor-help" title="${t("ci_legend_95")}: [${ci95LowPct.toFixed(1)}% ~ ${ci95HighPct.toFixed(1)}%] (${t("ci_legend_68")}: [${ci68LowPct.toFixed(1)}% ~ ${ci68HighPct.toFixed(1)}%])">
              [${ci95LowPct.toFixed(1)}% ~ ${ci95HighPct.toFixed(1)}%]
            </span>
          </div>
        </div>

        <!-- Statistical Error Bar Coordinate Strip (0% to 100%) -->
        <div class="col-span-12 sm:col-span-7">
          <div class="relative w-full h-9 bg-slate-100 dark:bg-slate-950 border border-slate-200 dark:border-slate-800/90 rounded-md flex items-center px-1 group cursor-default" title="${tooltipText}">
            <!-- Vertical Grid Reference Lines at 0%, 25%, 50%, 75%, 100% -->
            <div class="absolute inset-0 flex justify-between pointer-events-none opacity-25">
              <div class="border-r border-slate-400 dark:border-slate-600 h-full w-0"></div>
              <div class="border-r border-slate-400 dark:border-slate-600 h-full w-0"></div>
              <div class="border-r border-slate-400 dark:border-slate-600 h-full w-0"></div>
              <div class="border-r border-slate-400 dark:border-slate-600 h-full w-0"></div>
              <div class="border-r border-slate-400 dark:border-slate-600 h-full w-0"></div>
            </div>

            <!-- 95% Confidence Interval Whisker Line (Thin Whisker) -->
            <div class="absolute h-1 rounded-full cursor-help hover:h-1.5 transition-all z-0" style="left: ${ci95LowPct}%; width: ${ci95WidthPct}%; background-color: ${color}; opacity: 0.7;" title="${t("ci_legend_95")}: [${ci95LowPct.toFixed(1)}% ~ ${ci95HighPct.toFixed(1)}%]"></div>

            <!-- Left Whisker End Cap -->
            <div class="absolute w-1 h-3.5 rounded-xs cursor-help hover:w-1.5 transition-all z-0" style="left: ${ci95LowPct}%; background-color: ${color}; top: 50%; transform: translateY(-50%); opacity: 0.85;" title="${t("ci_legend_95")}: [${ci95LowPct.toFixed(1)}% ~ ${ci95HighPct.toFixed(1)}%]"></div>

            <!-- Right Whisker End Cap -->
            <div class="absolute w-1 h-3.5 rounded-xs cursor-help hover:w-1.5 transition-all z-0" style="left: ${ci95HighPct}%; background-color: ${color}; top: 50%; transform: translateY(-50%); opacity: 0.85;" title="${t("ci_legend_95")}: [${ci95LowPct.toFixed(1)}% ~ ${ci95HighPct.toFixed(1)}%]"></div>

            <!-- 68% Core Confidence Interval Pill Band (Thick Band) -->
            <div class="absolute h-3.5 rounded-full cursor-help transition-all shadow-2xs hover:h-4.5 hover:opacity-60 z-10" style="left: ${ci68LowPct}%; width: ${ci68WidthPct}%; background-color: ${color}; opacity: 0.38; border: 1.5px solid ${color}90;" title="${t("ci_legend_68")}: [${ci68LowPct.toFixed(1)}% ~ ${ci68HighPct.toFixed(1)}%]"></div>

            <!-- Point Estimate Marker (Center Circle) -->
            <div class="absolute w-4 h-4 rounded-full border-2 border-white dark:border-slate-950 shadow-md z-20 cursor-help transition-transform hover:scale-135" style="left: ${centerPct}%; background-color: ${color}; top: 50%; transform: translate(-50%, -50%);" title="${t("ci_point_tip")}: ${probPct}% (LL: ${ll})"></div>
          </div>
        </div>
      </div>
    `;
  });

  html += `</div>`;

  // Render Collapsible Full Statistical Diagnostics Table
  if (Object.keys(modelStats).length > 0) {
    html += `
      <div class="pt-3">
        <button type="button" onclick="toggleStatDetails()" class="inline-flex items-center gap-1.5 text-xs font-mono font-semibold text-sky-600 hover:text-sky-500 dark:text-sky-400 dark:hover:text-sky-300 transition cursor-pointer select-none">
          <svg id="stat-details-arrow" class="w-3.5 h-3.5 transition-transform duration-200" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
          <span>${t("stat_details_toggle")}</span>
        </button>

        <div id="stat-details-panel" class="hidden mt-2.5 overflow-x-auto rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/60 shadow-2xs">
          <table class="w-full text-xs font-mono text-left divide-y divide-slate-200 dark:divide-slate-800">
            <thead class="bg-slate-50 dark:bg-slate-950 text-slate-600 dark:text-slate-400">
              <tr>
                <th class="p-2.5">${t("th_stat_model")}</th>
                <th class="p-2.5">${t("th_stat_fitness")}</th>
                <th class="p-2.5">${t("th_stat_ci68")}</th>
                <th class="p-2.5">${t("th_stat_ci95")}</th>
                <th class="p-2.5">${t("th_stat_delta_ll")}</th>
                <th class="p-2.5">${t("th_stat_ll")}</th>
                <th class="p-2.5">${t("th_stat_se")}</th>
                <th class="p-2.5">${t("th_stat_nref")}</th>
              </tr>
            </thead>
            <tbody class="divide-y divide-slate-100 dark:divide-slate-800/60 text-slate-700 dark:text-slate-300">
    `;

    sortedEntries.forEach(([model, prob]) => {
      const color = modelColorMap[model] || "#0284c7";
      const probPct = (prob * 100).toFixed(1);
      const ci95 = confidenceIntervals[model] || [prob, prob];
      const ci68 = ci68Map[model] || [prob, prob];
      const mStat = modelStats[model] || {};
      const deltaVal = mStat.delta_ll !== undefined ? (mStat.delta_ll > 0 ? `+${mStat.delta_ll}` : `${mStat.delta_ll}`) : "-";
      const llVal = mStat.log_likelihood !== undefined ? mStat.log_likelihood : "-";
      const seVal = mStat.standard_error !== undefined ? mStat.standard_error : "-";
      const refN = mStat.ref_samples !== undefined ? mStat.ref_samples : "-";

      html += `
        <tr class="hover:bg-slate-50/70 dark:hover:bg-slate-800/30 transition">
          <td class="p-2.5 font-bold flex items-center gap-1.5">
            <span class="w-2.5 h-2.5 rounded-full shrink-0" style="background:${color}"></span>
            <span class="truncate">${model}</span>
          </td>
          <td class="p-2.5 font-bold text-sky-600 dark:text-sky-400">${probPct}%</td>
          <td class="p-2.5 text-slate-800 dark:text-slate-200 font-semibold">[${(ci68[0] * 100).toFixed(1)}% ~ ${(ci68[1] * 100).toFixed(1)}%]</td>
          <td class="p-2.5 text-slate-500">[${(ci95[0] * 100).toFixed(1)}% ~ ${(ci95[1] * 100).toFixed(1)}%]</td>
          <td class="p-2.5 ${mStat.delta_ll > 0 ? 'text-emerald-600 dark:text-emerald-400 font-bold' : 'text-slate-400'}">${deltaVal}</td>
          <td class="p-2.5 text-slate-500">${llVal}</td>
          <td class="p-2.5 text-slate-600 dark:text-slate-400">${seVal}</td>
          <td class="p-2.5 text-slate-600 dark:text-slate-400">${refN}</td>
        </tr>
      `;
    });

    html += `
            </tbody>
          </table>
        </div>
      </div>
    `;
  }

  container.innerHTML = html;
}

function toggleStatDetails() {
  const panel = document.getElementById("stat-details-panel");
  const arrow = document.getElementById("stat-details-arrow");
  if (!panel) return;
  panel.classList.toggle("hidden");
  if (arrow) {
    arrow.classList.toggle("rotate-180");
  }
}

/**
 * Draw True Five-Pointed Star on Canvas
 */
function drawStar(ctx, cx, cy, spikes = 5, outerRadius = 9.5, innerRadius = 4.8, fillStyle = "#f59e0b", strokeStyle = "#ffffff", lineWidth = 2.0) {
  let rot = (Math.PI / 2) * 3;
  let x = cx;
  let y = cy;
  const step = Math.PI / spikes;

  ctx.beginPath();
  ctx.moveTo(cx, cy - outerRadius);
  for (let i = 0; i < spikes; i++) {
    x = cx + Math.cos(rot) * outerRadius;
    y = cy + Math.sin(rot) * outerRadius;
    ctx.lineTo(x, y);
    rot += step;

    x = cx + Math.cos(rot) * innerRadius;
    y = cy + Math.sin(rot) * innerRadius;
    ctx.lineTo(x, y);
    rot += step;
  }
  ctx.lineTo(cx, cy - outerRadius);
  ctx.closePath();

  if (fillStyle) {
    ctx.fillStyle = fillStyle;
    ctx.fill();
  }
  if (strokeStyle) {
    ctx.strokeStyle = strokeStyle;
    ctx.lineWidth = lineWidth;
    ctx.stroke();
  }
}

// 2D PCA Interactive Highlight State
let pcaHighlightedModel = null;
let pcaLockedModel = null;
let pcaInteractiveElements = [];
let pcaListenersSetup = false;

/**
 * 2D PCA Cluster Canvas with Adaptive Scaling, Confidence Regions, Interactive Legend & Highlights
 */
function renderClusterCanvas(clusterData, overrideHighlight = undefined) {
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

  // Active highlighted model (null if all active)
  const activeModel = overrideHighlight !== undefined ? overrideHighlight : (pcaHighlightedModel || pcaLockedModel);

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

  // 1. True Adaptive Bounding Box
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  clusterData.clusters.forEach(c => {
    (c.points || []).forEach(p => {
      minX = Math.min(minX, p[0]);
      maxX = Math.max(maxX, p[0]);
      minY = Math.min(minY, p[1]);
      maxY = Math.max(maxY, p[1]);
    });
    if (c.center) {
      minX = Math.min(minX, c.center[0]);
      maxX = Math.max(maxX, c.center[0]);
      minY = Math.min(minY, c.center[1]);
      maxY = Math.max(maxY, c.center[1]);
    }
  });

  if (clusterData.user_point) {
    minX = Math.min(minX, clusterData.user_point[0]);
    maxX = Math.max(maxX, clusterData.user_point[0]);
    minY = Math.min(minY, clusterData.user_point[1]);
    maxY = Math.max(maxY, clusterData.user_point[1]);
  }

  if (!isFinite(minX) || !isFinite(maxX) || minX === maxX) {
    minX = -0.5; maxX = 0.5; minY = -0.5; maxY = 0.5;
  }

  // Generous padding so ellipses and labels never clip
  const spanX = Math.max(maxX - minX, 0.1);
  const spanY = Math.max(maxY - minY, 0.1);
  const padX = spanX * 0.28;
  const padY = spanY * 0.28;
  minX -= padX; maxX += padX;
  minY -= padY; maxY += padY;

  function toScreen(x, y) {
    const sx = ((x - minX) / (maxX - minX)) * (w - 70) + 35;
    const sy = h - (((y - minY) / (maxY - minY)) * (h - 70) + 35);
    return [sx, sy];
  }

  const scaleX = (w - 70) / (maxX - minX);
  const scaleY = (h - 70) / (maxY - minY);

  // Helper: compute covariance confidence ellipse
  function computeConfidenceEllipse(points) {
    if (!points || points.length < 3) return null;
    const n = points.length;
    let mx = 0, my = 0;
    for (const p of points) { mx += p[0]; my += p[1]; }
    mx /= n; my /= n;

    let sxx = 0, syy = 0, sxy = 0;
    for (const p of points) {
      const dx = p[0] - mx;
      const dy = p[1] - my;
      sxx += dx * dx;
      syy += dy * dy;
      sxy += dx * dy;
    }
    sxx /= n; syy /= n; sxy /= n;

    const trace = sxx + syy;
    const det = sxx * syy - sxy * sxy;
    const disc = Math.sqrt(Math.max(0, (trace * trace) / 4 - det));
    const l1 = Math.max(0.00001, trace / 2 + disc);
    const l2 = Math.max(0.00001, trace / 2 - disc);
    const angle = 0.5 * Math.atan2(2 * sxy, sxx - syy);

    const scale = 2.45; // ~95% confidence coverage
    return {
      mx, my,
      rx: Math.sqrt(l1) * scale,
      ry: Math.sqrt(l2) * scale,
      angle: angle
    };
  }

  pcaInteractiveElements = [];

  // 2. Pass 1: Draw Confidence Region (范围)
  clusterData.clusters.forEach(c => {
    const isSelected = activeModel === c.model_name;
    const isMuted = activeModel !== null && !isSelected;
    const baseColor = c.color || "#0284c7";

    const ell = computeConfidenceEllipse(c.points);
    if (ell) {
      const [csx, csy] = toScreen(ell.mx, ell.my);
      const sRx = Math.max(14, ell.rx * scaleX);
      const sRy = Math.max(14, ell.ry * scaleY);

      ctx.save();
      ctx.translate(csx, csy);
      ctx.rotate(-ell.angle);

      if (isMuted) {
        ctx.fillStyle = isDark ? "rgba(100, 116, 139, 0.04)" : "rgba(203, 213, 225, 0.15)";
        ctx.beginPath();
        ctx.ellipse(0, 0, sRx, sRy, 0, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = isDark ? "rgba(71, 85, 105, 0.15)" : "rgba(203, 213, 225, 0.35)";
        ctx.lineWidth = 1.0;
        ctx.setLineDash([3, 4]);
        ctx.stroke();
      } else if (isSelected) {
        ctx.fillStyle = baseColor + "38"; // ~22% opacity
        ctx.beginPath();
        ctx.ellipse(0, 0, sRx, sRy, 0, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = baseColor;
        ctx.lineWidth = 2.0;
        ctx.setLineDash([]);
        ctx.stroke();
      } else {
        // Normal all-active state
        ctx.fillStyle = baseColor + "18"; // ~10% opacity
        ctx.beginPath();
        ctx.ellipse(0, 0, sRx, sRy, 0, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = baseColor + "66"; // ~40% opacity
        ctx.lineWidth = 1.2;
        ctx.setLineDash([4, 4]);
        ctx.stroke();
      }

      ctx.restore();
    }
  });

  // 3. Pass 2: Draw Individual Scatter Points (点)
  clusterData.clusters.forEach(c => {
    const isSelected = activeModel === c.model_name;
    const isMuted = activeModel !== null && !isSelected;
    const baseColor = c.color || "#0284c7";

    if (isMuted) {
      ctx.fillStyle = isDark ? "rgba(100, 116, 139, 0.25)" : "rgba(203, 213, 225, 0.5)";
      c.points.forEach(p => {
        const [sx, sy] = toScreen(p[0], p[1]);
        ctx.beginPath();
        ctx.arc(sx, sy, 1.5, 0, Math.PI * 2);
        ctx.fill();
      });
    } else if (isSelected) {
      ctx.fillStyle = baseColor + "ff";
      c.points.forEach(p => {
        const [sx, sy] = toScreen(p[0], p[1]);
        ctx.beginPath();
        ctx.arc(sx, sy, 2.5, 0, Math.PI * 2);
        ctx.fill();
      });
    } else {
      ctx.fillStyle = baseColor + "aa";
      c.points.forEach(p => {
        const [sx, sy] = toScreen(p[0], p[1]);
        ctx.beginPath();
        ctx.arc(sx, sy, 2.0, 0, Math.PI * 2);
        ctx.fill();
      });
    }

    // Centroid Anchor Point
    const [cx, cy] = toScreen(c.center[0], c.center[1]);
    pcaInteractiveElements.push({
      type: "model",
      name: c.model_name,
      color: baseColor,
      sx: cx,
      sy: cy,
      x: c.center[0],
      y: c.center[1],
      points_len: c.sample_count || (c.points || []).length
    });

    if (isMuted) {
      ctx.beginPath();
      ctx.arc(cx, cy, 3.0, 0, Math.PI * 2);
      ctx.fillStyle = isDark ? "rgba(71, 85, 105, 0.4)" : "rgba(203, 213, 225, 0.7)";
      ctx.fill();
    } else if (isSelected) {
      ctx.beginPath();
      ctx.arc(cx, cy, 6.0, 0, Math.PI * 2);
      ctx.fillStyle = baseColor;
      ctx.fill();
      ctx.strokeStyle = isDark ? "#ffffff" : "#0f172a";
      ctx.lineWidth = 2.0;
      ctx.stroke();

      // Show floating label badge ONLY for highlighted/selected model to keep canvas clean!
      ctx.font = "bold 12px Inter, sans-serif";
      const textWidth = ctx.measureText(c.model_name).width;
      const offsetX = cx > w * 0.5 ? 12 : -textWidth - 16;
      const offsetY = cy > h * 0.5 ? 16 : -10;
      const labelX = cx + offsetX;
      const labelY = cy + offsetY;

      ctx.fillStyle = isDark ? "rgba(15, 23, 42, 0.92)" : "rgba(255, 255, 255, 0.96)";
      ctx.fillRect(labelX - 4, labelY - 13, textWidth + 8, 18);
      ctx.strokeStyle = baseColor;
      ctx.lineWidth = 1.5;
      ctx.strokeRect(labelX - 4, labelY - 13, textWidth + 8, 18);

      ctx.fillStyle = isDark ? "#f8fafc" : "#0f172a";
      ctx.fillText(c.model_name, labelX, labelY);
    } else {
      ctx.beginPath();
      ctx.arc(cx, cy, 4.2, 0, Math.PI * 2);
      ctx.fillStyle = baseColor;
      ctx.fill();
      ctx.strokeStyle = isDark ? "#ffffff" : "#0f172a";
      ctx.lineWidth = 1.4;
      ctx.stroke();
    }
  });

  // 4. Pass 3: Draw User Test Sample Point (★ True Star on the TOPMOST layer)
  if (clusterData.user_point) {
    const [ux, uy] = toScreen(clusterData.user_point[0], clusterData.user_point[1]);
    pcaInteractiveElements.push({
      type: "user",
      name: t("pca_user_point"),
      color: "#f59e0b",
      sx: ux,
      sy: uy,
      x: clusterData.user_point[0],
      y: clusterData.user_point[1]
    });

    // Outer gentle glowing pulse aura ring
    ctx.fillStyle = "rgba(245, 158, 11, 0.22)";
    ctx.beginPath();
    ctx.arc(ux, uy, 16, 0, Math.PI * 2);
    ctx.fill();

    // Central crisp golden 5-pointed STAR
    drawStar(
      ctx,
      ux,
      uy,
      5,
      9.5,   // outer radius
      4.5,   // inner radius
      "#f59e0b",
      isDark ? "#ffffff" : "#0f172a",
      2.0
    );

    // Callout pill tag (ALWAYS on top)
    ctx.font = "bold 12px Inter, sans-serif";
    const userText = t("pca_user_point");
    const uWidth = ctx.measureText(userText).width;
    const uX = Math.max(10, Math.min(w - uWidth - 22, ux - uWidth / 2));
    const uY = uy > 42 ? uy - 18 : uy + 28;

    ctx.fillStyle = isDark ? "rgba(15, 23, 42, 0.95)" : "rgba(255, 255, 255, 0.98)";
    ctx.fillRect(uX - 6, uY - 13, uWidth + 12, 18);
    ctx.strokeStyle = "#f59e0b";
    ctx.lineWidth = 1.8;
    ctx.strokeRect(uX - 6, uY - 13, uWidth + 12, 18);

    ctx.fillStyle = isDark ? "#fbbf24" : "#b45309";
    ctx.fillText(userText, uX, uY);
  }

  // 5. Update Interactive Legend Bar
  renderClusterLegend(clusterData, activeModel);

  // 6. Setup Mouse Listeners (Once)
  setupPcaCanvasListeners(canvas, clusterData);
}

/**
 * Render Interactive Legend Chips above PCA Canvas
 */
function renderClusterLegend(clusterData, activeModel) {
  const container = document.getElementById("pca-legend-container");
  if (!container) return;

  const isAllActive = activeModel === null;

  // Build chips if not already built
  if (container.children.length === 0) {
    let html = `
      <button type="button" class="pca-legend-chip px-2.5 py-1 rounded-md text-xs font-mono font-bold transition flex items-center gap-1.5 cursor-pointer border shadow-2xs" data-model="__ALL__">
        <span class="w-2 h-2 rounded-full chip-dot"></span>
        <span>${t("pca_legend_all")}</span>
      </button>
    `;

    clusterData.clusters.forEach(c => {
      const color = c.color || "#0284c7";
      html += `
        <button type="button" class="pca-legend-chip px-2.5 py-1 rounded-md text-xs font-mono font-medium transition flex items-center gap-1.5 cursor-pointer border shadow-2xs" data-model="${c.model_name}" data-color="${color}">
          <span class="w-2.5 h-2.5 rounded-full shrink-0 shadow-2xs" style="background-color: ${color};"></span>
          <span class="truncate max-w-[130px]">${c.model_name}</span>
        </button>
      `;
    });

    if (clusterData.user_point) {
      html += `
        <button type="button" class="pca-legend-chip px-2.5 py-1 rounded-md text-xs font-mono font-bold transition flex items-center gap-1.5 cursor-pointer border border-amber-500/60 bg-amber-50 dark:bg-amber-950/40 text-amber-600 dark:text-amber-400" data-model="__USER__">
          <span>${t("pca_user_point")}</span>
        </button>
      `;
    }

    container.innerHTML = html;

    container.querySelectorAll(".pca-legend-chip").forEach(chip => {
      const mName = chip.getAttribute("data-model");

      chip.addEventListener("mouseenter", () => {
        if (mName === "__ALL__") {
          renderClusterCanvas(clusterData, null);
        } else {
          renderClusterCanvas(clusterData, mName);
        }
      });

      chip.addEventListener("mouseleave", () => {
        renderClusterCanvas(clusterData, pcaLockedModel);
      });

      chip.addEventListener("click", () => {
        if (mName === "__ALL__") {
          pcaLockedModel = null;
        } else {
          pcaLockedModel = (pcaLockedModel === mName ? null : mName);
        }
        renderClusterCanvas(clusterData, pcaLockedModel);
      });
    });
  }

  // Update styles of existing chips based on activeModel
  container.querySelectorAll(".pca-legend-chip").forEach(chip => {
    const mName = chip.getAttribute("data-model");
    const color = chip.getAttribute("data-color") || "#0284c7";

    if (mName === "__ALL__") {
      const dot = chip.querySelector(".chip-dot");
      if (isAllActive) {
        chip.className = "pca-legend-chip px-2.5 py-1 rounded-md text-xs font-mono font-bold transition flex items-center gap-1.5 cursor-pointer border bg-slate-800 text-white dark:bg-white dark:text-slate-900 border-transparent shadow-xs";
        if (dot) dot.className = "w-2 h-2 rounded-full chip-dot bg-sky-400";
      } else {
        chip.className = "pca-legend-chip px-2.5 py-1 rounded-md text-xs font-mono font-medium transition flex items-center gap-1.5 cursor-pointer border bg-slate-100 text-slate-600 dark:bg-slate-800/80 dark:text-slate-400 border-slate-200 dark:border-slate-700/80 hover:bg-slate-200 dark:hover:bg-slate-700 shadow-2xs";
        if (dot) dot.className = "w-2 h-2 rounded-full chip-dot bg-slate-400";
      }
    } else if (mName === "__USER__") {
      const isUserSelected = activeModel === "__USER__";
      if (isUserSelected) {
        chip.className = "pca-legend-chip px-2.5 py-1 rounded-md text-xs font-mono font-bold transition flex items-center gap-1.5 cursor-pointer border border-amber-500 ring-2 ring-amber-500/30 bg-amber-100 dark:bg-amber-950/60 text-amber-700 dark:text-amber-300 shadow-xs";
      } else {
        chip.className = "pca-legend-chip px-2.5 py-1 rounded-md text-xs font-mono font-bold transition flex items-center gap-1.5 cursor-pointer border border-amber-500/60 bg-amber-50 dark:bg-amber-950/40 text-amber-600 dark:text-amber-400";
      }
    } else {
      const isSelected = activeModel === mName;
      const isMuted = activeModel !== null && !isSelected;

      if (isSelected) {
        chip.className = "pca-legend-chip px-2.5 py-1 rounded-md text-xs font-mono font-bold transition flex items-center gap-1.5 cursor-pointer border border-sky-500 shadow-xs ring-2 ring-sky-500/20";
        chip.style.backgroundColor = `${color}1a`;
        chip.style.borderColor = color;
        chip.style.color = color;
      } else if (isMuted) {
        chip.className = "pca-legend-chip px-2.5 py-1 rounded-md text-xs font-mono font-medium transition flex items-center gap-1.5 cursor-pointer border opacity-35 hover:opacity-100 border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-900 text-slate-500 shadow-2xs";
        chip.style.backgroundColor = "";
        chip.style.borderColor = "";
        chip.style.color = "";
      } else {
        chip.className = "pca-legend-chip px-2.5 py-1 rounded-md text-xs font-mono font-medium transition flex items-center gap-1.5 cursor-pointer border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 text-slate-700 dark:text-slate-300 hover:border-slate-300 dark:hover:border-slate-700 shadow-2xs";
        chip.style.backgroundColor = "";
        chip.style.borderColor = "";
        chip.style.color = "";
      }
    }
  });
}

/**
 * Setup Mouse Hover / Crosshair Listeners on PCA Canvas
 */
function setupPcaCanvasListeners(canvas, clusterData) {
  if (pcaListenersSetup) return;
  pcaListenersSetup = true;

  const tooltip = document.getElementById("pca-tooltip");

  canvas.addEventListener("mousemove", (e) => {
    if (!currentClusterData) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    let nearest = null;
    let minDist = 26;

    for (const el of pcaInteractiveElements) {
      const dist = Math.hypot(el.sx - mx, el.sy - my);
      if (dist < minDist) {
        minDist = dist;
        nearest = el;
      }
    }

    if (nearest) {
      if (tooltip) {
        tooltip.classList.remove("hidden");
        const tipX = Math.min(rect.width - 160, Math.max(10, mx + 14));
        const tipY = my > 50 ? my - 45 : my + 18;
        tooltip.style.left = `${tipX}px`;
        tooltip.style.top = `${tipY}px`;

        if (nearest.type === "user") {
          tooltip.className = "absolute pointer-events-none px-3 py-1.5 rounded-lg text-xs font-mono shadow-xl border z-30 transition-all duration-75 bg-amber-50 dark:bg-slate-900 border-amber-500 text-amber-700 dark:text-amber-400";
          tooltip.innerHTML = `<span class="font-bold">${t("pca_user_point")}</span><div class="text-[11px] text-slate-500 dark:text-slate-400">${t("pca_coord_label")}(${nearest.x.toFixed(3)}, ${nearest.y.toFixed(3)})</div>`;
        } else {
          tooltip.className = "absolute pointer-events-none px-3 py-1.5 rounded-lg text-xs font-mono shadow-xl border z-30 transition-all duration-75 bg-white dark:bg-slate-900 border-slate-300 dark:border-slate-700 text-slate-800 dark:text-slate-200";
          tooltip.innerHTML = `
            <div class="flex items-center gap-1.5 font-bold">
              <span class="w-2.5 h-2.5 rounded-full" style="background:${nearest.color}"></span>
              <span>${nearest.name}</span>
            </div>
            <div class="text-[11px] text-slate-500 dark:text-slate-400 mt-0.5">
              ${t("pca_coord_label")}(${nearest.x.toFixed(3)}, ${nearest.y.toFixed(3)}) | ${t("pca_samples_label")}${nearest.points_len}
            </div>
          `;
        }
      }
      renderClusterCanvas(currentClusterData, nearest.name);
    } else {
      if (tooltip) tooltip.classList.add("hidden");
      renderClusterCanvas(currentClusterData, pcaLockedModel);
    }
  });

  canvas.addEventListener("mouseleave", () => {
    if (tooltip) tooltip.classList.add("hidden");
    renderClusterCanvas(currentClusterData, pcaLockedModel);
  });
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
    cachedStats = stats;
    updateHeaderStats(stats);

    document.getElementById("db-total-samples").innerText = stats.total_samples;
    document.getElementById("db-total-tokens").innerText = (stats.token_usage?.grand_total_tokens || tokenUsage.overall?.grand_total_tokens || 96967).toLocaleString();
    document.getElementById("db-api-calls").innerText = stats.token_usage?.recorded_api_calls || 240;
    const posCount = stats.positional_token_count || 345;
    const posEl = document.getElementById("db-pos-tokens");
    if (posEl) posEl.innerText = `${posCount} ${t("db_pos_unit")}`;
    
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

/**
 * Archive Export & Import System
 */
async function exportArchiveZip() {
  const btn = document.getElementById("btn-export-archive");
  const origText = btn ? btn.innerHTML : "";
  showToast(t("toast_exporting"), "info");

  try {
    if (btn) btn.classList.add("opacity-60", "pointer-events-none");
    const response = await fetch("/api/archive/export");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    const disposition = response.headers.get("Content-Disposition");
    let filename = `statllm_archive_${new Date().toISOString().slice(0, 10)}.zip`;
    if (disposition && disposition.includes("filename=")) {
      const match = disposition.match(/filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/);
      if (match && match[1]) filename = match[1].replace(/['"]/g, '');
    }

    const blob = await response.blob();
    const downloadUrl = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = downloadUrl;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.URL.revokeObjectURL(downloadUrl);
  } catch (err) {
    showToast(t("toast_export_fail") + err.message, "error");
  } finally {
    if (btn) {
      btn.classList.remove("opacity-60", "pointer-events-none");
      btn.innerHTML = origText;
    }
  }
}

function openImportModal() {
  const modal = document.getElementById("import-modal");
  if (modal) modal.style.display = "flex";
}

function closeImportModal() {
  const modal = document.getElementById("import-modal");
  if (modal) modal.style.display = "none";
  const input = document.getElementById("archive-file-input");
  if (input) input.value = "";
}

async function onArchiveFileSelected(inputEl) {
  const file = inputEl.files && inputEl.files[0];
  if (!file) return;

  const mode = document.getElementById("import-mode-select")?.value || "merge";
  closeImportModal();
  showToast(t("toast_importing"), "info");

  try {
    const res = await fetch(`/api/archive/import?mode=${encodeURIComponent(mode)}`, {
      method: "POST",
      headers: { "Content-Type": "application/zip" },
      body: file
    });
    const data = await res.json();
    if (!res.ok || data.status !== "success") {
      throw new Error(data.detail || "Import error");
    }

    const successMsg = t("toast_import_success")
      .replace("{n}", data.imported_samples)
      .replace("{skip}", data.skipped_duplicates || 0);
    showToast(successMsg, "success");

    // Refresh database statistics and headers
    await refreshDbStats();
    updateHeaderStats({ total_samples: data.total_samples });
  } catch (err) {
    showToast(t("toast_import_fail") + err.message, "error");
  } finally {
    inputEl.value = "";
  }
}

function showToast(msg, type = "info") {
  const container = document.getElementById("toast-container");
  if (!container) return;

  const toast = document.createElement("div");
  const bg = type === "error" 
    ? "bg-rose-600 text-white" 
    : type === "success" 
      ? "bg-emerald-600 text-white" 
      : "bg-slate-800 text-white dark:bg-slate-100 dark:text-slate-900";

  toast.className = `pointer-events-auto px-4 py-2.5 rounded-xl shadow-lg text-xs sm:text-sm font-semibold flex items-center gap-2 transition transform translate-y-2 opacity-0 duration-200 ${bg}`;
  toast.textContent = msg;

  container.appendChild(toast);
  requestAnimationFrame(() => {
    toast.classList.remove("translate-y-2", "opacity-0");
  });

  setTimeout(() => {
    toast.classList.add("opacity-0", "translate-y-2");
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}

