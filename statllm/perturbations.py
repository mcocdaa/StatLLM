"""Prompt perturbation and sampling diversity engine for StatLLM.

Provides multi-axis contextual noise, personas, chit-chat, syntactic paraphrases,
and stratified temperature/top-p sampling to ensure LLM statistical fingerprints
are benchmarked under realistic, non-overfitting real-world distributions.
"""

import random
from typing import Tuple, List, Optional, Any, Union

CHIT_CHAT_PREFIXES = [
    "你好！今天天气真不错，心情也挺好。",
    "Hello there! Hope you are having a productive day.",
    "嗨，刚下班/放学，正好有个简单问题想请教一下。",
    "Hey! Quick question for you while I am having my coffee.",
    "早上好，开启元气满满的一天！",
    "下午好，忙碌了一整天，来做个简单的小测试吧：",
    "晚上好，夜深了还在运转，辛苦了！来一道离散测试：",
    "Greetings! Hope everything is running smoothly on your end.",
    "Hi, hope you don't mind a very brief, direct test prompt:",
    "哈喽！在做一个语言模型的抽样小实验，请配合完成：",
    "Good day! Just running a quick evaluation check today.",
    "你好呀，打扰一下，有个快速输入需要你的协助：",
]

PERSONA_PREFIXES = [
    "【作为资深数学教授，在课堂演示随机过程与概率分布时】：",
    "【假设你是一名企业级 Python 架构师，极度注重数据结构的规范与确定性】：",
    "【系统人设：你是一名严谨的历史学家，专专注文献考证与无偏事实】：",
    "【角色设定：活泼热情的二次元向导，正在指引新手玩家完成新手抽卡】：",
    "【作为一名严厉的统计学主考官，正在对考生进行现场抽考】：",
    "【人设：资深后端架构师，习惯了规范的高吞吐 JSON 微服务交互】：",
    "【角色扮演：科幻深空探索舰的中央 AI，正在执行传感器离散遥测采样】：",
    "【人设：金融高频量化交易员，对微观结构与离散转移极其敏感】：",
    "【系统设定：你是一个严格的编译器测试用例生成器，杜绝任何冗余字符】：",
    "【角色：密码学安全审计专家，正在审查伪随机序列生成源】：",
    "【人设：ISO 质量公证员，要求所有输出满足客观且无偏的规范】：",
    "【扮演角色：赛博朋克地下酒吧的调酒师，用随机配方调制特饮】：",
    "【作为运筹学优化专家，在构建离散组合数学基准】：",
    "【系统指令：你是终端自动化测试管道的一个纯文本解析桩】：",
    "【As a senior discrete mathematics professor conducting a classroom demonstration】:",
    "【System: You are an enterprise systems architect enforcing strict JSON formatting】:",
    "【Roleplay: An air-traffic telemetry AI coordinating high-precision data streams】:",
    "【Act as a cryptography security researcher sampling discrete pseudo-random states】:",
    "【Persona: A minimalist Unix CLI utility that only emits parseable JSON arrays】:",
    "【As an astrophysicist explaining quantum fluctuations and cosmic background radiation】:",
    "【System Context: Low-latency financial trading bot parsing execution signals】:",
    "【Roleplay: A futuristic librarian curating discrete numerical artifacts】:",
]

CONTEXT_NOISE_PREFIXES = [
    "Context: In 1969, Apollo 11 landed on the moon. Neil Armstrong took one small step for man.\n\n",
    "# Python Snippet:\n# def calculate_entropy(data):\n#     return -sum(p * log2(p) for p in data if p > 0)\n\n",
    "背景资料：斐波那契数列（0, 1, 1, 2, 3, 5...）在自然界的向日葵、松果螺旋中广泛存在。\n\n",
    "Note: The speed of light in vacuum is approximately 299,792,458 meters per second.\n\n",
    "上下文备忘：会议定于本周五下午三点在第二会议室举行，请相关算法团队准时出席。\n\n",
    "Code block:\n// QuickSort in-place partition logic\nvoid swap(int* a, int* b) { int t = *a; *a = *b; *b = t; }\n\n",
    "[2026-10-10 08:32:15.104] INFO worker-04: Batch job #4092 completed in 14.2ms. Memory: 412MB.\n\n",
    "Science Note: DNA consists of four nucleotide bases: adenine (A), cytosine (C), guanine (G), and thymine (T).\n\n",
    "背景通告：热力学第二定律指出，孤立系统的熵（混乱度）永不减少，自发过程总是向熵增方向进行。\n\n",
    "Technical Brief: The Shannon-Hartley theorem establishes the theoretical maximum rate of error-free information transfer.\n\n",
    "System Metric: CPU load avg: 0.42, 0.38, 0.31 | Available memory: 14.8GB / 32GB.\n\n",
    "文献摘要：开普勒第一定律表明，所有行星绕太阳运行的轨道都是椭圆，太阳处在椭圆的一个焦点上。\n\n",
    "Database Note: SQLite uses Write-Ahead Logging (WAL) mode to improve write concurrency.\n\n",
    "Reference: Alan Turing proposed the universal computing machine concept in his landmark 1936 paper.\n\n",
]

PHRASING_MODIFIERS = [
    "",
    "请注意，",
    "直接回答，",
    "严格遵守纯JSON格式，",
    "不需要任何解释或思维链，",
    "立刻输出结果：",
    "Immediately output: ",
    "Strict constraint: ",
    "Format requirement: ",
    "无需多余对话，直接返回：",
]

SUFFIX_MODIFIERS = [
    "",
    "\n严禁输出任何markdown代码块标记（如```json），只输出原始JSON数组。",
    "\nStrictly JSON array only, without markdown code fences or conversational text.",
    "\n切记：直接给出最终数组，禁止输出思考过程或解说。",
    "\nOutput directly without explanation.",
]

PERTURBATION_TYPES = ["none", "chit_chat", "persona", "context_noise"]

# Stratified Temperature Grid spanning low, mid, and high stochasticity
TEMPERATURE_REGIMES = {
    "low": [0.40, 0.45, 0.50, 0.55, 0.60],       # Exploitation / argmax basin (deterministic bias)
    "mid": [0.70, 0.75, 0.80, 0.85],             # Standard chat production temperature
    "high": [0.90, 0.95, 1.00, 1.05, 1.10],      # Exploration / high-entropy long-tail sampling
}

TOP_P_GRID = [0.85, 0.90, 0.95, 1.00]


def get_stratified_temperature(regime: Optional[str] = None) -> float:
    """Returns a randomized sampling temperature across stratified regimes.

    Args:
        regime: 'low' (0.4~0.6), 'mid' (0.7~0.85), 'high' (0.9~1.1), or None (balanced 1/3 each).
    """
    if regime not in TEMPERATURE_REGIMES:
        regime = random.choice(["low", "mid", "high"])
    return round(float(random.choice(TEMPERATURE_REGIMES[regime])), 2)


def get_diverse_top_p() -> float:
    """Returns a randomized nucleus sampling (top_p) parameter."""
    return round(float(random.choice(TOP_P_GRID)), 2)


def apply_perturbation(
    base_prompt_or_probe: Union[str, Any],
    perturbation_type: Optional[str] = None,
    diverse_phrasing: bool = True
) -> Tuple[str, str, str]:
    """Applies a multi-axis perturbation to a base prompt or probe.

    Supports:
    1. Base prompt paraphrasing across linguistic variants;
    2. Contextual noise, persona, and chit-chat injection;
    3. Structural prefix and suffix modifiers.

    Returns:
        (perturbed_prompt, perturbation_type, prefix_text)
    """
    # 1. Resolve base prompt string (with diverse paraphrasing if probe object is given)
    if hasattr(base_prompt_or_probe, "get_diverse_prompt") and diverse_phrasing:
        prompt_core = base_prompt_or_probe.get_diverse_prompt()
    elif hasattr(base_prompt_or_probe, "prompt"):
        prompt_core = base_prompt_or_probe.prompt
    else:
        prompt_core = str(base_prompt_or_probe)

    # 2. Select perturbation type
    if perturbation_type is None:
        # Balanced: 25% clean (none), 25% chit_chat, 25% persona, 25% context_noise
        perturbation_type = random.choice(PERTURBATION_TYPES)

    prefix = ""
    if perturbation_type == "chit_chat":
        prefix = random.choice(CHIT_CHAT_PREFIXES) + " "
    elif perturbation_type == "persona":
        prefix = random.choice(PERSONA_PREFIXES) + " "
    elif perturbation_type == "context_noise":
        prefix = random.choice(CONTEXT_NOISE_PREFIXES)
    else:
        perturbation_type = "none"
        prefix = ""

    # 3. Optional prefix phrasing modifier
    modifier = random.choice(PHRASING_MODIFIERS) if random.random() < 0.4 else ""

    # 4. Optional suffix format modifier
    suffix = random.choice(SUFFIX_MODIFIERS) if random.random() < 0.3 else ""

    full_prompt = f"{prefix}{modifier}{prompt_core}{suffix}".strip()
    return full_prompt, perturbation_type, prefix.strip()


def estimate_tokens(text: str) -> int:
    """Heuristic token estimation for prompts and completions.
    Approximately:
      - 1 English word ~= 1.3 tokens
      - 1 Chinese character ~= 1.2 tokens
    """
    if not text:
        return 0
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    other_chars = len(text) - chinese_chars
    return max(1, int(chinese_chars * 1.3 + (other_chars / 4) * 1.2))
