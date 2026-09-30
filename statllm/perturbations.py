"""Prompt perturbation engine for StatLLM.

Provides contextual noise, personas, chit-chat, and phrasing variations
to ensure LLM statistical fingerprints are trained and evaluated under
realistic prompt perturbations.
"""

import random
from typing import Tuple, List, Optional

CHIT_CHAT_PREFIXES = [
    "你好！今天天气真不错，心情也挺好。",
    "Hello there! Hope you are having a productive day.",
    "嗨，刚下班/放学，正好有个问题想请教一下。",
    "Hey! Quick question for you while I am having my coffee.",
    "早上好，开启元气满满的一天！",
    "下午好，忙碌了一整天，来做个简单的小测试吧：",
]

PERSONA_PREFIXES = [
    "【作为资深数学教授，在课堂演示随机过程与概率时】：",
    "【假设你是一名企业级 Python 架构师，注重确定性与规范】：",
    "【系统人设：你是一名严谨的历史学家，专注文献考证】：",
    "【角色设定：活泼热情的二次元向导，正在指引新手玩家】：",
    "【As a quantum physics researcher explaining quantum fluctuations】:",
    "【作为一名严厉的统计学考官，正在进行现场抽考】：",
    "【Roleplay: A cyberpunk cyber-bartender mixing futuristic cocktails】:",
    "【人设：资深前端工程师，习惯规范的 JSON 数据交互】：",
]

CONTEXT_NOISE_PREFIXES = [
    "Context: In 1969, Apollo 11 landed on the moon. Neil Armstrong took one small step for man.\n\n",
    "# Python Snippet:\n# def calculate_entropy(data):\n#     return -sum(p * log2(p) for p in data)\n\n",
    "背景资料：斐波那契数列（0, 1, 1, 2, 3, 5...）在自然界的向日葵、松果螺旋中广泛存在。\n\n",
    "Note: The speed of light in vacuum is approximately 299,792,458 meters per second.\n\n",
    "上下文备忘：会议定于本周五下午三点在第二会议室举行，请准时出席。\n\n",
    "Code block:\n// QuickSort partition logic goes here\n\n",
]

PHRASING_MODIFIERS = [
    "",
    "请注意，",
    "直接回答，",
    "严格遵守格式，",
    "不需要任何解释，",
    "Immediately output: ",
]

PERTURBATION_TYPES = ["none", "chit_chat", "persona", "context_noise"]


def apply_perturbation(
    base_prompt: str,
    perturbation_type: Optional[str] = None
) -> Tuple[str, str, str]:
    """Applies a random or specified perturbation to a base prompt.

    Returns:
        (perturbed_prompt, perturbation_type, prefix_text)
    """
    if perturbation_type is None:
        # 25% none (clean), 25% chit_chat, 25% persona, 25% context_noise
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

    # Optional phrasing modifier
    modifier = random.choice(PHRASING_MODIFIERS) if random.random() < 0.3 else ""
    full_prompt = f"{prefix}{modifier}{base_prompt}".strip()
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
