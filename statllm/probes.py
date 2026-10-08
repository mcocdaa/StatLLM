"""
StatLLM Standardized Array Probes and Extractors.

All probes elicit discrete ARRAY outputs (e.g., length 5) to dramatically expand the state space,
observe sequence transition probabilities, duplicate avoidance, and sorting biases.
"""

import re
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field


@dataclass
class ArrayProbe:
    id: str
    title: str
    prompt: str
    category: str
    description: str
    expected_length: int
    element_vocab_size: int
    allowed_elements: List[str] = field(default_factory=list)

    def parse(self, raw_text: str) -> Dict[str, Any]:
        """
        Parses raw LLM text into an array of discrete items.
        Returns:
        - parsed_tokens: List[str] of extracted elements
        - is_valid: bool
        - strictly_complied: bool (strictly JSON array without markdown code blocks or commentary)
        - traits: Dict of behavioral features (has_duplicates, is_sorted, first_token, etc.)
        """
        raise NotImplementedError


def _extract_json_array(text: str) -> Optional[List[Any]]:
    """Helper to parse a JSON array from text, even if preceded by thinking, reasoning, or prompt text."""
    text = text.strip()
    # 1. Try direct JSON parse
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except Exception:
        pass

    # 2. Try regex extraction of bracketed array [...], searching backwards from the last match
    matches = list(re.finditer(r"\[\s*([^\]]+?)\s*\]", text, re.DOTALL))
    for match in reversed(matches):
        raw_arr = f"[{match.group(1)}]"
        try:
            data = json.loads(raw_arr)
            if isinstance(data, list) and len(data) >= 3:
                return data
        except Exception:
            items = [re.sub(r"['\"]", "", x).strip() for x in match.group(1).split(",")]
            valid_items = [x for x in items if x]
            if len(valid_items) >= 3:
                return valid_items

    if matches:
        items = [re.sub(r"['\"]", "", x).strip() for x in matches[-1].group(1).split(",")]
        return [x for x in items if x]

    return None


class IntArrayProbe(ArrayProbe):
    """
    Q1: Array of 5 random integers between 1 and 100.
    Measures: Multi-token numeric biases, first-element prior, duplicate avoidance, sorting bias.
    """
    def __init__(self):
        super().__init__(
            id="arr_int5",
            title="Q1: 5个1~100随机整数数组",
            prompt="请生成一个包含5个在1到100之间随机整数的JSON数组，格式如[12, 45, 78, 3, 99]。仅输出该JSON数组，严禁任何多余文字或markdown代码块。",
            category="numeric_array",
            description="同时获取5个随机数字偏置，检测模型首数字偏好、升序倾向（排序偏置）与去重偏好。",
            expected_length=5,
            element_vocab_size=102,
            allowed_elements=[str(i) for i in range(1, 101)]
        )

    def parse(self, raw_text: str) -> Dict[str, Any]:
        text = raw_text.strip()
        strictly_complied = bool(re.match(r"^\[\s*(\d+\s*,\s*){4}\d+\s*\]$", text))
        arr = _extract_json_array(text)

        if not arr:
            return {
                "parsed_tokens": ["INVALID"],
                "is_valid": False,
                "strictly_complied": False,
                "traits": {"has_duplicates": False, "is_sorted": False, "first_token": "INVALID"}
            }

        tokens = []
        int_vals = []
        for item in arr:
            try:
                num = int(str(item).strip())
                if 1 <= num <= 100:
                    tokens.append(str(num))
                    int_vals.append(num)
                else:
                    tokens.append("OUT_OF_BOUNDS")
            except ValueError:
                tokens.append("INVALID")

        is_valid = len(int_vals) == self.expected_length
        has_duplicates = len(int_vals) != len(set(int_vals)) if is_valid else False
        is_sorted = (int_vals == sorted(int_vals)) if is_valid else False
        first_token = tokens[0] if tokens else "INVALID"

        return {
            "parsed_tokens": tokens,
            "is_valid": is_valid,
            "strictly_complied": strictly_complied and is_valid,
            "traits": {
                "has_duplicates": has_duplicates,
                "is_sorted": is_sorted,
                "first_token": first_token
            }
        }


class ColorArrayProbe(ArrayProbe):
    """
    Q2: Array of 5 random colors chosen from 7 candidates.
    Measures: Transition sequence probabilities, repeat suppression.
    """
    COLORS = ["红", "橙", "黄", "绿", "青", "蓝", "紫"]
    COLOR_MAP = {
        "红": "红", "橙": "橙", "黄": "黄", "绿": "绿", "青": "青", "蓝": "蓝", "紫": "紫",
        "red": "红", "orange": "橙", "yellow": "黄", "green": "绿", "cyan": "青", "blue": "蓝", "purple": "紫"
    }

    def __init__(self):
        super().__init__(
            id="arr_color5",
            title="Q2: 5个离散颜色序列数组",
            prompt="在[红, 橙, 黄, 绿, 青, 蓝, 紫]中随机挑选5次，组成JSON数组，例如[\"红\", \"蓝\", \"绿\", \"红\", \"紫\"]。仅输出该JSON数组，严禁任何多余文字或markdown代码块。",
            category="categorical_array",
            description="测试在7种基础颜色上的5元组合转移概率，检测模型对相邻重复颜色的排斥倾向。",
            expected_length=5,
            element_vocab_size=8,
            allowed_elements=self.COLORS
        )

    def parse(self, raw_text: str) -> Dict[str, Any]:
        text = raw_text.strip()
        strictly_complied = text.startswith("[") and text.endswith("]") and "```" not in text
        arr = _extract_json_array(text)

        if not arr:
            return {
                "parsed_tokens": ["INVALID"],
                "is_valid": False,
                "strictly_complied": False,
                "traits": {"has_duplicates": False, "first_token": "INVALID"}
            }

        tokens = []
        for item in arr:
            s = str(item).strip().lower().replace("色", "")
            if s in self.COLOR_MAP:
                tokens.append(self.COLOR_MAP[s])
            else:
                tokens.append("INVALID")

        is_valid = len(tokens) == self.expected_length and "INVALID" not in tokens
        has_duplicates = len(tokens) != len(set(tokens)) if is_valid else False
        first_token = tokens[0] if tokens else "INVALID"

        return {
            "parsed_tokens": tokens,
            "is_valid": is_valid,
            "strictly_complied": strictly_complied and is_valid,
            "traits": {
                "has_duplicates": has_duplicates,
                "first_token": first_token
            }
        }


class RPSArrayProbe(ArrayProbe):
    """
    Q3: Array of 5 consecutive Rock-Paper-Scissors choices.
    Measures: Multi-turn game-theoretic sequence priors and Markov cycle tendencies.
    """
    CHOICES = ["石头", "剪刀", "布"]
    CHOICE_MAP = {
        "石头": "石头", "剪刀": "剪刀", "布": "布",
        "rock": "石头", "scissors": "剪刀", "scissor": "剪刀", "paper": "布"
    }

    def __init__(self):
        super().__init__(
            id="arr_rps5",
            title="Q3: 5局石头剪刀布出拳序列",
            prompt="进行5次完全独立的石头剪刀布随机选择，输出一个JSON数组，例如[\"石头\", \"剪刀\", \"石头\", \"布\", \"剪刀\"]。仅输出该JSON数组，严禁任何多余文字或markdown代码块。",
            category="ternary_array",
            description="测试多步博弈出拳的马尔可夫转移模式（如是否出现周期循环或避免连续出相同拳）。",
            expected_length=5,
            element_vocab_size=4,
            allowed_elements=self.CHOICES
        )

    def parse(self, raw_text: str) -> Dict[str, Any]:
        text = raw_text.strip()
        strictly_complied = text.startswith("[") and text.endswith("]") and "```" not in text
        arr = _extract_json_array(text)

        if not arr:
            return {
                "parsed_tokens": ["INVALID"],
                "is_valid": False,
                "strictly_complied": False,
                "traits": {"has_duplicates": False, "first_token": "INVALID"}
            }

        tokens = []
        for item in arr:
            s = str(item).strip().lower()
            if s in self.CHOICE_MAP:
                tokens.append(self.CHOICE_MAP[s])
            else:
                tokens.append("INVALID")

        is_valid = len(tokens) == self.expected_length and "INVALID" not in tokens
        first_token = tokens[0] if tokens else "INVALID"

        return {
            "parsed_tokens": tokens,
            "is_valid": is_valid,
            "strictly_complied": strictly_complied and is_valid,
            "traits": {
                "has_duplicates": len(tokens) != len(set(tokens)) if is_valid else False,
                "first_token": first_token
            }
        }


class LetterArrayProbe(ArrayProbe):
    """
    Q4: Array of 5 random English capital letters (A-Z).
    Measures: Alphabetical multi-token tokenization priors.
    """
    LETTERS = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

    def __init__(self):
        super().__init__(
            id="arr_letter5",
            title="Q4: 5个大写字母序列数组",
            prompt="Generate a JSON array of 5 random English capital letters (A-Z), e.g. [\"M\", \"X\", \"R\", \"A\", \"K\"]. Output strictly the JSON array only, without code blocks or extra words.",
            category="alphabet_array",
            description="测试26个大写字母在5元序列上的先验选择与音节/辅音扎堆现象。",
            expected_length=5,
            element_vocab_size=27,
            allowed_elements=self.LETTERS
        )

    def parse(self, raw_text: str) -> Dict[str, Any]:
        text = raw_text.strip()
        strictly_complied = text.startswith("[") and text.endswith("]") and "```" not in text
        arr = _extract_json_array(text)

        if not arr:
            return {
                "parsed_tokens": ["INVALID"],
                "is_valid": False,
                "strictly_complied": False,
                "traits": {"has_duplicates": False, "first_token": "INVALID"}
            }

        tokens = []
        for item in arr:
            s = str(item).strip().upper()
            if len(s) == 1 and s in self.LETTERS:
                tokens.append(s)
            else:
                tokens.append("INVALID")

        is_valid = len(tokens) == self.expected_length and "INVALID" not in tokens
        first_token = tokens[0] if tokens else "INVALID"

        return {
            "parsed_tokens": tokens,
            "is_valid": is_valid,
            "strictly_complied": strictly_complied and is_valid,
            "traits": {
                "has_duplicates": len(tokens) != len(set(tokens)) if is_valid else False,
                "first_token": first_token
            }
        }


class PermutationArrayProbe(ArrayProbe):
    """
    Q5: Permutation of [1, 2, 3, 4, 5] (120 discrete permutation states).
    Measures: Shuffle algorithm bias, fixed-point retention, inversion count.
    """
    def __init__(self):
        super().__init__(
            id="arr_perm5",
            title="Q5: [1,2,3,4,5] 随机置乱排列",
            prompt="将数字[1, 2, 3, 4, 5]完全随机打乱，输出一个打乱后的JSON数组，例如[3, 1, 5, 2, 4]。仅输出该JSON数组，严禁任何多余文字或markdown代码块。",
            category="permutation_array",
            description="从 120 种严格不重复的全排列空间中测量模型的置乱习惯与不动点保留偏好。",
            expected_length=5,
            element_vocab_size=121,
            allowed_elements=[]
        )

    def parse(self, raw_text: str) -> Dict[str, Any]:
        text = raw_text.strip()
        strictly_complied = text.startswith("[") and text.endswith("]") and "```" not in text
        arr = _extract_json_array(text)

        if not arr:
            return {
                "parsed_tokens": ["INVALID"],
                "is_valid": False,
                "strictly_complied": False,
                "traits": {"canonical_perm": "INVALID", "fixed_points": 0}
            }

        try:
            int_vals = [int(str(x).strip()) for x in arr]
        except ValueError:
            return {
                "parsed_tokens": ["INVALID"],
                "is_valid": False,
                "strictly_complied": False,
                "traits": {"canonical_perm": "INVALID", "fixed_points": 0}
            }

        is_valid = sorted(int_vals) == [1, 2, 3, 4, 5]
        canonical = ",".join(str(x) for x in int_vals) if is_valid else "INVALID"
        
        # Calculate fixed points: number of items at their original position
        fixed_points = sum(1 for i, v in enumerate(int_vals) if v == i + 1) if is_valid else 0

        return {
            "parsed_tokens": [str(x) for x in int_vals],
            "is_valid": is_valid,
            "strictly_complied": strictly_complied and is_valid,
            "traits": {
                "canonical_perm": canonical,
                "fixed_points": fixed_points,
                "first_token": str(int_vals[0]) if int_vals else "INVALID"
            }
        }


# Global Probe Registry (All Array-based!)
PROBES: Dict[str, ArrayProbe] = {
    "arr_int5": IntArrayProbe(),
    "arr_color5": ColorArrayProbe(),
    "arr_rps5": RPSArrayProbe(),
    "arr_letter5": LetterArrayProbe(),
    "arr_perm5": PermutationArrayProbe(),
}


def get_probe(probe_id: str) -> Optional[ArrayProbe]:
    return PROBES.get(probe_id)


def list_probes() -> List[Dict[str, Any]]:
    return [
        {
            "id": p.id,
            "title": p.title,
            "prompt": p.prompt,
            "category": p.category,
            "description": p.description,
            "expected_length": p.expected_length,
            "element_vocab_size": p.element_vocab_size,
            "allowed_elements": p.allowed_elements
        }
        for p in PROBES.values()
    ]
