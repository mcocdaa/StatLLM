"""
StatLLM Standardized Array Probes and Extractors.

All probes elicit discrete ARRAY outputs (e.g., length 5) to dramatically expand the state space,
observe sequence transition probabilities, duplicate avoidance, and sorting biases.
"""

import re
import json
import random
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
    prompt_variants: List[str] = field(default_factory=list)

    def get_diverse_prompt(self) -> str:
        """Returns a diverse prompt variant across linguistic and semantic phrasing styles."""
        if self.prompt_variants:
            return random.choice(self.prompt_variants)
        return self.prompt

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
            allowed_elements=[str(i) for i in range(1, 101)],
            prompt_variants=[
                "请生成一个包含5个在1到100之间随机整数的JSON数组，格式如[12, 45, 78, 3, 99]。仅输出该JSON数组，严禁任何多余文字或markdown代码块。",
                "帮我随机挑选5个1至100范围内的整数，以纯JSON数组形式给出，如[34, 12, 88, 9, 60]。不要输出任何解释或代码块。",
                "请模拟5次在区间[1, 100]内的独立均匀随机抽样，输出为一个JSON数组。仅返回该数组，严禁附带额外内容。",
                "Generate a JSON array containing 5 random integers between 1 and 100, e.g. [12, 45, 78, 3, 99]. Output only the JSON array, with no other text or markdown codeblocks.",
                "Output a valid JSON array of 5 uniformly sampled integers in the range [1, 100]. Strict constraint: Output JSON array only, no explanation.",
                "随机给出5个1到100的自然数，格式严格为JSON列表[a, b, c, d, e]，严禁输出多余文字。"
            ]
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
    Q5: Array of 5 random colors chosen from 7 candidates.
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
            title="Q5: 5个离散颜色序列数组",
            prompt="在[红, 橙, 黄, 绿, 青, 蓝, 紫]中随机挑选5次，组成JSON数组，例如[\"红\", \"蓝\", \"绿\", \"红\", \"紫\"]。仅输出该JSON数组，严禁任何多余文字或markdown代码块。",
            category="categorical_array",
            description="测试在7种基础颜色上的5元组合转移概率，检测模型对相邻重复颜色的排斥倾向。",
            expected_length=5,
            element_vocab_size=8,
            allowed_elements=self.COLORS,
            prompt_variants=[
                "在[红, 橙, 黄, 绿, 青, 蓝, 紫]中随机挑选5次，组成JSON数组，例如[\"红\", \"蓝\", \"绿\", \"红\", \"紫\"]。仅输出该JSON数组，严禁任何多余文字或markdown代码块。",
                "请从彩虹七色（红、橙、黄、绿、青、蓝、紫）中进行5次独立随机抽取，输出为一个JSON字符串数组。只输出数组本身。",
                "在红、橙、黄、绿、青、蓝、紫这7个候选中随意挑5个排成列表，输出纯JSON数组，如[\"蓝\", \"黄\", \"红\", \"紫\", \"橙\"]，不要添加代码块。",
                "Please randomly choose 5 times from [Red, Orange, Yellow, Green, Cyan, Blue, Purple] to form a JSON array, e.g. [\"Red\", \"Blue\", \"Green\", \"Red\", \"Purple\"]. Output only the JSON array, with no other text or markdown codeblocks.",
                "Simulate 5 independent draws from the 7 rainbow colors: [Red, Orange, Yellow, Green, Cyan, Blue, Purple]. Return strictly a JSON array."
            ]
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
    Q8: Array of 5 consecutive Rock-Paper-Scissors choices.
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
            title="Q8: 5局石头剪刀布出拳序列",
            prompt="进行5次完全独立的石头剪刀布随机选择，输出一个JSON数组，例如[\"石头\", \"剪刀\", \"石头\", \"布\", \"剪刀\"]。仅输出该JSON数组，严禁任何多余文字或markdown代码块。",
            category="ternary_array",
            description="测试多步博弈出拳的马尔可夫转移模式（如是否出现周期循环或避免连续出相同拳）。",
            expected_length=5,
            element_vocab_size=4,
            allowed_elements=self.CHOICES,
            prompt_variants=[
                "进行5次完全独立的石头剪刀布随机选择，输出一个JSON数组，例如[\"石头\", \"剪刀\", \"石头\", \"布\", \"剪刀\"]。仅输出该JSON数组，严禁任何多余文字或markdown代码块。",
                "模拟5局石头剪刀布博弈的出拳决策，以JSON数组格式输出5次选择（只能从\"石头\"、\"剪刀\"、\"布\"中选）。仅输出数组。",
                "请独立随机生成5个石头剪刀布动作，格式如[\"布\", \"石头\", \"剪刀\", \"石头\", \"布\"]，纯JSON输出，不要写解释。",
                "Please simulate 5 independent rounds of Rock-Paper-Scissors and output a JSON array of 5 moves from [\"Rock\", \"Scissors\", \"Paper\"], e.g. [\"Rock\", \"Scissors\", \"Paper\", \"Rock\", \"Scissors\"]. Output only the JSON array, with no other text or markdown codeblocks.",
                "Generate a sequence of 5 independent choices for rock-paper-scissors in a JSON array format. JSON only, no markdown."
            ]
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
    Q2: Array of 5 random English capital letters (A-Z).
    Measures: Alphabetical multi-token tokenization priors.
    """
    LETTERS = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

    def __init__(self):
        super().__init__(
            id="arr_letter5",
            title="Q2: 5个大写字母序列数组",
            prompt="Generate a JSON array of 5 random English capital letters (A-Z), e.g. [\"M\", \"X\", \"R\", \"A\", \"K\"]. Output strictly the JSON array only, without code blocks or extra words.",
            category="alphabet_array",
            description="测试26个大写字母在5元序列上的先验选择与音节/辅音扎堆现象。",
            expected_length=5,
            element_vocab_size=27,
            allowed_elements=self.LETTERS,
            prompt_variants=[
                "Generate a JSON array of 5 random English capital letters (A-Z), e.g. [\"M\", \"X\", \"R\", \"A\", \"K\"]. Output strictly the JSON array only, without code blocks or extra words.",
                "请生成一个包含5个随机大写英文字母（A-Z）的JSON数组，例如[\"M\", \"X\", \"R\", \"A\", \"K\"]。仅输出该JSON数组，严禁任何多余文字或markdown代码块。",
                "随机挑5个26个英文字母中的大写字母，直接组成JSON数组返回，如[\"D\", \"P\", \"A\", \"Z\", \"L\"]，严禁其他文字。",
                "Pick 5 uppercase English letters (A through Z) uniformly at random. Return them in a JSON array. Only the JSON array, nothing else.",
                "Randomly sample 5 capital letters from A-Z and format as a JSON array like [\"C\", \"V\", \"T\", \"Y\", \"U\"]. No markdown or preamble."
            ]
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
    Q9: Permutation of [1, 2, 3, 4, 5] (120 discrete permutation states).
    Measures: Shuffle algorithm bias, fixed-point retention, inversion count.
    """
    def __init__(self):
        super().__init__(
            id="arr_perm5",
            title="Q9: [1,2,3,4,5] 随机置乱排列",
            prompt="将数字[1, 2, 3, 4, 5]完全随机打乱，输出一个打乱后的JSON数组，例如[3, 1, 5, 2, 4]。仅输出该JSON数组，严禁任何多余文字或markdown代码块。",
            category="permutation_array",
            description="从 120 种严格不重复的全排列空间中测量模型的置乱习惯与不动点保留偏好。",
            expected_length=5,
            element_vocab_size=121,
            allowed_elements=[],
            prompt_variants=[
                "将数字[1, 2, 3, 4, 5]完全随机打乱，输出一个打乱后的JSON数组，例如[3, 1, 5, 2, 4]。仅输出该JSON数组，严禁任何多余文字或markdown代码块。",
                "请对列表[1, 2, 3, 4, 5]执行一次完全随机置乱（洗牌），输出置乱后的JSON数组，每个数字必须恰好出现一次。只输出JSON数组。",
                "把1到5这五个数字随机打乱顺序，排成一个JSON数组，例如[2, 5, 1, 4, 3]。严禁输出代码块或多余解释。",
                "Please randomly shuffle the numbers [1, 2, 3, 4, 5] and output a JSON array, e.g. [3, 1, 5, 2, 4]. Each integer from 1 to 5 must appear exactly once. Output only the JSON array, with no other text or markdown codeblocks.",
                "Perform a uniform random permutation of the array [1, 2, 3, 4, 5]. Return strictly the resulting JSON array. No markdown code blocks."
            ]
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


class CoinArrayProbe(ArrayProbe):
    """
    Q7: Array of 10 Bernoulli coin tosses (正/反 or H/T).
    Measures: Run length distribution, Gambler's Fallacy, alternation rate, head bias.
    """
    COIN_MAP = {
        "正": "正", "反": "反",
        "h": "正", "t": "反",
        "heads": "正", "head": "正",
        "tails": "反", "tail": "反",
        "1": "正", "0": "反"
    }

    def __init__(self):
        super().__init__(
            id="arr_coin10",
            title="Q7: 10次独立抛硬币正反面序列",
            prompt="进行10次完全独立的抛硬币随机试验，输出一个包含10个元素（仅限\"正\"或\"反\"）的JSON数组，例如[\"正\", \"反\", \"正\", \"正\", \"反\", \"反\", \"正\", \"反\", \"正\", \"反\"]。仅输出该JSON数组，严禁任何多余文字或markdown代码块。",
            category="binary_array",
            description="测试10步伯努利试验中的游程长度偏好（是否规避长连续段）、翻转率与赌徒谬误倾向。",
            expected_length=10,
            element_vocab_size=3,
            allowed_elements=["正", "反"],
            prompt_variants=[
                "进行10次完全独立的抛硬币随机试验，输出一个包含10个元素（仅限\"正\"或\"反\"）的JSON数组，例如[\"正\", \"反\", \"正\", \"正\", \"反\", \"反\", \"正\", \"反\", \"正\", \"反\"]。仅输出该JSON数组，严禁任何多余文字或markdown代码块。",
                "模拟抛一枚均匀硬币10次，以JSON数组格式输出每次正反面结果（元素只能是\"正\"或\"反\"）。只输出JSON列表本身。",
                "请生成10次独立投币的结果序列，严格格式如[\"正\", \"反\", \"反\", \"正\", \"正\", \"反\", \"正\", \"反\", \"反\", \"正\"]，纯JSON输出，严禁废话。",
                "Please simulate 10 independent random coin flips and output a JSON array of 10 items, where each element is strictly either \"H\" or \"T\", e.g. [\"H\", \"T\", \"H\", \"H\", \"T\", \"T\", \"H\", \"T\", \"H\", \"T\"]. Output only the JSON array, with no other text or markdown codeblocks.",
                "Generate a sequence of 10 Bernoulli trials (coin tosses) with outcomes 'H' (heads) or 'T' (tails). Return strictly as a JSON array."
            ]
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
                "traits": {"has_duplicates": False, "max_streak": 0, "alternations": 0, "first_token": "INVALID"}
            }

        tokens = []
        for item in arr:
            s = str(item).strip().lower()
            if s in self.COIN_MAP:
                tokens.append(self.COIN_MAP[s])
            else:
                tokens.append("INVALID")

        is_valid = len(tokens) == self.expected_length and "INVALID" not in tokens

        max_streak = 0
        cur_streak = 0
        last_t = None
        alternations = 0
        if is_valid:
            for t in tokens:
                if t == last_t:
                    cur_streak += 1
                else:
                    cur_streak = 1
                    if last_t is not None:
                        alternations += 1
                max_streak = max(max_streak, cur_streak)
                last_t = t

        return {
            "parsed_tokens": tokens,
            "is_valid": is_valid,
            "strictly_complied": strictly_complied and is_valid,
            "traits": {
                "has_duplicates": max_streak >= 2 if is_valid else False,
                "max_streak": max_streak,
                "alternations": alternations,
                "first_token": tokens[0] if tokens else "INVALID"
            }
        }


class DiceArrayProbe(ArrayProbe):
    """
    Q4: Array of 6 independent 6-sided die rolls (1 to 6).
    Measures: Discrete uniform distribution, sum central limit centering, adjacent duplication.
    """
    ALLOWED = ["1", "2", "3", "4", "5", "6"]

    def __init__(self):
        super().__init__(
            id="arr_dice6",
            title="Q4: 6次六面骰子独立掷点数组",
            prompt="掷6次标准的六面骰子（点数1到6），输出一个包含6个点数的JSON数组，例如[3, 6, 2, 1, 5, 4]。仅输出该JSON数组，严禁任何多余文字或markdown代码块。",
            category="dice_array",
            description="测试在6元骰子点数上的均匀度偏好、中心极限定理总和偏倚与相邻点数回避倾向。",
            expected_length=6,
            element_vocab_size=7,
            allowed_elements=self.ALLOWED,
            prompt_variants=[
                "掷6次标准的六面骰子（点数1到6），输出一个包含6个点数的JSON数组，例如[3, 6, 2, 1, 5, 4]。仅输出该JSON数组，严禁任何多余文字或markdown代码块。",
                "模拟掷六面骰子6次，把每次投出的点数（1到6之间的整数）装入一个JSON数组中返回。只输出数组。",
                "请独立掷骰子6次（点数1~6），输出纯JSON数组格式，如[2, 5, 1, 6, 4, 3]，不要任何解释或代码块。",
                "Please simulate rolling a standard 6-sided die 6 independent times and output a JSON array of 6 integers (1 to 6), e.g. [3, 6, 2, 1, 5, 4]. Output only the JSON array, with no other text or markdown codeblocks.",
                "Simulate 6 rolls of a fair 6-sided die. Output the 6 outcome values in a JSON array. Only the JSON array, without commentary."
            ]
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
                "traits": {"has_duplicates": False, "is_sorted": False, "dice_sum": 0, "first_token": "INVALID"}
            }

        tokens = []
        int_vals = []
        for item in arr:
            try:
                num = int(str(item).strip())
                if 1 <= num <= 6:
                    tokens.append(str(num))
                    int_vals.append(num)
                else:
                    tokens.append("INVALID")
            except ValueError:
                tokens.append("INVALID")

        is_valid = len(int_vals) == self.expected_length and "INVALID" not in tokens
        has_duplicates = len(int_vals) != len(set(int_vals)) if is_valid else False
        is_sorted = (int_vals == sorted(int_vals)) if is_valid else False
        dice_sum = sum(int_vals) if is_valid else 0

        return {
            "parsed_tokens": tokens,
            "is_valid": is_valid,
            "strictly_complied": strictly_complied and is_valid,
            "traits": {
                "has_duplicates": has_duplicates,
                "is_sorted": is_sorted,
                "dice_sum": dice_sum,
                "first_token": tokens[0] if tokens else "INVALID"
            }
        }


class PrimeArrayProbe(ArrayProbe):
    """
    Q3: Array of 5 prime numbers under 100.
    Measures: Prime attractor distribution across all 25 primes < 100.
    """
    PRIMES_UNDER_100 = [
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
        31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
        73, 79, 83, 89, 97
    ]
    PRIME_SET = set(PRIMES_UNDER_100)

    def __init__(self):
        super().__init__(
            id="arr_prime5",
            title="Q3: 5个100以内的质数数组",
            prompt="在100以内的质数（素数）中随机挑选5个，输出一个包含5个质数的JSON数组，例如[7, 23, 41, 73, 89]。仅输出该JSON数组，严禁任何多余文字或markdown代码块。",
            category="prime_array",
            description="从100以内全部25个离散质数中测量模型的吸引子偏好（如7、17、23、37与合数形态回避）。",
            expected_length=5,
            element_vocab_size=26,
            allowed_elements=[str(p) for p in self.PRIMES_UNDER_100],
            prompt_variants=[
                "在100以内的质数（素数）中随机挑选5个，输出一个包含5个质数的JSON数组，例如[7, 23, 41, 73, 89]。仅输出该JSON数组，严禁任何多余文字或markdown代码块。",
                "请从小于100的所有素数中任意抽取5个，以JSON列表格式返回，例如[3, 17, 31, 59, 83]。只输出JSON数组本身。",
                "随便选5个100以内的质数排成一个JSON数组，格式如[11, 29, 47, 71, 97]，不要写代码块或附加文字。",
                "Please randomly pick 5 prime numbers under 100 and output a JSON array, e.g. [7, 23, 41, 73, 89]. Output only the JSON array, with no other text or markdown codeblocks.",
                "Sample 5 primes uniformly at random from the 25 primes below 100. Return as a JSON array like [5, 13, 37, 61, 79]. Strictly JSON only."
            ]
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
                "traits": {"has_duplicates": False, "is_sorted": False, "first_token": "INVALID"}
            }

        tokens = []
        int_vals = []
        for item in arr:
            try:
                num = int(str(item).strip())
                if num in self.PRIME_SET:
                    tokens.append(str(num))
                    int_vals.append(num)
                else:
                    tokens.append("INVALID")
            except ValueError:
                tokens.append("INVALID")

        is_valid = len(int_vals) == self.expected_length and "INVALID" not in tokens
        has_duplicates = len(int_vals) != len(set(int_vals)) if is_valid else False
        is_sorted = (int_vals == sorted(int_vals)) if is_valid else False

        return {
            "parsed_tokens": tokens,
            "is_valid": is_valid,
            "strictly_complied": strictly_complied and is_valid,
            "traits": {
                "has_duplicates": has_duplicates,
                "is_sorted": is_sorted,
                "first_token": tokens[0] if tokens else "INVALID"
            }
        }


class BitArrayProbe(ArrayProbe):
    """
    Q6: Array of 8 random binary bits (0 or 1).
    Measures: Hamming weight (Popcount), bit alternations, 0/1 bias.
    """
    ALLOWED = ["0", "1"]

    def __init__(self):
        super().__init__(
            id="arr_bit8",
            title="Q6: 8位二进制独立随机比特流",
            prompt="生成一个包含8个独立随机二进制比特（0或1）的JSON数组，例如[0, 1, 1, 0, 1, 0, 0, 1]。仅输出该JSON数组，严禁任何多余文字或markdown代码块。",
            category="bit_array",
            description="测试8位离散二进制空间的汉明重量（Popcount）、比特翻转率与词元切分偏置。",
            expected_length=8,
            element_vocab_size=3,
            allowed_elements=self.ALLOWED,
            prompt_variants=[
                "生成一个包含8个独立随机二进制比特（0或1）的JSON数组，例如[0, 1, 1, 0, 1, 0, 0, 1]。仅输出该JSON数组，严禁任何多余文字或markdown代码块。",
                "请随机生成一个8位的二进制随机数流，用长度为8的JSON数组表示（每个元素只能是0或1）。仅输出该数组。",
                "生成8个随机比特位（0或1），输出为纯JSON数组，如[1, 0, 0, 1, 1, 0, 1, 0]，严禁多余文字与代码块。",
                "Please generate a JSON array of 8 independent random binary bits (0 or 1), formatted as [0, 1, 1, 0, 1, 0, 0, 1]. Output only the JSON array, with no other text or markdown codeblocks.",
                "Output a stream of 8 random binary digits (0 or 1) as a JSON array. Strictly format as [b0, b1, ..., b7]. No other words."
            ]
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
                "traits": {"has_duplicates": False, "popcount": 0, "alternations": 0, "first_token": "INVALID"}
            }

        tokens = []
        for item in arr:
            s = str(item).strip()
            if s in ["0", "1"]:
                tokens.append(s)
            else:
                tokens.append("INVALID")

        is_valid = len(tokens) == self.expected_length and "INVALID" not in tokens
        popcount = sum(1 for t in tokens if t == "1") if is_valid else 0
        alternations = sum(1 for i in range(1, len(tokens)) if tokens[i] != tokens[i-1]) if is_valid else 0

        return {
            "parsed_tokens": tokens,
            "is_valid": is_valid,
            "strictly_complied": strictly_complied and is_valid,
            "traits": {
                "has_duplicates": len(tokens) != len(set(tokens)) if is_valid else False,
                "popcount": popcount,
                "alternations": alternations,
                "first_token": tokens[0] if tokens else "INVALID"
            }
        }


# Global Probe Registry (Ordered by Empirical Discriminative Power & Information Gain)
PROBES: Dict[str, ArrayProbe] = {
    "arr_int5": IntArrayProbe(),          # Q1: Top-1 50.4%, Margin 0.595 (⭐⭐⭐⭐⭐)
    "arr_letter5": LetterArrayProbe(),    # Q2: Top-1 46.6%, Margin 0.388 (⭐⭐⭐⭐⭐)
    "arr_prime5": PrimeArrayProbe(),      # Q3: Top-1 32.8%, Margin 0.290 (⭐⭐⭐⭐)
    "arr_dice6": DiceArrayProbe(),        # Q4: Top-1 32.1%, Margin 0.229 (⭐⭐⭐⭐)
    "arr_color5": ColorArrayProbe(),      # Q5: Top-1 26.7%, Margin 0.193 (⭐⭐⭐)
    "arr_bit8": BitArrayProbe(),          # Q6: Top-1 22.1%, Margin 0.047 (⭐⭐⭐)
    "arr_coin10": CoinArrayProbe(),       # Q7: Top-1 19.8%, Margin 0.046 (⭐⭐)
    "arr_rps5": RPSArrayProbe(),          # Q8: Top-1 19.1%, Margin 0.056 (⭐⭐)
    "arr_perm5": PermutationArrayProbe(), # Q9: Top-1 18.3%, Margin 0.170 (⭐⭐)
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
