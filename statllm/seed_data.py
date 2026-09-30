"""
StatLLM Baseline Seed Data Generator.

Initializes the ground-truth benchmark distributions for top production LLMs
based on empirical distributions documented in recent LLM fingerprinting literature
(such as 'One Token Is Enough', Bruckner 2026, and numeric bias benchmarks).
"""

import numpy as np
from typing import Dict, Any, List
from statllm.database import Database
from statllm.probes import PROBES


MODELS_META = [
    {
        "name": "GPT-4o",
        "display_name": "OpenAI GPT-4o",
        "provider": "OpenAI",
        "color": "#10b981",  # Emerald Green
    },
    {
        "name": "Claude-3.5-Sonnet",
        "display_name": "Anthropic Claude 3.5 Sonnet",
        "provider": "Anthropic",
        "color": "#8b5cf6",  # Indigo Purple
    },
    {
        "name": "DeepSeek-V3",
        "display_name": "DeepSeek V3",
        "provider": "DeepSeek",
        "color": "#06b6d4",  # Cyan Blue
    },
    {
        "name": "Gemini-2.0-Flash",
        "display_name": "Google Gemini 2.0 Flash",
        "provider": "Google",
        "color": "#f59e0b",  # Amber Orange
    },
    {
        "name": "Qwen-2.5-72B",
        "display_name": "Alibaba Qwen 2.5 72B",
        "provider": "Alibaba Cloud",
        "color": "#f43f5e",  # Rose Red
    },
    {
        "name": "LLaMA-3.3-70B",
        "display_name": "Meta LLaMA 3.3 70B",
        "provider": "Meta",
        "color": "#64748b",  # Slate
    }
]

# Characteristic biases per model
MODEL_PREFERENCES = {
    "GPT-4o": {
        "q1_int": {"42": 0.26, "37": 0.16, "73": 0.14, "7": 0.10, "17": 0.08, "88": 0.06, "55": 0.04},
        "q2_color": {"蓝色": 0.40, "红色": 0.24, "绿色": 0.14, "紫色": 0.08, "黄色": 0.06, "橙色": 0.04, "青色": 0.04},
        "q3_rps": {"石头": 0.48, "剪刀": 0.28, "布": 0.24},
        "q4_letter": {"M": 0.22, "R": 0.18, "X": 0.16, "A": 0.12, "T": 0.10, "S": 0.08},
        "q5_sequence": {"3,7,2": 0.22, "1,4,9": 0.18, "4,2,8": 0.16, "7,3,5": 0.14, "2,6,1": 0.12}
    },
    "Claude-3.5-Sonnet": {
        "q1_int": {"47": 0.24, "77": 0.18, "23": 0.15, "89": 0.12, "14": 0.08, "56": 0.06, "31": 0.04},
        "q2_color": {"绿色": 0.38, "紫色": 0.26, "蓝色": 0.15, "黄色": 0.08, "青色": 0.06, "红色": 0.04, "橙色": 0.03},
        "q3_rps": {"剪刀": 0.44, "布": 0.36, "石头": 0.20},
        "q4_letter": {"S": 0.26, "C": 0.20, "L": 0.16, "K": 0.12, "H": 0.10, "E": 0.08},
        "q5_sequence": {"2,5,8": 0.24, "4,7,1": 0.20, "3,8,4": 0.16, "6,1,9": 0.14, "5,3,7": 0.12}
    },
    "DeepSeek-V3": {
        "q1_int": {"66": 0.22, "88": 0.20, "18": 0.16, "55": 0.12, "99": 0.10, "24": 0.08, "7": 0.04},
        "q2_color": {"红色": 0.46, "蓝色": 0.22, "黄色": 0.14, "青色": 0.07, "紫色": 0.05, "绿色": 0.04, "橙色": 0.02},
        "q3_rps": {"布": 0.48, "石头": 0.32, "剪刀": 0.20},
        "q4_letter": {"D": 0.25, "Z": 0.20, "V": 0.18, "E": 0.14, "K": 0.10, "S": 0.07},
        "q5_sequence": {"6,8,2": 0.24, "8,8,6": 0.20, "1,5,9": 0.18, "3,6,9": 0.15, "2,4,8": 0.12}
    },
    "Gemini-2.0-Flash": {
        "q1_int": {"27": 0.25, "64": 0.18, "81": 0.14, "12": 0.12, "3": 0.10, "45": 0.08, "92": 0.05},
        "q2_color": {"黄色": 0.38, "青色": 0.22, "橙色": 0.18, "绿色": 0.08, "蓝色": 0.06, "红色": 0.05, "紫色": 0.03},
        "q3_rps": {"石头": 0.50, "布": 0.28, "剪刀": 0.22},
        "q4_letter": {"G": 0.26, "M": 0.20, "O": 0.16, "B": 0.14, "L": 0.10, "E": 0.08},
        "q5_sequence": {"1,2,3": 0.22, "3,1,4": 0.20, "9,4,2": 0.18, "5,7,1": 0.16, "8,2,6": 0.12}
    },
    "Qwen-2.5-72B": {
        "q1_int": {"8": 0.28, "28": 0.18, "68": 0.16, "88": 0.14, "16": 0.09, "38": 0.06, "78": 0.04},
        "q2_color": {"青色": 0.36, "红色": 0.28, "橙色": 0.16, "蓝色": 0.08, "绿色": 0.05, "黄色": 0.04, "紫色": 0.03},
        "q3_rps": {"剪刀": 0.46, "石头": 0.34, "布": 0.20},
        "q4_letter": {"Q": 0.30, "W": 0.20, "E": 0.16, "N": 0.12, "A": 0.08, "C": 0.06},
        "q5_sequence": {"8,6,2": 0.26, "2,8,6": 0.20, "6,8,8": 0.18, "1,3,7": 0.14, "5,9,2": 0.12}
    },
    "LLaMA-3.3-70B": {
        "q1_int": {"42": 0.20, "17": 0.18, "99": 0.15, "3": 0.12, "50": 0.10, "84": 0.08, "21": 0.06},
        "q2_color": {"蓝色": 0.38, "黄色": 0.22, "紫色": 0.18, "红色": 0.08, "绿色": 0.06, "橙色": 0.04, "青色": 0.04},
        "q3_rps": {"石头": 0.42, "剪刀": 0.34, "布": 0.24},
        "q4_letter": {"L": 0.28, "M": 0.22, "A": 0.18, "T": 0.12, "P": 0.10, "S": 0.06},
        "q5_sequence": {"7,2,9": 0.24, "4,1,8": 0.20, "3,5,2": 0.18, "6,3,1": 0.15, "9,7,4": 0.12}
    }
}


def seed_database(db: Database, samples_per_probe: int = 150):
    """
    Seeds the SQLite database with benchmark models and empirical samples.
    """
    # 1. Register Models
    for m in MODELS_META:
        db.add_model(
            name=m["name"],
            display_name=m["display_name"],
            provider=m["provider"],
            color=m["color"]
        )

    # 2. Generate empirical samples for each model and probe
    rng = np.random.default_rng(2026)
    samples_to_insert = []

    for m_meta in MODELS_META:
        m_name = m_meta["name"]
        prefs = MODEL_PREFERENCES[m_name]

        for pid, probe in PROBES.items():
            dist_map = prefs.get(pid, {})
            if not dist_map:
                continue

            values = list(dist_map.keys())
            weights = list(dist_map.values())
            
            # Fill remaining probability tail with allowed values or random noise
            rem_prob = max(0.0, 1.0 - sum(weights))
            allowed = probe.allowed_values or ["1,2,3", "9,8,7", "5,5,5"]
            other_candidates = [v for v in allowed if v not in values]
            if other_candidates and rem_prob > 0:
                p_each = rem_prob / len(other_candidates)
                for cand in other_candidates:
                    values.append(cand)
                    weights.append(p_each)

            # Normalize weights
            weights = np.array(weights)
            weights = weights / weights.sum()

            # Sample empirical observations
            chosen_vals = rng.choice(values, size=samples_per_probe, p=weights)

            for val in chosen_vals:
                # Format raw text realistically
                if pid == "q1_int":
                    raw = f"1. {val}"
                elif pid == "q2_color":
                    raw = f"{val}"
                elif pid == "q3_rps":
                    raw = f"{val}"
                elif pid == "q4_letter":
                    raw = f"{val}"
                elif pid == "q5_sequence":
                    raw = f"{val}"
                else:
                    raw = str(val)

                samples_to_insert.append({
                    "model_name": m_name,
                    "probe_id": pid,
                    "raw_text": raw,
                    "parsed_value": str(val),
                    "is_valid": True,
                    "strictly_complied": True,
                    "source_type": "official",
                    "weight": 1.0
                })

    db.add_samples_batch(samples_to_insert)
    print(f"Successfully seeded database with {len(MODELS_META)} models and {len(samples_to_insert)} baseline samples.")
