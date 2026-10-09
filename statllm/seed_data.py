"""
StatLLM Baseline Seed Data Generator for Array Probes.

Initializes baseline benchmark distributions across modern LLMs,
including Grok-4.7, DeepSeek-V4.1-Flash, GPT-5.6-Luna, Claude-3.5-Sonnet, Gemini-2.0, and Qwen-2.5.
"""

import json
import numpy as np
from typing import Dict, Any, List
from statllm.database import Database
from statllm.probes import PROBES


MODELS_META = [
    {
        "name": "Grok-4.7",
        "display_name": "xAI Grok 4.7",
        "provider": "xAI",
        "color": "#38bdf8",  # Sky Blue
    },
    {
        "name": "DeepSeek-V4.1-Flash",
        "display_name": "DeepSeek V4.1 Flash",
        "provider": "DeepSeek",
        "color": "#06b6d4",  # Cyan Blue
    },
    {
        "name": "GPT-5.6-Luna",
        "display_name": "OpenAI GPT-5.6 Luna",
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
    }
]

# Empirical array preferences
MODEL_ARRAY_PREFERENCES = {
    "Grok-4.7": {
        "arr_int5": {
            "token_bias": {"27": 0.22, "44": 0.18, "83": 0.12, "5": 0.10, "61": 0.10, "63": 0.08, "89": 0.08, "8": 0.06, "91": 0.06},
            "has_dup": False,
            "is_sorted": False
        },
        "arr_color5": {
            "token_bias": {"黄": 0.28, "青": 0.24, "橙": 0.16, "绿": 0.14, "红": 0.10, "蓝": 0.05, "紫": 0.03},
            "has_dup": False
        },
        "arr_rps5": {
            "token_bias": {"布": 0.42, "剪刀": 0.36, "石头": 0.22},
            "has_dup": True
        },
        "arr_letter5": {
            "token_bias": {"K": 0.24, "W": 0.20, "B": 0.18, "R": 0.16, "M": 0.12, "X": 0.10},
            "has_dup": False
        },
        "arr_perm5": {
            "perms": {"4,1,5,3,2": 0.35, "3,5,1,4,2": 0.35, "2,5,1,4,3": 0.30}
        }
    },
    "DeepSeek-V4.1-Flash": {
        "arr_int5": {
            "token_bias": {"66": 0.24, "88": 0.20, "18": 0.16, "55": 0.12, "99": 0.10, "24": 0.08, "7": 0.05, "33": 0.05},
            "has_dup": False,
            "is_sorted": False
        },
        "arr_color5": {
            "token_bias": {"红": 0.38, "蓝": 0.26, "黄": 0.16, "青": 0.08, "紫": 0.05, "绿": 0.04, "橙": 0.03},
            "has_dup": False
        },
        "arr_rps5": {
            "token_bias": {"布": 0.46, "石头": 0.34, "剪刀": 0.20},
            "has_dup": True
        },
        "arr_letter5": {
            "token_bias": {"D": 0.25, "Z": 0.20, "V": 0.18, "E": 0.15, "K": 0.12, "S": 0.10},
            "has_dup": False
        },
        "arr_perm5": {
            "perms": {"5,3,1,4,2": 0.35, "2,4,1,5,3": 0.35, "3,1,5,2,4": 0.30}
        }
    },
    "GPT-5.6-Luna": {
        "arr_int5": {
            "token_bias": {"42": 0.25, "37": 0.18, "73": 0.14, "7": 0.12, "17": 0.10, "88": 0.08, "55": 0.05, "23": 0.05},
            "has_dup": False,
            "is_sorted": True  # Strong sorting bias!
        },
        "arr_color5": {
            "token_bias": {"蓝": 0.36, "红": 0.25, "绿": 0.15, "紫": 0.10, "黄": 0.06, "橙": 0.04, "青": 0.04},
            "has_dup": False
        },
        "arr_rps5": {
            "token_bias": {"石头": 0.48, "剪刀": 0.30, "布": 0.22},
            "has_dup": True
        },
        "arr_letter5": {
            "token_bias": {"M": 0.24, "R": 0.20, "X": 0.16, "A": 0.14, "T": 0.12, "S": 0.08},
            "has_dup": False
        },
        "arr_perm5": {
            "perms": {"1,3,5,2,4": 0.38, "2,1,4,3,5": 0.34, "3,1,2,5,4": 0.28}
        }
    },
    "Claude-3.5-Sonnet": {
        "arr_int5": {
            "token_bias": {"47": 0.22, "77": 0.18, "23": 0.16, "89": 0.12, "14": 0.10, "56": 0.08, "31": 0.07, "92": 0.07},
            "has_dup": False,
            "is_sorted": False
        },
        "arr_color5": {
            "token_bias": {"绿": 0.38, "紫": 0.26, "蓝": 0.16, "黄": 0.08, "青": 0.05, "红": 0.04, "橙": 0.03},
            "has_dup": False
        },
        "arr_rps5": {
            "token_bias": {"剪刀": 0.45, "布": 0.35, "石头": 0.20},
            "has_dup": True
        },
        "arr_letter5": {
            "token_bias": {"S": 0.26, "C": 0.22, "L": 0.18, "K": 0.14, "H": 0.10, "E": 0.10},
            "has_dup": False
        },
        "arr_perm5": {
            "perms": {"4,2,5,1,3": 0.36, "3,5,2,1,4": 0.34, "5,1,3,2,4": 0.30}
        }
    },
    "Gemini-2.0-Flash": {
        "arr_int5": {
            "token_bias": {"27": 0.20, "64": 0.18, "81": 0.16, "12": 0.14, "3": 0.12, "45": 0.08, "92": 0.06, "19": 0.06},
            "has_dup": False,
            "is_sorted": False
        },
        "arr_color5": {
            "token_bias": {"黄": 0.36, "青": 0.24, "橙": 0.18, "绿": 0.08, "蓝": 0.06, "红": 0.05, "紫": 0.03},
            "has_dup": False
        },
        "arr_rps5": {
            "token_bias": {"石头": 0.50, "布": 0.28, "剪刀": 0.22},
            "has_dup": True
        },
        "arr_letter5": {
            "token_bias": {"G": 0.28, "M": 0.20, "O": 0.16, "B": 0.14, "L": 0.12, "E": 0.10},
            "has_dup": False
        },
        "arr_perm5": {
            "perms": {"1,4,2,5,3": 0.38, "3,1,4,2,5": 0.32, "4,1,3,5,2": 0.30}
        }
    },
    "Qwen-2.5-72B": {
        "arr_int5": {
            "token_bias": {"8": 0.26, "28": 0.18, "68": 0.16, "88": 0.14, "16": 0.10, "38": 0.08, "78": 0.04, "46": 0.04},
            "has_dup": False,
            "is_sorted": False
        },
        "arr_color5": {
            "token_bias": {"青": 0.34, "红": 0.28, "橙": 0.16, "蓝": 0.08, "绿": 0.06, "黄": 0.05, "紫": 0.03},
            "has_dup": False
        },
        "arr_rps5": {
            "token_bias": {"剪刀": 0.46, "石头": 0.34, "布": 0.20},
            "has_dup": True
        },
        "arr_letter5": {
            "token_bias": {"Q": 0.32, "W": 0.20, "E": 0.16, "N": 0.12, "A": 0.10, "C": 0.10},
            "has_dup": False
        },
        "arr_perm5": {
            "perms": {"2,5,3,1,4": 0.38, "5,2,4,1,3": 0.32, "3,4,1,5,2": 0.30}
        }
    }
}


def seed_database(db: Database, samples_per_probe: int = 100):
    """
    Seeds database with array observations.
    """
    db.clear_database()
    rng = np.random.default_rng(2026)

    # 1. Register Models
    for m in MODELS_META:
        db.add_model(
            name=m["name"],
            display_name=m["display_name"],
            provider=m["provider"],
            color=m["color"]
        )

    # 2. Insert array samples
    samples_to_insert = []
    for m_meta in MODELS_META:
        m_name = m_meta["name"]
        prefs = MODEL_ARRAY_PREFERENCES[m_name]

        for pid, probe in PROBES.items():
            probe_pref = prefs.get(pid, {})
            if not probe_pref:
                continue

            for _ in range(samples_per_probe):
                if pid == "arr_perm5":
                    # Sample permutation
                    perms = list(probe_pref["perms"].keys())
                    p_weights = list(probe_pref["perms"].values())
                    perm_str = rng.choice(perms, p=p_weights)
                    tokens = perm_str.split(",")
                    raw_text = f"[{', '.join(tokens)}]"
                    traits = {"canonical_perm": perm_str, "first_token": tokens[0]}
                else:
                    token_bias = probe_pref["token_bias"]
                    cand_tokens = list(token_bias.keys())
                    cand_weights = list(token_bias.values())

                    # Normalize weights
                    w_arr = np.array(cand_weights) / sum(cand_weights)
                    
                    # Choose 5 elements
                    chosen = rng.choice(cand_tokens, size=5, replace=True, p=w_arr)
                    tokens = [str(x) for x in chosen]
                    
                    # Formatting
                    if pid == "arr_int5":
                        if probe_pref.get("is_sorted", False):
                            tokens = sorted(tokens, key=lambda x: int(x))
                        raw_text = f"[{', '.join(tokens)}]"
                        traits = {
                            "has_duplicates": len(tokens) != len(set(tokens)),
                            "is_sorted": tokens == sorted(tokens, key=lambda x: int(x)),
                            "first_token": tokens[0]
                        }
                    else:
                        raw_text = json.dumps(tokens, ensure_ascii=False)
                        traits = {
                            "has_duplicates": len(tokens) != len(set(tokens)),
                            "first_token": tokens[0]
                        }

                samples_to_insert.append({
                    "model_name": m_name,
                    "probe_id": pid,
                    "raw_text": raw_text,
                    "parsed_tokens": tokens,
                    "is_valid": True,
                    "strictly_complied": True,
                    "traits": traits,
                    "source_type": "official",
                    "weight": 1.0
                })

    db.add_samples_batch(samples_to_insert)
    print(f"Seeded {len(MODELS_META)} models with {len(samples_to_insert)} array samples.")
