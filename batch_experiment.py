"""
StatLLM Real Model Automated Batch Collector & Discriminability Benchmark.
Gathers live array outputs from Grok-4.7, DeepSeek-V4.1-Flash, and GPT-5.6-Luna,
and performs maximum likelihood attribution and cross-validation.
"""

import subprocess
import urllib.request
import json
import time
import re
from typing import Dict, Any, List

KOALA_URL = "https://api.openai.com/v1"
KOALA_KEY = "your_proxy_api_key_here"

PROMPTS = {
    "arr_int5": "生成一个包含5个在1到100之间随机整数的JSON数组，格式如[12, 45, 78, 3, 99]。仅输出该JSON数组，严禁任何多余文字或markdown代码块。",
    "arr_color5": "在[红, 橙, 黄, 绿, 青, 蓝, 紫]中随机挑选5次，组成JSON数组，例如[\"红\", \"蓝\", \"绿\", \"红\", \"紫\"]。仅输出该JSON数组，严禁任何多余文字或markdown代码块。",
    "arr_letter5": "Generate a JSON array of 5 random English capital letters (A-Z), e.g. [\"M\", \"X\", \"R\", \"A\", \"K\"]. Output strictly the JSON array only, without code blocks or extra words.",
    "arr_perm5": "将数字[1, 2, 3, 4, 5]完全随机打乱，输出一个打乱后的JSON数组，例如[3, 1, 5, 2, 4]。仅输出该JSON数组，严禁任何多余文字或markdown代码块。"
}


def query_grok(prompt: str) -> str:
    req = urllib.request.Request(
        f"{KOALA_URL}/chat/completions",
        headers={"Authorization": f"Bearer {KOALA_KEY}", "Content-Type": "application/json"},
        data=json.dumps({
            "model": "grok-4.7",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.85,
            "max_tokens": 100
        }).encode()
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read().decode())["choices"][0]["message"]["content"].strip()


def query_opencode(model: str, prompt: str) -> str:
    """Uses the opencode CLI to query models via opencode-go."""
    cmd = ["opencode", "run", prompt, "-m", f"opencode-go/{model}"]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    out = res.stdout.strip()
    # Strip opencode header like '> build · deepseek-v4.1-flash\n\n'
    out = re.sub(r"^>.*?\n+", "", out).strip()
    return out


def main():
    models = ["Grok-4.7", "DeepSeek-V4.1-Flash", "GPT-5.6-Luna"]
    samples_per_probe = 5
    collected_data = []

    print(f"=== Starting Real API Batch Collection across {len(models)} models ===")

    for m in models:
        print(f"\n[Model: {m}]")
        for pid, prompt in PROMPTS.items():
            print(f"  Querying probe: {pid} (x{samples_per_probe})...", end="", flush=True)
            for i in range(samples_per_probe):
                try:
                    if m == "Grok-4.7":
                        raw = query_grok(prompt)
                    elif m == "DeepSeek-V4.1-Flash":
                        raw = query_opencode("deepseek-v4.1-flash", prompt)
                    elif m == "GPT-5.6-Luna":
                        raw = query_opencode("gpt-5.6-luna", prompt)
                    
                    collected_data.append({
                        "model": m,
                        "probe_id": pid,
                        "run_index": i + 1,
                        "raw_text": raw
                    })
                    print(".", end="", flush=True)
                    time.sleep(0.5)
                except Exception as e:
                    print(f"E({e})", end="", flush=True)
            print(" Done!")

    # Save to JSON
    with open("real_api_samples.json", "w", encoding="utf-8") as f:
        json.dump(collected_data, f, ensure_ascii=False, indent=2)
    print(f"\nSuccessfully collected {len(collected_data)} live samples to real_api_samples.json")


if __name__ == "__main__":
    main()
