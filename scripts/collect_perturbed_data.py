"""Automated perturbed data collection across real models.

Applies random perturbations (chit-chat, persona, context noise, varied phrasing)
to standardized array probes, queries real API endpoints, extracts response arrays,
computes token usage, and stores records into SQLite database.
"""

import argparse
import json
import re
import subprocess
import time
import urllib.request
from typing import Dict, Any, Optional

from statllm.database import Database
from statllm.perturbations import apply_perturbation, estimate_tokens
from statllm.probes import PROBES

KOALA_URL = "https://pool.creative-koala-llm.top/v1"
KOALA_KEY = "sk-ad3e4bdafe5c4c832cefe1562eb82dce20c2d103b54716cf266c7a4a0cf1fa5f"


def query_grok(prompt: str) -> Dict[str, Any]:
    req = urllib.request.Request(
        f"{KOALA_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {KOALA_KEY}",
            "Content-Type": "application/json",
        },
        data=json.dumps(
            {
                "model": "grok-4.7",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.85,
                "max_tokens": 120,
            }
        ).encode(),
    )
    with urllib.request.urlopen(req, timeout=18) as r:
        data = json.loads(r.read().decode())
        choice = data["choices"][0]
        raw_text = choice["message"]["content"].strip()
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", estimate_tokens(prompt))
        completion_tokens = usage.get(
            "completion_tokens", estimate_tokens(raw_text)
        )
        return {
            "raw_text": raw_text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }


def query_opencode(model: str, prompt: str) -> Dict[str, Any]:
    cmd = ["opencode", "run", prompt, "-m", f"opencode-go/{model}"]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=35)
    out = res.stdout.strip()
    raw_text = re.sub(r"^>.*?\n+", "", out).strip()
    p_tokens = estimate_tokens(prompt)
    c_tokens = estimate_tokens(raw_text)
    return {
        "raw_text": raw_text,
        "prompt_tokens": p_tokens,
        "completion_tokens": c_tokens,
        "total_tokens": p_tokens + c_tokens,
    }


def collect_samples(
    models: list,
    probes: list,
    runs_per_probe: int = 2,
    db_path: str = "statllm.db",
    delay: float = 0.8,
):
    db = Database(db_path)
    print("=" * 70)
    print(f"StatLLM 扰动数据批量生成与采集启动")
    print(f"目标模型: {models}")
    print(f"目标探针: {probes}")
    print(f"每探针轮数: {runs_per_probe} 轮 (附带随机前缀扰动与人设包装)")
    print("=" * 70)

    total_added = 0
    total_tokens_consumed = 0

    for model_name in models:
        print(f"\n>>> 正在为模型 [{model_name}] 采集扰动数据...")
        for probe_id in probes:
            probe = PROBES.get(probe_id)
            if not probe:
                continue

            for run_idx in range(runs_per_probe):
                # Apply random perturbation (none, chit_chat, persona, context_noise)
                full_prompt, pert_type, pert_prefix = apply_perturbation(
                    probe.prompt
                )
                print(
                    f"  [{probe_id}] 轮次 #{run_idx+1} (扰动类型: {pert_type})"
                )

                try:
                    if model_name == "Grok-4.7":
                        result = query_grok(full_prompt)
                    elif model_name == "DeepSeek-V4.1-Flash":
                        result = query_opencode(
                            "deepseek-v4.1-flash", full_prompt
                        )
                    elif model_name == "GPT-5.6-Luna":
                        result = query_opencode("gpt-5.6-luna", full_prompt)
                    else:
                        print(f"  [跳过] 未知模型: {model_name}")
                        continue

                    raw_text = result["raw_text"]
                    p_tokens = result["prompt_tokens"]
                    c_tokens = result["completion_tokens"]
                    t_tokens = result["total_tokens"]
                    total_tokens_consumed += t_tokens

                    # Parse output using probe's native parser
                    parse_res = probe.parse(raw_text)
                    is_valid = parse_res["is_valid"]
                    parsed_tokens = parse_res["parsed_tokens"]
                    traits = parse_res["traits"]
                    strictly_complied = parse_res["strictly_complied"]

                    print(
                        f"    输出: {raw_text[:50]}... | Token消耗: {t_tokens} (Prompt: {p_tokens}, Completion: {c_tokens})"
                    )

                    # Store into DB
                    db.add_sample(
                        model_name=model_name,
                        probe_id=probe_id,
                        raw_text=raw_text,
                        parsed_tokens=parsed_tokens,
                        traits=traits,
                        is_valid=is_valid,
                        strictly_complied=strictly_complied,
                        source_type="official",
                        weight=1.0,
                        prompt_tokens=p_tokens,
                        completion_tokens=c_tokens,
                        total_tokens=t_tokens,
                        perturbation_type=pert_type,
                        perturbation_prefix=pert_prefix,
                    )
                    total_added += 1

                except Exception as e:
                    print(f"    [调用异常]: {e}")

                time.sleep(delay)

    print("\n" + "=" * 70)
    print(f"采集完成！新增入库样本数: {total_added}")
    print(f"本次运行新增 Token 消耗: {total_tokens_consumed}")
    print("=" * 70)

    # Output updated token stats
    token_stats = db.get_token_usage_stats()
    print("\n[当前数据库累计 Token 统计]:")
    print(json.dumps(token_stats["overall"], indent=2, ensure_ascii=False))
    print("\n[分模型累计 Token 统计]:")
    for m in token_stats["by_model"]:
        if m["api_call_count"] > 0:
            print(
                f"  - {m['model_name']}: {m['api_call_count']} 次调用, 总计 {m['total_tokens']} tokens (Prompt: {m['prompt_tokens']}, Completion: {m['completion_tokens']})"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StatLLM Perturbed Data Collector")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["Grok-4.7", "DeepSeek-V4.1-Flash", "GPT-5.6-Luna"],
    )
    parser.add_argument(
        "--probes",
        nargs="+",
        default=["arr_int5", "arr_color5", "arr_rps5", "arr_letter5"],
    )
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()

    collect_samples(models=args.models, probes=args.probes, runs_per_probe=args.runs)
