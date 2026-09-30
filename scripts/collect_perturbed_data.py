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


def query_grok(prompt: str, temperature: float = 0.8) -> Dict[str, Any]:
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
                "temperature": float(temperature),
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


def query_opencode(model: str, prompt: str, temperature: float = 0.8) -> Dict[str, Any]:
    cmd = ["opencode", "run", prompt, "-m", f"opencode-go/{model}"]
    if temperature <= 0.6:
        cmd.extend(["--variant", "minimal"])
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


import concurrent.futures
import threading

db_lock = threading.Lock()


def process_single_sample(
    model_name: str,
    probe_id: str,
    run_idx: int,
    target_runs: int,
    db_path: str = "statllm.db",
) -> Optional[int]:
    db = Database(db_path)
    probe = PROBES.get(probe_id)
    if not probe:
        return None

    full_prompt, pert_type, pert_prefix = apply_perturbation(probe.prompt)

    # Temperature schedule: 25% 0.5, 50% 0.8, 25% 1.0
    if target_runs <= 1:
        current_temp = 0.8
    elif run_idx < max(1, int(target_runs * 0.25)):
        current_temp = 0.5
    elif run_idx >= target_runs - max(1, int(target_runs * 0.25)):
        current_temp = 1.0
    else:
        current_temp = 0.8

    try:
        if model_name == "Grok-4.7":
            result = query_grok(full_prompt, temperature=current_temp)
        elif model_name == "DeepSeek-V4.1-Flash":
            result = query_opencode(
                "deepseek-v4.1-flash", full_prompt, temperature=current_temp
            )
        elif model_name == "GPT-5.6-Luna":
            result = query_opencode(
                "gpt-5.6-luna", full_prompt, temperature=current_temp
            )
        else:
            return None

        raw_text = result["raw_text"]
        p_tokens = result["prompt_tokens"]
        c_tokens = result["completion_tokens"]
        t_tokens = result["total_tokens"]

        parse_res = probe.parse(raw_text)
        is_valid = parse_res["is_valid"]
        parsed_tokens = parse_res["parsed_tokens"]
        traits = parse_res["traits"]
        strictly_complied = parse_res["strictly_complied"]

        with db_lock:
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
                temperature=current_temp,
            )

        print(
            f"  [√] [{model_name}] [{probe_id}] #{run_idx+1}/{target_runs} (T={current_temp}, {pert_type}): {parsed_tokens} | {t_tokens} tok"
        )
        return t_tokens
    except Exception as e:
        print(f"  [X] [{model_name}] [{probe_id}] #{run_idx+1} 出错: {e}")
        return None


def collect_target_samples(
    models: list,
    probes: list,
    target_per_probe: int = 16,
    workers: int = 3,
    db_path: str = "statllm.db",
):
    db = Database(db_path)
    print("=" * 70)
    print("StatLLM 目标样本量智能补齐与扰动采集启动")
    print(f"目标模型: {models}")
    print(f"目标探针: {probes}")
    print(f"单探针目标样本数: {target_per_probe} (单模型目标: {target_per_probe * len(probes)} 条)")
    print(f"并发工作线程: {workers}")
    print("=" * 70)

    # 1. Calculate missing tasks per model and interleave round-robin
    all_model_tasks = []
    print("\n[当前数据库现状与补齐计划]:")
    for model_name in models:
        total_curr = 0
        model_tasks = []
        for probe_id in probes:
            curr = db.get_sample_count(model_name, probe_id)
            total_curr += curr
            needed = max(0, target_per_probe - curr)
            for idx in range(curr, target_per_probe):
                model_tasks.append((model_name, probe_id, idx, target_per_probe, db_path))
        print(f"  - {model_name}: 当前已有 {total_curr} 条, 待采集 {len(model_tasks)} 条 (达到目标 {target_per_probe * len(probes)} 条)")
        all_model_tasks.append(model_tasks)

    tasks = []
    max_len = max(len(t) for t in all_model_tasks) if all_model_tasks else 0
    for i in range(max_len):
        for m_tasks in all_model_tasks:
            if i < len(m_tasks):
                tasks.append(m_tasks[i])

    print(f"\n>>> 总计待执行采集任务数: {len(tasks)} 次 API 调用 (已按模型交织并发优化)")
    if not tasks:
        print("所有模型已达到目标样本量，无需采集！")
        return

    # 2. Execute concurrently
    total_added = 0
    total_tokens_consumed = 0
    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_single_sample, *task) for task in tasks]
        for f in concurrent.futures.as_completed(futures):
            res_tokens = f.result()
            if res_tokens is not None:
                total_added += 1
                total_tokens_consumed += res_tokens

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"采集完成！本次新增入库样本数: {total_added}/{len(tasks)}")
    print(f"本次运行新增 Token 消耗: {total_tokens_consumed}")
    print(f"总计耗时: {elapsed:.1f} 秒 (平均 {(elapsed/max(1, total_added)):.2f} 秒/样本)")
    print("=" * 70)

    # Output updated token stats
    token_stats = db.get_token_usage_stats()
    print("\n[当前数据库累计 Token 统计]:")
    print(json.dumps(token_stats["overall"], indent=2, ensure_ascii=False))
    print("\n[分模型累计 Token 与样本统计]:")
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
        default=["arr_int5", "arr_color5", "arr_rps5", "arr_letter5", "arr_perm5"],
    )
    parser.add_argument("--target-total", type=int, default=80)
    parser.add_argument("--target-per-probe", type=int, default=None)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    probes = args.probes
    if args.target_per_probe is not None:
        target_per_probe = args.target_per_probe
    else:
        target_per_probe = max(1, args.target_total // len(probes))

    collect_target_samples(
        models=args.models,
        probes=probes,
        target_per_probe=target_per_probe,
        workers=args.workers,
    )
