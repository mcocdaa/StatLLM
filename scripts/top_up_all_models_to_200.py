"""
StatLLM Model Fingerprint Benchmark Top-Up Script.

Top up every model in statllm.db to EXACTLY 200 authentic empirical samples.
Greedily selects underrepresented probes for each model to maximize probe balance.

Complies with:
- Invariant 1: Zero-Synthetic Empirical Invariant (100% real API calls).
- Invariant 4: Zero Secret Leakage Invariant (key in .env).
- Invariant 5: SQLite WAL Checkpoint & PCA fit.
"""

import os
import sys
import json
import time
import re
import random
import urllib.request
import urllib.error
import subprocess
import threading
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from statllm.database import Database
from statllm.probes import PROBES
from statllm.perturbations import (
    apply_perturbation,
    estimate_tokens,
    get_stratified_temperature,
    get_diverse_top_p,
)
from statllm.cluster import ClusterProjector

TARGET_PER_MODEL = 200
DB_PATH = "statllm.db"

# Safely load environment variables from .env
def load_env(env_path: str = ".env") -> None:
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

load_env()
OPENROUTER_KEY = os.environ.get("OPENROUTER_KEY", "")

# Model configurations
MODEL_CONFIGS = {
    "DeepSeek-V4.1-Flash": {"channel": "openrouter", "model_id": "deepseek/deepseek-v4.1-flash", "max_tokens": 300},
    "Grok-4.7":            {"channel": "openrouter", "model_id": "x-ai/grok-4.7", "max_tokens": 250},
    "Claude-Sonnet-5.5":   {"channel": "openrouter", "model_id": "anthropic/claude-sonnet-5.5", "max_tokens": 250},
    "GPT-6-Astra":         {"channel": "openrouter", "model_id": "openai/gpt-6-astra", "max_tokens": 250},
    "Gemini-2.5-Flash":    {"channel": "openrouter", "model_id": "google/gemini-2.5-flash", "max_tokens": 250},
    "MiniMax-M3":          {"channel": "openrouter", "model_id": "minimax/minimax-m3", "max_tokens": 300},
    "Qwen-3.8-Max":        {"channel": "openrouter", "model_id": "qwen/qwen3.8-max-prime", "max_tokens": 250},
    "Kimi-K3":             {"channel": "openrouter", "model_id": "moonshotai/kimi-k3", "max_tokens": 300},
    "GPT-5.6-Luna":        {"channel": "openrouter", "model_id": "openai/gpt-5.6-luna", "max_tokens": 250},
    "Llama-3.3-70B":       {"channel": "openrouter", "model_id": "meta-llama/llama-3.3-70b-instruct", "max_tokens": 250},
    "GLM-5.3":             {"channel": "openrouter", "model_id": "z-ai/glm-5.3-flash", "max_tokens": 250},
    "GPT-6-Luna":          {"channel": "openrouter", "model_id": "openai/gpt-6-luna", "max_tokens": 250},
    "Gemma-4-31B":         {"channel": "openrouter", "model_id": "google/gemma-4-31b-it", "max_tokens": 250},
    "Gemini-Pro-Latest":   {"channel": "openrouter", "model_id": "~google/gemini-pro-latest", "max_tokens": 450},
    "Meta-Muse-Spark-1.3": {"channel": "opencode",   "model_id": "opencode/muse-spark-1.3-contributor-free", "max_tokens": 250},
}

db_lock = threading.Lock()


def query_openrouter(
    model_id: str,
    prompt: str,
    max_tokens: int = 250,
    temperature: float = 0.8,
    top_p: float = 0.95
) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://statllm.ai",
        "X-Title": "StatLLM"
    }
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p
    }

    for attempt in range(3):
        try:
            req = urllib.request.Request(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                data=json.dumps(payload).encode()
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
                choice = data.get("choices", [{}])[0]
                msg = choice.get("message", {})
                content = msg.get("content") or choice.get("text") or ""
                reasoning = msg.get("reasoning") or ""
                usage = data.get("usage", {})
                p_tokens = usage.get("prompt_tokens", estimate_tokens(prompt))
                c_tokens = usage.get("completion_tokens", estimate_tokens(content))
                t_tokens = usage.get("total_tokens", p_tokens + c_tokens)
                return {
                    "content": content.strip() if content else "",
                    "reasoning": reasoning.strip() if reasoning else "",
                    "prompt_tokens": p_tokens,
                    "completion_tokens": c_tokens,
                    "total_tokens": t_tokens,
                }
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(2.0 * (attempt + 1))
            else:
                time.sleep(1.0 + attempt)
        except Exception:
            time.sleep(1.5 * (attempt + 1))

    return {"content": "", "reasoning": "", "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def query_opencode(model_id: str, prompt: str) -> Dict[str, Any]:
    cmd = ["opencode", "run", prompt, "-m", model_id]
    for attempt in range(2):
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=35)
            out = re.sub(r"^>.*?\n+", "", res.stdout).strip()
            if out:
                p_tok = estimate_tokens(prompt)
                c_tok = estimate_tokens(out)
                return {
                    "content": out,
                    "reasoning": "",
                    "prompt_tokens": p_tok,
                    "completion_tokens": c_tok,
                    "total_tokens": p_tok + c_tok
                }
        except Exception:
            time.sleep(1.0)
    return {"content": "", "reasoning": "", "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def sample_single_unit(model_name: str, probe_id: str) -> bool:
    """Attempts to collect and store one valid sample for (model_name, probe_id)."""
    cfg = MODEL_CONFIGS.get(model_name)
    if not cfg:
        return False
    probe = PROBES.get(probe_id)
    if not probe:
        return False

    channel = cfg["channel"]
    mid = cfg["model_id"]
    max_tok = cfg.get("max_tokens", 250)

    # Retry up to 3 times with fresh perturbations if model response is invalid
    for attempt in range(3):
        full_prompt, actual_pert_type, pert_prefix = apply_perturbation(probe, diverse_phrasing=True)
        temp = get_stratified_temperature()
        top_p = get_diverse_top_p()

        if channel == "openrouter":
            res = query_openrouter(mid, full_prompt, max_tokens=max_tok, temperature=temp, top_p=top_p)
        else:
            res = query_opencode(mid, full_prompt)

        raw_text = res["content"]
        if not raw_text and res["reasoning"]:
            parsed_reasoning = probe.parse(res["reasoning"])
            if parsed_reasoning["is_valid"]:
                raw_text = res["reasoning"]

        if not raw_text:
            continue

        parsed = probe.parse(raw_text)
        if not parsed["is_valid"]:
            continue

        p_tok = res["prompt_tokens"] or estimate_tokens(full_prompt)
        c_tok = res["completion_tokens"] or estimate_tokens(raw_text)
        t_tok = res["total_tokens"] or (p_tok + c_tok)

        record = {
            "model_name": model_name,
            "probe_id": probe_id,
            "raw_text": raw_text,
            "parsed_tokens": parsed["parsed_tokens"],
            "traits": parsed.get("traits", {}),
            "is_valid": parsed["is_valid"],
            "strictly_complied": parsed["strictly_complied"],
            "source_type": "official",
            "weight": 1.0,
            "prompt_tokens": p_tok,
            "completion_tokens": c_tok,
            "total_tokens": t_tok,
            "perturbation_type": actual_pert_type,
            "perturbation_prefix": pert_prefix,
            "temperature": temp
        }

        with db_lock:
            db = Database(DB_PATH)
            db.add_sample(**record)

        print(f"  [√] [{model_name}] {probe_id} (T={temp}, {actual_pert_type}): {parsed['parsed_tokens']} (tok={t_tok})")
        return True

    print(f"  [X] [{model_name}] {probe_id} FAILED after 3 attempts.")
    return False


def main():
    db = Database(DB_PATH)
    all_probes = list(PROBES.keys())

    print("=" * 80)
    print("StatLLM 全模型基准样本统一对齐工程：统一补齐至 200 条样本")
    print(f"目标定额: 每模型各 {TARGET_PER_MODEL} 条样本 | 15 模型总计 3,000 条")
    print("=" * 80)

    # Check current counts
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT name FROM models ORDER BY name;")
        all_models = [r[0] for r in c.fetchall()]

    tasks = []
    print("\n当前各模型样本储备与补齐规划:")
    for m in all_models:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("SELECT probe_id, count(*) FROM samples WHERE model_name=? GROUP BY probe_id;", (m,))
            counts = {p: 0 for p in all_probes}
            counts.update(dict(c.fetchall()))
        
        cur_total = sum(counts.values())
        needed = max(0, TARGET_PER_MODEL - cur_total)
        print(f"  • {m:22}: 当前 {cur_total:3d} 条 | 需补齐 {needed:3d} 条")

        # Greedily assign to probes with minimum count
        for _ in range(needed):
            p_min = min(all_probes, key=lambda p: counts[p])
            tasks.append((m, p_min))
            counts[p_min] += 1

    total_tasks = len(tasks)
    print(f"\n总计调度执行 {total_tasks} 次真实 API/模型调用...")

    # Split tasks: OpenRouter vs OpenCode
    openrouter_tasks = [t for t in tasks if MODEL_CONFIGS[t[0]]["channel"] == "openrouter"]
    opencode_tasks   = [t for t in tasks if MODEL_CONFIGS[t[0]]["channel"] == "opencode"]

    random.shuffle(openrouter_tasks)
    print(f"  - OpenRouter 任务: {len(openrouter_tasks)} 个 (12 并发 Worker)")
    print(f"  - OpenCode 任务:   {len(opencode_tasks)} 个 (4 并发 Worker)")

    start_time = time.time()
    successful = 0

    # Execute OpenRouter and OpenCode concurrently
    with ThreadPoolExecutor(max_workers=12) as or_exec, ThreadPoolExecutor(max_workers=4) as oc_exec:
        futures = []
        for t in openrouter_tasks:
            futures.append(or_exec.submit(sample_single_unit, *t))
        for t in opencode_tasks:
            futures.append(oc_exec.submit(sample_single_unit, *t))

        for f in as_completed(futures):
            try:
                if f.result():
                    successful += 1
            except Exception as e:
                print(f"Task exception: {e}")

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"补齐采样全部执行完毕！成功入库: {successful}/{total_tasks} ({successful/total_tasks*100:.1f}%)")
    print(f"总耗时: {elapsed:.1f} 秒 ({elapsed/60:.1f} 分钟)")
    print("=" * 80)

    # Invariant 5: Checkpoint SQLite WAL
    print("正在执行 PRAGMA wal_checkpoint(TRUNCATE)...")
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
    print("SQLite WAL 归档刷写完成。")

    # Re-fit 2D PCA cluster projector
    print("正在基于最新全量真实数据重新拟合 2D PCA 聚类投影器...")
    projector = ClusterProjector(db)
    projector.fit()
    print("PCA 聚类投影器重新拟合完成！")

    # Output final summary matching Web UI table
    token_usage = db.get_token_usage_stats()
    print("\n[最新数据库全景对齐大盘 (Web 表格对照)]:")
    print(f"{'模型':22} | {'样本数':>6} | {'输入 Token':>11} | {'生成 Token':>10} | {'总 Token':>11}")
    print("-" * 72)
    for row in token_usage.get("by_model", []):
        print(f"{row['model_name']:22} | {row['sample_count']:6d} | {row['prompt_tokens']:11,d} | {row['completion_tokens']:10,d} | {row['total_tokens']:11,d}")


if __name__ == "__main__":
    main()
