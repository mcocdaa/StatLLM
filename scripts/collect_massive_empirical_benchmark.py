"""
StatLLM Massive Empirical Benchmark Collection & Probe Expansion Script.

Uses real OpenRouter API calls to collect empirical samples across 14 frontier LLMs
over all 9 discrete mathematical probes (cold-starting the 4 new probes to >= 150 samples
and augmenting the 5 existing probes with temperature-stratified and perturbed samples).

Complies with:
- Invariant 1: Zero-Synthetic Empirical Invariant (100% real API calls).
- Invariant 4: Zero Secret Leakage Invariant (reads key from .env / env var).
- Invariant 5: SQLite WAL Checkpoint Invariant (wal_checkpoint and PCA re-fit).
"""

import os
import sys
import json
import time
import random
import urllib.request
import urllib.error
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

# Safely load environment variables from .env if present
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

if not OPENROUTER_KEY:
    print("[ERROR] OPENROUTER_KEY not found in environment or .env file.")
    sys.exit(1)

# Frontier models across top AI companies
ACTIVE_MODELS = [
    {
        "name": "DeepSeek-V4.1-Flash",
        "display_name": "DeepSeek V4.1 Flash",
        "provider": "DeepSeek",
        "color": "#06b6d4",
        "model_id": "deepseek/deepseek-v4.1-flash",
        "max_tokens": 300,
    },
    {
        "name": "Claude-Sonnet-5.5",
        "display_name": "Anthropic Claude Sonnet 5.5",
        "provider": "Anthropic",
        "color": "#7c3aed",
        "model_id": "anthropic/claude-sonnet-5.5",
        "max_tokens": 250,
    },
    {
        "name": "GPT-6-Astra",
        "display_name": "OpenAI GPT-6 Astra",
        "provider": "OpenAI",
        "color": "#10b981",
        "model_id": "openai/gpt-6-astra",
        "max_tokens": 250,
    },
    {
        "name": "GPT-5.6-Luna",
        "display_name": "OpenAI GPT-5.6 Luna",
        "provider": "OpenAI",
        "color": "#14b8a6",
        "model_id": "openai/gpt-5.6-luna",
        "max_tokens": 250,
    },
    {
        "name": "GPT-6-Luna",
        "display_name": "OpenAI GPT-6 Luna",
        "provider": "OpenAI",
        "color": "#059669",
        "model_id": "openai/gpt-6-luna",
        "max_tokens": 250,
    },
    {
        "name": "Gemini-2.5-Flash",
        "display_name": "Google Gemini 2.5 Flash",
        "provider": "Google",
        "color": "#d97706",
        "model_id": "google/gemini-2.5-flash",
        "max_tokens": 250,
    },
    {
        "name": "Gemini-Pro-Latest",
        "display_name": "Google Gemini Pro (Latest)",
        "provider": "Google",
        "color": "#b45309",
        "model_id": "~google/gemini-pro-latest",
        "max_tokens": 450,  # extra margin for reasoning tokens
    },
    {
        "name": "Gemma-4-31B",
        "display_name": "Google Gemma 4 31B",
        "provider": "Google",
        "color": "#0284c7",
        "model_id": "google/gemma-4-31b-it",
        "max_tokens": 250,
    },
    {
        "name": "Llama-3.3-70B",
        "display_name": "Meta Llama 3.3 70B",
        "provider": "Meta",
        "color": "#2563eb",
        "model_id": "meta-llama/llama-3.3-70b-instruct",
        "max_tokens": 250,
    },
    {
        "name": "MiniMax-M3",
        "display_name": "MiniMax M3",
        "provider": "MiniMax",
        "color": "#ec4899",
        "model_id": "minimax/minimax-m3",
        "max_tokens": 300,
    },
    {
        "name": "Kimi-K3",
        "display_name": "MoonshotAI Kimi K3",
        "provider": "MoonshotAI",
        "color": "#8b5cf6",
        "model_id": "moonshotai/kimi-k3",
        "max_tokens": 300,
    },
    {
        "name": "Grok-4.7",
        "display_name": "xAI Grok 4.7",
        "provider": "xAI",
        "color": "#38bdf8",
        "model_id": "x-ai/grok-4.7",
        "max_tokens": 250,
    },
    {
        "name": "GLM-5.3",
        "display_name": "Zhipu GLM-5.3",
        "provider": "Zhipu AI",
        "color": "#6366f1",
        "model_id": "z-ai/glm-5.3-flash",
        "max_tokens": 250,
    },
    {
        "name": "Qwen-3.8-Max",
        "display_name": "Alibaba Qwen 3.8 Max",
        "provider": "Alibaba Cloud",
        "color": "#ea580c",
        "model_id": "qwen/qwen3.8-max-prime",
        "max_tokens": 250,
    },
]

# Probes schedule
NEW_PROBES = ["arr_coin10", "arr_dice6", "arr_prime5", "arr_bit8"]
EXISTING_PROBES = ["arr_color5", "arr_int5", "arr_letter5", "arr_perm5", "arr_rps5"]

# Runs per probe per model
RUNS_PER_NEW_PROBE = 15      # 15 * 14 = 210 samples per new probe (> 100 benchmark goal)
RUNS_PER_EXISTING_PROBE = 5  # 5 * 14 = 70 additional diverse samples per existing probe

db_lock = threading.Lock()


def query_openrouter(
    model_id: str,
    prompt: str,
    max_tokens: int = 250,
    temperature: float = 0.8,
    top_p: float = 0.95
) -> Dict[str, Any]:
    """Execute physical API call against OpenRouter endpoint with retries."""
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


def sample_worker(
    model_cfg: Dict[str, Any],
    probe_id: str,
    run_idx: int,
    total_runs: int,
    db_path: str = "statllm.db"
) -> Optional[Dict[str, Any]]:
    probe = PROBES.get(probe_id)
    if not probe:
        return None

    # Multi-axis diverse prompt perturbation
    full_prompt, actual_pert_type, pert_prefix = apply_perturbation(probe, diverse_phrasing=True)
    temp = get_stratified_temperature()
    top_p = get_diverse_top_p()

    m_name = model_cfg["name"]
    mid = model_cfg["model_id"]
    max_tok = model_cfg.get("max_tokens", 250)

    res = query_openrouter(mid, full_prompt, max_tokens=max_tok, temperature=temp, top_p=top_p)
    raw_text = res["content"]

    # If content was empty due to reasoning buffer, check reasoning block for valid output
    if not raw_text and res["reasoning"]:
        parsed_from_reasoning = probe.parse(res["reasoning"])
        if parsed_from_reasoning["is_valid"]:
            raw_text = res["reasoning"]

    if not raw_text:
        print(f"  [X] [{m_name}] {probe_id} #{run_idx+1}/{total_runs} -> EMPTY")
        return None

    parsed = probe.parse(raw_text)
    if not parsed["is_valid"]:
        print(f"  [!] [{m_name}] {probe_id} #{run_idx+1}/{total_runs} -> INVALID: {raw_text[:50]}")
        return None

    p_tok = res["prompt_tokens"] or estimate_tokens(full_prompt)
    c_tok = res["completion_tokens"] or estimate_tokens(raw_text)
    t_tok = res["total_tokens"] or (p_tok + c_tok)

    record = {
        "model_name": m_name,
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
        db = Database(db_path)
        db.add_sample(**record)

    print(f"  [√] [{m_name}] {probe_id} #{run_idx+1}/{total_runs} (T={temp}, {actual_pert_type}): {parsed['parsed_tokens']} (tok={t_tok})")
    return record


def main():
    db_path = "statllm.db"
    db = Database(db_path)

    print("=" * 80)
    print("StatLLM 全球顶尖 AI 模型全量离散数学探针大规模物理采样补齐与扩充管线")
    print(f"覆盖模型: {len(ACTIVE_MODELS)} 家全球顶级旗舰")
    print(f"探针体系: {len(PROBES)} 大探针 (4 个新离散数理探针 + 5 个标准探针)")
    print(f"新探针目标: 每模型 {RUNS_PER_NEW_PROBE} 轮 (每题 {RUNS_PER_NEW_PROBE * len(ACTIVE_MODELS)} 样本，稳超 100 样本大关)")
    print(f"已有探针目标: 每模型 {RUNS_PER_EXISTING_PROBE} 轮扰动补齐 (每题累计达 300+ 样本)")
    print("=" * 80)

    # 1. Register/ensure metadata for all models
    for m in ACTIVE_MODELS:
        db.add_model(
            name=m["name"],
            display_name=m["display_name"],
            provider=m["provider"],
            color=m["color"]
        )

    # 2. Build task list
    tasks = []
    # New probes (prioritize heavily to bootstrap distributions)
    for model_cfg in ACTIVE_MODELS:
        for p_id in NEW_PROBES:
            for idx in range(RUNS_PER_NEW_PROBE):
                tasks.append((model_cfg, p_id, idx, RUNS_PER_NEW_PROBE, db_path))

    # Existing probes
    for model_cfg in ACTIVE_MODELS:
        for p_id in EXISTING_PROBES:
            for idx in range(RUNS_PER_EXISTING_PROBE):
                tasks.append((model_cfg, p_id, idx, RUNS_PER_EXISTING_PROBE, db_path))

    random.shuffle(tasks)
    total_tasks = len(tasks)
    print(f"总计规划物理调用: {total_tasks} 次真实 API 请求...")

    start_time = time.time()
    successful = 0
    total_tokens = 0

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_map = {executor.submit(sample_worker, *t): t for t in tasks}
        for future in as_completed(future_map):
            try:
                rec = future.result()
                if rec:
                    successful += 1
                    total_tokens += rec["total_tokens"]
            except Exception as e:
                task_t = future_map[future]
                print(f"  [X] Exception on {task_t[0]['name']}: {e}")

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"大规模物理采样全部执行完毕！成功入库: {successful}/{total_tasks} ({successful/total_tasks*100:.1f}%)")
    print(f"总耗时: {elapsed:.1f} 秒 ({elapsed/60:.1f} 分钟) | 累计消耗 Token: {total_tokens:,}")
    print("=" * 80)

    # Invariant 5: Checkpoint SQLite WAL
    print("正在执行 PRAGMA wal_checkpoint(TRUNCATE)...")
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
    print("SQLite WAL 归档刷写完成。")

    # Re-fit 2D PCA cluster projector
    print("正在基于最新全量真实数据重新拟合 2D PCA 聚类投影器...")
    projector = ClusterProjector(db)
    projector.fit()
    print("PCA 聚类投影器重新拟合完成！")

    # Output statistical summary
    stats = db.get_stats()
    print("\n[最新数据库统计大盘]:")
    print(f"  • 总样本数: {stats['total_samples']:,} 条")
    print(f"  • 扰动类型分布: {stats['perturbation_breakdown']}")

    print("\n[各探针样本量分布 (目标 >= 100)]:")
    with sqlite3.connect(db_path) as conn:
        c = conn.cursor()
        c.execute("SELECT probe_id, count(*) FROM samples GROUP BY probe_id ORDER BY count(*) DESC;")
        for p_row in c.fetchall():
            flag = "✓ (达标 >= 100)" if p_row[1] >= 100 else "! (< 100)"
            print(f"    - {p_row[0]:15}: {p_row[1]:4d} 条 {flag}")

    print("\n[各模型样本量分布]:")
    for m in stats["model_breakdown"]:
        print(f"    - {m['model_name']:22}: {m['count']:4d} 条")


if __name__ == "__main__":
    main()
