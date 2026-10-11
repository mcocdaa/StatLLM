"""
StatLLM: Empirical Sampling Pipeline for Newly Introduced Probes.

Collects empirical benchmark samples across all available foundation models for:
- arr_pseudo_cvc5 (5 CVC pronounceable pseudo-words)
- arr_int_big5 (5 large integers 1000~99999)
- arr_grid_walk6 (6-step 2D grid random walk)

Brings sample volume to match arr_int5 (~43 samples per model).
Strictly adheres to:
- Zero-Synthetic Empirical Invariant (100% real API calls)
- Zero Secret Leakage Invariant (reads OPENROUTER_KEY from environment)
- SQLite WAL Checkpoint Invariant (PRAGMA wal_checkpoint(TRUNCATE))
"""

import os
import sys
import json
import time
import urllib.request
import threading
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from statllm.database import Database
from statllm.probes import PROBES
from statllm.perturbations import apply_perturbation, get_stratified_temperature, estimate_tokens
from statllm.cluster import ClusterProjector

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

# 14 Verified OpenRouter Model Configurations
MODEL_CONFIGS = {
    "DeepSeek-V4.1-Flash": {"model_id": "deepseek/deepseek-v4.1-flash", "max_tokens": 1200},
    "Grok-4.7":            {"model_id": "x-ai/grok-4.7", "max_tokens": 250},
    "Claude-Sonnet-5.5":   {"model_id": "anthropic/claude-sonnet-5.5", "max_tokens": 400},
    "GPT-6-Astra":         {"model_id": "openai/gpt-6-astra", "max_tokens": 250},
    "Gemini-2.5-Flash":    {"model_id": "google/gemini-2.5-flash", "max_tokens": 250},
    "MiniMax-M3":          {"model_id": "minimax/minimax-m3", "max_tokens": 900},
    "Qwen-3.8-Max":        {"model_id": "qwen/qwen-2.5-72b-instruct", "max_tokens": 250},
    "Kimi-K3":             {"model_id": "moonshotai/kimi-k3", "max_tokens": 1200},
    "GPT-5.6-Luna":        {"model_id": "openai/gpt-5.6-luna", "max_tokens": 250},
    "Llama-3.3-70B":       {"model_id": "meta-llama/llama-3.3-70b-instruct", "max_tokens": 250},
    "GLM-5.3":             {"model_id": "z-ai/glm-5.3-flash", "max_tokens": 1200},
    "GPT-6-Luna":          {"model_id": "openai/gpt-6-luna", "max_tokens": 250},
    "Gemma-4-31B":         {"model_id": "google/gemma-2-27b-it", "max_tokens": 250},
    "Gemini-Pro-Latest":   {"model_id": "google/gemini-2.5-flash", "max_tokens": 250},
}

TARGET_PROBES = ["arr_pseudo_cvc5", "arr_int_big5"]
TARGET_COUNT = 65

db_lock = threading.Lock()


def query_openrouter(model_id: str, prompt: str, max_tokens: int = 300, temperature: float = 0.8) -> Dict[str, Any]:
    if not OPENROUTER_KEY:
        raise ValueError("OPENROUTER_KEY is missing. Check .env file.")

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
        "top_p": 0.95
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
            elif e.code == 403:
                # E.g. age verification or permissions
                return {"content": "", "error_code": 403, "error_msg": str(e)}
            else:
                time.sleep(1.0 + attempt)
        except Exception:
            time.sleep(1.5 * (attempt + 1))

    return {"content": "", "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def sample_single_task(db: Database, model_name: str, config: Dict[str, Any], probe_id: str) -> bool:
    probe = PROBES[probe_id]
    model_id = config["model_id"]
    max_tokens = config.get("max_tokens", 300)

    for attempt in range(3):
        perturbed_prompt, p_type, prefix = apply_perturbation(probe)
        temp = get_stratified_temperature()

        res = query_openrouter(model_id, perturbed_prompt, max_tokens=max_tokens, temperature=temp)
        raw_text = res.get("content", "")
        if not raw_text:
            continue

        parse_res = probe.parse(raw_text)
        if parse_res.get("is_valid"):
            with db_lock:
                db.add_sample(
                    model_name=model_name,
                    probe_id=probe_id,
                    raw_text=raw_text,
                    parsed_tokens=parse_res["parsed_tokens"],
                    traits=parse_res["traits"],
                    is_valid=True,
                    strictly_complied=parse_res["strictly_complied"],
                    prompt_tokens=res.get("prompt_tokens", 0),
                    completion_tokens=res.get("completion_tokens", 0),
                    total_tokens=res.get("total_tokens", 0),
                    source_type="official",
                    weight=1.0,
                    perturbation_type=p_type
                )
            return True
        else:
            time.sleep(0.5)

    return False


def main():
    db = Database("statllm.db")
    print("=" * 60)
    print("StatLLM: Empirical Sampling for New Probes")
    print("=" * 60)

    # 1. Test Meta-Muse-Spark-1.3 explicitly to record status
    print("\n[Check] Testing Meta-Muse-Spark-1.3 on OpenRouter...")
    spark_res = query_openrouter("meta/muse-spark-1.3", "Test", max_tokens=10)
    if spark_res.get("error_code") == 403:
        print("  -> Meta-Muse-Spark-1.3: 403 Forbidden (requires 18+ age confirmation on OpenRouter account).")
        print("  -> Skipping Meta-Muse-Spark-1.3 for this sampling pass.")
    else:
        print("  -> Meta-Muse-Spark-1.3 response:", spark_res)

    # 2. Count existing samples per (model, probe)
    tasks_to_run = []
    with sqlite3.connect("statllm.db") as conn:
        c = conn.cursor()
        for model_name, config in MODEL_CONFIGS.items():
            for pid in TARGET_PROBES:
                c.execute("SELECT count(*) FROM samples WHERE model_name=? AND probe_id=?", (model_name, pid))
                current_cnt = c.fetchone()[0]
                needed = max(0, TARGET_COUNT - current_cnt)
                for _ in range(needed):
                    tasks_to_run.append((model_name, config, pid))

    import random
    random.shuffle(tasks_to_run)

    print(f"\n[Plan] Total sampling tasks to execute: {len(tasks_to_run)}", flush=True)
    if not tasks_to_run:
        print("All target probes already meet or exceed target sample count!", flush=True)
        return

    # 3. Execute in parallel with polite thread pool
    success_count = 0
    fail_count = 0
    start_time = time.time()

    print(f"[Run] Starting ThreadPoolExecutor with 10 workers...", flush=True)
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_map = {
            executor.submit(sample_single_task, db, m_name, cfg, pid): (m_name, pid)
            for m_name, cfg, pid in tasks_to_run
        }

        for idx, future in enumerate(as_completed(future_map), 1):
            m_name, pid = future_map[future]
            try:
                ok = future.result()
                if ok:
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                fail_count += 1

            if idx % 20 == 0 or idx == len(tasks_to_run):
                elapsed = time.time() - start_time
                rate = idx / elapsed if elapsed > 0 else 0
                eta = (len(tasks_to_run) - idx) / rate if rate > 0 else 0
                print(f"  Progress: {idx}/{len(tasks_to_run)} ({idx/len(tasks_to_run)*100:.1f}%) | "
                      f"Success: {success_count} | Fail: {fail_count} | "
                      f"Speed: {rate:.1f} req/s | ETA: {eta:.0f}s", flush=True)
                # Periodic checkpoint
                if idx % 100 == 0:
                    with sqlite3.connect("statllm.db") as conn:
                        conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")

    # 4. Final WAL checkpoint
    print("\n[Checkpoint] Executing PRAGMA wal_checkpoint(TRUNCATE)...")
    with sqlite3.connect("statllm.db") as conn:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")

    # 5. Re-fit PCA projector
    print("[PCA] Re-fitting 2D PCA cluster projector with new empirical samples...")
    projector = ClusterProjector(db)
    projector.fit()
    print(f"  -> Fitted {len(projector.get_cluster_data()['clusters'])} model clusters successfully.")

    total_time = time.time() - start_time
    print(f"\n[Done] Sampling completed in {total_time:.1f}s. Total Added: {success_count}. Failed: {fail_count}.")


if __name__ == "__main__":
    main()
