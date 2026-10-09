"""
StatLLM Meta Muse-Spark 1.3 & Google Gemini-Pro-Latest / Gemma-4 Onboarding Script.

Adds and evaluates:
1. Meta Muse-Spark 1.3 (Meta's 1.3 flagship via OpenCode muse-spark-1.3-contributor-free)
2. Google Gemini Pro Latest (~google/gemini-pro-latest on OpenRouter with reasoning)
3. Google Gemma 4 31B (google/gemma-4-31b-it on OpenRouter)

All samples generated with randomized prompt perturbations (persona, chit-chat, context noise, phrasing) across all 5 standard probes.
"""

import os
import sys
import json
import time
import re
import random
import urllib.request
import subprocess
import threading
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from statllm.database import Database
from statllm.probes import PROBES
from statllm.perturbations import apply_perturbation, estimate_tokens
from statllm.cluster import ClusterProjector

OPENROUTER_KEY = os.environ.get("OPENROUTER_KEY", "")

NEW_MODELS = [
    {
        "name": "Meta-Muse-Spark-1.3",
        "display_name": "Meta Muse-Spark 1.3",
        "provider": "Meta",
        "color": "#1d4ed8",
        "channel": "opencode",
        "model_id": "opencode/muse-spark-1.3-contributor-free",
        "max_tokens": 200,
        "runs_per_probe": 14  # 5 x 14 = 70 samples
    },
    {
        "name": "Gemini-Pro-Latest",
        "display_name": "Google Gemini Pro (Latest)",
        "provider": "Google",
        "color": "#b45309",
        "channel": "openrouter",
        "model_id": "~google/gemini-pro-latest",
        "max_tokens": 600,
        "runs_per_probe": 14  # 5 x 14 = 70 samples
    },
    {
        "name": "Gemma-4-31B",
        "display_name": "Google Gemma 4 31B",
        "provider": "Google",
        "color": "#0284c7",
        "channel": "openrouter",
        "model_id": "google/gemma-4-31b-it",
        "max_tokens": 200,
        "runs_per_probe": 12  # 5 x 12 = 60 samples
    }
]

db_lock = threading.Lock()


def query_opencode(model_id: str, prompt: str, temperature: float = 0.8) -> str:
    for attempt in range(2):
        try:
            cmd = ["opencode", "run", prompt, "-m", model_id]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=35)
            out = re.sub(r"^>.*?\n+", "", res.stdout).strip()
            if out:
                return out
        except Exception:
            time.sleep(1.0)
    return ""


def query_openrouter(model_id: str, prompt: str, max_tokens: int = 400, temperature: float = 0.8) -> str:
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://statllm.ai",
                    "X-Title": "StatLLM"
                },
                data=json.dumps({
                    "model": model_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }).encode()
            )
            with urllib.request.urlopen(req, timeout=25) as r:
                data = json.loads(r.read().decode())
                choice = data["choices"][0]
                msg = choice.get("message", {})
                content = msg.get("content") or choice.get("text") or ""
                if content:
                    return content.strip()
        except Exception:
            time.sleep(1.5 + attempt)
    return ""


def sample_task(model_cfg: Dict[str, Any], probe_id: str, run_idx: int, db_path: str = "statllm.db") -> Optional[Dict[str, Any]]:
    probe = PROBES.get(probe_id)
    if not probe:
        return None

    full_prompt, actual_pert_type, pert_prefix = apply_perturbation(probe.prompt)
    m_name = model_cfg["name"]
    channel = model_cfg["channel"]
    mid = model_cfg["model_id"]
    max_tok = model_cfg.get("max_tokens", 300)

    temp = round(random.choice([0.7, 0.8, 0.85, 0.9, 0.95]), 2)

    if channel == "opencode":
        raw_text = query_opencode(mid, full_prompt, temperature=temp)
    else:
        raw_text = query_openrouter(mid, full_prompt, max_tokens=max_tok, temperature=temp)

    if not raw_text:
        print(f"  [X] [{m_name}] {probe_id} #{run_idx+1} ({actual_pert_type}) -> EMPTY / TIMEOUT")
        return None

    parsed = probe.parse(raw_text)
    if not parsed["is_valid"]:
        print(f"  [!] [{m_name}] {probe_id} #{run_idx+1} ({actual_pert_type}) -> INVALID: {raw_text[:60]}")
        return None

    p_tok = estimate_tokens(full_prompt)
    c_tok = estimate_tokens(raw_text)
    t_tok = p_tok + c_tok

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

    print(f"  [√] [{m_name}] {probe_id} #{run_idx+1} ({actual_pert_type}, T={temp}): {parsed['parsed_tokens']} (tok={t_tok})")
    return record


def main():
    db_path = "statllm.db"
    db = Database(db_path)
    probes_list = list(PROBES.keys())

    print("=" * 80)
    print("StatLLM Meta Muse-Spark 1.3 & Google 最新旗舰全随机提示词采样入库")
    print(f"目标模型: {[m['name'] for m in NEW_MODELS]}")
    print(f"标准探针: {probes_list}")
    print("=" * 80)

    # 1. Register new models
    for m in NEW_MODELS:
        db.add_model(
            name=m["name"],
            display_name=m["display_name"],
            provider=m["provider"],
            color=m["color"]
        )
    print("已注册模型元数据至 SQLite。")

    # 2. Build tasks
    tasks = []
    for model_cfg in NEW_MODELS:
        runs = model_cfg.get("runs_per_probe", 12)
        for p_id in probes_list:
            for idx in range(runs):
                tasks.append((model_cfg, p_id, idx, db_path))

    random.shuffle(tasks)
    print(f"总计规划执行 {len(tasks)} 次真实 API 采样...")

    start_time = time.time()
    successful = 0
    total_tokens = 0

    with ThreadPoolExecutor(max_workers=5) as executor:
        future_map = {executor.submit(sample_task, *t): t for t in tasks}
        for future in as_completed(future_map):
            try:
                rec = future.result()
                if rec:
                    successful += 1
                    total_tokens += rec["total_tokens"]
            except Exception as e:
                task_t = future_map[future]
                print(f"Exception on {task_t[0]['name']}: {e}")

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"采样完成！成功入库: {successful}/{len(tasks)} ({successful/len(tasks)*100:.1f}%)")
    print(f"总耗时: {elapsed:.1f} 秒 | 消耗 Token: {total_tokens}")
    print("=" * 80)

    # Checkpoint SQLite WAL
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
    print("已成功执行 PRAGMA wal_checkpoint(TRUNCATE)。")

    # Refit 2D PCA
    print("正在重新拟合 2D PCA 聚类投影器...")
    projector = ClusterProjector(db)
    projector.fit()
    print("PCA 聚类投影器拟合成功！")

    # Print summary
    stats = db.get_stats()
    print("\n[最新数据库统计概览]:")
    print(f"  • 总样本数: {stats['total_samples']}")
    print(f"  • 扰动类型分布: {stats['perturbation_breakdown']}")
    print(f"  • 各模型样本量:")
    for m in stats["model_breakdown"]:
        print(f"    - {m['model_name']}: {m['count']} 条")


if __name__ == "__main__":
    main()
