"""
StatLLM Comprehensive Flagship Model Benchmark Expansion Script.

Expands benchmark across every major foundation AI frontier company's BEST flagship model:
1. Zhipu AI (智谱) -> GLM-5.3
2. DeepSeek (深度求索) -> DeepSeek-V4.1-Flash
3. Alibaba Cloud (阿里千问) -> Qwen-3.8-Max
4. MiniMax (稀宇科技) -> MiniMax-M3
5. Moonshot AI (月之暗面) -> Kimi-K3
6. OpenAI -> GPT-6-Astra
7. Anthropic -> Claude-Sonnet-5.5
8. Google -> Gemini-2.5-Flash
9. Meta -> Llama-3.3-70B
10. xAI -> Grok-4.7

Every sample query is generated with a randomized prompt perturbation + standardized probe question.
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

# Registry of each company's BEST flagship model
FRONTIER_FLAGSHIPS = [
    {
        "name": "GLM-5.3",
        "display_name": "Zhipu GLM-5.3",
        "provider": "Zhipu AI",
        "color": "#6366f1",  # Modern Indigo
        "channel": "opencode",
        "model_id": "glm-5.3",
        "fallback_channel": "openrouter",
        "fallback_id": "z-ai/glm-5.3-flash",
        "max_tokens": 250,
        "target_runs_per_probe": 15  # 5 x 15 = 75 samples for brand new model
    },
    {
        "name": "DeepSeek-V4.1-Flash",
        "display_name": "DeepSeek V4.1 Flash",
        "provider": "DeepSeek",
        "color": "#06b6d4",
        "channel": "openrouter",
        "model_id": "deepseek/deepseek-v4.1-flash",
        "max_tokens": 260,
        "target_runs_per_probe": 2   # 5 x 2 = 10 samples
    },
    {
        "name": "Qwen-3.8-Max",
        "display_name": "Alibaba Qwen 3.8 Max",
        "provider": "Alibaba Cloud",
        "color": "#ea580c",
        "channel": "opencode",
        "model_id": "qwen3.8-max",
        "fallback_channel": "openrouter",
        "fallback_id": "qwen/qwen3.8-max-prime",
        "max_tokens": 200,
        "target_runs_per_probe": 2   # 5 x 2 = 10 samples
    },
    {
        "name": "MiniMax-M3",
        "display_name": "MiniMax M3",
        "provider": "MiniMax",
        "color": "#ec4899",
        "channel": "openrouter",
        "model_id": "minimax/minimax-m3",
        "max_tokens": 300,
        "target_runs_per_probe": 2   # 5 x 2 = 10 samples
    },
    {
        "name": "Kimi-K3",
        "display_name": "MoonshotAI Kimi K3",
        "provider": "MoonshotAI",
        "color": "#8b5cf6",
        "channel": "openrouter",
        "model_id": "moonshotai/kimi-k3",
        "max_tokens": 300,
        "target_runs_per_probe": 2   # 5 x 2 = 10 samples
    },
    {
        "name": "GPT-6-Astra",
        "display_name": "OpenAI GPT-6 Astra",
        "provider": "OpenAI",
        "color": "#10b981",
        "channel": "openrouter",
        "model_id": "openai/gpt-6-astra",
        "max_tokens": 200,
        "target_runs_per_probe": 2   # 5 x 2 = 10 samples
    },
    {
        "name": "Claude-Sonnet-5.5",
        "display_name": "Anthropic Claude Sonnet 5.5",
        "provider": "Anthropic",
        "color": "#7c3aed",
        "channel": "openrouter",
        "model_id": "anthropic/claude-sonnet-5.5",
        "max_tokens": 200,
        "target_runs_per_probe": 2   # 5 x 2 = 10 samples
    },
    {
        "name": "Gemini-2.5-Flash",
        "display_name": "Google Gemini 2.5 Flash",
        "provider": "Google",
        "color": "#d97706",
        "channel": "openrouter",
        "model_id": "google/gemini-2.5-flash",
        "max_tokens": 200,
        "target_runs_per_probe": 2   # 5 x 2 = 10 samples
    },
    {
        "name": "Llama-3.3-70B",
        "display_name": "Meta Llama 3.3 70B",
        "provider": "Meta",
        "color": "#2563eb",
        "channel": "openrouter",
        "model_id": "meta-llama/llama-3.3-70b-instruct",
        "max_tokens": 200,
        "target_runs_per_probe": 3   # 5 x 3 = 15 samples
    },
    {
        "name": "Grok-4.7",
        "display_name": "xAI Grok 4.7",
        "provider": "xAI",
        "color": "#38bdf8",
        "channel": "openrouter",
        "model_id": "x-ai/grok-4.7",
        "max_tokens": 200,
        "target_runs_per_probe": 2   # 5 x 2 = 10 samples
    }
]

db_lock = threading.Lock()


def query_opencode(model_id: str, prompt: str, temperature: float = 0.8) -> str:
    for attempt in range(2):
        try:
            cmd = ["opencode", "run", prompt, "-m", f"opencode-go/{model_id}"]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=35)
            out = re.sub(r"^>.*?\n+", "", res.stdout).strip()
            if out:
                return out
        except Exception:
            time.sleep(1.0)
    return ""


def query_openrouter(model_id: str, prompt: str, max_tokens: int = 200, temperature: float = 0.8) -> str:
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


def sample_single(
    model_cfg: Dict[str, Any],
    probe_id: str,
    run_idx: int,
    db_path: str = "statllm.db"
) -> Optional[Dict[str, Any]]:
    probe = PROBES.get(probe_id)
    if not probe:
        return None

    # Every single test uses a randomized prompt perturbation + question
    full_prompt, actual_pert_type, pert_prefix = apply_perturbation(probe.prompt)
    m_name = model_cfg["name"]
    channel = model_cfg["channel"]
    mid = model_cfg["model_id"]
    max_tok = model_cfg.get("max_tokens", 200)

    # Randomized temperature between 0.7 and 0.95
    temp = round(random.choice([0.7, 0.8, 0.85, 0.9, 0.95]), 2)

    raw_text = ""
    if channel == "opencode":
        raw_text = query_opencode(mid, full_prompt, temperature=temp)
        if not raw_text and model_cfg.get("fallback_id"):
            raw_text = query_openrouter(model_cfg["fallback_id"], full_prompt, max_tokens=max_tok, temperature=temp)
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
    print("StatLLM 全球顶尖 AI 厂商旗舰大模型（含智谱 GLM）全量全随机提示词采样扩充")
    print(f"厂商数: {len(FRONTIER_FLAGSHIPS)}")
    print(f"标准探针: {probes_list}")
    print("=" * 80)

    # 1. Ensure all models (including GLM-5.3) are registered in the models table
    for m in FRONTIER_FLAGSHIPS:
        db.add_model(
            name=m["name"],
            display_name=m["display_name"],
            provider=m["provider"],
            color=m["color"]
        )
    print(f"已在底库注册 {len(FRONTIER_FLAGSHIPS)} 家厂商旗舰模型元数据。")

    # 2. Build task list
    tasks = []
    for model_cfg in FRONTIER_FLAGSHIPS:
        runs = model_cfg.get("target_runs_per_probe", 2)
        for p_id in probes_list:
            for idx in range(runs):
                tasks.append((model_cfg, p_id, idx, db_path))

    # Shuffle to interleave tasks and maximize API parallelism across providers
    random.shuffle(tasks)
    print(f"总计规划执行 {len(tasks)} 次高并发真实 API 采样 (含 GLM-5.3 全量基准 + 各大厂商旗舰数据补齐)...")

    start_time = time.time()
    successful = 0
    total_tokens = 0

    with ThreadPoolExecutor(max_workers=5) as executor:
        future_map = {executor.submit(sample_single, *t): t for t in tasks}
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
    print(f"扩充完成！成功采样入库: {successful}/{len(tasks)} ({successful/len(tasks)*100:.1f}%)")
    print(f"总耗时: {elapsed:.1f} 秒 | 消耗 Token: {total_tokens}")
    print("=" * 80)

    # Checkpoint SQLite WAL
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
    print("已成功执行 PRAGMA wal_checkpoint(TRUNCATE)。")

    # Refit 2D PCA cluster projector
    print("正在基于最新全量数据重新拟合 2D PCA 聚类投影器...")
    projector = ClusterProjector(db)
    projector.fit()
    print("PCA 聚类投影器拟合成功！")

    # Print summary
    stats = db.get_stats()
    print("\n[最新数据库统计概览]:")
    print(f"  • 总样本数: {stats['total_samples']}")
    print(f"  • 扰动类型分布: {stats['perturbation_breakdown']}")
    print(f"  • 各厂商旗舰模型样本量:")
    for m in stats["model_breakdown"]:
        print(f"    - {m['model_name']}: {m['count']} 条")


if __name__ == "__main__":
    main()
