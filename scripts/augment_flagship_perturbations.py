"""
StatLLM Flagship Models Prompt Perturbation Data Augmentation.

Augments baseline benchmark with authentic empirical samples under prompt perturbations:
- Chit-chat prefix (前置闲聊)
- Persona prefix (系统人设/角色设定)
- Context noise (长上下文/代码段背景噪音)

Targets flagship models across top frontier AI labs:
- Qwen-3.8-Max (Alibaba Cloud) via OpenCode
- MiniMax-M3 (MiniMax) via OpenRouter
- Kimi-K3 (MoonshotAI) via OpenRouter
- GPT-6-Astra (OpenAI) via OpenRouter
- Claude-Sonnet-5.5 (Anthropic) via OpenRouter
- Gemini-2.5-Flash (Google) via OpenRouter
- DeepSeek-V4.1-Flash (DeepSeek) via OpenRouter
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from statllm.database import Database
from statllm.probes import PROBES
from statllm.perturbations import apply_perturbation, estimate_tokens
from statllm.cluster import ClusterProjector

OPENROUTER_KEY = os.environ.get("OPENROUTER_KEY", "")

TARGET_FLAGSHIPS = [
    {
        "name": "Qwen-3.8-Max",
        "channel": "opencode",
        "model_id": "qwen3.8-max",
        "max_tokens": 160
    },
    {
        "name": "MiniMax-M3",
        "channel": "openrouter",
        "model_id": "minimax/minimax-m3",
        "max_tokens": 300
    },
    {
        "name": "Kimi-K3",
        "channel": "openrouter",
        "model_id": "moonshotai/kimi-k3",
        "max_tokens": 300
    },
    {
        "name": "GPT-6-Astra",
        "channel": "openrouter",
        "model_id": "openai/gpt-6-astra",
        "max_tokens": 200
    },
    {
        "name": "Claude-Sonnet-5.5",
        "channel": "openrouter",
        "model_id": "anthropic/claude-sonnet-5.5",
        "max_tokens": 200
    },
    {
        "name": "Gemini-2.5-Flash",
        "channel": "openrouter",
        "model_id": "google/gemini-2.5-flash",
        "max_tokens": 200
    },
    {
        "name": "DeepSeek-V4.1-Flash",
        "channel": "openrouter",
        "model_id": "deepseek/deepseek-v4.1-flash",
        "max_tokens": 260
    }
]

db_lock = threading.Lock()


def query_opencode(model_id: str, prompt: str, temperature: float = 0.8) -> str:
    for attempt in range(3):
        try:
            cmd = ["opencode", "run", prompt, "-m", f"opencode-go/{model_id}"]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=35)
            out = re.sub(r"^>.*?\n+", "", res.stdout).strip()
            if out:
                return out
        except Exception as e:
            time.sleep(1.0)
    return ""


def query_openrouter(model_id: str, prompt: str, max_tokens: int = 200, temperature: float = 0.8) -> str:
    for attempt in range(4):
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
        except Exception as e:
            time.sleep(1.5 + attempt)
    return ""


def collect_one(model_cfg: Dict[str, Any], probe_id: str, pert_type: str, run_idx: int, db_path: str = "statllm.db") -> Optional[Dict[str, Any]]:
    probe = PROBES.get(probe_id)
    if not probe:
        return None

    full_prompt, actual_pert_type, pert_prefix = apply_perturbation(probe.prompt, perturbation_type=pert_type)
    m_name = model_cfg["name"]
    channel = model_cfg["channel"]
    mid = model_cfg["model_id"]
    max_tok = model_cfg.get("max_tokens", 200)

    # Varied temperature schedule
    temp = 0.7 if run_idx % 3 == 0 else (0.85 if run_idx % 3 == 1 else 1.0)

    if channel == "opencode":
        raw_text = query_opencode(mid, full_prompt, temperature=temp)
    else:
        raw_text = query_openrouter(mid, full_prompt, max_tokens=max_tok, temperature=temp)

    if not raw_text:
        print(f"  [X] [{m_name}] {probe_id} (run {run_idx}, {actual_pert_type}) -> EMPTY / TIMEOUT")
        return None

    parsed = probe.parse(raw_text)
    if not parsed["is_valid"]:
        print(f"  [!] [{m_name}] {probe_id} (run {run_idx}, {actual_pert_type}) -> INVALID: {raw_text[:60]}")
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

    # Thread-safe database insertion
    with db_lock:
        db = Database(db_path)
        db.add_sample(**record)

    print(f"  [√] [{m_name}] {probe_id} #{run_idx+1} ({actual_pert_type}): {parsed['parsed_tokens']} (tok={t_tok})")
    return record


def main():
    db_path = "statllm.db"
    db = Database(db_path)
    probes_list = list(PROBES.keys())  # 5 probes: arr_int5, arr_color5, arr_rps5, arr_letter5, arr_perm5
    pert_cycle = ["chit_chat", "persona", "context_noise"]

    print("=" * 75)
    print("StatLLM 旗舰大模型多层级扰动提示词数据扩充引擎")
    print(f"目标旗舰模型数: {len(TARGET_FLAGSHIPS)}")
    print(f"候选探针数: {len(probes_list)} ({', '.join(probes_list)})")
    print(f"扰动类型覆盖: {pert_cycle}")
    print("=" * 75)

    # Plan collection tasks:
    # For each flagship model:
    # Collect 3 perturbed samples per probe (1 chit_chat, 1 persona, 1 context_noise)
    # 5 probes x 3 perturbations = 15 new perturbed samples per model
    # Total tasks = 7 models x 15 = 105 empirical calls
    tasks = []
    for model_cfg in TARGET_FLAGSHIPS:
        m_name = model_cfg["name"]
        for p_id in probes_list:
            for idx, p_type in enumerate(pert_cycle):
                tasks.append((model_cfg, p_id, p_type, idx, db_path))

    # Shuffle tasks to interleave across models and prevent burst rate-limits
    random.shuffle(tasks)
    print(f"生成交织采集计划: 总计 {len(tasks)} 次真实 API 调用任务...")

    start_time = time.time()
    successful_count = 0
    total_tokens_consumed = 0

    # Execute with 4 parallel worker threads
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_map = {
            executor.submit(collect_one, *t): t for t in tasks
        }
        for future in as_completed(future_map):
            try:
                rec = future.result()
                if rec:
                    successful_count += 1
                    total_tokens_consumed += rec["total_tokens"]
            except Exception as e:
                task_info = future_map[future]
                print(f"Task exception on {task_info[0]['name']} {task_info[1]}: {e}")

    elapsed = time.time() - start_time
    print("\n" + "=" * 75)
    print(f"数据扩充采集完成！成功率: {successful_count}/{len(tasks)} ({successful_count/len(tasks)*100:.1f}%)")
    print(f"总计耗时: {elapsed:.1f}s | 新增 Token 消耗: {total_tokens_consumed}")
    print("=" * 75)

    # Checkpoint SQLite WAL
    import sqlite3
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
    print("已成功执行 PRAGMA wal_checkpoint(TRUNCATE)，WAL 状态已完全固化至二进制数据库。")

    # Rebuild token counts & refit PCA cluster projector
    print("正在基于最新全量真实数据重新拟合 2D PCA 聚类投影器...")
    projector = ClusterProjector(db)
    projector.fit()
    print("PCA 聚类模型拟合完成！")

    # Output new database statistics
    stats = db.get_stats()
    print("\n[更新后数据库全量统计]:")
    print(f"  - 总样本数: {stats['total_samples']}")
    print(f"  - 扰动类型分布: {stats['perturbation_breakdown']}")
    print(f"  - 分模型样本数:")
    for m in stats["model_breakdown"]:
        print(f"    * {m['model_name']}: {m['count']} 条")


if __name__ == "__main__":
    main()
