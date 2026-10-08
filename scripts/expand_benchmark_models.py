"""
StatLLM Concurrent Model Fingerprint Expansion Script.
Assigns 1 worker per model (total 6 workers):
- 3 models via OpenRouter (Claude-Sonnet-5.5, Llama-3.3-70B, Gemini-2.5-Flash)
- 3 models via OpenCode-Go (Qwen-3.8-Max, MiniMax-M3, GPT-6-Luna)

Collects 4 high-quality empirical samples per probe (5 probes x 4 runs = 20 samples/model).
Safe thread-isolated querying with batch SQLite commit.
"""

import os
import sys
import json
import time
import re
import urllib.request
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from statllm.database import Database
from statllm.probes import PROBES
from statllm.perturbations import estimate_tokens

OPENROUTER_KEY = os.environ.get("OPENROUTER_KEY", "")

MODELS_CONFIG = [
    {
        "name": "Qwen-3.8-Max",
        "display_name": "Alibaba Qwen 3.8 Max",
        "provider": "Alibaba Cloud",
        "color": "#ea580c",
        "channel": "opencode",
        "model_id": "qwen3.8-max"
    },
    {
        "name": "MiniMax-M3",
        "display_name": "MiniMax M3",
        "provider": "MiniMax",
        "color": "#ec4899",
        "channel": "opencode",
        "model_id": "minimax-m3"
    },
    {
        "name": "GPT-6-Luna",
        "display_name": "OpenAI GPT-6 Luna",
        "provider": "OpenAI",
        "color": "#059669",
        "channel": "opencode",
        "model_id": "gpt-6-luna"
    },
    {
        "name": "Claude-Sonnet-5.5",
        "display_name": "Anthropic Claude Sonnet 5.5",
        "provider": "Anthropic",
        "color": "#7c3aed",
        "channel": "openrouter",
        "model_id": "anthropic/claude-sonnet-5.5"
    },
    {
        "name": "Llama-3.3-70B",
        "display_name": "Meta Llama 3.3 70B",
        "provider": "Meta",
        "color": "#2563eb",
        "channel": "openrouter",
        "model_id": "meta-llama/llama-3.3-70b-instruct"
    },
    {
        "name": "Gemini-2.5-Flash",
        "display_name": "Google Gemini 2.5 Flash",
        "provider": "Google",
        "color": "#d97706",
        "channel": "openrouter",
        "model_id": "google/gemini-2.5-flash"
    }
]


def query_opencode(model_id: str, prompt: str) -> str:
    for attempt in range(2):
        try:
            cmd = ["opencode", "run", prompt, "-m", f"opencode-go/{model_id}"]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=35)
            out = re.sub(r"^>.*?\n+", "", res.stdout).strip()
            if out:
                return out
        except Exception:
            time.sleep(1)
    return ""


def query_openrouter(model_id: str, prompt: str) -> str:
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
                    "max_tokens": 120,
                    "temperature": 0.85
                }).encode()
            )
            with urllib.request.urlopen(req, timeout=25) as r:
                data = json.loads(r.read().decode())
                choice = data["choices"][0]
                content = choice["message"].get("content") or ""
                if content:
                    return content.strip()
        except Exception:
            time.sleep(1.5)
    return ""


def collect_for_model(model_cfg: Dict[str, Any], samples_per_probe: int = 4) -> List[Dict[str, Any]]:
    m_name = model_cfg["name"]
    m_chan = model_cfg["channel"]
    m_id = model_cfg["model_id"]
    results = []

    print(f"[{m_name}] Worker started...")
    for pid, probe in PROBES.items():
        for run_idx in range(1, samples_per_probe + 1):
            if m_chan == "opencode":
                raw_text = query_opencode(m_id, probe.prompt)
            else:
                raw_text = query_openrouter(m_id, probe.prompt)

            if not raw_text:
                print(f"[{m_name}] {pid}#{run_idx} FAILED (empty response)")
                continue

            parsed = probe.parse(raw_text)
            p_tok = estimate_tokens(probe.prompt)
            c_tok = estimate_tokens(raw_text)
            t_tok = p_tok + c_tok
            results.append({
                "model_name": m_name,
                "probe_id": pid,
                "raw_text": raw_text,
                "parsed_tokens": parsed["parsed_tokens"],
                "traits": parsed.get("traits", {}),
                "is_valid": parsed["is_valid"],
                "strictly_complied": parsed["strictly_complied"],
                "source_type": "official",
                "weight": 1.0,
                "temperature": 0.85,
                "prompt_tokens": p_tok,
                "completion_tokens": c_tok,
                "total_tokens": t_tok
            })
            print(f"[{m_name}] {pid}#{run_idx} OK (valid={parsed['is_valid']}, tok={t_tok})")
            time.sleep(0.4)

    print(f"[{m_name}] Finished! Collected {len(results)} samples.")
    return results


def main():
    db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "statllm.db")
    db = Database(db_path)

    # 1. Register model metadata first
    for m in MODELS_CONFIG:
        db.add_model(
            name=m["name"],
            display_name=m["display_name"],
            provider=m["provider"],
            color=m["color"]
        )
    print(f"Registered {len(MODELS_CONFIG)} models in database.")

    # 2. Parallel collection (1 worker per model)
    samples_per_probe = 8
    total_expected = len(MODELS_CONFIG) * len(PROBES) * samples_per_probe
    print(f"Beginning concurrent collection: {len(MODELS_CONFIG)} models x {len(PROBES)} probes x {samples_per_probe} runs = {total_expected} total calls.")

    all_records = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=len(MODELS_CONFIG)) as executor:
        futures = [executor.submit(collect_for_model, m, samples_per_probe) for m in MODELS_CONFIG]
        for f in futures:
            all_records.extend(f.result())

    duration = time.time() - t0
    print(f"\nAll workers completed in {duration:.1f}s. Total collected: {len(all_records)} / {total_expected} samples.")

    # 3. Batch insert into database
    for rec in all_records:
        db.add_sample(
            model_name=rec["model_name"],
            probe_id=rec["probe_id"],
            raw_text=rec["raw_text"],
            parsed_tokens=rec["parsed_tokens"],
            traits=rec["traits"],
            is_valid=rec["is_valid"],
            strictly_complied=rec["strictly_complied"],
            source_type=rec["source_type"],
            weight=rec["weight"],
            prompt_tokens=rec["prompt_tokens"],
            completion_tokens=rec["completion_tokens"],
            total_tokens=rec["total_tokens"],
            temperature=rec["temperature"]
        )

    # Save backup json
    out_json = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "expanded_models_samples.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    # Checkpoint WAL so docker container sees changes immediately
    import sqlite3
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")

    print(f"Successfully inserted {len(all_records)} samples into statllm.db, checkpointed WAL, and saved to {out_json}")


if __name__ == "__main__":
    main()
