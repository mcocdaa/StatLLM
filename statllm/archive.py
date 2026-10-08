"""
StatLLM Archive Manager.

Handles exporting and importing standardized .zip archive bundles for easy
migration, backup, and sharing of empirical baseline datasets.
"""

import io
import json
import zipfile
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional

from statllm.database import Database


ARCHIVE_FORMAT_VERSION = "1.0.0"


def export_archive_bundle(db: Database) -> bytes:
    """
    Exports the current database state into a self-contained ZIP archive.
    
    Contains:
      - manifest.json: Metadata, checksums, and dataset summary.
      - models.json: Candidate model configuration and visual palette.
      - samples.jsonl: Line-by-line raw & parsed empirical records.
      - statllm.db: Raw SQLite binary snapshot for zero-config instant restore.
    """
    buf = io.BytesIO()
    models = db.list_models()
    samples = db.get_all_samples()
    stats = db.get_stats()
    token_usage = db.get_token_usage_stats()

    # Serialize models and samples
    models_json_str = json.dumps(models, indent=2, ensure_ascii=False)
    
    samples_jsonl_lines = []
    for s in samples:
        samples_jsonl_lines.append(json.dumps(s, ensure_ascii=False))
    samples_jsonl_str = "\n".join(samples_jsonl_lines) + ("\n" if samples_jsonl_lines else "")

    # Read current SQLite DB binary if available
    db_bytes = b""
    try:
        with open(db.db_path, "rb") as f:
            db_bytes = f.read()
    except Exception:
        db_bytes = b""

    # Compute checksums
    models_hash = hashlib.sha256(models_json_str.encode("utf-8")).hexdigest()
    samples_hash = hashlib.sha256(samples_jsonl_str.encode("utf-8")).hexdigest()

    manifest = {
        "format": "StatLLM-Archive-Bundle",
        "version": ARCHIVE_FORMAT_VERSION,
        "exported_at": datetime.utcnow().isoformat() + "Z",
        "dataset_summary": {
            "total_samples": len(samples),
            "total_models": len(models),
            "official_samples": stats.get("official_samples", 0),
            "user_samples": stats.get("user_samples", 0),
            "grand_total_tokens": token_usage.get("overall", {}).get("grand_total_tokens", 0),
            "models": [m["name"] for m in models],
            "probe_breakdown": stats.get("probe_breakdown", [])
        },
        "checksums": {
            "models_json_sha256": models_hash,
            "samples_jsonl_sha256": samples_hash
        }
    }
    manifest_json_str = json.dumps(manifest, indent=2, ensure_ascii=False)

    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", manifest_json_str)
        zf.writestr("models.json", models_json_str)
        zf.writestr("samples.jsonl", samples_jsonl_str)
        if db_bytes:
            zf.writestr("statllm.db", db_bytes)

    buf.seek(0)
    return buf.getvalue()


def import_archive_bundle(db: Database, zip_bytes: bytes, mode: str = "merge") -> Dict[str, Any]:
    """
    Imports a ZIP archive bundle into the database.
    
    Modes:
      - 'merge': Appends new samples without deleting existing ones (skipping exact duplicates).
      - 'replace': Clears existing samples and performs a full restore.
    """
    if mode not in ("merge", "replace"):
        raise ValueError("Mode must be either 'merge' or 'replace'.")

    buf = io.BytesIO(zip_bytes)
    if not zipfile.is_zipfile(buf):
        raise ValueError("Uploaded file is not a valid ZIP archive.")

    with zipfile.ZipFile(buf, "r") as zf:
        namelist = zf.namelist()
        if "manifest.json" not in namelist:
            raise ValueError("Invalid archive: missing manifest.json.")

        manifest_data = json.loads(zf.read("manifest.json").decode("utf-8"))
        
        # Models
        imported_models = []
        if "models.json" in namelist:
            imported_models = json.loads(zf.read("models.json").decode("utf-8"))
            for m in imported_models:
                db.add_model(
                    name=m["name"],
                    display_name=m.get("display_name", m["name"]),
                    provider=m.get("provider", ""),
                    color=m.get("color", "#38bdf8")
                )

        # Clear existing data if replace mode
        if mode == "replace":
            db.clear_all_data()

        # Samples
        imported_count = 0
        skipped_count = 0

        # Existing signatures for deduplication if merging
        existing_signatures = set()
        if mode == "merge":
            for s in db.get_all_samples():
                # signature: (model_name, probe_id, raw_text)
                existing_signatures.add((s["model_name"], s["probe_id"], s["raw_text"]))

        if "samples.jsonl" in namelist:
            lines = zf.read("samples.jsonl").decode("utf-8").splitlines()
            samples_to_insert = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    s = json.loads(line)
                except Exception:
                    continue

                sig = (s.get("model_name"), s.get("probe_id"), s.get("raw_text"))
                if mode == "merge" and sig in existing_signatures:
                    skipped_count += 1
                    continue

                parsed_tokens = []
                try:
                    parsed_tokens = json.loads(s.get("parsed_value", "[]"))
                except Exception:
                    parsed_tokens = []

                samples_to_insert.append({
                    "model_name": s.get("model_name"),
                    "probe_id": s.get("probe_id"),
                    "raw_text": s.get("raw_text", ""),
                    "parsed_tokens": parsed_tokens,
                    "is_valid": bool(s.get("is_valid", 1)),
                    "strictly_complied": bool(s.get("strictly_complied", 1)),
                    "source_type": s.get("source_type", "official"),
                    "weight": float(s.get("weight", 1.0))
                })
                existing_signatures.add(sig)

            if samples_to_insert:
                db.add_samples_batch(samples_to_insert)
                imported_count = len(samples_to_insert)

        # Synchronize token frequency indices
        db.rebuild_token_counts()

        final_stats = db.get_stats()
        return {
            "status": "success",
            "mode": mode,
            "archive_version": manifest_data.get("version", "1.0.0"),
            "imported_samples": imported_count,
            "skipped_duplicates": skipped_count,
            "total_samples": final_stats.get("total_samples", 0),
            "official_samples": final_stats.get("official_samples", 0),
            "user_samples": final_stats.get("user_samples", 0)
        }
