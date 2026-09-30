"""
StatLLM Command Line Interface.
"""

import argparse
import sys
import json
import uvicorn
from statllm.database import Database
from statllm.engine import LikelihoodEvaluator
from statllm.seed_data import seed_database
from statllm.probes import list_probes


def main():
    parser = argparse.ArgumentParser(
        prog="statllm",
        description="StatLLM: Statistical LLM Fingerprinting & Model Attribution Framework"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: seed
    seed_p = subparsers.add_parser("seed", help="Initialize and seed database with benchmark distributions")
    seed_p.add_argument("--db", default="statllm.db", help="Database file path (default: statllm.db)")
    seed_p.add_argument("--samples", type=int, default=150, help="Samples per probe per model (default: 150)")

    # Command: stats
    stats_p = subparsers.add_parser("stats", help="Show database overview and sample counts")
    stats_p.add_argument("--db", default="statllm.db", help="Database file path")

    # Command: probes
    subparsers.add_parser("probes", help="List all standardized probe questions")

    # Command: serve
    serve_p = subparsers.add_parser("serve", help="Start the StatLLM Web UI server")
    serve_p.add_argument("--host", default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    serve_p.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    serve_p.add_argument("--db", default="statllm.db", help="Database file path")

    # Command: eval
    eval_p = subparsers.add_parser("eval", help="Evaluate a sample from CLI")
    eval_p.add_argument("--db", default="statllm.db", help="Database file path")
    eval_p.add_argument("--probe", required=True, help="Probe ID (e.g. q1_int, q2_color)")
    eval_p.add_argument("text", help="Raw model response text")

    args = parser.parse_args()

    if args.command == "seed":
        db = Database(args.db)
        seed_database(db, samples_per_probe=args.samples)
        print("Database successfully seeded.")

    elif args.command == "stats":
        db = Database(args.db)
        stats = db.get_stats()
        print(json.dumps(stats, indent=2, ensure_ascii=False))

    elif args.command == "probes":
        for p in list_probes():
            print(f"[{p['id']}] {p['title']}")
            print(f"  Prompt: {p['prompt']}")
            print(f"  Category: {p['category']} | Vocab size: {p['vocab_size']}\n")

    elif args.command == "serve":
        from web.app import create_app
        app = create_app(db_path=args.db)
        print(f"Starting StatLLM server at http://{args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port)

    elif args.command == "eval":
        db = Database(args.db)
        evaluator = LikelihoodEvaluator(db)
        res = evaluator.evaluate([{"probe_id": args.probe, "raw_text": args.text}])
        print(f"\n--- StatLLM Evaluation Result ---")
        print(f"Top Model: {res['top_model']} (P = {res['top_probability'] * 100:.1f}%)")
        print(f"95% CI: {res['confidence_intervals'].get(res['top_model'])}")
        print(f"Posteriors: {json.dumps(res['posteriors'], indent=2, ensure_ascii=False)}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
