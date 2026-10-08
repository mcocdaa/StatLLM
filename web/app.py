"""
StatLLM Web API Application.
"""

import os
from datetime import datetime
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from statllm.database import Database
from statllm.engine import LikelihoodEvaluator
from statllm.cluster import ClusterProjector
from statllm.probes import list_probes, get_probe
from statllm.archive import export_archive_bundle, import_archive_bundle



class SubmissionItem(BaseModel):
    probe_id: str
    raw_text: str


class EvaluateRequest(BaseModel):
    submissions: List[SubmissionItem] = Field(..., min_items=1)
    consent_to_collect: bool = False
    claimed_model: Optional[str] = None
    positional_lambda: Optional[float] = Field(0.5, ge=0.0, le=1.0)


class ContributeRequest(BaseModel):
    model_name: str
    probe_id: str
    raw_text: str
    weight: float = 0.2


def create_app(db_path: str = "statllm.db") -> FastAPI:
    app = FastAPI(
        title="StatLLM API",
        description="Statistical Large Language Model Fingerprinting & Attribution Engine",
        version="0.1.0"
    )

    db = Database(db_path)
    evaluator = LikelihoodEvaluator(db, positional_lambda=0.5)
    cluster_projector = ClusterProjector(db)
    cluster_projector.fit()

    static_dir = os.path.join(os.path.dirname(__file__), "static")
    if os.path.exists(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    def serve_index():
        index_path = os.path.join(static_dir, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        return JSONResponse({"message": "StatLLM API is running. Static frontend not found."})

    @app.get("/api/probes")
    def get_probes():
        return list_probes()

    @app.get("/api/models")
    def get_models():
        return db.list_models()

    @app.get("/api/stats")
    def get_stats():
        return db.get_stats()

    @app.get("/api/token-usage")
    def get_token_usage():
        return db.get_token_usage_stats()

    @app.get("/api/cluster")
    def get_cluster():
        return cluster_projector.get_cluster_data()

    @app.post("/api/evaluate")
    def evaluate(req: EvaluateRequest):
        items = [{"probe_id": s.probe_id, "raw_text": s.raw_text} for s in req.submissions]
        lam = req.positional_lambda if req.positional_lambda is not None else 0.5
        try:
            eval_res = evaluator.evaluate(items, positional_lambda=lam, n_boot=800)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Project user submission to 2D cluster map
        user_point = cluster_projector.project_user_submission(eval_res["parsed_submissions"])
        cluster_data = cluster_projector.get_cluster_data(user_point=user_point)

        # Optional crowdsource collection with user consent
        if req.consent_to_collect:
            assigned_label = req.claimed_model if req.claimed_model else eval_res["top_model"]
            for parsed_rec in eval_res["parsed_submissions"]:
                if parsed_rec["is_valid"]:
                    db.add_sample(
                        model_name=assigned_label,
                        probe_id=parsed_rec["probe_id"],
                        raw_text=parsed_rec["raw_text"],
                        parsed_tokens=parsed_rec["parsed_tokens"],
                        traits=parsed_rec.get("traits", {}),
                        is_valid=parsed_rec["is_valid"],
                        strictly_complied=parsed_rec["strictly_complied"],
                        source_type="user",
                        weight=0.2
                    )

        return {
            "evaluation": eval_res,
            "cluster_data": cluster_data
        }

    @app.post("/api/contribute")
    def contribute(req: ContributeRequest):
        probe = get_probe(req.probe_id)
        if not probe:
            raise HTTPException(status_code=400, detail="Invalid probe ID")
        
        parse_res = probe.parse(req.raw_text)
        if not parse_res["is_valid"]:
            raise HTTPException(status_code=400, detail="Response could not be parsed into a valid array")

        sample_id = db.add_sample(
            model_name=req.model_name,
            probe_id=req.probe_id,
            raw_text=req.raw_text,
            parsed_tokens=parse_res["parsed_tokens"],
            traits=parse_res.get("traits", {}),
            is_valid=parse_res["is_valid"],
            strictly_complied=parse_res["strictly_complied"],
            source_type="user",
            weight=req.weight
        )

        return {
            "status": "success",
            "sample_id": sample_id,
            "parsed_tokens": parse_res["parsed_tokens"],
            "weight": req.weight,
            "updated_stats": db.get_stats()
        }

    @app.get("/api/archive/export")
    def export_archive():
        try:
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            zip_bytes = export_archive_bundle(db)
            filename = f"statllm_archive_{ts}.zip"
            return Response(
                content=zip_bytes,
                media_type="application/zip",
                headers={
                    "Content-Disposition": f"attachment; filename={filename}",
                    "Access-Control-Expose-Headers": "Content-Disposition"
                }
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to export archive: {str(e)}")

    @app.post("/api/archive/import")
    async def import_archive(request: Request, mode: str = "merge"):
        try:
            content = await request.body()
            if not content:
                raise HTTPException(status_code=400, detail="Uploaded file is empty.")
            res = import_archive_bundle(db, content, mode=mode)
            # Re-fit cluster projector with new data
            cluster_projector.fit()
            return res
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Import failed: {str(e)}")

    return app


# Default app for uvicorn
app = create_app()
