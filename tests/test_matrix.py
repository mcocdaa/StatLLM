import pytest
import os
import tempfile
from statllm.database import Database
from statllm.seed_data import seed_database
from statllm.matrix import get_probe_matrix
from statllm.engine import LikelihoodEvaluator
from web.app import create_app, SingleProbeTestRequest


@pytest.fixture
def temp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = Database(path)
    seed_database(db, samples_per_probe=20)
    yield db
    if os.path.exists(path):
        os.remove(path)


def test_get_probe_matrix(temp_db):
    from statllm.matrix import rank_probes_by_discrimination
    evaluator = LikelihoodEvaluator(temp_db, positional_lambda=0.25)
    res = get_probe_matrix(temp_db, evaluator, "arr_int5")

    assert res["probe_id"] == "arr_int5"
    assert "matrix" in res
    assert "overall_avg_margin" in res
    assert "mdi" in res
    assert "tier" in res
    assert 0.0 <= res["mdi"] <= 100.0
    assert res["tier"] in ["S", "A", "B", "C", "D"]
    assert "models" in res
    assert len(res["models"]) > 0

    # Test ranking
    ranking = rank_probes_by_discrimination(temp_db, evaluator)
    assert len(ranking) > 0
    assert "mdi" in ranking[0]
    assert ranking[0]["rank"] == 1
    # Check that ranking is strictly sorted by MDI descending
    mdis = [r["mdi"] for r in ranking]
    assert mdis == sorted(mdis, reverse=True)


def test_api_matrix_and_single_probe(temp_db):
    app = create_app(db_path=temp_db.db_path)

    # 1. Test GET /api/probes/{probe_id}/matrix endpoint function
    matrix_func = None
    single_func = None
    for route in app.routes:
        if getattr(route, "path", None) == "/api/probes/{probe_id}/matrix":
            matrix_func = route.endpoint
        elif getattr(route, "path", None) == "/api/evaluate/single-probe":
            single_func = route.endpoint

    assert matrix_func is not None
    assert single_func is not None

    data = matrix_func("arr_int5")
    assert data["probe_id"] == "arr_int5"
    assert "overall_avg_margin" in data
    assert "matrix" in data

    # 2. Test POST /api/evaluate/single-probe
    payload = SingleProbeTestRequest(
        probe_id="arr_int5",
        raw_text="[42, 17, 89, 5, 63]",
        model_A="Grok-4.7",
        model_B="GPT-5.6-Luna"
    )
    data2 = single_func(payload)
    assert data2["is_valid"] is True
    assert "pairwise_diagnostic" in data2
    assert data2["pairwise_diagnostic"]["model_A"] == "Grok-4.7"
    assert "instant_delta" in data2["pairwise_diagnostic"]
