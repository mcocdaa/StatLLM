import pytest
import os
import tempfile
from statllm.database import Database
from statllm.engine import LikelihoodEvaluator
from statllm.seed_data import seed_database


@pytest.fixture
def temp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = Database(path)
    seed_database(db, samples_per_probe=80)
    yield db
    if os.path.exists(path):
        os.remove(path)


def test_engine_evaluation(temp_db):
    evaluator = LikelihoodEvaluator(temp_db)

    # Simulate typical GPT-4o inputs: loves 42, loves 蓝色, loves 石头
    submissions = [
        {"probe_id": "q1_int", "raw_text": "1. 42"},
        {"probe_id": "q1_int", "raw_text": "1. 37"},
        {"probe_id": "q2_color", "raw_text": "蓝色"},
        {"probe_id": "q3_rps", "raw_text": "石头"},
    ]

    result = evaluator.evaluate(submissions, n_boot=200)

    # Posteriors sum to 1.0
    posteriors = result["posteriors"]
    assert pytest.approx(sum(posteriors.values()), rel=1e-4) == 1.0

    # Top model should be GPT-4o
    assert result["top_model"] == "GPT-4o"
    assert posteriors["GPT-4o"] > 0.4

    # Confidence interval checks
    ci = result["confidence_intervals"]
    assert "GPT-4o" in ci
    lower, upper = ci["GPT-4o"]
    assert 0.0 <= lower <= upper <= 1.0

    # Diagnostics
    assert result["sample_count"] == 4
    assert result["unique_probes_tested"] == 3


def test_weighted_crowdsourcing(temp_db):
    evaluator = LikelihoodEvaluator(temp_db)

    # Initial stats
    stats_before = temp_db.get_stats()
    initial_user_samples = stats_before["user_samples"]

    # User adds a crowdsourced sample with weight 0.2
    temp_db.add_sample(
        model_name="GPT-4o",
        probe_id="q1_int",
        raw_text="1. 42",
        parsed_value="42",
        source_type="user",
        weight=0.2
    )

    stats_after = temp_db.get_stats()
    assert stats_after["user_samples"] == initial_user_samples + 1
