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
    seed_database(db, samples_per_probe=40)
    yield db
    if os.path.exists(path):
        os.remove(path)


def test_engine_array_evaluation(temp_db):
    evaluator = LikelihoodEvaluator(temp_db)

    # Grok-4.7 characteristic array inputs
    submissions = [
        {"probe_id": "arr_int5", "raw_text": "[27, 83, 5, 61, 44]"},
        {"probe_id": "arr_color5", "raw_text": '["黄", "青", "红", "紫", "橙"]'},
        {"probe_id": "arr_letter5", "raw_text": '["K", "W", "B", "R", "M"]'},
    ]

    result = evaluator.evaluate(submissions, n_boot=200)

    posteriors = result["posteriors"]
    assert pytest.approx(sum(posteriors.values()), rel=1e-4) == 1.0

    # Top model should be Grok-4.7
    assert result["top_model"] == "Grok-4.7"
    assert posteriors["Grok-4.7"] > 0.4

    ci = result["confidence_intervals"]
    assert "Grok-4.7" in ci
    lower, upper = ci["Grok-4.7"]
    assert 0.0 <= lower <= upper <= 1.0


def test_weighted_crowdsourcing_array(temp_db):
    stats_before = temp_db.get_stats()
    initial_user = stats_before["user_samples"]

    temp_db.add_sample(
        model_name="Grok-4.7",
        probe_id="arr_int5",
        raw_text="[27, 44, 83, 5, 61]",
        parsed_tokens=["27", "44", "83", "5", "61"],
        traits={"has_duplicates": False, "first_token": "27"},
        source_type="user",
        weight=0.2
    )

    stats_after = temp_db.get_stats()
    assert stats_after["user_samples"] == initial_user + 1
