import pytest
import os
import tempfile
from web.app import create_app, EvaluateRequest, SubmissionItem, ContributeRequest
from statllm.database import Database
from statllm.seed_data import seed_database


@pytest.fixture
def test_app_and_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = Database(path)
    seed_database(db, samples_per_probe=20)
    app = create_app(path)
    yield app, db
    if os.path.exists(path):
        os.remove(path)


def test_evaluate_without_claimed_model_does_not_poison(test_app_and_db):
    app, db = test_app_and_db
    stats_before = db.get_stats()
    initial_user_samples = stats_before["user_samples"]

    # Find the evaluate route endpoint function
    eval_func = None
    for route in app.routes:
        if getattr(route, "path", None) == "/api/evaluate":
            eval_func = route.endpoint
            break
    assert eval_func is not None

    req = EvaluateRequest(
        submissions=[SubmissionItem(probe_id="arr_int5", raw_text="[1, 2, 3, 4, 5]")],
        consent_to_collect=True,
        claimed_model=None
    )
    result = eval_func(req)

    assert result["saved_samples_count"] == 0
    assert result["claimed_model"] is None

    # Verify no sample was written into database
    stats_after = db.get_stats()
    assert stats_after["user_samples"] == initial_user_samples


def test_evaluate_with_explicit_claimed_model_saves_and_registers(test_app_and_db):
    app, db = test_app_and_db
    stats_before = db.get_stats()
    initial_user_samples = stats_before["user_samples"]

    eval_func = None
    for route in app.routes:
        if getattr(route, "path", None) == "/api/evaluate":
            eval_func = route.endpoint
            break
    assert eval_func is not None

    req = EvaluateRequest(
        submissions=[SubmissionItem(probe_id="arr_int5", raw_text="[14, 28, 55, 72, 91]")],
        consent_to_collect=True,
        claimed_model="New-Community-Model-7B"
    )
    result = eval_func(req)

    assert result["saved_samples_count"] == 1
    assert result["claimed_model"] == "New-Community-Model-7B"

    # Verify sample was written and model was registered
    stats_after = db.get_stats()
    assert stats_after["user_samples"] == initial_user_samples + 1
    models = [m["name"] for m in db.list_models()]
    assert "New-Community-Model-7B" in models


def test_contribute_endpoint(test_app_and_db):
    app, db = test_app_and_db

    contrib_func = None
    for route in app.routes:
        if getattr(route, "path", None) == "/api/contribute":
            contrib_func = route.endpoint
            break
    assert contrib_func is not None

    req = ContributeRequest(
        model_name="Custom-FineTuned-Model",
        probe_id="arr_rps5",
        raw_text="['rock', 'paper', 'scissors', 'rock', 'rock']",
        weight=0.2
    )
    result = contrib_func(req)

    assert result["status"] == "success"
    models = [m["name"] for m in db.list_models()]
    assert "Custom-FineTuned-Model" in models
