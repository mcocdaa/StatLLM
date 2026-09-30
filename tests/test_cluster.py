import pytest
import os
import tempfile
from statllm.database import Database
from statllm.cluster import ClusterProjector
from statllm.seed_data import seed_database


@pytest.fixture
def temp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = Database(path)
    seed_database(db, samples_per_probe=60)
    yield db
    if os.path.exists(path):
        os.remove(path)


def test_cluster_projection(temp_db):
    projector = ClusterProjector(temp_db)
    projector.fit(n_points_per_model=20)

    cluster_data = projector.get_cluster_data()
    assert "clusters" in cluster_data
    assert len(cluster_data["clusters"]) >= 2

    # Check each cluster structure
    for c in cluster_data["clusters"]:
        assert "model_name" in c
        assert "center" in c
        assert len(c["center"]) == 2
        assert "points" in c
        assert len(c["points"]) == 20

    # Test user projection
    parsed_submissions = [
        {"probe_id": "q1_int", "parsed_value": "42"},
        {"probe_id": "q2_color", "parsed_value": "蓝色"},
        {"probe_id": "q3_rps", "parsed_value": "石头"},
    ]
    user_pt = projector.project_user_submission(parsed_submissions)
    assert user_pt is not None
    assert len(user_pt) == 2
