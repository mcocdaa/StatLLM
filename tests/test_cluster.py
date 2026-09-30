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
    seed_database(db, samples_per_probe=30)
    yield db
    if os.path.exists(path):
        os.remove(path)


def test_cluster_projection_array(temp_db):
    projector = ClusterProjector(temp_db)
    projector.fit(n_points_per_model=15)

    cluster_data = projector.get_cluster_data()
    assert "clusters" in cluster_data
    assert len(cluster_data["clusters"]) >= 2

    parsed_submissions = [
        {"probe_id": "arr_int5", "parsed_tokens": ["27", "83", "5", "61", "44"], "traits": {"has_duplicates": False}},
        {"probe_id": "arr_color5", "parsed_tokens": ["黄", "青", "红", "紫", "橙"], "traits": {"has_duplicates": False}},
    ]
    user_pt = projector.project_user_submission(parsed_submissions)
    assert user_pt is not None
    assert len(user_pt) == 2
