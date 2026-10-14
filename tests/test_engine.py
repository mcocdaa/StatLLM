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
        {"probe_id": "arr_prime5", "raw_text": "[11, 23, 47, 71, 89]"},
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


def test_engine_probability_conservation(temp_db):
    """
    Verifies that all conditional, positional, and interpolated probabilities
    strictly sum to 1.0 (Kolmogorov's first axiom of probability conservation).
    """
    from statllm.probes import get_probe
    evaluator = LikelihoodEvaluator(temp_db, beta=0.5, positional_lambda=0.5)
    matrix = temp_db.get_all_model_probe_counts()

    probe = get_probe("arr_letter5")
    vocab = probe.allowed_elements
    vocab_size = probe.element_vocab_size

    for m in matrix.keys():
        counts_dict, tot = matrix[m].get("arr_letter5", ({}, 0.0))
        assert tot > 0

        # 1. Global probability sums to 1.0 across the full vocabulary simplex (including OOV slot)
        p_global_sum = sum(evaluator.get_token_prob(counts_dict, tot, v, vocab_size) for v in vocab)
        p_global_sum += evaluator.get_token_prob(counts_dict, tot, "INVALID", vocab_size)
        assert pytest.approx(p_global_sum, rel=1e-5) == 1.0

        # 2. Position-specific and interpolated probability sum to 1.0 for every position
        for pos in range(5):
            for lam in [0.0, 0.3, 0.5, 0.8, 1.0]:
                p_interp_sum = sum(
                    evaluator.get_interpolated_token_prob(counts_dict, tot, v, pos, vocab_size, 5, lam)
                    for v in vocab
                )
                p_interp_sum += evaluator.get_interpolated_token_prob(counts_dict, tot, "INVALID", pos, vocab_size, 5, lam)
                assert pytest.approx(p_interp_sum, rel=1e-5) == 1.0


def test_source_type_and_primary_token_denominator_invariants(temp_db):
    """
    Verifies that:
    1. Non-standard source_type strings (e.g., 'official_api') are normalized at write time,
       so total_samples == official_samples + user_samples always holds.
    2. Multi-feature probes (arr_int_big5, arr_pseudo_cvc5) only count primary element tokens
       in the denominator `tot`, preventing 4x denominator inflation.
    3. The robust tournament ledger is strictly monotonically non-increasing in both
       champion-anchored margin (delta_ll) and posterior probability.
    """
    temp_db.add_sample(
        model_name="Grok-4.7",
        probe_id="arr_int_big5",
        raw_text="[12345, 67890, 23456, 34567, 45678]",
        parsed_tokens=["12345", "67890", "23456", "34567", "45678"],
        traits={"has_duplicates": False},
        source_type="official_api",
        weight=1.0
    )
    temp_db.add_sample(
        model_name="Grok-4.7",
        probe_id="arr_pseudo_cvc5",
        raw_text='["BAK", "MUP", "ZOT", "RIL", "VEN"]',
        parsed_tokens=["BAK", "MUP", "ZOT", "RIL", "VEN"],
        traits={"has_duplicates": False},
        source_type="official_batch",
        weight=1.0
    )

    stats = temp_db.get_stats()
    assert stats["total_samples"] == stats["official_samples"] + stats["user_samples"]

    matrix = temp_db.get_all_model_probe_counts()
    _, tot_big5 = matrix["Grok-4.7"]["arr_int_big5"]
    _, tot_cvc5 = matrix["Grok-4.7"]["arr_pseudo_cvc5"]
    assert tot_big5 == 5.0
    assert tot_cvc5 == 5.0

    evaluator = LikelihoodEvaluator(temp_db)
    res = evaluator.evaluate([
        {"probe_id": "arr_int_big5", "raw_text": "[12345, 67890, 23456, 34567, 45678]"},
        {"probe_id": "arr_pseudo_cvc5", "raw_text": '["BAK", "MUP", "ZOT", "RIL", "VEN"]'},
    ], n_boot=100)
    mb = res["robust_attribution"]["model_breakdown"]
    assert len(mb) >= 2
    assert mb[0]["delta_ll"] == 0.0
    for i in range(len(mb) - 1):
        assert mb[i]["delta_ll"] >= mb[i + 1]["delta_ll"]
        assert mb[i]["posterior"] >= mb[i + 1]["posterior"]

