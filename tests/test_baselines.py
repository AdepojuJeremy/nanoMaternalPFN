from nanomaternalpfn.baselines import evaluate_baselines


def test_baselines_return_valid_metrics():
    results = evaluate_baselines(
        n_tasks=2,
        seed_start=200_000,
        rf_trees=5,
    )

    assert {result.name for result in results} == {
        "logistic_regression",
        "random_forest",
    }

    for result in results:
        assert result.loss > 0
        assert 0.0 <= result.accuracy <= 1.0
