import math

import pytest
import torch

from gmm_mc import (
    MinimumChangeGMMRouter,
    RouterConfig,
    TaskRouter,
    build_argparser,
    certify_candidate,
    component_responsibilities,
    hoeffding_upper_bound,
    paired_bootstrap_gain_lcb,
    solve_minimum_change_qp,
    split_task_features,
)
from gmm_pbc import WeightedDiagonalGMM


def fitted_gmm(values: torch.Tensor, components: int = 1) -> WeightedDiagonalGMM:
    gmm = WeightedDiagonalGMM(
        n_components=components,
        variance_floor=0.05,
        eps=1e-8,
    )
    return gmm.fit(values.float(), em_iters=30, tol=1e-6, seed=7)


def test_component_responsibilities_are_normalized_and_drive_score_correction():
    values = torch.tensor([[-2.2], [-2.0], [-1.8], [1.8], [2.0], [2.2]])
    gmm = fitted_gmm(values, components=2)
    query = torch.tensor([[-2.0], [0.0], [2.0]])
    gamma = component_responsibilities(gmm, query)

    assert gamma.shape == (3, 2)
    torch.testing.assert_close(gamma.sum(dim=1), torch.ones(3))
    correction = torch.tensor([0.75, -0.25])
    router = MinimumChangeGMMRouter(
        RouterConfig(tasks=("task",), routing_dim=1, gmm_components=2)
    )
    router.tasks.append(TaskRouter("task", 0, gmm, correction))

    expected = gmm.log_prob(query) + gamma @ correction
    torch.testing.assert_close(router.predict_scores(query)[:, 0], expected)
    torch.testing.assert_close(router.predict_baseline_scores(query)[:, 0], gmm.log_prob(query))


def test_three_way_split_is_deterministic_disjoint_and_non_empty():
    features = torch.arange(60.0).reshape(20, 3)
    first = split_task_features(features, 0.70, 0.15, seed=13)
    second = split_task_features(features, 0.70, 0.15, seed=13)

    assert [len(part) for part in first] == [14, 3, 3]
    for left, right in zip(first, second):
        torch.testing.assert_close(left, right)
    rows = [tuple(row.tolist()) for part in first for row in part]
    assert len(rows) == len(set(rows)) == len(features)


def test_dual_qp_solver_returns_minimum_norm_feasible_correction():
    solution = solve_minimum_change_qp(
        new_responsibilities=torch.ones(2, 1),
        new_deficits=torch.ones(2),
        old_responsibilities=torch.ones(1, 1),
        old_upper_bounds=torch.tensor([2.0]),
        c_new=100.0,
        c_old=100.0,
        max_iter=500,
        tolerance=1e-9,
    )

    assert solution.success is True
    assert solution.correction.item() == pytest.approx(1.0, abs=1e-5)
    assert solution.objective == pytest.approx(0.5, abs=1e-5)
    assert solution.new_slack_sum == pytest.approx(0.0, abs=1e-6)
    assert solution.old_slack_sum == pytest.approx(0.0, abs=1e-6)


def test_paired_bootstrap_lcb_uses_example_level_pairing():
    baseline = torch.zeros(20, dtype=torch.bool)
    candidate = torch.ones(20, dtype=torch.bool)
    observed, lcb = paired_bootstrap_gain_lcb(
        baseline,
        candidate,
        bootstrap_replicates=200,
        alpha=0.05,
        seed=11,
    )

    assert observed == pytest.approx(1.0)
    assert lcb == pytest.approx(1.0)


def test_historical_ucb_is_nonzero_when_no_sampled_disturbance_occurs():
    observations = torch.zeros(100)
    expected = math.sqrt(math.log(20.0) / 200.0)

    assert hoeffding_upper_bound(observations, 0.05) == pytest.approx(expected)


def test_candidate_certification_reports_both_one_sided_bounds():
    certification = certify_candidate(
        baseline_new_correct=torch.zeros(40, dtype=torch.bool),
        candidate_new_correct=torch.ones(40, dtype=torch.bool),
        historical_disturbance=torch.zeros(1000, dtype=torch.bool),
        bootstrap_replicates=200,
        alpha=0.05,
        seed=3,
    )

    assert certification.new_gain_lcb == pytest.approx(1.0)
    assert certification.historical_disturbance == pytest.approx(0.0)
    assert certification.historical_disturbance_ucb > 0.0


def test_no_certified_gain_falls_back_to_exact_raw_gmm_and_round_trips(tmp_path):
    cfg = RouterConfig(
        tasks=("old", "new"),
        routing_dim=1,
        gmm_components=1,
        em_iters=20,
        variance_floor=0.05,
        gmm_fit_fraction=0.60,
        correction_opt_fraction=0.20,
        old_pseudo_opt_samples=100,
        old_pseudo_cert_samples=1000,
        bootstrap_replicates=100,
        old_disturbance_budget=1.0,
        seed=17,
    )
    router = MinimumChangeGMMRouter(cfg)
    router.fit_new_task("old", 0, torch.linspace(-4.0, -2.0, 60).unsqueeze(1))
    old_correction = router.tasks[0].correction.clone()
    router.fit_new_task("new", 1, torch.linspace(2.0, 4.0, 60).unsqueeze(1))

    assert router.correction_records[1].accepted is False
    assert router.correction_records[1].new_gain_lcb <= 0.0
    torch.testing.assert_close(router.tasks[0].correction, old_correction)
    torch.testing.assert_close(router.tasks[1].correction, torch.zeros(1))
    query = torch.tensor([[-3.0], [3.0]])
    torch.testing.assert_close(router.predict_scores(query), router.predict_baseline_scores(query))

    router.representation_manifest = {"schema_version": 1, "fingerprint": "test-space"}
    checkpoint = router.save(tmp_path, step=1)
    restored = MinimumChangeGMMRouter.load(checkpoint)
    torch.testing.assert_close(restored.predict_scores(query), router.predict_scores(query))
    assert restored.predict(query).tolist() == router.predict(query).tolist()
    assert restored.correction_records == router.correction_records
    assert restored.representation_manifest == router.representation_manifest


def test_cli_exposes_minimum_change_and_certification_controls():
    args = build_argparser().parse_args(
        [
            "--correction_opt_fraction",
            "0.2",
            "--old_pseudo_opt_samples",
            "123",
            "--old_pseudo_cert_samples",
            "456",
            "--historical_margin",
            "0.7",
            "--new_margin",
            "0.1",
            "--old_margin",
            "0.2",
            "--c_new",
            "2.0",
            "--c_old",
            "3.0",
            "--old_disturbance_budget",
            "0.03",
            "--confidence_alpha",
            "0.01",
        ]
    )

    assert args.correction_opt_fraction == pytest.approx(0.2)
    assert args.old_pseudo_opt_samples == 123
    assert args.old_pseudo_cert_samples == 456
    assert args.historical_margin == pytest.approx(0.7)
    assert args.new_margin == pytest.approx(0.1)
    assert args.old_margin == pytest.approx(0.2)
    assert args.c_new == pytest.approx(2.0)
    assert args.c_old == pytest.approx(3.0)
    assert args.old_disturbance_budget == pytest.approx(0.03)
    assert args.confidence_alpha == pytest.approx(0.01)


@pytest.mark.parametrize(
    "override",
    [
        {"gmm_components": 0},
        {"gmm_fit_fraction": 0.9, "correction_opt_fraction": 0.1},
        {"c_new": 0.0},
        {"c_old": float("nan")},
        {"confidence_alpha": 1.0},
        {"old_disturbance_budget": -0.1},
        {"tasks": ("duplicate", "duplicate")},
    ],
)
def test_router_rejects_invalid_configuration(override):
    with pytest.raises(ValueError):
        MinimumChangeGMMRouter(RouterConfig(**override))
