import json
import sys
import types

import pytest
import torch

from gmm_pbc import (
    PBCGMMRouter,
    RouterConfig,
    apply_selective_pairwise_boundaries,
    build_feature_cache_manifest,
    build_argparser,
    certify_boundary,
    find_optimal_boundary,
    get_pbc_features,
    merge_resume_config,
    prepare_feature_cache_config,
    split_task_features,
    validate_feature_tensor,
    validate_representation_manifest,
    run,
)


def test_split_task_features_is_disjoint_complete_and_deterministic():
    features = torch.arange(60, dtype=torch.float32).reshape(20, 3)

    first = split_task_features(
        features,
        gmm_fit_fraction=0.70,
        boundary_fit_fraction=0.15,
        seed=17,
    )
    second = split_task_features(
        features,
        gmm_fit_fraction=0.70,
        boundary_fit_fraction=0.15,
        seed=17,
    )

    gmm_fit, boundary_fit, boundary_cert = first
    assert [len(part) for part in first] == [14, 3, 3]
    assert all(torch.equal(a, b) for a, b in zip(first, second))

    recovered_rows = torch.cat([gmm_fit, boundary_fit, boundary_cert], dim=0)
    assert recovered_rows.unique(dim=0).shape[0] == len(features)
    assert {tuple(row.tolist()) for row in recovered_rows} == {
        tuple(row.tolist()) for row in features
    }


def test_find_optimal_boundary_maximizes_balanced_accuracy():
    new_task_margins = torch.tensor([0.40, 0.50, 0.60, 0.70])
    old_task_margins = torch.tensor([0.10, 0.20, 0.30, 0.35])

    result = find_optimal_boundary(new_task_margins, old_task_margins)

    assert result.boundary == pytest.approx(0.375)
    assert result.balanced_accuracy == pytest.approx(1.0)
    assert result.baseline_balanced_accuracy == pytest.approx(0.5)
    assert result.fit_gain == pytest.approx(0.5)


def test_certify_boundary_accepts_only_a_strictly_positive_paired_lcb():
    new_task_margins = torch.tensor([0.40, 0.50, 0.60, 0.70])
    old_task_margins = torch.tensor([0.10, 0.20, 0.30, 0.35])

    improved = certify_boundary(
        new_task_margins,
        old_task_margins,
        candidate_boundary=0.375,
        bootstrap_replicates=200,
        alpha=0.05,
        seed=23,
    )
    unchanged = certify_boundary(
        new_task_margins,
        old_task_margins,
        candidate_boundary=0.0,
        bootstrap_replicates=200,
        alpha=0.05,
        seed=23,
    )

    assert improved.observed_gain == pytest.approx(0.5)
    assert improved.lower_confidence_bound == pytest.approx(0.5)
    assert improved.accepted is True
    assert unchanged.observed_gain == pytest.approx(0.0)
    assert unchanged.lower_confidence_bound == pytest.approx(0.0)
    assert unchanged.accepted is False


def test_selective_routing_changes_only_low_margin_top_two_decisions():
    scores = torch.tensor(
        [
            [3.0, 1.0, 0.0],
            [1.0, 0.8, 0.0],
            [0.0, 0.8, 1.0],
        ]
    )

    predictions = apply_selective_pairwise_boundaries(
        scores=scores,
        task_ids=[0, 1, 2],
        boundaries={(1, 0): -0.3},
        margin_threshold=0.5,
    )

    assert predictions.tolist() == [0, 1, 2]


def test_router_checkpoint_round_trip_preserves_scores_and_boundaries(tmp_path):
    cfg = RouterConfig(
        tasks=("old", "new"),
        routing_dim=1,
        gmm_components=1,
        em_iters=10,
        gmm_fit_fraction=0.60,
        boundary_fit_fraction=0.20,
        old_pseudo_fit_samples=40,
        old_pseudo_cert_samples=40,
        bootstrap_replicates=100,
        margin_threshold=0.5,
        seed=31,
    )
    router = PBCGMMRouter(cfg)
    old_features = torch.linspace(-2.0, 0.0, 60).unsqueeze(1)
    new_features = torch.linspace(0.5, 2.5, 60).unsqueeze(1)
    router.fit_new_task("old", 0, old_features)
    router.fit_new_task("new", 1, new_features)
    router.representation_manifest = {
        "schema_version": 1,
        "fingerprint": "test-coordinate-system",
    }

    query = torch.tensor([[-1.0], [0.25], [1.5]])
    scores_before = router.predict_scores(query)
    predictions_before = router.predict(query)
    checkpoint = router.save(tmp_path, step=1)

    restored = PBCGMMRouter.load(checkpoint)

    torch.testing.assert_close(restored.predict_scores(query), scores_before)
    assert restored.predict(query).tolist() == predictions_before.tolist()
    assert restored.boundaries == router.boundaries
    assert restored.boundary_records == router.boundary_records
    assert restored.representation_manifest == router.representation_manifest


def test_cli_exposes_reproducible_pbc_controls():
    args = build_argparser().parse_args(
        [
            "--gmm_fit_fraction",
            "0.6",
            "--boundary_fit_fraction",
            "0.2",
            "--old_pseudo_fit_samples",
            "123",
            "--old_pseudo_cert_samples",
            "456",
            "--bootstrap_replicates",
            "789",
            "--bootstrap_alpha",
            "0.025",
            "--margin_threshold",
            "0.75",
            "--cache_tag",
            "dataset-revision-2",
        ]
    )

    assert args.gmm_fit_fraction == pytest.approx(0.6)
    assert args.boundary_fit_fraction == pytest.approx(0.2)
    assert args.old_pseudo_fit_samples == 123
    assert args.old_pseudo_cert_samples == 456
    assert args.bootstrap_replicates == 789
    assert args.bootstrap_alpha == pytest.approx(0.025)
    assert args.margin_threshold == pytest.approx(0.75)
    assert args.cache_tag == "dataset-revision-2"


def test_resume_preserves_checkpoint_method_parameters_and_updates_io():
    checkpoint_cfg = RouterConfig(
        tasks=("old", "new"),
        output_dir="old-output",
        routing_dim=17,
        bootstrap_alpha=0.01,
        margin_threshold=0.2,
    )
    runtime_cfg = RouterConfig(
        tasks=("old", "new", "future"),
        output_dir="new-output",
        feature_cache_dir="shared-cache",
        eval_k=77,
        eval_split="validation",
        routing_dim=256,
        bootstrap_alpha=0.05,
        margin_threshold=0.5,
        save_features=False,
        force_recompute_features=True,
    )

    merged = merge_resume_config(
        checkpoint_cfg,
        runtime_cfg,
        loaded_task_names=("old", "new"),
    )

    assert merged.routing_dim == 17
    assert merged.bootstrap_alpha == pytest.approx(0.01)
    assert merged.margin_threshold == pytest.approx(0.2)
    assert merged.tasks == ("old", "new", "future")
    assert merged.output_dir == "new-output"
    assert merged.feature_cache_dir == "shared-cache"
    assert merged.eval_k == 77
    assert merged.eval_split == "validation"
    assert merged.save_features is False
    assert merged.force_recompute_features is True


def test_feature_cache_manifest_changes_with_routing_coordinate_system():
    class DummyConfig:
        _commit_hash = "model-revision-a"

    class DummyModel:
        config = DummyConfig()

    class DummyTokenizer:
        name_or_path = "dummy-tokenizer"

    class DummyExtractor:
        model = DummyModel()
        tokenizer = DummyTokenizer()
        P = torch.eye(2)

    cfg = RouterConfig(
        model_name="dummy-model",
        routing_dim=2,
        seed=7,
        dataset_source="codetask",
    )
    first = build_feature_cache_manifest(cfg, DummyExtractor())
    changed_extractor = DummyExtractor()
    changed_extractor.P = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    changed_projection = build_feature_cache_manifest(cfg, changed_extractor)
    changed_seed = build_feature_cache_manifest(
        RouterConfig(
            model_name="dummy-model",
            routing_dim=2,
            seed=8,
            dataset_source="codetask",
        ),
        DummyExtractor(),
    )
    changed_cache_tag = build_feature_cache_manifest(
        RouterConfig(
            model_name="dummy-model",
            routing_dim=2,
            seed=7,
            dataset_source="codetask",
            cache_tag="dataset-revision-2",
        ),
        DummyExtractor(),
    )

    assert first["projection_sha256"] != changed_projection["projection_sha256"]
    assert first["fingerprint"] != changed_projection["fingerprint"]
    assert first["fingerprint"] != changed_seed["fingerprint"]
    assert first["fingerprint"] != changed_cache_tag["fingerprint"]


def test_feature_cache_config_uses_manifest_fingerprint_namespace(tmp_path):
    class DummyConfig:
        _commit_hash = "model-revision-a"

        def to_dict(self):
            return {"hidden_size": 2}

    class DummyModel:
        config = DummyConfig()

    class DummyTokenizer:
        name_or_path = "dummy-tokenizer"
        init_kwargs = {"_commit_hash": "tokenizer-revision-a"}

    class DummyExtractor:
        model = DummyModel()
        tokenizer = DummyTokenizer()
        P = torch.eye(2)

    cfg = RouterConfig(
        model_name="dummy-model",
        output_dir=str(tmp_path),
        routing_dim=2,
    )

    cache_cfg, manifest, namespace = prepare_feature_cache_config(
        cfg, DummyExtractor()
    )

    assert cache_cfg.feature_cache_dir == str(namespace)
    assert namespace.name == manifest["fingerprint"]
    with open(namespace / "cache_manifest.json", encoding="utf-8") as handle:
        assert json.load(handle) == manifest


def test_feature_tensor_validation_rejects_wrong_or_non_finite_coordinates():
    cfg = RouterConfig(routing_dim=2)
    valid = torch.zeros(3, 2)

    assert validate_feature_tensor(valid, cfg, "fixture") is valid
    with pytest.raises(ValueError, match="routing_dim"):
        validate_feature_tensor(torch.zeros(3, 1), cfg, "fixture")
    with pytest.raises(ValueError, match="finite"):
        validate_feature_tensor(
            torch.tensor([[0.0, float("inf")]]), cfg, "fixture"
        )


def test_resume_rejects_a_different_representation_manifest():
    stored = {"schema_version": 1, "fingerprint": "coordinate-system-a"}

    validate_representation_manifest(stored, dict(stored))
    with pytest.raises(ValueError, match="representation"):
        validate_representation_manifest(
            stored,
            {"schema_version": 1, "fingerprint": "coordinate-system-b"},
        )


@pytest.mark.parametrize(
    "override",
    [
        {"gmm_components": 0},
        {"em_iters": 0},
        {"eps": 0.0},
        {"variance_floor": 0.0},
        {"em_tol": float("nan")},
        {"eps": float("nan")},
        {"margin_threshold": float("nan")},
        {"gmm_fit_fraction": 0.9, "boundary_fit_fraction": 0.1},
        {"tasks": ("duplicate", "duplicate")},
    ],
)
def test_router_rejects_invalid_method_configuration(override):
    with pytest.raises(ValueError):
        PBCGMMRouter(RouterConfig(**override))


def test_router_rejects_non_finite_training_features():
    router = PBCGMMRouter(
        RouterConfig(
            tasks=("bad",),
            routing_dim=1,
            gmm_components=1,
            old_pseudo_fit_samples=2,
            old_pseudo_cert_samples=2,
            bootstrap_replicates=2,
        )
    )

    with pytest.raises(ValueError, match="finite"):
        router.fit_new_task(
            "bad",
            0,
            torch.tensor([[0.0], [1.0], [float("nan")], [2.0]]),
        )


def test_checkpoint_load_rejects_invalid_gmm_state(tmp_path):
    cfg = RouterConfig(
        tasks=("only",),
        routing_dim=1,
        gmm_components=1,
        old_pseudo_fit_samples=2,
        old_pseudo_cert_samples=2,
        bootstrap_replicates=2,
    )
    router = PBCGMMRouter(cfg)
    router.fit_new_task("only", 0, torch.arange(8.0).unsqueeze(1))
    checkpoint = router.save(tmp_path, step=0)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["tasks"][0]["gmm"]["var"][0, 0] = -1.0
    corrupt_checkpoint = tmp_path / "corrupt.pt"
    torch.save(payload, corrupt_checkpoint)

    with pytest.raises(ValueError, match="GMM"):
        PBCGMMRouter.load(corrupt_checkpoint)


def test_checkpoint_load_rejects_inconsistent_boundary_record(tmp_path):
    cfg = RouterConfig(
        tasks=("old", "new"),
        routing_dim=1,
        gmm_components=1,
        old_pseudo_fit_samples=20,
        old_pseudo_cert_samples=20,
        bootstrap_replicates=20,
    )
    router = PBCGMMRouter(cfg)
    router.fit_new_task("old", 0, torch.linspace(-1.0, 0.0, 20).unsqueeze(1))
    router.fit_new_task("new", 1, torch.linspace(0.1, 1.1, 20).unsqueeze(1))
    checkpoint = router.save(tmp_path, step=1)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["boundary_records"][0]["accepted"] = True
    payload["boundary_records"][0]["lower_confidence_bound"] = 0.0
    corrupt_checkpoint = tmp_path / "bad-record.pt"
    torch.save(payload, corrupt_checkpoint)

    with pytest.raises(ValueError, match="boundary record"):
        PBCGMMRouter.load(corrupt_checkpoint)


def test_run_resumes_with_projection_and_preserves_prior_results(
    tmp_path,
    monkeypatch,
):
    class DummyModelConfig:
        _commit_hash = "dummy-model-revision"

        def to_dict(self):
            return {"hidden_size": 1}

    class DummyModel:
        config = DummyModelConfig()

    class DummyTokenizer:
        name_or_path = "dummy-tokenizer"
        init_kwargs = {"_commit_hash": "dummy-tokenizer-revision"}

        def get_vocab(self):
            return {"token": 0}

    class DummyExtractor:
        def __init__(
            self,
            model_name,
            feature_layers,
            routing_dim,
            device,
            seed,
        ):
            self.model = DummyModel()
            self.tokenizer = DummyTokenizer()
            self.P = torch.eye(routing_dim)

        def save_projection(self, output_dir):
            torch.save(self.P, output_dir / "projection_P.pt")

        def load_projection(self, path):
            self.P = torch.load(path, map_location="cpu", weights_only=True)

    def fake_get_features(extractor, cfg, task, split, k):
        sample_count = 20 if split == "train" else 8
        offset = 0.0 if task == "old" else 0.5
        return (
            torch.linspace(-1.0 + offset, 1.0 + offset, sample_count)
            .unsqueeze(1)
            .repeat(1, cfg.routing_dim)
        )

    fake_gmm = types.SimpleNamespace(
        RoutingFeatureExtractor=DummyExtractor,
        get_device=lambda: torch.device("cpu"),
        get_or_extract_features=fake_get_features,
        set_seed=torch.manual_seed,
    )
    monkeypatch.setitem(sys.modules, "gmm", fake_gmm)

    output_dir = tmp_path / "pbc-run"
    initial_cfg = RouterConfig(
        model_name="dummy-model",
        output_dir=str(output_dir),
        tasks=("old",),
        routing_dim=1,
        train_k=20,
        eval_k=8,
        gmm_components=1,
        em_iters=5,
        old_pseudo_fit_samples=20,
        old_pseudo_cert_samples=20,
        bootstrap_replicates=20,
    )
    run(initial_cfg)

    resumed_cfg = RouterConfig(
        model_name="dummy-model",
        output_dir=str(output_dir),
        tasks=("old", "new"),
        routing_dim=1,
        train_k=20,
        eval_k=8,
        gmm_components=1,
        em_iters=5,
        old_pseudo_fit_samples=20,
        old_pseudo_cert_samples=20,
        bootstrap_replicates=20,
    )
    run(
        resumed_cfg,
        resume_from=str(output_dir / "router_step0.pt"),
    )

    restored = PBCGMMRouter.load(output_dir / "router_step1.pt")
    assert [task.task_name for task in restored.tasks] == ["old", "new"]
    assert len(restored.boundary_records) == 1
    assert (output_dir / "projection_P.pt").exists()
    with open(output_dir / "routing_results.json", encoding="utf-8") as handle:
        results = json.load(handle)
    assert set(results) == {"step0", "step1"}


def test_invalid_cached_feature_shape_is_recomputed(monkeypatch):
    calls = []

    def fake_get_features(**kwargs):
        calls.append(kwargs["cfg"].force_recompute_features)
        if len(calls) == 1:
            return torch.zeros(3, 1)
        return torch.zeros(3, 2)

    monkeypatch.setitem(
        sys.modules,
        "gmm",
        types.SimpleNamespace(get_or_extract_features=fake_get_features),
    )
    cfg = RouterConfig(routing_dim=2, save_features=True)

    features = get_pbc_features(None, cfg, "task", "train", 3)

    assert features.shape == (3, 2)
    assert calls == [False, True]
