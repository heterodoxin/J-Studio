from jstudio.services import hf_cache


def test_dense_lens_discovery_requires_complete_viewing_evidence(tmp_path):
    import torch

    lens = tmp_path / "lens.pt"
    checkpoint = {
        "metadata": {
            "estimator": "projected-dense-jacobian-v2",
            "quality_stage": "Stable",
        },
        "geometry": {0: {"calibrated": True}},
    }
    torch.save(checkpoint, lens)

    assert hf_cache.inspect_lens_file(lens).state is hf_cache.LensState.NEEDS_FIT

    checkpoint["metadata"].update(
        {
            "quality_gate_version": "jspace-viewing-v2",
            "viewing_passed": "3",
            "viewing_total": "3",
        }
    )
    torch.save(checkpoint, lens)
    inspection = hf_cache.inspect_lens_file(lens)

    assert inspection.state is hf_cache.LensState.STABLE
    assert "3/3" in inspection.detail


def test_scan_hf_cache_finds_cached_models_without_downloads(tmp_path):
    hub = tmp_path / "hub"
    repo = hub / "models--Qwen--Qwen2.5-7B-Instruct"
    repo.mkdir(parents=True)
    (hub / "models--not-a-model-file").write_text("ignored")
    (hub / "datasets--Org--Data").mkdir()

    models = hf_cache.scan_hf_cache(cache_roots=(hub,), lens_roots=(tmp_path / "lenses",))

    assert [model.model_id for model in models] == ["Qwen/Qwen2.5-7B-Instruct"]
    assert models[0].cache_path == repo
    assert models[0].lens_state == hf_cache.LensState.MISSING


def test_scan_hf_cache_detects_existing_jstudio_lens(tmp_path, monkeypatch):
    hub = tmp_path / "hub"
    (hub / "models--Qwen--Qwen2.5-7B-Instruct").mkdir(parents=True)
    lens = tmp_path / "lenses" / "Qwen--Qwen2.5-7B-Instruct" / "lens.pt"
    lens.parent.mkdir(parents=True)
    lens.write_bytes(b"lens")

    monkeypatch.setattr(
        hf_cache,
        "inspect_lens_file",
        lambda path: hf_cache.LensInspection(
            hf_cache.LensState.STABLE,
            path,
            "Stable calibrated lens",
        ),
    )

    models = hf_cache.scan_hf_cache(cache_roots=(hub,), lens_roots=(tmp_path / "lenses",))

    assert models[0].lens_state == hf_cache.LensState.STABLE
    assert models[0].lens_path == lens
    assert models[0].lens_detail == "Stable calibrated lens"


def test_explicit_lens_file_is_reported_even_outside_jstudio_cache(
    tmp_path, monkeypatch
):
    lens = tmp_path / "manual-lens.pt"
    lens.write_bytes(b"lens")
    monkeypatch.setattr(
        hf_cache,
        "inspect_lens_file",
        lambda path: hf_cache.LensInspection(hf_cache.LensState.STABLE, path, "Manual"),
    )

    inspection = hf_cache.inspect_model_lens(
        "Qwen/Qwen2.5-7B-Instruct",
        lens_roots=(tmp_path / "empty",),
        explicit_lens_path=lens,
    )

    assert inspection.state == hf_cache.LensState.STABLE
    assert inspection.path == lens
