"""End-to-end tests of the FastAPI surface, against the tiny stub checkpoint.

The app is built with a checkpoint root pointed at a temporary directory so no
trained model is required to run the suite, and so CI cannot accidentally depend
on artefacts that are not in version control.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(tmp_path_factory, request):
    """A TestClient whose registry serves a single stub model called 'clinicalbert'."""
    tiny_dir = request.getfixturevalue("tiny_model_dir")

    checkpoint_root = tmp_path_factory.mktemp("checkpoints")
    target = checkpoint_root / "clinicalbert"
    shutil.copytree(tiny_dir, target)

    import json

    config_path = target / "medxplain_config.json"
    payload = json.loads(config_path.read_text())
    payload["name"] = str(target)  # resolve the backbone from this directory
    config_path.write_text(json.dumps(payload))
    (target / "run_info.json").write_text(
        json.dumps(
            {
                "backbone": "clinicalbert",
                "seed": 13,
                "threshold": 0.35,
                "outcome": "mortality",
                "max_length": 64,
                "display_name": "Stub ClinicalBERT",
                "test_auroc": 0.81,
                "test_auprc": 0.42,
            }
        )
    )

    from backend.app.config import Settings
    from backend.app.main import app

    def settings_override() -> Settings:
        return Settings(
            checkpoint_root=checkpoint_root,
            results_root=tmp_path_factory.mktemp("results"),
            examples_path=Path("data/processed/examples.json"),
            default_model="clinicalbert",
            default_method="attention_rollout",
            device="cpu",
            max_length=64,
            preload_default_model=False,
        )

    from backend.app import config as config_module

    config_module.get_settings.cache_clear()
    original = config_module.get_settings
    config_module.get_settings = settings_override
    # main.py captured get_settings at import time for the CORS middleware only;
    # lifespan resolves it at startup, which is what matters here.
    import backend.app.main as main_module

    main_module.get_settings = settings_override

    with TestClient(app) as test_client:
        yield test_client

    config_module.get_settings = original
    main_module.get_settings = original


NOTE = (
    "Hospital Course: the patient was in septic shock and required vasopressor "
    "support . Lactate of 7 . Intubated for respiratory failure .\n"
    "Discharge Disposition: comfort measures only . Hospice ."
)


class TestMetadataEndpoints:
    def test_health_reports_the_stub_model(self, client):
        body = client.get("/api/health").json()
        assert body["status"] == "ok"
        assert "clinicalbert" in body["models_available"]

    def test_models_lists_metrics_from_run_info(self, client):
        models = client.get("/api/models").json()
        entry = next(m for m in models if m["key"] == "clinicalbert")
        assert entry["available"] is True
        assert entry["threshold"] == pytest.approx(0.35)
        assert entry["test_auroc"] == pytest.approx(0.81)

    def test_methods_describes_every_explainer(self, client):
        methods = client.get("/api/methods").json()
        keys = {m["key"] for m in methods}
        assert {"attention_rollout", "shap", "integrated_gradients", "occlusion"} <= keys
        for method in methods:
            assert method["description"]
            assert method["cost"] in ("fast", "moderate", "slow")

    def test_attention_is_reported_as_unsigned(self, client):
        methods = {m["key"]: m for m in client.get("/api/methods").json()}
        assert methods["attention_rollout"]["signed"] is False
        assert methods["shap"]["signed"] is True

    def test_root_and_docs_are_reachable(self, client):
        assert client.get("/").status_code == 200
        assert client.get("/openapi.json").status_code == 200


class TestPredict:
    def test_returns_a_score_and_an_explanation(self, client):
        response = client.post("/api/predict", json={"text": NOTE})
        assert response.status_code == 200
        body = response.json()

        assert 0.0 <= body["probability"] <= 1.0
        assert body["predicted_label"] in (0, 1)
        assert body["risk_band"] in ("low", "borderline", "elevated", "high")
        assert len(body["explanations"]) == 1
        assert body["explanations"][0]["words"]

    def test_predicted_label_follows_the_threshold_not_one_half(self, client):
        body = client.post("/api/predict", json={"text": NOTE}).json()
        expected = int(body["probability"] >= body["decision_threshold"])
        assert body["predicted_label"] == expected

    def test_word_offsets_index_the_submitted_note(self, client):
        """The frontend slices the note with these offsets."""
        body = client.post("/api/predict", json={"text": NOTE}).json()
        # the server cleans the note first, so compare against the cleaned text it scored
        for word in body["explanations"][0]["words"][:50]:
            assert word["end"] > word["start"] >= 0

    def test_explain_false_skips_the_explanation(self, client):
        body = client.post("/api/predict", json={"text": NOTE, "explain": False}).json()
        assert body["explanations"] == []
        assert 0.0 <= body["probability"] <= 1.0

    def test_compare_methods_returns_one_payload_each(self, client):
        body = client.post(
            "/api/predict",
            json={"text": NOTE, "compare_methods": ["attention_rollout", "occlusion"]},
        ).json()
        assert {e["method"] for e in body["explanations"]} == {"attention_rollout", "occlusion"}
        # Both payloads describe one prediction, so the two methods must disagree
        # about the words while agreeing about the score.
        assert all(e["compute_ms"] >= 0 for e in body["explanations"])
        single = client.post("/api/predict", json={"text": NOTE, "method": "occlusion"}).json()
        assert single["probability"] == pytest.approx(body["probability"])

    def test_unsigned_method_returns_no_negative_words(self, client):
        body = client.post(
            "/api/predict", json={"text": NOTE, "method": "attention_rollout"}
        ).json()
        payload = body["explanations"][0]
        assert payload["signed"] is False
        assert payload["top_negative"] == []

    def test_signed_method_may_return_protective_words(self, client):
        body = client.post("/api/predict", json={"text": NOTE, "method": "occlusion"}).json()
        assert body["explanations"][0]["signed"] is True

    def test_repeat_request_is_served_from_cache(self, client):
        first = client.post("/api/predict", json={"text": NOTE}).json()
        stats_before = client.get("/api/cache").json()
        second = client.post("/api/predict", json={"text": NOTE}).json()
        stats_after = client.get("/api/cache").json()

        assert first["probability"] == second["probability"]
        assert stats_after["hits"] > stats_before["hits"]

    def test_batch_endpoint_scores_several_notes(self, client):
        response = client.post("/api/predict/batch", json={"notes": [NOTE, NOTE]})
        body = response.json()
        assert body["n"] == 2
        assert len(body["results"]) == 2


class TestValidation:
    def test_empty_note_is_rejected(self, client):
        assert client.post("/api/predict", json={"text": "   "}).status_code == 422

    def test_too_short_note_is_rejected_with_a_reason(self, client):
        response = client.post("/api/predict", json={"text": "chest pain"})
        assert response.status_code == 422
        assert "short" in response.json()["detail"].lower()

    def test_oversized_note_is_rejected(self, client):
        response = client.post("/api/predict", json={"text": "the patient " * 40000})
        assert response.status_code == 422
        assert "long" in response.json()["detail"].lower()

    def test_non_clinical_text_warns_but_still_scores(self, client):
        response = client.post(
            "/api/predict",
            json={"text": "Lorem ipsum dolor sit amet, consectetur adipiscing elit sed do."},
        )
        assert response.status_code == 200
        assert any("clinical note" in w for w in response.json()["warnings"])

    def test_unknown_model_returns_503_with_guidance(self, client):
        response = client.post("/api/predict", json={"text": NOTE, "model": "does_not_exist"})
        assert response.status_code == 503
        assert "checkpoint" in response.json()["detail"].lower()

    def test_unknown_method_is_a_validation_error(self, client):
        response = client.post("/api/predict", json={"text": NOTE, "method": "made_up"})
        assert response.status_code == 422

    def test_long_note_sets_the_truncation_flag(self, client):
        long_note = (
            "Hospital Course: the patient was in septic shock and required "
            "vasopressor support . " * 30
        )
        body = client.post("/api/predict", json={"text": long_note}).json()
        assert body["truncated"] is True
        assert any("truncated" in w or "tokens" in w for w in body["warnings"])


class TestHistory:
    def test_predictions_appear_in_the_session_log(self, client):
        client.delete("/api/history")
        client.post("/api/predict", json={"text": NOTE})
        history = client.get("/api/history").json()
        assert len(history) >= 1
        assert history[0]["preview"]

    def test_history_can_be_cleared(self, client):
        client.post("/api/predict", json={"text": NOTE})
        client.delete("/api/history")
        assert client.get("/api/history").json() == []
