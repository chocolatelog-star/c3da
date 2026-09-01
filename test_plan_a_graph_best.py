import json
from pathlib import Path

from run_plan_a_graph_best import build_adapter_manifest, required_adapter_paths


def test_adapter_manifest_maps_phase_a_treatment(tmp_path: Path):
    treatment = tmp_path / "phase_a" / "treatment"
    model = treatment / "models" / "extractor" / "best"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}", encoding="utf-8")
    (treatment / "target_pseudo_selected.jsonl").write_text('{"text":"x","label":""}\n', encoding="utf-8")
    adapter = tmp_path / "adapter"
    manifest = build_adapter_manifest(treatment, adapter, source="laptop14", target="rest15", seed=1000)
    assert manifest["target_test_access"] is False
    assert manifest["source"] == str(treatment.resolve())
    assert all(path.exists() for path in required_adapter_paths(adapter))
    state = json.loads((adapter / "target_pseudo_generation_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "complete"
    assert state["target_test_access"] is False
