from pathlib import Path


SOURCE = Path(__file__).with_name("m1_syntactic_rgat_pseudo_quick_ablation.py").read_text(encoding="utf-8")


def test_full_phase_a_treatment_inference_does_not_enable_graph_adapter():
    """Training keeps the graph adapter; historical inference must stay plain."""
    assert '_run_pipeline_command(args, variant_dirs["treatment"], "evaluate", False' in SOURCE
    assert '_run_pipeline_command(args, variant_dirs["treatment"], "pseudo", False' in SOURCE
