from pathlib import Path


def test_matrix_unit_paths_are_isolated(tmp_path: Path):
    from experiment_runner_common import matrix_unit_dir

    assert matrix_unit_dir(tmp_path, 8, 2).name == "batch8_accum2"
    assert matrix_unit_dir(tmp_path, 16, 1) != matrix_unit_dir(tmp_path, 8, 2)


def test_atomic_json_round_trip(tmp_path: Path):
    from experiment_runner_common import atomic_write_json, read_json

    path = tmp_path / "status.json"
    atomic_write_json(path, {"status": "complete"})
    assert read_json(path)["status"] == "complete"


def test_gpu_assignment_is_round_robin():
    from experiment_runner_common import assign_gpus

    assert assign_gpus(["a", "b", "c"], ["0", "2"]) == [("a", "0"), ("b", "2"), ("c", "0")]


def test_gpu_queues_keep_each_gpu_serial():
    from experiment_runner_common import group_units_by_gpu

    assert group_units_by_gpu(["a", "b", "c"], ["0", "2"]) == {"0": ["a", "c"], "2": ["b"]}
