import tempfile
import unittest
from pathlib import Path

from reproducibility import ReproducibilityError, RunContext, write_json_atomic
from run_reproducible_pipeline import (
    Stage,
    collect_stage_input_hashes,
    import_controlled_stage_reuse,
    initialize_run_mode_manifest,
)


class TargetAnchorStageReuseTest(unittest.TestCase):
    def _context(self, root: Path, run_id: str, recipe_id: str) -> RunContext:
        return RunContext.open_or_create(
            root,
            run_id,
            recipe_id,
            "candidate123",
            "feature/target-anchored-augmentation-v1",
        )

    @staticmethod
    def _stages(root: Path, changed_generator_command: bool = False) -> list[Stage]:
        prepared = root / "prepared.jsonl"
        model_dir = root / "models" / "generator" / "best"
        command = ["python", "train.py", "--run_dir", str(root)]
        if changed_generator_command:
            command.append("--changed")
        return [
            Stage(
                "prepare",
                ("python", "prepare.py", "--run_dir", str(root)),
                (prepared,),
            ),
            Stage(
                "generator",
                tuple(command),
                (model_dir / "model.safetensors", model_dir / "config.json"),
                inputs=(prepared,),
            ),
            Stage(
                "augment",
                ("python", "augment.py", "--run_dir", str(root)),
                (root / "augment.jsonl",),
                inputs=(model_dir / "model.safetensors", model_dir / "config.json"),
            ),
        ]

    def _seal_parent(self, output_root: Path, reuse_depth: int = 0):
        root = output_root / "step1" / "parent"
        context = self._context(root, "parent", "step1")
        context.manifest.update(
            {
                "source_dataset": "laptop14",
                "target_dataset": "rest15",
                "seed": 1000,
                "reuse_depth": reuse_depth,
                "run_type": "full_from_scratch" if reuse_depth == 0 else "controlled_stage_reuse",
                "git_worktree_clean": True,
                "recipe_semantic_sha256": "B" * 64,
                "external_inputs": {"matched": True, "sha256": {"data": "A" * 64}},
            }
        )
        write_json_atomic(context.manifest_path, context.manifest)
        stages = self._stages(root)
        for stage in stages[:2]:
            for output in stage.outputs:
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(output.name, encoding="utf-8")
            hashes = collect_stage_input_hashes(stage, root)
            context.mark_stage_complete(stage.name, stage.outputs, hashes, stage.argv)
            context._update_stage_status(stage.name, {"status": "completed", "exit_code": 0})
        return root, stages

    @staticmethod
    def _recipe():
        return {
            "source_dataset": "laptop14",
            "target_dataset": "rest15",
            "seed": 1000,
            "reuse": {
                "mode": "controlled_stage_reuse",
                "parent_recipe_id": "step1",
                "parent_run_id": "parent",
                "through_stage": "generator",
                "parent_recipe_semantic_sha256": "B" * 64,
            },
        }

    def test_imports_every_declared_model_file_and_records_depth_one(self):
        with tempfile.TemporaryDirectory() as temp:
            output_root = Path(temp) / "runs"
            self._seal_parent(output_root)
            child_root = output_root / "step2" / "child"
            child = self._context(child_root, "child", "step2")
            initialize_run_mode_manifest(child, self._recipe(), git_worktree_clean=True)

            reused = import_controlled_stage_reuse(
                context=child,
                recipe=self._recipe(),
                stages=self._stages(child_root),
                output_root=output_root,
                current_git_commit="candidate123",
                current_external_hashes={"data": "A" * 64},
                dry_run=False,
            )

            self.assertEqual(reused, ("prepare", "generator"))
            self.assertTrue((child_root / "models" / "generator" / "best" / "model.safetensors").is_file())
            self.assertTrue((child_root / "models" / "generator" / "best" / "config.json").is_file())
            self.assertEqual(child.manifest["reuse_depth"], 1)
            self.assertEqual(child.manifest["reuse_parent"]["run_id"], "parent")

    def test_rejects_missing_auxiliary_model_file(self):
        with tempfile.TemporaryDirectory() as temp:
            output_root = Path(temp) / "runs"
            parent_root, _stages = self._seal_parent(output_root)
            (parent_root / "models" / "generator" / "best" / "config.json").unlink()
            child_root = output_root / "step2" / "child"
            child = self._context(child_root, "child", "step2")
            initialize_run_mode_manifest(child, self._recipe(), git_worktree_clean=True)
            with self.assertRaisesRegex(ReproducibilityError, "missing"):
                import_controlled_stage_reuse(
                    context=child,
                    recipe=self._recipe(),
                    stages=self._stages(child_root),
                    output_root=output_root,
                    current_git_commit="candidate123",
                    current_external_hashes={"data": "A" * 64},
                    dry_run=False,
                )

    def test_rejects_reuse_chain_and_changed_upstream_command(self):
        with tempfile.TemporaryDirectory() as temp:
            output_root = Path(temp) / "runs"
            self._seal_parent(output_root, reuse_depth=1)
            child_root = output_root / "step2" / "child"
            child = self._context(child_root, "child", "step2")
            initialize_run_mode_manifest(child, self._recipe(), git_worktree_clean=True)
            with self.assertRaisesRegex(ReproducibilityError, "reuse_depth=0"):
                import_controlled_stage_reuse(
                    context=child,
                    recipe=self._recipe(),
                    stages=self._stages(child_root),
                    output_root=output_root,
                    current_git_commit="candidate123",
                    current_external_hashes={"data": "A" * 64},
                    dry_run=False,
                )

    def test_rejects_parent_identity_and_recipe_hash_mismatch(self):
        for changed_key, changed_value, expected_error in (
            ("run_id", "wrong-parent", "run_id"),
            ("recipe_id", "wrong-step", "recipe_id"),
            ("recipe_semantic_sha256", "C" * 64, "recipe semantic SHA256"),
        ):
            with self.subTest(changed_key=changed_key), tempfile.TemporaryDirectory() as temp:
                output_root = Path(temp) / "runs"
                parent_root, _stages = self._seal_parent(output_root)
                manifest_path = parent_root / "manifest.json"
                manifest = __import__("json").loads(manifest_path.read_text(encoding="utf-8"))
                manifest[changed_key] = changed_value
                write_json_atomic(manifest_path, manifest)
                child_root = output_root / "step2" / "child"
                child = self._context(child_root, "child", "step2")
                initialize_run_mode_manifest(child, self._recipe(), git_worktree_clean=True)
                with self.assertRaisesRegex(ReproducibilityError, expected_error):
                    import_controlled_stage_reuse(
                        context=child,
                        recipe=self._recipe(),
                        stages=self._stages(child_root),
                        output_root=output_root,
                        current_git_commit="candidate123",
                        current_external_hashes={"data": "A" * 64},
                        dry_run=False,
                    )

    def test_rejects_changed_upstream_command(self):
        with tempfile.TemporaryDirectory() as temp:
            output_root = Path(temp) / "runs"
            self._seal_parent(output_root)
            child_root = output_root / "step2" / "child"
            child = self._context(child_root, "child", "step2")
            initialize_run_mode_manifest(child, self._recipe(), git_worktree_clean=True)
            with self.assertRaisesRegex(ReproducibilityError, "command fingerprint"):
                import_controlled_stage_reuse(
                    context=child,
                    recipe=self._recipe(),
                    stages=self._stages(child_root, changed_generator_command=True),
                    output_root=output_root,
                    current_git_commit="candidate123",
                    current_external_hashes={"data": "A" * 64},
                    dry_run=False,
                )


if __name__ == "__main__":
    unittest.main()
