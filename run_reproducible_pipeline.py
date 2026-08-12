from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from reproducibility import (
    GoldenMismatchError,
    ReproducibilityError,
    RunContext,
    compare_observed_rows,
    read_jsonl,
    semantic_text_label_sha256,
    semantic_training_rows_sha256,
    sha256_file,
    validate_metrics,
    write_json_atomic,
)


PROJECT_ROOT = Path(__file__).resolve().parent
FINAL_TAG = "strict_aug150_w020_label_to_text_gen_complete_multi2_w025_pw065"
RESULT_TAG = f"{FINAL_TAG}_sentiment_contrastive_l001_source_balanced"
MODEL_SNAPSHOT_FILES = (
    "model.safetensors",
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "spiece.model",
    "tokenizer.json",
)


@dataclass(frozen=True)
class Stage:
    name: str
    argv: tuple[str, ...]
    outputs: tuple[Path, ...]
    golden_key: str = ""
    inputs: tuple[Path, ...] = ()


def load_recipe(path: Path) -> dict:
    recipe = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "recipe_id",
        "source_dataset",
        "target_dataset",
        "models",
        "training",
    }
    missing = sorted(required - set(recipe))
    if missing:
        raise ValueError(f"recipe missing required keys: {missing}")
    reuse = recipe.get("reuse", {})
    mode = reuse.get("mode", "full_from_scratch")
    if mode not in {"full_from_scratch", "controlled_stage_reuse"}:
        raise ValueError(f"unsupported reuse mode: {mode}")
    if mode == "controlled_stage_reuse":
        required_reuse = {
            "parent_recipe_id",
            "parent_run_id",
            "parent_recipe_semantic_sha256",
            "through_stage",
        }
        missing_reuse = sorted(required_reuse - set(reuse))
        if missing_reuse:
            raise ValueError(f"controlled reuse config is missing: {missing_reuse}")
        if reuse["through_stage"] != "generator":
            raise ValueError("target-anchor ablations must reuse exactly through generator")
    target_anchor = recipe.get("augment", {}).get("target_anchor_mode")
    if target_anchor is not None:
        if target_anchor != "local_span_edit":
            raise ValueError(f"unsupported target_anchor_mode: {target_anchor}")
        if int(recipe["augment"].get("selection_limit", 0)) <= 0:
            raise ValueError("target-anchor selection_limit must be positive")
        if float(recipe["augment"].get("total_weight_cap", 0)) <= 0:
            raise ValueError("target-anchor total_weight_cap must be positive")
    return recipe


def _model_snapshot_outputs(model_dir: Path) -> tuple[Path, ...]:
    return tuple(model_dir / name for name in MODEL_SNAPSHOT_FILES)


def _fraction_tag(value: float) -> str:
    scaled = round(float(value) * 100)
    return f"{scaled:03d}"


def _target_anchor_tags(recipe: dict) -> dict[str, str]:
    augment = recipe["augment"]
    final = recipe["final"]
    step = int(augment["experiment_step"])
    augment_tag = f"target_anchor_step{step}_v1"
    final_tag = (
        f"{augment_tag}_complete_multi2_w025_pw{_fraction_tag(final['effective_pseudo_weight'])}"
    )
    result_tag = f"{final_tag}_sentiment_contrastive_l001_source_balanced"
    return {"augment": augment_tag, "final": final_tag, "result": result_tag}


def validate_git_state(project_root: Path, allow_dirty: bool) -> tuple[str, str]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=project_root, text=True
    ).strip()
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=project_root, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=project_root, text=True
    ).strip()
    if dirty and not allow_dirty:
        raise ReproducibilityError("formal run requires a clean git worktree")
    return commit, branch


def validate_external_inputs(recipe: dict) -> dict[str, str]:
    validated = {}
    for name, declaration in recipe.get("external_inputs", {}).items():
        path = Path(declaration["path"])
        if not path.is_file():
            raise ReproducibilityError(f"external input is missing for {name}: {path}")
        actual_hash = sha256_file(path)
        expected_hash = declaration["sha256"].upper()
        if actual_hash != expected_hash:
            raise ReproducibilityError(
                f"external input hash mismatch for {name}: "
                f"{actual_hash} != {expected_hash}"
            )
        validated[name] = actual_hash
    return validated


def initialize_recipe_manifest(
    context: RunContext, recipe: dict, recipe_path: Path
) -> None:
    semantic_payload = json.dumps(
        recipe,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    values = {
        "source_dataset": recipe["source_dataset"],
        "target_dataset": recipe["target_dataset"],
        "seed": recipe["seed"],
        "recipe_path": str(Path(recipe_path).resolve()),
        "recipe_sha256": sha256_file(Path(recipe_path)),
        "recipe_semantic_sha256": hashlib.sha256(semantic_payload).hexdigest().upper(),
    }
    for key, value in values.items():
        existing = context.manifest.get(key)
        if existing is not None and existing != value:
            raise ReproducibilityError(f"run recipe identity mismatch for {key}")
        context.manifest[key] = value
    write_json_atomic(context.manifest_path, context.manifest)


def initialize_run_mode_manifest(
    context: RunContext, recipe: dict, *, git_worktree_clean: bool
) -> None:
    mode = recipe.get("reuse", {}).get("mode", "full_from_scratch")
    if mode == "full_from_scratch":
        values = {"run_type": mode, "reuse_depth": 0}
    elif mode == "controlled_stage_reuse":
        values = {"run_type": mode, "reuse_depth": 1}
    else:
        raise ReproducibilityError(f"unsupported reuse mode: {mode}")
    values["git_worktree_clean"] = bool(git_worktree_clean)
    for key, value in values.items():
        existing = context.manifest.get(key)
        if existing is not None and existing != value:
            raise ReproducibilityError(f"run mode manifest mismatch for {key}")
        context.manifest[key] = value
    write_json_atomic(context.manifest_path, context.manifest)


def validate_golden_artifact(stage: Stage, recipe: dict) -> dict | None:
    golden = recipe.get("golden", {})
    if not stage.golden_key or stage.golden_key not in golden:
        return None
    expected = golden[stage.golden_key]
    artifact_path = stage.outputs[-1] if stage.golden_key in {"base_pseudo", "predictions"} else stage.outputs[0]
    if not artifact_path.is_file():
        raise GoldenMismatchError(f"golden artifact is missing: {artifact_path}")

    result = {
        "stage": stage.name,
        "golden_key": stage.golden_key,
        "path": str(artifact_path.resolve()),
    }
    observed_rows = expected.get("observed_golden_rows")
    if observed_rows is not None:
        row_comparison = compare_observed_rows(
            stage.name, read_jsonl(artifact_path), observed_rows
        )
        result["rows"] = row_comparison
        if not row_comparison["matched"]:
            raise GoldenMismatchError(
                f"observed rows mismatch for {stage.name}: "
                f"{row_comparison['actual_rows']} != {observed_rows}"
            )

    expected_hash = expected.get("sha256")
    if expected_hash:
        actual_hash = sha256_file(artifact_path)
        result["sha256"] = actual_hash
        result["sha256_matched"] = actual_hash == expected_hash
        if actual_hash != expected_hash:
            raise GoldenMismatchError(
                f"golden hash mismatch for {stage.name}: "
                f"{actual_hash} != {expected_hash}"
            )

    expected_semantic_hash = expected.get("semantic_sha256")
    if expected_semantic_hash:
        actual_semantic_hash = semantic_text_label_sha256(artifact_path)
        result["semantic_sha256"] = actual_semantic_hash
        result["semantic_sha256_matched"] = (
            actual_semantic_hash == expected_semantic_hash
        )
        if actual_semantic_hash != expected_semantic_hash:
            raise GoldenMismatchError(
                f"golden semantic hash mismatch for {stage.name}: "
                f"{actual_semantic_hash} != {expected_semantic_hash}"
            )

    expected_training_hash = expected.get("training_semantic_sha256")
    if expected_training_hash:
        actual_training_hash = semantic_training_rows_sha256(artifact_path)
        result["training_semantic_sha256"] = actual_training_hash
        result["training_semantic_sha256_matched"] = (
            actual_training_hash == expected_training_hash
        )
        if actual_training_hash != expected_training_hash:
            raise GoldenMismatchError(
                f"golden training semantic hash mismatch for {stage.name}: "
                f"{actual_training_hash} != {expected_training_hash}"
            )

    if stage.name == "evaluate" and "metrics" in recipe:
        raw_metrics = json.loads(stage.outputs[0].read_text(encoding="utf-8"))
        fixed_metrics = json.loads(stage.outputs[1].read_text(encoding="utf-8"))
        validate_metrics(raw_metrics, recipe["metrics"]["raw"])
        validate_metrics(fixed_metrics, recipe["metrics"]["fixed"])
        result["metrics_matched"] = True
    return result


def collect_internal_input_hashes(stage: Stage, run_root: Path) -> dict[str, str]:
    run_root = Path(run_root).resolve()
    output_paths = {path.resolve() for path in stage.outputs}
    output_parents = {path.parent for path in output_paths}
    hashes = {}
    candidates = [Path(path) for path in stage.inputs]
    candidates.extend(Path(token) for token in stage.argv)
    for candidate in candidates:
        if not candidate.exists():
            continue
        resolved = candidate.resolve()
        if (
            not resolved.is_relative_to(run_root)
            or resolved in output_paths
            or resolved in output_parents
        ):
            continue
        hash_target = resolved
        if resolved.is_dir():
            model_weights = resolved / "model.safetensors"
            if not model_weights.is_file():
                continue
            hash_target = model_weights
        if hash_target.is_file():
            hashes[str(resolved)] = sha256_file(hash_target)
    return hashes


def collect_stage_input_hashes(stage: Stage, run_root: Path) -> dict[str, str]:
    hashes = collect_internal_input_hashes(stage, run_root)
    if stage.name == "prepare_final":
        pseudo_source = Path(run_root).resolve() / "target_pseudo.jsonl"
        if pseudo_source.is_file():
            hashes[str(pseudo_source.resolve())] = sha256_file(pseudo_source)
    return hashes


def _normalize_run_path_token(token: str, run_root: Path) -> str:
    text = str(token)
    root_text = str(Path(run_root).resolve())
    if text.casefold() == root_text.casefold():
        return "{run_root}"
    prefix = root_text + os.sep
    if text.casefold().startswith(prefix.casefold()):
        return "{run_root}/" + text[len(prefix):].replace("\\", "/")
    return text


def _normalized_hash_map(values: dict[str, str], run_root: Path) -> dict[str, str]:
    return {
        _normalize_run_path_token(path, run_root): value for path, value in values.items()
    }


def _import_reused_file(source: Path, destination: Path, expected_hash: str) -> str:
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    if destination.is_file():
        if sha256_file(destination) != expected_hash:
            raise ReproducibilityError(f"existing reuse destination hash mismatch: {destination}")
        return "existing_validated"
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        method = "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        method = "copy"
    if sha256_file(destination) != expected_hash:
        raise ReproducibilityError(f"imported artifact hash mismatch: {destination}")
    return method


def import_controlled_stage_reuse(
    *,
    context: RunContext,
    recipe: dict,
    stages: list[Stage],
    output_root: Path,
    current_git_commit: str,
    current_external_hashes: dict[str, str],
    dry_run: bool,
) -> tuple[str, ...]:
    reuse = recipe.get("reuse", {})
    if reuse.get("mode", "full_from_scratch") != "controlled_stage_reuse":
        return ()
    stage_names = [stage.name for stage in stages]
    through_stage = str(reuse["through_stage"])
    if through_stage not in stage_names:
        raise ReproducibilityError(f"reuse through_stage is unknown: {through_stage}")
    reused_stages = stages[: stage_names.index(through_stage) + 1]
    output_root = Path(output_root).resolve()
    parent_root = (
        output_root / str(reuse["parent_recipe_id"]) / str(reuse["parent_run_id"])
    ).resolve()
    if not parent_root.is_relative_to(output_root) or parent_root == context.run_root:
        raise ReproducibilityError(f"unsafe reuse parent path: {parent_root}")
    manifest_path = parent_root / "manifest.json"
    status_path = parent_root / "stage_status.json"
    if not manifest_path.is_file() or not status_path.is_file():
        raise ReproducibilityError(f"reuse parent is missing manifest or stage status: {parent_root}")
    parent = json.loads(manifest_path.read_text(encoding="utf-8"))
    if parent.get("recipe_id") != reuse["parent_recipe_id"]:
        raise ReproducibilityError("reuse parent recipe_id mismatch")
    if parent.get("run_id") != reuse["parent_run_id"]:
        raise ReproducibilityError("reuse parent run_id mismatch")
    if parent.get("recipe_semantic_sha256") != str(
        reuse["parent_recipe_semantic_sha256"]
    ).upper():
        raise ReproducibilityError("reuse parent recipe semantic SHA256 mismatch")
    statuses = json.loads(status_path.read_text(encoding="utf-8"))
    if parent.get("reuse_depth") != 0:
        raise ReproducibilityError("reuse parent must have reuse_depth=0")
    if parent.get("run_type") != "full_from_scratch":
        raise ReproducibilityError("reuse parent must be a full_from_scratch run")
    if parent.get("git_worktree_clean") is not True:
        raise ReproducibilityError("reuse parent must record a clean git worktree")
    if parent.get("git_commit") != current_git_commit:
        raise ReproducibilityError("reuse parent git commit must equal current candidate commit")
    for key in ("source_dataset", "target_dataset", "seed"):
        if parent.get(key) != recipe.get(key):
            raise ReproducibilityError(f"reuse parent {key} mismatch")
    external = parent.get("external_inputs", {})
    if external.get("matched") is not True or external.get("sha256") != current_external_hashes:
        raise ReproducibilityError("reuse parent external input hashes do not match")

    provenance = {
        "schema_version": 1,
        "reuse_depth": 1,
        "parent_run_dir": str(parent_root),
        "parent_recipe_id": parent.get("recipe_id"),
        "parent_run_id": parent.get("run_id"),
        "parent_git_commit": parent.get("git_commit"),
        "through_stage": through_stage,
        "stages": {},
    }
    for stage in reused_stages:
        parent_stage = parent.get("stages", {}).get(stage.name)
        parent_status = statuses.get(stage.name)
        if not parent_stage or parent_stage.get("status") != "completed":
            raise ReproducibilityError(f"reuse parent stage is not complete: {stage.name}")
        if not parent_status or parent_status.get("status") != "completed" or parent_status.get("exit_code") != 0:
            raise ReproducibilityError(f"reuse parent stage status is not complete: {stage.name}")
        parent_argv = [_normalize_run_path_token(token, parent_root) for token in parent_stage.get("argv", [])]
        child_argv = [_normalize_run_path_token(token, context.run_root) for token in stage.argv]
        if parent_argv != child_argv:
            raise ReproducibilityError(f"reused stage command fingerprint mismatch: {stage.name}")
        parent_outputs = [Path(path).resolve() for path in parent_stage.get("outputs", [])]
        if len(parent_outputs) != len(stage.outputs):
            raise ReproducibilityError(f"reused stage output count mismatch: {stage.name}")
        output_records = []
        for source, destination in zip(parent_outputs, stage.outputs):
            if not source.is_relative_to(parent_root):
                raise ReproducibilityError(f"reuse parent output is outside parent run: {source}")
            if source.relative_to(parent_root) != Path(destination).resolve().relative_to(context.run_root):
                raise ReproducibilityError(f"reused stage output layout mismatch: {stage.name}")
            artifact = parent.get("artifacts", {}).get(str(source))
            if artifact is None or not source.is_file():
                raise ReproducibilityError(f"reuse parent artifact is missing: {source}")
            expected = str(artifact.get("sha256", ""))
            if sha256_file(source) != expected:
                raise ReproducibilityError(f"reuse parent artifact hash mismatch: {source}")
            method = "validated_only" if dry_run else _import_reused_file(source, destination, expected)
            output_records.append({"parent_path": str(source), "child_path": str(Path(destination).resolve()), "sha256": expected, "import_method": method})
        if not dry_run:
            child_hashes = collect_stage_input_hashes(stage, context.run_root)
            if _normalized_hash_map(parent_stage.get("input_hashes", {}), parent_root) != _normalized_hash_map(child_hashes, context.run_root):
                raise ReproducibilityError(f"reused stage input hash mismatch: {stage.name}")
            context.mark_stage_complete(stage.name, stage.outputs, child_hashes, stage.argv)
            context.manifest["stages"][stage.name]["reused_from"] = {
                "parent_run_id": parent.get("run_id"),
                "parent_recipe_id": parent.get("recipe_id"),
                "parent_git_commit": parent.get("git_commit"),
                "outputs": output_records,
            }
            context._update_stage_status(stage.name, {"status": "completed", "exit_code": 0, "reused": True, "parent_run_id": parent.get("run_id")})
        provenance["stages"][stage.name] = {"outputs": output_records}
    context.manifest["reuse_parent"] = {
        "run_dir": str(parent_root),
        "recipe_id": parent.get("recipe_id"),
        "run_id": parent.get("run_id"),
        "git_commit": parent.get("git_commit"),
        "through_stage": through_stage,
        "validation": "dry_run" if dry_run else "imported_and_validated",
    }
    write_json_atomic(context.manifest_path, context.manifest)
    if not dry_run:
        provenance_path = context.run_root / "reuse_provenance.json"
        write_json_atomic(provenance_path, provenance)
        context.record_artifact("reuse_import", provenance_path)
    return tuple(stage.name for stage in reused_stages)


def execute_stages(
    stages: list[Stage],
    context: RunContext,
    recipe: dict,
    project_root: Path,
    dry_run: bool,
) -> None:
    total = len(stages)
    for index, stage in enumerate(stages, start=1):
        input_hashes = collect_stage_input_hashes(stage, context.run_root)

        if stage.name in context.manifest.get("stages", {}):
            if context.validate_completed_stage(stage.name, stage.outputs, input_hashes, stage.argv):
                validate_golden_artifact(stage, recipe)
                print(
                    f"[native-repro] SKIP {index}/{total} {stage.name} "
                    "(validated checkpoint)",
                    flush=True,
                )
                continue

        print(f"[native-repro] START {index}/{total} {stage.name}", flush=True)
        print(subprocess.list2cmdline(stage.argv), flush=True)
        context.run_command(
            stage.name,
            list(stage.argv),
            cwd=project_root,
            dry_run=dry_run,
        )
        if dry_run:
            print(f"[native-repro] DONE {index}/{total} {stage.name}", flush=True)
            continue

        if stage.name == "prepare_final":
            source = context.run_root / "target_pseudo.jsonl"
            destination = context.run_root / "final_data" / "target_pseudo.jsonl"
            context.require_internal_artifact(source)
            context.require_internal_artifact(destination)
            shutil.copy2(source, destination)

        context.mark_stage_complete(stage.name, stage.outputs, input_hashes, stage.argv)
        try:
            golden_result = validate_golden_artifact(stage, recipe)
        except GoldenMismatchError as error:
            context.manifest.setdefault("golden_comparisons", {})[stage.name] = {
                "matched": False,
                "error": str(error),
            }
            write_json_atomic(context.manifest_path, context.manifest)
            context.render_run_record_cn()
            raise
        if golden_result is not None:
            context.manifest.setdefault("golden_comparisons", {})[stage.name] = {
                "matched": True,
                **golden_result,
            }
            write_json_atomic(context.manifest_path, context.manifest)
        print(f"[native-repro] DONE {index}/{total} {stage.name}", flush=True)


def _command(python: Path, script: Path, *arguments: str) -> tuple[str, ...]:
    return (str(python), str(script), *(str(argument) for argument in arguments))


def build_best_v1_stages(
    project_root: Path,
    run_root: Path,
    recipe: dict,
    python_executable: Path,
    cuda: str,
) -> list[Stage]:
    project_root = Path(project_root).resolve()
    run_root = Path(run_root).resolve()
    final_data = run_root / "final_data"
    pipeline = project_root / "t5_aste_pipeline.py"
    trainer = project_root / "t5_absa_train.py"
    model = recipe["models"]["t5_base"]
    nli_model = recipe["models"]["nli"]
    seed = str(recipe["seed"])
    training = recipe["training"]
    pseudo = recipe["pseudo"]
    augment = recipe["augment"]
    complete = recipe["complete_multi"]
    final = recipe["final"]
    target_anchor_enabled = augment.get("target_anchor_mode") == "local_span_edit"
    tags = (
        _target_anchor_tags(recipe)
        if target_anchor_enabled
        else {"augment": "strict_aug150_w020_label_to_text_gen", "final": FINAL_TAG, "result": RESULT_TAG}
    )

    extractor_dir = run_root / "models" / "extractor_ep25_plain_last"
    extractor = extractor_dir / "best"
    generator_dir = run_root / "models" / "generator_label_to_text_gen_ep8"
    generator = generator_dir / "best"
    selected_augment = (
        run_root / f"target_anchored_selected_{tags['augment']}.jsonl"
        if target_anchor_enabled
        else run_root / "c3da_two_channel_augmented_selected_strict_aug150_w020_label_to_text_gen.jsonl"
    )
    complete_dir = final_data / "pseudo_variants" / "hp1_complete2_dist5_w025"
    complete_pseudo = complete_dir / "target_pseudo_high_precision.jsonl"
    final_train = final_data / f"final_train_{tags['final']}.jsonl"
    final_dev = final_data / f"final_dev_{tags['final']}.jsonl"
    final_model_dir = (
        final_data
        / "models"
        / f"final_dann_l0.03_{tags['result']}_ep5"
    )
    final_model = final_model_dir / "best"

    common_train = (
        "--per_device_train_batch_size",
        str(training["train_batch_size"]),
        "--per_device_eval_batch_size",
        str(training["eval_batch_size"]),
        "--gradient_accumulation_steps",
        str(training["gradient_accumulation_steps"]),
        "--learning_rate",
        str(training["learning_rate"]),
        "--fp16",
        "--gradient_checkpointing",
        "--cuda",
        str(cuda),
        "--seed",
        seed,
    )

    return [
        Stage(
            "prepare",
            _command(
                python_executable,
                pipeline,
                "prepare",
                "--source_dataset",
                recipe["source_dataset"],
                "--target_dataset",
                recipe["target_dataset"],
                "--run_dir",
                str(run_root),
                "--seed",
                seed,
                "--augment_prompt_style",
                "label_to_text",
                "--augment_channel_mode",
                "all",
                "--domain_prefix_style",
                "text",
                "--generator_output_tag",
                "label_to_text_gen",
                "--no_task_prefix",
            ),
            (
                run_root / "source_train.jsonl",
                run_root / "source_dev.jsonl",
                run_root / "extract_train.jsonl",
                run_root / "extract_dev.jsonl",
                run_root / "c3da_generator_train_label_to_text_gen.jsonl",
                run_root / "c3da_generator_dev_label_to_text_gen.jsonl",
                run_root / "target_unlabeled.jsonl",
                run_root / "target_test.jsonl",
            ),
        ),
        Stage(
            "extractor",
            _command(
                python_executable,
                trainer,
                "--model_path",
                model,
                "--train_file",
                str(run_root / "extract_train.jsonl"),
                "--dev_file",
                str(run_root / "extract_dev.jsonl"),
                "--output_dir",
                str(extractor_dir),
                "--num_train_epochs",
                str(training["extractor_epochs"]),
                "--source_weight",
                "1.0",
                "--pseudo_weight",
                "0.5",
                "--augment_weight",
                "0.2",
                "--lambda_structure_loss",
                "0",
                "--lambda_consistency_loss",
                "0",
                "--lambda_pairing_loss",
                "0",
                "--multi_triplet_loss_gain",
                "0",
                "--neutral_loss_gain",
                "0",
                "--checkpoint_selection",
                training["extractor_checkpoint_selection"],
                "--resume_from_checkpoint",
                "auto",
                *common_train,
            ),
            _model_snapshot_outputs(extractor),
            "extractor",
            (
                run_root / "extract_train.jsonl",
                run_root / "extract_dev.jsonl",
            ),
        ),
        Stage(
            "pseudo",
            _command(
                python_executable,
                pipeline,
                "pseudo",
                "--run_dir",
                str(run_root),
                "--model_path",
                str(extractor),
                "--batch_size",
                str(training["eval_batch_size"]),
                "--num_beams",
                str(pseudo["num_beams"]),
                "--max_new_tokens",
                str(pseudo["max_new_tokens"]),
                "--no_constrained_decoding",
                "--cuda",
                str(cuda),
                "--no_task_prefix",
                "--pseudo_model_variant",
                "last",
                "--high_precision_max_triplets",
                str(pseudo["high_precision_max_triplets"]),
                "--high_precision_max_token_distance",
                str(pseudo["high_precision_max_token_distance"]),
            ),
            (
                run_root / "target_pseudo.jsonl",
                run_root / "target_pseudo_selected.jsonl",
                run_root / "target_pseudo_high_precision.jsonl",
            ),
            "base_pseudo",
            (
                run_root / "target_unlabeled.jsonl",
                extractor / "model.safetensors",
            ),
        ),
        Stage(
            "generator",
            _command(
                python_executable,
                trainer,
                "--model_path",
                model,
                "--train_file",
                str(run_root / "c3da_generator_train_label_to_text_gen.jsonl"),
                "--dev_file",
                str(run_root / "c3da_generator_dev_label_to_text_gen.jsonl"),
                "--output_dir",
                str(generator_dir),
                "--num_train_epochs",
                str(training["generator_epochs"]),
                "--source_weight",
                "1.0",
                "--pseudo_weight",
                "1.0",
                "--augment_weight",
                "1.0",
                "--checkpoint_selection",
                training["generator_checkpoint_selection"],
                "--resume_from_checkpoint",
                "auto",
                *common_train,
            ),
            _model_snapshot_outputs(generator),
            "generator",
            (
                run_root / "c3da_generator_train_label_to_text_gen.jsonl",
                run_root / "c3da_generator_dev_label_to_text_gen.jsonl",
            ),
        ),
        Stage(
            "augment",
            (
                _command(
                    python_executable,
                    project_root / "target_anchored_augment.py",
                    "--run_dir",
                    str(run_root),
                    "--extractor_model_path",
                    str(extractor),
                    "--nli_model_path",
                    nli_model,
                    "--output_tag",
                    tags["augment"],
                    "--selection_limit",
                    str(augment["selection_limit"]),
                    "--max_per_base",
                    str(augment["max_per_base"]),
                    "--per_anchor",
                    str(augment["per_anchor"]),
                    "--seed",
                    seed,
                    "--batch_size",
                    str(training["eval_batch_size"]),
                    "--max_new_tokens",
                    "96",
                    "--model_filter_mode",
                    str(augment["model_filter_mode"]),
                    "--base_weight",
                    str(augment["sample_weight"]),
                    "--total_weight_cap",
                    str(augment["total_weight_cap"]),
                    "--min_selected_rows",
                    str(augment["min_selected_rows"]),
                    *(
                        (
                            "--gap_aware_selection",
                            "--aspect_ratio_min",
                            str(augment["aspect_ratio_min"]),
                            "--aspect_ratio_max",
                            str(augment["aspect_ratio_max"]),
                            "--opinion_ratio_min",
                            str(augment["opinion_ratio_min"]),
                            "--opinion_ratio_max",
                            str(augment["opinion_ratio_max"]),
                            "--neutral_max_ratio",
                            str(augment["neutral_max_ratio"]),
                            "--multi_triplet_target_ratio",
                            str(augment["multi_triplet_target_ratio"]),
                        )
                        if augment.get("gap_aware_selection") is True
                        else ()
                    ),
                    *(
                        (
                            "--tiered_weights",
                            "--high_weight",
                            str(augment["high_weight"]),
                            "--medium_weight",
                            str(augment["medium_weight"]),
                            "--neutral_high_weight",
                            str(augment["neutral_high_weight"]),
                            "--neutral_medium_weight",
                            str(augment["neutral_medium_weight"]),
                        )
                        if augment.get("tiered_weights") is True
                        else ()
                    ),
                    "--cuda",
                    str(cuda),
                )
                if target_anchor_enabled
                else _command(
                    python_executable,
                    pipeline,
                    "augment",
                "--run_dir",
                str(run_root),
                "--model_path",
                str(generator),
                "--nli_model_path",
                nli_model,
                "--augment_prompt_style",
                augment["prompt_style"],
                "--augment_channel_mode",
                augment["channel_mode"],
                "--domain_prefix_style",
                augment["domain_prefix_style"],
                "--compatibility_profile",
                augment["compatibility_profile"],
                "--augment_output_tag",
                "strict_aug150_w020_label_to_text_gen",
                "--final_train_output_tag",
                "strict_aug150_w020_label_to_text_gen",
                "--augment_select_max_rows",
                str(augment["selection_limit"]),
                "--augment_select_max_per_base",
                str(augment["max_per_base"]),
                "--augment_select_weight",
                str(augment["sample_weight"]),
                "--augment_select_require_raw_exact",
                "--augment_select_require_model_filter_passed",
                "--pseudo_train_source",
                "high_precision",
                "--high_precision_max_triplets",
                str(pseudo["high_precision_max_triplets"]),
                "--high_precision_max_token_distance",
                str(pseudo["high_precision_max_token_distance"]),
                "--model_filter_path",
                str(extractor),
                "--model_filter_mode",
                "fixed",
                "--model_filter_batch_size",
                str(training["eval_batch_size"]),
                "--model_filter_num_beams",
                "1",
                "--model_filter_no_constrained_decoding",
                "--model_filter_channel_aware",
                "--cuda",
                str(cuda),
                "--no_task_prefix",
                )
            ),
            (
                (
                    selected_augment,
                    run_root / f"target_anchored_analysis_{tags['augment']}.json",
                    run_root / f"target_anchored_{tags['augment']}_candidates.jsonl",
                    run_root / f"target_anchored_{tags['augment']}_nli.jsonl",
                    run_root / f"target_anchored_{tags['augment']}_model_kept.jsonl",
                    run_root / f"target_anchored_{tags['augment']}_model_removed.jsonl",
                    run_root / f"target_anchored_{tags['augment']}_exploratory_audit.jsonl",
                )
                if target_anchor_enabled
                else (selected_augment,)
            ),
            "" if target_anchor_enabled else "augment",
            (
                run_root / "source_train.jsonl",
                run_root / "source_dev.jsonl",
                run_root / "target_pseudo_selected.jsonl",
                extractor / "model.safetensors",
            ),
        ),
        Stage(
            "prepare_final",
            _command(
                python_executable,
                pipeline,
                "prepare",
                "--source_dataset",
                recipe["source_dataset"],
                "--target_dataset",
                recipe["target_dataset"],
                "--run_dir",
                str(final_data),
                "--seed",
                seed,
                "--augment_prompt_style",
                "label_to_text",
                "--augment_channel_mode",
                "all",
                "--domain_prefix_style",
                "text",
                "--generator_output_tag",
                "label_to_text_gen",
                "--no_task_prefix",
            ),
            (
                final_data / "source_train.jsonl",
                final_data / "source_dev.jsonl",
                final_data / "target_test.jsonl",
                final_data / "target_pseudo.jsonl",
            ),
            inputs=(run_root / "target_pseudo.jsonl",),
        ),
        Stage(
            "complete_multi2",
            _command(
                python_executable,
                pipeline,
                "select_complete_multi_pseudo",
                "--run_dir",
                str(final_data),
                "--output_dir",
                str(complete_dir),
                "--base_pseudo_file",
                str(run_root / "target_pseudo_high_precision.jsonl"),
                "--min_pseudo_weight",
                str(final["pseudo_weight"]),
                "--high_precision_max_token_distance",
                str(complete["max_token_distance"]),
                "--complete_multi_extra_weight",
                str(complete["extra_weight"]),
            ),
            (
                complete_pseudo,
                complete_dir / "target_pseudo_high_precision_analysis.json",
            ),
            "complete_pseudo",
            (
                final_data / "target_pseudo.jsonl",
                run_root / "target_pseudo_high_precision.jsonl",
            ),
        ),
        Stage(
            "build_final_train",
            _command(
                python_executable,
                pipeline,
                "build_final_train_from_files",
                "--run_dir",
                str(final_data),
                "--pseudo_train_file",
                str(complete_pseudo),
                "--selected_augment_file",
                str(selected_augment),
                "--selected_augment_weight",
                "0.0" if target_anchor_enabled else str(final["augment_weight"]),
                "--final_train_output_tag",
                tags["final"],
                "--no_task_prefix",
            ),
            (final_train, final_dev),
            "final_train",
            (
                final_data / "source_train.jsonl",
                final_data / "source_dev.jsonl",
                complete_pseudo,
                selected_augment,
            ),
        ),
        Stage(
            "final_train",
            _command(
                python_executable,
                trainer,
                "--model_path",
                model,
                "--train_file",
                str(final_train),
                "--dev_file",
                str(final_dev),
                "--output_dir",
                str(final_model_dir),
                "--num_train_epochs",
                str(training["final_epochs"]),
                "--source_weight",
                "1.0",
                "--pseudo_weight",
                str(final["pseudo_weight"]),
                *(
                    ("--pseudo_weight_scale", str(final["pseudo_weight_scale"]))
                    if "pseudo_weight_scale" in final
                    else ()
                ),
                "--augment_weight",
                str(final["augment_weight"]),
                "--checkpoint_selection",
                training["final_checkpoint_selection"],
                "--resume_from_checkpoint",
                "auto",
                "--lambda_domain_adv",
                str(final["lambda_domain_adv"]),
                "--domain_adv_grl_lambda",
                "1.0",
                "--domain_adv_hidden_size",
                "256",
                "--domain_adv_exclude_augment",
                "--lambda_sentiment_contrastive",
                str(final["lambda_sentiment_contrastive"]),
                "--lambda_pairing_loss",
                "0.0",
                "--pairing_temperature",
                "0.1",
                "--sentiment_contrastive_temperature",
                "0.1",
                "--sentiment_contrastive_min_weight",
                str(final["pseudo_weight"]),
                "--neutral_generation_loss_gain",
                "0.0",
                "--neutral_generation_max_effective_weight",
                "0.0",
                "--sentiment_contrastive_source_only",
                "--sentiment_contrastive_class_balanced",
                *common_train,
            ),
            _model_snapshot_outputs(final_model),
            "final_model",
            (final_train, final_dev),
        ),
        Stage(
            "evaluate",
            _command(
                python_executable,
                pipeline,
                "evaluate",
                "--run_dir",
                str(final_data),
                "--model_path",
                str(final_model),
                "--batch_size",
                str(training["eval_batch_size"]),
                "--num_beams",
                "4",
                "--max_new_tokens",
                "96",
                "--cuda",
                str(cuda),
                "--no_task_prefix",
                "--no_constrained_decoding",
                "--output_tag",
                tags["result"],
            ),
            (
                final_data / f"aste_metrics_raw_{tags['result']}.json",
                final_data / f"aste_metrics_fixed_{tags['result']}.json",
                final_data / f"aste_predictions_raw_fixed_{tags['result']}.jsonl",
            ),
            "predictions",
            (
                final_data / "target_test.jsonl",
                final_model / "model.safetensors",
            ),
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--cuda", default="0")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--allow_dirty", action="store_true")
    parser.add_argument("--user_command", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recipe = load_recipe(Path(args.recipe))
    commit, branch = validate_git_state(PROJECT_ROOT, args.allow_dirty)
    run_root = (
        Path(args.output_root).resolve()
        / recipe["recipe_id"]
        / args.run_id
    )
    context = RunContext.open_or_create(
        run_root,
        args.run_id,
        recipe["recipe_id"],
        commit,
        branch,
    )
    initialize_recipe_manifest(context, recipe, Path(args.recipe))
    initialize_run_mode_manifest(
        context,
        recipe,
        git_worktree_clean=not args.allow_dirty,
    )
    user_command = args.user_command
    if user_command.startswith("base64:"):
        user_command = base64.b64decode(user_command.removeprefix("base64:")).decode(
            "utf-8"
        )
    if not user_command:
        user_command = subprocess.list2cmdline([sys.executable, *sys.argv])
    context.write_user_command(user_command)
    if not args.dry_run:
        try:
            external_hashes = validate_external_inputs(recipe)
        except ReproducibilityError as error:
            context.manifest["external_inputs"] = {
                "matched": False,
                "error": str(error),
            }
            write_json_atomic(context.manifest_path, context.manifest)
            context.render_run_record_cn()
            raise
        context.manifest["external_inputs"] = {
            "matched": True,
            "sha256": external_hashes,
        }
        write_json_atomic(context.manifest_path, context.manifest)
        model_paths = [
            Path(recipe["external_inputs"][name]["path"])
            for name in ("t5_weights", "nli_weights")
            if name in recipe.get("external_inputs", {})
        ]
        context.capture_environment(sys.executable, model_paths)
    else:
        external_hashes = {
            name: str(declaration["sha256"]).upper()
            for name, declaration in recipe.get("external_inputs", {}).items()
        }
    stages = build_best_v1_stages(
        PROJECT_ROOT, run_root, recipe, Path(sys.executable), args.cuda
    )
    reused_stage_names = import_controlled_stage_reuse(
        context=context,
        recipe=recipe,
        stages=stages,
        output_root=Path(args.output_root),
        current_git_commit=commit,
        current_external_hashes=external_hashes,
        dry_run=args.dry_run,
    )
    execution_stages = stages
    if reused_stage_names:
        reused = set(reused_stage_names)
        execution_stages = [stage for stage in stages if stage.name not in reused]
    execute_stages(execution_stages, context, recipe, PROJECT_ROOT, args.dry_run)
    context.render_run_record_cn()


if __name__ == "__main__":
    main()
