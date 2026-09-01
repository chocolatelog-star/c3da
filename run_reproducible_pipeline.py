from __future__ import annotations

import argparse
import base64
import json
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
    canonical_json_sha256,
    resolve_project_path,
    validate_metrics,
    write_json_atomic,
)


PROJECT_ROOT = Path(__file__).resolve().parent
FINAL_TAG = "strict_aug150_w020_label_to_text_gen_complete_multi2_w025_pw065"
RESULT_TAG = f"{FINAL_TAG}_sentiment_contrastive_l001_source_balanced"


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
    return recipe


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


def validate_external_inputs(recipe: dict, project_root: Path | None = None) -> dict[str, str]:
    project_root = Path(project_root or PROJECT_ROOT)
    validated = {}
    for name, declaration in recipe.get("external_inputs", {}).items():
        path = resolve_project_path(declaration["path"], project_root, env_var="C3DA_MODEL_ROOT" if "model" in name else None)
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


def resolve_recipe_paths(recipe: dict, project_root: Path) -> dict:
    resolved = json.loads(json.dumps(recipe))
    for declaration in resolved.get("external_inputs", {}).values():
        declaration["path"] = str(resolve_project_path(declaration["path"], project_root, env_var="C3DA_MODEL_ROOT" if "model" in declaration.get("path", "").lower() else None))
    for key, value in resolved.get("models", {}).items():
        resolved["models"][key] = str(resolve_project_path(value, project_root, env_var="C3DA_MODEL_ROOT"))
    return resolved


def apply_cli_overrides(recipe: dict, args: argparse.Namespace) -> dict:
    resolved = json.loads(json.dumps(recipe))
    training = resolved.setdefault("training", {})
    mapping = {
        "train_batch_size": "train_batch_size",
        "eval_batch_size": "eval_batch_size",
        "gradient_accumulation_steps": "gradient_accumulation_steps",
    }
    for argument, key in mapping.items():
        value = getattr(args, argument, None)
        if value is not None:
            training[key] = value
    if "train_batch_size" not in training and "extractor_train_batch_size" in training:
        training["train_batch_size"] = training["extractor_train_batch_size"]
    if "eval_batch_size" not in training and "extractor_eval_batch_size" in training:
        training["eval_batch_size"] = training["extractor_eval_batch_size"]
    if "train_batch_size" not in training:
        raise ValueError("recipe training.train_batch_size is required")
    effective = training["train_batch_size"] * training.get("gradient_accumulation_steps", 1)
    training["effective_batch_size"] = effective
    return resolved


def ensure_resolved_config(context: RunContext, recipe: dict) -> str:
    config_hash = canonical_json_sha256(recipe)
    existing_hash = context.manifest.get("resolved_config_sha256")
    if existing_hash and existing_hash != config_hash:
        raise ReproducibilityError("resolved configuration changed; refusing resume")
    context.manifest["resolved_config_sha256"] = config_hash
    context.manifest["resolved_config"] = recipe
    write_json_atomic(context.manifest_path, context.manifest)
    return config_hash


def select_stages(stages: list[Stage], stop_after: str | None = None) -> list[Stage]:
    """Return a prefix of the recipe stages for upstream-only screening."""
    if not stop_after:
        return stages
    names = [stage.name for stage in stages]
    if stop_after not in names:
        raise ValueError(f"unknown stop stage: {stop_after}; choices: {names}")
    return stages[: names.index(stop_after) + 1]


def initialize_recipe_manifest(
    context: RunContext, recipe: dict, recipe_path: Path
) -> None:
    context.manifest.update(
        {
            "source_dataset": recipe["source_dataset"],
            "target_dataset": recipe["target_dataset"],
            "seed": recipe["seed"],
            "recipe_path": str(Path(recipe_path).resolve()),
            "recipe_sha256": sha256_file(Path(recipe_path)),
        }
    )
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
    hashes = {}
    candidates = [Path(path) for path in stage.inputs]
    candidates.extend(Path(token) for token in stage.argv)
    for candidate in candidates:
        if not candidate.exists():
            continue
        resolved = candidate.resolve()
        if not resolved.is_relative_to(run_root) or resolved in output_paths:
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


def execute_stages(
    stages: list[Stage],
    context: RunContext,
    recipe: dict,
    project_root: Path,
    dry_run: bool,
    strict_golden_validation: bool = False,
) -> None:
    total = len(stages)
    for index, stage in enumerate(stages, start=1):
        input_hashes = collect_internal_input_hashes(stage, context.run_root)
        if stage.name == "prepare_final":
            pseudo_source = context.run_root / "target_pseudo.jsonl"
            if pseudo_source.is_file():
                input_hashes[str(pseudo_source.resolve())] = sha256_file(pseudo_source)

        if stage.name in context.manifest.get("stages", {}):
            if context.validate_completed_stage(stage.name, stage.outputs, input_hashes):
                try:
                    validate_golden_artifact(stage, recipe)
                except GoldenMismatchError as error:
                    if strict_golden_validation:
                        raise
                    context.manifest.setdefault("golden_comparisons", {})[stage.name] = {"matched": False, "error": str(error)}
                    write_json_atomic(context.manifest_path, context.manifest)
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

        context.mark_stage_complete(stage.name, stage.outputs, input_hashes)
        try:
            golden_result = validate_golden_artifact(stage, recipe)
        except GoldenMismatchError as error:
            context.manifest.setdefault("golden_comparisons", {})[stage.name] = {
                "matched": False,
                "error": str(error),
            }
            write_json_atomic(context.manifest_path, context.manifest)
            context.render_run_record_cn()
            if strict_golden_validation:
                raise
            print(f"[native-repro] GOLDEN MISMATCH RECORDED {stage.name}: {error}", flush=True)
            continue
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

    extractor_dir = run_root / "models" / "extractor_ep25_plain_last"
    extractor = extractor_dir / "best"
    generator_dir = run_root / "models" / "generator_label_to_text_gen_ep8"
    generator = generator_dir / "best"
    selected_augment = (
        run_root
        / "c3da_two_channel_augmented_selected_strict_aug150_w020_label_to_text_gen.jsonl"
    )
    complete_dir = final_data / "pseudo_variants" / "hp1_complete2_dist5_w025"
    complete_pseudo = complete_dir / "target_pseudo_high_precision.jsonl"
    final_train = final_data / f"final_train_{FINAL_TAG}.jsonl"
    final_dev = final_data / f"final_dev_{FINAL_TAG}.jsonl"
    final_model_dir = (
        final_data
        / "models"
        / f"final_dann_l0.03_{RESULT_TAG}_ep5"
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
            (extractor / "model.safetensors",),
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
            (generator / "model.safetensors",),
            "generator",
            (
                run_root / "c3da_generator_train_label_to_text_gen.jsonl",
                run_root / "c3da_generator_dev_label_to_text_gen.jsonl",
            ),
        ),
        Stage(
            "augment",
            _command(
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
            ),
            (selected_augment,),
            "augment",
            (
                run_root / "source_train.jsonl",
                run_root / "source_dev.jsonl",
                run_root / "target_pseudo_selected.jsonl",
                generator / "model.safetensors",
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
                str(final["augment_weight"]),
                "--final_train_output_tag",
                FINAL_TAG,
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
            (final_model / "model.safetensors",),
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
                RESULT_TAG,
            ),
            (
                final_data / f"aste_metrics_raw_{RESULT_TAG}.json",
                final_data / f"aste_metrics_fixed_{RESULT_TAG}.json",
                final_data / f"aste_predictions_raw_fixed_{RESULT_TAG}.jsonl",
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
    parser.add_argument("--train_batch_size", type=int)
    parser.add_argument("--eval_batch_size", type=int)
    parser.add_argument("--gradient_accumulation_steps", type=int)
    parser.add_argument("--stop_after_stage", choices=["prepare", "extractor", "pseudo", "generator", "augment", "prepare_final", "complete_multi2", "build_final_train", "final_train", "evaluate"])
    parser.add_argument("--strict_golden_validation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recipe_path = Path(args.recipe).resolve()
    recipe = load_recipe(recipe_path)
    recipe = apply_cli_overrides(recipe, args)
    recipe = resolve_recipe_paths(recipe, PROJECT_ROOT)
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
    initialize_recipe_manifest(context, recipe, recipe_path)
    ensure_resolved_config(context, recipe)
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
            external_hashes = validate_external_inputs(recipe, PROJECT_ROOT)
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
    stages = build_best_v1_stages(
        PROJECT_ROOT, run_root, recipe, Path(sys.executable), args.cuda
    )
    stages = select_stages(stages, args.stop_after_stage)
    context.manifest["execution_policy"] = {
        "stop_after_stage": args.stop_after_stage,
        "strict_golden_validation": args.strict_golden_validation,
    }
    write_json_atomic(context.manifest_path, context.manifest)
    execute_stages(stages, context, recipe, PROJECT_ROOT, args.dry_run, args.strict_golden_validation)
    context.render_run_record_cn()


if __name__ == "__main__":
    main()
