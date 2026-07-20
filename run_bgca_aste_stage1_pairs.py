from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from t5_aste_pipeline import (
    dynamic_multitriplet_config_tag,
    dynamic_pseudo_filter_tag,
    positive_finite_float,
)


ASTE_PAIRS = [
    ("rest14", "laptop14"),
    ("rest15", "laptop14"),
    ("rest16", "laptop14"),
    ("laptop14", "rest14"),
    ("laptop14", "rest15"),
    ("laptop14", "rest16"),
]


def run_command(command: list[str], dry_run: bool = False) -> None:
    print(" ".join(command), flush=True)
    if dry_run:
        return
    subprocess.run(command, check=True)


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def stage_done(
    status: dict,
    stage: str,
    outputs: list[Path],
    rerun: bool,
    legacy_stages: tuple[str, ...] = (),
) -> bool:
    marked_done = bool(status.get(stage)) or any(bool(status.get(name)) for name in legacy_stages)
    return marked_done and not rerun and all(path.exists() for path in outputs)


def validate_pseudo_provenance(
    run_dir: Path,
    model_path: Path,
    pseudo_source_tag: str,
) -> tuple[bool, str]:
    required_paths = (
        run_dir / "target_pseudo.jsonl",
        run_dir / "target_pseudo_high_precision.jsonl",
        run_dir / "target_pseudo_high_precision_analysis.json",
        run_dir / "target_pseudo_analysis.json",
        run_dir / "target_pseudo_generation_state.json",
    )
    missing = [path.name for path in required_paths if not path.exists()]
    if missing:
        return False, f"missing required pseudo artifacts: {', '.join(missing)}"

    try:
        analysis = read_json(run_dir / "target_pseudo_analysis.json")
        state = read_json(run_dir / "target_pseudo_generation_state.json")
        if not isinstance(analysis, dict) or not isinstance(state, dict):
            return False, "pseudo provenance metadata must contain JSON objects"
        analysis_model_path = Path(analysis.get("model_path", "")).resolve()
        state_model_path = Path(state.get("resolved_model_path", "")).resolve()
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False, "pseudo provenance metadata is unreadable"

    if state.get("status") != "complete":
        return False, f"generation state is {state.get('status')!r}, expected 'complete'"
    if state_model_path != analysis_model_path:
        return False, "state and analysis model paths do not match"
    if state.get("pseudo_source_tag") != analysis.get("pseudo_source_tag"):
        return False, "state and analysis pseudo source tags do not match"
    if state_model_path != model_path.resolve():
        return False, "recorded model path does not match the reused extractor best path"
    if state.get("pseudo_source_tag") != pseudo_source_tag:
        return False, "recorded pseudo source tag does not match the current extractor tag"
    return True, ""


def validate_dynamic_pseudo_selection(
    output_dir: Path,
    pseudo_source_tag: str,
    min_pseudo_weight: float,
    max_token_distance: int,
    strict: bool = False,
) -> tuple[bool, str]:
    required_paths = (
        output_dir / "target_pseudo_high_precision.jsonl",
        output_dir / "target_pseudo_high_precision_analysis.json",
        output_dir / "target_pseudo_generation_state.json",
    )
    missing = [path.name for path in required_paths if not path.exists()]
    if missing:
        return False, f"missing required dynamic pseudo artifacts: {', '.join(missing)}"

    try:
        state = read_json(output_dir / "target_pseudo_generation_state.json")
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False, "dynamic pseudo selection state is unreadable"

    if state.get("status") != "complete":
        return False, f"dynamic pseudo selection state is {state.get('status')!r}, expected 'complete'"
    if state.get("selection_mode") != "dynamic_high_precision":
        return False, "dynamic pseudo selection mode does not match"
    if bool(state.get("strict", False)) != bool(strict):
        return False, "dynamic pseudo strict mode does not match"
    if state.get("base_pseudo_source_tag") != pseudo_source_tag:
        return False, "dynamic pseudo selection source tag does not match"
    try:
        recorded_min_weight = float(state.get("min_pseudo_weight", -1.0))
        recorded_token_distance = int(state.get("max_token_distance", -1))
    except (TypeError, ValueError):
        return False, "dynamic pseudo selection parameters are invalid"
    if recorded_min_weight != float(min_pseudo_weight):
        return False, "dynamic pseudo min weight does not match"
    if recorded_token_distance != int(max_token_distance):
        return False, "dynamic pseudo token distance does not match"
    return True, ""


def mark_done(status_path: Path, status: dict, stage: str) -> None:
    status[stage] = True
    write_json(status_path, status)


def pair_run_dir(root: Path, source: str, target: str) -> Path:
    return root / f"{source}_to_{target}"


def summary_output_paths(output_root: Path, output_tag: str = "") -> tuple[Path, Path]:
    suffix = f"_{output_tag}" if output_tag else ""
    return (
        output_root / f"results_bgca_aste_stage1{suffix}.csv",
        output_root / f"results_bgca_aste_stage1{suffix}_CN.md",
    )


def generator_tag(prompt_style: str) -> str:
    if prompt_style == "label_to_text":
        return "label_to_text_gen"
    if prompt_style == "masked_mutual":
        return "masked_mutual_gen"
    if prompt_style == "mixed":
        return "mixed_l2t_masked_aspect_masked_opinion"
    raise ValueError(f"unsupported generator prompt style: {prompt_style}")


def pseudo_filter_tag(max_triplets: int, max_token_distance: int) -> str:
    if max_triplets < 1:
        raise ValueError("high_precision_max_triplets must be at least 1")
    if max_token_distance < 0:
        raise ValueError("high_precision_max_token_distance must be non-negative")
    return f"hp{max_triplets}_dist{max_token_distance}"


def neutral_weight_tag(neutral_loss_gain: float, neutral_max_effective_weight: float) -> str:
    gain_tag = int(round(neutral_loss_gain * 100))
    max_tag = int(round(neutral_max_effective_weight * 100))
    return f"neutral_gain{gain_tag}_max{max_tag}"


def complete_multi_weight_tag(extra_weight: float) -> str:
    if extra_weight <= 0 or extra_weight > 1:
        raise ValueError("complete_multi_extra_weight must be in (0, 1]")
    return f"complete_multi2_w{int(round(extra_weight * 100)):03d}"


def append_sentiment_summary_tag(
    summary_tag: str,
    loss_weight: float,
    source_only: bool,
    class_balanced: bool,
) -> str:
    if loss_weight <= 0:
        return summary_tag
    loss_tag = str(loss_weight).replace(".", "")
    suffix = f"sentiment_contrastive_l{loss_tag}"
    if source_only:
        suffix += "_source"
    if class_balanced:
        suffix += "_balanced"
    return f"{summary_tag}_{suffix}".strip("_")


def legacy_hp1_stage_names(generator_output_tag: str) -> dict[str, tuple[str, ...]]:
    return {
        "augment": (f"augment_{generator_output_tag}",),
        "train_final": (f"train_final_{generator_output_tag}",),
        "evaluate": (f"evaluate_{generator_output_tag}",),
    }


def augment_experiment_tag(
    base_tag: str,
    opinion_replacement_mode: str,
    sentiment_vector_backend: str = "t5",
    use_polarity_axis: bool = False,
) -> str:
    if opinion_replacement_mode == "coupled_random":
        return base_tag
    if opinion_replacement_mode == "semantic_same_sentiment":
        return f"{base_tag}_semantic_same_sentiment"
    if opinion_replacement_mode == "sentiment_vector":
        suffix = "sentiment_vector_glove" if sentiment_vector_backend == "glove" else "sentiment_vector"
        polarity_suffix = "_polarity_axis" if use_polarity_axis else ""
        return f"{base_tag}_{suffix}{polarity_suffix}"
    raise ValueError(f"unsupported opinion replacement mode: {opinion_replacement_mode}")


def metric_value(data: dict, *keys: str):
    current = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return ""
        current = current[key]
    return current


def run_pair(args: argparse.Namespace, source: str, target: str) -> dict:
    run_dir = pair_run_dir(Path(args.output_root), source, target)
    upstream_run_dir = Path(args.reuse_upstream_run_dir) if args.reuse_upstream_run_dir else None
    dynamic_multitriplet = getattr(args, "dynamic_multitriplet", False) or getattr(
        args,
        "dynamic_multitriplet_strict",
        False,
    )
    dynamic_multitriplet_strict = getattr(args, "dynamic_multitriplet_strict", False)
    if not args.dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "stage_status.json"
    status = read_json(status_path)
    gen_tag = generator_tag(args.generator_prompt_style)
    generator_train_file = run_dir / f"c3da_generator_train_{gen_tag}.jsonl"
    generator_dev_file = run_dir / f"c3da_generator_dev_{gen_tag}.jsonl"
    prepare_stage = f"prepare_{gen_tag}"
    extract_train_file = run_dir / "extract_train.jsonl"
    shared_prepare_outputs = [
        run_dir / "extract_dev.jsonl",
        run_dir / "source_train.jsonl",
        run_dir / "source_dev.jsonl",
        run_dir / "target_unlabeled.jsonl",
        run_dir / "target_train_gold_analysis.jsonl",
        run_dir / "target_test.jsonl",
        generator_train_file,
        generator_dev_file,
    ]
    prepare_outputs = [extract_train_file, *shared_prepare_outputs]
    dynamic_config_tag = ""
    if dynamic_multitriplet:
        source_weights = (
            getattr(args, "source_count1_weight", 1.0),
            getattr(args, "source_count2_weight", 1.15),
            getattr(args, "source_count3_weight", 1.25),
            getattr(args, "source_count4plus_weight", 1.30),
        )
        dynamic_config_tag = dynamic_multitriplet_config_tag(*source_weights)
        prepare_stage = f"prepare_{dynamic_config_tag}_{gen_tag}"
        extract_train_file = run_dir / f"extract_train_{dynamic_config_tag}.jsonl"
        prepare_outputs = [
            extract_train_file,
            run_dir / f"extract_train_multitriplet_weight_analysis_{dynamic_config_tag}.json",
            *shared_prepare_outputs,
        ]

    extractor_tag = (
        "extractor_ep25_aste_f1"
        if dynamic_multitriplet
        else "extractor_ep25_plain_last"
    )
    if dynamic_multitriplet:
        extractor_tag += f"_{dynamic_config_tag}"
    if args.extractor_lambda_sentiment_contrastive > 0:
        extractor_lambda_tag = str(args.extractor_lambda_sentiment_contrastive).replace(".", "")
        extractor_tag += f"_sentiment_contrastive_l{extractor_lambda_tag}_source_balanced"
        if args.sentiment_prototype_initialize_from_context:
            extractor_tag += "_encoder_context_init"
    extractor_dir = (upstream_run_dir or run_dir) / "models" / extractor_tag
    extractor_stage = f"train_{extractor_tag}"
    if upstream_run_dir is not None:
        upstream_valid, upstream_reason = validate_pseudo_provenance(
            upstream_run_dir,
            extractor_dir / "best",
            extractor_tag,
        )
        if not (extractor_dir / "best" / "config.json").exists():
            upstream_valid = False
            upstream_reason = "upstream extractor best/config.json is missing"
        if not upstream_valid:
            message = f"cannot validate upstream pseudo provenance: {upstream_reason}"
            if args.dry_run:
                print(message, flush=True)
            else:
                raise RuntimeError(message)

    py = sys.executable
    common_train = [
        "--per_device_train_batch_size",
        "1",
        "--per_device_eval_batch_size",
        "2",
        "--gradient_accumulation_steps",
        "16",
        "--learning_rate",
        str(args.learning_rate),
        "--fp16",
        "--gradient_checkpointing",
        "--cuda",
        args.cuda,
        "--seed",
        str(args.seed),
    ]

    if not stage_done(
        status,
        prepare_stage,
        prepare_outputs,
        args.rerun,
    ):
        run_command(
            [
                py,
                "t5_aste_pipeline.py",
                "prepare",
                "--source_dataset",
                source,
                "--target_dataset",
                target,
                "--run_dir",
                str(run_dir),
                "--seed",
                str(args.seed),
                "--augment_prompt_style",
                args.generator_prompt_style,
                "--augment_channel_mode",
                "all",
                "--domain_prefix_style",
                args.domain_prefix_style,
                "--generator_output_tag",
                gen_tag,
                "--no_task_prefix",
                *(
                    [
                        "--dynamic_multitriplet",
                        "--source_count1_weight",
                        str(source_weights[0]),
                        "--source_count2_weight",
                        str(source_weights[1]),
                        "--source_count3_weight",
                        str(source_weights[2]),
                        "--source_count4plus_weight",
                        str(source_weights[3]),
                    ]
                    if dynamic_multitriplet
                    else []
                ),
            ],
            args.dry_run,
        )
        if not args.dry_run:
            mark_done(status_path, status, prepare_stage)

    if upstream_run_dir is None and not stage_done(
        status,
        extractor_stage,
        [extractor_dir / "best" / "config.json"],
        args.rerun,
        legacy_stages=() if dynamic_multitriplet else ("train_extractor",),
    ):
        run_command(
            [
                py,
                "t5_absa_train.py",
                "--model_path",
                args.extractor_model_path,
                "--train_file",
                str(extract_train_file),
                "--dev_file",
                str(run_dir / "extract_dev.jsonl"),
                "--output_dir",
                str(extractor_dir),
                "--num_train_epochs",
                str(args.extractor_epochs),
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
                "aste_f1" if dynamic_multitriplet else "last",
                "--resume_from_checkpoint",
                "auto",
                "--lambda_sentiment_contrastive",
                str(args.extractor_lambda_sentiment_contrastive),
                *(["--sentiment_contrastive_source_only", "--sentiment_contrastive_class_balanced"] if args.extractor_lambda_sentiment_contrastive > 0 else []),
                *(["--sentiment_prototype_initialize_from_context", "--sentiment_prototype_init_batch_size", str(args.sentiment_prototype_init_batch_size)] if args.extractor_lambda_sentiment_contrastive > 0 and args.sentiment_prototype_initialize_from_context else []),
                *common_train,
            ],
            args.dry_run,
        )
        if not args.dry_run:
            mark_done(status_path, status, extractor_stage)

    pseudo_stage = f"pseudo_{extractor_tag}"
    pseudo_outputs = [
        run_dir / "target_pseudo.jsonl",
        run_dir / "target_pseudo_high_precision.jsonl",
        run_dir / "target_pseudo_high_precision_analysis.json",
    ]
    provenance_matches, _provenance_reason = validate_pseudo_provenance(
        run_dir,
        extractor_dir / "best",
        extractor_tag,
    )
    pseudo_is_reusable = stage_done(
        status,
        pseudo_stage,
        pseudo_outputs,
        args.rerun,
        legacy_stages=() if dynamic_multitriplet else ("pseudo",),
    ) and provenance_matches
    if upstream_run_dir is None and not pseudo_is_reusable:
        run_command(
            [
                py,
                "t5_aste_pipeline.py",
                "pseudo",
                "--run_dir",
                str(run_dir),
                "--model_path",
                str(extractor_dir / "best"),
                "--batch_size",
                str(args.eval_batch_size),
                "--num_beams",
                "1",
                "--max_new_tokens",
                "128",
                "--no_constrained_decoding",
                "--cuda",
                args.cuda,
                "--no_task_prefix",
                "--pseudo_model_variant",
                "last",
                "--pseudo_source_tag",
                extractor_tag,
                "--high_precision_max_triplets",
                "1",
                "--high_precision_max_token_distance",
                "5",
            ],
            args.dry_run,
        )
        if not args.dry_run:
            mark_done(status_path, status, pseudo_stage)

    pseudo_input_run_dir = upstream_run_dir or run_dir
    pseudo_train_file = pseudo_input_run_dir / "target_pseudo_high_precision.jsonl"
    pseudo_analysis_file = pseudo_input_run_dir / "target_pseudo_high_precision_analysis.json"
    pseudo_tag = pseudo_filter_tag(
        args.high_precision_max_triplets,
        args.high_precision_max_token_distance,
    )
    use_legacy_pseudo_filter = (
        not dynamic_multitriplet
        and args.high_precision_max_triplets == 1
        and args.high_precision_max_token_distance == 5
    )
    if dynamic_multitriplet:
        pseudo_tag = dynamic_pseudo_filter_tag(
            args.high_precision_max_token_distance,
            strict=dynamic_multitriplet_strict,
        )
        pseudo_variant_dir = pseudo_input_run_dir / "pseudo_variants" / pseudo_tag
        pseudo_train_file = pseudo_variant_dir / "target_pseudo_high_precision.jsonl"
        pseudo_analysis_file = pseudo_variant_dir / "target_pseudo_high_precision_analysis.json"
        pseudo_selection_state_file = pseudo_variant_dir / "target_pseudo_generation_state.json"
        pseudo_filter_stage = f"select_dynamic_pseudo_{extractor_tag}_{pseudo_tag}"
        selection_valid, _selection_reason = validate_dynamic_pseudo_selection(
            pseudo_variant_dir,
            extractor_tag,
            0.65,
            args.high_precision_max_token_distance,
            strict=dynamic_multitriplet_strict,
        )
        dynamic_selection_reusable = stage_done(
            status,
            pseudo_filter_stage,
            [pseudo_train_file, pseudo_analysis_file, pseudo_selection_state_file],
            args.rerun,
        ) and selection_valid
        if upstream_run_dir is None and not dynamic_selection_reusable:
            run_command(
                [
                    py,
                    "t5_aste_pipeline.py",
                    "select_dynamic_pseudo",
                    "--run_dir",
                    str(run_dir),
                    "--output_dir",
                    str(pseudo_variant_dir),
                    "--min_pseudo_weight",
                    "0.65",
                    "--high_precision_max_token_distance",
                    str(args.high_precision_max_token_distance),
                    *(["--dynamic_strict"] if dynamic_multitriplet_strict else []),
                ],
                args.dry_run,
            )
            if not args.dry_run:
                mark_done(status_path, status, pseudo_filter_stage)
    elif not use_legacy_pseudo_filter:
        pseudo_variant_dir = pseudo_input_run_dir / "pseudo_variants" / pseudo_tag
        pseudo_train_file = pseudo_variant_dir / "target_pseudo_high_precision.jsonl"
        pseudo_analysis_file = pseudo_variant_dir / "target_pseudo_high_precision_analysis.json"
        pseudo_filter_stage = f"select_pseudo_{extractor_tag}_{pseudo_tag}"
        if upstream_run_dir is None and not stage_done(
            status,
            pseudo_filter_stage,
            [pseudo_train_file, pseudo_analysis_file],
            args.rerun,
        ):
            run_command(
                [
                    py,
                    "t5_aste_pipeline.py",
                    "select_pseudo",
                    "--run_dir",
                    str(run_dir),
                    "--output_dir",
                    str(pseudo_variant_dir),
                    "--min_pseudo_weight",
                    "0.65",
                    "--high_precision_max_triplets",
                    str(args.high_precision_max_triplets),
                    "--high_precision_max_token_distance",
                    str(args.high_precision_max_token_distance),
                ],
                args.dry_run,
            )
            if not args.dry_run:
                mark_done(status_path, status, pseudo_filter_stage)

    complete_multi_tag = ""
    if args.complete_multi_extra_weight > 0:
        if not use_legacy_pseudo_filter:
            raise ValueError("complete multi-triplet supplementation requires the hp1_dist5 base filter")
        complete_multi_tag = complete_multi_weight_tag(args.complete_multi_extra_weight)
        pseudo_variant_dir = pseudo_input_run_dir / "pseudo_variants" / f"hp1_{complete_multi_tag.replace('complete_multi2', 'complete2_dist5')}"
        pseudo_train_file = pseudo_variant_dir / "target_pseudo_high_precision.jsonl"
        pseudo_analysis_file = pseudo_variant_dir / "target_pseudo_high_precision_analysis.json"
        complete_stage = f"select_pseudo_{extractor_tag}_{complete_multi_tag}"
        if upstream_run_dir is None and not stage_done(
            status,
            complete_stage,
            [pseudo_train_file, pseudo_analysis_file],
            args.rerun,
        ):
            run_command(
                [
                    py,
                    "t5_aste_pipeline.py",
                    "select_complete_multi_pseudo",
                    "--run_dir",
                    str(run_dir),
                    "--output_dir",
                    str(pseudo_variant_dir),
                    "--base_pseudo_file",
                    str(run_dir / "target_pseudo_high_precision.jsonl"),
                    "--min_pseudo_weight",
                    "0.65",
                    "--high_precision_max_token_distance",
                    "5",
                    "--complete_multi_extra_weight",
                    str(args.complete_multi_extra_weight),
                ],
                args.dry_run,
            )
            if not args.dry_run:
                mark_done(status_path, status, complete_stage)

    if upstream_run_dir is not None and not args.dry_run:
        required_upstream_paths = [
            extractor_dir / "best" / "config.json",
            pseudo_input_run_dir / "target_pseudo.jsonl",
            pseudo_train_file,
            pseudo_analysis_file,
        ]
        if dynamic_multitriplet:
            required_upstream_paths.append(pseudo_selection_state_file)
        missing_paths = [path for path in required_upstream_paths if not path.exists()]
        if missing_paths:
            missing_text = ", ".join(str(path) for path in missing_paths)
            raise FileNotFoundError(f"missing required upstream artifacts: {missing_text}")
        if dynamic_multitriplet:
            selection_valid, selection_reason = validate_dynamic_pseudo_selection(
                pseudo_train_file.parent,
                extractor_tag,
                0.65,
                args.high_precision_max_token_distance,
                strict=dynamic_multitriplet_strict,
            )
            if not selection_valid:
                raise RuntimeError(f"cannot validate upstream dynamic pseudo selection: {selection_reason}")

    generator_dir = run_dir / "models" / f"generator_{gen_tag}_ep{args.generator_epochs}"
    if not stage_done(status, f"train_generator_{gen_tag}", [generator_dir / "best" / "config.json"], args.rerun):
        run_command(
            [
                py,
                "t5_absa_train.py",
                "--model_path",
                args.generator_model_path,
                "--train_file",
                str(generator_train_file),
                "--dev_file",
                str(generator_dev_file),
                "--output_dir",
                str(generator_dir),
                "--num_train_epochs",
                str(args.generator_epochs),
                "--source_weight",
                "1.0",
                "--pseudo_weight",
                "1.0",
                "--augment_weight",
                "1.0",
                "--checkpoint_selection",
                "best",
                "--resume_from_checkpoint",
                "auto",
                *common_train,
            ],
            args.dry_run,
        )
        if not args.dry_run:
            mark_done(status_path, status, f"train_generator_{gen_tag}")

    pseudo_suffix = "" if use_legacy_pseudo_filter else f"_{pseudo_tag}"
    if complete_multi_tag:
        pseudo_suffix = f"_{complete_multi_tag}"
    final_tag = augment_experiment_tag(
        f"strict_aug150_w020_{gen_tag}{pseudo_suffix}",
        args.opinion_replacement_mode,
        args.sentiment_vector_backend,
        args.sentiment_vector_use_polarity_axis,
    )
    final_train_file = run_dir / f"final_train_{final_tag}.jsonl"
    final_dev_file = run_dir / f"final_dev_{final_tag}.jsonl"
    reuse_for_auxiliary_loss = (
        (args.lambda_sentiment_contrastive > 0 or args.lambda_pairing_loss > 0)
        and final_train_file.exists()
        and final_dev_file.exists()
        and not args.rerun
    )
    legacy_stage_names = (
        legacy_hp1_stage_names(gen_tag)
        if use_legacy_pseudo_filter and not complete_multi_tag
        else {}
    )
    augment_legacy_stages = legacy_stage_names.get("augment", ())
    if complete_multi_tag:
        base_augment_tag = augment_experiment_tag(
            f"strict_aug150_w020_{gen_tag}",
            args.opinion_replacement_mode,
            args.sentiment_vector_backend,
            args.sentiment_vector_use_polarity_axis,
        )
        selected_augment_file = run_dir / f"c3da_two_channel_augmented_selected_{base_augment_tag}.jsonl"
        build_final_stage = f"build_final_{final_tag}"
        if not stage_done(
            status,
            build_final_stage,
            [final_train_file, final_dev_file],
            args.rerun,
        ):
            run_command(
                [
                    py,
                    "t5_aste_pipeline.py",
                    "build_final_train_from_files",
                    "--run_dir",
                    str(run_dir),
                    "--pseudo_train_file",
                    str(pseudo_train_file),
                    "--selected_augment_file",
                    str(selected_augment_file),
                    "--final_train_output_tag",
                    final_tag,
                    "--no_task_prefix",
                ],
                args.dry_run,
            )
            if not args.dry_run:
                mark_done(status_path, status, build_final_stage)
    elif not reuse_for_auxiliary_loss and not stage_done(
        status,
        f"augment_{final_tag}",
        [final_train_file],
        args.rerun,
        legacy_stages=augment_legacy_stages,
    ):
        run_command(
            [
                py,
                "t5_aste_pipeline.py",
                "augment",
                "--run_dir",
                str(run_dir),
                *(
                    ["--augmentation_input_run_dir", str(upstream_run_dir)]
                    if upstream_run_dir is not None
                    else []
                ),
                "--model_path",
                str(generator_dir / "best"),
                "--nli_model_path",
                args.nli_model_path,
                "--augment_prompt_style",
                args.augment_prompt_style,
                "--augment_channel_mode",
                "all",
                "--domain_prefix_style",
                args.domain_prefix_style,
                "--opinion_replacement_mode",
                args.opinion_replacement_mode,
                "--sentiment_vector_model_path",
                args.sentiment_vector_model_path,
                "--sentiment_vector_backend",
                args.sentiment_vector_backend,
                "--glove_path",
                args.glove_path,
                "--sentiment_vector_min_margin",
                str(args.sentiment_vector_min_margin),
                "--sentiment_vector_min_old_similarity",
                str(args.sentiment_vector_min_old_similarity),
                "--sentiment_vector_no_cooccurrence_min_similarity",
                str(args.sentiment_vector_no_cooccurrence_min_similarity),
                *(["--sentiment_vector_use_polarity_axis"] if args.sentiment_vector_use_polarity_axis else []),
                "--augment_output_tag",
                final_tag,
                "--final_train_output_tag",
                final_tag,
                "--augment_select_max_rows",
                "150",
                "--augment_select_max_per_base",
                "1",
                "--augment_select_weight",
                "0.2",
                "--augment_select_max_opinion_ratio",
                str(args.augment_select_max_opinion_ratio),
                "--augment_select_require_raw_exact",
                "--augment_select_require_model_filter_passed",
                "--pseudo_train_source",
                "high_precision",
                "--pseudo_train_file",
                str(pseudo_train_file),
                "--high_precision_max_triplets",
                str(args.high_precision_max_triplets),
                "--high_precision_max_token_distance",
                str(args.high_precision_max_token_distance),
                "--model_filter_path",
                str(extractor_dir / "best"),
                "--model_filter_mode",
                "fixed",
                "--model_filter_batch_size",
                "2",
                "--model_filter_num_beams",
                "1",
                "--model_filter_no_constrained_decoding",
                "--model_filter_channel_aware",
                "--model_filter_opinion_similarity_min",
                str(args.model_filter_opinion_similarity_min),
                *(["--model_filter_require_opinion_polarity"] if args.model_filter_require_opinion_polarity else []),
                "--cuda",
                args.cuda,
                "--no_task_prefix",
            ],
            args.dry_run,
        )
        if not args.dry_run:
            mark_done(status_path, status, f"augment_{final_tag}")

    result_tag = final_tag
    if args.lambda_sentiment_contrastive > 0:
        lambda_tag = str(args.lambda_sentiment_contrastive).replace(".", "")
        result_tag = f"{final_tag}_sentiment_contrastive_l{lambda_tag}"
        if args.sentiment_contrastive_source_only:
            result_tag += "_source"
        if args.sentiment_contrastive_class_balanced:
            result_tag += "_balanced"
        if args.sentiment_prototype_initialize_from_context:
            result_tag += "_encoder_context_init"
    if args.lambda_pairing_loss > 0:
        pairing_lambda_tag = str(args.lambda_pairing_loss).replace(".", "")
        result_tag += f"_pairing_encoder_l{pairing_lambda_tag}"
        if args.pairing_source_only:
            result_tag += "_source_only"
    use_neutral_weight_variant = (
        args.neutral_generation_loss_gain > 0
        or args.neutral_generation_max_effective_weight > 0
    )
    if use_neutral_weight_variant:
        neutral_max_weight = (
            args.neutral_generation_max_effective_weight
            if args.neutral_generation_max_effective_weight > 0
            else 1.0
        )
        result_tag += f"_{neutral_weight_tag(args.neutral_generation_loss_gain, neutral_max_weight)}"
    final_dir = run_dir / "models" / f"final_dann_l0.03_{result_tag}_ep{args.final_epochs}"
    if not stage_done(
        status,
        f"train_final_{result_tag}",
        [final_dir / "best" / "config.json"],
        args.rerun,
        legacy_stages=legacy_stage_names.get("train_final", ()),
    ):
        run_command(
            [
                py,
                "t5_absa_train.py",
                "--model_path",
                args.extractor_model_path,
                "--train_file",
                str(final_train_file),
                "--dev_file",
                str(final_dev_file),
                "--output_dir",
                str(final_dir),
                "--num_train_epochs",
                str(args.final_epochs),
                "--source_weight",
                "1.0",
                "--pseudo_weight",
                "0.5",
                "--augment_weight",
                "0.2",
                "--checkpoint_selection",
                "best",
                "--resume_from_checkpoint",
                "auto",
                "--lambda_domain_adv",
                "0.03",
                "--domain_adv_grl_lambda",
                "1.0",
                "--domain_adv_hidden_size",
                "256",
                "--domain_adv_exclude_augment",
                "--lambda_sentiment_contrastive",
                str(args.lambda_sentiment_contrastive),
                "--lambda_pairing_loss",
                str(args.lambda_pairing_loss),
                "--pairing_temperature",
                str(args.pairing_temperature),
                "--sentiment_contrastive_temperature",
                str(args.sentiment_contrastive_temperature),
                "--sentiment_contrastive_min_weight",
                str(args.sentiment_contrastive_min_weight),
                "--neutral_generation_loss_gain",
                str(args.neutral_generation_loss_gain),
                "--neutral_generation_max_effective_weight",
                str(args.neutral_generation_max_effective_weight),
                *(["--sentiment_contrastive_exclude_augment"] if args.sentiment_contrastive_exclude_augment else []),
                *(["--sentiment_contrastive_source_only"] if args.sentiment_contrastive_source_only else []),
                *(["--sentiment_contrastive_class_balanced"] if args.sentiment_contrastive_class_balanced else []),
                *(["--sentiment_prototype_initialize_from_context", "--sentiment_prototype_init_batch_size", str(args.sentiment_prototype_init_batch_size)] if args.sentiment_prototype_initialize_from_context else []),
                *(["--pairing_source_only"] if args.pairing_source_only else []),
                *common_train,
            ],
            args.dry_run,
        )
        if not args.dry_run:
            mark_done(status_path, status, f"train_final_{result_tag}")

    metrics_tag = result_tag
    raw_metrics_path = run_dir / f"aste_metrics_raw_{metrics_tag}.json"
    fixed_metrics_path = run_dir / f"aste_metrics_fixed_{metrics_tag}.json"
    sentiment_metrics_path = run_dir / f"aste_metrics_by_sentiment_{metrics_tag}.json"
    structure_metrics_path = run_dir / f"aste_metrics_by_structure_{metrics_tag}.json"
    error_analysis_path = run_dir / f"aste_error_analysis_{metrics_tag}.json"
    legacy_raw_metrics_path = run_dir / f"aste_metrics_raw_{gen_tag}.json"
    legacy_fixed_metrics_path = run_dir / f"aste_metrics_fixed_{gen_tag}.json"
    if (
        use_legacy_pseudo_filter
        and not complete_multi_tag
        and not use_neutral_weight_variant
        and args.lambda_sentiment_contrastive == 0
        and args.lambda_pairing_loss == 0
        and legacy_raw_metrics_path.exists()
        and legacy_fixed_metrics_path.exists()
    ):
        metrics_tag = gen_tag
        raw_metrics_path = legacy_raw_metrics_path
        fixed_metrics_path = legacy_fixed_metrics_path
    if not stage_done(
        status,
        f"evaluate_{result_tag}",
        [
            raw_metrics_path,
            fixed_metrics_path,
            *(
                [sentiment_metrics_path, structure_metrics_path, error_analysis_path]
                if use_neutral_weight_variant or args.lambda_pairing_loss > 0
                else []
            ),
        ],
        args.rerun,
        legacy_stages=legacy_stage_names.get("evaluate", ()),
    ):
        run_command(
            [
                py,
                "t5_aste_pipeline.py",
                "evaluate",
                "--run_dir",
                str(run_dir),
                "--model_path",
                str(final_dir / "best"),
                "--batch_size",
                str(args.eval_batch_size),
                "--num_beams",
                "4",
                "--max_new_tokens",
                "96",
                "--cuda",
                args.cuda,
                "--no_task_prefix",
                "--no_constrained_decoding",
                "--output_tag",
                result_tag,
            ],
            args.dry_run,
        )
        if not args.dry_run:
            mark_done(status_path, status, f"evaluate_{result_tag}")

    return summarize_pair(
        run_dir,
        source,
        target,
        final_tag,
        result_tag,
        args.generator_prompt_style,
        args.augment_prompt_style,
        args.domain_prefix_style,
        args.opinion_replacement_mode,
        pseudo_analysis_file,
        metrics_tag,
    )


def summarize_pair(
    run_dir: Path,
    source: str,
    target: str,
    final_tag: str,
    result_tag: str,
    generator_prompt_style: str,
    configured_augment_prompt_style: str,
    configured_domain_prefix_style: str,
    configured_opinion_replacement_mode: str,
    pseudo_analysis_file: Path,
    metrics_tag: str,
) -> dict:
    pseudo_hp = read_json(pseudo_analysis_file)
    augment = read_json(run_dir / f"c3da_augment_analysis_{final_tag}.json")
    final_comp = read_json(run_dir / f"final_train_composition_analysis_{final_tag}.json")
    raw = read_json(run_dir / f"aste_metrics_raw_{metrics_tag}.json")
    fixed = read_json(run_dir / f"aste_metrics_fixed_{metrics_tag}.json")
    hp_eval = pseudo_hp.get("hidden_gold_eval", {})
    hp_raw = hp_eval.get("raw_scores", {})
    return {
        "source": source,
        "target": target,
        "generator_prompt_style": generator_prompt_style,
        "augment_prompt_style": augment.get("prompt_style", configured_augment_prompt_style),
        "domain_prefix_style": augment.get("domain_prefix_style", configured_domain_prefix_style),
        "opinion_replacement_mode": augment.get("opinion_replacement_mode", configured_opinion_replacement_mode),
        "run_dir": str(run_dir),
        "source_rows": metric_value(final_comp, "source_rows_used"),
        "pseudo_hp_rows": pseudo_hp.get("selected_rows", ""),
        "pseudo_hp_precision": hp_raw.get("precision", ""),
        "pseudo_hp_recall": hp_raw.get("recall", ""),
        "pseudo_hp_f1": hp_raw.get("micro_f1", ""),
        "augment_selected_rows": augment.get(
            "selected_augmented_rows",
            final_comp.get("selected_augmented_rows", ""),
        ),
        "final_train_rows": final_comp.get("final_train_rows", ""),
        "raw_precision": raw.get("precision", ""),
        "raw_recall": raw.get("recall", ""),
        "raw_f1": raw.get("micro_f1", ""),
        "fixed_precision": fixed.get("precision", ""),
        "fixed_recall": fixed.get("recall", ""),
        "fixed_f1": fixed.get("micro_f1", ""),
    }


def write_summary_legacy(output_root: Path, rows: list[dict]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "results_bgca_aste_stage1.csv"
    fieldnames = list(rows[0].keys()) if rows else [
        "source",
        "target",
        "generator_prompt_style",
        "augment_prompt_style",
        "domain_prefix_style",
        "opinion_replacement_mode",
        "run_dir",
        "source_rows",
        "pseudo_hp_rows",
        "pseudo_hp_precision",
        "pseudo_hp_recall",
        "pseudo_hp_f1",
        "augment_selected_rows",
        "final_train_rows",
        "raw_precision",
        "raw_recall",
        "raw_f1",
        "fixed_precision",
        "fixed_recall",
        "fixed_f1",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md_path = output_root / "results_bgca_aste_stage1_CN.md"
    lines = [
        "# BGCA ASTE 跨域 Stage1 基线结果",
        "",
        "主指标使用 raw F1（原始F1），fixed F1（修正F1）仅作辅助分析。",
        "",
        "| 迁移方向 | 生成器训练方式 | 增强方式 | 领域前缀 | 观点词替换模式 | 伪标签F1 | 增强条数 | 最终训练条数 | raw P | raw R | raw F1 | fixed F1 |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        pair = f"{row['source']} -> {row['target']}"
        lines.append(
            "| "
            + " | ".join(
                [
                    pair,
                    str(row.get("generator_prompt_style", "")),
                    str(row.get("augment_prompt_style", "")),
                    str(row.get("domain_prefix_style", "")),
                    str(row.get("opinion_replacement_mode", "")),
                    fmt(row.get("pseudo_hp_f1")),
                    str(row.get("augment_selected_rows", "")),
                    str(row.get("final_train_rows", "")),
                    fmt(row.get("raw_precision")),
                    fmt(row.get("raw_recall")),
                    fmt(row.get("raw_f1")),
                    fmt(row.get("fixed_f1")),
                ]
            )
            + " |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({"csv": str(csv_path), "md": str(md_path)}, flush=True)


def fmt(value) -> str:
    if value == "" or value is None:
        return ""
    return f"{float(value) * 100:.2f}"


def write_summary(output_root: Path, rows: list[dict], output_tag: str = "") -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path, md_path = summary_output_paths(output_root, output_tag)
    fieldnames = list(rows[0].keys()) if rows else [
        "source",
        "target",
        "generator_prompt_style",
        "augment_prompt_style",
        "domain_prefix_style",
        "opinion_replacement_mode",
        "run_dir",
        "source_rows",
        "pseudo_hp_rows",
        "pseudo_hp_precision",
        "pseudo_hp_recall",
        "pseudo_hp_f1",
        "augment_selected_rows",
        "final_train_rows",
        "raw_precision",
        "raw_recall",
        "raw_f1",
        "fixed_precision",
        "fixed_recall",
        "fixed_f1",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# BGCA ASTE 跨域 Stage1 基线结果",
        "",
        "主指标使用 raw F1（原始 F1），fixed F1（修正 F1）仅作为辅助分析。",
        "",
        "| 跨域方向 | 生成器训练方式 | 增强方式 | 领域前缀 | 观点词替换模式 | 高精度伪标签 F1 | 增强条数 | 最终训练条数 | raw P | raw R | raw F1 | fixed F1 |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        pair = f"{row['source']} -> {row['target']}"
        lines.append(
            "| "
            + " | ".join(
                [
                    pair,
                    str(row.get("generator_prompt_style", "")),
                    str(row.get("augment_prompt_style", "")),
                    str(row.get("domain_prefix_style", "")),
                    str(row.get("opinion_replacement_mode", "")),
                    fmt(row.get("pseudo_hp_f1")),
                    str(row.get("augment_selected_rows", "")),
                    str(row.get("final_train_rows", "")),
                    fmt(row.get("raw_precision")),
                    fmt(row.get("raw_recall")),
                    fmt(row.get("raw_f1")),
                    fmt(row.get("fixed_f1")),
                ]
            )
            + " |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({"csv": str(csv_path), "md": str(md_path)}, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", default=r"runs\bgca_aste_stage1_baseline")
    parser.add_argument("--pairs", default="all", help="all or comma list like rest16:laptop14,laptop14:rest16")
    parser.add_argument(
        "--reuse_upstream_run_dir",
        default="",
        help="reuse one completed pair run's extractor and pseudo labels while writing new outputs to output_root",
    )
    parser.add_argument("--extractor_model_path", default=r"J:\nlp\models\t5-base-py")
    parser.add_argument("--generator_model_path", default=r"J:\nlp\models\t5-base-py")
    parser.add_argument(
        "--generator_prompt_style",
        choices=["label_to_text", "masked_mutual", "mixed"],
        default="label_to_text",
    )
    parser.add_argument("--augment_prompt_style", choices=["label_to_text", "masked_mutual"], default="masked_mutual")
    parser.add_argument("--domain_prefix_style", choices=["none", "text", "bracket"], default="none")
    parser.add_argument(
        "--opinion_replacement_mode",
        choices=["coupled_random", "semantic_same_sentiment", "sentiment_vector"],
        default="coupled_random",
    )
    parser.add_argument("--sentiment_vector_model_path", default=r"J:\nlp\models\t5-base-py")
    parser.add_argument("--sentiment_vector_backend", choices=["t5", "glove"], default="t5")
    parser.add_argument("--glove_path", default=r"J:\models\glove.6B.300d.txt")
    parser.add_argument("--sentiment_vector_min_margin", type=float, default=0.05)
    parser.add_argument("--sentiment_vector_use_polarity_axis", action="store_true")
    parser.add_argument("--sentiment_vector_min_old_similarity", type=float, default=0.35)
    parser.add_argument("--sentiment_vector_no_cooccurrence_min_similarity", type=float, default=0.50)
    parser.add_argument("--model_filter_opinion_similarity_min", type=float, default=0.0)
    parser.add_argument("--model_filter_require_opinion_polarity", action="store_true")
    parser.add_argument("--augment_select_max_opinion_ratio", type=float, default=1.0)
    parser.add_argument("--nli_model_path", default=r"J:\nlp\models\nli-deberta-v3-base-mnli-fever-anli")
    parser.add_argument("--extractor_epochs", type=int, default=25)
    parser.add_argument("--generator_epochs", type=int, default=8)
    parser.add_argument("--final_epochs", type=int, default=5)
    parser.add_argument("--extractor_lambda_sentiment_contrastive", type=float, default=0.0)
    parser.add_argument("--lambda_sentiment_contrastive", type=float, default=0.0)
    parser.add_argument("--lambda_pairing_loss", type=float, default=0.0)
    parser.add_argument("--pairing_temperature", type=float, default=0.1)
    parser.add_argument("--pairing_source_only", action="store_true")
    parser.add_argument("--sentiment_contrastive_temperature", type=float, default=0.1)
    parser.add_argument("--sentiment_contrastive_min_weight", type=float, default=0.65)
    parser.add_argument("--sentiment_contrastive_exclude_augment", action="store_true")
    parser.add_argument("--sentiment_contrastive_source_only", action="store_true")
    parser.add_argument("--sentiment_contrastive_class_balanced", action="store_true")
    parser.add_argument("--sentiment_prototype_initialize_from_context", action="store_true")
    parser.add_argument("--sentiment_prototype_init_batch_size", type=int, default=2)
    parser.add_argument("--neutral_generation_loss_gain", type=float, default=0.0)
    parser.add_argument("--neutral_generation_max_effective_weight", type=float, default=0.0)
    parser.add_argument("--high_precision_max_triplets", type=int, default=1)
    parser.add_argument("--high_precision_max_token_distance", type=int, default=5)
    parser.add_argument("--complete_multi_extra_weight", type=float, default=0.0)
    parser.add_argument("--dynamic_multitriplet", action="store_true")
    parser.add_argument("--dynamic_multitriplet_strict", action="store_true")
    parser.add_argument("--source_count1_weight", type=positive_finite_float, default=1.0)
    parser.add_argument("--source_count2_weight", type=positive_finite_float, default=1.15)
    parser.add_argument("--source_count3_weight", type=positive_finite_float, default=1.25)
    parser.add_argument("--source_count4plus_weight", type=positive_finite_float, default=1.30)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--cuda", default="0")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    return parser.parse_args()


def selected_pairs(pairs_text: str) -> list[tuple[str, str]]:
    if pairs_text == "all":
        return ASTE_PAIRS
    pairs = []
    for item in pairs_text.split(","):
        source, target = item.split(":")
        pairs.append((source.strip(), target.strip()))
    return pairs


def main() -> None:
    args = parse_args()
    rows = []
    pairs = selected_pairs(args.pairs)
    if args.reuse_upstream_run_dir and len(pairs) != 1:
        raise ValueError("--reuse_upstream_run_dir requires exactly one source:target pair")
    if args.dynamic_multitriplet_strict:
        args.dynamic_multitriplet = True
    pseudo_tag = (
        dynamic_pseudo_filter_tag(
            args.high_precision_max_token_distance,
            strict=args.dynamic_multitriplet_strict,
        )
        if args.dynamic_multitriplet
        else pseudo_filter_tag(
            args.high_precision_max_triplets,
            args.high_precision_max_token_distance,
        )
    )
    summary_tag = "" if pseudo_tag == "hp1_dist5" else pseudo_tag
    if args.neutral_generation_loss_gain > 0 or args.neutral_generation_max_effective_weight > 0:
        neutral_max_weight = (
            args.neutral_generation_max_effective_weight
            if args.neutral_generation_max_effective_weight > 0
            else 1.0
        )
        neutral_tag = neutral_weight_tag(args.neutral_generation_loss_gain, neutral_max_weight)
        summary_tag = f"{summary_tag}_{neutral_tag}".strip("_")
    if args.lambda_pairing_loss > 0:
        pairing_lambda_tag = str(args.lambda_pairing_loss).replace(".", "")
        pairing_tag = f"pairing_encoder_l{pairing_lambda_tag}"
        if args.pairing_source_only:
            pairing_tag += "_source_only"
        summary_tag = f"{summary_tag}_{pairing_tag}".strip("_")
    if args.complete_multi_extra_weight > 0:
        complete_tag = complete_multi_weight_tag(args.complete_multi_extra_weight)
        summary_tag = f"{summary_tag}_{complete_tag}".strip("_")
        summary_tag = append_sentiment_summary_tag(
            summary_tag,
            args.lambda_sentiment_contrastive,
            args.sentiment_contrastive_source_only,
            args.sentiment_contrastive_class_balanced,
        )
    for source, target in pairs:
        rows.append(run_pair(args, source, target))
        if not args.dry_run:
            write_summary(Path(args.output_root), rows, summary_tag)
    if not args.dry_run:
        write_summary(Path(args.output_root), rows, summary_tag)


if __name__ == "__main__":
    main()
