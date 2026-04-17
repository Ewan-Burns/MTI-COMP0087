import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse

from _bootstrap import bootstrap_repo_path
bootstrap_repo_path()

from statnlp_bench.config import (  
    GenerativeDetectionConfig,
    QuickRunConfig,
    SupervisedTrainingConfig,
    build_artifact_paths,
)
from statnlp_bench.methods.profiles import available_method_profiles, resolve_method_profile  # noqa: E402
from statnlp_bench.runtime_checks import ensure_publication_detection_runtime  # noqa: E402
from statnlp_bench.tracks.generative_detection import run_full_generative_detection_pipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the publication-fair RAID detection benchmark.")
    parser.add_argument("--artifact-root", default="artifacts_publication")
    parser.add_argument("--dataset-name", default="publication_detection")
    parser.add_argument("--question-file", default="datasets/mt_bench_prompts/raw/question.jsonl")
    parser.add_argument("--method-profile", choices=available_method_profiles(), default="publication_core")
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--max-prompts", type=int, default=2000)
    parser.add_argument("--split-seed", type=int, default=26)
    parser.add_argument("--training-seed", type=int, default=26)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--detector-device", default="auto")
    parser.add_argument("--target-fpr", type=float, default=0.05)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--skip-unsupervised", action="store_true")
    parser.add_argument("--raid-revision", default=None)
    parser.add_argument("--use-human-shift", action="store_true")
    parser.add_argument("--human-shift-revision", default=None)
    parser.add_argument("--human-shift-max-items", type=int, default=1000)
    args = parser.parse_args()

    selected_method_profile = None if args.methods else args.method_profile
    resolved_methods = args.methods or resolve_method_profile(selected_method_profile)
    ensure_publication_detection_runtime(require_mbr="MBR_16_BERTSCORE" in resolved_methods)

    artifacts = build_artifact_paths(args.artifact_root)
    output = run_full_generative_detection_pipeline(
        config=GenerativeDetectionConfig(
            artifacts=artifacts,
            dataset_name=args.dataset_name,
            split_seed=args.split_seed,
            train_ratio=0.5,
            val_ratio=0.0,
            max_prompts=None,
            methods=resolved_methods,
            method_profile=selected_method_profile,
            use_external_detection_data=True,
            include_local_mt_bench=False,
            use_external_human_shift=args.use_human_shift,
            force_regenerate=args.force,
            target_fpr=args.target_fpr,
            split_profile="dubois_exact",
            publication_mode=True,
        ),
        quick_config=QuickRunConfig.from_env(),
        training_config=SupervisedTrainingConfig(
            epochs=args.epochs,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            device=args.detector_device,
            seed=args.training_seed,
            force_retrain=args.force_retrain,
            target_fpr=args.target_fpr,
        ),
        question_file=args.question_file,
        external_detection_kwargs={
            "dataset_name": args.dataset_name,
            "repo_id": "liamdugan/raid",
            "revision": args.raid_revision,
            "split": "train",
            "prompt_field": "prompt",
            "human_field": "generation",
            "prompt_id_field": "source_id",
            "category_field": "domain",
            "max_items": args.max_prompts,
            "seed": args.split_seed,
        },
        external_human_shift_kwargs=(
            {
                "dataset_name": "cc_news_shift",
                "repo_id": "vblagoje/cc_news",
                "revision": args.human_shift_revision,
                "split": "train",
                "text_field": "text",
                "max_items": args.human_shift_max_items,
                "seed": args.split_seed,
            }
            if args.use_human_shift
            else None
        ),
        hf_detector_ids=[],
        run_unsupervised=not args.skip_unsupervised,
    )
    print(f"Publication detection results written to {output['result_dir']}")


if __name__ == "__main__":
    main()
