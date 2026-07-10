"""
CLI entrypoint for the preprocessing pipelines.

Dispatches to the tabular survival dataset builder and/or the
longitudinal ConDOR sequence builder depending on the --mode flag.

Usage:
    python -m xai_decision_making.preprocessing.run_preprocessing --mode survival --dataset dev
    python -m xai_decision_making.preprocessing.run_preprocessing --mode longitudinal --pacap_path ... --clinical_path ...
    python -m xai_decision_making.preprocessing.run_preprocessing --mode both --dataset dev --pacap_path ... --clinical_path ...
"""
import argparse

from xai_decision_making.preprocessing.builders.longitudinal_dataset_builder import LongitudinalDataFactory
from xai_decision_making.preprocessing.builders.survival_dataset_builder import build_survival_dataset


def run_survival_pipeline(dataset_type: str):
    df = build_survival_dataset(dataset_type=dataset_type)
    print("Survival pipeline complete\n")


def run_longitudinal_pipeline(pacap_path: str, clinical_path: str):
    factory = LongitudinalDataFactory(
        pacap_path=pacap_path, clinical_path=clinical_path, include_treatment=True
    )
    loader = factory.get_loaders(batch_size=32)
    print("Longitudinal pipeline complete\n")

    return loader


def main():
    parser = argparse.ArgumentParser(
        description="Preprocessing entrypoint for survival and longitudinal pipelines."
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["survival", "longitudinal", "both"],
        default="survival",
        help="Which preprocessing pipeline to run.",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["dev", "test", "black_envelop_test"],
        default="dev",
        help="Dataset type for survival builder.",
    )

    parser.add_argument(
        "--pacap_path",
        type=str,
        default=None,
        help="Path to PACAP dataset (required for longitudinal mode).",
    )

    parser.add_argument(
        "--clinical_path",
        type=str,
        default=None,
        help="Path to clinical dataset (required for longitudinal mode).",
    )

    args = parser.parse_args()

    if args.mode in ["survival", "both"]:
        run_survival_pipeline(dataset_type=args.dataset)

    if args.mode in ["longitudinal", "both"]:
        if not args.pacap_path or not args.clinical_path:
            raise ValueError(
                "Longitudinal mode requires --pacap_path and --clinical_path."
            )

        run_longitudinal_pipeline(
            pacap_path=args.pacap_path, clinical_path=args.clinical_path
        )


if __name__ == "__main__":
    main()
