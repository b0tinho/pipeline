#!/usr/bin/env python
"""
Punkt wejscia do uruchomienia pipeline'u SER.

Uzycie:
    python run.py [opcje]

Opcje:
    --models MODEL1,MODEL2    Tylko wybrane modele (domyslnie: wszystkie)
    --datasets DS1,DS2        Tylko wybrane datasety (domyslnie: wszystkie)
    --combined                Tryb polaczonych danych (6 klas)
    --epochs N                Liczba epok (domyslnie: 50)
    --batch-size N            Rozmiar batcha (domyslnie: 32)

Przyklady:
    python run.py
    python run.py --combined
    python run.py --models dscnn,lstm --datasets SAVEE
    python run.py --models gender_aware --datasets TESS,CREMA-D --epochs 30
"""

from __future__ import annotations

import argparse
from typing import Optional

from ser_pipeline.main import main
from ser_pipeline.model import MODEL_REGISTRY


def _parse_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SER Pipeline - Speech Emotion Recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Dostepne modele:  %(models)s
Dostepne datasety: TESS, RAVDESS, SAVEE, CREMA-D
        """.strip()
        % {"models": ", ".join(sorted(MODEL_REGISTRY.keys()))},
    )
    parser.add_argument(
        "--models",
        type=_parse_list,
        default=None,
        help="Lista modeli oddzielona przecinkami (domyslnie: wszystkie)",
    )
    parser.add_argument(
        "--datasets",
        type=_parse_list,
        default=None,
        help="Lista datasetow oddzielona przecinkami (domyslnie: wszystkie)",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        default=False,
        help="Tryb polaczonych danych (TESS+RAVDESS+SAVEE+CREMA-D, 6 klas)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Liczba epok treningu (domyslnie: 50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Rozmiar batcha (domyslnie: 32)",
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    # Nadpisanie domyslnych parametrow, jesli podano z CLI
    if args.epochs is not None or args.batch_size is not None:
        from ser_pipeline.engine import TrainingConfig

        if args.epochs is not None:
            TrainingConfig.num_epochs = args.epochs
        if args.batch_size is not None:
            TrainingConfig.batch_size = args.batch_size  # type: ignore[attr-defined]

    main(
        model_names=args.models,
        dataset_names=args.datasets,
        combined=args.combined,
    )
