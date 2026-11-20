#!/usr/bin/env python
"""
Punkt wejscia do uruchomienia pipeline'u SER.

Uzycie:
    python run.py [opcje]

Opcje:
    --models MODEL1,MODEL2    Tylko wybrane modele (domyslnie: wszystkie)
    --datasets DS1,DS2        Tylko wybrane datasety (domyslnie: wszystkie)
    --combined                Tryb polaczonych danych (6 klas)
    --preprocessed            Czytaj dane z data_preprocessed/ (VAD+preemfaza)
    --deltas                  3 kanały wejściowe (statyka + delta + delta-delta)
    --epochs N                Liczba epok (domyslnie: 50)
    --batch-size N            Rozmiar batcha (domyslnie: 32)
    --seed N                  Ziarno treningu (domyslnie: 42); podzial danych
                              jest zawsze ten sam, inny seed -> podfolder seed_<N>/
    --deterministic           Deterministyczne algorytmy CUDA (wolniej, ale
                              ten sam seed daje ten sam wynik)

Przyklady:
    python run.py
    python run.py --combined
    python run.py --preprocessed
    python run.py --deltas
    python run.py --models dscnn,lstm --datasets SAVEE
    python run.py --models gender_aware --datasets TESS,CREMA-D --epochs 30
    python run.py --combined --models gender_aware --seed 7 --deterministic
"""

from __future__ import annotations

import argparse

from ser_pipeline.main import DEFAULT_TRAINING_SEED, main
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
        "--preprocessed",
        action="store_true",
        default=False,
        help="Czytaj dane z data_preprocessed/ (VAD + preemfaza)",
    )
    parser.add_argument(
        "--deltas",
        action="store_true",
        default=False,
        help="3 kanały wejściowe: statyka + delta + delta-delta",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Liczba epok treningu (domyslnie: 50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Rozmiar batcha (domyslnie: 32)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_TRAINING_SEED,
        help=(
            "Ziarno treningu: wagi, kolejnosc batchy, augmentacja "
            f"(domyslnie: {DEFAULT_TRAINING_SEED}). Podzial danych sie nie zmienia."
        ),
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=False,
        help="Deterministyczne algorytmy CUDA (wolniejszy trening)",
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs i --batch-size musza byc dodatnie")
    if args.seed < 0:
        parser.error("--seed musi byc nieujemny")

    main(
        model_names=args.models,
        dataset_names=args.datasets,
        combined=args.combined,
        use_preprocessed=args.preprocessed,
        use_deltas=args.deltas,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        deterministic=args.deterministic,
    )
