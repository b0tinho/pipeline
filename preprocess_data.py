"""
Skrypt do jednorazowego preprocessingu offline wszystkich datasetów.

Stosuje VAD (wycięcie ciszy) + preemfazę i zapisuje wyniki do osobnego
katalogu `data_preprocessed/`, zachowując oryginalne nazwy i strukturę.

Użycie:
    python preprocess_data.py                 # wszystkie 4 datasety
    python preprocess_data.py --datasets SAVEE,TESS
    python preprocess_data.py --top-db 25     # inny próg ciszy
    python preprocess_data.py --clean         # usuń data_preprocessed/ najpierw

Po zakończeniu trening można uruchomić na przetworzonych danych:
    python run.py --preprocessed
"""

from __future__ import annotations

import argparse

from ser_pipeline.preprocessing import (
    PREPROCESSED_DATA_ROOT,
    RAW_DATA_ROOT,
    PreprocessingConfig,
    cleanup_preprocessed,
    preprocess_all_datasets,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline preprocessing audio: VAD + preemfaza",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--datasets",
        type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
        default=None,
        help="Lista datasetów oddzielona przecinkami (domyślnie: wszystkie)",
    )
    parser.add_argument(
        "--src", default=str(RAW_DATA_ROOT), help="Katalog źródłowy (domyślnie: data/)"
    )
    parser.add_argument(
        "--dst",
        default=str(PREPROCESSED_DATA_ROOT),
        help="Katalog docelowy (domyślnie: data_preprocessed/)",
    )
    parser.add_argument(
        "--sample-rate", type=int, default=16000, help="Częstotliwość próbkowania"
    )
    parser.add_argument(
        "--top-db", type=float, default=20.0, help="Próg ciszy VAD w dB"
    )
    parser.add_argument(
        "--preemphasis-coef",
        type=float,
        default=0.97,
        help="Współczynnik preemfazy",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        default=False,
        help="Usuń katalog docelowy przed przetwarzaniem",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()

    if args.clean:
        cleanup_preprocessed(args.dst)

    config = PreprocessingConfig(
        sample_rate=args.sample_rate,
        top_db=args.top_db,
        preemphasis_coef=args.preemphasis_coef,
    )

    preprocess_all_datasets(
        src_root=args.src,
        dst_root=args.dst,
        config=config,
        dataset_names=args.datasets,
    )

    print("\nPreprocessing zakończony.")
    print(f"Trening na przetworzonych danych: python run.py --preprocessed")
