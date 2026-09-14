"""
Preprocessing danych audio: VAD (wycięcie ciszy) + preemfaza.

Moduł przetwarza WSZYSTKIE pliki .wav danego zbioru danych JEDNORAZOWO
(offline) i zapisuje wyniki do osobnego katalogu, zachowując oryginalne
nazwy plików i strukturę folderów.

Dzięki temu:
    - VAD i preemfaza (operacje deterministyczne) są wykonywane tylko raz,
      a nie w każdej epoce treningu,
    - kolejność filtrów względem augmentacji jest jawna i stała,
    - oryginalne dane w `data/` pozostają nietknięte.

Augmentacja audio (losowa) NIE jest tu stosowana - pozostaje w locie,
wyłącznie dla splitu treningowego (patrz `augmentation.py`).

Uwaga: HPSS NIE jest częścią tego pipeline'u (został odrzucony ze względu
na koszt i możliwą utratę informacji prosodycznej).
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from .data_parsing import DATASET_PARSERS

# Domyślne katalogi: źródło (surowe) i cel (po preprocessingu).
RAW_DATA_ROOT = Path("data")
PREPROCESSED_DATA_ROOT = Path("data_preprocessed")


@dataclass
class PreprocessingConfig:
    """Konfiguracja preprocessingu offline.

    Attributes:
        sample_rate: Docelowa częstotliwość próbkowania.
        top_db: Próg ciszy dla VAD (dB).
        preemphasis_coef: Współczynnik preemfazy.
        min_duration_samples: Minimalna długość sygnału po VAD, poniżej
            której plik jest uznawany za "pusty" (sama cisza) i zamiast
            niego zapisywany jest oryginał (fallback).
    """

    sample_rate: int = 16_000
    top_db: float = 20.0
    preemphasis_coef: float = 0.97
    min_duration_samples: int = 1600  # 0.1 s przy 16 kHz


def preprocess_signal(
    signal: np.ndarray,
    sr: int,
    config: PreprocessingConfig,
) -> tuple[np.ndarray, bool]:
    """Stosuje VAD + preemfazę do pojedynczego sygnału.

    Args:
        signal: Sygnał audio 1D.
        sr: Częstotliwość próbkowania (nieużywana bezpośrednio, zachowana
            dla przyszłej rozszerzalności - np. filtrów zależnych od sr).
        config: Konfiguracja preprocessingu.

    Returns:
        Krotka (sygnał_po_filtracji, czy_uzyto_fallback). `czy_uzyto_fallback`
        jest True, gdy po VAD sygnał okazał się zbyt krótki (cisza) i
        zwrócono oryginał zamiast przefiltrowanej wersji.
    """
    del sr  # zarezerwowane na przyszłość

    trimmed, _ = librosa.effects.trim(signal, top_db=config.top_db)

    # Fallback: jeśli po VAD zostaje praktycznie cisza, zwróć oryginał.
    if trimmed.shape[0] < config.min_duration_samples:
        return signal, True

    filtered = librosa.effects.preemphasis(trimmed, coef=config.preemphasis_coef)
    return filtered, False


def preprocess_file(
    src_path: Path,
    dst_path: Path,
    config: PreprocessingConfig,
) -> tuple[str, str, bool]:
    """Przetwarza pojedynczy plik .wav i zapisuje do katalogu docelowego.

    Args:
        src_path: Ścieżka źródłowa pliku.
        dst_path: Ścieżka docelowa (zostanie utworzona wraz z katalogiem).
        config: Konfiguracja preprocessingu.

    Returns:
        Krotka (status, komunikat, czy_uzyto_fallback), gdzie status to
        "ok" lub "fallback".
    """
    signal, sr = librosa.load(src_path, sr=config.sample_rate, mono=True)
    filtered, used_fallback = preprocess_signal(signal, sr, config)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dst_path, filtered, config.sample_rate)

    status = "fallback" if used_fallback else "ok"
    message = "cisza - zapisano oryginał" if used_fallback else "przetworzono"
    return status, message, used_fallback


def preprocess_dataset(
    dataset_name: str,
    src_root: str | Path,
    dst_root: str | Path,
    config: PreprocessingConfig,
) -> dict:
    """Przetwarza wszystkie pliki .wav JEDNEGO datasetu.

    Zachowuje strukturę katalogów i oryginalne nazwy plików (kluczowe -
    parsery emocji i płci w `data_parsing.py` operują na nazwach).

    Args:
        dataset_name: Nazwa datasetu (TESS, RAVDESS, SAVEE, CREMA-D).
        src_root: Katalog źródłowy.
        dst_root: Katalog docelowy.
        config: Konfiguracja preprocessingu.

    Returns:
        Słownik ze statystykami: liczba przetworzonych, fallbacków,
        pominiętych (nierozpoznana konwencja) plików.

    Raises:
        ValueError: Gdy dataset_name jest nieobsługiwany.
    """
    if dataset_name not in DATASET_PARSERS:
        raise ValueError(
            f"Nieobsługiwany dataset: {dataset_name}. "
            f"Dostępne: {list(DATASET_PARSERS.keys())}"
        )

    src_root = Path(src_root)
    dst_root = Path(dst_root)
    if not src_root.exists():
        raise FileNotFoundError(f"Katalog źródłowy nie istnieje: {src_root}")

    parser_fn = DATASET_PARSERS[dataset_name]
    stats = {"processed": 0, "fallback": 0, "skipped": 0}

    wav_files = sorted(src_root.rglob("*.wav"))
    print(f"[{dataset_name}] Znaleziono {len(wav_files)} plików .wav")

    for src_path in wav_files:
        # Pomijamy pliki o nierozpoznanej konwencji nazw (nie trafią do
        # DataFrame w pipeline, więc ich przetwarzanie jest zbędne).
        if parser_fn(src_path) is None:
            stats["skipped"] += 1
            continue

        # Zachowujemy względną ścieżkę (strukturę folderów + nazwę pliku).
        rel_path = src_path.relative_to(src_root)
        dst_path = dst_root / rel_path

        try:
            _status, _msg, used_fallback = preprocess_file(src_path, dst_path, config)
            if used_fallback:
                stats["fallback"] += 1
            else:
                stats["processed"] += 1
        except Exception as exc:  # noqa: BLE001 - logujemy i kontynuujemy
            print(f"  [BŁĄD] {src_path}: {exc}")
            stats["skipped"] += 1

    return stats


def preprocess_all_datasets(
    src_root: str | Path = RAW_DATA_ROOT,
    dst_root: str | Path = PREPROCESSED_DATA_ROOT,
    config: PreprocessingConfig | None = None,
    dataset_names: list[str] | None = None,
) -> dict[str, dict]:
    """Przetwarza wszystkie (lub wybrane) datasety offline.

    Args:
        src_root: Katalog główny z surowymi danymi (domyślnie `data/`).
        dst_root: Katalog główny wynikowy (domyślnie `data_preprocessed/`).
        config: Konfiguracja preprocessingu (domyślnie `PreprocessingConfig()`).
        dataset_names: Lista nazw datasetów do przetworzenia. None = wszystkie
            z rejestru `DATASET_PARSERS`.

    Returns:
        Słownik `{dataset_name: stats}` ze statystykami per dataset.
    """
    config = config or PreprocessingConfig()
    names = dataset_names if dataset_names is not None else list(DATASET_PARSERS.keys())

    results: dict[str, dict] = {}
    for name in names:
        src = Path(src_root) / name
        dst = Path(dst_root) / name
        print(f"\n=== Preprocessing: {name} ===")
        stats = preprocess_dataset(name, src, dst, config)
        print(
            f"[{name}] Przetworzono: {stats['processed']}, "
            f"fallback: {stats['fallback']}, pominięto: {stats['skipped']}"
        )
        results[name] = stats

    return results


def cleanup_preprocessed(dst_root: str | Path = PREPROCESSED_DATA_ROOT) -> None:
    """Usuwa katalog z przetworzonymi danymi (przydatne przed ponownym runem).

    Args:
        dst_root: Katalog docelowy do usunięcia.
    """
    dst = Path(dst_root)
    if dst.exists():
        shutil.rmtree(dst)
        print(f"Usunięto: {dst}")
