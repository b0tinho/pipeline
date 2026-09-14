"""
Narzędzie do wizualizacji porównawczej Mel-spektrogramów.

Moduł pomocniczy (nie jest częścią głównej pętli treningowej) pozwala
porównać wizualnie oryginalny Mel-spektrogram nagrania z wersją po
zastosowaniu filtrów wstępnych:
    - VAD (Voice Activity Detection) - wycięcie ciszy z początku/końca,
    - preemfaza - wyostrzenie wysokich częstotliwości (formanty).

Przydatne do szybkiej oceny jakości nagrań i wpływu preprocessingu na
widoczność formantów (kluczowych dla rozpoznawania emocji).

Użycie z CLI:
    python -m ser_pipeline.spectrogram_analysis sciezka.wav "Tytul" [--save wynik.png]

Użycie programowe:
    from ser_pipeline.spectrogram_analysis import (
        generate_comparative_spectrograms,
    )
    generate_comparative_spectrograms("plik.wav", "Tytul", save_path="wynik.png")
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from .audio_processing import AudioProcessingConfig


def apply_vad(signal: np.ndarray, top_db: float = 20.0) -> np.ndarray:
    """Wycina ciszę z początku i końca sygnału (Voice Activity Detection).

    Args:
        signal: Sygnał audio 1D.
        top_db: Próg w dB poniżej maksymalnej amplitudy, poniżej którego
            próbki traktowane są jako cisza (domyślnie 20 dB).

    Returns:
        Sygnał po przycięciu ciszy.
    """
    trimmed, _ = librosa.effects.trim(signal, top_db=top_db)
    return trimmed


def apply_preemphasis(signal: np.ndarray, coef: float = 0.97) -> np.ndarray:
    """Stosuje preemfazę (filtr wyostrzający wysokie tony).

    Args:
        signal: Sygnał audio 1D.
        coef: Współczynnik preemfazy (typowe: 0.95-0.97).

    Returns:
        Sygnał po preemfazie.
    """
    return librosa.effects.preemphasis(signal, coef=coef)


def apply_hpss(signal: np.ndarray, margin: float = 2.0) -> np.ndarray:
    """Separacja harmoniczno-perkusyjna (HPSS) - zwraca składową harmoniczną.

    Składowa harmoniczna zawiera tonalne elementy mowy (formanty, samogłoski),
    natomiast składowa perkusyjna to szumy, stuknięcia i tło. Dla SER
    składowa harmoniczna jest zwykle bardziej informatywna.

    Args:
        signal: Sygnał audio 1D.
        margin: Parametr margin dla separacji (domyślnie 2.0).

    Returns:
        Składowa harmoniczna sygnału.
    """
    harmonic, _percussive = librosa.effects.hpss(signal, margin=margin)
    return harmonic


def apply_noise_reduction(
    signal: np.ndarray,
    sr: int,
    metric: str = "cosine",
) -> np.ndarray:
    """Redukuje szum tła metodą NMF (`librosa.decompose.nn_filter`).

    Filtr `nn_filter` usuwa powtarzalne, nie-tonalne wzorce ze spektrogramu
    (tło akustyczne), zachowując oryginalną fazę sygnału. Szczególnie
    przydatne dla nagrań crowd-sourced (CREMA-D), gdzie jakość mikrofonów
    jest zróżnicowana.

    Args:
        signal: Sygnał audio 1D.
        sr: Częstotliwość próbkowania.
        metric: Metryka podobieństwa dla `nn_filter` (domyślnie "cosine").

    Returns:
        Sygnał po redukcji szumu.
    """
    stft = librosa.stft(signal)
    magnitude = np.abs(stft)
    magnitude_filtered = librosa.decompose.nn_filter(
        magnitude, aggregate=np.median, metric=metric
    )
    # Zachowujemy oryginalną fazę i odtwarzamy sygnał w dziedzinie czasu.
    phase = np.exp(1j * np.angle(stft))
    stft_clean = magnitude_filtered * phase
    return librosa.istft(stft_clean, length=len(signal))


def apply_peak_normalization(signal: np.ndarray, peak: float = 0.95) -> np.ndarray:
    """Normalizuje amplitudę sygnału do zadanej wartości szczytowej.

    Args:
        signal: Sygnał audio 1D.
        peak: Docelowa maksymalna amplituda (domyślnie 0.95).

    Returns:
        Sygnał znormalizowany amplitudowo.
    """
    max_abs = np.max(np.abs(signal))
    if max_abs == 0:
        return signal
    return signal * (peak / max_abs)


# Rejestr filtrów: nazwa -> (funkcja, opis). Umożliwia składanie filtrów
# w dowolny łańcuch (patrz `apply_filter_chain`).
FILTER_REGISTRY: dict[str, tuple] = {
    "vad": (apply_vad, "VAD - wycięcie ciszy"),
    "preemphasis": (apply_preemphasis, "Preemfaza - wyostrzenie formantów"),
    "hpss": (apply_hpss, "HPSS - składowa harmoniczna"),
    "noise_reduction": (apply_noise_reduction, "NMF - redukcja szumu tła"),
    "normalize": (apply_peak_normalization, "Normalizacja amplitudy"),
}


def apply_filter_chain(
    signal: np.ndarray,
    filter_names: list[str],
    sr: int,
) -> np.ndarray:
    """Stosuje sekwencję filtrów w podanej kolejności.

    Args:
        signal: Sygnał audio 1D.
        filter_names: Lista kluczy z `FILTER_REGISTRY`.
        sr: Częstotliwość próbkowania (potrzebna dla redukcji szumu).

    Returns:
        Sygnał po przetworzeniu całego łańcucha filtrów.

    Raises:
        ValueError: Gdy podano nieznaną nazwę filtru.
    """
    result = signal
    for name in filter_names:
        if name not in FILTER_REGISTRY:
            raise ValueError(
                f"Nieznany filtr: '{name}'. Dostępne: {list(FILTER_REGISTRY.keys())}"
            )
        fn, _ = FILTER_REGISTRY[name]
        # Redukcja szumu wymaga sr; pozostałe filtry nie.
        if name == "noise_reduction":
            result = fn(result, sr)
        else:
            result = fn(result)
    return result


def _mel_to_db(signal: np.ndarray, config: AudioProcessingConfig) -> np.ndarray:
    """Oblicza Mel-spektrogram i konwertuje go do skali decybelowej."""
    mel = librosa.feature.melspectrogram(
        y=signal,
        sr=config.sample_rate,
        n_mels=config.n_mels,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
    )
    return librosa.power_to_db(mel, ref=np.max)


def generate_comparative_spectrograms(
    audio_path: str,
    title: str,
    save_path: Optional[str] = None,
    config: Optional[AudioProcessingConfig] = None,
    top_db: float = 20.0,
    preemphasis_coef: float = 0.97,
    fmax: float = 8000.0,
) -> None:
    """Generuje wykres porównawczy: oryginalny vs przefiltrowany spektrogram.

    Filtry zastosowane do wersji "po filtracji":
        1. VAD - wycięcie ciszy (`librosa.effects.trim`),
        2. Preemfaza - wyostrzenie wysokich tonów (`librosa.effects.preemphasis`).

    Args:
        audio_path: Ścieżka do pliku .wav.
        title: Tytuł (np. "RAVDESS - Smutek").
        save_path: Jeśli podane, zapisuje wykres do pliku (PNG) zamiast
            wyświetlać okno. Zalecane na maszynach bez GUI.
        config: Konfiguracja audio (domyślnie `AudioProcessingConfig()`).
        top_db: Próg ciszy dla VAD (dB).
        preemphasis_coef: Współczynnik preemfazy.
        fmax: Maksymalna częstotliwość na osi Y (Hz).

    Raises:
        FileNotFoundError: Gdy plik `audio_path` nie istnieje.
    """
    config = config or AudioProcessingConfig()

    # 1. Załadowanie oryginalnego dźwięku.
    y, sr = librosa.load(audio_path, sr=config.sample_rate, mono=True)

    # 2. Aplikacja filtrów.
    y_trimmed = apply_vad(y, top_db=top_db)
    y_filtered = apply_preemphasis(y_trimmed, coef=preemphasis_coef)

    # 3. Obliczenie Mel-spektrogramów (oryginalny i po filtracji).
    s_orig_db = _mel_to_db(y, config)
    s_filtered_db = _mel_to_db(y_filtered, config)

    # 4. Generowanie wykresów.
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

    img_orig = librosa.display.specshow(
        s_orig_db, x_axis="time", y_axis="mel", sr=sr, fmax=fmax, ax=axes[0]
    )
    axes[0].set(title=f"Oryginał ({title}) - Widoczna cisza i miękkie krawędzie")

    img_filt = librosa.display.specshow(
        s_filtered_db, x_axis="time", y_axis="mel", sr=sr, fmax=fmax, ax=axes[1]
    )
    axes[1].set(
        title=f"Po filtracji (VAD + preemfaza) - Skupienie na słowie, ostre formanty"
    )

    # Rezerwujemy miejsce na wykresy (right=0.85), a pasek koloru
    # tworzymy jako OSOBNĄ oś o jawnie podanym położeniu - dzięki temu
    # legenda nigdy nie najeżdża na spektrogramy.
    fig.subplots_adjust(left=0.08, right=0.85, top=0.95, bottom=0.08, hspace=0.45)
    cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
    fig.colorbar(img_filt, cax=cbar_ax, format="%+2.0f dB")

    if save_path:
        output = Path(save_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Wykres zapisany: {output}")
    else:
        plt.show()


def generate_filter_chain_comparison(
    audio_path: str,
    title: str,
    filter_names: list[str],
    save_path: Optional[str] = None,
    config: Optional[AudioProcessingConfig] = None,
    fmax: float = 8000.0,
) -> None:
    """Wizualizuje każdy etap łańcucha filtrów w siatce 2 kolumn (2xN).

    Układ 2-kolumnowy zapewnia przejrzystość przy większej liczbie
    etapów: pierwszy panel to zawsze sygnał oryginalny, a kolejne
    pokazują efekt każdego filtru w łańcuchu. Przy nieparzystej liczbie
    paneli ostatnia (nieużywana) oś jest ukrywana.

    Args:
        audio_path: Ścieżka do pliku .wav.
        title: Tytuł.
        filter_names: Lista kluczy filtrów z `FILTER_REGISTRY`.
        save_path: Jeśli podane, zapisuje wykres do pliku (PNG).
        config: Konfiguracja audio.
        fmax: Maksymalna częstotliwość na osi Y (Hz).

    Raises:
        ValueError: Gdy podano nieznany filtr.
    """
    config = config or AudioProcessingConfig()
    y, sr = librosa.load(audio_path, sr=config.sample_rate, mono=True)

    # Budowanie etapów: sygnał oryginalny + wynik po każdym filtrze.
    stages = [("Oryginał", y)]
    current = y
    for name in filter_names:
        if name not in FILTER_REGISTRY:
            raise ValueError(
                f"Nieznany filtr: '{name}'. Dostępne: {list(FILTER_REGISTRY.keys())}"
            )
        fn, desc = FILTER_REGISTRY[name]
        if name == "noise_reduction":
            current = fn(current, sr)
        else:
            current = fn(current)
        stages.append((desc, current))

    # Układ 2-kolumnowy: liczba wierszy = ceil(n_etapów / 2).
    n_cols = 2
    n_rows = (len(stages) + 1) // 2
    fig, axes = plt.subplots(
        nrows=n_rows, ncols=n_cols, figsize=(16, 5 * n_rows)
    )
    axes = axes.flatten()

    img = None
    for ax, (stage_title, signal) in zip(axes, stages):
        mel_db = _mel_to_db(signal, config)
        img = librosa.display.specshow(
            mel_db, x_axis="time", y_axis="mel", sr=sr, fmax=fmax, ax=ax
        )
        ax.set(title=f"{stage_title} ({title})")

    # Ukryj nieużywane osie (gdy liczba etapów jest nieparzysta).
    for ax in axes[len(stages):]:
        ax.axis("off")

    # Pasek koloru jako osobna oś (nie najeżdża na wykresy).
    fig.subplots_adjust(
        left=0.06, right=0.86, top=0.95, bottom=0.05, hspace=0.35, wspace=0.25
    )
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(img, cax=cbar_ax, format="%+2.0f dB")

    if save_path:
        output = Path(save_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Wykres zapisany: {output}")
    else:
        plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Porównanie Mel-spektrogramów (oryginał vs VAD+preemfaza)"
    )
    parser.add_argument("audio_path", help="Ścieżka do pliku .wav")
    parser.add_argument("title", help="Tytuł wykresu")
    parser.add_argument(
        "--save", default=None, help="Opcjonalna ścieżka zapisu wykresu (PNG)"
    )
    parser.add_argument("--fmax", type=float, default=8000.0)
    parser.add_argument(
        "--chain",
        default=None,
        help="Łańcuch filtrów oddzielony przecinkami, np. 'vad,preemphasis,normalize'",
    )
    args = parser.parse_args()

    if args.chain:
        generate_filter_chain_comparison(
            args.audio_path,
            args.title,
            filter_names=[f.strip() for f in args.chain.split(",") if f.strip()],
            save_path=args.save,
            fmax=args.fmax,
        )
    else:
        generate_comparative_spectrograms(
            args.audio_path, args.title, save_path=args.save, fmax=args.fmax
        )
