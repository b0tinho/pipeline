"""
Przetwarzanie sygnału audio: wczytywanie plików, ujednolicanie długości
nagrań (padding / truncation) oraz ekstrakcja cech akustycznych
(MFCC lub Mel-spektrogram).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import librosa
import numpy as np

FeatureType = Literal["mfcc", "melspectrogram"]


@dataclass
class AudioProcessingConfig:
    """Konfiguracja przetwarzania sygnału audio.

    Ta sama konfiguracja jest współdzielona przez wszystkie datasety w
    obrębie jednego uruchomienia pipeline'u (dla porównywalności
    eksperymentów), ale każdy dataset jest przetwarzany z jej użyciem
    całkowicie niezależnie.

    Attributes:
        sample_rate: Docelowa częstotliwość próbkowania (Hz).
        duration_seconds: Ustalona długość nagrania w sekundach, do
            której przycinane/dopełniane są wszystkie próbki.
        n_mfcc: Liczba współczynników MFCC (używana, gdy
            feature_type="mfcc").
        n_mels: Liczba pasm Mel (używana, gdy
            feature_type="melspectrogram").
        n_fft: Rozmiar okna FFT.
        hop_length: Przesunięcie okna analizy (w próbkach).
        feature_type: Rodzaj ekstrahowanych cech: "melspectrogram" lub
            "mfcc". Domyślną konfiguracją pipeline'u jest
            Mel-spektrogram - lepiej niż MFCC zachowuje on informację
            o widmie (MFCC odrzuca część informacji poprzez kompresję
            DCT), co zazwyczaj przekłada się na wyższą skuteczność w
            zadaniach SER przy architekturach opartych o CNN/CRNN.
    """

    sample_rate: int = 16_000
    duration_seconds: float = 3.0
    n_mfcc: int = 40
    n_mels: int = 128
    n_fft: int = 1024
    hop_length: int = 256
    feature_type: FeatureType = "melspectrogram"

    @property
    def target_length_samples(self) -> int:
        """Docelowa liczba próbek odpowiadająca ustalonej długości nagrania."""
        return int(self.sample_rate * self.duration_seconds)

    @property
    def feature_dim(self) -> int:
        """Wymiar częstotliwościowy wektora cech (zależny od wybranego typu)."""
        return self.n_mfcc if self.feature_type == "mfcc" else self.n_mels


def load_audio(file_path: str, sample_rate: int) -> np.ndarray:
    """Wczytuje plik audio i resampluje go do zadanej częstotliwości.

    Args:
        file_path: Ścieżka do pliku audio.
        sample_rate: Docelowa częstotliwość próbkowania (Hz).

    Returns:
        Jednowymiarowa tablica próbek (mono) typu float32.
    """
    signal, _ = librosa.load(file_path, sr=sample_rate, mono=True)
    return signal.astype(np.float32)


def pad_or_truncate(signal: np.ndarray, target_length: int) -> np.ndarray:
    """Ujednolica długość sygnału poprzez zerowy padding lub ucięcie.

    Krótsze nagrania są dopełniane ciszą (zerami) na końcu, dłuższe są
    przycinane do zadanej liczby próbek.

    Args:
        signal: Wejściowy sygnał 1D.
        target_length: Docelowa liczba próbek.

    Returns:
        Sygnał o długości dokładnie `target_length` próbek.
    """
    current_length = signal.shape[0]
    if current_length == target_length:
        return signal
    if current_length > target_length:
        return signal[:target_length]
    padding = np.zeros(target_length - current_length, dtype=signal.dtype)
    return np.concatenate([signal, padding])


def extract_features(signal: np.ndarray, config: AudioProcessingConfig) -> np.ndarray:
    """Ekstrahuje cechy akustyczne z sygnału - MFCC lub Mel-spektrogram.

    Args:
        signal: Sygnał audio 1D o ustalonej długości (patrz
            `pad_or_truncate`).
        config: Konfiguracja przetwarzania audio, decydująca m.in.
            o typie ekstrahowanych cech (`config.feature_type`).

    Returns:
        Macierz cech 2D o kształcie (n_features, n_time_frames).

    Raises:
        ValueError: Gdy `config.feature_type` nie jest rozpoznane.
    """
    if config.feature_type == "mfcc":
        features = librosa.feature.mfcc(
            y=signal,
            sr=config.sample_rate,
            n_mfcc=config.n_mfcc,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
        )
    elif config.feature_type == "melspectrogram":
        mel_spec = librosa.feature.melspectrogram(
            y=signal,
            sr=config.sample_rate,
            n_mels=config.n_mels,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
        )
        features = librosa.power_to_db(mel_spec, ref=np.max)
    else:
        raise ValueError(f"Nieznany typ cech: {config.feature_type}")

    return features.astype(np.float32)


def preprocess_and_extract(file_path: str, config: AudioProcessingConfig) -> np.ndarray:
    """Wykonuje pełny potok przetwarzania pojedynczego pliku audio.

    Kolejność kroków: wczytanie -> ujednolicenie długości -> ekstrakcja
    cech.

    Args:
        file_path: Ścieżka do pliku audio.
        config: Konfiguracja przetwarzania audio.

    Returns:
        Macierz cech 2D gotowa do podania na wejście sieci neuronowej.
    """
    signal = load_audio(file_path, config.sample_rate)
    signal = pad_or_truncate(signal, config.target_length_samples)
    return extract_features(signal, config)
