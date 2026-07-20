"""
Proste metody augmentacji surowego sygnału audio.

Zgodnie z założeniami projektu, augmentacja jest aplikowana WYŁĄCZNIE do
zbioru treningowego aktualnie przetwarzanego datasetu - klasy z tego
modułu nie powinny być używane w kontekście zbiorów walidacyjnego ani
testowego (patrz `dataset.SpeechEmotionDataset`, gdzie `augmenter` jest
przekazywany tylko dla splitu train).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import librosa
import numpy as np


@dataclass
class AugmentationConfig:
    """Konfiguracja augmentacji danych audio.

    Attributes:
        apply_probability: Prawdopodobieństwo zaaplikowania jakiejkolwiek
            augmentacji do danej próbki treningowej.
        noise_factor: Amplituda białego szumu dodawanego do sygnału.
        pitch_shift_semitones_range: Zakres losowego przesunięcia
            wysokości tonu, wyrażony w półtonach (min, max).
        time_stretch_rate_range: Zakres losowego współczynnika zmiany
            tempa (1.0 = bez zmian, <1.0 = wolniej, >1.0 = szybciej).
    """

    apply_probability: float = 0.5
    noise_factor: float = 0.005
    pitch_shift_semitones_range: tuple[float, float] = field(default=(-2.0, 2.0))
    time_stretch_rate_range: tuple[float, float] = field(default=(0.9, 1.1))


class AudioAugmenter:
    """Aplikuje losowo wybraną augmentację (lub żadną) do sygnału audio.

    Dostępne metody: wstrzykiwanie białego szumu, zmiana wysokości tonu
    (pitch shift) oraz zmiana tempa (time stretch).
    """

    def __init__(self, config: AugmentationConfig, sample_rate: int) -> None:
        """
        Args:
            config: Konfiguracja augmentacji.
            sample_rate: Częstotliwość próbkowania sygnałów wejściowych.
        """
        self.config = config
        self.sample_rate = sample_rate

    def add_white_noise(self, signal: np.ndarray) -> np.ndarray:
        """Dodaje biały szum gaussowski o zadanej amplitudzie.

        Args:
            signal: Sygnał wejściowy 1D.

        Returns:
            Sygnał z dodanym szumem.
        """
        noise = np.random.randn(len(signal)).astype(np.float32)
        return signal + self.config.noise_factor * noise

    def pitch_shift(self, signal: np.ndarray) -> np.ndarray:
        """Przesuwa wysokość tonu o losową liczbę półtonów.

        Args:
            signal: Sygnał wejściowy 1D.

        Returns:
            Sygnał po przesunięciu wysokości tonu.
        """
        low, high = self.config.pitch_shift_semitones_range
        n_steps = random.uniform(low, high)
        shifted = librosa.effects.pitch_shift(
            y=signal, sr=self.sample_rate, n_steps=n_steps
        )
        return shifted.astype(np.float32)

    def time_stretch(self, signal: np.ndarray) -> np.ndarray:
        """Zmienia tempo nagrania bez zmiany wysokości tonu.

        Args:
            signal: Sygnał wejściowy 1D.

        Returns:
            Sygnał po zmianie tempa (o innej długości niż wejściowy -
            docelowa długość jest ujednolicana później przez
            `audio_processing.pad_or_truncate`).
        """
        low, high = self.config.time_stretch_rate_range
        rate = random.uniform(low, high)
        stretched = librosa.effects.time_stretch(y=signal, rate=rate)
        return stretched.astype(np.float32)

    def apply_random(self, signal: np.ndarray) -> np.ndarray:
        """Z prawdopodobieństwem `apply_probability` aplikuje jedną,
        losowo wybraną metodę augmentacji spośród dostępnych. W
        przeciwnym razie zwraca sygnał niezmieniony.

        Args:
            signal: Sygnał wejściowy 1D.

        Returns:
            Sygnał (ewentualnie) poddany augmentacji.
        """
        if random.random() > self.config.apply_probability:
            return signal

        augmentation_fn = random.choice(
            [self.add_white_noise, self.pitch_shift, self.time_stretch]
        )
        try:
            return augmentation_fn(signal)
        except Exception:
            # Niektóre transformacje librosa mogą zgłosić wyjątek dla
            # skrajnie krótkich/cichych sygnałów - w takim wypadku
            # bezpiecznie zwracamy oryginalny, nienaruszony sygnał.
            return signal
