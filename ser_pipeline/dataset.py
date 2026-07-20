"""
Podział danych (train/val/test) ze stratyfikacją oraz niestandardowa
klasa `torch.utils.data.Dataset` i budowa DataLoaderów dla pojedynczego,
niezależnie przetwarzanego zbioru danych SER.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset

from .audio_processing import (
    AudioProcessingConfig,
    extract_features,
    load_audio,
    pad_or_truncate,
)
from .augmentation import AudioAugmenter
from .data_parsing import map_to_coarse_emotion


def stratified_split(
    dataframe: pd.DataFrame,
    label_column: str = "emotion",
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Dzieli DataFrame JEDNEGO datasetu na zbiory train/val/test z
    zachowaniem proporcji klas emocji (podział stratyfikowany).

    Podział wykonywany jest dwuetapowo:
        1. Wydzielenie zbioru testowego z całości danych.
        2. Wydzielenie zbioru walidacyjnego z pozostałej części.

    W obu krokach stosowana jest stratyfikacja względem kolumny z
    etykietami, dzięki czemu rozkład klas emocji w train/val/test
    pozostaje zbliżony do rozkładu w oryginalnym zbiorze.

    Args:
        dataframe: Ramka danych pojedynczego datasetu (nigdy połączona
            z innymi datasetami).
        label_column: Nazwa kolumny z etykietami emocji.
        test_size: Ułamek całości danych przeznaczony na zbiór testowy.
        val_size: Ułamek CAŁEGO zbioru (nie pozostałości) przeznaczony
            na zbiór walidacyjny.
        random_state: Ziarno losowości dla powtarzalności podziału.

    Returns:
        Krotka (train_df, val_df, test_df) z zresetowanymi indeksami.
    """
    train_val_df, test_df = train_test_split(
        dataframe,
        test_size=test_size,
        stratify=dataframe[label_column],
        random_state=random_state,
    )
    relative_val_size = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        stratify=train_val_df[label_column],
        random_state=random_state,
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


class SpeechEmotionDataset(Dataset):
    """Niestandardowy PyTorch `Dataset` obsługujący pojedynczy split
    (train/val/test) JEDNEGO zbioru danych SER.

    Odpowiada za: wczytanie pliku audio, opcjonalną augmentację (tylko
    dla splitu treningowego), ujednolicenie długości nagrania oraz
    ekstrakcję cech (MFCC lub Mel-spektrogram) wraz ze standaryzacją.

    Każda próbka zwraca czwórkę:
        (features, fine_label, coarse_label, gender_label)
    gdzie:
        - `fine_label` - oryginalna klasa emocji datasetu,
        - `coarse_label` - kategoria walencji (Negative/Neutral/Positive),
        - `gender_label` - Female/Male (na podst. nazwy pliku).

    Etykiety pomocnicze (`coarse_label`, `gender_label`) są używane
    wyłącznie przez modele wielozadaniowe (`HierarchicalMultiTaskCRNN`,
    `GenderAwareSERModel`) w połączonej funkcji straty -
    patrz `engine._run_one_epoch`. Pozostałe modele je ignorują.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        label_encoder: LabelEncoder,
        coarse_label_encoder: LabelEncoder,
        gender_label_encoder: LabelEncoder,
        audio_config: AudioProcessingConfig,
        augmenter: Optional[AudioAugmenter] = None,
    ) -> None:
        """
        Args:
            dataframe: Ramka danych ze ścieżkami plików i etykietami,
                należąca do JEDNEGO datasetu i JEDNEGO splitu.
            label_encoder: `LabelEncoder` dopasowany do pełnego zestawu
                etykiet precyzyjnych danego datasetu (współdzielony
                między splitami, aby indeksy klas były spójne).
            coarse_label_encoder: `LabelEncoder` dopasowany do kategorii
                zgrubnych (walencji) wyprowadzonych z etykiet precyzyjnych
                tego datasetu.
            gender_label_encoder: `LabelEncoder` dopasowany do etykiet
                płci (Female/Male) wyprowadzonych z nazw plików.
            audio_config: Konfiguracja przetwarzania audio.
            augmenter: Obiekt augmentacji - należy podać WYŁĄCZNIE dla
                splitu treningowego. Dla walidacyjnego/testowego
                pozostawić None.
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.label_encoder = label_encoder
        self.coarse_label_encoder = coarse_label_encoder
        self.gender_label_encoder = gender_label_encoder
        self.audio_config = audio_config
        self.augmenter = augmenter

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.dataframe.iloc[index]

        signal = load_audio(row["file_path"], self.audio_config.sample_rate)
        if self.augmenter is not None:
            signal = self.augmenter.apply_random(signal)

        signal = pad_or_truncate(signal, self.audio_config.target_length_samples)
        features = extract_features(signal, self.audio_config)

        # Standaryzacja cech (per-próbka) - stabilizuje i przyspiesza trening.
        mean, std = features.mean(), features.std() + 1e-8
        features = (features - mean) / std

        feature_tensor = torch.from_numpy(features).unsqueeze(0)  # (1, F, T)

        # Etykieta precyzyjna (fine) - oryginalna emocja.
        label_index = int(self.label_encoder.transform([row["emotion"]])[0])
        label_tensor = torch.tensor(label_index, dtype=torch.long)

        # Etykieta zgrubna (coarse) - walencja.
        coarse_emotion = map_to_coarse_emotion(row["emotion"])
        coarse_index = int(self.coarse_label_encoder.transform([coarse_emotion])[0])
        coarse_label_tensor = torch.tensor(coarse_index, dtype=torch.long)

        # Etykieta płci (gender) - Female/Male.
        gender_index = int(self.gender_label_encoder.transform([row["gender"]])[0])
        gender_label_tensor = torch.tensor(gender_index, dtype=torch.long)

        return feature_tensor, label_tensor, coarse_label_tensor, gender_label_tensor


@dataclass
class DataLoaders:
    """Kontener na DataLoadery train/val/test wraz z metadanymi datasetu."""

    train: DataLoader
    val: DataLoader
    test: DataLoader
    label_encoder: LabelEncoder
    num_classes: int
    coarse_label_encoder: LabelEncoder
    num_coarse_classes: int
    gender_label_encoder: LabelEncoder
    num_gender_classes: int


def build_dataloaders(
    dataframe: pd.DataFrame,
    audio_config: AudioProcessingConfig,
    augmenter: Optional[AudioAugmenter],
    batch_size: int = 32,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    num_workers: int = 0,
) -> DataLoaders:
    """Buduje pełny zestaw DataLoaderów (train/val/test) dla JEDNEGO
    datasetu, ze stratyfikowanym podziałem klas i augmentacją
    zastosowaną wyłącznie do zbioru treningowego.

    Args:
        dataframe: DataFrame pojedynczego datasetu (wynik
            `data_parsing.load_dataset_dataframe`).
        audio_config: Konfiguracja przetwarzania audio.
        augmenter: Obiekt augmentacji stosowany tylko do splitu train
            (może być None, jeśli augmentacja ma być wyłączona).
        batch_size: Rozmiar batcha DataLoaderów.
        test_size: Ułamek danych na zbiór testowy.
        val_size: Ułamek danych na zbiór walidacyjny.
        random_state: Ziarno losowości.
        num_workers: Liczba procesów roboczych DataLoadera.

    Returns:
        Obiekt `DataLoaders` z train/val/test DataLoaderami, dopasowanym
        `LabelEncoder` (etykiety precyzyjne), `coarse_label_encoder`
        (etykiety walencji) oraz `gender_label_encoder` (płeć z nazwy
        pliku, wyprowadzona przez `data_parsing.GENDER_PARSERS`), wraz z
        liczbą unikalnych klas w każdej taksonomii.
    """
    label_encoder = LabelEncoder()
    label_encoder.fit(dataframe["emotion"])

    coarse_label_encoder = LabelEncoder()
    coarse_label_encoder.fit(dataframe["emotion"].map(map_to_coarse_emotion))

    gender_label_encoder = LabelEncoder()
    gender_label_encoder.fit(dataframe["gender"])

    train_df, val_df, test_df = stratified_split(
        dataframe,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
    )

    train_dataset = SpeechEmotionDataset(
        train_df,
        label_encoder,
        coarse_label_encoder,
        gender_label_encoder,
        audio_config,
        augmenter=augmenter,
    )
    val_dataset = SpeechEmotionDataset(
        val_df,
        label_encoder,
        coarse_label_encoder,
        gender_label_encoder,
        audio_config,
        augmenter=None,
    )
    test_dataset = SpeechEmotionDataset(
        test_df,
        label_encoder,
        coarse_label_encoder,
        gender_label_encoder,
        audio_config,
        augmenter=None,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return DataLoaders(
        train=train_loader,
        val=val_loader,
        test=test_loader,
        label_encoder=label_encoder,
        num_classes=len(label_encoder.classes_),
        coarse_label_encoder=coarse_label_encoder,
        num_coarse_classes=len(coarse_label_encoder.classes_),
        gender_label_encoder=gender_label_encoder,
        num_gender_classes=len(gender_label_encoder.classes_),
    )


# ---------------------------------------------------------------------------
# Tryb COMBINED: laczenie wszystkich datasetow z ochrona przed speaker leakage
# ---------------------------------------------------------------------------
COMMON_EMOTION_CLASSES = [
    "Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad",
]


def build_combined_dataloaders(
    dataset_root_paths: dict[str, str],
    audio_config: AudioProcessingConfig,
    augmenter: Optional[AudioAugmenter],
    batch_size: int = 32,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    num_workers: int = 0,
) -> DataLoaders:
    """Buduje DataLoadery na POLACZONYCH danych z wszystkich datasetow.

    Kazdy zbior jest NAJPIERW dzielony na train/val/test OSOBNO
    (stratyfikacja wewnatrz datasetu), a dopiero potem odpowiednie
    splity sa laczone. Dzieki temu glos tego samego aktora nigdy nie
    trafia jednoczesnie do train i test (ochrona przed speaker leakage).

    Wszystkie datasety sa filtrowane do wspolnych 6 klas emocji
    (COMMON_EMOTION_CLASSES) - klasa Surprise jest usuwana z TESS,
    RAVDESS i SAVEE, poniewaz nie wystepuje w CREMA-D.
    """
    from .data_parsing import load_dataset_dataframe  # noqa: F811

    all_train_dfs = []
    all_val_dfs = []
    all_test_dfs = []

    for dataset_name, root_dir in dataset_root_paths.items():
        df = load_dataset_dataframe(dataset_name, root_dir)

        n_before = len(df)
        df = df[df["emotion"].isin(COMMON_EMOTION_CLASSES)].copy()
        n_removed = n_before - len(df)
        if n_removed > 0:
            print(
                f"[COMBINED] {dataset_name}: usunieto {n_removed} probek"
            )
        if df.empty:
            raise RuntimeError(
                f"Po filtracji dataset {dataset_name} jest pusty"
            )

        train_df, val_df, test_df = stratified_split(
            df, label_column="emotion",
            test_size=test_size, val_size=val_size, random_state=random_state,
        )
        all_train_dfs.append(train_df)
        all_val_dfs.append(val_df)
        all_test_dfs.append(test_df)

    combined_train = pd.concat(all_train_dfs, ignore_index=True)
    combined_val = pd.concat(all_val_dfs, ignore_index=True)
    combined_test = pd.concat(all_test_dfs, ignore_index=True)

    print(
        f"[COMBINED] train={len(combined_train)} "
        f"val={len(combined_val)} test={len(combined_test)}"
    )

    label_encoder = LabelEncoder()
    label_encoder.fit(COMMON_EMOTION_CLASSES)

    coarse_label_encoder = LabelEncoder()
    coarse_label_encoder.fit(
        pd.Series(COMMON_EMOTION_CLASSES).map(map_to_coarse_emotion)
    )

    gender_label_encoder = LabelEncoder()
    gender_label_encoder.fit(combined_train["gender"])

    train_dataset = SpeechEmotionDataset(
        combined_train, label_encoder, coarse_label_encoder,
        gender_label_encoder, audio_config, augmenter=augmenter,
    )
    val_dataset = SpeechEmotionDataset(
        combined_val, label_encoder, coarse_label_encoder,
        gender_label_encoder, audio_config, augmenter=None,
    )
    test_dataset = SpeechEmotionDataset(
        combined_test, label_encoder, coarse_label_encoder,
        gender_label_encoder, audio_config, augmenter=None,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
    )

    return DataLoaders(
        train=train_loader, val=val_loader, test=test_loader,
        label_encoder=label_encoder,
        num_classes=len(label_encoder.classes_),
        coarse_label_encoder=coarse_label_encoder,
        num_coarse_classes=len(coarse_label_encoder.classes_),
        gender_label_encoder=gender_label_encoder,
        num_gender_classes=len(gender_label_encoder.classes_),
    )
