"""
Główny orchestrator pipeline'u SER.

Uruchamia PEŁNY, NIEZALEŻNY proces (parsowanie danych -> stratyfikowany
podział -> augmentacja -> budowa modelu -> trening -> ewaluacja ->
raportowanie) dla KAŻDEJ kombinacji (dataset, architektura modelu):
cztery zbiory danych (TESS, RAVDESS, SAVEE, CREMA-D) x cztery
architektury z `model.MODEL_REGISTRY` (`dscnn`, `headfusion_acnn`,
`lstm`, `hierarchical_crnn`) - w sumie 16 niezależnych eksperymentów.

Datasety NIGDY nie są ze sobą łączone - każdy eksperyment operuje na
własnym DataFrame, własnym `LabelEncoder`, własnym modelu (z dopasowaną
do liczby klas warstwą wyjściową) i zapisuje wyniki do osobnego
podfolderu `results/per_dataset<WARIANT>/<NAZWA_DATASETU>/<NAZWA_MODELU>/`,
dzięki czemu wyniki różnych architektur na tym samym datasecie nigdy się
nie nadpisują i można je bezpośrednio porównać. Tryb combined zapisuje do
`results/combined<WARIANT>/<NAZWA_MODELU>/`. `<WARIANT>` to pusty napis
(dane surowe), `_filtry` (`--preprocessed`), `_deltas` (`--deltas`) lub
`_filtry_deltas` (obie flagi) - patrz `results_group_dir`.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .audio_processing import AudioProcessingConfig
from .augmentation import AudioAugmenter, AugmentationConfig
from .data_parsing import load_dataset_dataframe
from .dataset import build_combined_dataloaders, build_dataloaders
from .engine import TrainingConfig, evaluate_and_report, train_model
from .model import MODEL_REGISTRY, build_model

# ---------------------------------------------------------------------------
# Ścieżki do surowych danych. Dostosuj je do lokalizacji, w której
# rozpakowane zostały poszczególne zbiory danych na Twojej maszynie.
# Oczekiwana struktura: dowolna hierarchia folderów zawierająca pliki
# .wav zgodne z konwencją nazewnictwa danego zbioru (patrz
# `data_parsing.py`) - przeszukiwanie jest rekurencyjne.
# ---------------------------------------------------------------------------
DATASET_NAMES = ["TESS", "RAVDESS", "SAVEE", "CREMA-D"]

RAW_DATA_ROOT = "data"
PREPROCESSED_DATA_ROOT = "data_preprocessed"


def _build_root_paths(base_root: str) -> dict[str, str]:
    """Buduje mapę `{dataset: sciezka}` dla wybranego korzenia danych."""
    return {name: f"{base_root}/{name}" for name in DATASET_NAMES}


DATASET_ROOT_PATHS = _build_root_paths(RAW_DATA_ROOT)

RESULTS_DIR = Path("results")

# Ziarno podziału train/val/test. Jest STAŁE i niezależne od `--seed`,
# dzięki czemu wszystkie przebiegi (różne ziarna treningu, warianty
# surowe/filtry/deltas) są oceniane na tym samym zbiorze testowym.
SPLIT_SEED = 42

# Domyślne ziarno treningu: inicjalizacja wag, kolejność batchy,
# losowanie augmentacji. Zmieniane flagą `--seed`.
DEFAULT_TRAINING_SEED = 42


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    """Ustawia ziarno losowości we wszystkich bibliotekach dla
    zapewnienia powtarzalności eksperymentów.

    Wywoływana na początku KAŻDEGO eksperymentu (para dataset/model),
    aby wynik modelu nie zależał od tego, jakie modele były trenowane
    wcześniej w tym samym uruchomieniu.

    Args:
        seed: Wartość ziarna losowości.
        deterministic: Czy wymusić deterministyczne algorytmy cuDNN/CUDA.
            Wyłącza `cudnn.benchmark` (wolniejszy trening), ale ten sam
            seed daje wtedy ten sam wynik. Operacje bez deterministycznej
            implementacji zgłaszają tylko ostrzeżenie (`warn_only=True`).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Wymagane przez cuBLAS dla deterministycznych mnożeń macierzy;
        # musi być ustawione przed pierwszym użyciem cuBLAS.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    elif torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True


def _seeded_train_loader(loader: DataLoader, seed: int) -> DataLoader:
    """Tworzy nowy DataLoader treningowy z własnym generatorem losowości.

    Generator ustala kolejność batchy oraz ziarna procesów roboczych
    (`random`, `numpy`, `torch` w workerach - czyli losowanie
    augmentacji). Nowy loader oznacza też nowe procesy robocze, więc
    przy `persistent_workers=True` stan losowości nie przechodzi
    z jednego modelu na kolejny.

    Args:
        loader: Bazowy DataLoader treningowy (z `build_*_dataloaders`).
        seed: Ziarno treningu.

    Returns:
        DataLoader o tych samych ustawieniach z generatorem `seed`.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    uses_workers = loader.num_workers > 0
    return DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        shuffle=True,
        num_workers=loader.num_workers,
        drop_last=loader.drop_last,
        pin_memory=loader.pin_memory,
        persistent_workers=loader.persistent_workers,
        prefetch_factor=loader.prefetch_factor if uses_workers else None,
        generator=generator,
    )


def _experiment_output_dir(base_dir: Path, seed: int) -> Path:
    """Katalog wyników: domyślny seed -> `base_dir` (jak dotychczas),
    inny seed -> `base_dir/seed_<N>/`, aby przebiegi się nie nadpisywały."""
    if seed == DEFAULT_TRAINING_SEED:
        return base_dir
    return base_dir / f"seed_{seed}"


def results_group_dir(
    combined: bool, use_preprocessed: bool, use_deltas: bool
) -> Path:
    """Katalog grupy wyników zależny od trybu i wariantu danych.

    Każdy wariant ma własny katalog, więc przebiegi surowe, z filtrami
    i z deltami nie nadpisują się nawzajem (układ jak w `wyniki/`):

        results/combined/               results/per_dataset/
        results/combined_filtry/        results/per_dataset_filtry/
        results/combined_deltas/        results/per_dataset_deltas/
        results/combined_filtry_deltas/ results/per_dataset_filtry_deltas/

    Args:
        combined: Tryb połączonych danych.
        use_preprocessed: Dane po VAD + preemfazie.
        use_deltas: Cechy delta i delta-delta.

    Returns:
        Ścieżka katalogu grupy (bez nazwy datasetu i modelu).
    """
    name = "combined" if combined else "per_dataset"
    if use_preprocessed:
        name += "_filtry"
    if use_deltas:
        name += "_deltas"
    return RESULTS_DIR / name


def _save_run_config(
    output_dir: Path,
    seed: int,
    deterministic: bool,
    training_config: TrainingConfig,
    batch_size: int,
    results_group: str,
) -> None:
    """Zapisuje ustawienia przebiegu do `run_config.json`."""
    config = {
        "results_group": results_group,
        "seed": seed,
        "split_seed": SPLIT_SEED,
        "deterministic": deterministic,
        "num_epochs": training_config.num_epochs,
        "batch_size": batch_size,
        "learning_rate": training_config.learning_rate,
        "early_stopping_patience": training_config.early_stopping_patience,
    }
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def run_pipeline_for_dataset(
    dataset_name: str,
    root_dir: str,
    model_name: str,
    audio_config: AudioProcessingConfig,
    augmentation_config: AugmentationConfig,
    training_config: TrainingConfig,
    batch_size: int = 32,
    model_kwargs: dict | None = None,
    seed: int = DEFAULT_TRAINING_SEED,
    deterministic: bool = False,
    results_root: Path | None = None,
) -> None:
    """Uruchamia kompletny, niezależny eksperyment SER dla JEDNEGO
    datasetu i JEDNEJ architektury modelu.

    Kroki (zgodnie z wymaganiami projektu):
        1. Ładowanie i parsowanie danych specyficzne dla `dataset_name`.
        2. Stratyfikowany podział na train/val/test.
        3. Augmentacja audio zastosowana wyłącznie do splitu treningowego.
        4. Budowa modelu (`model_name`) z liczbą klas wyjściowych
           dopasowaną do unikalnych etykiet tego datasetu.
        5. Trening z checkpointingiem najlepszego modelu i early stopping.
        6. Ewaluacja na zbiorze testowym oraz zapis logów, wag modelu i
           macierzy pomyłek do `<results_root>/<dataset_name>/<model_name>/`.

    Args:
        dataset_name: Nazwa datasetu ("TESS", "RAVDESS", "SAVEE" lub
            "CREMA-D").
        root_dir: Katalog z surowymi plikami .wav danego datasetu.
        model_name: Klucz architektury z `model.MODEL_REGISTRY`
            ("dscnn", "headfusion_acnn", "lstm", "hierarchical_crnn").
        audio_config: Konfiguracja przetwarzania audio.
        augmentation_config: Konfiguracja augmentacji (stosowana tylko
            do zbioru treningowego).
        training_config: Hiperparametry treningu.
        batch_size: Rozmiar batcha DataLoaderów.
        model_kwargs: Dodatkowe argumenty konstruktora przekazywane do
            wybranej architektury (np. `{"dropout": 0.4}`).
        seed: Ziarno treningu (wagi, kolejność batchy, augmentacja).
            Podział danych zawsze używa `SPLIT_SEED`.
        deterministic: Czy wymusić deterministyczne algorytmy CUDA.
        results_root: Katalog grupy wyników (domyślnie
            `results/per_dataset/`, patrz `results_group_dir`).
    """
    if results_root is None:
        results_root = results_group_dir(False, False, False)
    print(
        f"\n{'=' * 70}\n"
        f"Eksperyment: dataset={dataset_name} | model={model_name} "
        f"| seed={seed}\n"
        f"{'=' * 70}"
    )
    set_global_seed(seed, deterministic)

    output_dir = _experiment_output_dir(results_root / dataset_name / model_name, seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ładowanie i parsowanie danych - niezależnie dla tego datasetu.
    dataframe = load_dataset_dataframe(dataset_name, root_dir)
    print(
        f"[{dataset_name}/{model_name}] Wczytano {len(dataframe)} próbek, "
        f"klasy: {sorted(dataframe['emotion'].unique())}"
    )

    # 2 i 3. Stratyfikowany podział + augmentacja (aplikowana wewnątrz
    # `SpeechEmotionDataset` wyłącznie do splitu treningowego).
    augmenter = AudioAugmenter(augmentation_config, audio_config.sample_rate)
    data_loaders = build_dataloaders(
        dataframe,
        audio_config=audio_config,
        augmenter=augmenter,
        batch_size=batch_size,
        random_state=SPLIT_SEED,
    )
    class_names = list(data_loaders.label_encoder.classes_)
    print(
        f"[{dataset_name}/{model_name}] Liczba klas: "
        f"{data_loaders.num_classes} -> {class_names}"
    )

    # 4. Budowa modelu z warstwą wyjściową dopasowaną do liczby klas
    # w bieżącym datasecie oraz liczbą kanałów wejściowych zależną od
    # konfiguracji audio (1 lub 3 dla delta/delta-delta).
    model = build_model(
        model_name,
        num_classes=data_loaders.num_classes,
        input_freq_dim=audio_config.feature_dim,
        input_channels=audio_config.num_channels,
        **(model_kwargs or {}),
    )

    # 5. Trening z checkpointingiem i early stopping.
    checkpoint_path = output_dir / "best_model.pt"
    log_path = output_dir / "training_log.csv"
    model = train_model(
        model,
        _seeded_train_loader(data_loaders.train, seed),
        data_loaders.val,
        training_config,
        checkpoint_path=checkpoint_path,
        log_path=log_path,
    )

    # 6. Ewaluacja na zbiorze testowym + zapis raportu i macierzy pomyłek.
    evaluate_and_report(
        model,
        data_loaders.test,
        class_names=class_names,
        device=training_config.device,
        output_dir=output_dir,
        dataset_name=f"{dataset_name} / {model_name}",
        gender_class_names=list(data_loaders.gender_label_encoder.classes_),
    )

    # Zapis mapowania indeks -> etykieta, aby wyniki były w pełni odtwarzalne.
    with open(output_dir / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump({i: name for i, name in enumerate(class_names)}, f, indent=2)
    _save_run_config(
        output_dir, seed, deterministic, training_config, batch_size,
        results_group=results_root.name,
    )

    print(f"[{dataset_name}/{model_name}] Zakończono. Wyniki zapisane w: {output_dir}")



def run_combined_pipeline(
    model_names: list[str],
    dataset_root_paths: dict[str, str],
    audio_config: AudioProcessingConfig,
    augmentation_config: AugmentationConfig,
    training_config: TrainingConfig,
    batch_size: int = 32,
    seed: int = DEFAULT_TRAINING_SEED,
    deterministic: bool = False,
    results_root: Path | None = None,
) -> None:
    """Uruchamia wszystkie wybrane modele na POLACZONYCH danych
    z czterech datasetow (TESS+RAVDESS+SAVEE+CREMA-D).

    Kazdy zbior jest dzielony na train/val/test OSOBNO przed
    polaczeniem (ochrona przed speaker leakage). Wszystkie datasety
    sa filtrowane do 6 wspolnych klas emocji (bez Surprise).
    Wyniki zapisywane sa w <results_root>/<MODEL>/ (domyslnie
    results/combined/<MODEL>/).

    Args:
        model_names: Lista kluczy architektur do wytrenowania.
        dataset_root_paths: Mapa `{dataset: sciezka}` (surowa lub
            preprocessed, zaleznie od `main`).
        audio_config: Konfiguracja audio.
        augmentation_config: Konfiguracja augmentacji.
        training_config: Hiperparametry treningu.
        batch_size: Rozmiar batcha.
        seed: Ziarno treningu, ustawiane od nowa przed KAZDYM modelem
            (razem z nowym loaderem treningowym), wiec wynik modelu nie
            zalezy od tego, jakie modele trenowano przed nim.
        deterministic: Czy wymusic deterministyczne algorytmy CUDA.
        results_root: Katalog grupy wynikow (patrz `results_group_dir`).
    """
    if results_root is None:
        results_root = results_group_dir(True, False, False)
    augmenter = AudioAugmenter(augmentation_config, audio_config.sample_rate)

    data_loaders = build_combined_dataloaders(
        dataset_root_paths,
        audio_config=audio_config,
        augmenter=augmenter,
        batch_size=batch_size,
        random_state=SPLIT_SEED,
    )
    class_names = list(data_loaders.label_encoder.classes_)

    total = len(model_names)
    for idx, model_name in enumerate(model_names, 1):
        print(
            f"\n{'=' * 70}\n"
            f"Eksperyment: dataset=COMBINED | model={model_name} "
            f"| seed={seed} ({idx}/{total})\n"
            f"{'=' * 70}"
        )
        set_global_seed(seed, deterministic)
        output_dir = _experiment_output_dir(results_root / model_name, seed)
        output_dir.mkdir(parents=True, exist_ok=True)

        model = build_model(
            model_name,
            num_classes=data_loaders.num_classes,
            input_freq_dim=audio_config.feature_dim,
            input_channels=audio_config.num_channels,
        )

        checkpoint_path = output_dir / "best_model.pt"
        log_path = output_dir / "training_log.csv"
        model = train_model(
            model, _seeded_train_loader(data_loaders.train, seed), data_loaders.val,
            training_config,
            checkpoint_path=checkpoint_path, log_path=log_path,
        )
        evaluate_and_report(
            model, data_loaders.test,
            class_names=class_names,
            device=training_config.device,
            output_dir=output_dir,
            dataset_name=f"COMBINED / {model_name}",
            gender_class_names=list(data_loaders.gender_label_encoder.classes_),
        )
        with open(output_dir / "label_mapping.json", "w", encoding="utf-8") as f:
            json.dump({i: name for i, name in enumerate(class_names)}, f, indent=2)
        _save_run_config(
            output_dir, seed, deterministic, training_config, batch_size,
            results_group=results_root.name,
        )

        print(f"[COMBINED/{model_name}] Zakończono. Wyniki: {output_dir}")



def main(
    model_names: list[str] | None = None,
    dataset_names: list[str] | None = None,
    combined: bool = False,
    use_preprocessed: bool = False,
    use_deltas: bool = False,
    num_epochs: int = 50,
    batch_size: int = 32,
    seed: int = DEFAULT_TRAINING_SEED,
    deterministic: bool = False,
) -> None:
    """Glowny punkt wejscia pipeline'u.

    Dwa tryby pracy:
    1. Per-dataset (domyslny, combined=False):
       Kazdy model trenowany niezaleznie na kazdym datasecie.
       Wyniki w results/per_dataset<WARIANT>/<DATASET>/<MODEL>/.

    2. Combined (combined=True):
       Wszystkie datasety laczone w jeden zbior (6 wspolnych klas).
       Kazdy model trenowany raz na polaczonych danych.
       Wyniki w results/combined<WARIANT>/<MODEL>/.

    Args:
        model_names: Lista kluczy architektur do przetestowania.
            None = wszystkie.
        dataset_names: Lista datasetow (ignorowane gdy combined=True).
        combined: Czy uzyc trybu polaczonych danych.
        use_preprocessed: Czy czytac dane z `data_preprocessed/`
            (VAD + preemfaza) zamiast surowych `data/`. Wymaga wczesniejszego
            uruchomienia `preprocess_data.py`.
        use_deltas: Czy dodać cechy delta i delta-delta (3 kanały wejściowe)
            obok statycznego Mel-spektrogramu/MFCC.
        num_epochs: Maksymalna liczba epok treningu (early stopping moze
            zakonczyc trening wczesniej).
        batch_size: Rozmiar batcha DataLoaderow.
        seed: Ziarno treningu (wagi, kolejnosc batchy, augmentacja).
            Podzial train/val/test zawsze uzywa `SPLIT_SEED`, wiec test
            jest ten sam dla kazdego seeda. Wyniki dla seeda innego niz
            domyslny trafiaja do podfolderu `seed_<N>/`.
        deterministic: Czy wymusic deterministyczne algorytmy CUDA
            (wolniej, ale ten sam seed daje ten sam wynik).
    """
    # Wybor korzenia danych: surowe vs preprocessed (VAD + preemfaza).
    data_root = PREPROCESSED_DATA_ROOT if use_preprocessed else RAW_DATA_ROOT
    dataset_root_paths = _build_root_paths(data_root)
    results_root = results_group_dir(combined, use_preprocessed, use_deltas)

    selected_models = (
        model_names if model_names is not None else list(MODEL_REGISTRY.keys())
    )
    unknown = set(selected_models) - set(MODEL_REGISTRY.keys())
    if unknown:
        raise ValueError(
            f"Nieznane modele: {sorted(unknown)}. "
            f"Dostepne: {list(MODEL_REGISTRY.keys())}"
        )
    set_global_seed(seed, deterministic)
    print(f"Seed treningu: {seed} | seed podzialu: {SPLIT_SEED} | "
          f"tryb deterministyczny: {'tak' if deterministic else 'nie'} | "
          f"wyniki: {results_root}")

    audio_config = AudioProcessingConfig(
        sample_rate=16_000, duration_seconds=3.0,
        n_mels=128, feature_type="melspectrogram",
        use_deltas=use_deltas,
    )
    augmentation_config = AugmentationConfig(apply_probability=0.5)
    training_config = TrainingConfig(
        num_epochs=num_epochs, learning_rate=1e-3, early_stopping_patience=7,
    )

    if combined:
        print(
            f"Tryb COMBINED: {len(selected_models)} modele na "
            f"polaczonych danych (TESS+RAVDESS+SAVEE+CREMA-D, 6 klas). "
            f"Zrodlo: {data_root}"
        )
        run_combined_pipeline(
            model_names=selected_models,
            dataset_root_paths=dataset_root_paths,
            audio_config=audio_config,
            augmentation_config=augmentation_config,
            training_config=training_config,
            batch_size=batch_size,
            seed=seed, deterministic=deterministic,
            results_root=results_root,
        )
        return

    selected_datasets = (
        dataset_names if dataset_names is not None else DATASET_NAMES
    )
    unknown_ds = set(selected_datasets) - set(DATASET_NAMES)
    if unknown_ds:
        raise ValueError(
            f"Nieznane datasety: {sorted(unknown_ds)}."
        )

    total = len(selected_datasets) * len(selected_models)
    completed = 0
    print(
        f"Plan: {len(selected_datasets)} datasety x "
        f"{len(selected_models)} modele = {total} kombinacji. "
        f"Zrodlo: {data_root}"
    )

    for dataset_name in selected_datasets:
        root_dir = dataset_root_paths[dataset_name]
        if not Path(root_dir).exists():
            print(f"[UWAGA] Katalog '{root_dir}' nie istnieje.")
            continue
        for model_name in selected_models:
            completed += 1
            print(f"\n>>> Kombinacja {completed}/{total}")
            try:
                run_pipeline_for_dataset(
                    dataset_name=dataset_name, root_dir=root_dir,
                    model_name=model_name,
                    audio_config=audio_config,
                    augmentation_config=augmentation_config,
                    training_config=training_config,
                    batch_size=batch_size,
                    seed=seed, deterministic=deterministic,
                    results_root=results_root,
                )
            except Exception as exc:
                print(
                    f"[BLAD] {dataset_name}/{model_name}: {exc}"
                )
                continue


if __name__ == "__main__":
    main()
