"""
Pętla treningowa, ewaluacja, early stopping oraz zapisywanie wyników
(logi treningu, checkpoint najlepszego modelu, raport klasyfikacji i
macierz pomyłek) do dedykowanego folderu wyników pojedynczego datasetu.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class TrainingConfig:
    """Hiperparametry treningu.

    Attributes:
        num_epochs: Maksymalna liczba epok treningu.
        learning_rate: Współczynnik uczenia optymalizatora Adam.
        weight_decay: Regularyzacja L2.
        early_stopping_patience: Liczba epok bez poprawy straty
            walidacyjnej, po której trening jest przerywany.
        device: Urządzenie obliczeniowe ("cuda" lub "cpu").
        multitask_alpha: Waga zadania pomocniczego w połączonej funkcji
            straty modeli wielozadaniowych:
            `L = alpha * L_aux + (1 - alpha) * L_fine`, gdzie `L_aux`
            to strata grupy emocji (`coarse`) dla
            `model.HierarchicalMultiTaskCRNN` lub strata płci dla
            `model.GenderAwareSERModel`.
            Wartość domyślna (0.2) odpowiada optymalnej wadze
            wyznaczonej empirycznie w oryginalnej pracy o architekturze
            3D-HMTL. Parametr jest ignorowany przez modele, które nie
            implementują metody `forward_multitask`.
    """

    num_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    early_stopping_patience: int = 7
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    multitask_alpha: float = 0.2


class EarlyStopping:
    """Zatrzymuje trening, gdy strata walidacyjna przestaje się poprawiać.

    Dodatkowo wskazuje, kiedy bieżąca epoka jest najlepszą dotychczas
    zaobserwowaną - w takim wypadku pętla treningowa zapisuje checkpoint
    wag modelu.
    """

    def __init__(self, patience: int = 7, min_delta: float = 1e-4) -> None:
        """
        Args:
            patience: Liczba epok tolerancji bez poprawy.
            min_delta: Minimalna zmiana straty uznawana za poprawę.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        """Aktualizuje stan na podstawie bieżącej straty walidacyjnej.

        Args:
            val_loss: Strata walidacyjna z bieżącej epoki.

        Returns:
            True, jeśli bieżący wynik jest najlepszym dotychczasowym
            (należy zapisać checkpoint modelu); False w przeciwnym razie.
        """
        is_best = val_loss < (self.best_loss - self.min_delta)
        if is_best:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return is_best


def _run_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    multitask_alpha: float = 0.2,
) -> Tuple[float, float]:
    """Wykonuje jedną epokę treningu (gdy podano optimizer) lub
    ewaluacji (gdy optimizer=None).

    Jeśli model implementuje metodę `forward_multitask` (patrz
    `model.HierarchicalMultiTaskCRNN`), automatycznie używana jest
    połączona funkcja straty:
    `L = multitask_alpha * L_coarse + (1 - multitask_alpha) * L_fine`,
    gdzie `L_coarse`/`L_fine` to strata na etykietach zgrubnych/precyzyjnych
    zwracanych przez `DataLoader` (patrz `dataset.SpeechEmotionDataset`).
    Dokładność jest raportowana zawsze na podstawie predykcji precyzyjnych
    (`fine`), zgodnie z rzeczywistą taksonomią klas danego datasetu.

    Args:
        model: Model sieci neuronowej.
        dataloader: DataLoader dla bieżącego etapu (train/val).
        criterion: Funkcja straty.
        device: Urządzenie obliczeniowe.
        optimizer: Optymalizator; None oznacza tryb ewaluacji (bez
            propagacji wstecznej).
        multitask_alpha: Waga zadania zgrubnego w połączonej funkcji
            straty (patrz `TrainingConfig.multitask_alpha`). Ignorowane
            dla modeli bez metody `forward_multitask`.

    Returns:
        Krotka (średnia strata, dokładność) uśredniona po całej epoce.
    """
    is_training = optimizer is not None
    model.train(mode=is_training)
    is_multitask = hasattr(model, "forward_multitask")

    total_loss = 0.0
    correct = 0
    total = 0

    # Jednorazowe wykrycie typu pomocniczego targetu. Zamiast
    # korzystac z dummy forwarda (ktory wymagalby dopasowania
    # wymiaru tensoru testowego do konkretnego modelu), sprawdzamy
    # liczbe neuronow wyjsciowych ostatniej warstwy liniowej
    # pomocniczej glowy: 2 -> gender (Female/Male),
    # 3 -> coarse/valence.
    aux_head_dim = 1  # fallback: uzyjemy coarse (3 klasy)
    if is_multitask:
        if hasattr(model, "gender_head"):
            last_linear = None
            for m in model.gender_head.modules():
                if isinstance(m, nn.Linear):
                    last_linear = m
            if last_linear is not None and last_linear.out_features == 2:
                aux_head_dim = 2  # gender
            else:
                aux_head_dim = 3  # coarse
        else:
            aux_head_dim = 3  # coarse (hierarchical_crnn)

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for features, labels, coarse_labels, gender_labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)

            if is_training:
                optimizer.zero_grad()

            if is_multitask:
                aux_logits, fine_logits = model.forward_multitask(features)
                if aux_head_dim == 2:
                    aux_targets = gender_labels.to(device)
                else:
                    aux_targets = coarse_labels.to(device)
                loss_aux = criterion(aux_logits, aux_targets)
                loss_fine = criterion(fine_logits, labels)
                loss = multitask_alpha * loss_aux + (1.0 - multitask_alpha) * loss_fine
                outputs = fine_logits
            else:
                outputs = model(features)
                loss = criterion(outputs, labels)

            if is_training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * features.size(0)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    training_config: TrainingConfig,
    checkpoint_path: Path,
    log_path: Path,
) -> nn.Module:
    """Pełna pętla treningowa z model checkpointingiem i early stopping.

    Po każdej epoce ewaluowana jest strata walidacyjna; jeśli jest ona
    najlepszą dotychczas zaobserwowaną, wagi modelu są zapisywane do
    `checkpoint_path`. Trening jest przerywany wcześniej, jeśli strata
    walidacyjna nie poprawia się przez `early_stopping_patience` epok.

    Dla modeli wielozadaniowych (implementujących `forward_multitask`,
    np. `model.HierarchicalMultiTaskCRNN`) strata treningowa i
    walidacyjna jest automatycznie liczona jako połączenie strat zadania
    zgrubnego i precyzyjnego, zgodnie z `training_config.multitask_alpha`
    (patrz `_run_one_epoch`). Dla pozostałych modeli zachowanie jest
    identyczne jak dotychczas (zwykła strata jednozadaniowa).

    Args:
        model: Model do wytrenowania (przenoszony na `training_config.device`).
        train_loader: DataLoader zbioru treningowego (z augmentacją).
        val_loader: DataLoader zbioru walidacyjnego (bez augmentacji).
        training_config: Hiperparametry treningu.
        checkpoint_path: Ścieżka zapisu wag najlepszego modelu (.pt).
        log_path: Ścieżka zapisu logu treningu w formacie CSV.

    Returns:
        Model z wagami wczytanymi z checkpointu najlepszej epoki.
    """
    device = training_config.device
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    early_stopping = EarlyStopping(patience=training_config.early_stopping_patience)

    history: List[dict] = []
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, training_config.num_epochs + 1):
        train_loss, train_acc = _run_one_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            multitask_alpha=training_config.multitask_alpha,
        )
        val_loss, val_acc = _run_one_epoch(
            model,
            val_loader,
            criterion,
            device,
            multitask_alpha=training_config.multitask_alpha,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        is_best = early_stopping.step(val_loss)
        if is_best:
            torch.save(model.state_dict(), checkpoint_path)

        if early_stopping.should_stop:
            print(f"Early stopping zatrzymał trening na epoce {epoch}.")
            break

    log_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(log_path, index=False)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model


def evaluate_and_report(
    model: nn.Module,
    test_loader: DataLoader,
    class_names: List[str],
    device: str,
    output_dir: Path,
    dataset_name: str,
    gender_class_names: Optional[List[str]] = None,
) -> dict:
    """Ewaluuje model na zbiorze testowym i zapisuje raport klasyfikacji
    oraz macierz pomyłek do `output_dir`.

    Dla każdego modelu zapisywany jest też `gender_report.json`. Zawiera
    on zawsze sekcję `emotion_by_gender`, czyli skuteczność rozpoznawania
    emocji osobno dla nagrań kobiet i mężczyzn (pozwala to porównać
    modele z płcią i bez niej). Dla modeli z głowicą płci (`gender_head`,
    np. `model.GenderAwareSERModel`) raport zawiera dodatkowo ocenę samej
    głowicy płci (accuracy, macierz pomyłek).

    Args:
        model: Wytrenowany model (najlepszy checkpoint).
        test_loader: DataLoader zbioru testowego.
        class_names: Lista nazw klas w kolejności zgodnej z
            `LabelEncoder.classes_` danego datasetu.
        device: Urządzenie obliczeniowe.
        output_dir: Folder wynikowy datasetu, np. `results/TESS`.
        dataset_name: Nazwa datasetu (do tytułów wykresów i logów).
        gender_class_names: Nazwy klas płci zgodne z
            `gender_label_encoder.classes_` (używane tylko dla modeli
            z głowicą płci).

    Returns:
        Słownik z raportem klasyfikacji (wynik `classification_report`
        z `output_dict=True`).
    """
    model.eval()
    all_predictions: List[int] = []
    all_labels: List[int] = []

    has_gender_head = hasattr(model, "gender_head") and hasattr(
        model, "forward_multitask"
    )
    gender_predictions: List[int] = []
    gender_labels_all: List[int] = []

    with torch.no_grad():
        for features, labels, _coarse_labels, gender_labels in test_loader:
            # Etykieta zgrubna (`_coarse_labels`) jest ignorowana - finalna
            # ewaluacja jest zawsze raportowana względem rzeczywistej
            # taksonomii klas datasetu (`model(features)` zwraca logity
            # zadania precyzyjnego dla wszystkich modeli - patrz kontrakt
            # `forward()` w `model.py`).
            features = features.to(device)
            if has_gender_head:
                gender_logits, outputs = model.forward_multitask(features)
                gender_predictions.extend(
                    gender_logits.argmax(dim=1).cpu().numpy().tolist()
                )
            else:
                outputs = model(features)
            gender_labels_all.extend(gender_labels.numpy().tolist())
            predictions = outputs.argmax(dim=1).cpu().numpy()
            all_predictions.extend(predictions.tolist())
            all_labels.extend(labels.numpy().tolist())

    report = classification_report(
        all_labels,
        all_predictions,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    conf_matrix = confusion_matrix(all_labels, all_predictions)
    disp = ConfusionMatrixDisplay(conf_matrix, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=True)
    ax.set_title(f"Confusion Matrix - {dataset_name}")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    print(f"[{dataset_name}] Test accuracy: {report['accuracy']:.4f}")

    _save_gender_report(
        gender_labels_all,
        gender_predictions if has_gender_head else None,
        gender_class_names,
        output_dir,
        dataset_name,
        emotion_labels=all_labels,
        emotion_predictions=all_predictions,
    )
    return report


def _emotion_by_gender(
    emotion_labels: List[int],
    emotion_predictions: List[int],
    gender_labels: List[int],
    gender_names: List[str],
) -> dict:
    """Skuteczność rozpoznawania emocji osobno dla każdej płci.

    Zwraca słownik `{nazwa_płci: {num_samples, accuracy, f1_macro}}`
    (tylko płcie obecne w zbiorze testowym) oraz `accuracy_gap`, czyli
    różnicę accuracy `gender_names[0] - gender_names[1]` (np. Female -
    Male), gdy obie płcie są obecne.
    """
    result: dict = {}
    for index, name in enumerate(gender_names[:2]):
        idx = [i for i, g in enumerate(gender_labels) if g == index]
        if not idx:
            continue
        true = [emotion_labels[i] for i in idx]
        pred = [emotion_predictions[i] for i in idx]
        sub_report = classification_report(
            true, pred, output_dict=True, zero_division=0
        )
        result[name] = {
            "num_samples": len(idx),
            "accuracy": sum(int(p == t) for p, t in zip(pred, true)) / len(idx),
            "f1_macro": sub_report["macro avg"]["f1-score"],
        }
    present = [n for n in gender_names[:2] if n in result]
    if len(present) == 2:
        result["accuracy_gap"] = {
            "definition": f"{present[0]} - {present[1]}",
            "value": result[present[0]]["accuracy"] - result[present[1]]["accuracy"],
        }
    return result


def _save_gender_report(
    labels: List[int],
    predictions: Optional[List[int]],
    class_names: Optional[List[str]],
    output_dir: Path,
    dataset_name: str,
    emotion_labels: Optional[List[int]] = None,
    emotion_predictions: Optional[List[int]] = None,
) -> None:
    """Zapisuje `gender_report.json`.

    Sekcja `emotion_by_gender` (skuteczność emocji osobno dla kobiet i
    mężczyzn) jest zapisywana dla każdego modelu, jeśli podano etykiety
    i predykcje emocji. Ocena głowicy płci (`accuracy`, macierz pomyłek,
    raport per klasa) jest dopisywana tylko, gdy `predictions` nie jest
    None, czyli dla modeli z `gender_head`.

    Głowica płci ma zawsze 2 wyjścia, ale w zbiorach jednopłciowych
    (TESS - same kobiety, SAVEE - sami mężczyźni) `LabelEncoder` zna
    tylko jedną klasę (indeks 0). Brakująca nazwa jest wtedy oznaczana
    jako nieobecna w zbiorze, żeby raport obejmował oba wyjścia.
    """
    names = list(class_names) if class_names else ["0", "1"]
    while len(names) < 2:
        names.append(f"{len(names)} (brak w zbiorze)")

    payload: dict = {"num_samples": len(labels), "classes": names[:2]}

    if emotion_labels is not None and emotion_predictions is not None:
        by_gender = _emotion_by_gender(
            emotion_labels, emotion_predictions, labels, names
        )
        payload["emotion_by_gender"] = by_gender
        summary = ", ".join(
            f"{n} {by_gender[n]['accuracy']:.4f}" for n in names[:2] if n in by_gender
        )
        print(f"[{dataset_name}] Test accuracy emocji wg płci: {summary}")

    if predictions is None:
        with open(output_dir / "gender_report.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return

    gender_report = classification_report(
        labels,
        predictions,
        labels=[0, 1],
        target_names=names[:2],
        output_dict=True,
        zero_division=0,
    )
    gender_accuracy = sum(int(p == t) for p, t in zip(predictions, labels)) / max(
        len(labels), 1
    )
    payload = {
        "accuracy": gender_accuracy,
        **payload,
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=[0, 1]
        ).tolist(),
        "classification_report": gender_report,
    }
    with open(output_dir / "gender_report.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"[{dataset_name}] Test accuracy (gender): {gender_accuracy:.4f}")
