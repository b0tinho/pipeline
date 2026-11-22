#!/usr/bin/env python
"""
Zbiera wyniki z wielu seedow do tabeli "srednia +- odchylenie standardowe".

Przeszukuje katalog wynikow (domyslnie `results/`) w poszukiwaniu
przebiegow (folderow z `classification_report.json`) i grupuje je po
eksperymencie (np. `combined_deltas/gender_aware`,
`per_dataset/TESS/lstm`). Przebiegi z roznymi seedami tego samego
eksperymentu leza w podfolderach `seed_<N>/` (patrz `ser_pipeline/main.py`).

Domyslnie liczone sa tylko przebiegi z `run_config.json`, czyli wykonane
po wprowadzeniu `--seed` (2026-11-20). Starsze wyniki (bez tego pliku)
mozna dolaczyc flaga `--include-legacy`.

Uzycie:
    python aggregate_results.py                     # results/
    python aggregate_results.py wyniki              # inny katalog
    python aggregate_results.py --seeds 1,2,3       # tylko wybrane seedy
    python aggregate_results.py --include-legacy    # z wynikami sprzed --seed
    python aggregate_results.py --csv summary.csv   # zapis do CSV

Kolumny:
    n       liczba przebiegow (seedow) w grupie
    acc     accuracy testowe, srednia +- odch. std. (probkowe, ddof=1)
    F1      macro F1 testowe, srednia +- odch. std.
    best_ep srednia epoka najlepszej val_loss (zapisany checkpoint)
    stop_ep srednia epoka zakonczenia treningu (early stopping lub limit)
    gender  accuracy glowicy plci (tylko modele z gender_head)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

SEED_DIR_PATTERN = re.compile(r"^seed_(\d+)$")
LEGACY_SEED = 42


@dataclass
class RunResult:
    """Wynik pojedynczego przebiegu (jeden seed)."""

    group: str
    model: str
    seed: int
    legacy: bool
    accuracy: float
    macro_f1: float
    best_epoch: int | None
    stop_epoch: int | None
    gender_accuracy: float | None
    config: dict = field(default_factory=dict)


def _read_epochs(log_path: Path) -> tuple[int | None, int | None]:
    """Zwraca (epoka najlepszej val_loss, liczba epok) z `training_log.csv`."""
    if not log_path.exists():
        return None, None
    with open(log_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None, None
    best = min(rows, key=lambda r: float(r["val_loss"]))
    return int(best["epoch"]), int(rows[-1]["epoch"])


def _load_run(run_dir: Path, root: Path) -> RunResult:
    """Wczytuje metryki jednego przebiegu i ustala jego grupe oraz seed."""
    parts = list(run_dir.relative_to(root).parts)
    seed_match = SEED_DIR_PATTERN.match(parts[-1])
    if seed_match:
        parts = parts[:-1]

    config_path = run_dir / "run_config.json"
    config: dict = {}
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)

    if seed_match:
        seed = int(seed_match.group(1))
    else:
        seed = int(config.get("seed", LEGACY_SEED))

    with open(run_dir / "classification_report.json", encoding="utf-8") as f:
        report = json.load(f)

    gender_accuracy = None
    gender_path = run_dir / "gender_report.json"
    if gender_path.exists():
        with open(gender_path, encoding="utf-8") as f:
            gender_accuracy = json.load(f).get("accuracy")

    best_epoch, stop_epoch = _read_epochs(run_dir / "training_log.csv")

    return RunResult(
        group="/".join(parts[:-1]) or ".",
        model=parts[-1],
        seed=seed,
        legacy=not config,
        accuracy=float(report["accuracy"]),
        macro_f1=float(report["macro avg"]["f1-score"]),
        best_epoch=best_epoch,
        stop_epoch=stop_epoch,
        gender_accuracy=gender_accuracy,
        config=config,
    )


def collect_runs(
    root: Path, seeds: set[int] | None, include_legacy: bool
) -> list[RunResult]:
    """Zbiera wszystkie przebiegi spelniajace filtry."""
    runs = []
    for report_path in sorted(root.rglob("classification_report.json")):
        run = _load_run(report_path.parent, root)
        if run.legacy and not include_legacy:
            continue
        if seeds is not None and run.seed not in seeds:
            continue
        runs.append(run)
    return runs


def _mean_std(values: list[float], digits: int = 3) -> str:
    """Formatuje "srednia +- odch. std."; dla jednej wartosci sama wartosc."""
    values = [v for v in values if v is not None]
    if not values:
        return "-"
    mean = statistics.mean(values)
    if len(values) == 1:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {statistics.stdev(values):.{digits}f}"


def _mean(values: list, digits: int = 1) -> str:
    values = [v for v in values if v is not None]
    return f"{statistics.mean(values):.{digits}f}" if values else "-"


# Ustawienia, ktore powinny byc wspolne dla wszystkich seedow jednej grupy.
_CONSISTENCY_KEYS = (
    "num_epochs", "batch_size", "deterministic", "split_seed",
    "early_stopping_patience",
)


def _epoch_caps_equivalent(items: list[RunResult]) -> bool:
    """Czy rozny limit epok w grupie nie mial wplywu na wyniki.

    Brak harmonogramu lr, wiec przebieg zatrzymany przez early stopping
    PRZED swoim limitem bylby identyczny przy dowolnym wiekszym limicie
    (w trybie deterministycznym). Rozny limit ma znaczenie tylko wtedy,
    gdy przebieg z mniejszym limitem w niego trafil.
    """
    caps = [r.config.get("num_epochs") for r in items]
    if any(c is None for c in caps):
        return False
    max_cap = max(caps)
    return all(
        r.stop_epoch is not None and r.stop_epoch < cap
        for r, cap in zip(items, caps)
        if cap < max_cap
    )


def summarize(runs: list[RunResult]) -> list[dict]:
    """Grupuje przebiegi po (grupa, model) i liczy statystyki."""
    grouped: dict[tuple[str, str], list[RunResult]] = {}
    for run in runs:
        grouped.setdefault((run.group, run.model), []).append(run)

    rows = []
    for (group, model), items in sorted(grouped.items()):
        items.sort(key=lambda r: r.seed)
        mixed = [
            key for key in _CONSISTENCY_KEYS
            if len({json.dumps(r.config.get(key)) for r in items}) > 1
            and not (key == "num_epochs" and _epoch_caps_equivalent(items))
        ]
        rows.append({
            "group": group,
            "model": model,
            "n": len(items),
            "seeds": ",".join(
                f"{r.seed}{'*' if r.legacy else ''}" for r in items
            ),
            "acc": _mean_std([r.accuracy for r in items]),
            "F1": _mean_std([r.macro_f1 for r in items]),
            "best_ep": _mean([r.best_epoch for r in items]),
            "stop_ep": _mean([r.stop_epoch for r in items]),
            "gender": _mean_std([r.gender_accuracy for r in items]),
            "acc_mean": statistics.mean(r.accuracy for r in items),
            "warning": (
                f"rozne ustawienia: {', '.join(mixed)}" if mixed else ""
            ),
        })
    return rows


_COLUMNS = ("model", "n", "seeds", "acc", "F1", "best_ep", "stop_ep", "gender")


def print_table(rows: list[dict]) -> None:
    """Wypisuje tabele, osobno dla kazdej grupy; modele wg sredniej acc."""
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["group"], []).append(row)

    for group, items in groups.items():
        items = sorted(items, key=lambda r: -r["acc_mean"])
        widths = {
            col: max(len(col), *(len(str(r[col])) for r in items))
            for col in _COLUMNS
        }
        print(f"\n== {group} ==")
        print("  ".join(col.ljust(widths[col]) for col in _COLUMNS))
        for row in items:
            print("  ".join(str(row[col]).ljust(widths[col]) for col in _COLUMNS))
            if row["warning"]:
                print(f"   ^ UWAGA: {row['warning']}")


def write_csv(rows: list[dict], path: Path) -> None:
    columns = ("group", *_COLUMNS, "warning")
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _parse_seeds(value: str) -> set[int]:
    return {int(item) for item in value.split(",") if item.strip()}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Srednia +- odch. std. wynikow z wielu seedow",
    )
    parser.add_argument(
        "root", nargs="?", default="results",
        help="Katalog wynikow (domyslnie: results)",
    )
    parser.add_argument(
        "--seeds", type=_parse_seeds, default=None,
        help="Tylko wybrane seedy, np. 1,2,3 (domyslnie: wszystkie)",
    )
    parser.add_argument(
        "--include-legacy", action="store_true",
        help="Dolacz przebiegi bez run_config.json (sprzed --seed), "
             "oznaczone gwiazdka jako seed 42*",
    )
    parser.add_argument(
        "--csv", type=Path, default=None,
        help="Zapisz tabele do pliku CSV",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        parser.error(f"Katalog '{root}' nie istnieje")

    runs = collect_runs(root, args.seeds, args.include_legacy)
    if not runs:
        hint = "" if args.include_legacy else " (sprobuj --include-legacy)"
        print(f"Brak przebiegow w '{root}' spelniajacych filtry{hint}.")
        return

    rows = summarize(runs)
    print_table(rows)
    single = sum(1 for r in rows if r["n"] == 1)
    if single:
        print(f"\n{single} grup(y) z jednym przebiegiem - brak odchylenia std.")
    if args.csv:
        write_csv(rows, args.csv)
        print(f"Zapisano: {args.csv}")


if __name__ == "__main__":
    main()
