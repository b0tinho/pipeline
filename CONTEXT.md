# Handoff — Pipeline SER (Speech Emotion Recognition)

## Cel projektu

Modularny pipeline PyTorch do rozpoznawania emocji z mowy na 4 zbiorach danych (**TESS, RAVDESS, SAVEE, CREMA-D**), z żelazną zasadą: **datasety nie są łączone** w treningu per-dataset. Dodatkowo tryb `combined` (mega-zbiór) do oceny generalizacji.

## Stan repozytorium

Ścieżka: `C:\Users\Admin\Documents\Github\pipeline`

```
ser_pipeline/
├── data_parsing.py        # parsery nazw plików → emocja + płeć (per dataset)
├── audio_processing.py    # load, pad/truncate, MFCC/Mel-spektrogram, delta
├── augmentation.py        # white noise, pitch shift, time stretch (tylko train)
├── dataset.py             # stratified split, SpeechEmotionDataset (4-tuple), build_dataloaders + build_combined_dataloaders
├── model.py               # 5 architektur + MODEL_REGISTRY + build_model()
├── engine.py              # pętla treningowa, early stopping, multi-task loss, raporty
├── main.py                # orchestrator: grid dataset×model, tryb combined, --preprocessed, --deltas
├── preprocessing.py       # offline VAD + preemfaza → data_preprocessed/
└── spectrogram_analysis.py# wizualizacja spektrogramów i łańcuchów filtrów
run.py                     # CLI (argparse): --models --datasets --combined --preprocessed --deltas --epochs --batch-size --seed --deterministic
preprocess_data.py         # CLI do preprocessingu offline
aggregate_results.py       # średnia ± odch. std. z wielu seedów (tabela + CSV)
setup.bat / setup.sh / install.bat / install.sh   # instalacja (venv + zależności + CUDA detekcja)
report.tex                 # pełny raport LaTeX (aktualny)
requirements.txt, pyproject.toml, .gitignore
```

Dane: `data/{TESS,RAVDESS,SAVEE,CREMA-D}` (TESS: 2800 plików — zduplikowany podfolder `TESS Toronto emotional speech set data` usunięty 2026-11-15). Pipeline zapisuje do `results/per_dataset<WARIANT>/<DATASET>/<MODEL>/` i `results/combined<WARIANT>/<MODEL>/`, gdzie `<WARIANT>` = `` (surowe), `_filtry` (`--preprocessed`), `_deltas` (`--deltas`), `_filtry_deltas` (obie) — funkcja `results_group_dir` w `main.py` (od 2026-11-20; wcześniej wszystkie warianty szły do jednego katalogu i nadpisywały się). Układ jest taki sam jak w `wyniki/`, do którego kopiowane są wyniki z maszyny zdalnej.

## Architektury (5, w `model.py`, klucze `MODEL_REGISTRY`)

| Klucz | Klasa | Parametry (6 kl., 128 mel, 1 kanał) |
|---|---|---|
| `dscnn` | DSCNNEmotionClassifier | 2.20 M |
| `headfusion_acnn` | HeadFusionACNN | 0.17 M |
| `lstm` | KumbharLSTMClassifier | 0.14 M (0.27 M @3 kanały) |
| `hierarchical_crnn` | HierarchicalMultiTaskCRNN | 13.00 M |
| `gender_aware` | GenderAwareSERModel | 1.15 M |

Wspólny kontrakt: wejście `(B, C, F, T)` (C=1 statyka lub C=3 z delta), wyjście logity `(B, num_classes)`. Modele multi-task (`hierarchical_crnn`, `gender_aware`) mają `forward_multitask()`.

## Kluczowe zaimplementowane funkcje

- **Grid per-dataset**: 4 datasety × 5 modeli = 20 eksperymentów, wyniki w `results/per_dataset/<DATASET>/<MODEL>/`.
- **Tryb combined**: 6 wspólnych klas (bez Surprise, bo CREMA-D jej nie ma), split per-dataset **przed** konkatenacją (ochrona przed speaker leakage).
- **Delta + delta-delta**: flaga `--deltas`, 3 kanały wejściowe, standaryzacja per-kanał.
- **Preprocessing offline**: VAD + preemfaza (`preprocess_data.py` → `data_preprocessed/`, flaga `--preprocessed`).
- **Multi-task loss**: `hierarchical_crnn` → target walencji (Negative/Neutral/Positive); `gender_aware` → target płci (wywodzony z nazwy pliku). Wykrywanie typu aux target w `engine.py` po liczbie neuronów `gender_head`.
- **Płeć z nazw plików**: TESS (OAF/YAF=Female), RAVDESS (nieparzysty aktor=M, parzysty=F), SAVEE (wszyscy 4 mówcy=M), CREMA-D (zbiór ID aktorek `CREMAD_FEMALE_ACTOR_IDS` z oficjalnego `VideoDemographics.csv`; 48 M / 43 F).
- **Ocena głowicy płci**: dla modeli z `gender_head` `evaluate_and_report` zapisuje dodatkowo `gender_report.json` (accuracy, macierz pomyłek, raport per klasa płci na teście). W zbiorach jednopłciowych (TESS, SAVEE) druga klasa oznaczona jako „brak w zbiorze”.
- **CLI `--epochs` / `--batch-size`**: przekazywane do `main(num_epochs=..., batch_size=...)` (domyślnie 50 / 32).
- **GPU**: `num_workers=12`, `pin_memory=True`, `persistent_workers=True`, `prefetch_factor=4`, `cudnn.benchmark=True` (w `set_global_seed`; wyłączane przez `--deterministic`).
- **Seed i powtarzalność** (2026-11-20): dwa ziarna w `main.py` — `SPLIT_SEED=42` (podział train/val/test, **stały**, więc test jest ten sam dla każdego seeda i wariantu) oraz ziarno treningu `--seed` (domyślnie `DEFAULT_TRAINING_SEED=42`: wagi, kolejność batchy, augmentacja). `set_global_seed` wywoływane **przed każdym eksperymentem**, a train loader tworzony od nowa z własnym `torch.Generator` (`_seeded_train_loader`) — wynik modelu nie zależy już od tego, jakie modele trenowano wcześniej w tym samym uruchomieniu. `--deterministic`: `cudnn.deterministic=True`, `benchmark=False`, `use_deterministic_algorithms(warn_only=True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Seed ≠ 42 → wyniki w podfolderze `<MODEL>/seed_<N>/`. Każdy przebieg zapisuje `run_config.json` (results_group, seed, split_seed, deterministic, epoki, batch, lr, patience). Sprawdzone: dwa przebiegi z tym samym seedem i `--deterministic` dają identyczny `training_log.csv`.
- **Agregacja seedów**: `aggregate_results.py [katalog]` grupuje przebiegi po eksperymencie (folder `seed_<N>` zdejmowany z klucza), liczy średnią ± odch. std. (ddof=1) dla accuracy i macro F1 oraz średnią epokę najlepszą / zatrzymania i accuracy głowicy płci. Domyślnie bierze tylko przebiegi z `run_config.json` (po 2026-11-20); `--include-legacy` dołącza stare (oznaczone `42*`), `--seeds 1,2,3` filtruje, `--csv plik` zapisuje. Ostrzega, gdy seedy w grupie mają różne ustawienia (epoki, batch, deterministic, split_seed).

## Wyniki (accuracy testowe)

**Per-dataset** (7 klas oprócz CREMA-D=6; wyniki z 2026-07-08, `gender_aware` z 2026-11-19, `lstm` z 2026-11-20 przy limicie 100 epok):
- TESS: ~0.995 (dscnn/hier/headfusion) — **nierzetelne** (tylko 2 aktorki, wyciek mówcy)
- RAVDESS: dscnn **0.607**, headfusion 0.551, lstm 0.528
- SAVEE: headfusion **0.653**, dscnn 0.514
- CREMA-D: headfusion **0.609**, dscnn 0.594
- `lstm` (100 epok, acc / macro F1): TESS 0.705 / 0.613, RAVDESS 0.528 / 0.474, SAVEE 0.361 / 0.208, CREMA-D 0.410 / 0.359 (przed: 0.669 / 0.610, 0.505 / 0.427, 0.333 / 0.159, 0.450 / 0.403). SAVEE i CREMA-D zatrzymane w ep. 29 i 24, więc limit epok nie miał tam znaczenia — zmiana to szum między przebiegami. LSTM pomija całe klasy: TESS recall Sad 0.00 / Surprise 0.02; SAVEE Angry, Disgust, Sad = 0.
- `gender_aware` (po poprawce płci): TESS 0.983, RAVDESS 0.565, SAVEE 0.458, CREMA-D 0.441 (przed: 0.921 / 0.468 / 0.444 / 0.482). CREMA-D niedouczony: train_acc 0.44, early stopping w ep. 25 (wahania val_loss), Fear recall 0.01.

**Mega-zbiór** (wszystkie warianty na tym samym teście 1728 próbek):

| Model | surowe | filtry | deltas |
|---|---|---|---|
| hierarchical_crnn* | **0.707** | 0.596 | **0.679** |
| dscnn | 0.664 | **0.679** | 0.666 |
| headfusion_acnn | 0.655 | 0.623 | 0.647 |
| gender_aware* | 0.671 | 0.668 | 0.625 |
| lstm* | 0.573 | 0.584 | 0.579 |

\* Powtórzone 2026-11-20 z limitem 100 epok (`hier` na filtrach nie powtarzany — zatrzymał się w ep. 43). Wyniki przy 50 epokach: hier 0.704 / — / 0.710; gender_aware 0.663 / 0.660 / 0.635 (przed poprawką płci: 0.653 / 0.629 / 0.584); lstm 0.573 / 0.573 / 0.549. Pozostałe modele zatrzymywały się przed 50. epoką, więc nie wymagały powtórki.

Epoka zatrzymania (najlepsza val_loss) przy limicie 100, surowe / filtry / deltas: hier 57 (50) / — / 44 (37); gender_aware 76 (69) / 54 (47) / 32 (25); lstm 74 (67) / 67 (60) / 76 (69). Głowica płci `gender_aware` na teście: 0.953 / 0.973 / 0.945.

**Wnioski**: (1) surowe dane najlepsze; (2) filtry VAD+preemfaza **szkodzą** większości modeli (hier −11.1 p.p.), ale `gender_aware` praktycznie nie (−0.3 p.p.), a `dscnn` i `lstm` lekko zyskują (+1.5 / +1.1 p.p.); (3) delta/delta-delta: brak spójnego efektu (hier −2.8, gender_aware −4.6, lstm +0.6 p.p.) — mieści się w szumie; (4) ranking na surowych: hier 0.707 > gender_aware 0.671 > dscnn 0.664 > headfusion 0.655 > lstm 0.573 (różnica gender_aware vs dscnn 0.7 p.p. — nierozstrzygająca); (5) **100 epok pomogło tylko tam, gdzie model dochodził do limitu**: `gender_aware` surowe +0.8 p.p. (teraz zbiega, najlepsza ep. 69), hier surowe +0.3 p.p. (najlepsza ep. 50 — limit praktycznie mu nie szkodził), lstm surowe bez zmian.
**Uwaga o wariancji**: brak harmonogramu lr, więc limit epok zmienia tylko przebiegi, które doszły do 50. epoki. Mimo to przebiegi zatrzymane wcześniej też się zmieniły: hier deltas 0.710 → 0.679 (ES w ep. 44), gender_aware deltas — wcześniej do ok. ep. 49, teraz ES w ep. 32; lstm CREMA-D −4.0 p.p., SAVEE +2.8 p.p.; wcześniej TESS dla `gender_aware` +6.2 p.p. bez zmiany etykiet. **Różnice poniżej ok. 4 p.p. między pojedynczymi przebiegami to szum.** Przyczyny: `cudnn.benchmark=True` (niedeterministyczne algorytmy GPU) oraz — w trybie combined — wspólny train loader z `persistent_workers=True` dla wszystkich modeli, przez co augmentacja i kolejność batchy zależały od modeli trenowanych wcześniej w tym samym uruchomieniu (inny zestaw `--models` = inny wynik). Oba źródła usunięte 2026-11-20 (patrz „Seed i powtarzalność”); wyniki powyżej pochodzą sprzed tej zmiany.

## Napotkane problemy i rozwiązania

1. **Wariancja podziału**: stary `combined` (2088 test) vs nowy (1728) dawał ~6 p.p. różnicy. Weryfikacja starych checkpointów na nowym teście wykazała, że to **efekt podziału** (ten sam model traci 5–7.6 p.p. przy zmianie testu), nie regresja modelu.
2. **Błędne etykiety płci** (2026-11-15) — RAVDESS (1–12=F zamiast parzysty=F, ~50% błędnych), SAVEE (JE/JK jako F, a wszyscy to M), CREMA-D (reguła parzystości błędna dla 42/91 aktorów, 46% nagrań). Poprawione w `data_parsing.py`; `gender_aware` powtórzony 2026-11-19 we wszystkich wariantach (pozostałe modele nie używają płci, podział stratyfikowany tylko po emocji).
3. **Nietypowy plik TESS** `OA_bite_neutral.wav` (2-literowy prefix) — fallback do nazwy folderu w `parse_tess_gender`.
4. **LSTM bez obsługi 3 kanałów** — dodano `input_channels` i konkatenację kanałów w `forward`.
5. **Colorbar najeżdżał na spektrogram** — dedykowana oś `fig.add_axes([0.87, ...])`.
6. **`setup.bat` błąd `ensurepip`** (Python ze Store) — fallback `--without-pip` + `get-pip.py`.
7. **Niedziałające `--epochs`/`--batch-size` w `run.py`** — ustawiały atrybuty klasy `TrainingConfig`, a `main()` miał na sztywno 50/32. Naprawione (2026-11-19): wartości idą parametrami `main()`.
8. **Ostrzeżenie numba `invalid value encountered in cast`** (librosa 1.0.1) — pochodzi z `phase_vocoder` → `util.phasor` przy pitch shift / time stretch. Niegroźne: wynik augmentacji skończony, korelacja z librosa 0.11 = 0.9999996, Mel-spektrogram identyczny (różnica ≤ 1.5e-5 dB).
9. **Nieprzypięte wersje** — `librosa>=0.10.0` dało na maszynie zdalnej librosa 1.0.1. Przypięto `librosa==1.0.1`, `numba==0.68.1`, `numpy==2.5.5` w `requirements.txt` i `pyproject.toml` (pozostałe pakiety nadal `>=`).
10. **Niepowtarzalne wyniki przy stałym seedzie** (2026-11-20) — patrz uwaga o wariancji w wynikach. Rozwiązanie: ziarno ustawiane przed każdym eksperymentem, nowy train loader z `torch.Generator` na model, opcjonalny `--deterministic`, rozdzielenie `SPLIT_SEED` (stały) od `--seed` (trening).

## Konwencje i decyzje projektowe

- Zawsze 4-tuple z `SpeechEmotionDataset`: `(features, fine_label, coarse_label, gender_label)`.
- Etykieta coarse = walencja (mapa `COARSE_EMOTION_MAP` w `data_parsing.py`).
- `build_model(model_name, num_classes, input_freq_dim, input_channels, **kwargs)`.
- Augmentacja tylko dla train; deterministyczna kolejność: VAD → preemfaza → augmentacja → pad → mel.
- Wyniki: `results/per_dataset<WARIANT>/<DATASET>/<MODEL>/` lub `results/combined<WARIANT>/<MODEL>/` z plikami `{best_model.pt, training_log.csv, classification_report.json, confusion_matrix.png, label_mapping.json, run_config.json}` + `gender_report.json` dla `gender_aware`; dla seeda ≠ 42 w podfolderze `seed_<N>/`.
- Podział danych zawsze z `SPLIT_SEED=42` — nie zmieniać, inaczej wyniki nie będą porównywalne z dotychczasowymi (test 1728 próbek na combined).

## Kolejne kroki (rekomendowane)

1. **Leave-one-speaker-out split** (zamiast losowego stratyfikowanego) — rzetelna ocena generalizacji; obecnie TESS ~99% to artefakt wycieku mówcy.
2. **Class weighting** dla RAVDESS (Neutral 288 vs reszta 192).
3. **Strojenie `multitask_alpha`** dla `hierarchical_crnn` (grid search).
4. **Eksperyment ablacyjny** VAD vs preemfaza (osobno), bo oba razem szkodzą.
5. **Uśrednianie po wielu seedach** — plan (2026-11-20): seedy 1, 2, 3, `--epochs 100 --deterministic`, nowe przebiegi (stary seed 42 nie mieszany z nowymi). Kolejność: (1) combined surowe, 5 modeli; (2) combined deltas; (3) combined filtry (min. hier, dscnn, gender_aware); (4) per dataset (najmniej istotne: TESS wyciek mówcy, SAVEE 72 próbki testowe = 1.4 p.p. na próbkę). Stare wyniki nie są błędne — to pojedyncze pomiary z dużym rozrzutem. Raportować średnią ± odch. std. z `aggregate_results.py`.
6. **Multi-GPU (DDP)** — obecnie jedna karta.
7. ~~Dłuższy trening na combined~~ — zrobione 2026-11-20 (`--epochs 100`), patrz wyniki. Dla kolejnych przebiegów na combined używać `--epochs 100` (`gender_aware` zbiega dopiero ok. ep. 69).
8. **Przypiąć pozostałe pakiety** (soundfile, scikit-learn, pandas, matplotlib) wg `pip freeze` z maszyny zdalnej.

## Jak uruchomić

```bash
python run.py                          # grid: 4 datasety × 5 modeli
python run.py --combined               # mega-zbiór, 6 klas
python run.py --combined --deltas      # + 3 kanały
python run.py --preprocessed           # dane po VAD+preemfazie
python run.py --models dscnn --datasets SAVEE
python run.py --combined --models gender_aware --epochs 100
python run.py --combined --models gender_aware --epochs 100 --seed 7 --deterministic   # → results/combined/gender_aware/seed_7/
python preprocess_data.py              # preprocessing offline (raz)

# wiele seedów (każdy wariant we własnym katalogu, nic się nie nadpisuje)
for s in 1 2 3; do
  python run.py --combined                --epochs 100 --seed $s --deterministic
  python run.py --combined --deltas       --epochs 100 --seed $s --deterministic
  python run.py --combined --preprocessed --epochs 100 --seed $s --deterministic
done
python aggregate_results.py                       # tabela średnia ± std z results/
python aggregate_results.py wyniki --csv wyniki/summary.csv
```

Zależności: `requirements.txt`; PyTorch dobierany przez `setup.sh`/`setup.bat` (CUDA 11.8 lub CPU).
