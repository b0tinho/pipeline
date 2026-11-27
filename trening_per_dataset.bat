@echo off
rem Pojedyncze korpusy z obiema plciami: RAVDESS i CREMA-D.
rem 7 modeli (5 bazowych + gender_aware_film + kontrola bez plci), seedy 1-3,
rem dane surowe. Procedura jak na zbiorze polaczonym: patience 15,
rem limit 150 epok, tryb deterministyczny.
rem   -> results\per_dataset_patience15\<KORPUS>\<MODEL>\seed_<N>
rem
rem TESS i SAVEE pominiete celowo: TESS ma 2 mowczynie (wyciek mowcy,
rem wyniki ok. 99%), SAVEE ma 72 nagrania testowe (1 nagranie = 1.4 p.p.)
rem i obu brakuje drugiej plci.
rem
rem Bledy pojedynczych eksperymentow nie przerywaja przebiegu - po
rem zakonczeniu sprawdzic log:  findstr /L /C:"[BLAD]" trening_per_dataset.log

set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set LOG=trening_per_dataset.log
set OPTS=--epochs 150 --deterministic --patience 15
set DATASETS=RAVDESS,CREMA-D
set MODELS=dscnn,headfusion_acnn,lstm,hierarchical_crnn,gender_aware,gender_aware_film,gender_aware_att_nogender

for %%s in (1 2 3) do (
    python run.py --datasets %DATASETS% --models %MODELS% --seed %%s %OPTS% >> %LOG% 2>&1
)

python aggregate_results.py results >> %LOG% 2>&1
echo Gotowe >> %LOG%
