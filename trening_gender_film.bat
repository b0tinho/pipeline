@echo off
rem Zwyciezca ablacji (gender_aware_film) i kontrola bez plci
rem (gender_aware_att_nogender) na filtrach i deltach, seedy 1-3.
rem Procedura jak w wynikach glownych: patience 15, limit 150 epok,
rem tryb deterministyczny.
rem   filtry -> results\combined_filtry_patience15\<MODEL>\seed_<N>
rem   deltas -> results\combined_deltas_patience15\<MODEL>\seed_<N>
rem
rem Na koncu powtorka na surowych (gender_aware, _film, _att_nogender),
rem tylko po to, zeby dopisac do gender_report.json skutecznosc emocji
rem osobno dla kobiet i mezczyzn (emotion_by_gender). W trybie
rem deterministycznym wyniki powinny byc identyczne z tymi w
rem wyniki_seedy\combined_patience15 - przy okazji sprawdza to powtarzalnosc.
rem   surowe -> results\combined_patience15\<MODEL>\seed_<N>

set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set LOG=trening_gender_film.log
set OPTS=--epochs 150 --deterministic --patience 15
set MODELS=gender_aware_film,gender_aware_att_nogender

for %%s in (1 2 3) do (
    python run.py --combined --preprocessed --models %MODELS% --seed %%s %OPTS% >> %LOG% 2>&1
    python run.py --combined --deltas       --models %MODELS% --seed %%s %OPTS% >> %LOG% 2>&1
)

for %%s in (1 2 3) do (
    python run.py --combined --models gender_aware,%MODELS% --seed %%s %OPTS% >> %LOG% 2>&1
)

python aggregate_results.py results >> %LOG% 2>&1
echo Gotowe >> %LOG%
