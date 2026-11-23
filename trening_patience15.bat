@echo off
rem Pelna spojnosc: wszystkie modele i warianty mega-zbioru z patience 15,
rem limit 150 epok, seedy 1-3, tryb deterministyczny.
rem Wyniki: results\combined_patience15, results\combined_deltas_patience15,
rem         results\combined_filtry_patience15 (podfoldery <MODEL>\seed_<N>).
rem
rem Pomijane (juz policzone z patience 15 i zatrzymane przed limitem 100,
rem wiec przy limicie 150 bylyby identyczne):
rem   deltas: hierarchical_crnn, gender_aware (seedy 1-3)
rem   filtry: gender_aware (seedy 1-3), hierarchical_crnn seed 1
rem Powtarzane: filtry / hierarchical_crnn seed 2 i 3 (trafily w limit 100).

set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set LOG=trening_patience15.log
set OPTS=--epochs 150 --deterministic --patience 15

for %%s in (1 2 3) do (
    python run.py --combined                --models dscnn,headfusion_acnn,lstm,hierarchical_crnn,gender_aware --seed %%s %OPTS% >> %LOG% 2>&1
    python run.py --combined --deltas       --models dscnn,headfusion_acnn,lstm --seed %%s %OPTS% >> %LOG% 2>&1
    python run.py --combined --preprocessed --models dscnn,headfusion_acnn,lstm --seed %%s %OPTS% >> %LOG% 2>&1
)

for %%s in (2 3) do (
    python run.py --combined --preprocessed --models hierarchical_crnn --seed %%s %OPTS% >> %LOG% 2>&1
)

python aggregate_results.py results >> %LOG% 2>&1
echo Gotowe >> %LOG%
