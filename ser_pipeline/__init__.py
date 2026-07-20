"""
ser_pipeline
============

Modularny pipeline do przetwarzania, treningu i ewaluacji modeli
rozpoznawania emocji z mowy (Speech Emotion Recognition, SER) na
czterech niezależnie przetwarzanych zbiorach danych: TESS, RAVDESS,
SAVEE oraz CREMA-D.

Kluczowe założenie: zbiory danych NIGDY nie są łączone. Każdy moduł w
tym pakiecie operuje na jednym, wskazanym datasecie na raz, a wyniki
(modele, logi, metryki, macierze pomyłek) są zapisywane do osobnych
podfolderów `results/<NAZWA_DATASETU>/`.
"""
