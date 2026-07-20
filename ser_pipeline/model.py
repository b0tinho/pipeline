"""
Architektury sieci neuronowych do rozpoznawania emocji z mowy (SER).

Moduł zawiera zestaw modeli scalonych do wspólnego interfejsu
pipeline'u:

    - Wejście: tensor `(batch, 1, n_features, n_frames)` - dokładnie
      taki, jaki produkuje `dataset.SpeechEmotionDataset` (MFCC lub
      Mel-spektrogram, jeden kanał).
    - Wyjście: tensor logitów `(batch, num_classes)`, gdzie
      `num_classes` jest parametrem konstruktora dopasowywanym
      dynamicznie do liczby unikalnych klas emocji w aktualnie
      przetwarzanym datasecie (patrz `main.run_pipeline_for_dataset`).

Dostępne architektury:
    - `DSCNNEmotionClassifier`     - Deep Stride CNN (Wani et al., 2020).
    - `HeadFusionACNN`             - ACNN z mechanizmem Head Fusion Attention
                                      (Xu et al., 2020).
    - `KumbharLSTMClassifier`      - lekki klasyfikator LSTM + Dense
                                      (Kumbhar et al., 2019).
    - `HierarchicalMultiTaskCRNN`  - hierarchiczny, wielozadaniowy CRNN
                                      z uwagą i fuzją cech (na podst. 3D-HMTL).

Moduł NIE definiuje żadnego "domyślnego" modelu projektu - orchestrator
(`main.py`) uruchamia KAżDY z powyższych czterech modeli na KAżDYM z
czterech datasetów (pełna kombinacja dataset x model), aby porównać ich
względną skuteczność.

Modele te zostały zaimportowane z osobnych plików (`model_3d_hmtl.py`,
`model_dscnn.py`, `model_headfusion.py`, `model_lstm.py`) i scalone
tutaj w jednym miejscu. W procesie scalania zaadaptowano je do
pipeline'u w następujący sposób:

    1. Zahardkodowaną liczbę klas wyjściowych (`num_emotions=4` itp.)
       zastąpiono parametrem `num_classes`, przekazywanym dynamicznie
       przez orchestrator (`main.py`) na podstawie liczby unikalnych
       etykiet w bieżącym datasecie.
    2. Zahardkodowane wymiary wejściowe (np. obraz 256x256, mapa cech
       40 pasm Mel x 300 klatek w 3 kanałach) zastąpiono albo
       operacjami niezależnymi od rozmiaru przestrzennego (globalny
       adaptacyjny pooling), albo parametrem `input_freq_dim`
       wyliczanym dynamicznie z konfiguracji audio
       (`AudioProcessingConfig.feature_dim`).
    3. Niejednorodne sygnatury `forward()` (niektóre zwracały krotki
       `(None, logits)` lub `(coarse, fine)`) sprowadzono do wspólnego
       kontraktu: `forward()` zwraca zawsze pojedynczy tensor logitów
       `(batch, num_classes)`, dzięki czemu każdy model jest gotowym
       zamiennikiem w istniejącej pętli treningowej `engine.py` - bez
       potrzeby jakichkolwiek modyfikacji tej pętli.

Wybór modelu do treningu odbywa się przez fabrykę `build_model()` lub
rejestr `MODEL_REGISTRY`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


# ===========================================================================
# 1. DSCNNEmotionClassifier - Deep Stride CNN (na podst. Wani et al., 2020)
# ===========================================================================
class DSCNNEmotionClassifier(nn.Module):
    """Deep Stride CNN (DSCNN) na podstawie architektury Wani et al. (2020).

    Kluczowa idea oryginału: pięć warstw konwolucyjnych ze stride=2
    zastępuje Max Pooling do redukcji wymiarów przestrzennych. Oryginał
    zakładał wejściowy obraz o zafiksowanym rozmiarze 256x256 oraz
    warstwę FC z zahardkodowanym rozmiarem wejścia (256 * 8 * 8).

    Adaptacja do pipeline'u: aby architektura współpracowała z cechami
    MFCC/Mel-spektrogramu o dowolnym, zmiennym rozmiarze przestrzennym
    (`n_features` x `n_frames`), warstwę Flatten + zahardkodowany FC
    zastąpiono globalnym adaptacyjnym average poolingiem
    (`nn.AdaptiveAvgPool2d(1)`) przed klasyfikatorem - dzięki temu model
    jest w pełni niezależny od wymiarów wejściowej mapy cech.
    """

    def __init__(
        self,
        num_classes: int,
        input_freq_dim: int | None = None,
        input_channels: int = 1,
        conv_channels: tuple[int, int, int, int, int] = (32, 64, 128, 128, 256),
        dropout: float = 0.5,
    ) -> None:
        """
        Args:
            num_classes: Liczba klas emocji w bieżącym datasecie.
            input_freq_dim: Nieużywane w tej architekturze - parametr
                zachowany wyłącznie dla zgodności z ujednoliconym
                interfejsem fabryki `build_model` (dzięki globalnemu
                poolingowi model jest niezależny od wymiaru
                częstotliwościowego wejścia).
            input_channels: Liczba kanałów wejściowych.
            conv_channels: Liczba filtrów w kolejnych 5 warstwach
                konwolucyjnych ze stride=2.
            dropout: Współczynnik dropout w klasyfikatorze.
        """
        super().__init__()
        del input_freq_dim  # Nieużywane - patrz docstring klasy.

        c1, c2, c3, c4, c5 = conv_channels
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, c1, kernel_size=11, stride=2, padding=5),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c3, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3, c4, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
            nn.Conv2d(c4, c5, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(c5),
            nn.ReLU(inplace=True),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(c5, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor wejściowy o kształcie (batch, 1, n_features, n_frames).

        Returns:
            Logity o kształcie (batch, num_classes).
        """
        x = self.conv_layers(x)
        x = self.global_pool(x).flatten(1)
        return self.classifier(x)


# ===========================================================================
# 2. HeadFusionACNN - ACNN z Head Fusion Attention (na podst. Xu et al., 2020)
# ===========================================================================
class HeadFusionAttention(nn.Module):
    """Uproszczony mechanizm uwagi wielogłowicowej (Head Fusion) działający
    w domenie przestrzennej mapy cech, na podstawie Xu et al. (2020).
    """

    def __init__(self, in_channels: int, n_head: int = 4) -> None:
        """
        Args:
            in_channels: Liczba kanałów wejściowej mapy cech.
            n_head: Liczba "głów" uwagi (analogicznie do multi-head attention).
        """
        super().__init__()
        self.n_head = n_head
        self.query_conv = nn.Conv2d(in_channels, in_channels * n_head, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels * n_head, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels * n_head, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.size()

        query = self.query_conv(x).view(batch * self.n_head, channels, -1)
        key = self.key_conv(x).view(batch * self.n_head, channels, -1)
        value = self.value_conv(x).view(batch * self.n_head, channels, -1)

        # Mapa uwagi: softmax(Q^T * K) * V.
        energy = torch.bmm(query.transpose(1, 2), key)
        attention = F.softmax(energy, dim=-1)
        out = torch.bmm(value, attention.transpose(1, 2))

        # Złożenie głów z powrotem i fuzja przez uśrednianie (Head Fusion).
        out = out.view(batch, self.n_head, channels, height, width)
        out = torch.mean(out, dim=1)
        return out


class HeadFusionACNN(nn.Module):
    """ACNN z mechanizmem Head Fusion Attention, na podstawie Xu et al. (2020).

    Dwie równoległe ścieżki konwolucyjne (Conv1a, Conv1b) o różnych
    kształtach jądra wychwytują lokalne wzorce w odmiennych proporcjach
    czas/częstotliwość; ich wynik jest łączony i przetwarzany przez
    dalsze bloki konwolucyjne oraz mechanizm uwagi `HeadFusionAttention`.

    Zakończenie globalnym average poolingiem (`AdaptiveAvgPool2d`) - tak
    jak w oryginale - sprawia, że architektura jest niezależna od
    rozmiaru przestrzennego wejścia i współpracuje bez zmian z cechami
    MFCC/Mel-spektrogramu o dowolnym `n_features` x `n_frames`.
    """

    def __init__(
        self,
        num_classes: int,
        input_freq_dim: int | None = None,
        input_channels: int = 1,
        n_head: int = 4,
    ) -> None:
        """
        Args:
            num_classes: Liczba klas emocji w bieżącym datasecie.
            input_freq_dim: Nieużywane - parametr zachowany wyłącznie
                dla zgodności z ujednoliconym interfejsem fabryki
                `build_model` (patrz docstring klasy).
            input_channels: Liczba kanałów wejściowych.
            n_head: Liczba głów uwagi w `HeadFusionAttention`.
        """
        super().__init__()
        del input_freq_dim  # Nieużywane - patrz docstring klasy.

        self.conv1a = nn.Conv2d(
            input_channels, 8, kernel_size=(10, 2), stride=1, padding=(4, 0)
        )
        self.conv1b = nn.Conv2d(
            input_channels, 8, kernel_size=(2, 8), stride=1, padding=(0, 3)
        )
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 80, kernel_size=3, padding=1),
            nn.BatchNorm2d(80),
            nn.ReLU(inplace=True),
        )

        self.attention = HeadFusionAttention(in_channels=80, n_head=n_head)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(80, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor wejściowy o kształcie (batch, 1, n_features, n_frames).

        Returns:
            Logity o kształcie (batch, num_classes).
        """
        branch_a = F.relu(self.conv1a(x))
        branch_b = F.relu(self.conv1b(x))

        # Dopasowanie rozmiarów obu ścieżek (padding może dać niewielkie
        # niedopasowanie wymiarów) przed konkatenacją kanałów.
        min_height = min(branch_a.size(2), branch_b.size(2))
        min_width = min(branch_a.size(3), branch_b.size(3))
        fused = torch.cat(
            (
                branch_a[:, :, :min_height, :min_width],
                branch_b[:, :, :min_height, :min_width],
            ),
            dim=1,
        )
        fused = self.bn1(fused)

        fused = self.conv2(fused)
        fused = self.conv3(fused)
        fused = self.conv4(fused)
        fused = self.conv5(fused)

        fused = self.attention(fused)
        fused = self.global_pool(fused).flatten(1)
        return self.classifier(fused)


# ===========================================================================
# 3. KumbharLSTMClassifier - lekki LSTM + Dense (na podst. Kumbhar et al., 2019)
# ===========================================================================
class KumbharLSTMClassifier(nn.Module):
    """Lekki klasyfikator LSTM + Dense, na podstawie Kumbhar et al. (2019).

    W przeciwieństwie do pozostałych modeli w tym module, ta
    architektura operuje na cechach jako sekwencji wektorów w czasie
    (`(batch, n_frames, n_features)`), a nie jako dwuwymiarowej "mapie
    obrazu" `(batch, 1, n_features, n_frames)`. Aby zachować zgodność z
    resztą pipeline'u, `forward()` automatycznie konwertuje tensor
    wejściowy do wymaganego układu wymiarów.
    """

    def __init__(
        self,
        num_classes: int,
        input_freq_dim: int,
        lstm_hidden_size: int = 128,
        dropout: float = 0.5,
    ) -> None:
        """
        Args:
            num_classes: Liczba klas emocji w bieżącym datasecie.
            input_freq_dim: Liczba cech na krok czasowy (np. `n_mfcc`
                lub `n_mels` z `AudioProcessingConfig`) - rozmiar
                wejścia warstwy LSTM.
            lstm_hidden_size: Rozmiar stanu ukrytego LSTM.
            dropout: Współczynnik dropout stosowany przed klasyfikatorem.
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_freq_dim,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        # W pełni połączone warstwy klasyfikacyjne zgodne z oryginałem.
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_size, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 8),
            nn.Tanh(),
            nn.Linear(8, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor wejściowy o kształcie (batch, 1, n_features, n_frames)
                lub już spłaszczony (batch, n_frames, n_features).

        Returns:
            Logity o kształcie (batch, num_classes).
        """
        if x.dim() == 4:
            # (batch, 1, n_features, n_frames) -> (batch, n_frames, n_features)
            x = x.squeeze(1).permute(0, 2, 1)

        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]

        out = self.dropout(last_hidden)
        return self.classifier(out)


# ===========================================================================
# 4. HierarchicalMultiTaskCRNN - hierarchiczny CRNN (na podst. 3D-HMTL)
# ===========================================================================
class AttentionLayer(nn.Module):
    """Mechanizm uwagi typu location-based, przekształcający cechy na
    poziomie klatki na cechy na poziomie całej wypowiedzi.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.W = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, rnn_outputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rnn_outputs: Tensor (batch, seq_len, hidden_size).

        Returns:
            Wektor kontekstu (batch, hidden_size), będący ważoną sumą
            kroków czasowych `rnn_outputs`.
        """
        scores = self.v(torch.tanh(self.W(rnn_outputs)))  # (batch, seq_len, 1)
        alphas = F.softmax(scores, dim=1)
        context_vector = torch.sum(rnn_outputs * alphas, dim=1)
        return context_vector


class SharedCNNExtractor(nn.Module):
    """Współdzielony ekstraktor cech CNN dla `HierarchicalMultiTaskCRNN`.

    Sześć bloków konwolucyjnych 2D (Conv2d -> BatchNorm -> LeakyReLU),
    z Max Poolingiem tylko w pierwszym bloku. Wymiar wejściowej mapy
    cech (`input_freq_dim`) oraz liczba kanałów (`input_channels`) są
    parametryzowane, dzięki czemu ekstraktor dopasowuje się dynamicznie
    do konfiguracji audio pipeline'u (w oryginale zahardkodowane jako
    3 kanały x 40 pasm Mel).
    """

    def __init__(
        self, input_channels: int, input_freq_dim: int, reduced_dim: int = 256
    ) -> None:
        super().__init__()

        def conv_block(
            in_channels: int,
            out_channels: int,
            pool_kernel: tuple[int, int] | None = None,
            pool_stride: tuple[int, int] | None = None,
        ) -> nn.Module:
            layers = [
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=(5, 3), padding=(2, 1)
                ),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.01),
            ]
            if pool_kernel is not None:
                layers.append(nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride))
            return nn.Sequential(*layers)

        # Uwaga: tensor wejściowy pipeline'u ma układ (batch, channels,
        # n_features/freq, n_frames/time) - w bloku 1 stosujemy pooling
        # (4, 2), tak by silniej zredukować oś częstotliwości (analogicznie
        # do oryginalnej redukcji 40 -> 10 pasm Mel) niż oś czasu.
        self.block1 = conv_block(
            input_channels, 32, pool_kernel=(4, 2), pool_stride=(4, 2)
        )
        self.block2 = conv_block(32, 64)
        self.block3 = conv_block(64, 64)
        self.block4 = conv_block(64, 64)
        self.block5 = conv_block(64, 64)
        self.block6 = conv_block(64, 128)

        freq_after_pool1 = max(input_freq_dim // 4, 1)
        self.flattened_dim = 128 * freq_after_pool1
        self.fc_reduce = nn.Linear(self.flattened_dim, reduced_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tensor wejściowy (batch, channels, n_features, n_frames).

        Returns:
            Krotka (cnn_features, cnn_raw_features):
                - `cnn_features`: sekwencja po redukcji wymiaru przez
                  `fc_reduce`, (batch, seq_len, reduced_dim) - wejście
                  dla RNN zadania zgrubnego.
                - `cnn_raw_features`: spłaszczona sekwencja przed
                  redukcją, (batch, seq_len, channels * freq_bins) -
                  używana później przy fuzji cech zadania precyzyjnego.
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)

        # (batch, channels, freq', time') -> (batch, time', channels, freq')
        x = x.permute(0, 3, 1, 2).contiguous()
        batch_size, seq_len, channels, freq_bins = x.size()
        x = x.reshape(batch_size, seq_len, channels * freq_bins)

        cnn_features = self.fc_reduce(x)
        return cnn_features, x


class HierarchicalMultiTaskCRNN(nn.Module):
    """Hierarchiczny, wielozadaniowy CRNN z mechanizmem uwagi, na
    podstawie architektury 3D-HMTL.

    Oryginalna architektura klasyfikuje emocje na dwóch poziomach
    hierarchii - zgrubnym (`coarse`) i precyzyjnym (`fine`) - łącząc
    (fuzja cech) reprezentację CNN z wektorem uwagi zadania zgrubnego
    przed klasyfikacją precyzyjną.

    Adaptacja do pipeline'u: ponieważ każdy z czterech datasetów SER
    (TESS, RAVDESS, SAVEE, CREMA-D) jest przetwarzany niezależnie i
    żaden z nich nie definiuje w tym projekcie hierarchii coarse/fine,
    model domyślnie działa w trybie single-task - `forward()` zwraca
    tylko logity `fine` o rozmiarze `num_classes`, co czyni go
    bezpośrednio zgodnym z pętlą treningową `engine.py`. Zadanie
    zgrubne (`num_coarse_types`, domyślnie równe `num_classes`) jest
    wciąż trenowane wewnętrznie jako pomocnicze (regularyzujące) -
    zaawansowani użytkownicy mogą uzyskać obie predykcje przez
    `forward_multitask()` i wytrenować model z łączoną funkcją straty
    `L = alpha * L_coarse + (1 - alpha) * L_fine`, tak jak w oryginale.
    """

    def __init__(
        self,
        num_classes: int,
        input_freq_dim: int,
        input_channels: int = 1,
        num_coarse_types: int | None = None,
        rnn_hidden_size: int = 256,
        reduced_dim: int = 256,
    ) -> None:
        """
        Args:
            num_classes: Liczba klas precyzyjnych (`fine`) - liczba
                unikalnych klas emocji w bieżącym datasecie.
            input_freq_dim: Wymiar częstotliwościowy cech wejściowych
                (`n_mfcc` lub `n_mels`).
            input_channels: Liczba kanałów wejściowych (domyślnie 1 -
                pipeline generuje jednokanałowe MFCC/Mel-spektrogramy;
                oryginał zakładał 3 kanały: statyka + delty + delta-delty).
            num_coarse_types: Liczba klas zgrubnych (`coarse`). Jeśli
                None, przyjmowana jest taka sama liczba jak
                `num_classes` (brak zdefiniowanej hierarchii).
            rnn_hidden_size: Rozmiar stanu ukrytego LSTM (na kierunek).
            reduced_dim: Rozmiar sekwencji cech po redukcji CNN->RNN.
        """
        super().__init__()
        num_coarse_types = num_coarse_types or num_classes

        self.cnn_extractor = SharedCNNExtractor(
            input_channels=input_channels,
            input_freq_dim=input_freq_dim,
            reduced_dim=reduced_dim,
        )

        # --- Moduł zadania zgrubnego (coarse) ---
        self.coarse_rnn = nn.LSTM(
            input_size=reduced_dim,
            hidden_size=rnn_hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.coarse_attention = AttentionLayer(hidden_size=rnn_hidden_size * 2)
        self.coarse_fc = nn.Linear(rnn_hidden_size * 2, 64)
        self.coarse_classifier = nn.Linear(64, num_coarse_types)

        # Pooling agregujący cechy CNN w czasie przed fuzją z uwagą zgrubną.
        self.fusion_pool = nn.AdaptiveMaxPool1d(1)

        # --- Moduł zadania precyzyjnego (fine) ---
        fused_size = self.cnn_extractor.flattened_dim + rnn_hidden_size * 2
        self.fine_rnn = nn.LSTM(
            input_size=fused_size,
            hidden_size=rnn_hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.fine_attention = AttentionLayer(hidden_size=rnn_hidden_size * 2)
        self.fine_fc = nn.Linear(rnn_hidden_size * 2, 64)
        self.fine_classifier = nn.Linear(64, num_classes)

    def forward_multitask(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Pełny przebieg hierarchiczny, zwracający obie głowy klasyfikacyjne.

        Args:
            x: Tensor wejściowy (batch, 1, n_features, n_frames).

        Returns:
            Krotka `(coarse_logits, fine_logits)` o kształtach
            `(batch, num_coarse_types)` i `(batch, num_classes)`.
        """
        cnn_seq_features, cnn_raw_features = self.cnn_extractor(x)

        # --- Zadanie zgrubne ---
        coarse_rnn_out, _ = self.coarse_rnn(cnn_seq_features)
        coarse_utterance_feature = self.coarse_attention(coarse_rnn_out)
        coarse_fc_out = F.relu(self.coarse_fc(coarse_utterance_feature))
        y_coarse = self.coarse_classifier(coarse_fc_out)

        # --- Fuzja cech ---
        cnn_pool = cnn_raw_features.permute(0, 2, 1)
        cnn_pool = self.fusion_pool(cnn_pool).squeeze(-1)
        fused_features = torch.cat((cnn_pool, coarse_utterance_feature), dim=1)

        seq_len = cnn_seq_features.size(1)
        fused_seq = fused_features.unsqueeze(1).repeat(1, seq_len, 1)

        # --- Zadanie precyzyjne ---
        fine_rnn_out, _ = self.fine_rnn(fused_seq)
        fine_utterance_feature = self.fine_attention(fine_rnn_out)
        fine_fc_out = F.relu(self.fine_fc(fine_utterance_feature))
        y_fine = self.fine_classifier(fine_fc_out)

        return y_coarse, y_fine

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Zwraca tylko logity zadania precyzyjnego (`fine`), aby zachować
        zgodność z jednozadaniową pętlą treningową `engine.py`.

        Args:
            x: Tensor wejściowy (batch, 1, n_features, n_frames).

        Returns:
            Logity o kształcie (batch, num_classes).
        """
        _, y_fine = self.forward_multitask(x)
        return y_fine


# ===========================================================================
# 5. GenderAwareSERModel - dwugłowicowy CNN+LSTM z fuzją płci (gender fusion)
# ===========================================================================
class GenderAwareSERModel(nn.Module):
    """Dwugłowicowy model SER z fuzją predykcji płci, zbudowany na bazie
    architektury CNN+LSTM.

    Kluczowa innowacja: model predykuje płeć mówcy (Male/Female) jako
    zadanie pomocnicze, a prawdopodobieństwa wyjściowe rozkładu płci
    (softmax) są doklejane do głównego wektora cech przed końcową
    klasyfikacją emocji (gender-aware feature fusion).

    Adaptacja do pipeline'u:
        1. Zahardkowane `cnn_out_features = 64 * 16` (dla mapy wejściowej
           128 pasm Mel w oryginalnym rozmiarze) zastąpiono wyliczeniem
           dynamicznym z `input_freq_dim`.
        2. `forward()` zwraca logity emocji (zgodność z pętlą
           treningową `engine.py`). Pełny tryb dwugłowicowy dostępny
           jest przez `forward_multitask()`, który zwraca
           `(gender_logits, emotion_logits)`. Trening multi-task z
           połączoną funkcją straty (identycznie jak dla
           `HierarchicalMultiTaskCRNN`) jest aktywowany automatycznie
           przez `engine._run_one_epoch`.

    Uwaga dot. danych: ponieważ datasety SER nie mają wprost
    zaetykietowanych mówców binarną płcią w wierszach DataFrame,
    etykieta płci w bieżącej implementacji NIE jest używana jako
    target (strategia "unsupervised auxiliary task" - wykrywanie
    `hasattr(model, "forward_multitask")` powoduje, że `engine.py`
    próbuje użyć `coarse_labels` z trzeciej pozycji batcha zamiast
    płci). Dla pełnego treningu dwugłowicowego z etykietami płci
    referencyjnych należy rozszerzyć `dataset.py` o dodatkową kolumnę
    `gender` w DataFrame i osobny `GenderLabelEncoder` - zostanie to
    zaimplementowane przy kolejnej iteracji.
    """

    def __init__(
        self,
        num_classes: int,
        input_freq_dim: int,
        input_channels: int = 1,
        lstm_hidden_size: int = 128,
        cnn_out_freq_reduced: int = 32,
    ) -> None:
        """
        Args:
            num_classes: Liczba unikalnych klas emocji w bieżącym datasecie.
            input_freq_dim: Wymiar częstotliwościowy cech wejściowych
                (`n_mfcc` lub `n_mels`).
            input_channels: Liczba kanałów wejściowych (domyślnie 1 -
                pipeline generuje jednokanałowe MFCC/Mel-spektrogramy).
            lstm_hidden_size: Rozmiar stanu ukrytego LSTM.
            cnn_out_freq_reduced: Oczekiwany wymiar częstotliwościowy po
                dwóch warstwach MaxPool2d (kernel_size=2). Domyślna
                wartość 32 odpowiada oryginalnemu 128 pasm Mel / 4.
        """
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels, out_channels=32, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        pooled_freq = max(input_freq_dim // 4, 1)
        cnn_out_features = 64 * pooled_freq

        self.lstm = nn.LSTM(
            input_size=cnn_out_features,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.gender_head = nn.Sequential(
            nn.Linear(lstm_hidden_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 2),  # 2 logity: Female / Male
        )

        self.emotion_head = nn.Sequential(
            nn.Linear(lstm_hidden_size + 2, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes),
        )

    def forward_multitask(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Pełny przebieg dwugłowicowy.

        Args:
            x: Tensor wejściowy (batch, 1, n_features, n_frames).

        Returns:
            Krotka `(gender_logits, emotion_logits)` o kształtach
            `(batch, 2)` i `(batch, num_classes)`.
        """
        x = self.cnn(x)  # (batch, 64, freq', time')
        x = x.permute(0, 3, 1, 2)  # (batch, time', 64, freq')

        batch_size, seq_len, channels, height = x.size()
        x = x.reshape(batch_size, seq_len, channels * height)

        lstm_out, _ = self.lstm(x)
        features = lstm_out[:, -1, :]  # ostatni krok czasowy

        gender_logits = self.gender_head(features)
        gender_probs = torch.softmax(gender_logits, dim=1)
        fused_features = torch.cat((features, gender_probs), dim=1)
        emotion_logits = self.emotion_head(fused_features)

        return gender_logits, emotion_logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Zwraca tylko logity emocji (zgodność z kontraktem pipeline'u).

        Args:
            x: Tensor wejściowy (batch, 1, n_features, n_frames).

        Returns:
            Logity emocji o kształcie (batch, num_classes).
        """
        _, emotion_logits = self.forward_multitask(x)
        return emotion_logits


# ===========================================================================
# Rejestr modeli i fabryka
# ===========================================================================
MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "dscnn": DSCNNEmotionClassifier,
    "headfusion_acnn": HeadFusionACNN,
    "lstm": KumbharLSTMClassifier,
    "hierarchical_crnn": HierarchicalMultiTaskCRNN,
    "gender_aware": GenderAwareSERModel,
}


def build_model(
    model_name: str, num_classes: int, input_freq_dim: int, **kwargs
) -> nn.Module:
    """Fabryka modeli - tworzy instancję wybranej architektury SER.

    Wszystkie modele zarejestrowane w `MODEL_REGISTRY` przyjmują ten
    sam, ujednolicony interfejs konstruktora (`num_classes`,
    `input_freq_dim`, ...) oraz ten sam kontrakt `forward()` (wejście:
    `(batch, 1, n_features, n_frames)`, wyjście: `(batch, num_classes)`),
    dzięki czemu mogą być używane wymiennie w
    `main.run_pipeline_for_dataset` bez zmian w reszcie pipeline'u.

    Args:
        model_name: Klucz z `MODEL_REGISTRY`, np. "dscnn",
            "headfusion_acnn", "lstm", "hierarchical_crnn".
        num_classes: Liczba unikalnych klas emocji w bieżącym datasecie.
        input_freq_dim: Wymiar częstotliwościowy cech wejściowych
            (`n_mfcc` lub `n_mels`), zależny od konfiguracji audio
            (`AudioProcessingConfig.feature_dim`). Niektóre modele
            (`dscnn`, `headfusion_acnn`) ignorują ten parametr, ponieważ
            wykorzystują globalny adaptacyjny pooling niezależny od
            rozmiaru wejścia.
        **kwargs: Dodatkowe argumenty przekazywane do konstruktora
            wybranego modelu (np. `dropout`, `lstm_hidden_size`).

    Returns:
        Nieprzeniesiona na urządzenie (CPU) instancja modelu.

    Raises:
        ValueError: Gdy `model_name` nie znajduje się w `MODEL_REGISTRY`.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Nieznany model: '{model_name}'. Dostępne opcje: "
            f"{list(MODEL_REGISTRY.keys())}"
        )
    model_cls = MODEL_REGISTRY[model_name]
    return model_cls(num_classes=num_classes, input_freq_dim=input_freq_dim, **kwargs)
