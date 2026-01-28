import torch
import torch.nn as nn

class LSTM_CLASSIF(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        dropout_p: float = 0.5,
    ):
        super().__init__()

        self.output_mode = "logits"  # or "embedding"

        self.acc_lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True
        )
        self.gyr_lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True
        )
        self.mag_lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True
        )
        self.mic_lstm = nn.LSTM(
            1, hidden_size, num_layers,
            batch_first=True
        )

        # Dropout after each modality
        self.modality_dropout = nn.Dropout(dropout_p)

        # FC head with dropout
        # self.fc = nn.Sequential(
        #     nn.Linear(hidden_size * 4, 64),
        #     nn.ReLU(),
        #     nn.Dropout(dropout_p),
        #     nn.Linear(64, num_classes),
        # )

        self.embedding = nn.Sequential(
            nn.Linear(hidden_size * 4, 64),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            )

        self.classifier = nn.Linear(64, num_classes)


    # def forward(self, X):
    #     Xt_acc_f, Xt_gyr_f, Xt_mag_f, Xt_mic_f = X

    #     _, (hn_acc, _) = self.acc_lstm(Xt_acc_f)
    #     acc = self.modality_dropout(hn_acc[-1])

    #     _, (hn_gyr, _) = self.gyr_lstm(Xt_gyr_f)
    #     gyr = self.modality_dropout(hn_gyr[-1])

    #     _, (hn_mag, _) = self.mag_lstm(Xt_mag_f)
    #     mag = self.modality_dropout(hn_mag[-1])

    #     _, (hn_mic, _) = self.mic_lstm(Xt_mic_f)
    #     mic = self.modality_dropout(hn_mic[-1])

    #     fused = torch.cat((acc, gyr, mag, mic), dim=1)
    #     return self.fc(fused)

    def forward(self, X):
        Xt_acc_f, Xt_gyr_f, Xt_mag_f, Xt_mic_f = X

        _, (hn_acc, _) = self.acc_lstm(Xt_acc_f)
        acc = self.modality_dropout(hn_acc[-1])

        _, (hn_gyr, _) = self.gyr_lstm(Xt_gyr_f)
        gyr = self.modality_dropout(hn_gyr[-1])

        _, (hn_mag, _) = self.mag_lstm(Xt_mag_f)
        mag = self.modality_dropout(hn_mag[-1])

        _, (hn_mic, _) = self.mic_lstm(Xt_mic_f)
        mic = self.modality_dropout(hn_mic[-1])

        fused = torch.cat((acc, gyr, mag, mic), dim=1)

        emb = self.embedding(fused)

        logits = self.classifier(emb)

        if self.output_mode == "embedding":
            return emb, logits

        return logits


