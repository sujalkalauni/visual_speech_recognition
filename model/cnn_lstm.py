import torch
import torch.nn as nn
from model.cnn import CNN

class CNNLSTM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.cnn = CNN()
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=1,
            batch_first=True
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: B x T x 1 x 64 x 64
        B, T, C, H, W = x.shape

        x = x.view(B * T, C, H, W)
        feats = self.cnn(x)           # (B*T) x 256
        feats = feats.view(B, T, -1)  # B x T x 256

        _, (hn, _) = self.lstm(feats)
        out = self.classifier(hn[-1]) # B x num_classes
        return out
