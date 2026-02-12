import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import LipDataset
from model.cnn_lstm import CNNLSTM
#made bby sujal kalauni
# ---- config ----
NUM_CLASSES = 1 + 0  # currently only label 0 exists
BATCH_SIZE = 1
EPOCHS = 3
LR = 1e-3



# ---- dataset ----
dataset = LipDataset(
    frames_root="data/mouth_frames",
    labels_csv="data/labels.csv"
)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ---- model ----
model = CNNLSTM(num_classes=1)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ---- training ----
for epoch in range(EPOCHS):
    for frames, label in loader:
        # frames: B x T x 1 x 64 x 64
        label = label.long()

        output = model(frames)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {loss.item():.4f}")

print("Training finished.")
