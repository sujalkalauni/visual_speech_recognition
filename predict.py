import torch
from model.cnn_lstm import CNNLSTM
from dataset import LipDataset

# load model
model = CNNLSTM(num_classes=1)
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

# load sample
dataset = LipDataset(
    frames_root="data/mouth_frames",
    labels_csv="data/labels.csv"
)

frames, _ = dataset[0]
frames = frames.unsqueeze(0)  # add batch dim

with torch.no_grad():
    output = model(frames)
    pred = torch.argmax(output, dim=1).item()

print("Predicted label:", pred)
