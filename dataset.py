import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import csv

class LipDataset(Dataset):
    def __init__(self, frames_root, labels_csv):
        self.frames_root = frames_root
        self.samples = []

        with open(labels_csv, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append((row["video"], int(row["label"])))

        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video, label = self.samples[idx]
        frame_dir = os.path.join(self.frames_root, video)

        frames = []
        for img_name in sorted(os.listdir(frame_dir)):
            img_path = os.path.join(frame_dir, img_name)
            img = Image.open(img_path)
            img = self.transform(img)
            frames.append(img)

        frames = torch.stack(frames)  # T x 1 x 64 x 64
        return frames, label
