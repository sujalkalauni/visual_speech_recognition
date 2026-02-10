from dataset import LipDataset
#made by me
dataset = LipDataset(
    frames_root="data/mouth_frames",
    labels_csv="data/labels.csv"
)

frames, label = dataset[0]
print("Frames shape:", frames.shape)
print("Label:", label)
