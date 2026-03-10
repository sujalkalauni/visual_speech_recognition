# Visual Speech Recognition

A deep learning model that reads lips — predicts spoken words from silent video by analyzing mouth movements frame by frame.

## Problem

Audio-based speech recognition fails in noisy environments or for the hearing-impaired. This project explores visual-only speech recognition using CNNs and sequence models to decode lip motion without any audio input.

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)

---

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/sujalkalauni/visual_speech_recognition.git
cd visual_speech_recognition

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## How to Run

```bash
# Preprocess the dataset (extract frames from videos)
python preprocess/extract_frames.py

# Prepare dataset splits
python dataset.py

# Train the model
python train.py

# Run prediction on a test sample
python predict.py --input data/test_sample.mp4

# Evaluate on the test set
python test_dataset.py
```

---

## Project Structure

```text
visual_speech_recognition/
├─ data/                   # Raw and processed video/frame data
├─ model/                  # Saved model weights and checkpoints
├─ preprocess/             # Frame extraction and preprocessing scripts
├─ dataset.py              # Dataset loading and split logic
├─ train.py                # Model training script
├─ predict.py              # Inference on new video input
├─ test_dataset.py         # Test set evaluation
├─ test_extract.py         # Unit tests for preprocessing
├─ requirements.txt
├─ .gitignore
├─ LICENSE
└─ README.md
```

---

## What I Learned

- Building CNN + sequence model pipelines for video data.
- Handling frame extraction, normalization, and batching for video inputs.
- Structuring an ML project with separate preprocessing, training, and inference stages.
- Working with temporal data and understanding how models learn from visual sequences.

## Future Improvements

- Replace custom model with a pretrained LipNet or similar architecture.
- Expand dataset coverage for better generalization across speakers.
- Add a REST API to serve predictions from a video upload.
- Containerize with Docker for reproducible training environments.

---

## License

MIT © 2026 Sujal Kalauni
