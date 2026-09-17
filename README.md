# AI vs. Human-Generated Image Classifier

A PyTorch image classifier that distinguishes AI-generated images from authentic, human-created ones. Built for the **[Detect AI vs. Human-Generated Images](https://www.kaggle.com/competitions/detect-ai-vs-human-generated-images)** Kaggle competition (2025 Women in AI Kaggle Challenge).

## Competition background

Generative AI has made it increasingly difficult to tell real images apart from synthetic ones, with implications for media trust, security, and misinformation. The challenge asked participants to build a binary classifier — AI-generated vs. human-created — evaluated by F1-score at a 0.5 decision threshold.

The dataset was provided by Shutterstock (authentic images) and DeepMedia (AI-generated counterparts): roughly 80,000 labeled training images and 5,540 unlabeled test images. Details and download: [competition dataset page](https://www.kaggle.com/datasets/alessandrasala79/ai-vs-human-generated-dataset).

## Approach

The model is a **ConvNeXt-Base** backbone (ImageNet-pretrained) fine-tuned for binary classification:

- Only the last two feature stages of the backbone are unfrozen; earlier stages keep their pretrained weights.
- The classifier head is replaced with global average pooling → BatchNorm → Linear(1024→512) → ReLU → Dropout(0.4) → Linear(512→2).
- Training uses discriminative learning rates (1e-5 for the backbone, 1e-4 for the head) with AdamW and a `StepLR` schedule (step size 5, gamma 0.7).
- Data augmentation: `RandomResizedCrop(224)`, horizontal flips, and color jitter; validation/test images are center-cropped to 224x224.
- Trained for 12 epochs, batch size 32, on a 95/5 stratified train/validation split.

## Results

| Metric | Value |
|---|---|
| Validation accuracy (held-out 5%) | 98.15% |
| Validation F1-score | 0.9815 |
| Kaggle public leaderboard F1-score | 0.733 |

The gap between local validation and the public leaderboard reflects the difference between the held-out split (drawn from the same training distribution) and the competition's independent test set.

## Repository structure

```
.
├── 0-73298-convnext-classifier.ipynb   # original, end-to-end exploratory/training notebook
├── src/
│   ├── dataset.py    # Dataset classes and image transforms
│   ├── model.py       # ConvNeXt-Base model + optimizer setup
│   ├── train.py       # Training / validation loop (script version of the notebook)
│   └── predict.py     # Inference + submission.csv generation
├── requirements.txt
└── LICENSE
```

The notebook (`0-73298-convnext-classifier.ipynb`) is the original, self-contained version used during the competition, run directly in a Kaggle Notebook environment. The `src/` scripts are a refactored, reusable version of the same pipeline for running outside Kaggle.

## Setup

```bash
pip install -r requirements.txt
```

### Data

The dataset is not included in this repository. Download it from the [competition data page](https://www.kaggle.com/competitions/detect-ai-vs-human-generated-images/data) or the [dataset page](https://www.kaggle.com/datasets/alessandrasala79/ai-vs-human-generated-dataset), and arrange it as:

```
data/
├── ai-vs-human-generated-dataset/   # image files
├── train.csv
└── test.csv
```

### Train

```bash
python src/train.py --epochs 12 --batch-size 32
```

### Predict

```bash
python src/predict.py --checkpoint model.pt
```

This writes `submission.csv` (Kaggle submission format: `id,label`) and `test_logits.csv` (raw pre-softmax logits per class).

## Notes

- GPU strongly recommended; the notebook was originally trained on a Kaggle GPU instance (~13 minutes/epoch on ConvNeXt-Base).
- The original notebook is kept as-is for reproducibility of the exact competition submission; the `src/` scripts are provided for easier reuse and are functionally equivalent.

## License

MIT — see [LICENSE](LICENSE).
