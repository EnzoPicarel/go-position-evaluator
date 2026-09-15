<div align="center">
  <h3 align="center">Go Position Evaluator</h3>

  <p align="center">
   A PyTorch convolutional value network estimating <strong>Black's win probability</strong> from an 8×8 Go position.
    <br />
    <a href="#-getting-started"><strong>Quick Start »</strong></a>
  </p>

![CI Status](https://img.shields.io/badge/build-passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)

</div>

## 🔍 About The Project

This project implements a complete deep-learning pipeline for evaluating 8×8 Go positions. Each board is encoded as two binary planes—one for Black stones and one for White stones—then processed by a convolutional neural network that predicts Black's empirical win probability.

The model was originally developed for a graded machine-learning practical at ENSEIRB-MATMECA. This portfolio version turns the coursework into a reproducible command-line project, adds automated tests and inference tooling, and corrects the validation split to prevent symmetric copies of the same board from appearing in both training and validation.

Built as an academic AI project focusing on spatial representation, convolutional networks, and reliable model evaluation.

## 🛠 Built With

- **Language:** Python 3.10+
- **Deep Learning:** PyTorch
- **Data Processing:** NumPy
- **Visualization:** Matplotlib
- **Experimentation:** Jupyter Notebook

## 📐 Architecture

### Technical Highlights

- **Spatial Board Encoding:** Represents each position as a `2 × 8 × 8` tensor, preserving the geometry of the Go board.
- **Convolutional Value Network:** Uses three convolutional blocks with Batch Normalization and LeakyReLU, followed by a dense prediction head.
- **Symmetry-Based Augmentation:** Generates all eight rotations and reflections of each training position.
- **Leakage-Safe Validation:** Groups equivalent boards before splitting the data, then applies augmentation only to the training set.
- **Reproducible Training Pipeline:** Supports deterministic seeds, CPU/CUDA/MPS execution, checkpoint export, learning curves, and baseline comparison.

| Component    | Architecture                                          |
| ------------ | ----------------------------------------------------- |
| Input        | 2 × 8 × 8 binary planes                               |
| Features     | Conv 2→32→64→64, 3×3 kernels, BatchNorm2d + LeakyReLU |
| Head         | Flatten → 128 → 64 → 1, dropout 0.3, sigmoid          |
| Optimization | Adam, learning rate 0.001, weight decay 0.0001        |

### File Organization

```text
├── go_model.py                    # Encoding, augmentation, CNN, training and inference
├── train.py                       # Training command-line interface
├── predict.py                     # Batch prediction command-line interface
├── go_position_evaluation.ipynb   # English experiment walkthrough
├── data/
│   └── samples-8x8.json            # Included 8×8 Go positions dataset
├── tests/
│   └── test_model.py              # Encoding, split, augmentation and checkpoint tests
├── VALIDATION.md                  # Verification results and limitations
└── requirements.txt               # Python dependencies
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- PIP
- The course dataset or another compatible 8×8 Go dataset

### Installation & Build

1. **Clone and set up the environment**

```bash
git clone https://github.com/EnzoPicarel/go-position-evaluator.git
cd go-position-evaluator

python -m venv .venv
source .venv/bin/activate
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

For GPU execution, install the PyTorch build recommended for your hardware in the [official installation guide](https://pytorch.org/get-started/locally/).

### Dataset

The repository includes the course dataset as `data/samples-8x8.json`. The original compressed source is available [here](https://www.labri.fr/perso/lsimon/static/inge2-ia/samples-8x8.json.gz). The instructor-provided assignment is not included in this repository.

The course handout describes **21,854 positions**, while the local dataset used for the final submission contains **41,553 positions**. Targets are empirical Black win frequencies from rollouts rather than exact game-theoretic values.

Expected JSON format:

```json
[
  {
    "black_stones": ["A1", "C3"],
    "white_stones": ["B2"],
    "black_wins": 60,
    "rollouts": 100
  }
]
```

Coordinates range from `A1` to `H8`. The example above only illustrates the schema.

## ⚡ Execution

Train the CNN and save its checkpoint, metrics, and learning history:

```bash
python train.py --data data/samples-8x8.json --epochs 20 --output artifacts
```

Optional commands:

```bash
# GPU training
python train.py --data data/samples-8x8.json --device cuda --output artifacts

# Short execution check
python train.py --data data/samples-8x8.json --limit 256 --epochs 1 --output artifacts
```

Generated files:

- `artifacts/model.pt` — trained model checkpoint.
- `artifacts/history.json` — training and validation loss history.
- `artifacts/metrics.json` — validation BCE/MAE and constant-prediction baseline.

Run batch inference on compatible positions:

```bash
python predict.py \
  --data data/positions.json.gz \
  --checkpoint artifacts/model.pt \
  --output artifacts/predictions.txt
```

The notebook `go_position_evaluation.ipynb` provides an interactive walkthrough and plots the saved learning curves.

## 🧪 Tests

Run the automated test suite:

```bash
python -m unittest discover -s tests -v
```

The tests verify:

1. Board-coordinate encoding and tensor shape.
2. Eightfold rotation/reflection augmentation.
3. Isolation of symmetric board groups across the train/validation split.
4. Model checkpoint creation and loading.

See [`VALIDATION.md`](VALIDATION.md) for the recorded verification results.

## ⚠️ Evaluation Notes

The original coursework augmented all positions before randomly splitting them, allowing equivalent boards to appear in both training and validation. Historical validation scores from that workflow are therefore not presented as evidence of generalization.

This version groups identical boards and all rotations/reflections before the split, then augments only the training positions. Positions from the same game may still cross the split because the dataset does not provide reliable game identifiers.

The network does not encode side to move, ko history, or komi and is not a complete Go-playing engine. Inference uses the supplied board orientation, so augmentation alone does not guarantee exact rotation/reflection invariance.

## 👥 Authors

- **Numa Guiot**
- **Enzo Picarel**
