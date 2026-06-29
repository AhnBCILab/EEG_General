# Sleep-EDF Raw EEG + PSD LOSO Classifier

This repository trains a 5-class sleep-stage classifier using raw EEG epochs and Welch PSD
features from the Sleep-EDF Expanded dataset.

## Files

- `main.py` - command-line entry point for leave-one-subject-out training.
- `data.py` - Sleep-EDF EDF and hypnogram loading.
- `preprocessing.py` - z-score normalization and Welch PSD extraction.
- `datasets.py` - PyTorch sequence dataset.
- `model.py` - raw EEG + PSD neural network.
- `trainer.py` - training, validation, and prediction loops.
- `metrics.py` - evaluation metrics and class weights.
- `plots.py` - confusion matrix plotting helper.

## Setup

Install Python dependencies:

```bash
pip install torch numpy pandas scipy scikit-learn mne tqdm matplotlib seaborn
```

Place the Sleep-EDF Expanded dataset in this layout inside this project folder,
or pass a custom location with --data-root:

```text
MouseData/sleep_data/sleep-edf-database-expanded-1.0.0/
└── sleep-edf-database-expanded-1.0.0/
    ├── RECORDS-v1
    ├── sleep-cassette/
    └── sleep-telemetry/
```

## Run

From this directory:

```bash
python main.py
```

Common options:

```bash
python main.py --epochs 50 --batch-size 64 --seq-len 15 --output-dir outputs/raw_psd_loso
```

Outputs are written as CSV files under `outputs/raw_psd_loso/`:

- fold metrics
- fold predictions
- fold training histories
- aggregate metrics
- aggregate confusion matrix

## Notes

The default run uses only Sleep-EDF cassette records (`SC*`) and channel
`EEG Fpz-Cz`, matching the notebook. Add `--include-telemetry` to include
telemetry records.
The included .gitignore excludes notebooks, datasets, virtual environments,
model checkpoints, and generated outputs so the GitHub repository can contain
only the reusable code.

## Data License

This project expects users to download the Sleep-EDF Expanded dataset separately
from PhysioNet. The dataset is distributed under the Open Data Commons
Attribution License v1.0 (ODC-BY 1.0). Users are responsible for following the
dataset license and citing Sleep-EDF/PhysioNet when using the data.
