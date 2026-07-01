# Data Format

Raw EEG data are not included in this repository.

This folder documents the expected input data format for running the MIMOME FgMDM transfer decoding scripts.

## Expected `.mat` Variables

Each input `.mat` file should contain:

| Variable | Description | Expected shape |
|---|---|---|
| `prepro_data` | Preprocessed EEG signal | channels × samples |
| `event` | Trigger vector | 1 × samples |
| `srate` | Sampling rate | scalar |
| `chanlocs` | Channel information | optional |

## Trigger Labels

| Trigger value | Meaning |
|---|---|
| `1` | Left-hand task |
| `2` | Right-hand task |

Only trials with trigger labels `1` and `2` are used for classification.

## Experimental Structure

The original study used three motor tasks:

| Task | Meaning |
|---|---|
| `ME` | Motor execution |
| `MI` | Motor imagery |
| `MO` | Motor observation |

Each task contains:

- 2 sessions
- 50 left-hand trials
- 50 right-hand trials
- 100 trials total per task

## Recommended Folder Structure

Example:

```text
data/
├── S01_ME_session1.mat
├── S01_ME_session2.mat
├── S01_MI_session1.mat
├── S01_MI_session2.mat
├── S01_MO_session1.mat
├── S01_MO_session2.mat
└── ...