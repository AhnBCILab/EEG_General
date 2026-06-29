# EEG General Research Repository

This repository is an EEG research workspace focused on motor task decoding, generative EEG modeling, and sleep-stage classification. It combines a MATLAB-based motor transfer learning project with additional EEG research modules.

## Overview

The primary project in this repository is the `MIMOME_FgMDM_transfer-main` folder, which implements FgMDM-based EEG decoding and task-to-task transfer learning across motor execution (ME), motor imagery (MI), and motor observation (MO).

In addition, this repository is associated with broader EEG research modules that include:

- EEG generative modeling and diffusion-based EEG synthesis
- EEGNet-based downstream evaluation of generated EEG
- Sleep-stage classification using raw EEG and PSD features

## MIMOME_FgMDM_transfer-main

This folder contains MATLAB code and documentation for a study of motor-task transfer learning in MI-BCI.

### Key goals

- Evaluate within-task decoding for ME, MI, and MO.
- Evaluate task-to-task transfer decoding across ME, MI, and MO.
- Determine whether easier motor tasks can reduce calibration burden for MI decoding.

### Core workflow

1. Load preprocessed EEG data from `.mat` files.
2. Band-pass filter the signal.
3. Epoch data around cue onset.
4. Separate left-hand and right-hand trials using trigger labels.
5. Compute covariance matrices from sliding windows.
6. Apply the FgMDM classifier.
7. Summarize classification accuracy across time windows and tasks.

### Main files and folders

- `MIMOME_FgMDM_transfer-main/README.md` : local project documentation.
- `MIMOME_FgMDM_transfer-main/data/README.md` : expected `.mat` input format and trigger definitions.
- `MIMOME_FgMDM_transfer-main/external/README.md` : external MATLAB toolbox requirements and installation notes.
- `MIMOME_FgMDM_transfer-main/private/` : helper utilities and internal code support.
- `MIMOME_FgMDM_transfer-main/results/` : output storage location for generated results.

### Data format and inputs

The local motor transfer learning project expects input `.mat` files with the following variables:

- `prepro_data` : preprocessed EEG signal (`channels × samples`)
- `event` : trigger labels indicating left/right trials (`1` for left, `2` for right)
- `srate` : sampling rate
- `chanlocs` : optional channel location metadata

The project assumes a motor task dataset organized by task and session, such as:

- `ME` : motor execution
- `MI` : motor imagery
- `MO` : motor observation

Each task is expected to include left-hand and right-hand trials.

### Dependency requirements

The local MATLAB project depends on external toolboxes:

- MATLAB
- FieldTrip
- AdaRGC / FgMDM implementation
- Covariance toolbox

The `external/README.md` documents how to set up these dependencies and verify required functions like `FgMDM` and `ft_preproc_bandpassfilter`.

## Associated EEG research modules

This repository is linked to additional EEG research work that extends beyond the local motor transfer learning project.

### EEG generation and evaluation (`GenAI`)

This module contains code for EEG generative modeling and evaluation, including:

- diffusion and DDPM-based EEG generation
- GAN and VAE baselines for EEG synthesis
- EFDM signal conversion workflows
- downstream evaluation using EEGNet and ARL EEGModels

Important notes:

- The primary contribution in this branch is synthetic EEG generation and evaluation.
- The EEGNet/ARL EEGModels code is provided as an evaluation component, not as the central method.
- Large datasets, model checkpoints, and generated artifacts are not included.

### Sleep-stage classification (`SleepStageClassification`)

This module contains Python code for training a sleep-stage classifier on the Sleep-EDF dataset.

Key components:

- `main.py` : training entry point for leave-one-subject-out (LOSO) experiments
- `data.py` : EDF and hypnogram loading
- `preprocessing.py` : z-score normalization and Welch PSD extraction
- `datasets.py` : PyTorch sequence dataset
- `model.py` : raw EEG + PSD neural network
- `trainer.py` : training, validation, and prediction loops
- `metrics.py` : evaluation metrics and class weighting
- `plots.py` : confusion matrix and result visualization

This module is designed for research on sleep-stage classification and requires separate access to the Sleep-EDF Expanded dataset.

## Usage guidance

### Running the motor transfer learning project

1. Install MATLAB and required toolboxes.
2. Prepare `.mat` input files in the expected format.
3. Configure local paths and toolbox locations in the project configuration.
4. Run the project scripts for within-task and cross-task decoding.

### Reproducibility notes

The local `MIMOME_FgMDM_transfer-main` project documents the analysis pipeline, but it does not include raw EEG data.

The associated GenAI and SleepStageClassification modules also document research workflows without including large training datasets or model checkpoints.

## Research reference

The motor transfer learning project is associated with the study:

Daeun Gwon and Minkyu Ahn, "Motor task-to-task transfer learning for motor imagery brain-computer interfaces," NeuroImage, 302, 120906, 2024.

## Summary

This repository is a combined EEG research workspace that includes:

- a MATLAB motor task transfer learning study using FgMDM
- EEG generation and diffusion modeling experiments
- EEGNet-based generated EEG evaluation code
- a sleep-stage classification research pipeline

It is intended for EEG researchers and practitioners who want to explore both classical Riemannian geometry-based EEG decoding and modern generative/deep learning EEG workflows.
