# EEG Generation Research Project

This folder contains a cleaned, GitHub-ready version of an EEG generation research workspace. The main contribution of this work is generating synthetic EEG data and studying how generated EEG can be evaluated and compared against real EEG.

The ARL EEGModels code is included only as a downstream evaluation component. It is used to test generated EEG through EEGNet-based classification experiments, not as the central contribution of this repository.

## Repository Layout

- eeg-diffusion-generation/ - primary project: EEG signal/image generation experiments using diffusion, GAN, VAE, DDPM, EFDM conversion, and generated-data comparison workflows.
- arl-eegmodels-eegnet-classification/ - evaluation support: ARL EEGModels/EEGNet scripts and notebooks used to evaluate generated EEG with downstream classification.

## Dataset

The experiments use the public Cho et al. 2017 motor imagery EEG dataset described in GigaScience as EEG datasets for motor imagery brain-computer interface. The dataset contains left-hand and right-hand motor imagery recordings from 52 subjects, collected with a 64-channel 10-10 EEG montage at 512 Hz, with simultaneous EMG recordings used to check actual hand movement.

Dataset reference: https://academic.oup.com/gigascience/article/6/7/gix034/3796323

The dataset and derived arrays are not included because they are too large for a normal GitHub repository. Without obtaining and preprocessing the original dataset, the training and evaluation notebooks are documentation of the research workflow rather than fully reproducible scripts.

## What was changed for GitHub

- Source code, notebooks, result figures, README files, dependency notes, and license files were organized into clear folders.
- Large generated artifacts such as TensorFlow checkpoints, PyTorch models, .pickle datasets, and generated data folders were intentionally left out.
- The READMEs describe the public dataset source and the preprocessing used, while making clear that the experiments are not directly reproducible without the dataset.

## License and Third-Party Code

See LICENSE.md for the license status of the original EEG generation work and the third-party ARL EEGModels evaluation code.

The original folders were not modified.
