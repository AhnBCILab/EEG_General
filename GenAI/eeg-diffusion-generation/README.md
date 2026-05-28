# EEG Diffusion and Generative Modeling Experiments

This is the primary contribution of the repository. It contains EEG generation experiments using diffusion models, DDPM-style workflows, GAN/VAE baselines, EFDM-to-signal conversion, and generated-data comparison notebooks.

The main goal is to generate synthetic EEG data and evaluate whether the generated samples preserve useful EEG structure. Evaluation is done with signal reconstruction/comparison notebooks, correlation analysis, visual inspection, and downstream classifiers such as EEGNet from ARL EEGModels.

## Repository Layout

- src/ - reusable Python scripts for diffusion/generation preprocessing and model runs.
- notebooks/ - experiment notebooks for diffusion, DDPM, GAN, VAE, EFDM conversion, generated-data comparison, and classification checks.
- results/ - exported plots from the original workspace.

## Main Workflows

1. Convert EEG or EFDM representations into signal-like training data.
2. Train or call diffusion/DDPM-style generators for 32-channel and 64-channel EEG representations.
3. Generate synthetic EEG samples for selected subjects/channels.
4. Compare generated and real data through visualizations, correlation analysis, and downstream EEG classification.
5. Use ARL EEGModels/EEGNet as an evaluation tool for generated EEG, not as the main method of this project.

## Dataset and Preprocessing

The project uses the public Cho et al. 2017 motor imagery EEG dataset from GigaScience/GigaDB, also described as EEG datasets for motor imagery brain-computer interface. It contains left-hand and right-hand motor imagery data from 52 subjects recorded with 64 EEG channels at 512 Hz, plus simultaneous EMG channels.

Dataset reference: https://academic.oup.com/gigascience/article/6/7/gix034/3796323

The preprocessing used in this project can be summarized as:

1. Load subject-level left-hand and right-hand motor imagery trials.
2. Use the 64 EEG channels, with C3 and C4 used in focused channel experiments.
3. Convert EEG trials into time-frequency / EFDM-like image representations using short-time Fourier transforms. Common settings in the scripts include Hann windows, nperseg 256 with noverlap 192 and nfft 512, plus a not-cropped variant using nperseg 128, noverlap 64, and nfft 768.
4. Separate left-hand and right-hand trials into two classes and one-hot encode labels for conditional generation/classification.
5. Normalize image-like tensors with dataset-level min-max scaling and model normalization layers.
6. Train diffusion/DDPM-style generators on the transformed EEG representations, then recover or compare generated samples against real EEG-derived representations.
7. Evaluate generated EEG using visual comparison, correlation analysis, signal reconstruction checks, and downstream EEGNet classification.

The original workspace contained about 42 GB of generated data, TensorFlow checkpoints, PyTorch weights, and .pickle arrays. These are not copied into the GitHub-ready project because they are too large for a normal Git repository. Without the original dataset and derived preprocessing outputs, this repository documents the workflow and code structure, but the experiments are not directly reproducible.

## Setup

1. Create a Python virtual environment.
2. Activate the environment.
3. Install dependencies from requirements.txt.

Because the dataset and derived training artifacts are not included, the notebooks should be treated as a record of the research workflow unless the original dataset is obtained and preprocessed again.

## Notes

This is a research workspace cleaned for upload, not a packaged Python library. The notebooks are preserved so the experimental history remains visible, while the large training outputs are kept out of Git.

The generation work in this folder is separate from the ARL EEGModels evaluation code. See the repository-level LICENSE.md for license and third-party-code notes.
