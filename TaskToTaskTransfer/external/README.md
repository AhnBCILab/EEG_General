# External Dependencies

This folder documents the external MATLAB toolboxes required to run the FgMDM-based MIMOME transfer decoding scripts.

By default, this repository does **not** include third-party toolbox source code. Users should install the required dependencies separately and specify their local paths in `config_local.m`.

## Required Toolboxes

The main scripts require the following external MATLAB toolboxes:

| Dependency         | Purpose                                             |
| ------------------ | --------------------------------------------------- |
| AdaRGC / FgMDM     | FgMDM and MDM classifier implementation             |
| FieldTrip          | EEG filtering and signal preprocessing utilities    |
| Covariance toolbox | Riemannian geometry and covariance matrix utilities |

## 1. AdaRGC / FgMDM

This repository uses the FgMDM implementation from the AdaRGC project.

### Source

```text
Repository: https://github.com/neurosatya/AdaRGC
Related paper: https://hal.inria.fr/hal-01924646
```

### Purpose in this project

AdaRGC provides the `FgMDM.m` function used for Riemannian geometry-based EEG classification.

In this project, FgMDM is used for:

```text
- within-task decoding
- task-to-task transfer decoding
- left/right motor task classification
```

### Expected MATLAB function

After installation, MATLAB should be able to find:

```matlab
FgMDM
```

You can check this with:

```matlab
which FgMDM
```

### License note

AdaRGC is distributed under the AGPL-3.0 license at the time this README was written.

Users are responsible for checking and complying with the license terms of the version they use.

### Recommended installation

Clone or download AdaRGC outside this repository:

```text
/path/to/AdaRGC-master/
```

Then add the path in `config_local.m`:

```matlab
cfg.adargc_dir = '/path/to/AdaRGC-master';
```

or on Windows:

```matlab
cfg.adargc_dir = 'C:\path\to\AdaRGC-master';
```

The main scripts will add the dependency using:

```matlab
addpath(genpath(cfg.adargc_dir));
```

## 2. FieldTrip

FieldTrip is used for EEG signal preprocessing utilities.

### Source

```text
Website: https://www.fieldtriptoolbox.org/
GitHub: https://github.com/fieldtrip/fieldtrip
```

### Purpose in this project

This repository uses FieldTrip mainly for band-pass filtering:

```matlab
ft_preproc_bandpassfilter
```

The classification preprocessing uses a practical online-style pipeline:

```text
- 5–40 Hz band-pass filtering
- no ICA
- epoch extraction around cue onset
```

### Expected MATLAB function

After installation, MATLAB should be able to find:

```matlab
ft_preproc_bandpassfilter
```

You can check this with:

```matlab
which ft_preproc_bandpassfilter
```

### Recommended installation

Download FieldTrip outside this repository:

```text
/path/to/fieldtrip/
```

Then add the path in `config_local.m`:

```matlab
cfg.fieldtrip_dir = '/path/to/fieldtrip';
```

or on Windows:

```matlab
cfg.fieldtrip_dir = 'C:\path\to\fieldtrip';
```

The main scripts should initialize FieldTrip with:

```matlab
addpath(cfg.fieldtrip_dir);
ft_defaults;
```

## 3. Covariance Toolbox

The covariance toolbox provides MATLAB functions for covariance matrix manipulation and Riemannian geometry operations.

### Source

```text
Repository: https://github.com/alexandrebarachant/covariancetoolbox
```

### Purpose in this project

This toolbox may be required by the FgMDM/AdaRGC implementation for operations involving symmetric positive definite covariance matrices.

### Recommended installation

Clone or download the covariance toolbox outside this repository:

```text
/path/to/covariancetoolbox/
```

Then add the path in `config_local.m`:

```matlab
cfg.covtoolbox_dir = '/path/to/covariancetoolbox';
```

or on Windows:

```matlab
cfg.covtoolbox_dir = 'C:\path\to\covariancetoolbox';
```

The main scripts will add the dependency using:

```matlab
addpath(genpath(cfg.covtoolbox_dir));
```

## Suggested Local Folder Layout

The external dependencies can be stored anywhere on your local machine.

Example:

```text
toolboxes/
├── AdaRGC-master/
├── fieldtrip-20200115/
└── covariancetoolbox-master/
```

Then set the paths in `config_local.m`:

```matlab
cfg.adargc_dir = 'C:\path\to\toolboxes\AdaRGC-master';
cfg.fieldtrip_dir = 'C:\path\to\toolboxes\fieldtrip-20200115';
cfg.covtoolbox_dir = 'C:\path\to\toolboxes\covariancetoolbox-master';
```

## Dependency Check

After setting paths, run:

```matlab
check_dependencies
```

The script should verify that MATLAB can find the required functions:

```matlab
FgMDM
ft_preproc_bandpassfilter
```

If MATLAB cannot find a required function, check:

```matlab
which FgMDM
which ft_preproc_bandpassfilter
```

If the output is empty, the corresponding toolbox path has not been correctly added.

## Version Tracking

For reproducibility, record the dependency versions or commit hashes used in your analysis.

Recommended format:

```text
AdaRGC:
- Source: https://github.com/neurosatya/AdaRGC
- Commit hash: <write commit hash here>
- Download date: <YYYY-MM-DD>

FieldTrip:
- Source: https://github.com/fieldtrip/fieldtrip
- Version or date: <write version/date here>

Covariance toolbox:
- Source: https://github.com/alexandrebarachant/covariancetoolbox
- Commit hash: <write commit hash here>
- Download date: <YYYY-MM-DD>
```

## Why Third-party Code Is Not Included

This repository is intended to provide the MIMOME FgMDM transfer decoding pipeline, not to redistribute external toolboxes.

External code is not included by default because:

```text
1. Third-party toolboxes have their own licenses.
2. Users should cite the original toolbox authors.
3. Dependency versions should be controlled explicitly by the user.
4. Keeping external code separate reduces repository size and license ambiguity.
```

If you choose to vendor or copy third-party code into this repository, review each dependency license first and preserve all required copyright and license notices.

## Citation

If you use FgMDM/AdaRGC, cite the original method and implementation source as appropriate.

Suggested references:

```bibtex
@inproceedings{kumar2019adaptive,
  title = {Towards Adaptive Classification using Riemannian Geometry approaches in Brain-Computer Interfaces},
  author = {Kumar, Satyam and Yger, Florian and Lotte, Fabien},
  booktitle = {2019 7th International Winter Conference on Brain-Computer Interface (BCI)},
  pages = {1--6},
  year = {2019},
  doi = {10.1109/IWW-BCI.2019.8737349}
}
```

```bibtex
@inproceedings{barachant2010riemannian,
  title = {Riemannian geometry applied to BCI classification},
  author = {Barachant, Alexandre and Bonnet, Sylvain and Congedo, Marco and Jutten, Christian},
  booktitle = {International Conference on Latent Variable Analysis and Signal Separation},
  pages = {629--636},
  year = {2010},
  publisher = {Springer}
}
```

Also cite FieldTrip and the covariance toolbox if they are used in your analysis.

## License Disclaimer

This file is not legal advice.

Each user is responsible for checking the license terms of the exact third-party dependency versions used in their own environment.
