# Excluded Artifacts

The following large artifact folders existed in the original workspace at `test_folder2/test_folder2/` and were not copied into the GitHub-ready project.

| Original path | Approx. size | Notes |
| --- | ---: | --- |
| `ddpm/` | 11 GB | DDPM training outputs/checkpoints |
| `generated_data32/` | 4.6 GB | Generated 32-channel data |
| `class64_cp/` | 1.9 GB | 64-channel classifier checkpoints |
| `S14_class64_cp/` | 1.3 GB | Subject/channel classifier checkpoints |
| `s01_class32_cp/` | 961 MB | 32-channel classifier checkpoints |
| `s03_class32_cp/` | 961 MB | 32-channel classifier checkpoints |
| `s05_class32_cp/` | 961 MB | 32-channel classifier checkpoints |
| `s07_class32_cp/` | 961 MB | 32-channel classifier checkpoints |
| `s10_class32_cp/` | 961 MB | 32-channel classifier checkpoints |
| `s11_class32_cp/` | 961 MB | 32-channel classifier checkpoints |
| `s14_class32_cp/` | 961 MB | 32-channel classifier checkpoints |
| `s18_class32_cp/` | 961 MB | 32-channel classifier checkpoints |
| `s20_class32_cp/` | 961 MB | 32-channel classifier checkpoints |
| `s23_class32_cp/` | 961 MB | 32-channel classifier checkpoints |
| `s25_class32_cp/` | 961 MB | 32-channel classifier checkpoints |
| `s28_class32_cp/` | 961 MB | 32-channel classifier checkpoints |
| `s30_class32_cp/` | 961 MB | 32-channel classifier checkpoints |
| `s31_class32_cp/` | 961 MB | 32-channel classifier checkpoints |
| `s34_class32_cp/` | 961 MB | 32-channel classifier checkpoints |
| `s39_class32_cp/` | 961 MB | 32-channel classifier checkpoints |
| `pytorch_data/` | 348 MB | PyTorch data/examples |
| `notcropped_class32_cp/` | 241 MB | Not-cropped classifier checkpoints |
| `s03_vae_cp/` | 532 MB | VAE checkpoints |

Smaller checkpoint folders and `.pickle`/`.pt` files were also excluded by `.gitignore`.
