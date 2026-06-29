"""Sleep-EDF loading and subject grouping."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import warnings

import numpy as np
from tqdm import tqdm

from preprocessing import compute_psd_dataset, zscore_with_scaler_per_channel


STAGE_MAPPING = {
    "Sleep stage W": 1,
    "Sleep stage 1": 2,
    "Sleep stage 2": 3,
    "Sleep stage 3": 4,
    "Sleep stage 4": 4,
    "Sleep stage R": 5,
}

EPOCH_EVENT_ID = {
    "Sleep stage W": 1,
    "Sleep stage 1": 2,
    "Sleep stage 2": 3,
    "Sleep stage 3/4": 4,
    "Sleep stage R": 5,
}


def load_sleep_edf_raw_psd(
    base_path: str | Path,
    channel: str = "EEG Fpz-Cz",
    sampling_rate: int = 100,
    wake_epochs: int = 60,
    records_file: str = "RECORDS-v1",
    cassette_only: bool = True,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[str]]:
    """Load Sleep-EDF records and return raw, PSD, and labels grouped by subject."""
    try:
        import mne
    except ImportError as exc:
        raise ImportError("Install mne before loading EDF files: pip install mne") from exc

    base_path = Path(base_path)
    records_path = base_path / records_file
    cassette_dir = base_path / "sleep-cassette"
    telemetry_dir = base_path / "sleep-telemetry"
    hypnogram_files = list(cassette_dir.glob("*-Hypnogram.edf")) + list(telemetry_dir.glob("*-Hypnogram.edf"))

    with records_path.open("r", encoding="utf-8") as file:
        relative_paths = [line.strip() for line in file if line.strip()]

    subject_raw: dict[str, list[np.ndarray]] = defaultdict(list)
    subject_psd: dict[str, list[np.ndarray]] = defaultdict(list)
    subject_labels: dict[str, list[np.ndarray]] = defaultdict(list)

    for rel_path in tqdm(relative_paths, desc="Loading Sleep-EDF records"):
        filename = Path(rel_path).name
        if cassette_only and not filename.startswith("SC"):
            continue

        psg_path = cassette_dir / filename if filename.startswith("SC") else telemetry_dir / filename
        record_id = psg_path.stem[:7]
        subject_id = record_id[3:5]

        matching_hypnograms = [path for path in hypnogram_files if path.name.startswith(record_id)]
        if not matching_hypnograms:
            print(f"No hypnogram found for {psg_path.name}")
            continue
        hypnogram_path = matching_hypnograms[0]

        print(f"\nLoading PSG: {psg_path.name}")
        print(f"Matching hypnogram: {hypnogram_path.name}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
            raw.pick([channel])
            annotations = mne.read_annotations(hypnogram_path)
            raw.set_annotations(annotations)

        events, _ = mne.events_from_annotations(
            raw,
            event_id=STAGE_MAPPING,
            chunk_duration=30.0,
            verbose=False,
        )

        tmax = 30.0 - 1.0 / raw.info["sfreq"]
        epochs = mne.Epochs(
            raw=raw,
            events=events,
            event_id=EPOCH_EVENT_ID,
            tmin=0.0,
            tmax=tmax,
            baseline=None,
            preload=True,
            verbose=False,
            on_missing="ignore",
        )

        data = epochs.get_data(copy=True)
        labels = epochs.events[:, 2] - 1

        sleep_indices = np.where(labels != 0)[0]
        if len(sleep_indices) == 0:
            print(f"No sleep epochs found in {filename}, skipping.")
            continue

        first_sleep_idx = sleep_indices[0]
        last_sleep_idx = sleep_indices[-1]
        wake_before_start = max(0, first_sleep_idx - wake_epochs)
        wake_after_end = min(len(labels), last_sleep_idx + 1 + wake_epochs)

        keep_indices = np.concatenate(
            [
                np.arange(wake_before_start, first_sleep_idx),
                np.arange(first_sleep_idx, last_sleep_idx + 1),
                np.arange(last_sleep_idx + 1, wake_after_end),
            ]
        )

        data = zscore_with_scaler_per_channel(data[keep_indices])
        labels = labels[keep_indices]

        print(f"Extracted {data.shape}")
        subject_raw[subject_id].append(data)
        subject_psd[subject_id].append(compute_psd_dataset(data, fs=sampling_rate))
        subject_labels[subject_id].append(labels)

    subject_ids = sorted(subject_raw)
    raw_by_subject = [np.vstack(subject_raw[subject_id]) for subject_id in subject_ids]
    psd_by_subject = [np.vstack(subject_psd[subject_id]) for subject_id in subject_ids]
    y_by_subject = [np.hstack(subject_labels[subject_id]) for subject_id in subject_ids]

    print(f"\nFinished loading. Subjects loaded: {len(raw_by_subject)}")
    print(f"Total epochs: {sum(x.shape[0] for x in raw_by_subject)}")
    return raw_by_subject, psd_by_subject, y_by_subject, subject_ids
