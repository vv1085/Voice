#!/usr/bin/env python
"""Build the model and trial files the VOICE web app serves.

    python build_artifacts.py                      # real BNCI2014-008 data (needs requirements-build.txt)
    python build_artifacts.py --source synthetic   # offline smoke-test data, clearly labelled
    python build_artifacts.py --check              # exit 0 if artifacts already exist

Pipeline (same as the notebook):
  1. MOABB P300 paradigm on BNCI2014-008: 0.5-10 Hz, epochs -0.2..0.8 s, 8 channels.
  2. Flash codes 3-14 rebuilt from each recording's "Flash stim" channel and checked
     against the Target/NonTarget annotations (35 trials x 120 flashes per subject).
  3. Features: microvolts, decimated 256 -> 64 Hz (8 x 65 = 520 values per epoch).
  4. One decoder per subject: shrinkage LDA (same as scikit-learn lsqr + 'auto') trained on the 7 other
     subjects plus this subject's first 10 trials. The other 25 trials are never seen
     in training; the web app spells only those.
  5. Each decoder is scored on its held-out trials and folded into weights + bias.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import voice_core as config  # noqa: E402
from voice_core import (  # noqa: E402
    LinearP300Model, SubjectTrials, TrialDecoder, artifacts_present, write_artifacts,
)

CHANNELS = ["Fz", "Cz", "Pz", "Oz", "P3", "P4", "PO7", "PO8"]


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# ---------------------------------------------------------------------------
# Data sources: both return {subject: (epochs_uv (4200, 8, 65) float32,
#                                      labels (4200,), flash_codes (4200,))}
# ---------------------------------------------------------------------------

def load_bnci(subjects: list[int], data_dir: str | None) -> dict[int, tuple]:
    if data_dir:
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        for key in ("MNE_DATA", "MNE_DATASETS_BNCI_PATH"):
            os.environ[key] = str(Path(data_dir).resolve())
    try:
        import mne
        from moabb.datasets import BNCI2014_008
        from moabb.paradigms import P300
    except ImportError as error:  # pragma: no cover
        sys.exit(f"{error}\nInstall build dependencies: pip install -r requirements-build.txt")

    mne.set_log_level("WARNING")
    dataset = BNCI2014_008()
    paradigm = P300(fmin=config.FILTER_LOW_HZ, fmax=config.FILTER_HIGH_HZ,
                    tmin=config.EPOCH_START_S, tmax=config.EPOCH_END_S)
    out = {}
    for s in subjects:
        log(f"Subject {s}: loading epochs (first run downloads the recording)...")
        X, y_raw, _ = paradigm.get_data(dataset=dataset, subjects=[s])
        labels = np.array([1 if str(v).strip().lower() == "target" else 0 for v in y_raw], dtype=np.int8)

        raw = next(iter(next(iter(dataset.get_data(subjects=[s])[s].values())).values()))
        flash = raw.get_data(picks=["Flash stim"])[0]
        previous = np.r_[0, flash[:-1]]
        codes = flash[np.flatnonzero((flash != 0) & (previous == 0))].astype(int)

        annotation_labels = np.array(
            [1 if str(d).lower() == "target" else 0 for d in raw.annotations.description], dtype=np.int8
        )
        expected = config.TRIALS_PER_SUBJECT * config.EVENTS_PER_TRIAL
        if not (len(X) == len(codes) == len(annotation_labels) == expected):
            sys.exit(f"Subject {s}: expected {expected} epochs/flashes/annotations, got "
                     f"{len(X)}/{len(codes)}/{len(annotation_labels)}.")
        if not np.array_equal(annotation_labels, labels):
            sys.exit(f"Subject {s}: epoch order does not match the annotation order.")

        out[s] = (preprocess(np.asarray(X)), labels, codes.astype(np.int8))
        del X, raw, flash
    return out


def preprocess(X_volts: np.ndarray) -> np.ndarray:
    x = X_volts[:, :, :: config.DECIMATION] * config.SCALE_TO_MICROVOLTS
    return np.clip(x, -60000, 60000).astype(np.float32)  # float16-safe range


def load_synthetic(subjects: list[int], seed: int = 7) -> dict[int, tuple]:
    """Simulated P300 data with the real recording's structure. For tests only."""
    rng = np.random.default_rng(seed)
    n_times = int(round((config.EPOCH_END_S - config.EPOCH_START_S) * config.SAMPLING_RATE)) + 1
    n_times = len(range(0, n_times, config.DECIMATION))
    t = np.linspace(config.EPOCH_START_S, config.EPOCH_END_S, n_times)
    erp = 2.5 * np.exp(-((t - 0.35) ** 2) / (2 * 0.06 ** 2))
    gain = np.array([0.6, 1.0, 1.2, 0.7, 0.9, 0.9, 0.8, 0.8])[:, None]
    out = {}
    for s in subjects:
        snr = rng.uniform(0.35, 0.8)
        codes, labels = [], []
        for _ in range(config.TRIALS_PER_SUBJECT):
            target = {int(rng.choice(config.ROW_CODES)), int(rng.choice(config.COLUMN_CODES))}
            for _ in range(config.REPETITIONS_PER_TRIAL):
                order = rng.permutation(config.FLASH_CODES)
                codes.extend(order)
                labels.extend(int(c in target) for c in order)
        codes, labels = np.array(codes, dtype=np.int8), np.array(labels, dtype=np.int8)
        x = rng.normal(0, 4.0, size=(len(codes), len(CHANNELS), n_times))
        x[labels == 1] += 0.6 * snr * gain * erp
        out[s] = (x.astype(np.float32), labels, codes)
    return out


# ---------------------------------------------------------------------------
# Validation, training, evaluation
# ---------------------------------------------------------------------------

def validate_structure(s: int, labels: np.ndarray, codes: np.ndarray) -> float:
    """Check 35 trials x 12 codes x 10 flashes; return share of trials whose two
    target codes are one row code and one column code."""
    n = config.EVENTS_PER_TRIAL
    mapped = 0
    for trial in range(config.TRIALS_PER_SUBJECT):
        c, y = codes[trial * n:(trial + 1) * n], labels[trial * n:(trial + 1) * n]
        counts = {k: int((c == k).sum()) for k in config.FLASH_CODES}
        if set(np.unique(c)) != set(config.FLASH_CODES) or set(counts.values()) != {config.REPETITIONS_PER_TRIAL}:
            sys.exit(f"Subject {s}, trial {trial}: unexpected flash structure {counts}.")
        targets = set(c[y == 1].tolist())
        if len(targets) != 2:
            sys.exit(f"Subject {s}, trial {trial}: expected 2 target codes, found {sorted(targets)}.")
        rows = targets & set(config.ROW_CODES)
        mapped += len(rows) == 1
    return mapped / config.TRIALS_PER_SUBJECT


def _class_shrunk_covariance(chunks: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, int]:
    """Ledoit-Wolf shrunk covariance of one class, accumulated chunk by chunk.

    Same maths as scikit-learn's LDA(solver="lsqr", shrinkage="auto"): standardize
    the class, apply ledoit_wolf, scale back. Only 520 x 520 sums stay in memory.
    """
    n = sum(len(c) for c in chunks)
    total = sum(c.sum(axis=0, dtype=np.float64) for c in chunks)
    mean = total / n
    var = sum(((c.astype(np.float64) - mean) ** 2).sum(axis=0) for c in chunks) / n
    scale = np.sqrt(var)
    scale[scale == 0] = 1.0

    p = len(mean)
    gram = np.zeros((p, p))
    gram_sq = np.zeros((p, p))
    for c in chunks:
        z = (c.astype(np.float64) - mean) / scale
        gram += z.T @ z
        z2 = z * z
        gram_sq += z2.T @ z2

    emp_cov = gram / n
    trace = np.diag(emp_cov).sum()
    mu = trace / p
    delta_ = np.sum(gram ** 2) / n ** 2
    beta = (np.sum(gram_sq) / n - delta_) / (p * n)
    delta = (delta_ - 2.0 * mu * trace + p * mu ** 2) / p
    beta = min(beta, delta)
    shrinkage = 0.0 if beta == 0 else beta / delta
    shrunk = (1.0 - shrinkage) * emp_cov
    shrunk.flat[:: p + 1] += shrinkage * mu
    return mean, scale[:, None] * shrunk * scale[None, :], n


def fit_shrinkage_lda(chunks: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, float]:
    """Binary shrinkage LDA from (features, labels) chunks; returns weights, bias
    such that p(target) = sigmoid(features @ weights + bias)."""
    by_class = {k: [x[y == k] for x, y in chunks if np.any(y == k)] for k in (0, 1)}
    stats = {k: _class_shrunk_covariance(by_class[k]) for k in (0, 1)}
    n_total = stats[0][2] + stats[1][2]
    priors = {k: stats[k][2] / n_total for k in (0, 1)}
    covariance = priors[0] * stats[0][1] + priors[1] * stats[1][1]
    means = np.vstack([stats[0][0], stats[1][0]])
    coef = np.linalg.lstsq(covariance, means.T, rcond=None)[0].T
    intercept = -0.5 * np.einsum("ij,ij->i", means, coef) + np.log([priors[0], priors[1]])
    return coef[1] - coef[0], float(intercept[1] - intercept[0])


def check_against_sklearn(X: np.ndarray, y: np.ndarray) -> None:
    """The streaming LDA must reproduce scikit-learn on a sample."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    feats = X.reshape(len(X), -1).astype(np.float64)
    weights, bias = fit_shrinkage_lda([(feats[: len(feats) // 2], y[: len(y) // 2]),
                                       (feats[len(feats) // 2:], y[len(y) // 2:])])
    reference = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(feats, y)
    ours = LinearP300Model(weights, bias, X.shape[1], X.shape[2]).predict_proba(X)
    if not np.allclose(ours, reference.predict_proba(feats)[:, 1], atol=1e-6):
        raise RuntimeError("Streaming LDA does not match scikit-learn.")
    log("Streaming LDA matches scikit-learn on a 3,000-epoch sample.")


def evaluate(model: LinearP300Model, trials: SubjectTrials) -> dict:
    adaptive, fixed, reps = [], [], []
    always_ten = TrialDecoder(model)
    always_ten.stopping.minimum_repetitions = config.REPETITIONS_PER_TRIAL
    for i in range(len(trials)):
        intended = trials.intended_character(i)
        if intended is None:
            continue
        repetitions = trials.repetitions(i)
        a = TrialDecoder(model).decode(repetitions)["state"]
        f = always_ten.decode(repetitions)["state"]
        adaptive.append(a["character"] == intended)
        fixed.append(f["character"] == intended)
        reps.append(a["repetitions"])
    return {
        "scored_trials": len(adaptive),
        "adaptive_accuracy": round(float(np.mean(adaptive)), 4) if adaptive else None,
        "fixed_10_repetition_accuracy": round(float(np.mean(fixed)), 4) if fixed else None,
        "mean_repetitions": round(float(np.mean(reps)), 2) if reps else None,
    }


def build(source: str, out_dir: Path, subjects: list[int], calibration_trials: int,
          data_dir: str | None) -> dict:
    started = time.time()
    data = load_synthetic(subjects) if source == "synthetic" else load_bnci(subjects, data_dir)
    n = config.EVENTS_PER_TRIAL
    split = calibration_trials * n

    mapping = {s: validate_structure(s, d[1], d[2]) for s, d in data.items()}
    if min(mapping.values()) < 0.95:
        sys.exit("Target codes do not split into one row code (3-8) and one column code (9-14): "
                 f"{mapping}. Check ROW_CODES / COLUMN_CODES in voice/config.py.")

    first = data[subjects[0]]
    check_against_sklearn(first[0][:3000], first[1][:3000])

    models, trial_files, evaluation = {}, {}, {}
    n_channels, n_times = next(iter(data.values()))[0].shape[1:]
    for s in subjects:
        others = [o for o in subjects if o != s]
        chunks = [(data[o][0].reshape(len(data[o][0]), -1), data[o][1]) for o in others]
        chunks.append((data[s][0][:split].reshape(split, -1), data[s][1][:split]))
        log(f"Subject {s}: training on {sum(len(c[1]) for c in chunks)} epochs "
            f"({len(others)} other subjects + {calibration_trials} calibration trials)")
        weights, bias = fit_shrinkage_lda(chunks)
        models[s] = (weights, bias)

        X_s, y_s, c_s = data[s]
        n_heldout = config.TRIALS_PER_SUBJECT - calibration_trials
        trial_files[s] = {
            "trial_numbers": np.arange(calibration_trials, config.TRIALS_PER_SUBJECT),
            "epochs": X_s[split:].reshape(n_heldout, n, n_channels, n_times),
            "flash_codes": c_s[split:].reshape(n_heldout, n),
            "labels": y_s[split:].reshape(n_heldout, n),
        }
        held_out = SubjectTrials(s, trial_files[s]["trial_numbers"],
                                 trial_files[s]["epochs"].astype(np.float16),  # evaluate what is served
                                 trial_files[s]["flash_codes"], trial_files[s]["labels"])
        evaluation[s] = evaluate(LinearP300Model(weights, bias, n_channels, n_times), held_out)
        log(f"Subject {s}: held-out accuracy {evaluation[s]['adaptive_accuracy']:.0%} "
            f"(adaptive, {evaluation[s]['mean_repetitions']} reps) | "
            f"{evaluation[s]['fixed_10_repetition_accuracy']:.0%} (10 reps)")

    manifest = {
        "source": "BNCI2014-008" if source == "bnci" else "synthetic",
        "synthetic": source == "synthetic",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "channels": CHANNELS,
        "n_channels": int(n_channels),
        "n_times": int(n_times),
        "units": "microvolts",
        "preprocessing": {
            "bandpass_hz": [config.FILTER_LOW_HZ, config.FILTER_HIGH_HZ],
            "epoch_s": [config.EPOCH_START_S, config.EPOCH_END_S],
            "sampling_rate_hz": config.SAMPLING_RATE / config.DECIMATION,
        },
        "classifier": "LDA with Ledoit-Wolf shrinkage (scikit-learn lsqr/auto equivalent), weights + bias",
        "calibration_trials": calibration_trials,
        "stopping": config.STOPPING,
        "subjects": {
            str(s): {"held_out_trials": int(len(trial_files[s]["trial_numbers"])),
                     "row_column_mapping_ok": mapping[s], **evaluation[s]}
            for s in subjects
        },
    }
    write_artifacts(out_dir, manifest, models, trial_files)
    log(f"Artifacts written to {out_dir} in {time.time() - started:.0f}s")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["bnci", "synthetic"],
                        default=os.environ.get("VOICE_BUILD_SOURCE", "bnci"))
    parser.add_argument("--out", type=Path, default=config.ARTIFACT_DIR)
    parser.add_argument("--data-dir", default=os.environ.get("VOICE_MNE_DATA"),
                        help="Where MOABB stores downloads (default: ~/mne_data)")
    parser.add_argument("--subjects", type=int, nargs="+", default=config.SUBJECTS)
    parser.add_argument("--calibration-trials", type=int, default=config.CALIBRATION_TRIALS)
    parser.add_argument("--skip-if-present", action="store_true")
    parser.add_argument("--check", action="store_true", help="Only report whether artifacts exist")
    args = parser.parse_args()

    if args.check:
        present = artifacts_present(args.out)
        print(f"Artifacts {'found' if present else 'missing'} in {args.out}")
        sys.exit(0 if present else 1)
    if args.skip_if_present and artifacts_present(args.out):
        log(f"Artifacts already present in {args.out}; skipping build.")
        return
    if not 1 <= args.calibration_trials < config.TRIALS_PER_SUBJECT:
        sys.exit("--calibration-trials must be between 1 and 34.")
    if len(args.subjects) < 2:
        sys.exit("Need at least 2 subjects (each decoder also trains on the others).")
    build(args.source, args.out, sorted(set(args.subjects)), args.calibration_trials, args.data_dir)


if __name__ == "__main__":
    main()
