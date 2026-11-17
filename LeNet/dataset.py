"""Read Pirrari NPZ files and split train/val without leaking 5-frame windows."""

from pathlib import Path
import re

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from .model import STATE_SHAPE
except ImportError:
    from model import STATE_SHAPE


def read_npz(path):
    """Validate and load (states, actions) from one collector NPZ file."""
    with np.load(path, allow_pickle=False) as data:
        if not {"states", "actions"}.issubset(data.files):
            raise ValueError(f"{path}: states/actions 키가 필요합니다.")
        states, actions = data["states"], data["actions"]
    if states.ndim != 5 or states.shape[1:] != STATE_SHAPE:
        raise ValueError(f"{path}: states는 (N, 5, 3, 32, 32)여야 합니다: {states.shape}")
    if states.dtype != np.uint8:
        raise ValueError(f"{path}: states는 uint8이어야 합니다: {states.dtype}")
    if actions.shape != (len(states),) or actions.dtype.kind not in "iu":
        raise ValueError(f"{path}: actions는 길이 N인 정수 배열이어야 합니다.")
    if not np.isin(actions, [0, 1, 2, 3]).all():
        raise ValueError(f"{path}: 허용 액션은 0, 1, 2, 3입니다.")
    return states, actions.astype(np.int64)


def extract_session(path):
    """Group multi-part NPZ chunks from the same recording into one session."""
    stem = Path(path).stem
    match = re.fullmatch(r"(\d{8}_\d{6})_\d+(?:_\d+)?", stem)
    if match:
        return match.group(1)
    # Handle custom names like 1117_basic_500_001 or 1117_midtwobox_500_001.
    match = re.fullmatch(r"(.+)_\d+", stem)
    return match.group(1) if match else stem


def collect_npz_paths(entries):
    """Resolve a list of NPZ files and/or directories into sorted unique paths."""
    if isinstance(entries, (str, Path)):
        entries = [entries]
    paths = {}
    for entry in entries:
        entry = Path(entry).expanduser().resolve()
        if entry.is_dir():
            candidates = sorted(entry.glob("*.npz"))
            if not candidates:
                candidates = sorted(entry.rglob("*.npz"))
        elif entry.is_file() and entry.suffix.lower() == ".npz":
            candidates = [entry]
        else:
            raise ValueError(f"NPZ 파일 또는 폴더를 찾을 수 없습니다: {entry}")
        if not candidates:
            raise ValueError(f"NPZ 파일이 없는 폴더입니다: {entry}")
        for path in candidates:
            paths[str(path)] = path
    return [paths[k] for k in sorted(paths)]


def load_data(entries):
    """Load and concatenate NPZ datasets from one or more paths/directories."""
    paths = collect_npz_paths(entries)
    states, actions, sessions, sources, rows = [], [], [], [], []
    for path in paths:
        x, y = read_npz(path)
        session = extract_session(path)
        states.append(x)
        actions.append(y)
        sessions.extend([session] * len(y))
        sources.extend([path.name] * len(y))
        rows.extend(range(len(y)))
    if not sum(map(len, actions)):
        raise ValueError("데이터셋이 비어 있습니다.")
    return (np.concatenate(states), np.concatenate(actions), np.array(sessions),
            np.array(sources), np.array(rows))


def split_indices(actions, sessions, val_fraction=0.2, seed=42):
    """Split indices into train/val, keeping a 4-sample gap on temporal splits."""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction은 0과 1 사이여야 합니다.")
    valid = np.flatnonzero(actions < 3)  # Exclude stop=3 if present.
    groups = np.unique(sessions[valid])
    if len(groups) >= 3:
        groups = np.random.default_rng(seed).permutation(groups)
        count = min(len(groups) - 1, max(1, int(round(len(groups) * val_fraction))))
        is_val = np.isin(sessions[valid], groups[:count])
        train, val = valid[~is_val], valid[is_val]
        method = "session_holdout"
    elif len(groups) >= 1:
        # Per-session temporal holdout with 4-frame gap so all sessions contribute to both splits.
        train_parts, val_parts = [], []
        gap = STATE_SHAPE[0] - 1
        for group in groups:
            indices = np.flatnonzero(sessions == group)
            cut = int(len(indices) * (1 - val_fraction))
            tr = indices[:max(0, cut - gap)]
            va = indices[cut:]
            train_parts.append(tr[actions[tr] < 3])
            val_parts.append(va[actions[va] < 3])
        train = np.concatenate(train_parts)
        val = np.concatenate(val_parts)
        method = "per_session_temporal_holdout_with_4_sample_gap"
    else:
        raise ValueError("학습할 주행 액션(0/1/2)이 없습니다.")
    if not len(train) or not len(val):
        raise ValueError("학습/검증 분리에 필요한 데이터가 부족합니다.")
    missing = np.flatnonzero(np.bincount(actions[train], minlength=3) == 0)
    if len(missing):
        raise ValueError(f"학습 데이터에 액션 {missing.tolist()}이 없습니다.")
    return train, val, method


class DrivingDataset(Dataset):
    """PyTorch Dataset returning float32 normalized [0, 1] states and action labels."""

    def __init__(self, states, actions, indices):
        self.states, self.actions, self.indices = states, actions, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row = self.indices[index]
        return (torch.from_numpy(self.states[row]).float().div_(255.0),
                int(self.actions[row]))
