#!/usr/bin/env python3
"""Train LeNet-5 behavior cloning policy on Pirrari NPZ datasets."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import random
import shutil

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

try:
    from .dataset import DrivingDataset, load_data, split_indices
    from .model import ARCH_NAME, ACTION_NAMES, STATE_SHAPE, LeNet5
except ImportError:
    from dataset import DrivingDataset, load_data, split_indices
    from model import ARCH_NAME, ACTION_NAMES, STATE_SHAPE, LeNet5

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = [
    REPO_ROOT / "dataset" / "1117_basic_2000",
    REPO_ROOT / "dataset" / "1117_midtwobox_500",
]


def evaluate(model, loader, device):
    """Evaluate loss, accuracy, balanced accuracy, and confusion matrix."""
    model.eval()
    loss_sum = 0.0
    confusion = torch.zeros(3, 3, dtype=torch.int64)
    with torch.inference_mode():
        for states, labels in loader:
            states, labels = states.to(device), labels.to(device)
            logits = model(states)
            loss_sum += nn.functional.cross_entropy(logits, labels, reduction="sum").item()
            bins = labels.cpu() * 3 + logits.argmax(1).cpu()
            confusion += torch.bincount(bins, minlength=9).reshape(3, 3)
    support = confusion.sum(1)
    recall = confusion.diag().float() / support.clamp_min(1)
    return {
        "loss": loss_sum / int(support.sum()),
        "accuracy": float(confusion.diag().sum() / support.sum()),
        "balanced_accuracy": float(recall[support > 0].mean()),
        "recall": [float(recall[i]) if support[i] else None for i in range(3)],
        "confusion_matrix": confusion.tolist(),
    }


def parser():
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", nargs="+", type=Path, default=DEFAULT_DATA,
                   help="학습할 .npz 파일 또는 폴더들 (기본: 1117_basic_2000, 1117_midtwobox_500)")
    p.add_argument("--output-dir", type=Path,
                   default=root / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--no-class-weights", action="store_true")
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if min(args.epochs, args.batch_size, args.patience, args.threads) < 1:
        p.error("epochs/batch-size/patience/threads는 1 이상이어야 합니다.")
    if not np.isfinite(args.lr) or args.lr <= 0 or not 0 < args.val_fraction < 1:
        p.error("lr은 양수, val-fraction은 0과 1 사이여야 합니다.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA를 사용할 수 없습니다. --device cpu를 사용하세요.")

    try:
        states, actions, sessions, sources, rows = load_data(args.data)
        train, val, split_method = split_indices(actions, sessions, args.val_fraction, args.seed)
    except (ValueError, OSError, KeyError) as exc:
        p.error(str(exc))

    counts = np.bincount(actions[train], minlength=3)
    val_counts = np.bincount(actions[val], minlength=3)
    print(f"arch={ARCH_NAME}; device={device}; split={split_method}")
    print(f"train={len(train)} {counts.tolist()}; val={len(val)} {val_counts.tolist()}")
    print(f"제외된 stop(3)={int((actions == 3).sum())}; 액션 순서={ACTION_NAMES}")

    train_loader = DataLoader(
        DrivingDataset(states, actions, train),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(DrivingDataset(states, actions, val), batch_size=args.batch_size)

    model = LeNet5().to(device)
    weights = None if args.no_class_weights else torch.tensor(
        len(train) / (3 * counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = {
        key: [str(v) for v in value] if isinstance(value, list) else (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
    }
    config.update(
        arch=ARCH_NAME, device=device, split_method=split_method,
        train_counts=counts.tolist(), val_counts=val_counts.tolist(),
        action_names=ACTION_NAMES, state_shape=list(STATE_SHAPE),
        normalization="uint8 RGB / 255.0", torch_version=str(torch.__version__),
    )
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    manifest = {
        name: [{"file": str(sources[i]), "row": int(rows[i]), "session": str(sessions[i])} for i in indices]
        for name, indices in [("train", train), ("val", val)]
    }
    (args.output_dir / "split.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    best_loss, stale, history = float("inf"), 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, weight_sum = 0.0, 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            denominator = float(weights[y].sum()) if weights is not None else len(y)
            loss_sum += loss.item() * denominator
            weight_sum += denominator
        metrics = evaluate(model, val_loader, device)
        record = {"epoch": epoch, "train_loss": loss_sum / weight_sum, "val": metrics}
        history.append(record)
        checkpoint = {
            "format_version": 1,
            "arch": ARCH_NAME,
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "action_names": ACTION_NAMES,
            "state_shape": list(STATE_SHAPE),
            "config": config,
            "metrics": record,
        }
        torch.save(checkpoint, args.output_dir / "last.pt")
        if metrics["loss"] < best_loss:
            best_loss, stale = metrics["loss"], 0
            torch.save(checkpoint, args.output_dir / "best.pt")
        else:
            stale += 1
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(
            f"epoch={epoch:03d} train_loss={record['train_loss']:.4f} "
            f"val_loss={metrics['loss']:.4f} accuracy={metrics['accuracy']:.3f} "
            f"balanced_accuracy={metrics['balanced_accuracy']:.3f}",
            flush=True,
        )
        if stale >= args.patience:
            print("검증 손실 개선이 없어 조기 종료합니다.")
            break

    latest_dir = Path(__file__).resolve().parent / "models"
    latest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.output_dir / "best.pt", latest_dir / "best.pt")
    print(f"학습 완료: {args.output_dir / 'best.pt'}")
    print(f"최신 모델 복사: {latest_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
