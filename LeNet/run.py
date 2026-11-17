#!/usr/bin/env python3
"""All-in-one launcher: Dataset -> LeNet-5 training -> autonomous camera/drive."""

import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run import run_child

ROOT = Path(__file__).resolve().parents[1]
LENET_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = [
    ROOT / "dataset" / "1117_basic_2000",
    ROOT / "dataset" / "1117_midtwobox_500",
]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", nargs="+", type=Path, default=DEFAULT_DATA,
                   help="학습할 NPZ 파일 또는 폴더들 (기본: 1117_basic_2000, 1117_midtwobox_500)")
    p.add_argument("--model", type=Path, help="학습 없이 기존 LeNet best.pt 실행")
    p.add_argument("--run-dir", type=Path, help="학습 결과 저장 폴더")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--drive", action="store_true", help="학습 완료 후 실제 모터 구동")
    mode.add_argument("--train-only", action="store_true", help="학습과 저장만 수행")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--speed", type=int, default=28)
    p.add_argument("--turn", type=int, default=25)
    p.add_argument("--min-confidence", type=float, default=0.0)
    p.add_argument("--duration", type=float, default=0)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.model and (args.train_only or args.run_dir):
        p.error("--train-only/--run-dir는 --model 없이 사용하세요.")
    if args.drive and not sys.stdin.isatty():
        p.error("실제 주행은 정지 키를 사용할 수 있는 터미널에서 실행하세요.")

    if args.model:
        model = args.model.expanduser().resolve()
    else:
        run_dir = (args.run_dir or LENET_DIR / "runs" /
                   ("lenet_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))).resolve()
        train_cmd = [
            sys.executable, str(LENET_DIR / "train.py"),
            "--data", *[str(d.expanduser().resolve()) for d in args.data],
            "--output-dir", str(run_dir),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--lr", str(args.lr),
            "--seed", str(args.seed),
            "--threads", str(args.threads),
            "--device", str(args.device),
        ]
        print(f"LeNet-5 학습을 시작합니다. 결과 폴더: {run_dir}", flush=True)
        run_child(train_cmd)
        model = run_dir / "best.pt"

    if not model.is_file():
        raise RuntimeError(f"실행할 모델이 없습니다: {model}")
    if args.train_only:
        return

    drive_cmd = [
        sys.executable, str(LENET_DIR / "drive.py"),
        "--model", str(model),
        "--camera", str(args.camera),
        "--speed", str(args.speed),
        "--turn", str(args.turn),
        "--min-confidence", str(args.min_confidence),
        "--threads", str(args.threads),
        "--duration", str(args.duration),
    ]
    if args.drive:
        drive_cmd.append("--drive")
    print("실제 주행을 시작합니다. Space/X/Q/ESC/Ctrl+C: 종료" if args.drive
          else "카메라 예측을 시작합니다 (모터 미사용).", flush=True)
    run_child(drive_cmd)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"실행 중단: {exc}", file=sys.stderr)
        sys.exit(1)
