#!/usr/bin/env python3
"""Camera -> LeNet-5 policy -> Raspbot. Motors enabled only with --drive."""

import argparse
from collections import deque
from pathlib import Path
import signal
import sys
import threading
import time

import cv2
import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autonomous.drive import KeyboardStop, choose_action, prepare_frame
from autonomous.hardware import load_raspbot_class
from autonomous.safety import MotorGuard
from autonomous.ssh_watch import SSHConnection, monitor_connection
from LeNet.predict import LeNetPolicy


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path,
                   default=Path(__file__).resolve().parent / "models" / "best.pt")
    p.add_argument("--drive", action="store_true", help="실제 모터 구동 (생략하면 카메라 예측만)")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--speed", type=int, default=28)
    p.add_argument("--turn", type=int, default=25)
    p.add_argument("--min-confidence", type=float, default=0.0,
                   help="기본 0: 확률 기반 정지 없이 가장 높은 방향 선택")
    p.add_argument("--timeout", type=float, default=0.5, help="카메라/추론 명령 유효 시간(초)")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--duration", type=float, default=0, help="자동 종료 시간(초), 0은 수동 종료")
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if not args.model.is_file():
        p.error(f"모델 파일이 없습니다: {args.model}. 먼저 LeNet/train.py로 학습하거나 --model을 지정하세요.")
    if not (1 <= args.speed <= 100 and 1 <= args.turn <= 100):
        p.error("speed/turn은 1~100 범위여야 합니다.")
    if not 0 <= args.min_confidence <= 1 or not 0.1 <= args.timeout <= 2:
        p.error("min-confidence는 0~1, timeout은 0.1~2초 범위여야 합니다.")
    if args.threads < 1 or not np.isfinite(args.duration) or args.duration < 0:
        p.error("threads는 양수, duration은 0 이상의 유한한 수여야 합니다.")
    if args.drive and not sys.stdin.isatty():
        p.error("실제 주행은 정지 키를 사용할 수 있는 터미널에서 실행하세요.")

    torch.set_num_threads(args.threads)
    policy = LeNetPolicy(args.model)
    policy.predict(np.zeros((5, 3, 32, 32), dtype=np.uint8))

    stop_event = threading.Event()
    previous_handlers = {}
    guard = None
    cap = None
    peer_thread = None
    try:
        for name in ("SIGINT", "SIGTERM", "SIGHUP"):
            sig = getattr(signal, name, None)
            if sig is not None:
                previous_handlers[sig] = signal.signal(sig, lambda *_: stop_event.set())
        connection = SSHConnection.from_environment() if args.drive else None
        if connection:
            reason = connection.failure_reason()
            if reason:
                raise RuntimeError(reason)
        bot = load_raspbot_class()() if args.drive else None
        guard = MotorGuard(bot, stop_event, args.speed, args.turn, args.timeout)
        if connection:
            peer_thread = threading.Thread(target=monitor_connection, args=(connection, guard), daemon=True)
            peer_thread.start()
        with KeyboardStop(guard):
            cap = cv2.VideoCapture(args.camera)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                raise RuntimeError(f"카메라 {args.camera}를 열 수 없습니다.")
            print(f"[LeNet5] {'실제 주행' if args.drive else '예측만 (모터 미사용)'} | {args.model}", flush=True)
            print("Space / X / Q / ESC / Ctrl+C: 정지 후 종료", flush=True)
            frames = deque(maxlen=5)
            started = time.monotonic()
            while not stop_event.is_set():
                tick = time.monotonic()
                if args.duration and tick - started >= args.duration:
                    guard.trip("설정 시간 종료")
                    break
                ok, frame = cap.read()
                if not ok or frame is None:
                    guard.trip("카메라 프레임 수신 실패")
                    break
                frames.append(prepare_frame(frame))
                if len(frames) < 5:
                    if not guard.command(None, tick):
                        break
                else:
                    prediction = policy.predict(np.stack(frames))
                    action = choose_action(prediction, args.min_confidence)
                    if not guard.command(action, tick):
                        break
                    label = prediction["name"] if action is not None else "STOP(low confidence)"
                    print(f"{label:22s} p={prediction['confidence']:.3f}", flush=True)
                stop_event.wait(max(0, 0.1 - (time.monotonic() - tick)))
    finally:
        stop_event.set()
        if guard:
            guard.close()
        if cap is not None:
            cap.release()
        if peer_thread:
            peer_thread.join(timeout=1.5)
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        if guard:
            print(f"종료: {guard.reason}", flush=True)
            if guard.stop_errors:
                print(f"모터 정지 통신 오류: {guard.stop_errors}", file=sys.stderr)


if __name__ == "__main__":
    main()
