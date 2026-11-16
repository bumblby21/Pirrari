#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pirrari TFLite 자율주행 실행 스크립트
경로: /home/pi/Pirrari/model/drive.py

- train.py로 학습된 pirrari_model.tflite를 불러와 10Hz로 실시간 추론 및 주행합니다.
- 안전을 위해 시작 시 정지(대기) 상태로 시작하며, Space 키로 자율주행 시작/일시정지를 토글합니다.
- 랜선/Wi-Fi 단절, Ctrl+C, 창 닫힘 시 모터를 즉시 정지합니다.
"""

import os
import sys
import time
import termios
import tty
import select
import atexit
import signal
import threading
import socket
import struct
import subprocess
from collections import deque
import numpy as np
import cv2
import tensorflow as tf
from Raspbot_Lib import Raspbot

# ===== 설정 =====
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pirrari_model.h5")
DEFAULT_SPEED = 28
DEFAULT_TURN = 25
SPEED_STEP = 2
HZ = 10
N_FRAMES = 5
IMG_SIZE = 32

ACTION_LABELS = {0: "FORWARD (전진)", 1: "RIGHT (우회전)", 2: "LEFT (좌회전)", 3: "STOP (정지)"}

# ===== 하드웨어 초기화 =====
bot = Raspbot()
speed = DEFAULT_SPEED
turn_speed = DEFAULT_TURN
is_shutting_down = False
current_action_name = "STOP (정지)"


def set_motor(l, r):
    try:
        bot.Ctrl_Muto(0, int(l))
        bot.Ctrl_Muto(1, int(l))
        bot.Ctrl_Muto(2, int(r))
        bot.Ctrl_Muto(3, int(r))
    except Exception:
        pass


def stop_motor():
    global current_action_name
    set_motor(0, 0)
    current_action_name = ACTION_LABELS[3]


def apply_action(action):
    global current_action_name
    if action == 0:
        set_motor(speed, speed)
    elif action == 1:
        set_motor(turn_speed, -turn_speed)
    elif action == 2:
        set_motor(-turn_speed, turn_speed)
    else:
        set_motor(0, 0)
        action = 3
    current_action_name = ACTION_LABELS.get(action, ACTION_LABELS[3])


def emergency_safe_shutdown(reason="알 수 없음"):
    global is_shutting_down
    if is_shutting_down:
        return
    is_shutting_down = True

    for _ in range(3):
        stop_motor()

    try:
        sys.stdout.write("\r\n\r\n" + "=" * 60 + "\r\n")
        sys.stdout.write(f" 🛑 [안전 정지 작동] 이유: {reason}\r\n")
        sys.stdout.write(" ⚙️  모든 모터가 안전하게 정지되었습니다.\r\n")
        sys.stdout.write("=" * 60 + "\r\n\r\n")
        sys.stdout.flush()
    except Exception:
        pass


atexit.register(emergency_safe_shutdown, "프로그램 종료 (atexit)")


def signal_handler(signum, frame):
    sig_names = {
        signal.SIGINT: "사용자 Ctrl+C 중단 (SIGINT)",
        signal.SIGTERM: "프로세스 강제 종료 신호 (SIGTERM)",
        signal.SIGHUP: "SSH / 터미널 창 닫힘 감지 (SIGHUP)",
    }
    emergency_safe_shutdown(sig_names.get(signum, f"시그널 {signum}"))
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
if hasattr(signal, "SIGHUP"):
    signal.signal(signal.SIGHUP, signal_handler)


# ===== 네트워크/랜선/Wi-Fi 연결 감시 워치독 =====
class NetworkWatchdog(threading.Thread):
    def __init__(self, callback, check_interval=0.1):
        super().__init__(daemon=True)
        self.callback = callback
        self.check_interval = check_interval
        self.running = True

        self.client_ip = None
        self.client_port = None
        ssh_conn = os.environ.get("SSH_CONNECTION", "")
        if ssh_conn:
            parts = ssh_conn.split()
            if len(parts) >= 2:
                self.client_ip = parts[0]
                self.client_port = int(parts[1])
        if not self.client_ip:
            ssh_client = os.environ.get("SSH_CLIENT", "")
            if ssh_client:
                parts = ssh_client.split()
                if len(parts) >= 2:
                    self.client_ip = parts[0]
                    self.client_port = int(parts[1])

        self.is_eth = False
        self.is_wlan = False
        ip_ref = self.client_ip or ""
        if "192.168.137." in ip_ref:
            self.is_eth = True
        elif "192.168.1." in ip_ref:
            self.is_wlan = True
        else:
            self.is_eth = self._is_eth_up()
            self.is_wlan = True

    def _is_eth_up(self):
        carrier_path = "/sys/class/net/eth0/carrier"
        if os.path.exists(carrier_path):
            try:
                with open(carrier_path, "r") as f:
                    return f.read().strip() == "1"
            except Exception:
                return False
        return False

    def _get_tcp_status(self):
        if not self.client_ip or not self.client_port:
            return None
        for tcp_file in ["/proc/net/tcp", "/proc/net/tcp6"]:
            if not os.path.exists(tcp_file):
                continue
            try:
                with open(tcp_file, "r") as f:
                    lines = f.readlines()
                for line in lines[1:]:
                    parts = line.strip().split()
                    if len(parts) < 10:
                        continue
                    try:
                        rem_ip_hex, rem_port_hex = parts[2].split(":")
                        if int(rem_port_hex, 16) != self.client_port:
                            continue
                        if len(rem_ip_hex) == 8:
                            rem_ip = socket.inet_ntoa(struct.pack("<I", int(rem_ip_hex, 16)))
                            if rem_ip == self.client_ip:
                                return {
                                    "found": True,
                                    "state": int(parts[3], 16),
                                    "tx_queue": int(parts[4].split(":")[0], 16),
                                    "retrnsmt": int(parts[6], 16),
                                    "tr": int(parts[5].split(":")[0], 16),
                                }
                    except Exception:
                        continue
            except Exception:
                pass
        return {"found": False}

    def run(self):
        eth_fail_count = 0
        tcp_fail_count = 0
        while self.running:
            time.sleep(self.check_interval)
            if not self.running:
                break
            if self.is_eth:
                eth_fail_count = 0 if self._is_eth_up() else eth_fail_count + 1
                if eth_fail_count >= 2:
                    self.callback("랜선(이더넷 케이블) 분리 감지")
                    os.kill(os.getpid(), signal.SIGTERM)
                    break
            if self.client_ip and self.client_port:
                tcp_info = self._get_tcp_status()
                broken = (
                    tcp_info is None
                    or not tcp_info.get("found", False)
                    or tcp_info.get("state") != 1
                    or (tcp_info.get("tr") == 1 and tcp_info.get("retrnsmt", 0) >= 2)
                    or (tcp_info.get("tx_queue", 0) > 8192 and tcp_info.get("retrnsmt", 0) >= 1)
                )
                tcp_fail_count = tcp_fail_count + 1 if broken else 0
                if tcp_fail_count >= 3:
                    reason = "와이파이(Wi-Fi) 통신 단절 감지" if self.is_wlan else "네트워크 통신 단절 감지"
                    self.callback(reason)
                    os.kill(os.getpid(), signal.SIGTERM)
                    break

    def stop(self):
        self.running = False


# ===== 비동기 카메라 스레드 =====
class Camera(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.frame = None
        self.lock = threading.Lock()
        self.running = True

    def run(self):
        while self.running:
            ok, f = self.cap.read()
            if ok:
                with self.lock:
                    self.frame = f
            else:
                time.sleep(0.01)

    def get(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False
        try:
            self.join(timeout=0.5)
        except Exception:
            pass
        if self.cap.isOpened():
            self.cap.release()


def get_key_nonblocking(timeout=0.01):
    try:
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
    except (select.error, ValueError):
        return "EOF"
    if not rlist:
        return None
    try:
        char = sys.stdin.read(1)
        if char == "":
            return "EOF"
    except Exception:
        return "EOF"
    return "ESC" if char == "\x1b" else char


def main():
    global speed, turn_speed

    model_path = sys.argv[1] if len(sys.argv) > 1 else MODEL_PATH
    if not os.path.isabs(model_path) and not os.path.exists(model_path):
        cand = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_path)
        if os.path.exists(cand):
            model_path = cand

    if not os.path.exists(model_path):
        print(f"❌ 학습된 모델 파일이 없습니다: {model_path}")
        print("   먼저 python3 /home/pi/Pirrari/model/train.py 를 실행해 모델을 학습해 주세요.")
        sys.exit(1)

    print(f"⏳ 모델 로딩 중: {os.path.basename(model_path)}")
    model = tf.keras.models.load_model(model_path, compile=False)
    model(np.zeros((1, IMG_SIZE, IMG_SIZE, N_FRAMES * 3), dtype=np.float32), training=False)

    stop_motor()
    cam = Camera()
    cam.start()

    net_watchdog = NetworkWatchdog(emergency_safe_shutdown)
    net_watchdog.start()

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    sys.stdout.write("\033[2J\033[H")
    sys.stdout.write(
        "===========================================================\r\n"
        " 🤖 Pirrari 자율주행 모드 (TFLite)\r\n"
        "===========================================================\r\n"
        "   Space     : 자율주행 시작 / 일시정지 토글\r\n"
        "   + / -     : 속도 조절 (±2)\r\n"
        "   Q / ESC   : 즉시 정지 및 안전 종료\r\n"
        "===========================================================\r\n\r\n"
    )
    sys.stdout.flush()

    driving = False
    buf = deque(maxlen=N_FRAMES)

    try:
        tty.setraw(fd)
        while True:
            t0 = time.time()

            # 1. 키보드 제어 (일시정지/재개/속도/종료)
            key = get_key_nonblocking(0.01)
            if key == "EOF":
                emergency_safe_shutdown("터미널 창 강제 종료 또는 통신 두절 (EOF)")
                break
            if key is not None:
                k = key.lower()
                if k == " ":
                    driving = not driving
                    if not driving:
                        stop_motor()
                        buf.clear()
                elif k in ["+", "="]:
                    speed = min(255, speed + SPEED_STEP)
                    turn_speed = min(255, turn_speed + SPEED_STEP)
                elif k in ["-", "_"]:
                    speed = max(10, speed - SPEED_STEP)
                    turn_speed = max(10, turn_speed - SPEED_STEP)
                elif k in ["q", "esc"]:
                    emergency_safe_shutdown("사용자 종료 명령 수신 ('Q'/'ESC')")
                    break

            # 2. 카메라 관찰 및 추론
            if driving:
                frame = cam.get()
                if frame is not None:
                    small = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
                    small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
                    buf.append(small)

                if len(buf) == N_FRAMES:
                    state = np.stack(buf)  # (5, 3, 32, 32)
                    x = state.transpose(2, 3, 0, 1).reshape(1, IMG_SIZE, IMG_SIZE, N_FRAMES * 3)
                    x = x.astype(np.float32) / 255.0
                    probs = model(x, training=False).numpy()[0]
                    action = int(np.argmax(probs))
                    conf = float(probs[action]) * 100.0
                    apply_action(action)
                    mode_text = f"🟢 주행 중 ({conf:4.1f}%)"
                else:
                    mode_text = f"🟡 프레임 수집 ({len(buf)}/{N_FRAMES})"
            else:
                stop_motor()
                mode_text = "⏸️ 일시정지 (Space:시작)"

            sys.stdout.write(
                f"\r\033[K▶ [{current_action_name:<14}] [{mode_text:<20}] 속도: {speed:>2} (Q:종료)"
            )
            sys.stdout.flush()

            elapsed = time.time() - t0
            time.sleep(max(0, (1.0 / HZ) - elapsed))

    except Exception as e:
        emergency_safe_shutdown(f"프로그램 예외 발생 ({e})")
    finally:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except Exception:
            pass
        net_watchdog.stop()
        cam.stop()
        emergency_safe_shutdown("메인 루프 종료")


if __name__ == "__main__":
    main()
