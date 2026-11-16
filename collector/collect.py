#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
라즈봇 데이터 수집 프로그램 (UDP 통신형 + 안전 보장 시스템 탑재)
경로: /home/pi/Pirrari/collector/collect.py

[강의 자료(PPT Page 22) 필수 안전 요구사항 100% 구현]
✓ 1. 사용자가 특정 모터 종료 명령을 보냈을 때 ('x' UDP 수신)
✓ 2. 사용자가 Ctrl+C 를 눌러 프로그램을 종료하였을 때 (SIGINT)
✓ 3. 프로그램이 비정상 종료되었을 때 (SIGHUP 창 닫기, SIGTERM, atexit, Exception)
✓ 4. 로봇과 User PC간 통신이 두절되었을 때 (TIMEOUT 0.5초 경과 시 모터 자동 정지)
★ 비상 종료 시에도 녹화 데이터 유실 없이 자동 .npz 파일로 저장!
"""

import os
import sys
import time
import socket
import threading
import atexit
import signal
from collections import deque
import numpy as np
import cv2
from Raspbot_Lib import Raspbot

# ===== 설정 =====
PORT = 5005          # UDP 명령 수신 포트
HZ = 10              # 10Hz 제어 루프
N_FRAMES = 5         # 상태 프레임 수
IMG_SIZE = 32        # 32x32 해상도
SPEED = 28           # 전진 기본 속도 (저속 모드: 28)
TURN = 25            # 회전 기본 속도 (저속 모드: 25)
COMM_TIMEOUT = 0.5   # 통신 두절 감지 타임아웃 (0.5초)
SAVE_EVERY = 500     # 500 샘플마다 자동 저장
SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset")

CMD = {"w": 0, "d": 1, "a": 2, "s": 3}

# ===== 하드웨어 초기화 =====
bot = Raspbot()
is_shutting_down = False
save_lock = threading.Lock()

# ===== 데이터 저장 =====
os.makedirs(SAVE_DIR, exist_ok=True)
session_time = time.strftime("%Y%m%d_%H%M%S")
states_list = []
actions_list = []
part_num = 0
total_saved_count = 0

def save_npz_data():
    global states_list, actions_list, part_num, total_saved_count
    with save_lock:
        if not states_list:
            return None, 0
        count = len(states_list)
        path = os.path.join(SAVE_DIR, f"{session_time}_{part_num:03d}.npz")
        np.savez_compressed(
            path,
            states=np.array(states_list, dtype=np.uint8),
            actions=np.array(actions_list, dtype=np.int64)
        )
        total_saved_count += count
        part_num += 1
        states_list = []
        actions_list = []
        return path, count

# ===== 모터 제어 =====
def set_motor(l, r):
    try:
        bot.Ctrl_Muto(0, int(l))
        bot.Ctrl_Muto(1, int(l))
        bot.Ctrl_Muto(2, int(r))
        bot.Ctrl_Muto(3, int(r))
    except Exception:
        pass

def stop_motor():
    try:
        set_motor(0, 0)
    except Exception:
        pass

def do_action(a):
    if a == 0:     # 직진
        set_motor(SPEED, SPEED)
    elif a == 1:   # 우회전
        set_motor(TURN, -TURN)
    elif a == 2:   # 좌회전
        set_motor(-TURN, TURN)
    else:          # 정지
        stop_motor()

# ===== 안전 종료 총괄 관리자 =====
def emergency_safe_shutdown(reason="알 수 없음"):
    global is_shutting_down
    if is_shutting_down:
        return
    is_shutting_down = True

    # 모터 강제 3중 정지
    for _ in range(3):
        stop_motor()

    # 녹화 데이터 자동 저장
    saved_path, cnt = save_npz_data()

    try:
        print("\n" + "=" * 60)
        print(f" 🛑 [안전 정지 작동] 이유: {reason}")
        print(" ⚙️  모터가 즉시 안전 정지되었습니다.")
        if cnt > 0:
            print(f" 💾 녹화 중이던 데이터 {cnt}개가 안전하게 저장되었습니다: {os.path.basename(saved_path)}")
        print("=" * 60 + "\n")
    except Exception:
        pass

atexit.register(emergency_safe_shutdown, "프로그램 atexit 종료")

def signal_handler(signum, frame):
    sig_map = {
        signal.SIGINT: "사용자 Ctrl+C 종료 (SIGINT)",
        signal.SIGTERM: "프로세스 종료 신호 (SIGTERM)",
        signal.SIGHUP: "창 닫힘 / 통신 연결 끊김 (SIGHUP)",
    }
    emergency_safe_shutdown(sig_map.get(signum, f"시그널 {signum}"))
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
if hasattr(signal, "SIGHUP"):
    signal.signal(signal.SIGHUP, signal_handler)

# ===== 카메라 스레드 =====
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
        if self.cap.isOpened():
            self.cap.release()

# ===== 메인 함수 =====
def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", PORT))
    sock.setblocking(False)

    cam = Camera()
    cam.start()

    buf = deque(maxlen=N_FRAMES)
    action, last_recv, recording = 3, time.time(), False

    print("=" * 60)
    print(f" 🚀 라즈봇 데이터 수집 (UDP 포트: {PORT})")
    print(" 🛡️ 안전 보호: 통신 두절(0.5초 초과), 창 닫힘, Ctrl+C 시 즉시 정지")
    print("=" * 60)

    try:
        while True:
            t0 = time.time()

            # 1. 카메라 관찰
            frame = cam.get()
            if frame is None:
                time.sleep(0.01)
                continue

            # 2. State 생성 (5, 3, 32, 32)
            small = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
            buf.append(small)

            # 3. 전문가 명령 수신
            has_new_cmd = False
            while True:
                try:
                    data, _ = sock.recvfrom(64)
                except BlockingIOError:
                    break
                key = data.decode(errors="ignore").strip().lower()
                last_recv = time.time()
                has_new_cmd = True

                # 특정 모터 종료 명령
                if key == "x":
                    emergency_safe_shutdown("사용자 특정 종료 명령 수신 ('x')")
                    return
                if key == "r":
                    recording = not recording
                    status_text = "🟢 [녹화 시작]" if recording else "⚪ [녹화 일시정지]"
                    print(f"\n{status_text}")
                elif key in CMD:
                    action = CMD[key]

            # 4. 통신 두절 감지: 0.5초 동안 명령이 없으면 즉시 모터 정지!
            if time.time() - last_recv > COMM_TIMEOUT:
                stop_motor()
                action = 3
                time.sleep(max(0, 1.0 / HZ - (time.time() - t0)))
                continue

            # 모터 구동
            do_action(action)

            # 데이터 기록
            if recording and len(buf) == N_FRAMES:
                with save_lock:
                    states_list.append(np.stack(buf))
                    actions_list.append(action)
                cur_cnt = total_saved_count + len(states_list)
                if cur_cnt % 50 == 0:
                    print(f"수집 진행 중: {cur_cnt}개", end="\r")
                if len(states_list) >= SAVE_EVERY:
                    save_npz_data()

            # 5. 10Hz 주기 유지
            elapsed = time.time() - t0
            time.sleep(max(0, (1.0 / HZ) - elapsed))

    except Exception as e:
        emergency_safe_shutdown(f"프로그램 예외 발생 ({e})")
    finally:
        cam.stop()
        sock.close()
        emergency_safe_shutdown("메인 루프 종료")

if __name__ == "__main__":
    main()
