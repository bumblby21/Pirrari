#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Raspbot 키보드 제어 및 데이터 수집 프로그램 (네트워크 단절 안전 보장 시스템)
경로: /home/pi/Pirrari/collector/keyboard_control.py

[조작 편의성 및 필수 안전 요구사항]
✓ 1. 조작 편의: 키를 계속 누르고 있을 필요 없이 방향키 1회 입력으로 연속 주행 가능
✓ 2. 비상 정지: Space 또는 X 키 입력 시 즉시 모터 정지
✓ 3. 정상 종료: Q 또는 ESC 키 입력 시 데이터 저장 후 안전 종료
✓ 4. 비정상 종료: Ctrl+C (SIGINT), 강제 종료 (SIGTERM), 창 닫기 (SIGHUP) 시 모터 즉시 0 정지
✓ 5. 네트워크 통신 단절 자동 감지:
     - 컴퓨터와 연결된 랜선(이더넷) 분리 즉시 감지 (0.2~0.3초 내 모터 정지)
     - 와이파이(Wi-Fi) 연결 단절 즉시 감지 (모터 정지)
     - 통신 단절 시 주행 중이던 데이터는 유실 없이 자동 .npz 압축 저장!
"""

import sys
import os
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
from Raspbot_Lib import Raspbot

# ===== 설정 =====
DEFAULT_SPEED = 28    # 직진 기본 속도 (작은 맵용 저속: 28)
DEFAULT_TURN = 25     # 회전 기본 속도 (작은 맵용 저속: 25)
SPEED_STEP = 2        # 속도 조절 단위 (±2)
HZ = 10               # 10Hz (초당 10회 루프)
N_FRAMES = 5          # 상태(State)로 묶을 최근 프레임 수
IMG_SIZE = 32         # 32x32 해상도
SAVE_EVERY = 500      # 500개마다 자동 저장
TARGET_GOAL = 5000    # 목표 수집 개수
SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset")

# ===== 하드웨어 초기화 =====
bot = Raspbot()

speed = DEFAULT_SPEED
turn_speed = DEFAULT_TURN
current_action_name = "STOP (정지)"
current_action_code = 3  # 0:전진, 1:우회전, 2:좌회전, 3:정지
recording = False
is_shutting_down = False
save_lock = threading.Lock()

# ===== 데이터 저장용 전역 변수 =====
os.makedirs(SAVE_DIR, exist_ok=True)
session_time = time.strftime("%Y%m%d_%H%M%S")
states_list = []
actions_list = []
part_num = 0
total_saved_count = 0

def save_npz_data():
    """모인 데이터를 .npz 파일로 저장 (스레드 안전)"""
    global states_list, actions_list, part_num, total_saved_count
    with save_lock:
        if not states_list:
            return None, 0
        count = len(states_list)
        file_path = os.path.join(SAVE_DIR, f"{session_time}_{part_num:03d}.npz")
        np.savez_compressed(
            file_path,
            states=np.array(states_list, dtype=np.uint8),
            actions=np.array(actions_list, dtype=np.int64)
        )
        total_saved_count += count
        part_num += 1
        states_list = []
        actions_list = []
        return file_path, count

# ===== 모터 제어 함수 =====
def set_motor(l, r):
    try:
        bot.Ctrl_Muto(0, int(l))
        bot.Ctrl_Muto(1, int(l))
        bot.Ctrl_Muto(2, int(r))
        bot.Ctrl_Muto(3, int(r))
    except Exception:
        pass

def stop_motor():
    global current_action_name, current_action_code
    set_motor(0, 0)
    current_action_name = "STOP (정지)"
    current_action_code = 3

def move_forward():
    global current_action_name, current_action_code
    set_motor(speed, speed)
    current_action_name = "FORWARD (전진)"
    current_action_code = 0

def move_backward():
    global current_action_name, current_action_code
    set_motor(-speed, -speed)
    current_action_name = "BACKWARD (후진)"
    current_action_code = 3

def turn_left():
    global current_action_name, current_action_code
    set_motor(-turn_speed, turn_speed)
    current_action_name = "LEFT (좌회전)"
    current_action_code = 2

def turn_right():
    global current_action_name, current_action_code
    set_motor(turn_speed, -turn_speed)
    current_action_name = "RIGHT (우회전)"
    current_action_code = 1

# ===== 안전 종료 총괄 관리자 (Safety Manager) =====
def emergency_safe_shutdown(reason="알 수 없음", save_data=True):
    """
    모든 비상/정상 종료 상황에서 모터를 즉시 정지시키고
    save_data=True인 경우 현재까지 녹화된 데이터를 100% 안전하게 디스크에 저장합니다.
    """
    global is_shutting_down
    if is_shutting_down:
        return
    is_shutting_down = True

    # 1. 모터 즉각 정지 (최우선: 확실한 정지를 위해 3회 전송)
    for _ in range(3):
        stop_motor()

    # 2. 녹화 중이던 데이터 디스크에 저장 (비정상 종료/연결 단절 시 무조건 자동 저장)
    saved_path, cnt = (None, 0)
    if save_data:
        saved_path, cnt = save_npz_data()

    # 터미널이 열려있는 경우에만 안내 출력
    try:
        sys.stdout.write("\r\n\r\n" + "=" * 60 + "\r\n")
        sys.stdout.write(f" 🛑 [안전 정지 작동] 이유: {reason}\r\n")
        sys.stdout.write(f" ⚙️  모든 모터가 안전하게 정지되었습니다.\r\n")
        if cnt > 0:
            sys.stdout.write(f" 💾 녹화 중이던 데이터 {cnt}개가 안전하게 저장되었습니다!\r\n")
            sys.stdout.write(f"    저장 파일: {os.path.basename(saved_path)}\r\n")
        elif not save_data:
            sys.stdout.write(" 🗑️  사용자 선택에 따라 미저장 데이터를 저장하지 않고 종료합니다.\r\n")
        sys.stdout.write("=" * 60 + "\r\n\r\n")
        sys.stdout.flush()
    except Exception:
        pass

# 파이썬 exit 시 무조건 호출
atexit.register(emergency_safe_shutdown, "프로그램 정상/비정상 종료 (atexit)")

def signal_handler(signum, frame):
    sig_names = {
        signal.SIGINT: "사용자 Ctrl+C 중단 (SIGINT)",
        signal.SIGTERM: "프로세스 강제 종료 신호 (SIGTERM)",
        signal.SIGHUP: "SSH / 터미널 창 닫힘 감지 (SIGHUP)",
    }
    name = sig_names.get(signum, f"시그널 {signum}")
    emergency_safe_shutdown(name)
    sys.exit(0)

# 주요 시스템 시그널 가로채기
signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # kill
if hasattr(signal, "SIGHUP"):
    signal.signal(signal.SIGHUP, signal_handler)  # SSH 창 X 버튼 닫기

# ===== 네트워크/랜선/Wi-Fi 연결 감시 워치독 =====
class NetworkWatchdog(threading.Thread):
    """
    1. 랜선(eth0) 물리 분리 감지: carrier 신호 실시간 감시 (0.2초 이내)
    2. Wi-Fi(wlan0) 물리 어댑터 감시: operstate 감시
    3. Wi-Fi 클라이언트 이탈 감시: Yahboom AP(핫스팟) 모드일 때 iw station dump에서 노트북 MAC 확인
    4. SSH 터미널 TCP 세션 실시간 감시 (/proc/net/tcp):
       - 화면 출력에 대한 TCP ACK 미수신, 재전송 타이머(tr==1) 및 재전송 카운트(retrnsmt>=2) 실시간 추적
       - 사용자가 노트북에서 Wi-Fi를 끄거나 신호가 끊기면 0.3~0.5초 이내에 포착하여 모터 차단
    5. ARP 상태 감시: 클라이언트 ARP가 INCOMPLETE(0x0)로 전환되면 즉시 차단
    """
    def __init__(self, callback, check_interval=0.1):
        super().__init__(daemon=True)
        self.callback = callback
        self.check_interval = check_interval
        self.running = True

        # SSH 접속 클라이언트 정보 파악 (SSH_CONNECTION, SSH_CLIENT, who 폴백 지원)
        self.client_ip = None
        self.client_port = None
        self.server_ip = None
        self.server_port = None

        ssh_conn = os.environ.get("SSH_CONNECTION", "")
        if ssh_conn:
            parts = ssh_conn.split()
            if len(parts) >= 4:
                self.client_ip = parts[0]
                self.client_port = int(parts[1])
                self.server_ip = parts[2]
                self.server_port = int(parts[3])

        if not self.client_ip:
            ssh_client = os.environ.get("SSH_CLIENT", "")
            if ssh_client:
                parts = ssh_client.split()
                if len(parts) >= 2:
                    self.client_ip = parts[0]
                    self.client_port = int(parts[1])

        # 클라이언트 MAC 주소 획득
        self.client_mac = self._get_client_mac(self.client_ip) if self.client_ip else None

        # 접속 유형 판별 (eth0 vs wlan0)
        self.is_eth = False
        self.is_wlan = False
        ip_ref = self.client_ip or self.server_ip or ""
        if "192.168.137." in ip_ref:
            self.is_eth = True
        elif "192.168.1." in ip_ref:
            self.is_wlan = True
        else:
            # 특정되지 않으면 둘 다 감시
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

    def _get_client_mac(self, client_ip):
        try:
            with open("/proc/net/arp", "r") as f:
                for line in f.readlines()[1:]:
                    p = line.strip().split()
                    if len(p) >= 6 and p[0] == client_ip:
                        flags = int(p[2], 16)
                        if flags != 0:
                            return p[3].lower()
        except Exception:
            pass
        return None

    def _is_arp_valid(self, client_ip):
        try:
            with open("/proc/net/arp", "r") as f:
                for line in f.readlines()[1:]:
                    p = line.strip().split()
                    if len(p) >= 6 and p[0] == client_ip:
                        flags = int(p[2], 16)
                        # 0x0 is INCOMPLETE (연결 끊김)
                        if flags == 0:
                            return False
                        return True
        except Exception:
            pass
        return True

    def _get_tcp_status(self):
        if not self.client_ip or not self.client_port:
            return None

        # /proc/net/tcp 및 /proc/net/tcp6 확인
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
                    rem_addr = parts[2]
                    try:
                        rem_ip_hex, rem_port_hex = rem_addr.split(":")
                        if int(rem_port_hex, 16) != self.client_port:
                            continue
                        if len(rem_ip_hex) == 8:
                            rem_ip = socket.inet_ntoa(struct.pack("<I", int(rem_ip_hex, 16)))
                            if rem_ip == self.client_ip:
                                st = int(parts[3], 16)
                                tx_q = int(parts[4].split(":")[0], 16)
                                tr = int(parts[5].split(":")[0], 16)
                                retrnsmt = int(parts[6], 16)
                                return {
                                    "found": True,
                                    "state": st,
                                    "tx_queue": tx_q,
                                    "tr": tr,
                                    "retrnsmt": retrnsmt,
                                }
                    except Exception:
                        continue
            except Exception:
                pass
        return {"found": False}

    def _is_station_in_ap(self, mac):
        if not mac:
            return True
        try:
            res = subprocess.run(
                ["iw", "dev", "wlan0", "station", "dump"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=0.1
            )
            return mac.lower() in res.stdout.lower()
        except Exception:
            return True

    def run(self):
        eth_fail_count = 0
        tcp_fail_count = 0
        ap_fail_count = 0
        arp_fail_count = 0

        while self.running:
            time.sleep(self.check_interval)
            if not self.running:
                break

            # 1. 랜선(eth0) 물리 분리 감시 (이더넷 연결 시)
            if self.is_eth:
                if not self._is_eth_up():
                    eth_fail_count += 1
                else:
                    eth_fail_count = 0

                if eth_fail_count >= 2:  # ~0.2초
                    self.callback("랜선(이더넷 케이블) 분리 감지")
                    os.kill(os.getpid(), signal.SIGTERM)
                    break

            # 2. SSH TCP 연결 상태 실시간 감시 (Wi-Fi 및 랜선 공통 핵심)
            if self.client_ip and self.client_port:
                tcp_info = self._get_tcp_status()
                is_tcp_broken = False

                if tcp_info is None or not tcp_info.get("found", False):
                    # 소켓이 닫혔거나 사라짐
                    is_tcp_broken = True
                elif tcp_info.get("state") != 1:  # TCP_ESTABLISHED 아님
                    is_tcp_broken = True
                elif tcp_info.get("tr") == 1 and tcp_info.get("retrnsmt", 0) >= 2:
                    # TCP 재전송 타이머 활성화 및 2회 이상 연속 실패 (와이파이 차단 등)
                    is_tcp_broken = True
                elif tcp_info.get("tx_queue", 0) > 8192 and tcp_info.get("retrnsmt", 0) >= 1:
                    # 화면 출력 송신 큐가 과도하게 쌓이고 재전송 실패
                    is_tcp_broken = True

                if is_tcp_broken:
                    tcp_fail_count += 1
                else:
                    tcp_fail_count = 0

                if tcp_fail_count >= 3:  # 약 0.3~0.4초 지속 시 비상 정지
                    reason = "와이파이(Wi-Fi) 통신 단절 감지 (TCP 재전송 실패)" if self.is_wlan else "네트워크 통신 단절 감지"
                    self.callback(reason)
                    os.kill(os.getpid(), signal.SIGTERM)
                    break

            # 3. AP 모드일 때 와이파이 클라이언트(Station) 연결 감시
            if self.is_wlan:
                # MAC 주소가 처음엔 없었을 수 있으므로 갱신 시도
                if not self.client_mac and self.client_ip:
                    self.client_mac = self._get_client_mac(self.client_ip)

                if self.client_mac:
                    # AP station dump 확인
                    if not self._is_station_in_ap(self.client_mac):
                        ap_fail_count += 1
                    else:
                        ap_fail_count = 0

                    if ap_fail_count >= 3:  # 약 0.3~0.4초
                        self.callback("와이파이(Wi-Fi) 연결 단절 감지 (AP 스테이션 이탈)")
                        os.kill(os.getpid(), signal.SIGTERM)
                        break

                # 4. ARP 상태 감시 (클라이언트 ARP 엔트리가 INCOMPLETE로 바뀐 경우)
                if self.client_ip:
                    if not self._is_arp_valid(self.client_ip):
                        arp_fail_count += 1
                    else:
                        arp_fail_count = 0

                    if arp_fail_count >= 3:  # 약 0.3~0.4초
                        self.callback("와이파이(Wi-Fi) 연결 단절 감지 (ARP 응답 없음)")
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
        try:
            if self.cap.isOpened():
                self.cap.release()
        except Exception:
            pass

    def get(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False
        try:
            self.join(timeout=0.5)
        except Exception:
            pass

# ===== 비동기 키 입력 감지 =====
def get_key_nonblocking(timeout=0.01):
    try:
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
    except (select.error, ValueError):
        return 'EOF'
    if not rlist:
        return None

    try:
        char = sys.stdin.read(1)
        if char == '':  # 창이 닫혀 stdin EOF 발생
            return 'EOF'
    except Exception:
        return 'EOF'

    if char == '\x1b':  # Escape sequence
        rlist, _, _ = select.select([sys.stdin], [], [], 0.02)
        if rlist:
            char2 = sys.stdin.read(1)
            if char2 == '[':
                rlist, _, _ = select.select([sys.stdin], [], [], 0.02)
                if rlist:
                    char3 = sys.stdin.read(1)
                    if char3 == 'A': return 'UP'
                    elif char3 == 'B': return 'DOWN'
                    elif char3 == 'C': return 'RIGHT'
                    elif char3 == 'D': return 'LEFT'
        return 'ESC'
    return char

# ===== 메인 함수 =====
def main():
    global speed, turn_speed, recording

    stop_motor()

    cam = Camera()
    cam.start()

    # 랜선 / 와이파이 단절 실시간 감시 워치독 시작
    net_watchdog = NetworkWatchdog(emergency_safe_shutdown)
    net_watchdog.start()

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    sys.stdout.write("\033[2J\033[H")
    banner = (
        "===========================================================\r\n"
        " Raspbot 키보드 주행 & 데이터 수집 (네트워크 안전 시스템 완비)\r\n"
        "===========================================================\r\n"
        " [조작 키 안내] (1회 누르면 계속 주행 유지)\r\n"
        "   W 또는 ↑  : 전진 (Action 0)\r\n"
        "   D 또는 →  : 우회전 (Action 1)\r\n"
        "   A 또는 ←  : 좌회전 (Action 2)\r\n"
        "   S 또는 ↓  : 후진\r\n"
        "   Space / X : 즉시 멈춤 (Action 3)\r\n"
        "   --------------------------------------------------------\r\n"
        "   R         : 🔴 녹화 시작 / ⚪ 일시 정지 (토글, 정지·후진은 자동 제외)\r\n"
        "   + / -     : 속도 조절 (현재 속도 ±2)\r\n"
        "   Q         : 데이터 저장 후 안전 종료 (명령어 종료)\r\n"
        "===========================================================\r\n"
        " 🛡️ [네트워크 안전 보호 기능]\r\n"
        "  • 랜선(이더넷) 또는 Wi-Fi 연결 끊김 시 0.3초 내 모터 즉시 0 정지\r\n"
        "  • 창 닫기(SIGHUP), Ctrl+C 시 즉시 정지 및 데이터 안전 저장\r\n"
        "  • 손을 떼어도 주행은 유지되며, Space/X 키로 정지 가능\r\n"
        "===========================================================\r\n\r\n"
    )
    sys.stdout.write(banner)
    sys.stdout.flush()

    buf = deque(maxlen=N_FRAMES)
    saved_msg = ""
    current_count = 0

    try:
        tty.setraw(fd)

        while True:
            t0 = time.time()

            # 1. 키보드 입력 감지 (창 닫힘 감지 포함)
            key = get_key_nonblocking(0.01)

            if key == 'EOF':
                emergency_safe_shutdown("터미널 창 강제 종료 또는 통신 두절 (EOF 감지)")
                break

            if key is not None:
                k = key.lower()
                if k in ['w', 'up']:
                    move_forward()
                elif k in ['s', 'down']:
                    move_backward()
                elif k in ['a', 'left']:
                    turn_left()
                elif k in ['d', 'right']:
                    turn_right()
                elif k in [' ', 'x']:
                    stop_motor()
                elif k == 'r':
                    recording = not recording
                elif k in ['+', '=']:
                    speed = min(255, speed + SPEED_STEP)
                    turn_speed = min(255, turn_speed + SPEED_STEP)
                    if "FORWARD" in current_action_name: move_forward()
                    elif "LEFT" in current_action_name: turn_left()
                    elif "RIGHT" in current_action_name: turn_right()
                elif k in ['-', '_']:
                    speed = max(10, speed - SPEED_STEP)
                    turn_speed = max(10, turn_speed - SPEED_STEP)
                    if "FORWARD" in current_action_name: move_forward()
                    elif "LEFT" in current_action_name: turn_left()
                    elif "RIGHT" in current_action_name: turn_right()
                elif k in ['q', 'esc']:
                    for _ in range(3):
                        stop_motor()
                    try:
                        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                    except Exception:
                        pass

                    save_choice = True
                    with save_lock:
                        pending_cnt = len(states_list)

                    if pending_cnt > 0:
                        sys.stdout.write(f"\r\n\r\n💾 아직 저장되지 않은 데이터 {pending_cnt}개가 있습니다.\r\n")
                        sys.stdout.flush()
                        while True:
                            try:
                                ans = input("   데이터셋을 저장하시겠습니까? (y/n): ").strip().lower()
                            except (EOFError, KeyboardInterrupt):
                                # 입력 도중 연결 단절 시 유실 방지를 위해 자동 저장
                                save_choice = True
                                break
                            if ans in ['y', 'yes']:
                                save_choice = True
                                break
                            elif ans in ['n', 'no']:
                                save_choice = False
                                with save_lock:
                                    states_list.clear()
                                    actions_list.clear()
                                break
                            else:
                                print("   'y' (예) 또는 'n' (아니오)를 입력해 주세요.")

                    emergency_safe_shutdown("사용자 종료 명령 수신 ('Q'/'ESC')", save_data=save_choice)
                    break

            # 2. 카메라 이미지 획득 및 State 생성
            # 정지/후진(3) 중에는 버퍼를 비워 재주행 시 이전 흐름이 5프레임에 섞이지 않도록 방지
            if current_action_code in (0, 1, 2):
                frame = cam.get()
                if frame is not None:
                    small = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
                    small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
                    buf.append(small)
            else:
                buf.clear()

            # 3. 데이터 기록 (전진=0, 우회전=1, 좌회전=2만 기록 — 정지/후진=3은 자동 제외)
            if recording and len(buf) == N_FRAMES and current_action_code in (0, 1, 2):
                with save_lock:
                    states_list.append(np.stack(buf))
                    actions_list.append(current_action_code)
                current_count = total_saved_count + len(states_list)

                # 500개마다 중간 자동 저장
                if len(states_list) >= SAVE_EVERY:
                    saved_path, cnt = save_npz_data()
                    saved_msg = f"[자동저장: {os.path.basename(saved_path)}]"

            # 4. 상태 출력
            if recording:
                if current_action_code in (0, 1, 2):
                    rec_icon = f"🔴 녹화 중 ({current_count}/{TARGET_GOAL})"
                else:
                    rec_icon = f"⏸️ 정지제외 ({current_count}/{TARGET_GOAL})"
            else:
                rec_icon = f"⚪ 녹화 대기 ({current_count}/{TARGET_GOAL})"
            status_line = (
                f"\r\033[K▶ [{current_action_name:<11}] [{rec_icon:<18}] "
                f"속도: {speed:>2} {saved_msg} (R:녹화, Q:종료)"
            )
            sys.stdout.write(status_line)
            sys.stdout.flush()

            # 5. 10Hz 주기 유지
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
        emergency_safe_shutdown("메인 루프 정상 종료")

if __name__ == "__main__":
    main()
