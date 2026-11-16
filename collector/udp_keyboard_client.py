#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UDP 키보드 컨트롤러 (원격 조종기)
경로: /home/pi/Pirrari/collector/udp_keyboard_client.py

PC나 노트북에서 라즈베리파이로 무선(UDP) 키보드 명령을 보낼 때 사용합니다.
(라즈베리파이 내부 로컬 테스트 시 대상 IP: 127.0.0.1)

[전송 명령]
- W / ↑ : 'w' (직진 / 액션 0)
- D / → : 'd' (우회전 / 액션 1)
- A / ← : 'a' (좌회전 / 액션 2)
- S / ↓ : 's' (정지 / 액션 3)
- Space : 's' (정지)
- R     : 'r' (데이터 녹화 시작/중지 토글)
- Q / X : 'x' (모터 종료 명령)
"""

import socket
import sys
import time
import termios
import tty
import select

def get_key(timeout=0.05):
    fd = sys.stdin.fileno()
    rlist, _, _ = select.select([sys.stdin], [], [], timeout)
    if not rlist:
        return None
    char = sys.stdin.read(1)
    if char == '\x1b':
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

def main():
    target_ip = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    target_port = int(sys.argv[2]) if len(sys.argv) > 2 else 5005

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    print("=" * 55)
    print(f" UDP 키보드 컨트롤러 -> 대상: {target_ip}:{target_port}")
    print("=" * 55)
    print("  W 또는 ↑ : 직진 (w)")
    print("  S 또는 ↓ : 정지/후진 (s)")
    print("  A 또는 ← : 좌회전 (a)")
    print("  D 또는 → : 우회전 (d)")
    print("  Space   : 정지 (s)")
    print("  R       : 데이터 녹화 토글 (r)")
    print("  Q / X   : 종료 명령 전송 및 프로그램 종료")
    print("=" * 55)

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        tty.setraw(fd)
        last_sent = ""
        while True:
            key = get_key(0.05)
            if key is not None:
                k = key.lower()
                cmd = None
                if k in ['w', 'up']:
                    cmd = 'w'
                elif k in ['s', 'down', ' ']:
                    cmd = 's'
                elif k in ['a', 'left']:
                    cmd = 'a'
                elif k in ['d', 'right']:
                    cmd = 'd'
                elif k == 'r':
                    cmd = 'r'
                elif k in ['x', 'q', 'esc']:
                    cmd = 'x'

                if cmd:
                    sock.sendto(cmd.encode(), (target_ip, target_port))
                    last_sent = cmd
                    if cmd == 'x':
                        break

            sys.stdout.write(f"\r[마지막 전송 명령: {last_sent:^5}] (종료: Q)   ")
            sys.stdout.flush()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        sock.close()
        print("\nUDP 컨트롤러가 종료되었습니다.\n")

if __name__ == "__main__":
    main()
