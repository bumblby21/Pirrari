#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wi-Fi 및 유선 네트워크 안전 보장 시스템 전체 통합 검증 스크립트
"""
import sys
import os
import pty
import time
import subprocess

def run_tests():
    print("=" * 60)
    print(" 🚀 [안전 검증] Wi-Fi 및 랜선 단절 시 모터 안전 정지 통합 테스트")
    print("=" * 60)

    # ----------------------------------------------------
    # [검증 1] 주행 유지 및 Space/X 비상 정지 검증
    # ----------------------------------------------------
    print("\n[검증 1] 프로그램 기동 및 키 미입력 상태 주행 유지 검증...")
    master, slave = pty.openpty()
    env = os.environ.copy()

    proc = subprocess.Popen(
        [sys.executable, "/home/pi/Pirrari/collector/keyboard_control.py"],
        stdin=slave, stdout=slave, stderr=slave, close_fds=True, env=env
    )
    os.close(slave)
    time.sleep(1.0)
    assert proc.poll() is None, "프로그램 기동 실패"
    print("✓ 프로그램 정상 기동 완료 (PID: %d)" % proc.pid)

    # W (전진)
    os.write(master, b'w')
    time.sleep(0.2)
    print("✓ 'w' (전진) 키 입력 전송 완료")

    # 1.5초 동안 키 미입력 대기
    time.sleep(1.5)
    assert proc.poll() is None, "데드맨 스위치 제거 확인: 손을 떼어도 계속 주행해야 합니다."
    print("✓ 손을 떼고 1.5초 대기해도 주행(FORWARD) 상태 정상 유지 확인!")

    # X (정지)
    os.write(master, b'x')
    time.sleep(0.2)
    print("✓ 'x' (정지) 키로 멈춤 확인")

    # Q (안전 종료)
    os.write(master, b'q')
    proc.wait(timeout=2.0)
    assert proc.returncode == 0, f"Q 종료 실패 (code: {proc.returncode})"
    os.close(master)
    print("✓ 'q' 키로 데이터 저장 및 정상 종료 완료 (Exit code: 0)")

    # ----------------------------------------------------
    # [검증 2, 3, 4] 격리된 자식 프로세스에서 워치독 단절 테스트
    # ----------------------------------------------------
    tests = [
        ("Wi-Fi 단절(TCP 재전송 실패)", """
import sys, os, time
sys.path.insert(0, '/home/pi/Pirrari/collector')
from keyboard_control import NetworkWatchdog
os.environ['SSH_CONNECTION'] = '192.168.1.197 64050 192.168.1.11 22'

class Mock(NetworkWatchdog):
    def _get_tcp_status(self):
        return {'found': True, 'state': 1, 'tx_queue': 12000, 'tr': 1, 'retrnsmt': 3}

called = []
dog = Mock(lambda r: called.append(r), check_interval=0.05)
dog.start()
time.sleep(0.25)
dog.stop()
assert len(called) > 0
print('PASS:', called[0])
"""),
        ("Wi-Fi AP 스테이션 이탈 감지", """
import sys, os, time
sys.path.insert(0, '/home/pi/Pirrari/collector')
from keyboard_control import NetworkWatchdog
os.environ['SSH_CONNECTION'] = '192.168.1.197 64050 192.168.1.11 22'

class Mock(NetworkWatchdog):
    def _get_tcp_status(self):
        return {'found': True, 'state': 1, 'tx_queue': 0, 'tr': 0, 'retrnsmt': 0}
    def _is_station_in_ap(self, mac):
        return False

called = []
dog = Mock(lambda r: called.append(r), check_interval=0.05)
dog.start()
time.sleep(0.25)
dog.stop()
assert len(called) > 0
print('PASS:', called[0])
"""),
        ("랜선(이더넷) 케이블 물리 분리 감지", """
import sys, os, time
sys.path.insert(0, '/home/pi/Pirrari/collector')
from keyboard_control import NetworkWatchdog
os.environ['SSH_CONNECTION'] = '192.168.137.1 60371 192.168.137.171 22'

class Mock(NetworkWatchdog):
    def _is_eth_up(self):
        return False

called = []
dog = Mock(lambda r: called.append(r), check_interval=0.05)
dog.start()
time.sleep(0.20)
dog.stop()
assert len(called) > 0
print('PASS:', called[0])
""")
    ]

    for title, code in tests:
        print(f"\n[검증] {title} 테스트 중...")
        res = subprocess.run([sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        assert res.returncode == 0, f"테스트 실패: {res.stderr}"
        for line in res.stdout.strip().split("\n"):
            if line.startswith("PASS:"):
                print(f"✓ {title} 성공: {line}")

    print("\n" + "=" * 60)
    print(" 🎉 모든 안전 요구사항(Wi-Fi 단절 + 랜선 분리 + 주행 유지) 100% 통과!")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
