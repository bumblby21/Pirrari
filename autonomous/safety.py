"""Motor commands with a latched stop and independent software watchdog."""

import threading
import time


class MotorGuard:
    def __init__(self, bot, stop_event, speed=28, turn=25, timeout=0.5):
        self.bot, self.stop_event = bot, stop_event
        self.speed, self.turn, self.timeout = speed, turn, timeout
        self.lock = threading.Lock()
        self.deadline = None
        self.reason = ""
        self.stop_errors = []
        self._zero()
        if self.stop_errors:
            raise RuntimeError(f"초기 모터 정지 실패: {self.stop_errors}")
        self.thread = threading.Thread(target=self._watch, daemon=True)
        self.thread.start()

    def _zero(self):
        if self.bot is not None:
            for motor in range(4):
                try:
                    self.bot.Ctrl_Muto(motor, 0)
                except Exception as exc:
                    self.stop_errors.append(f"motor {motor}: {exc}")

    def trip(self, reason):
        # Latch before acquiring the lock, so no subsequent command can resume.
        self.stop_event.set()
        with self.lock:
            if not self.reason:
                self.reason = reason
            self._zero()

    def command(self, action, captured_at):
        with self.lock:
            if self.stop_event.is_set():
                return False
            now = time.monotonic()
            if now - captured_at > self.timeout or captured_at > now:
                self.stop_event.set()
                self.reason = "오래된 카메라 입력/추론 지연"
                self._zero()
                return False
            pairs = {0: (self.speed, self.speed), 1: (self.turn, -self.turn),
                     2: (-self.turn, self.turn), None: (0, 0)}
            if action not in pairs:
                self.stop_event.set()
                self.reason = "잘못된 액션"
                self._zero()
                return False
            self.deadline = captured_at + self.timeout
            left, right = pairs[action]
            try:
                if self.bot is not None:
                    for motor, value in enumerate((left, left, right, right)):
                        if self.stop_event.is_set():
                            self._zero()
                            return False
                        self.bot.Ctrl_Muto(motor, value)
            except Exception:
                self.stop_event.set()
                self.reason = "모터 명령 실패"
                self._zero()
                raise
            return True

    def check(self):
        if self.stop_event.is_set():
            self.trip("중단 요청")
            return False
        with self.lock:
            expired = self.deadline is not None and time.monotonic() > self.deadline
        if expired:
            self.trip("카메라/추론 명령 갱신 시간 초과")
            return False
        return True

    def _watch(self):
        while not self.stop_event.wait(0.02):
            if not self.check():
                return
        self.trip("중단 요청")

    def close(self):
        self.trip("종료")
        self.thread.join(timeout=1)
