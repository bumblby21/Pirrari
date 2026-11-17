"""Hardware-free runtime tests."""

import tempfile
from pathlib import Path
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from autonomous.drive import choose_action, main, prepare_frame
from autonomous.safety import MotorGuard


class FakeBot:
    def __init__(self):
        self.values = [0] * 4
        self.fail_motor = None
        self.calls = []

    def Ctrl_Muto(self, motor, value):
        self.calls.append((motor, value))
        if self.fail_motor == motor:
            raise OSError("mock I2C failure")
        self.values[motor] = value


class RuntimeTests(unittest.TestCase):
    def guard(self):
        bot = FakeBot()
        guard = MotorGuard(bot, threading.Event(), timeout=0.15)
        self.addCleanup(guard.close)
        return bot, guard

    def test_mapping_and_latched_stop(self):
        bot, guard = self.guard()
        for action, expected in [(0, [28, 28, 28, 28]), (1, [25, 25, -25, -25]),
                                 (2, [-25, -25, 25, 25]), (None, [0, 0, 0, 0])]:
            self.assertTrue(guard.command(action, time.monotonic()))
            self.assertEqual(bot.values, expected)
        guard.trip("keyboard")
        self.assertFalse(guard.command(0, time.monotonic()))
        self.assertEqual(bot.values, [0] * 4)

    def test_watchdog_stops_without_main_loop(self):
        bot, guard = self.guard()
        guard.command(0, time.monotonic())
        self.assertTrue(guard.stop_event.wait(1))
        guard.thread.join(timeout=1)
        self.assertEqual(bot.values, [0] * 4)
        self.assertFalse(guard.command(0, time.monotonic()))

    def test_stale_prediction_never_moves(self):
        bot, guard = self.guard()
        self.assertFalse(guard.command(0, time.monotonic() - 1))
        self.assertEqual(bot.values, [0] * 4)

    def test_motor_failure_attempts_stop_on_all_channels(self):
        bot, guard = self.guard()
        bot.fail_motor = 1
        with self.assertRaises(OSError):
            guard.command(0, time.monotonic())
        self.assertTrue(guard.stop_event.is_set())
        for motor in range(4):
            self.assertIn((motor, 0), bot.calls[-4:])

    def test_preprocessing_and_confidence(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        frame[:] = [10, 20, 30]
        small = prepare_frame(frame)
        self.assertEqual(small.shape, (3, 32, 32))
        np.testing.assert_array_equal(small[:, 0, 0], [30, 20, 10])
        self.assertEqual(choose_action({"probabilities": [0.1, 0.8, 0.1]}, 0.6), 1)
        self.assertIsNone(choose_action({"probabilities": [0.3, 0.4, 0.3]}, 0.6))
        with self.assertRaises(ValueError):
            choose_action({"probabilities": [float("nan"), 0, 1]}, 0.6)

    def test_camera_loop_and_failure_cleanup(self):
        class Camera:
            def __init__(self):
                self.count = 0
                self.released = False

            def set(self, *args):
                pass

            def isOpened(self):
                return True

            def read(self):
                self.count += 1
                if self.count > 5:
                    return False, None
                return True, np.full((240, 320, 3), self.count, dtype=np.uint8)

            def release(self):
                self.released = True

        camera = Camera()
        observed = []

        class Policy:
            def __init__(self, *args):
                pass

            def predict(self, state):
                observed.append(state)
                return {"action": 0, "name": "forward", "confidence": 0.8,
                        "probabilities": [0.8, 0.1, 0.1]}

        with tempfile.TemporaryDirectory() as temp:
            model = Path(temp) / "best.pt"
            model.touch()
            with patch("autonomous.drive.DrivingPolicy", Policy), \
                    patch("autonomous.drive.cv2.VideoCapture", return_value=camera), \
                    patch("autonomous.drive.sys.stdin.isatty", return_value=False):
                main(["--model", str(model)])
        self.assertTrue(camera.released)
        self.assertEqual(len(observed), 2)  # warmup plus first complete camera window
        np.testing.assert_array_equal(observed[-1][:, 0, 0, 0], [1, 2, 3, 4, 5])


if __name__ == "__main__":
    unittest.main()
