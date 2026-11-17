"""SSH socket monitoring must not depend on ping or motor hardware."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from autonomous.ssh_watch import SSHConnection, decode_endpoint, monitor_connection


class SSHWatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.connection = SSHConnection("192.168.1.2", 50000, 22, "192.168.1.3", self.root)

    def write_socket(self, state="01", retransmits="0", peer_port="C350"):
        (self.root / "tcp").write_text(
            "header\n"
            f"0: 0301A8C0:0016 0201A8C0:{peer_port} {state} "
            f"00000000:00000000 01:00000000 {retransmits} 1000 0 123\n")

    def test_active_ssh_does_not_call_ping(self):
        self.write_socket()
        with patch("subprocess.run", side_effect=AssertionError("No ping allowed")):
            self.assertIsNone(self.connection.failure_reason())

    def test_closed_missing_and_wrong_socket_rejected(self):
        self.assertIsNotNone(self.connection.failure_reason())
        self.write_socket(state="08")
        self.assertIsNotNone(self.connection.failure_reason())
        self.write_socket(peer_port="C351")
        self.assertIsNotNone(self.connection.failure_reason())

    def test_retransmission_failure(self):
        self.write_socket(retransmits="2")
        self.assertIn("재전송", self.connection.failure_reason())

    def test_ipv6_and_ipv4_mapped(self):
        address, port = decode_endpoint("00000000000000000000000001000000:0016")
        self.assertEqual(str(address), "::1")
        self.assertEqual(port, 22)
        address, _ = decode_endpoint("0000000000000000FFFF00000201A8C0:C350")
        self.assertEqual(str(address), "192.168.1.2")

    def test_environment(self):
        c = SSHConnection.from_environment({"SSH_CONNECTION": "192.168.1.2 50000 192.168.1.3 22"})
        self.assertEqual(c.peer_port, 50000)
        self.assertEqual(c.local_port, 22)
        self.assertIsNone(SSHConnection.from_environment({}))
        with self.assertRaises(RuntimeError):
            SSHConnection.from_environment({"SSH_CONNECTION": "bad"})

    def test_monitor_trips_guard_on_disconnect(self):
        self.write_socket(state="08")
        guard = Mock()
        guard.stop_event.wait.return_value = False
        monitor_connection(self.connection, guard)
        guard.trip.assert_called_once()


if __name__ == "__main__":
    unittest.main()
