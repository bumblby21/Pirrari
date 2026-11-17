"""Watch the existing SSH TCP connection without requiring ICMP replies."""

import ipaddress
import os
from pathlib import Path
import sys


def normalized_ip(value):
    address = ipaddress.ip_address(value)
    return getattr(address, "ipv4_mapped", None) or address


def decode_endpoint(value):
    address, port = value.split(":")
    # /proc/net/tcp{,6} encodes each 32-bit address word in host byte order.
    packed = b"".join(int(address[i:i + 8], 16).to_bytes(4, sys.byteorder)
                      for i in range(0, len(address), 8))
    return normalized_ip(packed), int(port, 16)


class SSHConnection:
    def __init__(self, peer, peer_port, local_port, local=None, proc_root="/proc/net"):
        self.peer = normalized_ip(peer)
        self.peer_port = int(peer_port)
        self.local_port = int(local_port)
        self.local = normalized_ip(local) if local else None
        self.proc_root = Path(proc_root)

    @classmethod
    def from_environment(cls, environment=None):
        environment = os.environ if environment is None else environment
        connection = environment.get("SSH_CONNECTION", "").split()
        client = environment.get("SSH_CLIENT", "").split()
        try:
            if connection:
                peer, peer_port, local, local_port = connection
                return cls(peer, peer_port, local_port, local)
            if client:
                peer, peer_port, local_port = client
                return cls(peer, peer_port, local_port)
        except ValueError as exc:
            raise RuntimeError("SSH 연결 환경변수 형식이 잘못되었습니다. 새 SSH 터미널에서 실행하세요.") from exc
        return None

    def failure_reason(self):
        for filename in ("tcp", "tcp6"):
            try:
                lines = (self.proc_root / filename).read_text().splitlines()[1:]
            except FileNotFoundError:
                continue
            except OSError as exc:
                return f"SSH 연결 상태를 읽을 수 없습니다: {exc}"
            for line in lines:
                fields = line.split()
                if len(fields) < 7:
                    continue
                local_ip, local_port = decode_endpoint(fields[1])
                peer_ip, peer_port = decode_endpoint(fields[2])
                if (peer_ip != self.peer or peer_port != self.peer_port
                        or local_port != self.local_port
                        or (self.local is not None and local_ip != self.local)):
                    continue
                if fields[3] != "01":  # TCP_ESTABLISHED
                    return "SSH TCP 연결이 종료되었습니다."
                timer = int(fields[5].split(":")[0], 16)
                retransmits = int(fields[6], 16)
                if timer == 1 and retransmits >= 2:
                    return "SSH TCP 재전송이 반복 실패했습니다."
                return None
        return "현재 SSH 연결을 찾지 못했습니다. 새 SSH 터미널에서 직접 실행하세요."


def monitor_connection(connection, guard):
    while not guard.stop_event.wait(0.2):
        try:
            reason = connection.failure_reason()
        except Exception as exc:
            reason = f"SSH 연결 감시 오류: {exc}"
        if reason:
            guard.trip(reason)
            return
