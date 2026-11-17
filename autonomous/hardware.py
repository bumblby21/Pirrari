"""Resolve the robot's existing driver without opening I2C on import."""

import importlib
from pathlib import Path
import sys


def load_raspbot_class():
    try:
        return importlib.import_module("Raspbot_Lib").Raspbot
    except ModuleNotFoundError as exc:
        if exc.name != "Raspbot_Lib":
            raise
        # This Pi's torch venv does not expose the system-installed Raspbot egg.
        # The vendor's source package is already present here; reuse it unchanged.
        source = Path.home() / "py_install"
        if not (source / "Raspbot_Lib" / "__init__.py").is_file():
            raise RuntimeError("현재 Python 환경에 Raspbot_Lib를 설치해야 합니다.") from exc
        sys.path.insert(0, str(source))
        return importlib.import_module("Raspbot_Lib").Raspbot
