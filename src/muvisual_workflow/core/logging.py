"""Application logging configuration."""

from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path
import sys
from threading import Lock

from muvisual_workflow.core.paths import PROJECT_ROOT

LOG_DIR = PROJECT_ROOT / "log"
LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s"


class _LocalTimezoneFormatter(logging.Formatter):
    def formatTime(
        self,
        record: logging.LogRecord,
        datefmt: str | None = None,
    ) -> str:
        current = datetime.fromtimestamp(record.created).astimezone()
        return current.strftime(datefmt or "%Y-%m-%d %H:%M:%S%z")


class _DailyFileHandler(logging.Handler):
    def __init__(self, directory: Path) -> None:
        super().__init__()
        self.directory = directory
        self._date: str | None = None
        self._stream = None
        self._stream_lock = Lock()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            current_date = datetime.fromtimestamp(record.created).astimezone().strftime(
                "%Y-%m-%d"
            )
            with self._stream_lock:
                if current_date != self._date:
                    if self._stream is not None:
                        self._stream.close()
                    self.directory.mkdir(parents=True, exist_ok=True)
                    self._stream = (self.directory / f"log_{current_date}.log").open(
                        "a",
                        encoding="utf-8",
                    )
                    self._date = current_date
                self._stream.write(message + "\n")
                self._stream.flush()
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        with self._stream_lock:
            if self._stream is not None:
                self._stream.close()
                self._stream = None
        super().close()


def configure_logging() -> None:
    root = logging.getLogger()
    if getattr(root, "_muvisual_configured", False):
        return

    formatter = _LocalTimezoneFormatter(LOG_FORMAT)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    file_handler = _DailyFileHandler(LOG_DIR)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    root.setLevel(logging.DEBUG)
    root.addHandler(console)
    root.addHandler(file_handler)
    logging.captureWarnings(True)
    root._muvisual_configured = True


def get_logger(module: str) -> logging.Logger:
    return logging.getLogger(module)
