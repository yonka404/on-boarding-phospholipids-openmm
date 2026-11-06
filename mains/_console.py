"""Small standard-library console helpers, usable with system Python 3.8+."""

from __future__ import annotations

import logging
import os
import re
import shutil
import sys
import textwrap
from typing import TextIO

_CSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_TONES = {"cyan": "36", "green": "32", "yellow": "33", "red": "31", "bold": "1"}


def clean_text(value: object) -> str:
    """Keep subprocess messages and filenames from injecting terminal controls."""
    return _CONTROL.sub("", _CSI.sub("", str(value)))


def color_enabled(stream: TextIO) -> bool:
    return (
        bool(getattr(stream, "isatty", lambda: False)())
        and "NO_COLOR" not in os.environ
        and os.environ.get("TERM") != "dumb"
    )


def paint(value: str, tone: str, color: bool) -> str:
    return f"\x1b[{_TONES[tone]}m{value}\x1b[0m" if color else value


class Console:
    def __init__(self, stream: TextIO | None = None, *, width: int | None = None):
        self.stream = sys.stdout if stream is None else stream
        self.color = color_enabled(self.stream)
        self.width = width or shutil.get_terminal_size((120, 24)).columns

    def heading(self, value: str) -> None:
        self.line(paint(clean_text(value), "cyan", self.color))

    def line(self, value: str = "") -> None:
        print(value, file=self.stream, flush=True)

    def field(self, label: str, value: object, *, tone: str | None = None) -> None:
        prefix = f"{label:<12} "
        lines = []
        for paragraph in clean_text(value).splitlines() or [""]:
            lines.extend(textwrap.wrap(
                paragraph, width=max(20, self.width - len(prefix)),
                break_long_words=False, break_on_hyphens=False,
            ) or [""])
        for index, line in enumerate(lines):
            label_text = prefix if index == 0 else " " * len(prefix)
            if tone and index == 0:
                label_text = paint(label_text, tone, self.color)
            self.line(label_text + line)


class ConsoleFormatter(logging.Formatter):
    def __init__(self, stream: TextIO, *, timestamps: bool = True):
        prefix = "%(asctime)s " if timestamps else ""
        super().__init__(prefix + "[%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        self.stream = stream

    def format(self, record: logging.LogRecord) -> str:
        message = clean_text(super().format(record))
        if color_enabled(self.stream):
            tone = "red" if record.levelno >= logging.ERROR else "yellow" if record.levelno >= logging.WARNING else "cyan"
            label = f"[{record.levelname}]"
            message = message.replace(label, paint(label, tone, True), 1)
        return message


def configure_logging(*, timestamps: bool = True) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(ConsoleFormatter(handler.stream, timestamps=timestamps))
    logging.basicConfig(level=logging.INFO, handlers=[handler])
