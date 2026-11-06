import io
import logging
import os
import unittest
from unittest import mock

from mains import _console


class Terminal(io.StringIO):
    def isatty(self):
        return True


class ConsoleTests(unittest.TestCase):
    def test_colour_is_automatic_only_for_supported_terminal_streams(self):
        with mock.patch.dict(os.environ, {"TERM": "xterm-256color"}, clear=True):
            self.assertTrue(_console.color_enabled(Terminal()))
            self.assertFalse(_console.color_enabled(io.StringIO()))
            with mock.patch.dict(os.environ, {"NO_COLOR": ""}):
                self.assertFalse(_console.color_enabled(Terminal()))
            with mock.patch.dict(os.environ, {"TERM": "dumb"}):
                self.assertFalse(_console.color_enabled(Terminal()))

    def test_long_identifiers_and_paths_are_preserved_without_terminal_codes(self):
        stream = io.StringIO()
        console = _console.Console(stream, width=40)
        path = "mains/logs/abeta42_7q4m_8bfa_replica_2-321.log"
        console.field("Log", path)
        self.assertIn(path, stream.getvalue())
        self.assertNotIn("\x1b", stream.getvalue())

    def test_runtime_logging_has_timestamps_and_severity_without_module_or_source(self):
        formatter = _console.ConsoleFormatter(io.StringIO())
        record = logging.LogRecord("private.module", logging.WARNING, "private.py", 123, "Checkpoint kept", (), None)
        output = formatter.format(record)
        self.assertRegex(output, r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d")
        self.assertIn("[WARNING] Checkpoint kept", output)
        self.assertNotIn("private", output)
        self.assertNotIn("\x1b", output)

    def test_external_terminal_codes_are_removed_from_plain_logs(self):
        formatter = _console.ConsoleFormatter(io.StringIO())
        record = logging.LogRecord("app", logging.ERROR, "app.py", 1, "\x1b[31mbad GPU\x1b[0m", (), None)
        self.assertIn("bad GPU", formatter.format(record))
        self.assertNotIn("\x1b", formatter.format(record))
