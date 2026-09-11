"""Logging configuration: rich console output plus a rotating file.

Call :func:`setup_logging` once at process start (the CLI and scripts do this).
Library modules only ever do ``logging.getLogger(__name__)``.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rich.logging import RichHandler

_CONFIGURED = False


def setup_logging(
    level: str = "INFO", log_file: str | Path | None = None
) -> logging.Logger:
    """Configure the root logger. Later calls are no-ops.

    Args:
        level: standard logging level name, e.g. ``"INFO"`` or ``"DEBUG"``.
        log_file: path of the file log; its parent directory is created.
            ``None`` disables file logging.
    """
    global _CONFIGURED
    root = logging.getLogger()
    if _CONFIGURED:
        return root

    level_name = level.upper()
    root.setLevel(level_name)

    console = RichHandler(rich_tracebacks=True, show_path=False, markup=False)
    console.setLevel(level_name)
    console.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    root.addHandler(console)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            path, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        root.addHandler(file_handler)

    # Third-party libraries are chatty at INFO; keep them to warnings.
    for noisy in ("httpx", "httpcore", "chromadb", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True
    return root
