"""Rich console output plus a rotating file. Called once at process start;
library modules only ever do logging.getLogger(__name__).
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
    """Configure the root logger; later calls are no-ops. log_file=None disables
    file logging."""
    global _CONFIGURED
    root = logging.getLogger()
    if _CONFIGURED:
        return root

    level_name = level.upper()
    root.setLevel(level_name)

    console = RichHandler(rich_tracebacks=False, show_path=False, markup=False)
    console.setLevel(level_name)
    console.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    # Tracebacks belong in the file; the console gets one-line messages only.
    console.addFilter(lambda record: record.exc_info is None)
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
