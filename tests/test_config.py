"""Settings load with defaults and validate AS_OF_DATE and LOG_LEVEL."""

from datetime import date

import pytest
from pydantic import ValidationError

from cerulean_rag.config import Settings


def test_defaults_load() -> None:
    s = Settings(_env_file=None)
    assert s.GEN_MODEL == "qwen2.5:7b-instruct"
    assert s.EMBED_MODEL == "nomic-embed-text"
    assert s.AS_OF_DATE == date(2026, 8, 27)
    assert s.TOP_K == 8


def test_as_of_date_parses_iso_string() -> None:
    s = Settings(_env_file=None, AS_OF_DATE="2026-01-15")
    assert s.AS_OF_DATE == date(2026, 1, 15)


def test_as_of_date_rejects_garbage() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, AS_OF_DATE="not-a-date")


def test_log_level_normalised_and_validated() -> None:
    assert Settings(_env_file=None, LOG_LEVEL="debug").LOG_LEVEL == "DEBUG"
    with pytest.raises(ValidationError):
        Settings(_env_file=None, LOG_LEVEL="loud")
