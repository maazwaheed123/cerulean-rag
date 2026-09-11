"""Shared pytest configuration: skip `ollama`-marked tests when no server is up."""

from __future__ import annotations

import urllib.request

import pytest


def _ollama_reachable(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(base_url, timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    from cerulean_rag.config import get_settings

    if _ollama_reachable(get_settings().OLLAMA_BASE_URL):
        return
    skip = pytest.mark.skip(reason="Ollama server not reachable")
    for item in items:
        if "ollama" in item.keywords:
            item.add_marker(skip)
