"""Placeholder tests to verify the package installs and imports correctly."""

from __future__ import annotations


def test_version() -> None:
    from teddympnn import __version__

    assert __version__ == "0.0.1"


def test_import() -> None:
    import teddympnn

    assert hasattr(teddympnn, "__version__")
