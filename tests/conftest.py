import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def have(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def tiny_model(tmp_path_factory):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from folioocr.model.tiny import build_tiny
    path = tmp_path_factory.mktemp("tiny") / "model"
    build_tiny(str(path))
    return str(path)


def assets_dir():
    try:
        from folioocr.render.assets import find_assets_dir
        return find_assets_dir()
    except FileNotFoundError:
        return None


def chromium_ok() -> bool:
    if not have("playwright"):
        return False
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            b = p.chromium.launch()
            b.close()
        return True
    except Exception:
        return False
