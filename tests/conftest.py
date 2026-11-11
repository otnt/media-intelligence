import pytest


@pytest.fixture(autouse=True)
def isolated_icloud(tmp_path_factory, monkeypatch):
    """Never let a test archive into the real iCloud Drive: point the default archive at a temp dir."""
    import media_pipeline.config as config_module

    fake = tmp_path_factory.mktemp("icloud-drive")
    monkeypatch.setattr(config_module, "ICLOUD_DRIVE", fake)
    return fake
