import pytest


@pytest.fixture(autouse=True)
def isolated_icloud(tmp_path_factory, monkeypatch):
    """Never let a test archive into the real iCloud Drive: point the default archive at a temp dir."""
    import media_pipeline.config as config_module

    fake = tmp_path_factory.mktemp("icloud-drive")
    monkeypatch.setattr(config_module, "ICLOUD_DRIVE", fake)
    return fake


@pytest.fixture(autouse=True)
def isolated_bulk_jobs(tmp_path_factory, monkeypatch):
    """The pipeline looks up bulk downloads; keep tests away from the real ~/AIContent/bulk jobs."""
    import media_pipeline.config as config_module

    fake = tmp_path_factory.mktemp("bulk-jobs")
    monkeypatch.setattr(config_module, "BULK_ROOT", fake)
    return fake


@pytest.fixture(autouse=True)
def _no_b23_network(monkeypatch):
    """b23.tv short links are resolved over the network in production; tests keep them as they are."""
    from media_pipeline import media

    monkeypatch.setattr(media, "_resolve_b23", lambda url: url)
