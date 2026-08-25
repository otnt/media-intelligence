import threading

from media_pipeline.visual.vlm import MlxVlmProvider, _mlx_loop


def test_mlx_work_stays_on_a_daemon_thread_that_survives_close():
    provider = MlxVlmProvider("dummy")
    names: list[str] = []

    def work() -> int:
        names.append(threading.current_thread().name)
        return 7

    assert provider._on_thread(work) == 7
    assert names == ["mlx-vlm"]
    loop = _mlx_loop()
    assert loop._thread.daemon
    assert loop._thread.is_alive()
    provider.close()
    assert loop._thread.is_alive()


def test_mlx_loop_is_shared_and_reentrant_from_the_worker():
    provider = MlxVlmProvider("dummy")

    def inner() -> str:
        return threading.current_thread().name

    def outer() -> str:
        return provider._on_thread(inner)

    assert provider._on_thread(outer) == "mlx-vlm"
    assert _mlx_loop() is _mlx_loop()
