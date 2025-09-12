import pytest

from polar_flow.server import gpu_monitor as gm


def test_get_all_gpu_info_nvml_ok(monkeypatch):
    # 构造假的 NVML API（覆盖 34-55 行）
    class DummyMem:
        def __init__(self, total, free, used):
            self.total, self.free, self.used = total, free, used

    class DummyUtil:
        def __init__(self, gpu, memory):
            self.gpu, self.memory = gpu, memory

    def fake_init():
        pass

    def fake_count():
        return 2

    def fake_handle(idx):
        return f"H{idx}"

    def fake_mem(h):
        i = int(h[1:])
        return DummyMem(total=16 * 1024**3, free=(8 - i) * 1024**3, used=i * 1024**3)

    def fake_util(h):
        i = int(h[1:])
        return DummyUtil(gpu=10 * i, memory=5 * i)

    monkeypatch.setattr(gm, "nvmlInit", fake_init)
    monkeypatch.setattr(gm, "nvmlDeviceGetCount", fake_count)
    monkeypatch.setattr(gm, "nvmlDeviceGetHandleByIndex", fake_handle)
    monkeypatch.setattr(gm, "nvmlDeviceGetMemoryInfo", fake_mem)
    monkeypatch.setattr(gm, "nvmlDeviceGetUtilizationRates", fake_util)

    infos = gm.get_all_gpu_info()
    assert len(infos) == 2
    assert infos[0]["id"] == 0
    assert "memory_free" in infos[0]


def test_get_all_gpu_info_nvml_init_fail(monkeypatch):
    # nvmlInit 抛异常 → 早返回 []（覆盖异常分支）
    class DummyError(Exception):
        pass

    monkeypatch.setattr(gm, "NVMLError", DummyError, raising=False)

    def fake_init():
        raise DummyError("boom")

    monkeypatch.setattr(gm, "nvmlInit", fake_init)
    assert gm.get_all_gpu_info() == []


def test_monitor_loop_runs_once(monkeypatch, capsys):
    # 让循环只跑一次：sleep 抛出 KeyboardInterrupt（覆盖 63-67）
    monkeypatch.setattr(
        gm,
        "get_all_gpu_info",
        lambda: [
            {
                "id": 0,
                "memory_total": 0,
                "memory_free": 0,
                "memory_used": 0,
                "util_gpu": 0,
                "util_mem": 0,
            },
        ],
    )

    def fake_sleep(_):
        raise KeyboardInterrupt

    monkeypatch.setattr(gm.time, "sleep", fake_sleep)
    with pytest.raises(KeyboardInterrupt):
        gm.monitor_loop(0.01)
    out = capsys.readouterr().out
    assert "GPU infos:" in out
