# server/gpu_monitor.py
from __future__ import annotations

import logging
import time
from typing import TypedDict

from pynvml import (
    NVMLError,
    nvmlDeviceGetCount,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetUtilizationRates,
    nvmlInit,
    nvmlShutdown,
)

logger = logging.getLogger(__name__)


class GPUInfo(TypedDict):
    id: int
    memory_total: int
    memory_free: int
    memory_used: int
    util_gpu: int
    util_mem: int


def get_all_gpu_info() -> list[GPUInfo]:
    """
    返回所有 GPU 的状态列表
    """
    try:
        nvmlInit()
    except NVMLError:
        logger.exception("初始化 GPU 失败")
        return []

    try:
        gpu_count = nvmlDeviceGetCount()
        infos: list[GPUInfo] = []
        for i in range(gpu_count):
            try:
                handle = nvmlDeviceGetHandleByIndex(i)
                mem = nvmlDeviceGetMemoryInfo(handle)
                util = nvmlDeviceGetUtilizationRates(handle)
                info = GPUInfo(
                    id=i,
                    memory_total=int(mem.total),
                    memory_free=int(mem.free),
                    memory_used=int(mem.used),
                    util_gpu=int(util.gpu),
                    util_mem=int(util.memory),
                )
                infos.append(info)
            except NVMLError:
                logger.exception("读取 GPU[%s] 信息失败，已跳过", i)
        return infos
    finally:
        try:
            nvmlShutdown()
        except NVMLError:
            logger.debug("nvmlShutdown 失败，忽略", exc_info=True)


def monitor_loop(poll_interval: float = 5.0) -> None:
    """
    后台线程 /进程执行 GPU 信息监控，
    定期（默认每 poll_interval 秒）采集并可供调度器 /网页 UI 查询
    """
    while True:
        infos = get_all_gpu_info()
        # TODO 把 infos 存到全局缓存 /共享状态里
        print("GPU infos:", infos)
        time.sleep(poll_interval)
