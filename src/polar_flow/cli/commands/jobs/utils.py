import re
import time
from datetime import datetime

from polar_flow._vendor.slurm_client.models.v0043_uint_32_no_val_struct import (
    V0043Uint32NoValStruct,
)
from polar_flow._vendor.slurm_client.models.v0043_uint_64_no_val_struct import (
    V0043Uint64NoValStruct,
)


def parse_noval_ui64(value: str) -> V0043Uint64NoValStruct:
    v = value.strip().lower()
    if v in {"infinite", "inf", "unlimited"}:
        return V0043Uint64NoValStruct(set_=False, infinite=True, number=0)

    try:
        iv = int(v)
        return V0043Uint64NoValStruct(set_=True, infinite=False, number=iv)
    except ValueError:
        raise ValueError(f"{value} 无法被解析为整数") from None


def parse_noval_ui32(value: str) -> V0043Uint32NoValStruct:
    v = value.strip().lower()
    if v in {"infinite", "inf", "unlimited"}:
        return V0043Uint32NoValStruct(set_=False, infinite=True, number=0)

    try:
        iv = int(v)
        return V0043Uint32NoValStruct(set_=True, infinite=False, number=iv)
    except ValueError:
        raise ValueError(f"{value} 无法被解析为整数") from None


PATTERN = re.compile(
    r"""^
    (?:(?P<month>\d{1,2})/(?P<day>\d{1,2})
        (?:/(?P<year>\d{2,4}))?-)?   # 日期部分，可选
    (?P<hour>\d{1,2}):(?P<minute>\d{2})
    (?::(?P<second>\d{2}))?          # 秒数，可选
    $""",
    re.VERBOSE,
)


def parse_to_unix_local(s: str) -> int:
    """
    解析 [MM/DD[/YY]-]HH:MM[:SS] 字符串为本地时间的 UNIX 时间戳 (秒级)
    """
    m = PATTERN.match(s.strip())
    if not m:
        raise ValueError(f"无效时间格式: {s}")

    parts = m.groupdict()
    now = datetime.now()  # noqa: DTZ005

    # 年份
    if parts["year"] is None:
        year = now.year
    else:
        year = int(parts["year"])
        if year < 100:  # 两位数年份，补全为 2000+
            year += 2000

    month = int(parts["month"] or now.month)
    day = int(parts["day"] or now.day)
    hour = int(parts["hour"])
    minute = int(parts["minute"])
    second = int(parts["second"] or 0)

    # 构造 naive datetime（本地时间）
    try:
        dt = datetime(year, month, day, hour, minute, second)  # noqa: DTZ001
    except ValueError:
        raise ValueError("非法日期时间") from None

    # 转换为本地时间戳
    return int(time.mktime(dt.timetuple()))


def parse_time_type(value: str) -> V0043Uint64NoValStruct:
    """
    将字符串解析为 TimeType(NoValType)
    """
    v = value.strip().lower()
    if v in {"infinite", "inf", "unlimited"}:
        return V0043Uint64NoValStruct(set_=False, infinite=True, number=0)

    try:
        iv = int(v)
        return V0043Uint64NoValStruct(set_=True, infinite=False, number=iv)
    except ValueError:
        pass

    # [MM/DD[/YY]-]HH:MM[:SS]
    return V0043Uint64NoValStruct(set_=True, infinite=False, number=parse_to_unix_local(v))
