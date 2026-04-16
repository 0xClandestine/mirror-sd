#!/usr/bin/env python3
"""
ane_meter.py — Sudoless start/stop ANE power meter using macmon.

Install: brew install macmon

Usage:
    import ane_meter
    ane_meter.start()
    # ... run inference ...
    stats = ane_meter.stop()
    print(stats)

Context manager:
    with ane_meter.measure() as m:
        # ... run inference ...
    print(m.stats)
"""

import subprocess
import threading
import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional

_proc: Optional[subprocess.Popen] = None
_stop = threading.Event()
_samples: list[float] = []
_t0: float = 0.0
_lock = threading.Lock()


def _poll() -> None:
    global _proc
    _proc = subprocess.Popen(
        ["macmon", "raw", "--interval", "500"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    for line in _proc.stdout:
        if _stop.is_set():
            break
        try:
            data = json.loads(line)
            with _lock:
                _samples.append(data.get("ane_power", 0.0) * 1000)  # W -> mW
        except (json.JSONDecodeError, KeyError):
            pass
    _proc.terminate()


def start() -> None:
    """Begin sampling ANE power at 500ms intervals."""
    global _t0
    with _lock:
        _samples.clear()
    _stop.clear()
    _t0 = time.perf_counter()
    threading.Thread(target=_poll, daemon=True).start()


def stop() -> dict:
    """Stop sampling and return summary stats."""
    _stop.set()
    if _proc:
        _proc.terminate()
    time.sleep(0.3)  # drain last sample
    dur = time.perf_counter() - _t0
    with _lock:
        s = list(_samples)
    if not s:
        return {
            "samples": 0,
            "peak_mw": None,
            "avg_mw": None,
            "duration_s": round(dur, 3),
        }
    return {
        "samples": len(s),
        "peak_mw": round(max(s), 1),
        "avg_mw": round(sum(s) / len(s), 1),
        "energy_mj": round(sum(s) * 0.5, 1),  # 500ms intervals → mW * 0.5s = mJ
        "duration_s": round(dur, 3),
    }


@dataclass
class _Measurement:
    stats: dict = field(default_factory=dict)


@contextmanager
def measure():
    """Context manager: ``with ane_meter.measure() as m: ...  print(m.stats)``"""
    m = _Measurement()
    start()
    try:
        yield m
    finally:
        m.stats = stop()


if __name__ == "__main__":
    print("Sampling ANE for 5s… (run an inference workload now)")
    start()
    time.sleep(5)
    print(stop())
