#!/usr/bin/env python3
"""
ane_meter.py — Minimal start/stop ANE power meter for AI inference benchmarking.

Usage:
    import ane_meter

    ane_meter.start()
    # ... run your inference ...
    stats = ane_meter.stop()
    print(stats)  # {'samples': 12, 'peak_mw': 4820.0, 'avg_mw': 3100.5, 'duration_s': 6.1}

Requires: macOS Apple Silicon, sudo for powermetrics.
"""

import subprocess, threading, re, time

_thread = None
_stop = threading.Event()
_samples: list[float] = []
_t0 = 0.0

def _poll():
    while not _stop.is_set():
        try:
            r = subprocess.run(
                ["sudo", "powermetrics", "--samplers", "cpu_power", "-n", "1", "-i", "500"],
                capture_output=True, text=True, timeout=10,
            )
            m = re.search(r"ANE Power:\s*([\d.]+)\s*mW", r.stdout)
            if m:
                _samples.append(float(m.group(1)))
        except Exception:
            pass

def start():
    """Begin sampling ANE power in background."""
    global _thread, _t0
    _samples.clear()
    _stop.clear()
    _t0 = time.time()
    _thread = threading.Thread(target=_poll, daemon=True)
    _thread.start()

def stop() -> dict:
    """Stop sampling and return stats."""
    _stop.set()
    _thread.join(timeout=5)
    dur = time.time() - _t0
    if not _samples:
        return {"samples": 0, "peak_mw": None, "avg_mw": None, "duration_s": round(dur, 2)}
    return {
        "samples": len(_samples),
        "peak_mw": max(_samples),
        "avg_mw": round(sum(_samples) / len(_samples), 1),
        "duration_s": round(dur, 2),
    }

if __name__ == "__main__":
    print("Sampling ANE for 5s… (run an inference workload now)")
    start()
    time.sleep(5)
    print(stop())
