#!/usr/bin/env python3
"""
Verify that constexpr_affine_dequantize actually stores int8 weights at runtime.

The key question: after ANECCompile() processes a kernel built with
constexpr_quantized_weight_1x1, does the compiled ANEC IR store weights as
int8 (1 byte/element) or expand them to fp16 (2 bytes/element)?

Method:
  1. Compile a tiny q8 conv1x1 kernel
  2. Find the ANEC cache entry it created (newest dir under /var/folders/.../T/)
  3. Parse the 'data' blob — compare blob byte sizes against expected int8 vs fp16 sizes
  4. Report bandwidth impact: int8 = half the ANE memory bandwidth of fp16

Also compares against the CPU-side dequant path (quantized_weight_1x1) as a baseline.

Usage:
    uv run python scripts/test_constexpr_q8.py
"""

import glob
import os
import struct
import time
import tempfile
import subprocess

# ── ANEC cache parsing ─────────────────────────────────────────────────────────

DEADBEEF = 0xDEADBEEF
CACHE_BASES = [
    "/var/folders",
]


def find_anec_cache_dirs(newer_than: float) -> list[str]:
    """Find ANEC compile-cache dirs created after `newer_than` (epoch seconds)."""
    dirs = []
    for base in CACHE_BASES:
        for data_path in glob.glob(f"{base}/**/T/*/data", recursive=True):
            if os.path.getmtime(data_path) >= newer_than:
                dirs.append(os.path.dirname(data_path))
    return sorted(dirs, key=lambda d: os.path.getmtime(os.path.join(d, "data")))


def parse_blobs(data: bytes) -> list[dict]:
    """Parse DEADBEEF blob records from ANEC IR data file.

    Format:
      [64-byte global header]  -- NOT a DEADBEEF record, skip it
      For each weight:
        [64-byte DEADBEEF record]  (+0x00 magic, +0x04 chunk_ver, +0x08 byte_size,
                                    +0x0C 0, +0x10 file_offset, ...)
        [byte_size bytes of blob data]
    """
    blobs = []
    i = 64  # skip 64-byte global header
    while i + 64 <= len(data):
        magic, chunk_ver, byte_size, _, file_offset = struct.unpack_from("<5I", data, i)
        if magic == DEADBEEF:
            dtype = {0x01: "fp16/uint8", 0x04: "int8"}.get(chunk_ver, f"0x{chunk_ver:02x}")
            blobs.append({
                "offset": i,
                "chunk_ver": chunk_ver,
                "dtype": dtype,
                "byte_size": byte_size,
                "file_offset": file_offset,
            })
            i += 64 + byte_size
        else:
            i += 4  # scan forward for next DEADBEEF (shouldn't be needed, but be safe)
    return blobs


def analyze_cache_dir(path: str, oc: int, ic: int) -> dict:
    data_path = os.path.join(path, "data")
    if not os.path.exists(data_path):
        return {}
    data = open(data_path, "rb").read()
    blobs = parse_blobs(data)
    result = {
        "path": path,
        "file_size": len(data),
        "blobs": blobs,
        "oc": oc,
        "ic": ic,
    }
    # Classify weight blobs
    int8_blobs  = [b for b in blobs if b["chunk_ver"] == 0x04]
    fp16_blobs  = [b for b in blobs if b["chunk_ver"] == 0x01]
    result["int8_blobs"]  = int8_blobs
    result["fp16_blobs"]  = fp16_blobs

    # ANECCompile stores weights transposed+padded: [ic, padded_oc]
    # where padded_oc = oc * 5 // 4  (25% channel padding, confirmed empirically)
    padded_oc = oc * 5 // 4
    expected_int8_padded = ic * padded_oc          # transposed, int8 (1 byte/elem)
    expected_int8_flat   = oc * ic                 # non-transposed (fallback check)
    expected_fp16_padded = ic * padded_oc * 2      # transposed, fp16 (2 bytes/elem)
    expected_fp16_flat   = oc * ic * 2             # non-transposed fp16

    result["verdict"] = "unknown"
    for b in blobs:
        if b["byte_size"] in (expected_int8_padded, expected_int8_flat):
            result["verdict"] = "INT8 — weights kept as int8 in ANEC IR (half bandwidth)"
        elif b["byte_size"] in (expected_fp16_padded, expected_fp16_flat):
            result["verdict"] = "FP16 — weights expanded to fp16 in ANEC IR (no bandwidth gain)"
    return result


def print_report(label: str, result: dict, oc: int, ic: int):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    if not result:
        print("  No cache entry found.")
        return
    print(f"  Cache dir : {os.path.basename(result['path'])[:48]}")
    print(f"  File size : {result['file_size']} bytes")
    print(f"  Blobs     : {len(result['blobs'])} total  "
          f"({len(result['int8_blobs'])} int8, {len(result['fp16_blobs'])} fp16/uint8)")
    print()
    padded_oc        = oc * 5 // 4
    expected_int8_p  = ic * padded_oc       # transposed+padded int8
    expected_int8    = oc * ic              # flat int8
    expected_fp16_p  = ic * padded_oc * 2  # transposed+padded fp16
    expected_fp16    = oc * ic * 2         # flat fp16
    expected_scale   = oc * 2             # fp16 per-channel scale
    for i, b in enumerate(result["blobs"]):
        tag = ""
        if b["byte_size"] == expected_int8_p:
            tag = f" ← weight INT8  [{ic}×{padded_oc}] transposed+padded"
        elif b["byte_size"] == expected_int8:
            tag = f" ← weight INT8  [{oc}×{ic}] flat"
        elif b["byte_size"] == expected_fp16_p:
            tag = f" ← weight FP16  [{ic}×{padded_oc}] transposed+padded"
        elif b["byte_size"] == expected_fp16:
            tag = f" ← weight FP16  [{oc}×{ic}] flat"
        elif b["byte_size"] == expected_scale:
            tag = " ← scale fp16 (per-channel)"
        print(f"  blob[{i:2d}]  dtype={b['dtype']:<12}  size={b['byte_size']:>10} bytes{tag}")
    print()
    print(f"  VERDICT: {result.get('verdict', 'unknown')}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    try:
        import mirror_sd_ane as ane
    except ImportError:
        print("ERROR: mirror_sd_ane not importable. Run 'maturin develop' in ane/ first.")
        raise SystemExit(1)

    # DIMS_8B (must match the hardcoded Rust constants):
    #   hidden=4096, intermediate=8192, n_heads=32, n_kv_heads=8, head_dim=128
    HIDDEN = 4096
    INTER  = 8192
    NH     = 32
    NKV    = 8
    HD     = 128

    import numpy as np

    def make_q8(oc, ic):
        """Zero int8 weights + unit fp16 scales — minimal allocation."""
        w = np.zeros((oc, ic), dtype=np.int8)
        scales = np.ones(oc, dtype=np.float16)
        return w.tobytes(), scales.tobytes(), oc, ic

    wq     = make_q8(NH  * HD, HIDDEN)   # 4096 x 4096
    wk     = make_q8(NKV * HD, HIDDEN)   # 1024 x 4096
    wv     = make_q8(NKV * HD, HIDDEN)   # 1024 x 4096
    wo     = make_q8(HIDDEN,   NH * HD)  # 4096 x 4096
    w_gate = make_q8(INTER,    HIDDEN)   # 8192 x 4096
    w_up   = make_q8(INTER,    HIDDEN)   # 8192 x 4096
    w_down = make_q8(HIDDEN,   INTER)    # 4096 x 8192

    layer_w = ane.Q8LayerWeights(wq, wk, wv, wo, w_gate, w_up, w_down)

    print("Compiling q8 kernel (constexpr_affine_dequantize path)...")
    t0 = time.time()
    t_before = t0 - 0.5   # half-second margin before compile

    kernels = ane.compile_dflash_kernels_q8(
        seq_q=64,
        ctx_len=64,
        softcap=0.0,
        layer_weights=[layer_w],
        is_27b=False,
    )
    print(f"  Compiled {sum(len(k) for k in kernels)} kernels in {time.time()-t0:.2f}s")

    # Give the filesystem a moment to flush cache
    time.sleep(0.1)

    cache_dirs = find_anec_cache_dirs(t_before)
    print(f"  Found {len(cache_dirs)} new ANEC cache entries")

    if not cache_dirs:
        print("\nNo cache entries found — ANECCompile may cache to a different path.")
        print("Try: find /var/folders -name 'data' -newer /tmp -ls 2>/dev/null | head")
        return

    # The weight blobs we care about are the FFN gate/up/down (largest, easiest to identify)
    # gate: INTER x HIDDEN = 8192 x 4096 = 33554432 elements
    gate_oc = INTER
    gate_ic = HIDDEN
    results = []
    for d in cache_dirs:
        r = analyze_cache_dir(d, gate_oc, gate_ic)
        if r:
            results.append(r)

    if not results:
        print("Could not parse any cache entries.")
        return

    # Show the most recent
    result = results[-1]
    print_report("constexpr_affine_dequantize (constexpr_quantized_weight_1x1)", result, gate_oc, gate_ic)

    # Summary
    verdicts = [r.get("verdict", "") for r in results]
    int8_count = sum(1 for v in verdicts if "INT8" in v)
    fp16_count = sum(1 for v in verdicts if "FP16" in v)
    print(f"\nSummary across {len(results)} cache entries:")
    print(f"  INT8 (bandwidth halved) : {int8_count}")
    print(f"  FP16 (no gain)          : {fp16_count}")

    if int8_count > fp16_count:
        print("\nCONCLUSION: constexpr_affine_dequantize keeps weights as INT8 in ANEC IR.")
        print("  => ANE weight bandwidth is ~halved vs fp16 path.")
        print("  => Expected FFN kernel speedup: ~2x (30ms -> ~15ms for 27B)")
    elif fp16_count > int8_count:
        print("\nCONCLUSION: ANECCompile expands int8 -> fp16 at compile time.")
        print("  => No bandwidth reduction. constexpr path provides no speed benefit.")
        print("  => q8 advantage is precision regularization only (weights in [-127,127]).")
    else:
        print("\nCONCLUSION: Inconclusive — inspect blobs above manually.")


if __name__ == "__main__":
    main()
