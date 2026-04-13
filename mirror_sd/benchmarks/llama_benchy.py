"""Run llama-benchy benchmarks against spec and baseline servers.

Starts servers, runs llama-benchy at varying context depths, kills servers,
moves to next config. One command runs everything — start and walk away.

Usage:
    python -m mirror_sd.benchmarks.llama_benchy \\
      --model ~/.omlx/models/Qwen3.5-27B-4bit \\
      --draft z-lab/Qwen3.5-27B-DFlash

    # Just baseline + one spec config
    python -m mirror_sd.benchmarks.llama_benchy \\
      --model ~/.omlx/models/Qwen3.5-27B-4bit \\
      --draft z-lab/Qwen3.5-27B-DFlash \\
      --spec-only --kod
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import urllib.request


BENCHMARKS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

SPEC_CONFIGS = [
    {"label": "dflash", "kod": False},
    {"label": "dflash_kod", "kod": True},
]


def wait_for_server(url, proc, timeout=60):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        if proc.poll() is not None:
            return False
        try:
            urllib.request.urlopen(f"{url}/v1/models", timeout=2)
            return True
        except Exception:
            time.sleep(1)
    return False


def run_llama_benchy(base_url, model_name, depths, pp, tg, runs, tokenizer, save_result, latency_mode="generation"):
    cmd = [
        sys.executable, "-m", "llama_benchy",
        "--base-url", base_url,
        "--model", model_name,
        "--pp", str(pp),
        "--tg", str(tg),
        "--depth", *[str(d) for d in depths],
        "--runs", str(runs),
        "--latency-mode", latency_mode,
        "--enable-prefix-caching",
    ]
    if tokenizer:
        cmd.extend(["--tokenizer", tokenizer])
    if save_result:
        cmd.extend(["--save-result", save_result, "--format", "json"])

    print(f"\n  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode


def save_benchmark(raw_path, label, args, model_name, model_path, draft_path, cfg=None):
    if raw_path is None or not os.path.exists(raw_path):
        return

    with open(raw_path) as f:
        data = json.load(f)

    metadata = {
        "label": label,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": os.path.basename(model_path),
        "draft_model": os.path.basename(draft_path) if draft_path else None,
        "block_size": args.block_size,
        "quantize_draft": args.quantize_draft,
        "depths": args.depth,
        "pp": args.pp,
        "tg": args.tg,
        "runs": args.runs,
        "latency_mode": args.latency_mode,
        "platform": {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "system": platform.system(),
            "python": platform.python_version(),
        },
    }
    if cfg:
        metadata["kod"] = cfg.get("kod", False)
        metadata["auto_ar"] = cfg.get("auto_ar", False)

    data["metadata"] = metadata

    os.makedirs(BENCHMARKS_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    safe_label = label.replace(" ", "_").replace("+", "").lower()
    out_name = f"{ts}_{safe_label}.json"
    out_path = os.path.join(BENCHMARKS_DIR, out_name)

    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"  Saved: {out_path}")
    return out_path


def build_spec_cmd(model_path, draft_path, port, model_name, args, cfg):
    cmd = [
        sys.executable, "-m", "mirror_sd.server",
        "--model", model_path,
        "--draft", draft_path,
        "--port", str(port),
        "--model-name", model_name,
        "--cache-size", "10",
    ]
    if cfg.get("kod"):
        cmd.append("--kod")
    if args.no_adaptive:
        cmd.append("--no-adaptive")
    if args.block_size:
        cmd.extend(["--block-size", str(args.block_size)])
    if args.quantize_draft:
        cmd.extend(["--quantize-draft", str(args.quantize_draft)])
    if args.turboquant_bits > 0:
        cmd.extend(["--turboquant-bits", str(args.turboquant_bits)])
    return cmd


def main():
    parser = argparse.ArgumentParser(description="llama-benchy benchmark runner for Mirror-SD")
    parser.add_argument("--model", type=str, required=True, help="Target model path")
    parser.add_argument("--draft", type=str, required=True, help="DFlash draft model path")
    parser.add_argument("--model-name", type=str, default=None, help="Model name for API")
    parser.add_argument("--no-adaptive", action="store_true", help="Disable adaptive block size")
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--quantize-draft", type=int, default=None, choices=[4, 8])
    parser.add_argument("--spec-port", type=int, default=8989, help="Port for spec server")
    parser.add_argument("--baseline-port", type=int, default=8990, help="Port for baseline server")
    parser.add_argument("--depth", type=int, nargs="+", default=[0, 512, 2048, 4096, 8192, 16384],
                        help="Context depths to benchmark")
    parser.add_argument("--pp", type=int, default=128, help="Prompt processing tokens")
    parser.add_argument("--tg", type=int, default=512, help="Token generation count")
    parser.add_argument("--runs", type=int, default=3, help="Runs per test")
    parser.add_argument("--latency-mode", type=str, default="generation", choices=["api", "generation", "none"])
    parser.add_argument("--spec-only", action="store_true", help="Only benchmark one spec config (use --kod/--auto-ar)")
    parser.add_argument("--baseline-only", action="store_true", help="Only benchmark baseline")
    parser.add_argument("--no-baseline", action="store_true", help="Skip baseline benchmarking")
    parser.add_argument("--kod", action="store_true", help="Enable KOD (only with --spec-only)")
    parser.add_argument("--tokenizer", type=str, default=None, help="HuggingFace tokenizer name")
    parser.add_argument("--turboquant-bits", type=float, default=0.0, help="TurboQuant KV cache bit-width")
    args = parser.parse_args()

    model_path = os.path.expanduser(args.model)
    model_name = args.model_name or os.path.basename(model_path)
    tokenizer = args.tokenizer or args.model

    spec_url = f"http://localhost:{args.spec_port}"
    baseline_url = f"http://localhost:{args.baseline_port}"

    results = []

    def start_server(cmd, url, label):
        print(f"\n  Starting {label} server...")
        proc = subprocess.Popen(cmd)
        if not wait_for_server(url, proc):
            if proc.poll() is not None:
                print(f"ERROR: {label} server crashed (exit code {proc.returncode})")
            else:
                print(f"ERROR: {label} server did not respond within 60s")
                proc.kill()
            sys.exit(1)
        print(f"  {label} server ready.")
        return proc

    def stop_server(proc, label):
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        print(f"  {label} server stopped.")

    os.makedirs(BENCHMARKS_DIR, exist_ok=True)

    # --- Baseline ---
    if not args.no_baseline and not args.spec_only:
        tmp = os.path.join(BENCHMARKS_DIR, "_tmp_baseline.json")
        try:
            baseline_cmd = [
                sys.executable, "-m", "mlx_lm", "server",
                "--model", model_path,
                "--port", str(args.baseline_port),
            ]
            proc = start_server(baseline_cmd, baseline_url, "Baseline")

            print(f"\n{'='*60}")
            print(f"  BENCHMARKING: Baseline (autoregressive)")
            print(f"{'='*60}")
            rc = run_llama_benchy(
                baseline_url, model_name, args.depth, args.pp, args.tg, args.runs,
                tokenizer, tmp, args.latency_mode,
            )
            if rc != 0:
                print(f"  WARNING: llama-benchy exited with code {rc}")
            path = save_benchmark(tmp, "baseline", args, model_name, model_path, None)
            if path:
                results.append(("baseline", path))

            stop_server(proc, "Baseline")
            time.sleep(5)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    # --- Spec configs ---
    if not args.baseline_only:
        if args.spec_only:
            configs = [{"label": "dflash_spec", "kod": args.kod}]
        else:
            configs = SPEC_CONFIGS

        for cfg in configs:
            tmp = os.path.join(BENCHMARKS_DIR, f"_tmp_{cfg['label']}.json")
            try:
                cmd = build_spec_cmd(model_path, args.draft, args.spec_port, model_name, args, cfg)
                proc = start_server(cmd, spec_url, cfg["label"])

                print(f"\n{'='*60}")
                print(f"  BENCHMARKING: {cfg['label']}")
                print(f"{'='*60}")
                rc = run_llama_benchy(
                    spec_url, model_name, args.depth, args.pp, args.tg, args.runs,
                    tokenizer, tmp, args.latency_mode,
                )
                if rc != 0:
                    print(f"  WARNING: llama-benchy exited with code {rc}")
                path = save_benchmark(tmp, cfg["label"], args, model_name, model_path, args.draft, cfg)
                if path:
                    results.append((cfg["label"], path))

                stop_server(proc, cfg["label"])
                time.sleep(5)
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)

    # --- Summary ---
    if len(results) >= 2:
        print(f"\n{'='*60}")
        print(f"  RESULTS SUMMARY")
        print(f"{'='*60}")
        for label, path in results:
            print(f"  {label}: {path}")
        print(f"\n  Compare with:")
        print(f"  python -m mirror_sd.benchmarks.view {' '.join(p for _, p in results)} --chart comparison.png")


if __name__ == "__main__":
    main()
