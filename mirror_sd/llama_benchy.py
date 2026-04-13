"""Run llama-benchy benchmarks against spec and baseline servers.

Starts the DFlash spec server and (optionally) a baseline mlx_lm server,
then runs llama-benchy at varying context depths. Saves results with
metadata to benchmarks/ directory.

Usage:
    python -m mirror_sd.llama_benchy --model ~/.omlx/models/Qwen3.5-27B-4bit --draft z-lab/Qwen3.5-27B-DFlash --kod
    python -m mirror_sd.llama_benchy --model ~/.omlx/models/Qwen3.5-27B-4bit --draft z-lab/Qwen3.5-27B-DFlash --kod --baseline-only
    python -m mirror_sd.llama_benchy --model ~/.omlx/models/Qwen3.5-27B-4bit --draft z-lab/Qwen3.5-27B-DFlash --kod --spec-only
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time


BENCHMARKS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmarks")


def wait_for_server(url, timeout=120):
    import urllib.request
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        try:
            urllib.request.urlopen(f"{url}/v1/models", timeout=3)
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

    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode


def save_benchmark(raw_path, label, args, model_name, model_path, draft_path):
    if raw_path is None or not os.path.exists(raw_path):
        return

    with open(raw_path) as f:
        data = json.load(f)

    metadata = {
        "label": label,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "mode": "DFlash+KOD" if args.kod else ("DFlash+ADAPTIVE" if not args.no_adaptive else "DFlash"),
        "model": os.path.basename(model_path),
        "draft_model": os.path.basename(draft_path) if draft_path else None,
        "block_size": args.block_size,
        "quantize_draft": args.quantize_draft,
        "kod": args.kod,
        "adaptive": not args.no_adaptive,
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

    data["metadata"] = metadata

    os.makedirs(BENCHMARKS_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    safe_label = label.replace(" ", "_").replace("+", "").lower()
    out_name = f"{ts}_{safe_label}.json"
    out_path = os.path.join(BENCHMARKS_DIR, out_name)

    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved benchmark: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="llama-benchy benchmark runner for Mirror-SD")
    parser.add_argument("--model", type=str, required=True, help="Target model path")
    parser.add_argument("--draft", type=str, required=True, help="DFlash draft model path")
    parser.add_argument("--model-name", type=str, default=None, help="Model name for API")
    parser.add_argument("--kod", action="store_true", help="Kelly-Optimal Drafting")
    parser.add_argument("--no-adaptive", action="store_true", help="Disable adaptive block size")
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--quantize-draft", type=int, default=None, choices=[4, 8])
    parser.add_argument("--spec-port", type=int, default=8989, help="Port for spec server")
    parser.add_argument("--baseline-port", type=int, default=8990, help="Port for baseline server")
    parser.add_argument("--depth", type=int, nargs="+", default=[0, 512, 2048, 4096, 8192, 16384],
                        help="Context depths to benchmark")
    parser.add_argument("--pp", type=int, default=128, help="Prompt processing tokens")
    parser.add_argument("--tg", type=int, default=128, help="Token generation count")
    parser.add_argument("--runs", type=int, default=3, help="Runs per test")
    parser.add_argument("--latency-mode", type=str, default="generation", choices=["api", "generation", "none"])
    parser.add_argument("--spec-only", action="store_true", help="Only benchmark spec server (assume already running)")
    parser.add_argument("--baseline-only", action="store_true", help="Only benchmark baseline server")
    parser.add_argument("--no-baseline", action="store_true", help="Skip baseline benchmarking")
    parser.add_argument("--tokenizer", type=str, default=None, help="HuggingFace tokenizer name (defaults to model)")
    args = parser.parse_args()

    model_path = os.path.expanduser(args.model)
    model_name = args.model_name or os.path.basename(model_path)
    tokenizer = args.tokenizer or args.model

    spec_url = f"http://localhost:{args.spec_port}/v1"
    baseline_url = f"http://localhost:{args.baseline_port}/v1"

    procs = []

    def cleanup():
        for p in procs:
            try:
                p.terminate()
                p.wait(timeout=5)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass

    tmp_spec = os.path.join(BENCHMARKS_DIR, "_tmp_spec.json")
    tmp_baseline = os.path.join(BENCHMARKS_DIR, "_tmp_baseline.json")
    os.makedirs(BENCHMARKS_DIR, exist_ok=True)

    try:
        if not args.spec_only:
            spec_cmd = [
                sys.executable, "-m", "mirror_sd.server",
                "--model", model_path,
                "--draft", args.draft,
                "--port", str(args.spec_port),
                "--model-name", model_name,
                "--cache-size", "10",
            ]
            if args.kod:
                spec_cmd.append("--kod")
            if args.no_adaptive:
                spec_cmd.append("--no-adaptive")
            if args.block_size:
                spec_cmd.extend(["--block-size", str(args.block_size)])
            if args.quantize_draft:
                spec_cmd.extend(["--quantize-draft", str(args.quantize_draft)])

            print(f"Starting spec server: {' '.join(spec_cmd)}")
            spec_proc = subprocess.Popen(spec_cmd, stderr=subprocess.PIPE)
            procs.append(spec_proc)

            if not wait_for_server(spec_url, timeout=180):
                print("ERROR: Spec server failed to start")
                cleanup()
                sys.exit(1)
            print("Spec server ready.")

        if not args.spec_only and not args.no_baseline:
            baseline_cmd = [
                sys.executable, "-m", "mlx_lm.server",
                "--model", model_path,
                "--port", str(args.baseline_port),
            ]
            print(f"Starting baseline server: {' '.join(baseline_cmd)}")
            baseline_proc = subprocess.Popen(baseline_cmd, stderr=subprocess.PIPE)
            procs.append(baseline_proc)

            if not wait_for_server(baseline_url, timeout=180):
                print("ERROR: Baseline server failed to start")
                cleanup()
                sys.exit(1)
            print("Baseline server ready.")

        if not args.baseline_only:
            print("\n" + "=" * 60)
            print("  BENCHMARKING: DFlash Speculative Decoding")
            print("=" * 60)
            rc = run_llama_benchy(
                spec_url, model_name, args.depth, args.pp, args.tg, args.runs,
                tokenizer, tmp_spec, args.latency_mode,
            )
            if rc != 0:
                print(f"WARNING: llama-benchy exited with code {rc}")
            save_benchmark(tmp_spec, "dflash_spec", args, model_name, model_path, args.draft)

        if not args.spec_only and not args.no_baseline:
            print("\n" + "=" * 60)
            print("  BENCHMARKING: Baseline (autoregressive)")
            print("=" * 60)
            rc = run_llama_benchy(
                baseline_url, model_name, args.depth, args.pp, args.tg, args.runs,
                tokenizer, tmp_baseline, args.latency_mode,
            )
            if rc != 0:
                print(f"WARNING: llama-benchy exited with code {rc}")
            save_benchmark(tmp_baseline, "baseline", args, model_name, model_path, None)

    finally:
        cleanup()
        for f in [tmp_spec, tmp_baseline]:
            if os.path.exists(f):
                os.remove(f)


if __name__ == "__main__":
    main()
