"""OpenAI-compatible server wrapping DFlash speculative decoding.

Serves /v1/chat/completions (streaming + non-streaming) and /v1/models
so that llama-benchy can benchmark our speculative decoding pipeline.

Supports KV prompt cache persistence (like mlx_lm.server) so prefix-cached
benchmarking tools get correct decode-only tg measurements.

Usage:
    python -m mirror_sd.server --model Qwen/Qwen3-8B --draft z-lab/Qwen3-8B-DFlash-b16
    python -m mirror_sd.server --model ~/.omlx/models/Qwen3.5-27B-4bit --draft z-lab/Qwen3.5-27B-DFlash --kod
"""

import argparse
import copy
import json
import os
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

import mlx.core as mx

from mlx_lm import load as mlx_load
from mlx_lm.models.cache import LRUPromptCache, make_prompt_cache

from .loader import load_dflash_model
from .generate import spec_generate
from .prompt import get_stop_token_ids


class SpecServer:
    def __init__(self, target_model, draft_model, tokenizer, config, args):
        self.target_model = target_model
        self.draft_model = draft_model
        self.tokenizer = tokenizer
        self.config = config
        self.args = args
        self.model_name = args.model_name or os.path.basename(os.path.expanduser(args.model))
        self.model_key = args.model
        self.prompt_cache = LRUPromptCache(max_size=args.cache_size)

    def _log_cache(self):
        n = len(self.prompt_cache)
        nb = self.prompt_cache.nbytes
        print(f"Prompt Cache: {n} seqs, {nb / 1e9:.2f} GB", flush=True)

    def _fetch_cache(self, tokens):
        target_cache, rest = self.prompt_cache.fetch_nearest_cache(
            self.model_key, tokens
        )
        cache_count = len(tokens) - len(rest)
        return target_cache, rest, cache_count

    def _do_spec(self, tokens, max_tokens, temperature, stream_callback=None):
        eos_ids = get_stop_token_ids(self.tokenizer)

        target_cache, rest_tokens, cache_count = self._fetch_cache(tokens)
        self._log_cache()

        if target_cache is not None and len(rest_tokens) == 0:
            # Full cache hit — pass last prefill_step_size tokens so spec_generate
            # can capture target_hidden from capture_layers forward.
            # These tokens are already in the KV cache, so this re-prefills them,
            # but it's the only way to get target_hidden for the draft model.
            # After generation, LRUPromptCache will store the longer entry and
            # evict the old shorter one (since caches are trimmable).
            last_n = min(self.args.prefill_step_size, len(tokens))
            input_ids = mx.array(tokens[-last_n:])[None]
        elif target_cache is not None and len(rest_tokens) > 0:
            input_ids = mx.array(rest_tokens)[None]
        else:
            target_cache = None
            input_ids = mx.array(tokens)[None]

        output_ids, stats, final_cache, _, target_hidden = spec_generate(
            self.target_model, self.draft_model, input_ids,
            max_new_tokens=max_tokens, temperature=temperature,
            stop_token_ids=eos_ids,
            adaptive_block=not self.args.no_adaptive, kod=self.args.kod,
            stream_callback=stream_callback,
            prefill_step_size=self.args.prefill_step_size,
            prompt_cache=target_cache,
            lazy_logits=self.args.lazy_logits,
            logit_chunk_size=self.args.logit_chunk_size,
            compile_full=self.args.compile_full,
            compiled_whole=self.args.compiled_whole,
            turboquant_bits=self.args.turboquant_bits,
            auto_ar=self.args.auto_ar,
            auto_ar_threshold=self.args.auto_ar_threshold,
        )

        all_tokens = tokens + output_ids[0, len(tokens):].tolist()
        self.prompt_cache.insert_cache(
            self.model_key, all_tokens, final_cache
        )

        return output_ids, stats

    def _format_prompt(self, messages):
        kwargs = {"add_generation_prompt": True, "tokenize": False}
        if self.args.no_think:
            try:
                kwargs["enable_thinking"] = False
            except TypeError:
                pass
        return self.tokenizer.apply_chat_template(messages, **kwargs)

    def generate_streaming(self, messages, max_tokens=128, temperature=0.0, write_fn=None):
        prompt = self._format_prompt(messages)
        tokens = self.tokenizer.encode(prompt)

        request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        created = int(time.time())

        def _write_chunk(delta, finish_reason=None, usage=None):
            chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": self.model_name,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
            }
            if usage is not None:
                chunk["usage"] = usage
            write_fn(chunk)

        _write_chunk({"role": "assistant"})

        def on_token(tok_id):
            _write_chunk({"content": self.tokenizer.decode([tok_id], skip_special_tokens=True)})

        output_ids, stats = self._do_spec(tokens, max_tokens, temperature, stream_callback=on_token)

        gen_count = output_ids.shape[1] - len(tokens)
        _write_chunk({}, finish_reason="stop", usage={
            "prompt_tokens": len(tokens),
            "completion_tokens": gen_count,
            "total_tokens": len(tokens) + gen_count,
        })

    def generate(self, messages, max_tokens=128, temperature=0.0):
        prompt = self._format_prompt(messages)
        tokens = self.tokenizer.encode(prompt)

        output_ids, stats = self._do_spec(tokens, max_tokens, temperature)

        gen_tokens = output_ids[0, len(tokens):].tolist()
        text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop" if len(gen_tokens) < max_tokens else "length",
            }],
            "usage": {
                "prompt_tokens": len(tokens),
                "completion_tokens": len(gen_tokens),
                "total_tokens": len(tokens) + len(gen_tokens),
            },
        }


class Handler(BaseHTTPRequestHandler):
    server_instance: Optional[SpecServer] = None

    def log_message(self, format, *args):
        import sys
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] {args[0]}", file=sys.stderr)

    def _set_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "*")
        self.send_header("Access-Control-Allow-Headers", "*")

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Content-type", "application/json")
        self._set_cors_headers()
        self.end_headers()

    def do_GET(self):
        if self.path in ("/v1/models", "/models"):
            self._handle_models()
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path in ("/v1/chat/completions", "/chat/completions"):
            self._handle_chat_completions()
        else:
            self.send_response(404)
            self.end_headers()

    def _handle_models(self):
        srv = Handler.server_instance
        body = {
            "object": "list",
            "data": [{
                "id": srv.model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "mirror-sd",
            }],
        }
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self._set_cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps(body).encode())

    def _handle_chat_completions(self):
        srv = Handler.server_instance
        content_length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(content_length)
        body = json.loads(raw.decode())

        messages = body.get("messages", [])
        max_tokens = body.get("max_tokens", body.get("max_completion_tokens", 128))
        temperature = body.get("temperature", 0.0)
        stream = body.get("stream", False)

        if srv.args.no_think:
            has_system = any(m.get("role") == "system" for m in messages)
            if not has_system:
                messages = [{"role": "system", "content": "/no_think"}] + messages

        if stream:
            self.send_response(200)
            self.send_header("Content-type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self._set_cors_headers()
            self.end_headers()

            def write_sse(chunk):
                data = json.dumps(chunk)
                self.wfile.write(f"data: {data}\n\n".encode())
                self.wfile.flush()

            srv.generate_streaming(messages, max_tokens, temperature, write_fn=write_sse)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        else:
            result = srv.generate(messages, max_tokens, temperature)
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self._set_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())


def main():
    parser = argparse.ArgumentParser(description="OpenAI-compatible server for DFlash speculative decoding")
    parser.add_argument("--model", type=str, required=True, help="Target model path")
    parser.add_argument("--draft", type=str, required=True, help="DFlash draft model path")
    parser.add_argument("--model-name", type=str, default=None, help="Model name for API (defaults to --model)")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8989)
    parser.add_argument("--kod", action="store_true", help="Kelly-Optimal Drafting")
    parser.add_argument("--no-adaptive", action="store_true", help="Disable adaptive block size")
    parser.add_argument("--quantize-draft", type=int, default=None, choices=[4, 8])
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--prefill-step-size", type=int, default=512, help="Chunk size for prompt prefill")
    parser.add_argument("--cache-size", type=int, default=10, help="Max prompt cache entries")
    parser.add_argument("--no-think", action="store_true", help="Inject /no_think system prompt for Qwen3 thinking mode")
    parser.add_argument("--lazy-logits", action="store_true", help="Use lazy logits: compute lm_head in chunks, stopping at rejection")
    parser.add_argument("--logit-chunk-size", type=int, default=1, help="Chunk size for lazy logits (1=token-by-token)")
    parser.add_argument("--compile-full", action="store_true", help="Use mx.compile for full-attention layers during verify")
    parser.add_argument("--compiled-whole", action="store_true", help="Use mx.compile for entire 64-layer verify pass")
    parser.add_argument("--turboquant-bits", type=float, default=0.0, help="Enable TurboQuant KV cache at this bit-width (e.g. 2.5, 3.5)")
    parser.add_argument("--auto-ar", action="store_true", help="Auto fallback to AR when acceptance rate is below breakeven")
    parser.add_argument("--auto-ar-threshold", type=float, default=0.35, help="Auto-AR per-token acceptance threshold (default: 0.35)")
    args = parser.parse_args()

    model_path = os.path.expanduser(args.model)
    print(f"Loading target: {model_path}")
    target_model, tokenizer = mlx_load(model_path)
    print(f"Loading draft:  {args.draft}")
    draft_model, config = load_dflash_model(args.draft, quantize=args.quantize_draft)
    if args.block_size is not None:
        config.block_size = args.block_size
        draft_model.block_size = args.block_size

    srv = SpecServer(target_model, draft_model, tokenizer, config, args)
    Handler.server_instance = srv

    ThreadingHTTPServer.allow_reuse_address = True
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    mode = "DFlash+KOD" if args.kod else ("DFlash+ADAPTIVE" if not args.no_adaptive else "DFlash")
    print(f"Serving {mode} (block_size={config.block_size}) on http://{args.host}:{args.port}")
    print(f"  model: {srv.model_name}")
    print(f"  prompt cache: {args.cache_size} entries")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        httpd.shutdown()


if __name__ == "__main__":
    main()
