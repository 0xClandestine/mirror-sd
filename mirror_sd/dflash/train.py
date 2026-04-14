"""Train a DFlash draft model for Apple Silicon."""

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from .model import DFlashConfig, DFlashDraftModel, build_target_layer_ids


class DFlashTrainModel(DFlashDraftModel):
    def forward_train(
        self,
        input_ids: mx.array,
        target_hidden: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        noise_embedding = self.embed_tokens(input_ids)
        hidden = self.hidden_norm(self.fc(target_hidden))
        for layer in self.layers[:self.num_draft_layers]:
            hidden_states = layer(
                hidden_states=noise_embedding,
                target_hidden=hidden,
                mask=mask,
                cache=None,
            )
            noise_embedding = hidden_states
        return self.norm(noise_embedding)

    def compute_loss(
        self,
        input_ids: mx.array,
        target_ids: mx.array,
        target_hidden: mx.array,
        loss_weights: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        hidden = self.forward_train(input_ids, target_hidden, mask)
        logits = self.lm_head(hidden)

        log_probs = mx.log(mx.softmax(logits, axis=-1) + 1e-12)
        target_log_probs = mx.take_along_axis(
            log_probs,
            target_ids[:, :, None],
            axis=-1,
        ).squeeze(-1)

        weighted = target_log_probs * loss_weights
        n_tokens = mx.sum(loss_weights) + 1e-12
        return -mx.sum(weighted) / n_tokens


def prepare_data(args):
    from mlx_lm import load as mlx_load
    from mlx_lm.models import cache as cache_module
    from ..target import forward_with_hidden_states, extract_context_feature

    print(f"Loading target model: {args.model}")
    target_model, tokenizer = mlx_load(args.model)

    target_layer_ids = build_target_layer_ids(
        getattr(target_model, 'n_layers', 36), 5
    )
    print(f"Target layer IDs: {target_layer_ids}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts = _load_prompts(args)
    print(f"Processing {len(prompts)} prompts...")

    inner = target_model.model
    n_layers = len(inner.layers)

    for idx, prompt in enumerate(prompts):
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array(tokens)[None]

        cache = cache_module.make_prompt_cache(target_model)
        logits = target_model(input_ids, cache=cache)
        mx.eval(logits)
        mx.eval([c.state for c in cache])

        generated = [int(mx.argmax(logits[:, -1:, :], axis=-1)[0, 0])]
        for _ in range(args.max_response_length - 1):
            token_input = mx.array([[generated[-1]]])
            logits = target_model(token_input, cache=cache)
            mx.eval(logits)
            next_token = int(mx.argmax(logits[:, -1:, :], axis=-1)[0, 0])
            generated.append(next_token)
            if next_token == tokenizer.eos_token_id:
                break

        full_ids = mx.array([tokens + generated], dtype=mx.int32)

        from mlx_lm.models import cache as cache_module_inner
        fresh_cache = cache_module_inner.make_prompt_cache(target_model)
        h = inner.embed_tokens(full_ids)
        captured = {}
        for i in range(n_layers):
            if i in target_layer_ids:
                captured[i] = h
            h = inner.layers[i](h, None, cache=fresh_cache[i])
        mx.eval(h)

        selected = [captured[lid] for lid in target_layer_ids]
        target_hidden = mx.concatenate(selected, axis=-1)
        mx.eval(target_hidden)

        out_path = output_dir / f"sample_{idx:06d}.npz"
        mx.savez(
            str(out_path),
            input_ids=full_ids,
            target_hidden=target_hidden,
            prompt_len=mx.array([len(tokens)]),
        )

        if (idx + 1) % 100 == 0:
            print(f"  {idx + 1}/{len(prompts)} samples saved")

    meta = {
        "model": args.model,
        "target_layer_ids": target_layer_ids,
        "n_samples": len(prompts),
        "max_response_length": args.max_response_length,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Done. {len(prompts)} samples saved to {output_dir}")


def _load_prompts(args) -> List[str]:
    if args.data == "demo":
        return [
            "What is 15% of 200?",
            "Write a Python function to compute Fibonacci numbers:",
            "Explain quantum entanglement in simple terms:",
            "Solve for x: 3x + 7 = 22",
            "What is the capital of France?",
            "How does photosynthesis work?",
            "Implement binary search in Python:",
            "What is the meaning of life?",
            "Explain the theory of relativity:",
            "Write a function to check if a string is a palindrome:",
        ]
    raise NotImplementedError(f"Data source '{args.data}' not yet supported. Use --data demo")


@dataclass
class TrainConfig:
    num_layers: int = 3
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    block_size: int = 8
    max_seq_length: int = 3072
    anchors_per_seq: int = 512
    loss_gamma: float = 4.0
    learning_rate: float = 6e-4
    warmup_ratio: float = 0.04
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    epochs: int = 6
    batch_size: int = 1
    seed: int = 42
    use_oput: bool = False
    oput_ratio: float = 0.75


def build_training_mask(
    seq_len: int,
    block_starts: List[int],
    block_size: int,
    ctx_len: int,
) -> Optional[mx.array]:
    n_blocks = len(block_starts)
    if n_blocks == 0:
        return None

    total_len = ctx_len + seq_len
    mask = mx.full((1, 1, seq_len, total_len), -1e9, dtype=mx.float32)

    for i, start in enumerate(block_starts):
        end = min(start + block_size, seq_len)
        for q_pos in range(start, end):
            for k_pos in range(ctx_len):
                mask[0, 0, q_pos, k_pos] = 0.0
            for k_pos in range(start, end):
                mask[0, 0, q_pos, ctx_len + k_pos] = 0.0

    return mask


def build_loss_weights(
    seq_len: int,
    block_starts: List[int],
    block_size: int,
    gamma: float,
) -> mx.array:
    weights = mx.zeros((1, seq_len), dtype=mx.float32)
    for start in block_starts:
        end = min(start + block_size, seq_len)
        for k in range(start + 1, end):
            pos_in_block = k - start
            w = math.exp(-(pos_in_block - 1) / gamma)
            weights[0, k] = w
    return weights


def train(args):
    from mlx_lm import load as mlx_load

    cfg = TrainConfig(
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads if not args.no_gqa else args.num_attention_heads,
        block_size=args.block_size,
        learning_rate=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        loss_gamma=args.loss_gamma,
        use_oput=args.oput,
    )

    print(f"Loading target model: {args.model}")
    target_model, tokenizer = mlx_load(args.model)

    target_layer_ids = build_target_layer_ids(
        getattr(target_model, 'n_layers', 36), cfg.num_layers
    )

    mask_token_id = _get_mask_token_id(tokenizer)

    dflash_config = DFlashConfig(
        hidden_size=cfg.hidden_size,
        num_hidden_layers=cfg.num_layers,
        intermediate_size=cfg.intermediate_size,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        head_dim=128,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        max_position_embeddings=cfg.max_seq_length,
        vocab_size=len(tokenizer.get_vocab()),
        block_size=cfg.block_size,
        mask_token_id=mask_token_id,
        target_layer_ids=target_layer_ids,
        num_target_layers=getattr(target_model, 'n_layers', 36),
    )

    draft_model = DFlashTrainModel(dflash_config)

    draft_model.embed_tokens = target_model.model.embed_tokens
    draft_model.lm_head = target_model.lm_head

    trainable_keys = set()
    for name, _ in tree_flatten(draft_model.parameters()):
        if any(k in name for k in ['layers', 'fc', 'hidden_norm', 'norm']):
            trainable_keys.add(name)

    n_trainable = sum(
        v.size
        for k, v in tree_flatten(draft_model.parameters())
        if hasattr(v, 'size') and any(tk in k for tk in ['layers', 'fc', 'hidden_norm', 'norm'])
    )
    print(f"Trainable params: {n_trainable / 1e6:.1f}M")
    print(f"Config: {cfg.num_layers}L, hidden={cfg.hidden_size}, block_size={cfg.block_size}")

    data_dir = Path(args.train_data)
    samples = sorted(data_dir.glob("sample_*.npz"))
    if not samples:
        print(f"No training data found in {data_dir}")
        return
    print(f"Training data: {len(samples)} samples")

    total_steps = cfg.epochs * len(samples) // cfg.batch_size
    warmup_steps = int(total_steps * cfg.warmup_ratio)

    lr_schedule = mx.concatenate([
        mx.linspace(0, cfg.learning_rate, warmup_steps),
        cfg.learning_rate * 0.5 * (1 + mx.cos(mx.arange(total_steps - warmup_steps) * math.pi / (total_steps - warmup_steps))),
    ])

    optimizer = nn.optimizers.AdamW(learning_rate=cfg.learning_rate, weight_decay=cfg.weight_decay)

    def train_step(model, input_ids, target_ids, target_hidden, loss_weights, attn_mask):
        def loss_fn(model):
            return model.compute_loss(
                input_ids, target_ids, target_hidden, loss_weights, attn_mask
            )
        loss, grads = nn.value_and_grad(model, loss_fn)(model)
        grads, norm = nn.optimizers.clip_grad_norm(grads, cfg.grad_clip)
        optimizer.update(model, grads)
        return loss, norm

    key = mx.random.key(cfg.seed)
    step = 0
    t0 = time.perf_counter()

    for epoch in range(cfg.epochs):
        key, subkey = mx.random.split(key)
        perm = mx.random.permutation(subkey, len(samples))
        epoch_loss = 0.0
        n_steps = 0

        for si in range(0, len(samples), cfg.batch_size):
            data = mx.load(str(samples[int(perm[si])]))
            full_ids = data['input_ids']
            target_hidden_all = data['target_hidden']
            prompt_len = int(data['prompt_len'])

            seq_len = full_ids.shape[1]
            response_len = seq_len - prompt_len

            if response_len < cfg.block_size + 1:
                continue

            n_anchors = min(cfg.anchors_per_seq, response_len // cfg.block_size)
            key, subkey = mx.random.split(key)
            anchor_offsets = mx.argsort(mx.random.uniform(subkey, (response_len,)))[:n_anchors]
            block_starts = [prompt_len + int(a) for a in anchor_offsets]

            input_ids = mx.array(full_ids)
            for start in block_starts:
                end = min(start + cfg.block_size, seq_len)
                for pos in range(start + 1, end):
                    input_ids[0, pos] = mask_token_id

            target_ids = mx.array(full_ids)

            loss_weights = build_loss_weights(seq_len, block_starts, cfg.block_size, cfg.loss_gamma)

            attn_mask = build_training_mask(seq_len, block_starts, cfg.block_size, target_hidden_all.shape[1])

            lr = float(lr_schedule[min(step, len(lr_schedule) - 1)])
            optimizer.learning_rate = lr

            loss, grad_norm = train_step(
                draft_model, input_ids, target_ids,
                target_hidden_all, loss_weights, attn_mask,
            )
            mx.eval(loss, grad_norm)

            epoch_loss += float(loss)
            n_steps += 1
            step += 1

            if step % 10 == 0:
                elapsed = time.perf_counter() - t0
                print(f"  step {step:5d} | loss={float(loss):.4f} | grad_norm={float(grad_norm):.2f} | lr={lr:.2e} | {elapsed:.1f}s")

            if cfg.use_oput and step % 2 == 0:
                draft_out = draft_model.forward_train(input_ids, target_hidden_all, attn_mask)
                draft_logits = draft_model.lm_head(draft_out)
                mx.eval(draft_logits)

                predicted_ids = mx.argmax(draft_logits, axis=-1)
                on_policy_ids = mx.array(full_ids)
                for start in block_starts:
                    end = min(start + cfg.block_size, seq_len)
                    for pos in range(start + 1, end):
                        on_policy_ids[0, pos] = int(predicted_ids[0, pos])

                oput_loss, oput_grads = nn.value_and_grad(draft_model, lambda m: m.compute_loss(
                    on_policy_ids, target_ids, target_hidden_all, loss_weights, attn_mask
                ))(draft_model)
                mx.eval(oput_loss)
                oput_grads, _ = nn.optimizers.clip_grad_norm(oput_grads, cfg.grad_clip)
                optimizer.update(draft_model, oput_grads)

        avg_loss = epoch_loss / max(n_steps, 1)
        print(f"Epoch {epoch + 1}/{cfg.epochs}: avg_loss={avg_loss:.4f}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    weights = {}
    for k, v in tree_flatten(draft_model.parameters()):
        if hasattr(v, 'shape'):
            weights[k] = v

    mx.savez(str(output_dir / "weights.npz"), **weights)

    config_dict = {
        "hidden_size": cfg.hidden_size,
        "num_hidden_layers": cfg.num_layers,
        "intermediate_size": cfg.intermediate_size,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_key_value_heads,
        "head_dim": 128,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "max_position_embeddings": cfg.max_seq_length,
        "vocab_size": dflash_config.vocab_size,
        "block_size": cfg.block_size,
        "mask_token_id": mask_token_id,
        "target_layer_ids": target_layer_ids,
        "num_target_layers": dflash_config.num_target_layers,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    print(f"Saved to {output_dir}")


def _get_mask_token_id(tokenizer) -> int:
    vocab = tokenizer.get_vocab()
    for candidate in ["<|mask|>", "<mask>", " MASK "]:
        if candidate in vocab:
            return vocab[candidate]
    return vocab.get("<|mask|>", 151669)


def main():
    parser = argparse.ArgumentParser(description="Train DFlash draft model for Apple Silicon")
    subparsers = parser.add_subparsers(dest="command")

    prep = subparsers.add_parser("prepare-data", help="Extract target features from training data")
    prep.add_argument("--model", type=str, required=True)
    prep.add_argument("--data", type=str, default="demo")
    prep.add_argument("--output", type=str, required=True)
    prep.add_argument("--max-response-length", type=int, default=2048)

    trn = subparsers.add_parser("train", help="Train draft model")
    trn.add_argument("--model", type=str, required=True)
    trn.add_argument("--train-data", type=str, required=True)
    trn.add_argument("--output", type=str, required=True)
    trn.add_argument("--num-layers", type=int, default=3)
    trn.add_argument("--hidden-size", type=int, default=4096)
    trn.add_argument("--intermediate-size", type=int, default=12288)
    trn.add_argument("--num-attention-heads", type=int, default=32)
    trn.add_argument("--num-key-value-heads", type=int, default=8)
    trn.add_argument("--block-size", type=int, default=8)
    trn.add_argument("--no-gqa", action="store_true", help="Use MHA instead of GQA (ANE-friendly)")
    trn.add_argument("--lr", type=float, default=6e-4)
    trn.add_argument("--epochs", type=int, default=6)
    trn.add_argument("--batch-size", type=int, default=1)
    trn.add_argument("--loss-gamma", type=float, default=4.0, help="Loss decay gamma (4 for b8, 7 for b16)")
    trn.add_argument("--oput", action="store_true", help="Enable OPUT on-policy training from DMax")

    args = parser.parse_args()
    if args.command == "prepare-data":
        prepare_data(args)
    elif args.command == "train":
        train(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()