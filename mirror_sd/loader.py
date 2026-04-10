"""Weight loading and model conversion for DFlash on MLX.

Handles:
1. Loading DFlash draft model weights from HuggingFace safetensors
2. Converting PyTorch dtype tensors to MLX-compatible float16
3. Mapping HuggingFace weight names to MLX module structure
4. Auto-detecting configuration from HuggingFace config.json
"""

import json
from pathlib import Path
from typing import Optional, Tuple, Dict

import mlx.core as mx
import mlx.nn as nn

from .dflash import DFlashDraftModel, DFlashConfig


def load_dflash_model(
    model_path: str,
    config_overrides: Optional[Dict] = None,
    dtype: mx.Dtype = mx.float16,
) -> Tuple[DFlashDraftModel, DFlashConfig]:
    """Load a DFlash draft model from a HuggingFace model directory.

    Args:
        model_path: Path to local model directory or HuggingFace repo ID
        config_overrides: Optional config overrides
        dtype: Target dtype for model weights

    Returns:
        draft_model: DFlashDraftModel with loaded weights
        config: DFlashConfig used to create the model
    """
    from huggingface_hub import snapshot_download

    model_path = Path(model_path)
    if not model_path.exists():
        model_path = Path(snapshot_download(str(model_path)))

    config = _load_config(model_path, config_overrides)
    draft_model = DFlashDraftModel(config)

    weights = _load_safetensors(model_path, dtype)
    weights = draft_model.sanitize(weights)
    draft_model.load_weights(list(weights.items()))
    mx.eval(draft_model.parameters())

    return draft_model, config


def _load_config(model_path: Path, overrides: Optional[Dict] = None) -> DFlashConfig:
    """Load DFlashConfig from a HuggingFace config.json."""
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.json found at {config_path}")

    with open(config_path) as f:
        cfg = json.load(f)

    if overrides:
        cfg.update(overrides)

    return DFlashConfig.from_dict(cfg)


def _load_safetensors(
    model_path: Path,
    dtype: mx.Dtype = mx.float16,
) -> Dict[str, mx.array]:
    """Load weights from safetensors files using MLX native loader.

    Handles both single file and sharded (model-00001-of-000NN.safetensors) formats.
    Converts bfloat16 to float16 since MLX doesn't natively support bfloat16.
    """
    weights = {}

    # Try loading directly with mx.load first (handles single & sharded)
    single = model_path / "model.safetensors"
    index_path = model_path / "model.safetensors.index.json"

    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        weight_files = sorted(set(index["weight_map"].values()))
        for wf in weight_files:
            fpath = model_path / wf
            shard = mx.load(str(fpath))
            for k, v in shard.items():
                if v.dtype == mx.bfloat16:
                    v = v.astype(mx.float32).astype(dtype)
                elif v.dtype != dtype:
                    v = v.astype(dtype)
                weights[k] = v
    elif single.exists():
        loaded = mx.load(str(single))
        for k, v in loaded.items():
            if isinstance(v, mx.array):
                if v.dtype == mx.bfloat16:
                    v = v.astype(mx.float32).astype(dtype)
                elif v.dtype != dtype:
                    v = v.astype(dtype)
                weights[k] = v
    else:
        st_files = sorted(model_path.glob("*.safetensors"))
        for st_file in st_files:
            shard = mx.load(str(st_file))
            for k, v in shard.items():
                if isinstance(v, mx.array):
                    if v.dtype == mx.bfloat16:
                        v = v.astype(mx.float32).astype(dtype)
                    elif v.dtype != dtype:
                        v = v.astype(dtype)
                    weights[k] = v

    return weights


def convert_dflash_to_mlx(
    source_path: str,
    output_dir: str,
    dtype: mx.Dtype = mx.float16,
):
    """Convert a DFlash model from HuggingFace format to MLX format.

    Downloads the model from HuggingFace, converts weights to float16,
    and saves in MLX-compatible safetensors format.

    Args:
        source_path: HuggingFace repo ID (e.g., "z-lab/Qwen3-8B-DFlash-b16")
        output_dir: Local directory to save converted model
        dtype: Target dtype for conversion
    """
    from huggingface_hub import snapshot_download

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {source_path}...")
    local_path = Path(snapshot_download(source_path))

    config = _load_config(local_path)
    print(f"Config: hidden={config.hidden_size}, layers={config.num_hidden_layers}, "
          f"heads={config.num_attention_heads}, block_size={config.block_size}")

    weights = _load_safetensors(local_path, dtype)
    print(f"Loaded {len(weights)} weight tensors")

    mx.save_safetensors(str(output_dir / "weights.safetensors"), weights)

    cfg_dict = {
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "intermediate_size": config.intermediate_size,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
        "rms_norm_eps": config.rms_norm_eps,
        "rope_theta": config.rope_theta,
        "max_position_embeddings": config.max_position_embeddings,
        "vocab_size": config.vocab_size,
        "block_size": config.block_size,
        "mask_token_id": config.mask_token_id,
        "num_target_layers": config.num_target_layers,
        "target_layer_ids": config.target_layer_ids,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(cfg_dict, f, indent=2)

    print(f"Converted model saved to {output_dir}")
