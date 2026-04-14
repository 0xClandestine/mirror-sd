import mlx.core as mx


class DFlashKVCache:
    """KV cache for the DFlash draft model.

    Unlike the standard KVCache, this supports:
    - Explicit position tracking (not offset-based) for correct RoPE on
      concatenated [context, noise] keys
    - Cropping to discard rejected tokens after verification (like the
      PyTorch reference's DynamicCache.crop)
    - Full bidirectional attention within the draft (non-causal)
    - Optional sliding window with sink to prevent unbounded memory growth

    The cache stores K/V from the verified prefix so that subsequent
    draft blocks can attend to already-accepted tokens.

    When sink_size and window_size are set, the cache retains the first
    sink_size positions (sink) and the most recent window_size positions,
    evicting middle positions when the cache exceeds
    sink_size + window_size. The offset tracks the total number of
    positions processed (for correct RoPE), independent of actual
    cache length.
    """

    def __init__(self, sink_size: int = 0, window_size: int = 0):
        self.keys = None
        self.values = None
        self.offset = 0
        self.sink_size = int(sink_size)
        self.window_size = int(window_size)

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            self.keys = mx.concatenate([self.keys, keys], axis=2)
            self.values = mx.concatenate([self.values, values], axis=2)
        self.offset += keys.shape[2]
        self._apply_window()
        return self.keys, self.values

    def _apply_window(self):
        if self.sink_size <= 0 or self.window_size <= 0:
            return
        if self.keys is None or self.values is None:
            return
        cache_len = int(self.keys.shape[2])
        max_len = self.sink_size + self.window_size
        if cache_len <= max_len:
            return
        sink_k = self.keys[:, :, :self.sink_size, :]
        sink_v = self.values[:, :, :self.sink_size, :]
        window_k = self.keys[:, :, -self.window_size:, :]
        window_v = self.values[:, :, -self.window_size:, :]
        self.keys = mx.concatenate([sink_k, window_k], axis=2)
        self.values = mx.concatenate([sink_v, window_v], axis=2)

    def trim(self, n: int):
        """Remove the last n positions from the cache.

        After each draft step, trim(n) removes the noise positions (the
        speculative draft tokens) while keeping the context positions from
        the verified prefix. This matches dflash-mlx's
        trim_draft_cache(cache, block_size).
        """
        if self.keys is not None and n > 0:
            new_length = max(self.keys.shape[2] - n, 0)
            self.keys = self.keys[..., :new_length, :]
            self.values = self.values[..., :new_length, :]
            self.offset -= n

    def state(self):
        if self.keys is None:
            return []
        return [self.keys, self.values]
