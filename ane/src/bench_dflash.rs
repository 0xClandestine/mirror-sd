/// Benchmark: Rust private-API dispatch latency for Q8 DFlash kernels.
///
/// Mirrors the three kernels built by `demos/coreml_dflash.py` so latency can
/// be compared directly.  Uses `run_uncached` to match the production path in
/// `ane_model.py` (which calls run_uncached from the ANE background thread).
///
/// Run with:
///   cd ane && cargo test bench_dflash -- --ignored --nocapture
#[cfg(test)]
mod tests {
    use std::time::Instant;

    use ane::{NSQualityOfService, Shape, TensorData};

    use crate::dflash::{
        build_ffn_residual_kernel_lut4, build_ffn_residual_kernel_q8,
        build_mega_qkv_kernel_q8, build_o_proj_residual_kernel_q8,
        DFlashDims, Q8Weight, DIMS_8B,
    };

    const W_SQ:  usize = 64;   // align_width(32) = 64 (ANE minimum spatial width)
    const W_CTX: usize = 64;
    const N_WARMUP: usize = 20;
    const N_BENCH:  usize = 100;

    fn zero_q8(oc: usize, ic: usize) -> Q8Weight {
        Q8Weight {
            int8_data:  vec![0i8 as u8; oc * ic].into_boxed_slice(),
            scales_f16: vec![0x3C00u16; oc],  // fp16(1.0)
            oc,
            ic,
        }
    }

    fn bench<F: FnMut()>(label: &str, mut run: F) -> f64 {
        for _ in 0..N_WARMUP { run(); }
        let t0 = Instant::now();
        for _ in 0..N_BENCH { run(); }
        let ms = t0.elapsed().as_secs_f64() * 1e3 / N_BENCH as f64;
        println!("  {label:32} {ms:7.2} ms/call");
        ms
    }

    #[test]
    #[ignore = "benchmark; run with --ignored --nocapture"]
    fn bench_ffn_residual_q8() {
        let d = &DIMS_8B;
        let w_gate = zero_q8(d.intermediate, d.hidden);
        let w_up   = zero_q8(d.intermediate, d.hidden);
        let w_down = zero_q8(d.hidden,       d.intermediate);

        let g = build_ffn_residual_kernel_q8(d, W_SQ, 0.0, w_gate, w_up, w_down);
        let exe = g.compile(NSQualityOfService::Default).expect("compile");

        let h1      = TensorData::new(Shape { batch: 1, channels: d.hidden, height: 1, width: W_SQ });
        let norm_w  = TensorData::new(Shape { batch: 1, channels: d.hidden, height: 1, width: W_SQ });
        let out     = TensorData::new(Shape { batch: 1, channels: d.hidden, height: 1, width: W_SQ });

        println!("\nFFN residual Q8 (8B, w_sq={W_SQ}):");
        bench("run_uncached", || exe.run(&[&h1, &norm_w], &[&out]).unwrap());
        bench("run_cached",   || exe.run_cached(&[&h1, &norm_w], &[&out]).unwrap());
    }

    #[test]
    #[ignore = "benchmark; run with --ignored --nocapture"]
    fn bench_o_proj_residual_q8() {
        let d = &DIMS_8B;
        let wo = zero_q8(d.hidden, d.n_heads * d.head_dim);

        let g = build_o_proj_residual_kernel_q8(d, W_SQ, 0.0, wo);
        let exe = g.compile(NSQualityOfService::Default).expect("compile");

        let attn = TensorData::new(Shape { batch: 1, channels: d.n_heads * d.head_dim, height: 1, width: W_SQ });
        let h_in = TensorData::new(Shape { batch: 1, channels: d.hidden, height: 1, width: W_SQ });
        let out  = TensorData::new(Shape { batch: 1, channels: d.hidden, height: 1, width: W_SQ });

        println!("\no_proj residual Q8 (8B, w_sq={W_SQ}):");
        bench("run_uncached", || exe.run(&[&attn, &h_in], &[&out]).unwrap());
        bench("run_cached",   || exe.run_cached(&[&attn, &h_in], &[&out]).unwrap());
    }

    #[test]
    #[ignore = "benchmark; run with --ignored --nocapture"]
    fn bench_mega_qkv_q8() {
        let d = &DIMS_8B;
        let w_kv = W_CTX + W_SQ;

        let wq = zero_q8(d.n_heads    * d.head_dim, d.hidden);
        let wk = zero_q8(d.n_kv_heads * d.head_dim, d.hidden);
        let wv = zero_q8(d.n_kv_heads * d.head_dim, d.hidden);

        let g = build_mega_qkv_kernel_q8(d, W_SQ, W_CTX, wq, wk, wv);
        let exe = g.compile(NSQualityOfService::Default).expect("compile");

        let hidden    = TensorData::new(Shape { batch: 1, channels: d.hidden,        height: 1, width: W_SQ });
        let in_norm_w = TensorData::new(Shape { batch: 1, channels: d.hidden,        height: 1, width: W_SQ });
        let context   = TensorData::new(Shape { batch: 1, channels: d.hidden,        height: 1, width: W_CTX });
        let k_norm_w  = TensorData::new(Shape { batch: 1, channels: d.n_kv_heads,    height: d.head_dim, width: w_kv });
        let cos_k     = TensorData::new(Shape { batch: 1, channels: 1,               height: w_kv,       width: d.head_dim });
        let sin_k     = TensorData::new(Shape { batch: 1, channels: 1,               height: w_kv,       width: d.head_dim });
        let q_norm_w  = TensorData::new(Shape { batch: 1, channels: d.n_heads,       height: d.head_dim, width: W_SQ });
        let cos_q     = TensorData::new(Shape { batch: 1, channels: 1,               height: W_SQ,       width: d.head_dim });
        let sin_q     = TensorData::new(Shape { batch: 1, channels: 1,               height: W_SQ,       width: d.head_dim });

        let inputs = [&hidden, &in_norm_w, &context, &k_norm_w, &cos_k, &sin_k, &q_norm_w, &cos_q, &sin_q];

        // outputs: k_rope, v_4d_t, q_rope
        let k_out = TensorData::new(Shape { batch: 1, channels: d.n_kv_heads, height: d.head_dim, width: w_kv });
        let v_out = TensorData::new(Shape { batch: 1, channels: d.n_kv_heads, height: w_kv,       width: d.head_dim });
        let q_out = TensorData::new(Shape { batch: 1, channels: d.n_heads,    height: d.head_dim, width: W_SQ });
        let outputs = [&k_out, &v_out, &q_out];

        println!("\nmega_qkv Q8 (8B, w_sq={W_SQ}, w_ctx={W_CTX}):");
        bench("run_uncached", || exe.run(&inputs, &outputs).unwrap());
        bench("run_cached",   || exe.run_cached(&inputs, &outputs).unwrap());
    }

    #[test]
    #[ignore = "benchmark; run with --ignored --nocapture"]
    fn bench_all_dflash_q8() {
        println!("\n=== Rust private-API Q8 DFlash kernel latency (8B, w_sq={W_SQ}) ===");
        println!("(fp16 weights in IOSurface after CPU dequantize at graph-build time)\n");

        // run each sub-benchmark inline
        let d = &DIMS_8B;

        // FFN
        {
            let w_gate = zero_q8(d.intermediate, d.hidden);
            let w_up   = zero_q8(d.intermediate, d.hidden);
            let w_down = zero_q8(d.hidden,       d.intermediate);
            let g   = build_ffn_residual_kernel_q8(d, W_SQ, 0.0, w_gate, w_up, w_down);
            let exe = g.compile(NSQualityOfService::Default).expect("compile ffn");
            let h1     = TensorData::new(Shape { batch: 1, channels: d.hidden, height: 1, width: W_SQ });
            let norm_w = TensorData::new(Shape { batch: 1, channels: d.hidden, height: 1, width: W_SQ });
            let out    = TensorData::new(Shape { batch: 1, channels: d.hidden, height: 1, width: W_SQ });
            print!("  FFN residual    ");
            bench("run_uncached", || exe.run(&[&h1, &norm_w], &[&out]).unwrap());
        }

        // o_proj
        {
            let wo  = zero_q8(d.hidden, d.n_heads * d.head_dim);
            let g   = build_o_proj_residual_kernel_q8(d, W_SQ, 0.0, wo);
            let exe = g.compile(NSQualityOfService::Default).expect("compile oproj");
            let attn = TensorData::new(Shape { batch: 1, channels: d.n_heads * d.head_dim, height: 1, width: W_SQ });
            let h_in = TensorData::new(Shape { batch: 1, channels: d.hidden, height: 1, width: W_SQ });
            let out  = TensorData::new(Shape { batch: 1, channels: d.hidden, height: 1, width: W_SQ });
            print!("  o_proj residual ");
            bench("run_uncached", || exe.run(&[&attn, &h_in], &[&out]).unwrap());
        }

        // QKV
        {
            let w_kv = W_CTX + W_SQ;
            let wq   = zero_q8(d.n_heads    * d.head_dim, d.hidden);
            let wk   = zero_q8(d.n_kv_heads * d.head_dim, d.hidden);
            let wv   = zero_q8(d.n_kv_heads * d.head_dim, d.hidden);
            let g    = build_mega_qkv_kernel_q8(d, W_SQ, W_CTX, wq, wk, wv);
            let exe  = g.compile(NSQualityOfService::Default).expect("compile qkv");
            let hidden    = TensorData::new(Shape { batch: 1, channels: d.hidden,     height: 1, width: W_SQ });
            let in_norm_w = TensorData::new(Shape { batch: 1, channels: d.hidden,     height: 1, width: W_SQ });
            let context   = TensorData::new(Shape { batch: 1, channels: d.hidden,     height: 1, width: W_CTX });
            let k_norm_w  = TensorData::new(Shape { batch: 1, channels: d.n_kv_heads, height: d.head_dim, width: w_kv });
            let cos_k     = TensorData::new(Shape { batch: 1, channels: 1, height: w_kv, width: d.head_dim });
            let sin_k     = TensorData::new(Shape { batch: 1, channels: 1, height: w_kv, width: d.head_dim });
            let q_norm_w  = TensorData::new(Shape { batch: 1, channels: d.n_heads,    height: d.head_dim, width: W_SQ });
            let cos_q     = TensorData::new(Shape { batch: 1, channels: 1, height: W_SQ, width: d.head_dim });
            let sin_q     = TensorData::new(Shape { batch: 1, channels: 1, height: W_SQ, width: d.head_dim });
            let inputs = [&hidden, &in_norm_w, &context, &k_norm_w, &cos_k, &sin_k, &q_norm_w, &cos_q, &sin_q];
            let k_out = TensorData::new(Shape { batch: 1, channels: d.n_kv_heads, height: d.head_dim, width: w_kv });
            let v_out = TensorData::new(Shape { batch: 1, channels: d.n_kv_heads, height: w_kv, width: d.head_dim });
            let q_out = TensorData::new(Shape { batch: 1, channels: d.n_heads,    height: d.head_dim, width: W_SQ });
            print!("  mega_qkv        ");
            bench("run_uncached", || exe.run(&inputs, &[&k_out, &v_out, &q_out]).unwrap());
        }
    }

    /// Smoke test: build a tiny ffn_residual_lut4 kernel and check it compiles + runs.
    ///
    /// Uses tiny dimensions (hidden=64, intermediate=128) so compile is fast.
    ///
    /// If ANECCompile() rejects `constexpr_lut_to_dense` (as it does for
    /// `constexpr_affine_dequantize`), the error is printed and the test passes
    /// without panicking — this reports unsupported status cleanly.
    ///
    /// Run with:
    ///   cd ane && cargo test ffn_lut4_smoke -- --nocapture
    #[test]
    fn ffn_lut4_smoke() {
        const HIDDEN: usize = 64;
        const INTER:  usize = 128;
        const W_SQ:   usize = 64;

        let d = DFlashDims {
            hidden: HIDDEN, head_dim: 64, n_heads: 4, n_kv_heads: 2,
            intermediate: INTER, target_hidden: 5 * HIDDEN, gqa_ratio: 2,
        };

        // Tiny random-ish f32 weights (deterministic)
        let gate_f32: Vec<f32> = (0..INTER * HIDDEN).map(|i| ((i as f32) * 0.001 - 0.5)).collect();
        let up_f32:   Vec<f32> = (0..INTER * HIDDEN).map(|i| ((i as f32) * 0.002 - 1.0)).collect();
        let down_f32: Vec<f32> = (0..HIDDEN * INTER).map(|i| ((i as f32) * 0.003 - 0.7)).collect();

        let g = build_ffn_residual_kernel_lut4(&d, W_SQ, 0.0, &gate_f32, &up_f32, &down_f32);

        let exe = match g.compile(NSQualityOfService::Default) {
            Ok(exe) => exe,
            Err(e) => {
                println!("\nANECCompile() FAILED: {e}");
                println!("constexpr_lut_to_dense is NOT supported through the private API on this system.");
                println!("(This is expected — the same limitation affects constexpr_affine_dequantize.)");
                return;
            }
        };

        let h1     = TensorData::new(Shape { batch: 1, channels: HIDDEN, height: 1, width: W_SQ });
        let norm_w = TensorData::new(Shape { batch: 1, channels: HIDDEN, height: 1, width: W_SQ });
        let out    = TensorData::new(Shape { batch: 1, channels: HIDDEN, height: 1, width: W_SQ });

        exe.run(&[&h1, &norm_w], &[&out]).expect("run failed");
        println!("\nffn_lut4_smoke: compile + run succeeded.");
        println!("constexpr_lut_to_dense IS supported through the private API on this system.");
    }

    /// Benchmark LUT4 ffn_residual vs Q8 to measure actual latency difference.
    ///
    /// Run with:
    ///   cd ane && cargo test bench_ffn_lut4 -- --ignored --nocapture
    #[test]
    #[ignore = "benchmark; run with --ignored --nocapture"]
    fn bench_ffn_lut4() {
        let d = &DIMS_8B;

        let gate_f32: Vec<f32> = vec![0.0f32; d.intermediate * d.hidden];
        let up_f32:   Vec<f32> = vec![0.0f32; d.intermediate * d.hidden];
        let down_f32: Vec<f32> = vec![0.0f32; d.hidden * d.intermediate];

        let g = build_ffn_residual_kernel_lut4(d, W_SQ, 0.0, &gate_f32, &up_f32, &down_f32);
        let exe = match g.compile(NSQualityOfService::Default) {
            Ok(exe) => exe,
            Err(e) => {
                println!("\nANECCompile() FAILED: {e}");
                println!("constexpr_lut_to_dense not supported through private API — skipping benchmark.");
                return;
            }
        };

        let h1     = TensorData::new(Shape { batch: 1, channels: d.hidden, height: 1, width: W_SQ });
        let norm_w = TensorData::new(Shape { batch: 1, channels: d.hidden, height: 1, width: W_SQ });
        let out    = TensorData::new(Shape { batch: 1, channels: d.hidden, height: 1, width: W_SQ });

        println!("\nFFN residual LUT4 vs Q8 (8B, w_sq={W_SQ}):");
        println!("  LUT4:");
        bench("  run_uncached", || exe.run(&[&h1, &norm_w], &[&out]).unwrap());

        // Also run Q8 for comparison
        let w_gate = zero_q8(d.intermediate, d.hidden);
        let w_up   = zero_q8(d.intermediate, d.hidden);
        let w_down = zero_q8(d.hidden,       d.intermediate);
        let g_q8   = build_ffn_residual_kernel_q8(d, W_SQ, 0.0, w_gate, w_up, w_down);
        let exe_q8 = g_q8.compile(NSQualityOfService::Default).expect("q8 compile");
        println!("  Q8 (fp16 in IOSurface, for reference):");
        bench("  run_uncached", || exe_q8.run(&[&h1, &norm_w], &[&out]).unwrap());
    }
}
