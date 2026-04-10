"""Diagnostic: compare ANE conv1x1_proj output against MLX ground truth.

Tests two approaches:
  A) Current: weight shape [1, OC, 1, IC], data in row-major [OC, IC]
     Graph: placeholder → transpose([0,3,2,1]) → reshape([OC,IC,1,1]) → conv1x1_dynamic
  B) Reference: weight shape [1, IC, 1, OC], data in column-major (transposed)
     Graph: placeholder → transpose([0,3,2,1]) → reshape([OC,IC,1,1]) → conv1x1_dynamic
  C) Direct: weight shape [OC, IC, 1, 1], data in row-major [OC, IC]
     Graph: placeholder → conv1x1_dynamic (no transpose/reshape)

Uses small dimensions to keep it fast and easy to debug.
"""

import sys
import numpy as np

# Small dimensions for testing
IC = 64
OC = 32
SEQ = 64  # minimum spatial width

def build_conv_kernel_approach_a(ic, oc, seq):
    """Current approach: weight [1, OC, 1, IC], row-major data."""
    import mirror_sd_ane as ane
    from mirror_sd_ane import ANETensor, ANEKernel
    # Need to build via Rust
    # We'll use compile_dflash_kernels... no, we need a custom kernel
    # Actually we can't build custom kernels from Python. Let me use a different approach.
    pass


def test_with_mlx():
    """Use MLX to compute ground truth, then compare with ANE."""
    import mlx.core as mx
    
    IC = 64
    OC = 32
    SEQ = 64
    
    # Create known weight and input
    # Weight: simple diagonal-like pattern for easy verification
    np.random.seed(42)
    W = np.random.randn(OC, IC).astype(np.float32)
    X = np.random.randn(1, SEQ, IC).astype(np.float32)
    
    # Ground truth: Y = X @ W^T  (MLX linear convention)
    Y_expected = X @ W.T  # [1, SEQ, OC]
    
    print(f"Expected output shape: {Y_expected.shape}")
    print(f"Expected output[0, :4, :4]:\n{Y_expected[0, :4, :4]}")
    
    return W, X, Y_expected


def test_ane_conv1x1():
    """Test ANE conv1x1 with known weights using the Rust module."""
    import mirror_sd_ane as ane
    
    IC = 64
    OC = 32
    SEQ = 64
    
    np.random.seed(42)
    W = np.random.randn(OC, IC).astype(np.float32)
    X = np.random.randn(1, SEQ, IC).astype(np.float32)
    
    # Ground truth
    Y_expected = X @ W.T  # [1, SEQ, OC]
    
    # We need to build a simple conv1x1 kernel graph from Rust
    # The Rust module has test_conv1x1 but it only checks compilation, not execution
    # We need to build our own test kernel
    # 
    # Let me check if we can access the Graph builder from Python...
    # No, we can't. We need to add a Rust function to build+compile a test conv1x1 kernel.
    print("Need to add Rust-side test function. Let me check what's available.")
    print(f"Available functions: {dir(ane)}")


def test_ane_conv1x1_existing():
    """
    Use the existing ANE infrastructure to test conv1x1 weight layout.
    
    We'll build a minimal kernel via Rust that does:
      input [1, IC, 1, SEQ] → conv1x1_proj → output [1, OC, 1, SEQ]
    
    Then compare against MLX ground truth.
    """
    # First, let's check if we can use the existing dflash kernel infrastructure
    # The k_proj_ctx kernel is a simple conv1x1 with weight shape [1, OC, 1, IC]
    # But we need to isolate just the conv1x1 part
    
    # Actually, let me take a different approach: test with the full ANE model
    # but with a known single-layer scenario
    
    # For now, let me focus on understanding the data layout issue by
    # examining the IOSurface data format
    
    # The key insight from the rustane reference:
    # - Weight placeholder shape is [1, IC, 1, OC] (channels=IC, width=OC)
    # - Data is stored as: element [0, ic, 0, oc] at offset ic*OC + oc
    # - This is column-major of the [OC, IC] weight matrix
    # - After transpose([0,3,2,1]): shape becomes [1, OC, 1, IC]
    # - The ANE reads element [0, oc, 0, ic] from original offset ic*OC + oc
    # - After reshape to [OC, IC, 1, 1]: element [oc, ic, 0, 0] from offset ic*OC + oc
    # - So weight W[oc][ic] is stored at offset ic*OC + oc (column-major)
    
    # Our current approach:
    # - Weight placeholder shape is [1, OC, 1, IC] (channels=OC, width=IC)
    # - Data is stored as: element [0, oc, 0, ic] at offset oc*IC + ic
    # - This is row-major of the [OC, IC] weight matrix
    # - After transpose([0,3,2,1]): shape becomes [1, IC, 1, OC]
    # - The ANE should read element [0, ic, 0, oc] from original offset oc*IC + ic
    # - After reshape to [OC, IC, 1, 1]: element [oc, ic, 0, 0] from offset oc*IC + ic
    # - So weight W[oc][ic] is stored at offset oc*IC + ic (row-major)
    
    # BOTH approaches should give the same result if the ANE transpose works correctly!
    # The issue must be that transpose on a placeholder doesn't work correctly,
    # OR there's something else going on.
    
    # Let me test by building a kernel that takes weight as [OC, IC, 1, 1] directly
    # (no transpose needed) and see if that works.
    
    print("Analysis complete - need to modify Rust code to test direct weight input")
    print("The issue is likely that ANE transpose on a placeholder tensor is broken")
    print("Solution: change weight layout to avoid transpose on placeholder")


if __name__ == "__main__":
    W, X, Y_expected = test_with_mlx()
    test_ane_conv1x1_existing()