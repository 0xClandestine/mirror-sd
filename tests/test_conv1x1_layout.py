"""Diagnostic: compare ANE conv1x1 output against NumPy ground truth.

Tests two approaches for weight layout:
  A) transpose: weight placeholder [1, OC, 1, IC], data row-major, then graph transpose+reshape
  B) concat-slice: weight placeholder [1, IC, 1, OC], data column-major (transposed),
     then concat+slice+transpose+reshape (matches rustane reference)

Uses small dimensions (IC=64, OC=32, SEQ=64) for fast turnaround.
"""

import numpy as np

IC = 64
OC = 32
SEQ = 64


def numpy_ground_truth(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Y = X @ W^T, where X is [1, SEQ, IC] and W is [OC, IC]."""
    return X @ W.T


def test_approach_a():
    """Current approach: weight shape [1, OC, 1, IC], data in row-major order.
    
    Graph: placeholder(weight) → transpose([0,3,2,1]) → reshape([OC,IC,1,1]) → conv1x1_dynamic
    Weight data: W[oc][ic] at offset oc*IC + ic (row-major)
    """
    import mirror_sd_ane as ane

    np.random.seed(42)
    W = np.random.randn(OC, IC).astype(np.float32)
    X = np.random.randn(1, SEQ, IC).astype(np.float32)
    Y_expected = numpy_ground_truth(X, W)

    # Compile kernel
    kernels = ane.compile_conv1x1_transpose(IC, OC, SEQ)
    kernel = kernels[0]
    print(f"Approach A kernel: {kernel}")

    w_seq = ((SEQ + 63) // 64) * 64
    w_ic = ((IC + 63) // 64) * 64
    w_oc = ((OC + 63) // 64) * 64

    # Create input tensor [1, IC, 1, w_seq]
    input_tensor = ane.ANETensor(1, IC, 1, w_seq)
    # Create weight tensor [1, OC, 1, w_ic] — current approach
    weight_tensor = ane.ANETensor(1, OC, 1, w_ic)
    # Create output tensor [1, OC, 1, w_seq]
    output_tensor = ane.ANETensor(1, OC, 1, w_seq)

    # Fill input: X is [1, SEQ, IC], need to write as [1, IC, 1, w_seq] (channels-first)
    input_data = np.zeros((1, IC, w_seq), dtype=np.float32)
    input_data[:, :, :SEQ] = X.transpose(0, 2, 1)  # [1, SEQ, IC] → [1, IC, SEQ]
    input_tensor.write_f32(input_data.flatten().tolist())

    # Fill weight: W is [OC, IC] row-major, write as [1, OC, 1, IC] channels-first
    weight_data = np.zeros((1, OC, 1, w_ic), dtype=np.float32)
    weight_data[:, :, :, :IC] = W.reshape(OC, 1, IC)  # Nope, need 4D
    weight_4d = W.reshape(1, OC, 1, IC)  # [1, OC, 1, IC]
    weight_data[:, :, :, :IC] = weight_4d
    weight_tensor.write_f32(weight_data.flatten().tolist())

    # Run
    kernel.run([input_tensor, weight_tensor], [output_tensor])

    # Read output: [1, OC, 1, w_seq] → [1, SEQ, OC]
    output_flat = np.array(output_tensor.read_f32(), dtype=np.float32)
    output_4d = output_flat.reshape(1, OC, 1, w_seq)
    Y_ane = output_4d[:, :, 0, :SEQ].transpose(0, 2, 1)  # [1, OC, SEQ] → [1, SEQ, OC]

    # Compare
    cos_sim = np.sum(Y_expected * Y_ane) / (np.linalg.norm(Y_expected) * np.linalg.norm(Y_ane) + 1e-8)
    max_abs_err = np.max(np.abs(Y_expected - Y_ane))
    mean_abs_err = np.mean(np.abs(Y_expected - Y_ane))

    print(f"Approach A (transpose on placeholder [1, OC, 1, IC]):")
    print(f"  Cosine similarity: {cos_sim:.6f}")
    print(f"  Max absolute error: {max_abs_err:.6f}")
    print(f"  Mean absolute error: {mean_abs_err:.6f}")
    print(f"  Expected output[0,:4,:4]:\n{Y_expected[0,:4,:4]}")
    print(f"  ANE output[0,:4,:4]:\n{Y_ane[0,:4,:4]}")
    return cos_sim


def test_approach_b():
    """Reference approach: weight shape [1, IC, 1, OC], data in column-major order.
    
    Graph: concat(input, weight) → slice → transpose([0,3,2,1]) → reshape([OC,IC,1,1]) → conv1x1_dynamic
    Weight data: W[oc][ic] at offset ic*OC + oc (column-major / transposed)
    """
    import mirror_sd_ane as ane

    np.random.seed(42)
    W = np.random.randn(OC, IC).astype(np.float32)
    X = np.random.randn(1, SEQ, IC).astype(np.float32)
    Y_expected = numpy_ground_truth(X, W)

    # Compile kernel
    kernels = ane.compile_conv1x1_concat(IC, OC, SEQ)
    kernel = kernels[0]
    print(f"Approach B kernel: {kernel}")

    w_seq = ((SEQ + 63) // 64) * 64
    w_oc = ((OC + 63) // 64) * 64

    # Create activation tensor [1, IC, 1, w_seq]
    acts_tensor = ane.ANETensor(1, IC, 1, w_seq)
    # Create weight tensor [1, IC, 1, w_oc] — reference approach (channels=IC, width=OC)
    wts_tensor = ane.ANETensor(1, IC, 1, w_oc)
    # Create output tensor [1, OC, 1, w_seq]
    output_tensor = ane.ANETensor(1, OC, 1, w_seq)

    # Fill activations: X is [1, SEQ, IC], write as [1, IC, 1, w_seq]
    input_data = np.zeros((1, IC, w_seq), dtype=np.float32)
    input_data[:, :, :SEQ] = X.transpose(0, 2, 1)
    acts_tensor.write_f32(input_data.flatten().tolist())

    # Fill weights: W is [OC, IC], store as [1, IC, 1, OC] in column-major
    # Element [0, ic, 0, oc] at offset ic*OC + oc
    # W[oc][ic] should be at position (ic, oc) in the [IC, OC] view
    weight_data = np.zeros((1, IC, w_oc), dtype=np.float32)
    weight_data[:, :, :OC] = W.T  # W.T is [IC, OC], matches [1, IC, 1, OC] layout
    wts_tensor.write_f32(weight_data.flatten().tolist())

    # Run
    kernel.run([acts_tensor, wts_tensor], [output_tensor])

    # Read output: [1, OC, 1, w_seq] → [1, SEQ, OC]
    output_flat = np.array(output_tensor.read_f32(), dtype=np.float32)
    output_4d = output_flat.reshape(1, OC, 1, w_seq)
    Y_ane = output_4d[:, :, 0, :SEQ].transpose(0, 2, 1)

    # Compare
    cos_sim = np.sum(Y_expected * Y_ane) / (np.linalg.norm(Y_expected) * np.linalg.norm(Y_ane) + 1e-8)
    max_abs_err = np.max(np.abs(Y_expected - Y_ane))
    mean_abs_err = np.mean(np.abs(Y_expected - Y_ane))

    print(f"Approach B (concat-slice on [1, IC, 1, OC]):")
    print(f"  Cosine similarity: {cos_sim:.6f}")
    print(f"  Max absolute error: {max_abs_err:.6f}")
    print(f"  Mean absolute error: {mean_abs_err:.6f}")
    print(f"  Expected output[0,:4,:4]:\n{Y_expected[0,:4,:4]}")
    print(f"  ANE output[0,:4,:4]:\n{Y_ane[0,:4,:4]}")
    return cos_sim


def test_approach_c():
    """Approach C: weight shape [1, IC, 1, OC] with transposed data, direct transpose (no concat-slice).
    
    Tests whether transpose on a placeholder works at all with the [1, IC, 1, OC] shape.
    Graph: placeholder(weight) → transpose([0,3,2,1]) → reshape([OC,IC,1,1]) → conv1x1_dynamic
    Weight data: W[oc][ic] at offset ic*OC + oc (column-major / transposed)
    """
    import mirror_sd_ane as ane

    np.random.seed(42)
    W = np.random.randn(OC, IC).astype(np.float32)
    X = np.random.randn(1, SEQ, IC).astype(np.float32)
    Y_expected = numpy_ground_truth(X, W)

    # Compile kernel
    kernels = ane.compile_conv1x1_transpose_b(IC, OC, SEQ)
    kernel = kernels[0]
    print(f"Approach C kernel: {kernel}")

    w_seq = ((SEQ + 63) // 64) * 64
    w_oc = ((OC + 63) // 64) * 64

    # Create input tensor [1, IC, 1, w_seq]
    input_tensor = ane.ANETensor(1, IC, 1, w_seq)
    # Create weight tensor [1, IC, 1, w_oc] — transposed shape
    weight_tensor = ane.ANETensor(1, IC, 1, w_oc)
    # Create output tensor [1, OC, 1, w_seq]
    output_tensor = ane.ANETensor(1, OC, 1, w_seq)

    # Fill input
    input_data = np.zeros((1, IC, w_seq), dtype=np.float32)
    input_data[:, :, :SEQ] = X.transpose(0, 2, 1)
    input_tensor.write_f32(input_data.flatten().tolist())

    # Fill weight: W.T is [IC, OC], store as [1, IC, 1, OC]
    weight_data = np.zeros((1, IC, w_oc), dtype=np.float32)
    weight_data[:, :, :OC] = W.T
    weight_tensor.write_f32(weight_data.flatten().tolist())

    # Run
    kernel.run([input_tensor, weight_tensor], [output_tensor])

    # Read output
    output_flat = np.array(output_tensor.read_f32(), dtype=np.float32)
    output_4d = output_flat.reshape(1, OC, 1, w_seq)
    Y_ane = output_4d[:, :, 0, :SEQ].transpose(0, 2, 1)

    # Compare
    cos_sim = np.sum(Y_expected * Y_ane) / (np.linalg.norm(Y_expected) * np.linalg.norm(Y_ane) + 1e-8)
    max_abs_err = np.max(np.abs(Y_expected - Y_ane))
    mean_abs_err = np.mean(np.abs(Y_expected - Y_ane))

    print(f"Approach C (direct transpose on [1, IC, 1, OC]):")
    print(f"  Cosine similarity: {cos_sim:.6f}")
    print(f"  Max absolute error: {max_abs_err:.6f}")
    print(f"  Mean absolute error: {mean_abs_err:.6f}")
    print(f"  Expected output[0,:4,:4]:\n{Y_expected[0,:4,:4]}")
    print(f"  ANE output[0,:4,:4]:\n{Y_ane[0,:4,:4]}")
    return cos_sim


if __name__ == "__main__":
    print("=" * 60)
    print("Testing ANE conv1x1 weight layout approaches")
    print(f"IC={IC}, OC={OC}, SEQ={SEQ}")
    print("=" * 60)
    
    cos_a = test_approach_a()
    print()
    cos_b = test_approach_b()
    print()
    cos_c = test_approach_c()
    
    print()
    print("=" * 60)
    print(f"Approach A (transpose on [1,OC,1,IC]): {cos_a:.6f}")
    print(f"Approach B (concat-slice [1,IC,1,OC]): {cos_b:.6f}")
    print(f"Approach C (transpose on [1,IC,1,OC]): {cos_c:.6f}")
    if cos_b > max(cos_a, cos_c):
        print("=> Approach B (concat-slice) is BEST")
        print("=> ANE transpose on placeholder is broken for BOTH [1,OC,1,IC] and [1,IC,1,OC]")
        print("=> Must use concat+slice pattern for weight transposition")
    elif cos_c > cos_a:
        print("=> Approach C (transpose on [1,IC,1,OC]) works")
        print("=> Issue was with [1,OC,1,IC] shape specifically")
    else:
        print("=> Both approaches give similar results")