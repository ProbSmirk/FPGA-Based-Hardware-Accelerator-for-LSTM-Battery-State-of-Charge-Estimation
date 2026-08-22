import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.models import load_model

MODEL_FILE  = 'fpga_lstm_model_qat.h5'
OUTPUT_DIR  = '.'    # same folder — copy .txt files to your Vivado project

Q_FORMAT    = 16     # Q16.16 fixed point (multiply float by 2^16 = 65536)
NUM_INPUTS  = 4      # voltage, current, temperature, Ah_used
NUM_NEURONS = 16
NUM_GATES   = 4      # i, f, c, o



def hard_sigmoid_fpga(x):
    return tf.clip_by_value(0.25 * x + 0.5, 0.0, 1.0)


def hard_tanh_fpga(x):
    return tf.clip_by_value(x, -1.0, 1.0)


def float_to_q16(f):
    """Convert float to Q16.16 signed 32-bit integer, returned as 8-char hex."""
    v = int(round(f * 65536))
    v = max(-2147483648, min(2147483647, v))
    if v < 0:
        v = v + (1 << 32)
    return f"{v:08X}"


def save_hex_file(filename, values, description):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, 'w') as f:
        for v in values:
            f.write(float_to_q16(v) + '\n')
    print(f"  {filename:20} {len(values):5} entries  "
          f"[{min(values):+.4f}, {max(values):+.4f}]  <- {description}")


print("=" * 60)
print("  WEIGHT EXTRACTION")
print("=" * 60)

print(f"\nLoading {MODEL_FILE}...")
model = load_model(
    MODEL_FILE, compile=False,
    custom_objects={'hard_sigmoid_fpga': hard_sigmoid_fpga,
                     'hard_tanh_fpga': hard_tanh_fpga}
)
model.summary()


lstm_layer  = model.get_layer('lstm_layer')
dense_layer = model.get_layer('soc_output')
bn_layer    = model.get_layer('input_batchnorm')

W  = lstm_layer.get_weights()[0]   # shape (4, 64) — input kernel
U  = lstm_layer.get_weights()[1]   # shape (16, 64) — recurrent kernel
b  = lstm_layer.get_weights()[2]   # shape (64,) — bias
dW = dense_layer.get_weights()[0]  # shape (16, 1) — dense kernel
dB = dense_layer.get_weights()[1]  # shape (1,) — dense bias

# BatchNorm parameters
bn_gamma    = bn_layer.get_weights()[0]   # scale     shape (4,)
bn_beta     = bn_layer.get_weights()[1]   # offset    shape (4,)
bn_mean     = bn_layer.get_weights()[2]   # moving mean  shape (4,)
bn_var      = bn_layer.get_weights()[3]   # moving var   shape (4,)
bn_epsilon  = 1e-3                         # Keras default

print(f"\nWeight shapes:")
print(f"  W  : {W.shape}   (input kernel — 4 inputs × 64 gate-neurons)")
print(f"  U  : {U.shape}  (recurrent kernel — 16 neurons × 64)")
print(f"  b  : {b.shape}    (bias — 64)")
print(f"  dW : {dW.shape}   (dense kernel — 16 neurons → 1 output)")
print(f"  dB : {dB.shape}      (dense bias)")
print(f"  BN : gamma{bn_gamma.shape} beta{bn_beta.shape} "
      f"mean{bn_mean.shape} var{bn_var.shape}")

assert W.shape  == (NUM_INPUTS, NUM_GATES * NUM_NEURONS), \
    f"W shape mismatch: got {W.shape}, expected ({NUM_INPUTS}, {NUM_GATES*NUM_NEURONS})"
assert U.shape  == (NUM_NEURONS, NUM_GATES * NUM_NEURONS), \
    f"U shape mismatch: got {U.shape}"
assert b.shape  == (NUM_GATES * NUM_NEURONS,), \
    f"b shape mismatch: got {b.shape}"
assert dW.shape == (NUM_NEURONS, 1), \
    f"dW shape mismatch: got {dW.shape}"
print("\nShape verification: PASSED")

print("\n" + "="*60)
print("  SAVING WEIGHT FILES")
print("="*60)

lstm_W_flat = []
for inp in range(NUM_INPUTS):
    for col in range(NUM_GATES * NUM_NEURONS):
        lstm_W_flat.append(W[inp, col])

save_hex_file("lstm_W.txt", lstm_W_flat,
              "lstm input weights — 4 inputs × 64 = 256 entries")

lstm_U_flat = []
for k in range(NUM_NEURONS):
    for col in range(NUM_GATES * NUM_NEURONS):
        lstm_U_flat.append(U[k, col])

save_hex_file("lstm_U.txt", lstm_U_flat,
              "lstm recurrent weights — 16 × 64 = 1024 entries")

save_hex_file("lstm_b.txt", b.tolist(),
              "lstm bias — 64 entries")

save_hex_file("dense_W.txt", dW.flatten().tolist(),
              "dense kernel — 16 entries")

save_hex_file("dense_b.txt", dB.tolist(),
              "dense bias — 1 entry")

bn_scale  = bn_gamma / np.sqrt(bn_var + bn_epsilon)
bn_offset = bn_beta - bn_gamma * bn_mean / np.sqrt(bn_var + bn_epsilon)

save_hex_file("bn_scale.txt",  bn_scale.tolist(),
              "BatchNorm scale  — 4 entries (one per feature)")
save_hex_file("bn_offset.txt", bn_offset.tolist(),
              "BatchNorm offset — 4 entries (one per feature)")

# Also save scaler params for minmax_preprocess.v
scaler = np.load('scaler_params.npz')
x_min = scaler['x_min']   # shape (4,)
x_max = scaler['x_max']   # shape (4,)

mm_scale  = 1.0 / (x_max - x_min)
mm_offset = -x_min / (x_max - x_min)

save_hex_file("mm_scale.txt",  mm_scale.tolist(),
              "MinMaxScaler scale  — 4 entries")
save_hex_file("mm_offset.txt", mm_offset.tolist(),
              "MinMaxScaler offset — 4 entries")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  SUMMARY")
print("="*60)
print(f"\nFiles generated (copy ALL to your Vivado project sim/src folder):")
files = ["lstm_W.txt", "lstm_U.txt", "lstm_b.txt",
         "dense_W.txt", "dense_b.txt",
         "bn_scale.txt", "bn_offset.txt",
         "mm_scale.txt", "mm_offset.txt"]
for f in files:
    path = os.path.join(OUTPUT_DIR, f)
    if os.path.exists(path):
        size = os.path.getsize(path)
        print(f"  {f:20} {size:7} bytes")

print(f"\nBatchNorm parameters (will be hardcoded in bnorm_preprocess.v):")
feat_names = ['voltage', 'current', 'temperature', 'Ah_used']
for i, name in enumerate(feat_names):
    print(f"  [{i}] {name:12}: "
          f"scale={bn_scale[i]:+.6f}  offset={bn_offset[i]:+.6f}")

print(f"\nMinMaxScaler parameters (hardcoded in minmax_preprocess.v):")
for i, name in enumerate(feat_names):
    print(f"  [{i}] {name:12}: "
          f"raw_min={x_min[i]:.4f}  raw_max={x_max[i]:.4f}  "
          f"scale={mm_scale[i]:.6f}")

print(f"\nWorst-case pre-activation check:")
worst = NUM_INPUTS * np.abs(W).max() + NUM_NEURONS * np.abs(U).max() + np.abs(b).max()
print(f"  {NUM_INPUTS}×{np.abs(W).max():.3f} + "
      f"{NUM_NEURONS}×{np.abs(U).max():.3f} + {np.abs(b).max():.3f} = {worst:.2f}")
print(f"  LUT_DEPTH  = 16384  (covers ±32 in Q8.8)")
print(f"  LUT_OFFSET = 8192")

print("\nDone. Run generate_lut.py next.")