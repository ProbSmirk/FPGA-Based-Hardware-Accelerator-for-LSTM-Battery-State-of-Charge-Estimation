
import numpy as np
import csv
import os
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

MODEL_FILE = 'fpga_lstm_model_qat.h5'
ORDERED_CSV = r'C:\Users\sumuk\PycharmMiscProject\batteryfiles\Master_Training_Data_Ordered.csv'
TIME_STEPS = 50
CURRENT_CLIP = 5.0

VOLTAGE_GLITCH_THRESHOLD = 1.0

def hard_sigmoid_fpga(x):
    return tf.clip_by_value(0.25 * x + 0.5, 0.0, 1.0)


def hard_tanh_fpga(x):
    return tf.clip_by_value(x, -1.0, 1.0)


print("Loading model...")
model = load_model(
    MODEL_FILE, compile=False,
    custom_objects={'hard_sigmoid_fpga': hard_sigmoid_fpga,
                     'hard_tanh_fpga': hard_tanh_fpga}
)
print(f"  Input shape: {model.input_shape}")

bn_layer = model.get_layer('input_batchnorm')
bn_gamma = bn_layer.get_weights()[0]
bn_beta = bn_layer.get_weights()[1]
bn_mean = bn_layer.get_weights()[2]
bn_var = bn_layer.get_weights()[3]
bn_eps = 1e-3

bn_scale = bn_gamma / np.sqrt(bn_var + bn_eps)
bn_offset = bn_beta - bn_gamma * bn_mean / np.sqrt(bn_var + bn_eps)

print(f"\nRefitting scaler...")
df = pd.read_csv(ORDERED_CSV)
features = ['voltage_load', 'current_load', 'temperature_battery', 'Ah_used']
df['current_load'] = df['current_load'].clip(lower=-CURRENT_CLIP, upper=CURRENT_CLIP)

voltage_glitch_mask = (df['voltage_load'] < VOLTAGE_GLITCH_THRESHOLD).values
print(f"Voltage glitch rows (< {VOLTAGE_GLITCH_THRESHOLD}V): "
      f"{voltage_glitch_mask.sum():,} ({voltage_glitch_mask.sum()/len(df)*100:.2f}%) — excluded from scaler fit")

scaler_x = MinMaxScaler(feature_range=(0, 1))
scaler_x.fit(df.loc[~voltage_glitch_mask, features])

x_min = scaler_x.data_min_
x_max = scaler_x.data_max_
mm_scale = 1.0 / (x_max - x_min)
mm_offset = -x_min / (x_max - x_min)

feat_names = ['voltage', 'current', 'temperature', 'Ah_used']
print(f"  {'Feature':<14} {'min':>10} {'max':>10}")
for i, name in enumerate(feat_names):
    print(f"  {name:<14} {x_min[i]:>10.4f} {x_max[i]:>10.4f}")

if os.path.exists('scaler_params.npz'):
    saved = np.load('scaler_params.npz')
    all_match = all(abs(saved['x_min'][i] - x_min[i]) < 0.001 and
                    abs(saved['x_max'][i] - x_max[i]) < 0.001 for i in range(4))
    print(f"  Scaler vs saved: {'all match OK' if all_match else 'DIFFERS — using refitted'}")


def to_int16(x):
    x = int(x) & 0xFFFF
    if x >= 0x8000: x -= 0x10000 #chops off the top bits if it exceeds 16 bits and converts to negative in 2 complements
    return x


def float_to_q8(f):
    return to_int16(int(round(f * 256)))


def q8_to_float(q):
    return q / 256.0

def minmax_apply(x_q8, scale_f32, offset_f32):
    s = int(scale_f32  * 65536)   # truncate
    o = int(offset_f32 * 65536)   # truncate
    r = to_int16((int(x_q8) * s >> 16) + (o >> 8))
    if r < 0:      return 0
    if r > 0x0100: return 0x0100
    return r

def bn_apply(x_q8, scale_f32, offset_f32):
    s = int(scale_f32  * 65536)   # truncate, not round — matches Verilog integer cast
    o = int(offset_f32 * 65536)   # truncate, not round
    return to_int16((int(x_q8) * s >> 16) + (o >> 8))


def hard_tanh_cell(x_q8):
    if x_q8 > 0x0100: return 0x0100
    if x_q8 < -0x0100: return -0x0100
    return x_q8


def hard_sigmoid_q8(x_q8):

    if x_q8 > 0x0200: return 0x0100
    if x_q8 < -0x0200: return 0x0000
    return to_int16((x_q8 >> 2) + 0x0080)


# ── Load weights ──────────────────────────────────────────────────────────────
def load_hex_s32(filename):
    out = []
    for h in open(filename).read().strip().split():
        v = int(h, 16)
        if v >= 0x80000000: v -= 0x100000000
        out.append(v)
    return out


lstm_W = load_hex_s32('lstm_W.txt')
lstm_U = load_hex_s32('lstm_U.txt')
lstm_b = load_hex_s32('lstm_b.txt')
dense_W = load_hex_s32('dense_W.txt')
dense_b = load_hex_s32('dense_b.txt')

while len(lstm_W) < 256: lstm_W.append(0)

print(f"\nWeights: W={len(lstm_W)}, U={len(lstm_U)}, b={len(lstm_b)}, "
      f"dW={len(dense_W)}, db={len(dense_b)}")


# ── Pipeline functions ────────────────────────────────────────────────────────
def preprocess_q8(v_phys, i_phys, t_phys, ah_phys):
    v_q8 = float_to_q8(v_phys);
    i_q8 = float_to_q8(i_phys)
    t_q8 = float_to_q8(t_phys);
    ah_q8 = float_to_q8(ah_phys)

    v_mm = minmax_apply(v_q8, mm_scale[0], mm_offset[0])
    i_mm = minmax_apply(i_q8, mm_scale[1], mm_offset[1])
    t_mm = minmax_apply(t_q8, mm_scale[2], mm_offset[2])
    ah_mm = minmax_apply(ah_q8, mm_scale[3], mm_offset[3])

    return (bn_apply(v_mm, bn_scale[0], bn_offset[0]),
            bn_apply(i_mm, bn_scale[1], bn_offset[1]),
            bn_apply(t_mm, bn_scale[2], bn_offset[2]),
            bn_apply(ah_mm, bn_scale[3], bn_offset[3]))


def lstm_step(x_bn, cell_in, hidden_in):
    gate_offsets = {'input': 0, 'forget': 16, 'cell': 32, 'output': 48}
    gate_results = {}

    for gate_name, gate_offset in gate_offsets.items():
        results = []
        for n in range(16):
            acc = lstm_b[gate_offset + n] << 8
            for inp in range(4):
                acc += x_bn[inp] * lstm_W[inp * 64 + gate_offset + n]
            for k in range(16):
                acc += hidden_in[k] * lstm_U[k * 64 + gate_offset + n]

            act_in = to_int16((acc >> 16) & 0xFFFF)
            # gate_name == 'cell' -> CELL_CANDIDATE, uses hard_tanh in macarray.v
            # everything else -> sigmoid gates, use hard_sigmoid in macarray.v
            results.append(hard_tanh_cell(act_in) if gate_name == 'cell'
                           else hard_sigmoid_q8(act_in))
        gate_results[gate_name] = results

    f_t = gate_results['forget'];
    i_t = gate_results['input']
    c_t = gate_results['cell'];
    o_t = gate_results['output']

    # UPDATE_CELL: c = f*c_prev + i*c_tilde
    new_cell = [to_int16(((f_t[n] * cell_in[n]) >> 8) +
                         ((i_t[n] * c_t[n]) >> 8)) for n in range(16)]

    # UPDATE_H: h = o * tanh(c)  — apply hard_tanh_cell before multiply
    new_hidden = [to_int16((o_t[n] * hard_tanh_cell(new_cell[n])) >> 8)
                  for n in range(16)]

    return new_hidden, new_cell


def lstm_single_step_soc(v_bn, i_bn, t_bn, ah_bn):
    hidden, cell = lstm_step([v_bn, i_bn, t_bn, ah_bn], [0] * 16, [0] * 16)

    acc = dense_b[0] << 8
    for i in range(16): acc += hidden[i] * dense_W[i]
    act_in = to_int16((acc >> 16) & 0xFFFF)

    # Dense output activation — this was already the hard_sigmoid formula,
    # no change needed here.
    if act_in > 0x0200: return 0x0100
    if act_in < -0x0200: return 0x0000
    return to_int16((act_in >> 2) + 0x0080)


def keras_predict_steady(v_phys, i_phys, t_phys, ah_phys):
    v_sc = float(np.clip((v_phys - x_min[0]) / (x_max[0] - x_min[0]), 0, 1))
    i_sc = float(np.clip((i_phys - x_min[1]) / (x_max[1] - x_min[1]), 0, 1))
    t_sc = float(np.clip((t_phys - x_min[2]) / (x_max[2] - x_min[2]), 0, 1))
    ah_sc = float(np.clip((ah_phys - x_min[3]) / (x_max[3] - x_min[3]), 0, 1))
    seq = np.tile(np.array([v_sc, i_sc, t_sc, ah_sc]), (TIME_STEPS, 1))[np.newaxis]
    return float(model.predict(seq, verbose=0)[0][0])


# ── Debug: verify fix on 3 cases ──────────────────────────────────────────────
print("\n" + "=" * 65)
print("  PREPROCESSING DEBUG")
print("=" * 65)

debug_cases = [
    (float(np.median([x_min[0], x_max[0]])), 0.0, 25.0, 0.0, "Nominal"),
    (float(x_min[0]) + 0.1 * (float(x_max[0]) - float(x_min[0])), 5.0, 30.0, 100.0, "Discharging"),
    (float(x_max[0]) - 0.05 * (float(x_max[0]) - float(x_min[0])), 0.0, 20.0, 0.0, "Full charge"),
]

print(f"\n  {'Case':<14} {'Stage':<10} {'V':>8} {'I':>8} {'T':>8} {'Ah':>8}")
print("  " + "-" * 60)

for v_t, i_t, t_t, ah_t, label in debug_cases:
    v_mm_f = np.clip((v_t - x_min[0]) / (x_max[0] - x_min[0]), 0, 1)
    i_mm_f = np.clip((i_t - x_min[1]) / (x_max[1] - x_min[1]), 0, 1)
    t_mm_f = np.clip((t_t - x_min[2]) / (x_max[2] - x_min[2]), 0, 1)
    ah_mm_f = np.clip((ah_t - x_min[3]) / (x_max[3] - x_min[3]), 0, 1)

    v_bn_f = bn_scale[0] * v_mm_f + bn_offset[0]
    i_bn_f = bn_scale[1] * i_mm_f + bn_offset[1]
    t_bn_f = bn_scale[2] * t_mm_f + bn_offset[2]
    ah_bn_f = bn_scale[3] * ah_mm_f + bn_offset[3]

    v_bn, i_bn, t_bn, ah_bn = preprocess_q8(v_t, i_t, t_t, ah_t)
    soc_q8 = lstm_single_step_soc(v_bn, i_bn, t_bn, ah_bn)
    keras_s = keras_predict_steady(v_t, i_t, t_t, ah_t)

    print(f"\n  {label}")
    print(f"  {'':14} {'Post-BN':<10} {v_bn / 256:>8.4f} {i_bn / 256:>8.4f} "
          f"{t_bn / 256:>8.4f} {ah_bn / 256:>8.4f}")
    print(f"  {'':14} {'Float-BN':<10} {v_bn_f:>8.4f} {i_bn_f:>8.4f} "
          f"{t_bn_f:>8.4f} {ah_bn_f:>8.4f}")
    print(f"  {'':14} {'SoC-VLG':<10} {q8_to_float(soc_q8):>8.4f}")
    print(f"  {'':14} {'SoC-Keras':<10} {keras_s:>8.4f}")
    print(f"  {'':14} {'Error':<10} {abs(q8_to_float(soc_q8) - keras_s) * 100:>7.2f}%")

# ── Generate vectors ──────────────────────────────────────────────────────────
# Sweep the ACTUAL fitted range, not a hardcoded single-cell Li-ion guess.
# Your real data sits at ~4.9-8.8V (multi-cell pack) — the old 2.5/4.2V
# clip here made v_hi < v_lo against your real x_min/x_max, silently
# collapsing most golden vectors into a degenerate near-zero region.
v_lo = float(x_min[0]);
v_hi = float(x_max[0])
t_lo = max(float(x_min[2]), 0.0);
t_hi = min(float(x_max[2]), 50.0)
ah_hi = min(float(x_max[3]) * 0.8, 500.0)

v_range = np.linspace(v_lo, v_hi, 10)
i_levels = [0.0, 2.5, 5.0]
t_range = np.linspace(t_lo, t_hi, 5)
ah_range = np.linspace(0.0, ah_hi, 7)

total = len(v_range) * len(i_levels) * len(t_range) * len(ah_range)

print(f"\n{'=' * 65}")
print(f"  GENERATING {total} VECTORS")
print(f"{'=' * 65}")
print(f"  Voltage    : {v_range[0]:.2f}V → {v_range[-1]:.2f}V  ({len(v_range)} pts)")
print(f"  Current    : {i_levels} A")
print(f"  Temperature: {t_range[0]:.1f}°C → {t_range[-1]:.1f}°C  ({len(t_range)} pts)")
print(f"  Ah_used    : {ah_range[0]:.1f} → {ah_range[-1]:.1f} Ah  ({len(ah_range)} pts)")

vectors = []
count = 0

for ah in ah_range:
    for t in t_range:
        for i_val in i_levels:
            for v in v_range:
                v_bn, i_bn, t_bn, ah_bn = preprocess_q8(
                    float(v), float(i_val), float(t), float(ah))

                soc_q8 = lstm_single_step_soc(v_bn, i_bn, t_bn, ah_bn)
                keras_s = keras_predict_steady(
                    float(v), float(i_val), float(t), float(ah))

                verilog_f = q8_to_float(soc_q8)

                vectors.append({
                    'v': float(v), 'i': float(i_val), 't': float(t), 'ah': float(ah),
                    'v_bn_q8': v_bn, 'i_bn_q8': i_bn, 't_bn_q8': t_bn, 'ah_bn_q8': ah_bn,
                    'soc_verilog_q8': soc_q8, 'soc_verilog': verilog_f,
                    'soc_keras': keras_s,
                    'error_pct': abs(verilog_f - keras_s) * 100
                })
                count += 1
                if count % 100 == 0:
                    print(f"  {count}/{total} done...")

# ── Save mem files ────────────────────────────────────────────────────────────
for fname, key in [('golden_v.mem', 'v_bn_q8'), ('golden_i.mem', 'i_bn_q8'),
                   ('golden_t.mem', 't_bn_q8'), ('golden_ah.mem', 'ah_bn_q8'),
                   ('golden_soc.mem', 'soc_verilog_q8')]:
    with open(fname, 'w') as f:
        for vec in vectors:
            f.write(f"{vec[key] & 0xFFFF:04x}\n")

with open('golden_ref.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=vectors[0].keys())
    writer.writeheader();
    writer.writerows(vectors)

print(f"\nFiles saved: golden_*.mem ({total} vectors each), golden_ref.csv")

# ── Error summary ─────────────────────────────────────────────────────────────
errors = np.array([v['error_pct'] for v in vectors])
keras_vals = np.array([v['soc_keras'] for v in vectors])

print(f"\n{'=' * 60}")
print(f"  VERILOG FIXED-POINT vs KERAS (single-step, steady-state)")
print(f"  NOTE: this compares single-step Verilog vs 50-step Keras.")
print(f"  For sequence accuracy see verify_pipeline.py instead.")
print(f"{'=' * 60}")
print(f"  Mean error : {errors.mean():.2f}%")
print(f"  Median err : {np.median(errors):.2f}%")
print(f"  Max error  : {errors.max():.2f}%")
print(f"  Within 5%  : {(errors < 5).sum():,} / {total}")
print(f"  Within 10% : {(errors < 10).sum():,} / {total}")

print(f"\n  Voltage sweep (I=0A, T={t_range[2]:.1f}C, Ah=0):")
print(f"  {'Voltage':>8} {'Keras':>8} {'Verilog':>10} {'Err%':>8}")
print("  " + "-" * 40)
t_mid = float(t_range[2])
for vec in vectors:
    if abs(vec['i']) < 0.1 and abs(vec['t'] - t_mid) < 1.0 and vec['ah'] < 0.1:
        print(f"  {vec['v']:>8.3f}V  {vec['soc_keras']:>8.4f}  "
              f"{vec['soc_verilog']:>10.4f}  {vec['error_pct']:>7.2f}%")

print(f"\nUpdate tb_lstm_system.v: set NUM_VECTORS = {total}")
print("Copy golden_*.mem to Vivado project folder, then run simulation.")