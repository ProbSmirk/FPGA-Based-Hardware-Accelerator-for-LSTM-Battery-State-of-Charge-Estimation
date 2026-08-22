

import numpy as np
import pandas as pd
import os, warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

MODEL_FILE = 'fpga_lstm_model_qat.h5'
ORDERED_CSV = r'C:\Users\sumuk\PycharmMiscProject\batteryfiles\Master_Training_Data_Ordered.csv'
TIME_STEPS = 50


VOLTAGE_GLITCH_THRESHOLD = 1.0

def hard_sigmoid_fpga(x):
    return tf.clip_by_value(0.25 * x + 0.5, 0.0, 1.0)


def hard_tanh_fpga(x):
    return tf.clip_by_value(x, -1.0, 1.0)


print("Loading model and data...")
model = load_model(
    MODEL_FILE, compile=False,
    custom_objects={'hard_sigmoid_fpga': hard_sigmoid_fpga,
                     'hard_tanh_fpga': hard_tanh_fpga}
)
df = pd.read_csv(ORDERED_CSV)

features = ['voltage_load', 'current_load', 'temperature_battery', 'Ah_used']
df['current_load'] = df['current_load'].clip(lower=-5.0, upper=5.0)

voltage_glitch_mask = (df['voltage_load'] < VOLTAGE_GLITCH_THRESHOLD).values
print(f"Voltage glitch rows (< {VOLTAGE_GLITCH_THRESHOLD}V): "
      f"{voltage_glitch_mask.sum():,} ({voltage_glitch_mask.sum()/len(df)*100:.2f}%) — excluded from scaler fit")

scaler_x = MinMaxScaler(feature_range=(0, 1))
scaler_x.fit(df.loc[~voltage_glitch_mask, features])
x_min = scaler_x.data_min_
x_max = scaler_x.data_max_

bn_layer = model.get_layer('input_batchnorm')
bn_gamma = bn_layer.get_weights()[0]
bn_beta = bn_layer.get_weights()[1]
bn_mean = bn_layer.get_weights()[2]
bn_var = bn_layer.get_weights()[3]

bn_scale = bn_gamma / np.sqrt(bn_var + 1e-3)
bn_offset = bn_beta - bn_gamma * bn_mean / np.sqrt(bn_var + 1e-3)

print(f"\nBatchNorm parameters:")
feat_names = ['voltage', 'current', 'temperature', 'Ah_used']
for i, n in enumerate(feat_names):
    print(f"  {n:<12}: scale={bn_scale[i]:+.4f}  offset={bn_offset[i]:+.4f}")


def to_int16(x):
    x = int(x) & 0xFFFF
    if x >= 0x8000: x -= 0x10000
    return x


def float_to_q8(f):
    return to_int16(int(round(f * 256)))


def q8_to_float(q):
    return q / 256.0


def bn_apply_q8(x_q8, scale_f32, offset_f32):
    s = int(scale_f32 * 65536)   # truncate, not round
    o = int(offset_f32 * 65536)  # truncate, not round
    return to_int16((int(x_q8) * s >> 16) + (o >> 8))


def hard_tanh_cell(x_q8):
   
    if x_q8 > 0x0100: return 0x0100
    if x_q8 < -0x0100: return -0x0100
    return x_q8


def hard_sigmoid_q8(x_q8):
    if x_q8 > 0x0200: return 0x0100
    if x_q8 < -0x0200: return 0x0000
    return to_int16((x_q8 >> 2) + 0x0080)


def load_hex_s32(filename):
    out = []
    for h in open(filename).read().strip().split():
        v = int(h, 16)
        if v >= 0x80000000: v -= 0x100000000
        out.append(v)
    return out


lstm_W = load_hex_s32('lstm_W.txt');
lstm_U = load_hex_s32('lstm_U.txt')
lstm_b = load_hex_s32('lstm_b.txt');
dense_W = load_hex_s32('dense_W.txt')
dense_b = load_hex_s32('dense_b.txt')

while len(lstm_W) < 256: lstm_W.append(0)


def run_sequence_verilog(X_seq):
    """
    Run a 50-step sequence through the fixed-point Verilog model.
    X_seq: numpy array (50, 4) of MinMax-scaled float values.
    Returns SoC as float.
    """
    cell = [0] * 16
    hidden = [0] * 16

    for step in range(len(X_seq)):
        # Convert MinMax-scaled float to Q8.8, apply BN
        x_bn = []
        for fi in range(4):
            mm_q8 = float_to_q8(float(X_seq[step, fi]))
            x_bn.append(bn_apply_q8(mm_q8, bn_scale[fi], bn_offset[fi]))

        # LSTM gates
        gate_offsets = {'input': 0, 'forget': 16, 'cell': 32, 'output': 48}
        gate_results = {}

        for gate_name, gate_offset in gate_offsets.items():
            results = []
            for n in range(16):
                acc = lstm_b[gate_offset + n] << 8
                for inp in range(4):
                    acc += x_bn[inp] * lstm_W[inp * 64 + gate_offset + n]
                for k in range(16):
                    acc += hidden[k] * lstm_U[k * 64 + gate_offset + n]

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

        # UPDATE_CELL
        cell = [to_int16(((f_t[n] * cell[n]) >> 8) +
                         ((i_t[n] * c_t[n]) >> 8)) for n in range(16)]

        # UPDATE_H — apply hard_tanh_cell before multiply
        hidden = [to_int16((o_t[n] * hard_tanh_cell(cell[n])) >> 8)
                  for n in range(16)]

    acc = dense_b[0] << 8
    for i in range(16): acc += hidden[i] * dense_W[i]
    act_in = to_int16((acc >> 16) & 0xFFFF)

    if act_in > 0x0200: return q8_to_float(0x0100)
    if act_in < -0x0200: return q8_to_float(0x0000)
    return q8_to_float(to_int16((act_in >> 2) + 0x0080))


X_scaled = scaler_x.transform(df[features])
y_col = df['SoC'].values

soc_diff = np.abs(np.diff(y_col, prepend=y_col[0]))
boundary = soc_diff > 0.3
skip_all = boundary | voltage_glitch_mask

split = int(len(X_scaled) * 0.8)
X_test = X_scaled[split:]
y_test = y_col[split:]
bound_test = skip_all[split:]

print(f"\nRunning 200 real sequences through fixed-point Verilog model...")

vlg_preds = []
keras_preds = []
true_socs = []
n_tested = 0

for i in range(len(X_test) - TIME_STEPS):
    if bound_test[i:i + TIME_STEPS].any():
        continue

    if n_tested >= 200:
        break

    X_seq = X_test[i:i + TIME_STEPS]
    y_true = y_test[i + TIME_STEPS]

    vlg_soc = run_sequence_verilog(X_seq)
    keras_soc = float(model.predict(X_seq[np.newaxis], verbose=0)[0][0])

    vlg_preds.append(vlg_soc)
    keras_preds.append(keras_soc)
    true_socs.append(y_true)
    n_tested += 1

vlg_preds = np.array(vlg_preds)
keras_preds = np.array(keras_preds)
true_socs = np.array(true_socs)

vlg_err = np.abs(vlg_preds - true_socs) * 100
keras_err = np.abs(keras_preds - true_socs) * 100
hw_vs_krs = np.abs(vlg_preds - keras_preds) * 100

print(f"\n{'=' * 72}")
print(f"  REAL ACCURACY — {n_tested} sequences from actual battery test data")
print(f"{'=' * 72}")
print(f"  {'Metric':<35} {'Keras':>10} {'Verilog HW':>12} {'HW vs Keras':>13}")
print("  " + "-" * 72)
print(f"  {'Mean abs error vs true SoC':<35} {keras_err.mean():>9.2f}% "
      f"{vlg_err.mean():>11.2f}% {hw_vs_krs.mean():>12.2f}%")
print(f"  {'Median abs error vs true SoC':<35} {np.median(keras_err):>9.2f}% "
      f"{np.median(vlg_err):>11.2f}% {np.median(hw_vs_krs):>12.2f}%")
print(f"  {'Max abs error vs true SoC':<35} {keras_err.max():>9.2f}% "
      f"{vlg_err.max():>11.2f}% {hw_vs_krs.max():>12.2f}%")
print(f"  {'Within 5% of true SoC':<35} "
      f"{(keras_err < 5).sum():>8}/{n_tested} "
      f"{(vlg_err < 5).sum():>10}/{n_tested} "
      f"{(hw_vs_krs < 5).sum():>11}/{n_tested}")
print(f"  {'Within 10% of true SoC':<35} "
      f"{(keras_err < 10).sum():>8}/{n_tested} "
      f"{(vlg_err < 10).sum():>10}/{n_tested}")

print(f"\n  Interpretation:")
print(f"  ├─ 'Keras vs true SoC'    = model accuracy (training result)")
print(f"  ├─ 'Verilog HW vs true'   = real hardware accuracy on battery data")
print(f"  └─ 'HW vs Keras'          = fixed-point quantisation cost only")
print(f"       (now that activations match, this is pure Q8.8 rounding cost)")
print(f"       Target: HW vs Keras < 5%  →  fixed-point is faithful to model")

print(f"\n  SoC region breakdown (Verilog HW vs true):")
print(f"  {'Region':<22} {'Keras%':>8} {'VlogHW%':>9} {'HWvsKrs%':>10} {'Count':>7}")
print("  " + "-" * 58)

regions = [("0-20%  discharged", 0.0, 0.2), ("20-40% low", 0.2, 0.4),
           ("40-60% mid", 0.4, 0.6), ("60-80% high", 0.6, 0.8), ("80-100% full", 0.8, 1.0)]

for label, lo, hi in regions:
    mask = (true_socs >= lo) & (true_socs < hi)
    if mask.sum() == 0: continue
    print(f"  {label:<22} {keras_err[mask].mean():>8.2f} "
          f"{vlg_err[mask].mean():>9.2f} "
          f"{hw_vs_krs[mask].mean():>10.2f} {mask.sum():>7}")

print(f"\nDone.")
print(f"If 'HW vs Keras' mean is < 5%: fixed-point pipeline is correct.")
