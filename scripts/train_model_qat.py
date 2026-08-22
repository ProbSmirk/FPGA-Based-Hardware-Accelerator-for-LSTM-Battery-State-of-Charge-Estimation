import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_sample_weight
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

DATA_FILE      = r'C:\Users\sumuk\PycharmMiscProject\batteryfiles\Master_Training_Data_Ordered.csv'
MODEL_OUT      = 'fpga_lstm_model_qat.h5'
SCALER_OUT     = 'scaler_params.npz'

HIDDEN_NEURONS = 16
TIME_STEPS     = 50
NUM_FEATURES   = 4

MAX_WEIGHT_W   = 2.0
MAX_WEIGHT_DW  = 3.0

EPOCHS         = 300
BATCH_SIZE     = 512
LR_INITIAL     = 0.0005

# Any voltage_load reading below this is a sensor glitch/dropout, not a
# real operating point — confirmed via check_voltage_glitches.py:
# 22,122/571,430 rows (3.87%), scattered mid-recording, essentially zero
# overlap with SoC-jump file boundaries. Must be excluded from the scaler
# fit (was dragging x_min to -0.027 instead of the real ~4.9V floor) and
# from any training/eval window.
VOLTAGE_GLITCH_THRESHOLD = 1.0


def hard_sigmoid_fpga(x):
    # mirrors macarray.v hard_sigmoid: clip ±2.0, slope 0.25, offset 0.5
    return tf.clip_by_value(0.25 * x + 0.5, 0.0, 1.0)


def hard_tanh_fpga(x):
    # mirrors macarray.v hard_tanh: identity in [-1,1], clip outside
    return tf.clip_by_value(x, -1.0, 1.0)


def build_sequences(X_array, y_array, time_steps, skip_mask):
 
    X_seq, y_seq, w_idx = [], [], []

    for i in range(len(X_array) - time_steps):
        window_end = i + time_steps
        if skip_mask[i:window_end].any():
            continue
        X_seq.append(X_array[i:window_end])
        y_seq.append(y_array[window_end])
        w_idx.append(window_end)

    return np.array(X_seq), np.array(y_seq), np.array(w_idx)


def train_fpga_lstm():

    print("Loading ordered dataset...")
    df = pd.read_csv(DATA_FILE)
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")

    if 'Ah_used' not in df.columns:
        raise ValueError(
            "Ah_used column not found.\n"
            "Check your source CSV has this column — it was visible in your data."
        )

    features = ['voltage_load', 'current_load', 'temperature_battery', 'Ah_used']
    target   = ['SoC']

    print(f"\nFeatures ({len(features)}): {features}")
    print(f"Ah_used range in dataset: [{df['Ah_used'].min():.4f}, "
          f"{df['Ah_used'].max():.4f}] Ah")

    voltage_glitch_mask = (df['voltage_load'] < VOLTAGE_GLITCH_THRESHOLD).values
    print(f"\nVoltage glitch rows (< {VOLTAGE_GLITCH_THRESHOLD}V): "
          f"{voltage_glitch_mask.sum():,} ({voltage_glitch_mask.sum()/len(df)*100:.2f}%)")
    print("  Excluding these from scaler fit.")

    scaler_x = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))

    scaler_x.fit(df.loc[~voltage_glitch_mask, features])
    scaler_y.fit(df.loc[~voltage_glitch_mask, target])

    # Transform IS applied to the full array (indices must stay aligned with
    # the raw dataframe for build_sequences / skip_mask below) — glitch rows
    # get scaled too, but they'll be excluded from every window via skip_mask.
    X_scaled = scaler_x.transform(df[features])
    y_scaled = scaler_y.transform(df[target]).flatten()

    np.savez(SCALER_OUT,
             x_min=scaler_x.data_min_,
             x_max=scaler_x.data_max_,
             y_min=scaler_y.data_min_,
             y_max=scaler_y.data_max_,
             feature_names=np.array(features))
    print(f"\nScaler parameters saved to {SCALER_OUT}")
    for i, feat in enumerate(features):
        print(f"  {feat:22}: [{scaler_x.data_min_[i]:.4f}, {scaler_x.data_max_[i]:.4f}]")

    soc_raw = df['SoC'].values
    soc_diff = np.abs(np.diff(soc_raw, prepend=soc_raw[0]))
    boundary_mask = soc_diff > 0.3

    n_boundaries = boundary_mask.sum()
    print(f"\nFile boundaries detected: {n_boundaries}")
    print(f"  ({'OK' if 5 <= n_boundaries <= 20 else 'CHECK SOURCE DATA'})")

    skip_mask = boundary_mask | voltage_glitch_mask
    print(f"\nBuilding {TIME_STEPS}-step sequences with {NUM_FEATURES} features...")
    X_seq, y_seq, w_idx = build_sequences(X_scaled, y_scaled, TIME_STEPS, skip_mask)
    print(f"  Sequences built    : {len(X_seq):,}")
    print(f"  X_seq shape        : {X_seq.shape}")
    print(f"  y_seq shape        : {y_seq.shape}")

    print("\nComputing SoC sample weights...")
    soc_bins = pd.cut(pd.Series(y_seq), bins=20, labels=False)
    s_weights = compute_sample_weight('balanced', soc_bins)
    print(f"  Weight range: [{s_weights.min():.4f}, {s_weights.max():.4f}]")

    print("\nShuffling sequences...")
    shuffle_idx = np.random.RandomState(42).permutation(len(X_seq))
    X_seq     = X_seq[shuffle_idx]
    y_seq     = y_seq[shuffle_idx]
    s_weights = s_weights[shuffle_idx]
    print(f"  Shuffled {len(X_seq):,} sequences")

    inputs = Input(shape=(TIME_STEPS, NUM_FEATURES), name='sensor_input')

    x = BatchNormalization(axis=-1, name='input_batchnorm')(inputs)

    x = LSTM(
        HIDDEN_NEURONS,
        activation=hard_tanh_fpga,
        recurrent_activation=hard_sigmoid_fpga,
        kernel_constraint=MaxNorm(MAX_WEIGHT_W),
        recurrent_constraint=MaxNorm(1.5),
        bias_constraint=MaxNorm(MAX_WEIGHT_W),
        return_sequences=False,
        name='lstm_layer'
    )(x)

    outputs = Dense(
        1,
        activation=hard_sigmoid_fpga,
        kernel_constraint=MaxNorm(MAX_WEIGHT_DW),
        bias_constraint=None,
        name='soc_output'
    )(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='mse', optimizer=Adam(learning_rate=LR_INITIAL))
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=25,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=10, min_lr=1e-6, verbose=1),
        ModelCheckpoint(MODEL_OUT, monitor='val_loss',
                        save_best_only=True, verbose=1)
    ]

    print("\nStarting training...")
    history = model.fit(
        X_seq, y_seq,
        sample_weight=s_weights,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    print(f"\nModel saved to {MODEL_OUT}")

    print("\n" + "="*60)
    print("  POST-TRAINING WEIGHT CHECK")
    print("="*60)

    lstm_layer  = model.get_layer('lstm_layer')
    dense_layer = model.get_layer('soc_output')

    W  = lstm_layer.get_weights()[0]
    U  = lstm_layer.get_weights()[1]
    b  = lstm_layer.get_weights()[2]
    dW = dense_layer.get_weights()[0]
    dB = dense_layer.get_weights()[1]

    print(f"W  max={np.abs(W).max():.3f}  mean={np.abs(W).mean():.3f}  "
          f"(limit: {MAX_WEIGHT_W}) "
          f"{'OK' if np.abs(W).max() <= MAX_WEIGHT_W + 0.01 else 'EXCEEDED'}")
    print(f"U  max={np.abs(U).max():.3f}  mean={np.abs(U).mean():.3f}  "
          f"(limit: 1.5) "
          f"{'OK' if np.abs(U).max() <= 1.5 + 0.01 else 'EXCEEDED'}")
    print(f"b  max={np.abs(b).max():.3f}  mean={np.abs(b).mean():.3f}")
    print(f"dW max={np.abs(dW).max():.3f}  "
          f"(limit: {MAX_WEIGHT_DW}) "
          f"{'OK' if np.abs(dW).max() <= MAX_WEIGHT_DW + 0.01 else 'EXCEEDED'}")
    print(f"dB max={np.abs(dB).max():.3f}")

    worst_case = NUM_FEATURES * np.abs(W).max() + 16 * np.abs(U).max() + np.abs(b).max()
    print(f"\nWorst-case pre-activation : {worst_case:.2f}")
    print(f"  LUT must cover at least  : ±{np.ceil(worst_case):.0f}")

    print("\n" + "="*60)
    print("  VALIDATION ACCURACY")
    print("="*60)
    split = int(len(X_seq) * 0.8)
    X_val = X_seq[split:]; y_val = y_seq[split:]
    y_pred = model.predict(X_val, verbose=0).flatten()
    errors = np.abs(y_pred - y_val[:len(y_pred)])
    mae    = errors.mean()
    print(f"Overall MAE : {mae:.4f} ({mae*100:.2f}%)")

    regions = [
        ("0-20%  (discharged)", 0.0, 0.2),
        ("20-40% (low)",        0.2, 0.4),
        ("40-60% (mid)",        0.4, 0.6),
        ("60-80% (high)",       0.6, 0.8),
        ("80-100% (full)",      0.8, 1.0),
    ]
    y_true_all = y_val[:len(y_pred)]
    print(f"\n{'Region':<22} {'MAE':>8} {'Count':>8}")
    print("-" * 42)
    for label, lo, hi in regions:
        mask = (y_true_all >= lo) & (y_true_all < hi)
        if mask.sum() == 0:
            continue
        region_err = np.abs(y_pred[mask] - y_true_all[mask])
        print(f"{label:<22} {region_err.mean():>8.4f} {mask.sum():>8,}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history['loss'],     label='Train', color='tomato')
    axes[0].plot(history.history['val_loss'], label='Val',   color='steelblue')
    axes[0].set_title('Loss Curve (hard_tanh_fpga/hard_sigmoid_fpga)')
    axes[0].legend(); axes[0].grid(True)

    axes[1].scatter(y_true_all[:5000], y_pred[:5000], alpha=0.15, s=2, color='steelblue')
    axes[1].plot([0, 1], [0, 1], 'r--', linewidth=1)
    axes[1].set_title('Predicted vs True SoC')
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('train_qat_summary.png', dpi=150)
    print("\nPlot saved to train_qat_summary.png")


if __name__ == "__main__":
    train_fpga_lstm()
