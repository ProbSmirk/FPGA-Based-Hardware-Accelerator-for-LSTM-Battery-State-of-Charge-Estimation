
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_sample_weight #estimates sample weights for unbalanced datasets by assigning a weight to every individual row (sample) in your target array

INPUT_FILE       = r'C:\Users\sumuk\PycharmMiscProject\batteryfiles\Master_Training_Data_Light.csv'
OUTPUT_SHUFFLED  = r'C:\Users\sumuk\PycharmMiscProject\batteryfiles\Master_Training_Data_Balanced.csv'
OUTPUT_ORDERED   = r'C:\Users\sumuk\PycharmMiscProject\batteryfiles\Master_Training_Data_Ordered.csv'

IDLE_CURRENT_THRESHOLD = 0.5
IDLE_KEEP_FRACTION     = 0.20  # keep 20% of idle samples to prevent overfitting
CURRENT_CLIP_MAX       = 5.0   # amps — clips rare 25A outliers, capping outliers is highly beneficial for fixed-point hardware implementation later, as it prevents the MinMaxScaler from squashing normal data into tiny ranges to accommodate a rare 25A spike.
SAMPLE_RATE_HZ   = 1.0        #value of dt as soc is integral of current with dt in coloumb counting
CC_BOUNDARY_THRESHOLD = 0.3 #if the SOC jumps by more than this coloumb counting as it shows start of a new file


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")# used for printing


def show_distributions(df, title):
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] #divides soc into different chunks of data
    counts = pd.cut(df['SoC'], bins=bins).value_counts().sort_index() #puts the values in these chunks and counts and sorts how much in each
    print(f"\n{title} — SoC bin counts:")
    for interval, count in counts.items():
        bar = '█' * (count // max(1, counts.max() // 30))
        print(f"  {str(interval):20} {count:8,}  {bar}")
    print(f"\n  Current stats:")
    print(f"    near-zero (<{IDLE_CURRENT_THRESHOLD}A): "
          f"{(df['current_load'].abs() < IDLE_CURRENT_THRESHOLD).sum():,} samples "#counts how many samples in the resting phase
          f"({(df['current_load'].abs() < IDLE_CURRENT_THRESHOLD).mean()*100:.1f}%)")
    print(f"    max: {df['current_load'].max():.2f}A   "
          f"mean: {df['current_load'].mean():.3f}A   "
          f"std: {df['current_load'].std():.3f}A")


def compute_coulomb_counting(df, boundary_threshold, sample_rate_hz):
   
    dt = 1.0 / sample_rate_hz
    current = df['current_load'].values
    soc     = df['SoC'].values

    cc = np.zeros(len(df))
    accumulator = 0.0

    for i in range(1, len(df)):
        # Detect file boundary — SoC jump larger than threshold
        if abs(soc[i] - soc[i-1]) > boundary_threshold:
            accumulator = 0.0   # reset CC at each new battery file

        # Trapezoidal integration: area = (I[i-1] + I[i]) / 2 * dt
        accumulator += 0.5 * (current[i-1] + current[i]) * dt
        cc[i] = accumulator

    return cc


print_section("STEP 0 — Loading data")
df = pd.read_csv(INPUT_FILE)
print(f"Loaded {len(df):,} rows from {os.path.basename(INPUT_FILE)}")
print(f"Columns: {list(df.columns)}")
show_distributions(df, "ORIGINAL")


print_section("STEP 1 — Downsampling idle samples")

idle_mask   = df['current_load'].abs() < IDLE_CURRENT_THRESHOLD
active_mask = ~idle_mask

df_idle   = df[idle_mask]#sorted into 2 buckets
df_active = df[active_mask]

print(f"Idle samples   (|I| < {IDLE_CURRENT_THRESHOLD}A): {len(df_idle):,}")
print(f"Active samples (|I| >= {IDLE_CURRENT_THRESHOLD}A): {len(df_active):,}")

df_idle_downsampled = df_idle.sample(frac=IDLE_KEEP_FRACTION, random_state=42)
print(f"Kept {IDLE_KEEP_FRACTION*100:.0f}% of idle -> {len(df_idle_downsampled):,} samples") #keep 20% of idle bucket

# Combine but preserve original row order for sequencing and CC computation
df_combined = pd.concat([df_idle_downsampled, df_active], ignore_index=False)
df_combined = df_combined.sort_index() #combines and sorts in original row order
df_combined = df_combined.reset_index(drop=True)

print(f"After Step 1: {len(df_combined):,} total samples (temporally ordered)")
show_distributions(df_combined, "AFTER STEP 1")
#In real-world battery data, a battery might sit at 100% (fully charged) or 0% (fully depleted) for hours, but transition through the middle ranges (like 40% to 60%) relatively quickly. If you feed that raw data into a neural network, the model becomes biased. It gets mathematically "lazy" and becomes excellent at predicting 100% and 0%, but terrible at predicting the middle ranges, simply because it saw the extremes far more often.
print_section("STEP 2 — SoC rebalancing check")

df_combined['_soc_bin'] = pd.cut(df_combined['SoC'], bins=20, labels=False)#creates 20 buckets from 0 to 1
sample_weights_check = compute_sample_weight('balanced', df_combined['_soc_bin'])#weights assigned according how much each bucket present, low for buckets with many data points and vice versa, so that loss function has same values
df_combined = df_combined.drop(columns=['_soc_bin'])

min_w  = sample_weights_check.min()
max_w  = sample_weights_check.max() #checks ratio between most common battery state and least common and we want ratio to be <20 so that model can learn
mean_w = sample_weights_check.mean()
print(f"Sample weight range: min={min_w:.4f}  mean={mean_w:.4f}  max={max_w:.4f}")
print(f"Ratio max/min = {max_w/min_w:.1f}x")
print(f"(Target: ratio < 20)")
print(f"\nWeights are NOT saved to CSV — recomputed fresh inside train_model.py.")


print_section("STEP 3 — Current clipping (discrete lab data, log1p not applied)")

print("Unique current levels found in dataset:")
value_counts = df_combined['current_load'].round(1).value_counts().sort_index() #round off to 1 decimal place and then sort
print(value_counts.to_string())

df_combined['current_load'] = df_combined['current_load'].clip(
    lower=-CURRENT_CLIP_MAX,
    upper= CURRENT_CLIP_MAX #clips current to between 5 and -5
)

print(f"\nAfter clip: [{df_combined['current_load'].min():.3f}, "
      f"{df_combined['current_load'].max():.3f}]")


print_section("STEP 4 — Ah_used column (Coulomb counting from test equipment)")

print(f"Ah_used range: [{df_combined['Ah_used'].min():.4f}, "
      f"{df_combined['Ah_used'].max():.4f}] Ah")
print(f"Ah_used zeros: {(df_combined['Ah_used'] == 0).sum():,} rows") #checks if coloumb counting column has been used or not
print(f"Ah_used non-zero: {(df_combined['Ah_used'] != 0).sum():,} rows")

# No transformation needed — MinMaxScaler handles normalisation in train_model.py
# Just confirm the column is present and has meaningful values
sample_corr = df_combined[['Ah_used', 'SoC']].sample(10000, random_state=1).corr()
print(f"\nCorrelation between Ah_used and SoC: "
      f"{sample_corr.loc['Ah_used', 'SoC']:.4f}")
print("(Expected: negative — more Ah used = lower SoC)")

print_section("SAVING")

print(f"Columns in output: {list(df_combined.columns)}")

df_combined.to_csv(OUTPUT_ORDERED, index=False)
print(f"\nOrdered CSV saved ({len(df_combined):,} rows):")
print(f"  {OUTPUT_ORDERED}")
print(f"  <- USE THIS FILE in train_model.py")
print(f"  Features: voltage_load, current_load, temperature_battery, coulomb_count")

df_shuffled = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
df_shuffled.to_csv(OUTPUT_SHUFFLED, index=False)
print(f"\nShuffled CSV saved ({len(df_shuffled):,} rows):")
print(f"  {OUTPUT_SHUFFLED}")
print(f"  <- reference only, do NOT use for LSTM training")


print_section("VERIFICATION PLOTS")
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
fig.suptitle('Dataset Before vs After Balancing (with Coulomb Counting)', fontsize=13)

df_orig = pd.read_csv(INPUT_FILE)

axes[0, 0].hist(df_orig['SoC'], bins=50, color='tomato', alpha=0.7)
axes[0, 0].set_title('SoC — Original')
axes[0, 0].set_xlabel('SoC'); axes[0, 0].set_ylabel('Count')

axes[0, 1].hist(df_orig['current_load'].clip(-CURRENT_CLIP_MAX, CURRENT_CLIP_MAX),
                bins=30, color='tomato', alpha=0.7)
axes[0, 1].set_title('Current — Original')
axes[0, 1].set_xlabel('Current (A)')

sample_orig = df_orig.sample(min(5000, len(df_orig)), random_state=1)
axes[0, 2].scatter(sample_orig['current_load'].clip(-5, 25),
                   sample_orig['SoC'], alpha=0.1, s=1, color='tomato')
axes[0, 2].set_title('Current vs SoC — Original')
axes[0, 2].set_xlabel('Current (A)'); axes[0, 2].set_ylabel('SoC')

axes[0, 3].set_visible(False)  # placeholder

axes[1, 0].hist(df_combined['SoC'], bins=50, color='steelblue', alpha=0.7)
axes[1, 0].set_title('SoC — Balanced')
axes[1, 0].set_xlabel('SoC'); axes[1, 0].set_ylabel('Count')

axes[1, 1].hist(df_combined['current_load'], bins=30, color='steelblue', alpha=0.7)
axes[1, 1].set_title(f'Current — Clipped ±{CURRENT_CLIP_MAX}A')
axes[1, 1].set_xlabel('Current (A)')

sample_bal = df_combined.sample(min(5000, len(df_combined)), random_state=1)
axes[1, 2].scatter(sample_bal['current_load'],
                   sample_bal['SoC'], alpha=0.1, s=1, color='steelblue')
axes[1, 2].set_title('Current vs SoC — Balanced')
axes[1, 2].set_xlabel('Current (A)'); axes[1, 2].set_ylabel('SoC')

axes[1, 3].scatter(sample_bal['Ah_used'],
                   sample_bal['SoC'], alpha=0.1, s=1, color='darkorange')
axes[1, 3].set_title('Ah_used vs SoC (Coulomb Count)')
axes[1, 3].set_xlabel('Ah_used'); axes[1, 3].set_ylabel('SoC')

plt.tight_layout()
plt.show()

print_section("TEMPORAL ORDER SANITY CHECK")
soc_diff = df_combined['SoC'].diff().abs()
big_jumps = (soc_diff > 0.3).sum()
print(f"SoC jumps > 0.3: {big_jumps} (expected ~10 for 11 source files)")
print(f"Status: {'OK' if big_jumps < 50 else 'HIGH — check source data order'}")

print(f"\nAh_used feature sanity:")
ah_start = df_combined['Ah_used'].iloc[0]
ah_max   = df_combined['Ah_used'].abs().max()
print(f"  Ah_used at row 0       : {ah_start:.4f} Ah")
print(f"  Ah_used max magnitude  : {ah_max:.2f} Ah "
      f"({'reasonable' if ah_max < 5000 else 'very large — check units'})")
ah_soc_corr = df_combined[['Ah_used', 'SoC']].corr().loc['Ah_used', 'SoC']
print(f"  Ah_used / SoC correlation: {ah_soc_corr:.4f} "
      f"(expected negative — more Ah used = lower SoC)")

print("\nDone. Use Master_Training_Data_Ordered.csv in train_model.py.")
