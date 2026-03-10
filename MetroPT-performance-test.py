import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from fnn import VQVAE, evaluate_model_performance
import matplotlib.pyplot as plt


def plot_railway_reconstruction(model, original_window, window_idx):
    mse, reconstructed, c_idx = model.calculate_anomaly_score(original_window)

    original_2d = original_window.reshape(60, 7)
    reconstructed_2d = reconstructed.reshape(60, 7)

    plt.figure(figsize=(10, 4))
    plt.plot(original_2d[:, 0], label='Original TP2 (Pressure)', color='blue', alpha=0.7)
    plt.plot(reconstructed_2d[:, 0], label='Reconstructed', color='orange', linestyle='--')

    plt.title(f"Window {window_idx} - MSE: {mse:.4f} - Codebook Index: {c_idx}")
    plt.xlabel("Time steps (10s intervals)")
    plt.ylabel("Standardized Pressure")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_codebook_usage_timeline(model, data_set):
    indices = []
    mse_scores = []

    for sample in data_set:
        sample_col = sample.reshape(-1, 1)
        mse, _, c_idx = model.calculate_anomaly_score(sample_col)
        indices.append(c_idx)
        mse_scores.append(mse)

    fig, ax1 = plt.subplots(figsize=(15, 6))

    ax1.scatter(range(len(indices)), indices, alpha=0.4, s=12, c=indices, cmap='viridis')
    ax1.set_xlabel("Window Sequence")
    ax1.set_ylabel("Codebook Index (0-63)")
    ax1.set_yticks(range(0, 128, 8))
    ax1.grid(True, linestyle='--', alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(range(len(mse_scores)), mse_scores, color='red', linewidth=1, alpha=0.6)
    ax2.set_ylabel("Reconstruction MSE", color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    plt.title("Latent State Migration and Reconstruction Error")
    plt.tight_layout()
    plt.show()


def plot_smoothed_health(scores, window_size=50):
    s = pd.Series(scores)
    smoothed = s.rolling(window=window_size).mean()

    plt.figure(figsize=(15, 5))
    plt.plot(scores, alpha=0.2, color='gray', label='Raw MSE (Noise)')
    plt.plot(smoothed, color='red', linewidth=2, label=f'{window_size}-Window Trend')
    plt.axhline(y=threshold, color='black', linestyle='--', label='Alert Threshold')

    plt.title("Trend Analysis: Smoothing out the 'Jitter'")
    plt.ylabel("Filtered Anomaly Score")
    plt.legend()
    plt.show()


def plot_all_features(model, original_window, window_idx):
    mse, reconstructed, c_idx = model.calculate_anomaly_score(original_window)

    original_2d = original_window.reshape(60, 7)
    reconstructed_2d = reconstructed.reshape(60, 7)

    sensor_names = ['TP2', 'TP3', 'H1', 'DV_pressure', 'Reservoirs', 'Oil_temperature', 'Motor_current']

    fig, axes = plt.subplots(nrows=7, ncols=1, figsize=(12, 18), sharex=True)
    fig.suptitle(f"Window {window_idx} Analysis | Total MSE: {mse:.4f} | Codebook ID: {c_idx}", fontsize=16)

    for i in range(7):
        ax = axes[i]
        ax.plot(original_2d[:, i], label='Actual', color='blue', alpha=0.6)
        ax.plot(reconstructed_2d[:, i], label='Reconstructed', color='orange', linestyle='--')

        ax.set_ylabel(sensor_names[i])
        ax.legend(loc='upper right', fontsize='small')
        ax.grid(True, alpha=0.2)

    plt.xlabel("Time Steps (10s intervals)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

df = pd.read_csv("MetroPT3(AirCompressor).csv")

# Split into 2 datasets - healthy and unhealthy
failure_intervals = [
    ("2020-04-18 00:00:00", "2020-04-18 23:59:00"),
    ("2020-05-29 23:30:00", "2020-05-30 06:00:00"),
    ("2020-06-05 10:00:00", "2020-06-07 14:30:00"),
    ("2020-07-15 14:30:00", "2020-07-15 19:00:00")
]

mask = pd.Series(False, index=df.index)
for start, end in failure_intervals:
    mask |= (df['timestamp'] >= start) & (df['timestamp'] <= end)

df_failures = df[mask]
df_healthy = df[~mask]

# Remove unnecessary columns and empty spaces
df_healthy = df_healthy.drop(columns=['timestamp', 'Unnamed: 0']).dropna()
df_failures = df_failures.drop(columns=['timestamp', 'Unnamed: 0']).dropna()

# Scale
analog_sensors = ['TP2', 'TP3', 'H1', 'DV_pressure', 'Reservoirs', 'Oil_temperature', 'Motor_current']
digital_signals = ['COMP', 'DV_eletric', 'Towers', 'MPG', 'LPS', 'Pressure_switch', 'Oil_level']
scaler = StandardScaler()

train_analog = scaler.fit_transform(df_healthy[analog_sensors])
test_analog = scaler.transform(df_failures[analog_sensors])

train_combined = np.hstack([train_analog, df_healthy[digital_signals].values])
test_combined = np.hstack([test_analog, df_failures[digital_signals].values])

window_size = 60
def create_dataset(data, window_size=window_size):
    windows = []
    for i in range(len(data) - window_size):
        window = data[i:i + window_size].flatten()
        windows.append(window)
    return np.array(windows)

X_train = create_dataset(train_analog)
X_test = create_dataset(test_analog)

# Set up the model
input_dim = window_size * len(analog_sensors)
vqvae_shape = np.array([input_dim, 256, 128])
nr_of_codebook_entries = 128

enc_w = [w.astype(np.float64) for w in np.load("weights/vqvae_encoder_weights_v2.npy", allow_pickle=True)]
enc_b = [b.astype(np.float64) for b in np.load("weights/vqvae_encoder_biases_v2.npy", allow_pickle=True)]
dec_w = [w.astype(np.float64) for w in np.load("weights/vqvae_decoder_weights_v2.npy", allow_pickle=True)]
dec_b = [b.astype(np.float64) for b in np.load("weights/vqvae_decoder_biases_v2.npy", allow_pickle=True)]
loaded_codebook = np.load("weights/vqvae_codebook.npy", allow_pickle=True).astype(np.float64)

model = VQVAE(vqvae_shape, nr_of_codebook_entries,  enc_w, enc_b, dec_w, dec_b, loaded_codebook)

plot_all_features(model, X_train[100], 100)
plot_all_features(model, X_test[100], 100)

threshold, h_scores, f_scores = evaluate_model_performance(model, X_train, X_test)

codebook_check_set = df.dropna()
codebook_check_set["timestamp"] = pd.to_datetime(codebook_check_set["timestamp"])

cutoff = pd.Timestamp("2020-09-01 03:59:50")

codebook_check_set = codebook_check_set[codebook_check_set["timestamp"] <= cutoff]
codebook_check_set = codebook_check_set.drop(columns=['timestamp', 'Unnamed: 0'])
check_set_scaled = scaler.transform(codebook_check_set[analog_sensors])
X_check_windows = create_dataset(check_set_scaled)

plot_codebook_usage_timeline(model, X_check_windows)
plot_smoothed_health(f_scores)