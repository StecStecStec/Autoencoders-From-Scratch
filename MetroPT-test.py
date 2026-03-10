import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from fnn import VQVAE

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
vqvae_shape = np.array([input_dim, 256, 64])
nr_of_codebook_entries = 64
model = VQVAE(vqvae_shape, nr_of_codebook_entries)

epochs = 3
learning_rate = 0.001
beta = 0.50
batch_size = 64

model.train(X_train,  epochs=epochs, learning_rate=learning_rate, beta=beta, batch_size=batch_size)