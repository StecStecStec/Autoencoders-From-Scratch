# Anomaly Detection and Generative Models for MNIST and Industrial Data

This repository contains implementations of various neural network architectures for generative modeling and anomaly detection, including AutoEncoders (AE), Variational AutoEncoders (VAE), Conditional Variational AutoEncoders (CVAE), and Vector Quantized Variational AutoEncoders (VQVAE). It provides tools for evaluating these models on MNIST and industrial time-series datasets, such as MetroPT3 air compressor sensor data.

---

## Features

### Generative Models
* **AutoEncoder (AE):** Basic feedforward autoencoder for dimensionality reduction and reconstruction.
* **Variational AutoEncoder (VAE):** Probabilistic encoder-decoder model for generating latent representations and smooth transitions.
* **Conditional VAE (CVAE):** Extends VAE to generate data conditioned on specific class labels.
* **Vector Quantized VAE (VQVAE):** Discrete latent space with codebook entries for improved reconstruction and anomaly detection.

### Anomaly Detection
* Evaluate sensor data with reconstruction-based anomaly scoring.
* Identify failures in industrial datasets using MSE (Mean Squared Error) on reconstructed windows.
* Visualizations for:
    * Reconstruction vs. original signals
    * Latent codebook usage over time
    * Smoothed anomaly trends
    * Feature-wise reconstruction comparison

---

## Repository Structure

```text
.
├── fnn.py                     # Core model implementations: AE, VAE, CVAE, VQVAE
├── mnist_training.py           # Scripts for training models on MNIST
├── industrial_data_analysis.py # Loading and preprocessing MetroPT3 AirCompressor data
├── evaluation.py               # Functions for evaluating anomaly scores and plotting
├── weights/                    # Saved weights and codebook entries for models
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## Setup Instructions
1. Installation
Clone the repository and install the required dependencies:

Bash
```
git clone <repo-url>
cd <repo-folder>
pip install -r requirements.txt
mkdir weights
```

-2. Data Preparation
Download the MNIST dataset via the built-in utility:

Python
```
from fnn import load_mnist_data_np
(x_train, y_train), (x_test, y_test) = load_mnist_data_np()
```

For industrial data, ensure the MetroPT3 CSV is present and use the preprocessing script:

Python
```
import pandas as pd
from sklearn.preprocessing import StandardScaler
df = pd.read_csv("MetroPT3(AirCompressor).csv")
```

# Use industrial_data_analysis.py for scaling and windowing
Usage Examples
Training MNIST Models
Python
```
from fnn import AutoEncoder, load_mnist_data_np

(x_train, y_train), (x_test, y_test) = load_mnist_data_np()
x_train = x_train.reshape(-1, 784, 1) / 255.0

ae = AutoEncoder([784, 256, 64])
ae.train(x_train, epochs=3, learning_rate=0.001, batch_size=64)
Evaluating Anomaly Detection
Python
from fnn import VQVAE, evaluate_model_performance

model = VQVAE([420, 256, 64], number_of_codebook_entries=64)
threshold, healthy_scores, failure_scores = evaluate_model_performance(model, X_train, X_test)
```

Visualization and Analysis
Python
```
from evaluation import plot_railway_reconstruction, plot_codebook_usage_timeline

# Plot original vs reconstructed signal
plot_railway_reconstruction(model, X_test[0], window_idx=0)

# Plot codebook usage over time
plot_codebook_usage_timeline(model, X_test)
```

Technical Notes
Incremental Training: Models support saving and loading weights from the weights/ directory.

VQVAE Advantage: The VQVAE architecture is specifically optimized for time-series anomaly detection as the discrete codebook effectively maps standard operational states.

Dependencies: Requires Python 3.10+, NumPy, Matplotlib, Pandas, and Scikit-learn.
