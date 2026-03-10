# VarEnc: Variational Autoencoders for Anomaly Detection

This repository implements several autoencoder-based models for unsupervised anomaly detection, with a focus on variational and vector‑quantized architectures. It includes training on MNIST for demonstration and application to the **MetroPT3 (Air Compressor) dataset** for predictive maintenance.

## Implemented Models

- **AutoEncoder (AE)** – standard feedforward autoencoder.
- **Variational AutoEncoder (VAE)** – learns a probabilistic latent space.
- **Conditional Variational AutoEncoder (CVAE)** – conditions on class labels.
- **Vector‑Quantized Variational AutoEncoder (VQ‑VAE)** – discrete latent representations via a codebook.

All models are built from scratch using NumPy (no deep learning framework).

## Project Structure

- `fnn.py` – core classes: `FNN` (flexible feedforward network), `AutoEncoder`, `VariationalAutoEncoder`, `ConditionalVariationalAutoEncoder`, `VQVAE`.  
- `main.py` – trains/evaluates all models on MNIST, saves/loads weights, and generates reconstruction/transition plots.  
- `metropt-test.py` – trains a VQ‑VAE on the MetroPT3 dataset (analog sensor windows) for anomaly detection.  
- `metropt-performance-test.py` – evaluates the trained VQ‑VAE on MetroPT3, plots reconstructions, anomaly scores, and codebook usage over time.  
- `weights/` – directory where trained model weights (`.npy` files) are saved/loaded.

## Dependencies

- Python 3.x
- NumPy
- Matplotlib
- Pandas
- scikit‑learn

Install with:
```bash
pip install numpy matplotlib pandas scikit-learn
```
# Usage

## MNIST Demo
Run main.py to train all models or load pre‑trained weights (set the corresponding train_* flags). After training/evaluation, it displays:

- Reconstructions
- Latent space interpolations
- Random samples from the prior

## MetroPT3 Anomaly Detection
1. Place MetroPT3(AirCompressor).csv in the project folder.
2. Run metropt-test.py to train a VQ‑VAE on healthy data windows (analog sensors scaled, window size = 60 time steps).
3. Run metropt-performance-test.py to evaluate the model on failure periods and visualize:
   - Reconstruction of individual windows
   - Codebook index migration over time
   - Smoothed anomaly score trends
Weights for the MetroPT3 model are saved in the weights/ folder with version suffixes (e.g., v2).

## Key Features
- Fully NumPy‑based backpropagation with custom gradients.
- Support for different output activations (sigmoid, tanh, linear, softmax).
- Modular design – easy to extend with new architectures.
- Visualization utilities for reconstructions, latent transitions, and anomaly score distributions.

## Notes
- The MetroPT3 dataset contains telemetry from an air compressor; failure intervals are hardcoded based on domain knowledge.
- For VQ‑VAE, the codebook is updated during training using a simple vector quantization loss.
- Conditional VAE concatenates one‑hot labels to both encoder input and decoder input.
