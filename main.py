import numpy as np
from fnn import load_mnist_data_np, AutoEncoder, VariationalAutoEncoder, ConditionalVariationalAutoEncoder, VQVAE

train_auto_enc = False
train_var_enc = False
train_cond_var_enc = False
train_vq_vae = False

ae_shape = np.array([784, 256, 64])
vae_shape = np.array([784, 256, 128])
cvae_shape = np.array([784, 256, 128])
vqvae_shape = np.array([784, 256, 64])

(x_train, y_train), (x_test, y_test) = load_mnist_data_np()

x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

x_train = x_train.reshape(-1, 784, 1)
x_test = x_test.reshape(-1, 784, 1)

# Training the models
if train_auto_enc:
    auto_enc = AutoEncoder(ae_shape)
    epochs = 1
    learning_rate = 0.001
    batch_size = 64

    auto_enc.train(x_train, epochs=epochs, learning_rate=learning_rate, batch_size=batch_size)
else:
    enc_w = list(np.load("weights/auto_encoder_weights.npy", allow_pickle=True))
    enc_b = list(np.load("weights/auto_encoder_biases.npy", allow_pickle=True))
    dec_w = list(np.load("weights/auto_decoder_weights.npy", allow_pickle=True))
    dec_b = list(np.load("weights/auto_decoder_biases.npy", allow_pickle=True))

    auto_enc = AutoEncoder(ae_shape, enc_w, enc_b, dec_w, dec_b)

if train_var_enc:
    vae = VariationalAutoEncoder(vae_shape)
    epochs = 3
    learning_rate = 0.001
    beta_target = 0.1
    batch_size = 64

    vae.train(x_train, epochs=epochs, learning_rate=learning_rate, beta_target=beta_target, batch_size=batch_size)
else:
    enc_w = list(np.load("weights/var_encoder_weights.npy", allow_pickle=True))
    enc_b = list(np.load("weights/var_encoder_biases.npy", allow_pickle=True))
    dec_w = list(np.load("weights/var_decoder_weights.npy", allow_pickle=True))
    dec_b = list(np.load("weights/var_decoder_biases.npy", allow_pickle=True))

    vae = VariationalAutoEncoder(vae_shape, enc_w, enc_b, dec_w, dec_b)

if train_cond_var_enc:
    cvae = ConditionalVariationalAutoEncoder(cvae_shape, 10)
    epochs = 3
    learning_rate = 0.001
    beta_target = 0.1
    batch_size = 64

    cvae.train(x_train, y_train, epochs=epochs, learning_rate=learning_rate, beta_target=beta_target, batch_size=batch_size)
else:
    enc_w = list(np.load("weights/cond_var_encoder_weights.npy", allow_pickle=True))
    enc_b = list(np.load("weights/cond_var_encoder_biases.npy", allow_pickle=True))
    dec_w = list(np.load("weights/cond_var_decoder_weights.npy", allow_pickle=True))
    dec_b = list(np.load("weights/cond_var_decoder_biases.npy", allow_pickle=True))

    cvae = ConditionalVariationalAutoEncoder(cvae_shape, 10, enc_w, enc_b, dec_w, dec_b)

nr_of_codebook_entries = 64
if train_vq_vae:
    vqvae = VQVAE(vqvae_shape, nr_of_codebook_entries)
    epochs = 3
    learning_rate = 0.001
    beta = 0.25
    batch_size = 64

    vqvae.train(x_train,  epochs=epochs, learning_rate=learning_rate, beta=0.25, batch_size=batch_size)
else:
    enc_w = list(np.load("weights/vqvae_encoder_weights.npy", allow_pickle=True))
    enc_b = list(np.load("weights/vqvae_encoder_biases.npy", allow_pickle=True))
    dec_w = list(np.load("weights/vqvae_decoder_weights.npy", allow_pickle=True))
    dec_b = list(np.load("weights/vqvae_decoder_biases.npy", allow_pickle=True))
    loaded_codebook = list(np.load("weights/vqvae_codebook.npy", allow_pickle=True))

    vqvae = VQVAE(vqvae_shape, nr_of_codebook_entries,  enc_w, enc_b, dec_w, dec_b, loaded_codebook)

# Plots
nr_1 = x_test[16]
nr_2 = x_test[19]

activations_auto_enc_1, _ = auto_enc.encode(nr_1)
activations_auto_enc_2, _ = auto_enc.encode(nr_2)

activations_var_enc_1, _ = vae.encode(nr_1)
activations_var_enc_2, _ = vae.encode(nr_2)

# Compare reconstructions
auto_enc.show_reconstrutions(20, x_test)
vae.show_reconstrutions(20, x_test)
cvae.show_reconstrutions(20, x_test, y_test)
vqvae.show_reconstrutions(20, x_test)

# Compare transitions between AE and VAE
auto_enc.show_transition(10, activations_auto_enc_1[-1], activations_auto_enc_2[-1])
vae.show_transition(10, activations_var_enc_1[-1], activations_var_enc_2[-1])

# Compare sample grids between AE and VAE
auto_enc.show_sample_grid(5)
vae.show_sample_grid(5)