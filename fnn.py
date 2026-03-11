import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import gzip
import os


def evaluate_model_performance(model, x_healthy, x_failures):
    def get_errors(data):
        scores = []
        for sample in data:
            mse, _, _ = model.calculate_anomaly_score(sample)
            scores.append(mse)
        return np.array(scores)

    print("Calculating scores for Healthy data...")
    healthy_scores = get_errors(x_healthy[:5000])

    print("Calculating scores for Failure data...")
    failure_scores = get_errors(x_failures[:5000])

    threshold = np.mean(healthy_scores) + 3 * np.std(healthy_scores)

    plt.figure(figsize=(12, 6))
    plt.hist(healthy_scores, bins=50, alpha=0.5, label='Healthy (Normal)', color='green')
    plt.hist(failure_scores, bins=50, alpha=0.5, label='Failures (Anomalies)', color='red')
    plt.axvline(threshold, color='black', linestyle='--', label=f'Threshold (3σ): {threshold:.4f}')

    plt.title("Anomaly Score Distribution: Healthy vs. Failure")
    plt.xlabel("Reconstruction MSE")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

    return threshold, healthy_scores, failure_scores

def load_mnist_data_np():
    base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    files = [
        "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"
    ]

    def _load_data(file_path, is_images):
        with gzip.open(file_path, 'rb') as f:
            _ = np.frombuffer(f.read(4), dtype=np.uint32).byteswap()
            count = np.frombuffer(f.read(4), dtype=np.uint32).byteswap()[0]

            if is_images:
                rows = np.frombuffer(f.read(4), dtype=np.uint32).byteswap()[0]
                cols = np.frombuffer(f.read(4), dtype=np.uint32).byteswap()[0]
                data = np.frombuffer(f.read(), dtype=np.uint8)
                return data.reshape(count, rows, cols)
            else:
                data = np.frombuffer(f.read(), dtype=np.uint8)
                return data

    for file_name in files:
        if not os.path.exists(file_name):
            print(f"Downloading {file_name}...")
            urllib.request.urlretrieve(base_url + file_name, file_name)

    x_train = _load_data(files[0], is_images=True)
    y_train = _load_data(files[1], is_images=False)
    x_test = _load_data(files[2], is_images=True)
    y_test = _load_data(files[3], is_images=False)

    return (x_train, y_train), (x_test, y_test)

class FNN:
    def __init__(self, network_shape: np.ndarray,
                 loaded_weights: list = None,
                 loaded_biases: list = None):
        self.network_shape = np.array(network_shape, dtype=int)
        if loaded_weights is None:
            self.weights = []
            for in_size, out_size in zip(self.network_shape[:-1], self.network_shape[1:]):
                w = np.random.randn(out_size, in_size) / np.sqrt(in_size)
                self.weights.append(w)
        else:
            self.weights = loaded_weights

        if loaded_biases is None:
            self.biases = [np.zeros((out_size, 1)) for out_size in self.network_shape[1:]]
        else:
            self.biases = loaded_biases

    def sigmoid(self, x):
        x = np.asarray(x, dtype=np.float64)
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))

    def tanh(self, x):
        x = np.clip(x, -500, 500)
        return np.tanh(x)

    def sigmoid_derivative_from_activation(self, s):
        return s * (1.0 - s)

    def softmax(self, x):
        shift = x - np.max(x, axis=0, keepdims=True)
        exp_x = np.exp(shift)
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)

    def forward_pass(self,input,
                     final_activation_function=None):
        activation_functions = {
            None: self.sigmoid,
            "sigmoid": self.sigmoid,
            "tanh": self.tanh,
            "softmax": self.softmax,
            "linear": lambda x: x
        }

        if final_activation_function not in activation_functions:
            raise ValueError(
                f"final_activation_function must be one of {list(activation_functions.keys())}"
            )

        activations = [input]
        zs = []
        L = len(self.weights)
        for i in range(L):
            z = self.weights[i] @ activations[-1] + self.biases[i]
            zs.append(z)
            if i == L - 1:
                if final_activation_function == "softmax":
                    a = self.softmax(z)
                elif final_activation_function == "linear":
                    a = z
                elif final_activation_function == "tanh":
                    a = self.tanh(z)
                elif final_activation_function == "sigmoid":
                    a = self.sigmoid(z)
                else:
                    a = self.sigmoid(z)
            else:
                a = self.sigmoid(z)
            activations.append(a)
        return activations, zs

    def backpropagation(self, input,
                        y_or_delta,
                        learning_rate=0.01,
                        return_input_grad=False,
                        output_activation='sigmoid',
                        provided_delta=False):
        activations, zs = self.forward_pass(input,
                                            final_activation_function=output_activation)
        L = len(self.weights)
        grad_w = [np.zeros_like(w) for w in self.weights]
        grad_b = [np.zeros_like(b) for b in self.biases]

        if provided_delta:
            delta = y_or_delta
        else:
            aL = activations[-1]
            if output_activation == 'linear':
                delta = (aL - y_or_delta)
            elif output_activation == 'softmax':
                delta = (aL - y_or_delta)
            elif output_activation == 'tanh':
                delta = (aL - y_or_delta) * (1.0 - aL**2)
            else:
                delta = (aL - y_or_delta) * self.sigmoid_derivative_from_activation(aL)

        grad_w[L - 1] = delta @ activations[L - 1].T
        grad_b[L - 1] = np.sum(delta, axis=1, keepdims=True)

        for l in range(L - 2, -1, -1):
            a_l_plus_1 = activations[l + 1]
            delta = (self.weights[l + 1].T @ delta) * self.sigmoid_derivative_from_activation(a_l_plus_1)
            grad_w[l] = delta @ activations[l].T
            grad_b[l] = np.sum(delta, axis=1, keepdims=True)

        for l in range(L):
            self.weights[l] -= learning_rate * grad_w[l]
            self.biases[l] -= learning_rate * grad_b[l]

        if return_input_grad:
            return self.weights[0].T @ delta

class AutoEncoder:
    def __init__(self, network_shape: np.ndarray,
                 loaded_encoder_weights: list = None,
                 loaded_encoder_biases: list = None,
                 loaded_decoder_weights: list = None,
                 loaded_decoder_biases: list = None):
        network_shape = np.array(network_shape, dtype=int)
        self.encoder_shape = network_shape
        self.encoder = FNN(self.encoder_shape, loaded_encoder_weights, loaded_encoder_biases)
        self.decoder_shape = np.flip(network_shape, 0).astype(int)
        self.decoder = FNN(self.decoder_shape, loaded_decoder_weights, loaded_decoder_biases)

    def encode(self, input):
        return self.encoder.forward_pass(input, final_activation_function="linear")

    def decode(self, input):
        return self.decoder.forward_pass(input)

    def full_backpropagation(self, input_vector,
                             learning_rate=0.01):
        activations_enc, _ = self.encode(input_vector)
        enc_out = activations_enc[-1]

        activations_dec, _ = self.decode(enc_out)
        reconstructed_output = activations_dec[-1]

        d_recon_da = (reconstructed_output - input_vector)

        d_decoder = self.decoder.backpropagation(
            enc_out,
            d_recon_da,
            learning_rate=learning_rate,
            return_input_grad=True,
            output_activation='sigmoid',
            provided_delta=True
        )

        self.encoder.backpropagation(
            input_vector, d_decoder, learning_rate=learning_rate,
            return_input_grad=False, output_activation='linear', provided_delta=True
        )

        reconstruction_loss = -np.sum(
            input_vector * np.log(reconstructed_output + 1e-8) +
            (1 - input_vector) * np.log(1 - reconstructed_output + 1e-8)
        )

        return reconstruction_loss

    def train(self, x_train, epochs, learning_rate, batch_size):
        for epoch in range(epochs):
            total_recon_loss = 0.0

            indices = np.random.permutation(len(x_train))
            x_train_shuffled = x_train[indices]

            num_batches = len(x_train) // batch_size

            for i in range(0, len(x_train), batch_size):
                batch = x_train_shuffled[i:i + batch_size]
                batch_recon_loss = 0.0

                for sample in batch:
                    recon_loss = self.full_backpropagation(
                        sample, learning_rate=learning_rate
                    )
                    batch_recon_loss += recon_loss

                print(f"{i}")

                total_recon_loss += batch_recon_loss / batch_size

            avg_recon = total_recon_loss / num_batches
            print(
                f"Epoch {epoch + 1}/{epochs} - Avg Recon Loss: {avg_recon:.4f}")

        np.save("weights/auto_encoder_weights.npy", np.array(self.encoder.weights, dtype=object))
        np.save("weights/auto_encoder_biases.npy", np.array(self.encoder.biases, dtype=object))
        np.save("weights/auto_decoder_weights.npy", np.array(self.decoder.weights, dtype=object))
        np.save("weights/auto_decoder_biases.npy", np.array(self.decoder.biases, dtype=object))

    def show_reconstrutions(self, nr_of_examples, mnist_data):
        fig, axes = plt.subplots(2, nr_of_examples, figsize=(15, 6))
        for i in range(nr_of_examples):
            test_sample = mnist_data[i]
            activations_enc, _ = self.encode(test_sample)

            axes[0, i].imshow(test_sample.reshape(28, 28), cmap='gray')
            axes[0, i].axis('off')
            axes[0, i].set_title(f'i={i}')

            activations_dec, _ = self.decode(activations_enc[-1])
            reconstructed = activations_dec[-1].reshape(28, 28)

            axes[1, i].imshow(reconstructed, cmap='gray')
            axes[1, i].axis('off')
            axes[1, i].set_title(f'R{i}')

        plt.tight_layout()
        plt.show()

    def show_sample_grid(self, length):
        N = int(length**2)
        grid_size = length

        latent_dim_ae = int(self.decoder_shape[0])

        z_ae = np.random.randn(latent_dim_ae, 1, N)
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
        axes = axes.flatten()
        for i in range(N):
            latent_vector = z_ae[:, 0, i].reshape(-1, 1)
            activations, _ = self.decode(latent_vector)
            img = activations[-1].reshape(28, 28)
            axes[i].imshow(img, cmap='gray')
            axes[i].axis('off')
        plt.tight_layout()
        plt.show()

    def show_transition(self, nr_of_steps, latent_vector_1, latent_vector_2):
        fig, axes = plt.subplots(1, nr_of_steps + 2, figsize=(10, 4))
        difference = np.array(latent_vector_2) - np.array(latent_vector_1)

        for i in range(0, nr_of_steps + 2):
            latent_step = latent_vector_1 + ((difference / 10) * i)

            activations_dec, _ = self.decode(latent_step)
            reconstructed = activations_dec[-1].reshape(28, 28)

            axes[i].imshow(reconstructed, cmap='gray')
            axes[i].axis('off')
            axes[i].set_title(f'Step {i}')

        plt.tight_layout()
        plt.show()

class VariationalAutoEncoder:
    def __init__(self, network_shape: np.ndarray,
                 loaded_encoder_weights: list = None,
                 loaded_encoder_biases: list = None,
                 loaded_decoder_weights: list = None,
                 loaded_decoder_biases: list = None):
        network_shape = np.array(network_shape, dtype=int)
        self.latent_dim = max(1, int(network_shape[-1]) // 2)
        self.encoder_shape = network_shape
        self.encoder = FNN(self.encoder_shape, loaded_encoder_weights, loaded_encoder_biases)
        self.decoder_shape = np.flip(network_shape, 0).astype(int)
        self.decoder_shape[0] = self.latent_dim
        self.decoder = FNN(self.decoder_shape, loaded_decoder_weights, loaded_decoder_biases)


    def encode(self, input):
        return self.encoder.forward_pass(input, final_activation_function="linear")

    def decode(self, input):
        return self.decoder.forward_pass(input)

    def full_backpropagation(self, input_vector,
                             learning_rate=0.01,
                             beta=1.0):
        activations_enc, _ = self.encode(input_vector)
        enc_out = activations_enc[-1]
        mu, log_var = np.split(enc_out, 2, axis=0)
        sigma = np.exp(0.5 * log_var)
        epsilon = np.random.randn(*mu.shape)
        latent_sample = mu + sigma * epsilon

        activations_dec, _ = self.decode(latent_sample)
        reconstructed_output = activations_dec[-1]

        reconstruction_loss = -np.sum(
            input_vector * np.log(reconstructed_output + 1e-8) +
            (1 - input_vector) * np.log(1 - reconstructed_output + 1e-8)
        )

        kl_loss = -0.5 * np.sum(1 + log_var - mu ** 2 - np.exp(log_var))

        total_loss = reconstruction_loss + beta * kl_loss

        d_recon_da = (reconstructed_output - input_vector)

        dlatent = self.decoder.backpropagation(
            latent_sample,
            d_recon_da,
            learning_rate=learning_rate,
            return_input_grad=True,
            output_activation='sigmoid',
            provided_delta=True
        )

        dmu_from_decoder = dlatent
        dlogvar_from_decoder = dlatent * epsilon * 0.5 * sigma

        dKL_dmu = mu
        dKL_dlogvar = 0.5 * (np.exp(log_var) - 1)

        dmu = dmu_from_decoder + beta * dKL_dmu
        dlogvar = dlogvar_from_decoder + beta * dKL_dlogvar

        encoder_output_grad = np.vstack([dmu, dlogvar])

        self.encoder.backpropagation(
            input_vector, encoder_output_grad, learning_rate=learning_rate,
            return_input_grad=False, output_activation='linear', provided_delta=True
        )

        return reconstruction_loss, kl_loss, total_loss

    def train(self, x_train, epochs, learning_rate, beta_target, batch_size):
        for epoch in range(epochs):
            total_recon_loss = 0.0
            total_kl_loss = 0.0
            total_loss_sum = 0.0
            beta = beta_target * (epoch + 1) / epochs

            indices = np.random.permutation(len(x_train))
            x_train_shuffled = x_train[indices]

            num_batches = len(x_train) // batch_size

            for i in range(0, len(x_train), batch_size):
                batch = x_train_shuffled[i:i + batch_size]
                batch_recon_loss = 0.0
                batch_kl_loss = 0.0
                batch_total_loss = 0.0

                for sample in batch:
                    recon_loss, kl_loss, total_loss = self.full_backpropagation(
                        sample, learning_rate=learning_rate, beta=beta
                    )
                    batch_recon_loss += recon_loss
                    batch_kl_loss += kl_loss
                    batch_total_loss += total_loss

                print(f"{i}")

                total_recon_loss += batch_recon_loss / batch_size
                total_kl_loss += batch_kl_loss / batch_size
                total_loss_sum += batch_total_loss / batch_size

            avg_recon = total_recon_loss / num_batches
            avg_kl = total_kl_loss / num_batches
            avg_total = total_loss_sum / num_batches
            print(
                f"Epoch {epoch + 1}/{epochs} - Avg Recon Loss: {avg_recon:.4f}, Avg KL Loss: {avg_kl:.4f}, Avg Total Loss: {avg_total:.4f}")

        np.save("weights/var_encoder_weights.npy", np.array(self.encoder.weights, dtype=object))
        np.save("weights/var_encoder_biases.npy", np.array(self.encoder.biases, dtype=object))
        np.save("weights/var_decoder_weights.npy", np.array(self.decoder.weights, dtype=object))
        np.save("weights/var_decoder_biases.npy", np.array(self.decoder.biases, dtype=object))

    def show_reconstrutions(self, nr_of_examples, mnist_data):
        fig, axes = plt.subplots(2, nr_of_examples, figsize=(15, 6))
        for i in range(nr_of_examples):
            test_sample = mnist_data[i]
            activations_enc, _ = self.encode(test_sample)
            mu, log_var = np.split(activations_enc[-1], 2, axis=0)
            sigma = np.exp(0.5 * log_var)

            axes[0, i].imshow(test_sample.reshape(28, 28), cmap='gray')
            axes[0, i].axis('off')
            axes[0, i].set_title(f'i={i}')

            epsilon = np.random.randn(*mu.shape)
            latent = mu + sigma * epsilon

            activations_dec, _ = self.decode(latent)
            reconstructed = activations_dec[-1].reshape(28, 28)

            axes[1, i].imshow(reconstructed, cmap='gray')
            axes[1, i].axis('off')
            axes[1, i].set_title(f'R{i}')

        plt.tight_layout()
        plt.show()

    def show_transition(self, nr_of_steps, latent_vector_1, latent_vector_2):
        fig, axes = plt.subplots(1, nr_of_steps + 2, figsize=(10, 4))
        difference = np.array(latent_vector_2) - np.array(latent_vector_1)

        for i in range(0, nr_of_steps + 2):
            mu, log_var = np.split(latent_vector_1 + ((difference / 10) * i), 2, axis=0)
            sigma = np.exp(0.5 * log_var)
            epsilon = np.random.randn(*mu.shape)
            # Latent is mu because we are trying to get clean reconstructions
            latent = mu #+ sigma * epsilon

            activations_dec, _ = self.decode(latent)
            reconstructed = activations_dec[-1].reshape(28, 28)

            axes[i].imshow(reconstructed, cmap='gray')
            axes[i].axis('off')
            axes[i].set_title(f'Step {i}')

        plt.tight_layout()
        plt.show()

    def show_sample_grid(self, length):
        N = int(length**2)
        grid_size = length

        latent_dim_vae = int(self.decoder_shape[0])

        z_vae = np.random.randn(latent_dim_vae, 1, N)
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
        axes = axes.flatten()
        for i in range(N):
            latent_vector = z_vae[:, 0, i].reshape(-1, 1)
            activations, _ = self.decode(latent_vector)
            img = activations[-1].reshape(28, 28)
            axes[i].imshow(img, cmap='gray')
            axes[i].axis('off')
        plt.tight_layout()
        plt.show()


class ConditionalVariationalAutoEncoder:
    def __init__(self, network_shape: np.ndarray,
               number_of_classes: int,
               loaded_encoder_weights: list = None,
               loaded_encoder_biases: list = None,
               loaded_decoder_weights: list = None,
               loaded_decoder_biases: list = None):
        network_shape = np.array(network_shape, dtype=int)
        self.number_of_classes = number_of_classes
        self.latent_dim = max(1, int(network_shape[-1]) // 2)
        self.encoder_shape = network_shape.copy()
        self.encoder_shape[0] += number_of_classes
        self.encoder = FNN(self.encoder_shape, loaded_encoder_weights, loaded_encoder_biases)
        self.decoder_shape = np.flip(network_shape, 0).astype(int)
        self.decoder_shape[0] = self.latent_dim + number_of_classes
        self.decoder = FNN(self.decoder_shape, loaded_decoder_weights, loaded_decoder_biases)

    def encode(self, input_vector, class_id):
        one_hot = np.eye(1, self.number_of_classes, class_id, dtype=int).ravel()
        enc_in = np.concatenate((input_vector.reshape(-1), one_hot)).reshape(-1, 1)
        return self.encoder.forward_pass(enc_in)

    def decode(self, input_vector, class_id):
        one_hot = np.eye(1, self.number_of_classes, class_id, dtype=int).ravel()
        dec_in = np.concatenate((input_vector.reshape(-1), one_hot)).reshape(-1, 1)
        return self.decoder.forward_pass(dec_in)

    def full_backpropagation(self, input_vector,
                             class_id,
                             learning_rate=0.01,
                             beta=1.0):
        activations_enc, _ = self.encode(input_vector, class_id)
        enc_out = activations_enc[-1]
        mu, log_var = np.split(enc_out, 2, axis=0)
        sigma = np.exp(0.5 * log_var)
        epsilon = np.random.randn(*mu.shape)
        latent_sample = mu + sigma * epsilon

        activations_dec, _ = self.decode(latent_sample, class_id)
        reconstructed_output = activations_dec[-1]

        reconstruction_loss = -np.sum(
            input_vector * np.log(reconstructed_output + 1e-8) +
            (1 - input_vector) * np.log(1 - reconstructed_output + 1e-8)
        )

        kl_loss = -0.5 * np.sum(1 + log_var - mu ** 2 - np.exp(log_var))

        total_loss = reconstruction_loss + beta * kl_loss

        d_recon_da = (reconstructed_output - input_vector)

        one_hot = np.eye(1, self.number_of_classes, class_id, dtype=int).ravel()
        decoder_backprop_input = np.concatenate((latent_sample.reshape(-1), one_hot)).reshape(-1, 1)
        dlatent = self.decoder.backpropagation(
            decoder_backprop_input,
            d_recon_da,
            learning_rate=learning_rate,
            return_input_grad=True,
            output_activation='sigmoid',
            provided_delta=True
        )

        dlatent_latent = dlatent[:self.latent_dim]

        dmu_from_decoder = dlatent_latent
        dlogvar_from_decoder = dlatent_latent * epsilon * 0.5 * sigma

        dKL_dmu = mu
        dKL_dlogvar = 0.5 * (np.exp(log_var) - 1)

        dmu = dmu_from_decoder + beta * dKL_dmu
        dlogvar = dlogvar_from_decoder + beta * dKL_dlogvar

        encoder_output_grad = np.vstack([dmu, dlogvar])

        encoder_backprop_input = np.concatenate((input_vector.reshape(-1), one_hot)).reshape(-1, 1)
        self.encoder.backpropagation(
            encoder_backprop_input, encoder_output_grad, learning_rate=learning_rate,
            return_input_grad=False, output_activation='linear', provided_delta=True
        )

        return reconstruction_loss, kl_loss, total_loss

    def train(self, x_train, y_train, epochs, learning_rate, beta_target, batch_size):
        for epoch in range(epochs):
            total_recon_loss = 0.0
            total_kl_loss = 0.0
            total_loss_sum = 0.0
            beta = beta_target * (epoch + 1) / epochs

            indices = np.random.permutation(len(x_train))
            x_train_shuffled = x_train[indices]
            y_train_shuffled = y_train[indices]

            num_batches = len(x_train) // batch_size

            for i in range(0, len(x_train), batch_size):
                batch_x = x_train_shuffled[i:i + batch_size]
                batch_labels = y_train_shuffled[i:i + batch_size]
                batch_recon_loss = 0.0
                batch_kl_loss = 0.0
                batch_total_loss = 0.0

                for sample, label in zip(batch_x, batch_labels):
                    recon_loss, kl_loss, total_loss = self.full_backpropagation(
                        sample,
                        class_id=label,
                        learning_rate=learning_rate,
                        beta=beta
                    )
                    batch_recon_loss += recon_loss
                    batch_kl_loss += kl_loss
                    batch_total_loss += total_loss

                print(f"{i}")

                total_recon_loss += batch_recon_loss / batch_size
                total_kl_loss += batch_kl_loss / batch_size
                total_loss_sum += batch_total_loss / batch_size

            avg_recon = total_recon_loss / num_batches
            avg_kl = total_kl_loss / num_batches
            avg_total = total_loss_sum / num_batches
            print(
                f"Epoch {epoch + 1}/{epochs} - Avg Recon Loss: {avg_recon:.4f}, Avg KL Loss: {avg_kl:.4f}, Avg Total Loss: {avg_total:.4f}")

        np.save("weights/cond_var_encoder_weights.npy", np.array(self.encoder.weights, dtype=object))
        np.save("weights/cond_var_encoder_biases.npy", np.array(self.encoder.biases, dtype=object))
        np.save("weights/cond_var_decoder_weights.npy", np.array(self.decoder.weights, dtype=object))
        np.save("weights/cond_var_decoder_biases.npy", np.array(self.decoder.biases, dtype=object))

    def show_reconstrutions(self, nr_of_examples, mnist_data, labels):
        fig, axes = plt.subplots(2, nr_of_examples, figsize=(15, 6))
        for i in range(nr_of_examples):
            test_sample = mnist_data[i]
            label = labels[i]
            activations_enc, _ = self.encode(test_sample, label)
            mu, log_var = np.split(activations_enc[-1], 2, axis=0)
            sigma = np.exp(0.5 * log_var)

            axes[0, i].imshow(test_sample.reshape(28, 28), cmap='gray')
            axes[0, i].axis('off')
            axes[0, i].set_title(f'i={i}')

            epsilon = np.random.randn(*mu.shape)
            latent = mu + sigma * epsilon

            activations_dec, _ = self.decode(latent, label)
            reconstructed = activations_dec[-1].reshape(28, 28)

            axes[1, i].imshow(reconstructed, cmap='gray')
            axes[1, i].axis('off')
            axes[1, i].set_title(f'R{i}')

        plt.tight_layout()
        plt.show()

class VQVAE:
    def __init__(self, network_shape: np.ndarray,
                 number_of_codebook_entries: int,
                 loaded_encoder_weights: list = None,
                 loaded_encoder_biases: list = None,
                 loaded_decoder_weights: list = None,
                 loaded_decoder_biases: list = None,
                 loaded_codebook_entries: list = None):
        self.network_shape = network_shape
        self.number_of_codebook_entries = number_of_codebook_entries
        if loaded_encoder_weights:
            self.codebook_entries = loaded_codebook_entries
        else:
            self.codebook_entries = np.random.rand(number_of_codebook_entries, network_shape[-1])
        self.codebook_dim = network_shape[-1]
        self.encoder_shape = network_shape
        self.encoder = FNN(self.encoder_shape, loaded_encoder_weights, loaded_encoder_biases)
        self.decoder_shape = np.flip(network_shape, 0).astype(int)
        self.decoder_shape[0] = self.codebook_dim # not really necessary
        self.decoder = FNN(self.decoder_shape, loaded_decoder_weights, loaded_decoder_biases)

    def encode(self, input):
        return self.encoder.forward_pass(input, final_activation_function="linear")

    def decode(self, input):
        return self.decoder.forward_pass(input, final_activation_function="linear")

    def find_nearest_codebook_entry(self, input):
        input_flat = input.flatten()
        distances = np.sum((self.codebook_entries - input_flat) ** 2, axis=1)
        idx = np.argmin(distances)
        z_q = self.codebook_entries[idx].reshape(-1, 1)
        return z_q, idx

    def full_backpropagation(self, input_vector,
                            learning_rate=0.01,
                            beta=0.25):
        activations_enc, _ = self.encode(input_vector)
        z_e = activations_enc[-1]
        z_q, codebook_chosen_idx = self.find_nearest_codebook_entry(z_e)
        activations_dec, _ = self.decode(z_q)
        reconstructed_output = activations_dec[-1]

        reconstruction_loss = np.sum((reconstructed_output - input_vector) ** 2)
        codebook_loss = np.sum((z_q - z_e) ** 2)
        commit_loss = beta * np.sum((z_e - z_q) ** 2)
        total_loss = reconstruction_loss + codebook_loss + commit_loss

        dlatent = self.decoder.backpropagation(z_q, input_vector, learning_rate,
                                               return_input_grad=True, output_activation='linear', provided_delta=False)

        delta_into_encoder = dlatent + 2*beta*(z_e-z_q)
        z_e_flat = z_e.reshape(-1)
        z_q_flat = z_q.reshape(-1)

        self.codebook_entries[codebook_chosen_idx] += learning_rate * (z_e_flat - z_q_flat)

        self.encoder.backpropagation(
            input_vector, delta_into_encoder, learning_rate=learning_rate,
            return_input_grad=False, output_activation='linear', provided_delta=True
        )

        return reconstruction_loss, codebook_loss, commit_loss, total_loss

    def train(self, x_train, epochs, learning_rate, beta, batch_size):
        for epoch in range(epochs):
            total_recon_loss = 0.0
            total_codebook_loss = 0.0
            total_commit_loss = 0.0
            total_loss_sum = 0.0

            indices = np.random.permutation(len(x_train))
            x_train_shuffled = x_train[indices]

            num_batches = len(x_train) // batch_size

            for i in range(0, len(x_train), batch_size):
                batch = x_train_shuffled[i:i + batch_size]
                batch_recon_loss = 0.0
                batch_codebook_loss = 0.0
                batch_commit_loss = 0.0
                batch_total_loss = 0.0

                for sample in batch:
                    sample_vector = sample.reshape(-1, 1)
                    reconstruction_loss, codebook_loss, commit_loss, total_loss = self.full_backpropagation(
                        sample_vector,
                        learning_rate=learning_rate,
                        beta=beta
                    )
                    batch_recon_loss += reconstruction_loss
                    batch_codebook_loss += codebook_loss
                    batch_commit_loss += commit_loss
                    batch_total_loss += total_loss

                print(f"{i}")

                total_recon_loss += batch_recon_loss / batch_size
                total_codebook_loss += batch_codebook_loss / batch_size
                total_commit_loss += batch_commit_loss / batch_size
                total_loss_sum += batch_total_loss / batch_size

            avg_recon = total_recon_loss / num_batches
            avg_codebook = total_codebook_loss / num_batches
            avg_commit = total_commit_loss / num_batches
            avg_total = total_loss_sum / num_batches
            print(
                f"Epoch {epoch + 1}/{epochs} - Avg Recon Loss: {avg_recon:.4f}, Avg Codebook Loss: {avg_codebook:.4f}, Avg Commit Loss: {avg_commit:.4f}, Avg Total Loss: {avg_total:.4f}")

        np.save("weights/vqvae_encoder_weights.npy", np.array(self.encoder.weights, dtype=object))
        np.save("weights/vqvae_encoder_biases.npy", np.array(self.encoder.biases, dtype=object))
        np.save("weights/vqvae_decoder_weights.npy", np.array(self.decoder.weights, dtype=object))
        np.save("weights/vqvae_decoder_biases.npy", np.array(self.decoder.biases, dtype=object))
        np.save("weights/vqvae_codebook.npy", np.array(self.codebook_entries, dtype=object))

    def show_reconstrutions(self, nr_of_examples, mnist_data):
        fig, axes = plt.subplots(2, nr_of_examples, figsize=(15, 6))
        for i in range(nr_of_examples):
            test_sample = mnist_data[i]
            activations_enc, _ = self.encode(test_sample)

            axes[0, i].imshow(test_sample.reshape(28, 28), cmap='gray')
            axes[0, i].axis('off')
            axes[0, i].set_title(f'i={i}')

            z_q, codebook_chosen_idx = self.find_nearest_codebook_entry(activations_enc[-1])
            activations_dec, _ = self.decode(z_q)
            reconstructed = activations_dec[-1].reshape(28, 28)

            axes[1, i].imshow(reconstructed, cmap='gray')
            axes[1, i].axis('off')
            axes[1, i].set_title(f'R{i}')

        plt.tight_layout()
        plt.show()

    def calculate_anomaly_score(self, input_vector):
        sample_col = input_vector.reshape(-1, 1)

        activations_enc, _ = self.encode(sample_col)
        z_e = activations_enc[-1]

        z_q, codebook_idx = self.find_nearest_codebook_entry(z_e)

        activations_dec, _ = self.decode(z_q)
        reconstructed = activations_dec[-1]

        mse = np.mean((reconstructed - sample_col) ** 2)

        return mse, reconstructed, codebook_idx