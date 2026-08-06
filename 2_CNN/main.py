import numpy as np
import pandas as pd


class ConvolutionalNeuralNetwork:
    def __init__(self):
        self.w1 = np.random.randn(3, 3) * np.sqrt(2 / 9)
        self.w2 = np.random.randn(3, 3) * np.sqrt(2 / 9)
        self.w3 = np.random.randn(4, 2, 3, 3) * np.sqrt(2 / (3 * 3 * 2))
        self.w4 = np.random.randn(1024, 16) * np.sqrt(2 / 1024)
        self.b1 = np.zeros(16)
        self.w5 = np.random.randn(16) * np.sqrt(2 / 16)
        self.b2 = np.zeros(1)

        # Optimizer params
        self.mw1 = np.zeros_like(self.w1)
        self.vw1 = np.zeros_like(self.w1)

        self.mw2 = np.zeros_like(self.w2)
        self.vw2 = np.zeros_like(self.w2)

        self.mw3 = np.zeros_like(self.w3)
        self.vw3 = np.zeros_like(self.w3)

        self.mw4 = np.zeros_like(self.w4)
        self.vw4 = np.zeros_like(self.w4)

        self.mb1 = np.zeros_like(self.b1)
        self.vb1 = np.zeros_like(self.b1)

        self.mw5 = np.zeros_like(self.w5)
        self.vw5 = np.zeros_like(self.w5)

        self.mb2 = np.zeros_like(self.b2)
        self.vb2 = np.zeros_like(self.b2)

        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.e = 10**-8
        self.lr = 0.0001
        self.t = 0

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return (z > 0).astype(float)

    def conv2d(
        self,
        input_map: np.ndarray,
        kernel: np.ndarray,
        padding: int = 0,
        stride: int = 1,
    ):
        h_in, w_in = input_map.shape
        f = kernel.shape[0]

        h_out = (h_in - f + 2 * padding) // stride + 1
        w_out = (w_in - f + 2 * padding) // stride + 1

        regions = []
        for i in range(h_out):
            for j in range(w_out):
                start_i = i * stride
                start_j = j * stride
                region = input_map[start_i : start_i + f, start_j : start_j + f]
                regions.append(region.flatten())

        conv_result = np.array(regions) @ kernel.flatten()
        return conv_result.reshape(h_out, w_out)

    def conv2d_multi_channel(self, feature_map, kernels):
        # feature_map: (C_in, H, W)
        # kernels: (C_out, C_in, F, F)

        C_out, C_in, F, _ = kernels.shape
        _, H, W = feature_map.shape

        H_out = H - F + 1
        W_out = W - F + 1

        out = np.zeros((C_out, H_out, W_out))

        for c_out in range(C_out):
            for c_in in range(C_in):
                out[c_out] += self.conv2d(feature_map[c_in], kernels[c_out, c_in])

        return out

    def binary_cross_entropy(self, y_true, y_pred):
        # Epsilon untuk menghindari log(0)
        epsilon = 1e-7
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        # Rumus: -(y * log(p) + (1-y) * log(1-p))
        loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return np.mean(loss)

    def dw2d_derivative(self, input_map: np.ndarray, delta: np.ndarray):
        h_in, w_in = input_map.shape
        f, _ = delta.shape

        h_out = h_in - f + 1
        w_out = w_in - f + 1

        regions = []
        for i in range(h_out):
            for j in range(w_out):
                region = input_map[i : i + f, j : j + f]
                regions.append(region.flatten())

        conv_result = np.array(regions) @ delta.flatten()
        return conv_result.reshape(h_out, w_out)

    def dw_multi_channel_derivative(
        self, input_map: np.ndarray, delta: np.ndarray, w: np.ndarray
    ):
        p, _, _ = input_map.shape
        q, _, _ = delta.shape
        out = np.zeros_like(w)

        for i in range(q):
            for j in range(p):
                result = self.dw2d_derivative(input_map[j], delta[i])
                out[i, j] = result

        return out

    def dx2d_derivative(self, w: np.ndarray, delta: np.ndarray, x: np.ndarray):
        h_in, w_in = x.shape
        w_flip = np.flip(w)

        p, q = delta.shape
        f, _ = w.shape

        out = np.zeros_like(x)

        for i in range(p):
            for j in range(q):
                # region = out[i : i + f, j : j + f]
                out[i : i + f, j : j + f] += w_flip * delta[i, j]

        return out

    def dx_conv_multi_channel(self, w: np.ndarray, delta: np.ndarray, x: np.ndarray):
        """
        w     : (C_out, C_in, f, f)
        delta : (C_out, H_out, W_out)
        x     : (C_in, H_in, W_in)
        """
        C_out, C_in, f, _ = w.shape
        dx = np.zeros_like(x)

        for j in range(C_in):  # input channel
            for i in range(C_out):  # output channel
                dx[j] += self.dx2d_derivative(w[i, j], delta[i], x[j])

        return dx

    def adam_optimizer(
        self, m, v, theta_prev, dtheta, t
    ):  # Tambah parameter theta_prev
        # 1. Hitung Moment (M dan V yang baru)
        m_new = (self.beta_1 * m) + ((1 - self.beta_1) * dtheta)
        v_new = (self.beta_2 * v) + ((1 - self.beta_2) * np.square(dtheta))

        # 2. Koreksi Bias
        m_hat = m_new / (1 - (self.beta_1**t))
        v_hat = v_new / (1 - (self.beta_2**t))

        # 3. Hitung Pembaruan (Update Step)
        update = (self.lr / (np.sqrt(v_hat) + self.e)) * m_hat

        # 4. Terapkan Pembaruan
        theta_new = theta_prev - update

        # Kembalikan M dan V yang BARU, serta Theta yang BARU
        return m_new, v_new, theta_new

    def train(self, x_train: np.ndarray, y_train: np.ndarray, epochs: int):
        for epoch in range(epochs):
            total_loss = 0
            self.t += 1
            for idx, x in enumerate(x_train):
                x = x.reshape(20, 20)
                y = y_train[idx]
                # Forward pass conv
                feature_map_1 = self.conv2d(x, self.w1)
                feature_map_2 = self.conv2d(x, self.w2)
                z1 = np.stack((feature_map_1, feature_map_2))
                a1 = self.relu(z1)

                z2 = self.conv2d_multi_channel(a1, self.w3)
                a2 = self.relu(z2)

                flattened = a2.flatten()

                # Forward pass MLP
                z1_mlp = flattened @ self.w4 + self.b1
                a1_mlp = self.relu(z1_mlp)

                z2_mlp = a1_mlp @ self.w5 + self.b2
                y_pred = self.sigmoid(z2_mlp)

                # Loss
                loss = self.binary_cross_entropy(y, y_pred)
                total_loss += loss

                # Backward mlp
                dy_pred = y_pred - y
                dw5 = dy_pred * a1_mlp
                db2 = dy_pred * 1

                da1 = dy_pred * self.w5
                dz1_mlp = da1 * self.relu_derivative(z1_mlp)

                dw4 = np.outer(flattened, dz1_mlp)
                db1 = dz1_mlp * 1

                dx = dz1_mlp @ self.w4.T

                grad_from_mlp = dx.reshape(4, 16, 16)

                # backward conv
                da2 = grad_from_mlp * self.relu_derivative(z2)
                dw3 = self.dw_multi_channel_derivative(a1, da2, self.w3)
                da1 = self.dx_conv_multi_channel(self.w3, da2, a1)
                dz1 = da1 * self.relu_derivative(z1)
                dw2 = self.dw2d_derivative(x, dz1[1])
                dw1 = self.dw2d_derivative(x, dz1[0])

                self.mw5, self.vw5, self.w5 = self.adam_optimizer(
                    self.mw5, self.vw5, self.w5, dw5, self.t
                )

                self.mb2, self.vb2, self.b2 = self.adam_optimizer(
                    self.mb2, self.vb2, self.b2, db2, self.t
                )

                self.mw4, self.vw4, self.w4 = self.adam_optimizer(
                    self.mw4, self.vw4, self.w4, dw4, self.t
                )
                self.mb1, self.vb1, self.b1 = self.adam_optimizer(
                    self.mb1, self.vb1, self.b1, db1, self.t
                )

                # Update Weight Conv
                self.mw3, self.vw3, self.w3 = self.adam_optimizer(
                    self.mw3, self.vw3, self.w3, dw3, self.t
                )
                self.mw2, self.vw2, self.w2 = self.adam_optimizer(
                    self.mw2, self.vw2, self.w2, dw2, self.t
                )
                self.mw1, self.vw1, self.w1 = self.adam_optimizer(
                    self.mw1, self.vw1, self.w1, dw1, self.t
                )

            print(f"Epoch {epoch + 1} - Average Loss: {total_loss / len(x_train)}")

    def test(self, x_test: np.ndarray):
        result = []
        for idx, x in enumerate(x_test):
            x = x.reshape(20, 20)

            # Forward pass conv
            feature_map_1 = self.conv2d(x, self.w1)
            feature_map_2 = self.conv2d(x, self.w2)
            z1 = np.stack((feature_map_1, feature_map_2))
            a1 = self.relu(z1)
            z2 = self.conv2d_multi_channel(a1, self.w3)
            a2 = self.relu(z2)
            flattened = a2.flatten()
            # Forward pass MLP
            z1_mlp = flattened @ self.w4 + self.b1
            a1_mlp = self.relu(z1_mlp)
            z2_mlp = a1_mlp @ self.w5 + self.b2
            y_pred = self.sigmoid(z2_mlp)

            result.append(y_pred)

        return result


df = pd.read_csv("./dataset/mnist.csv")

x_train = df.iloc[0:30, 0:-1].values
y_train = df.iloc[0:30, -1].values

x_test = df.iloc[30:50, 0:-1].values
y_test = df.iloc[30:50, -1].values

model = ConvolutionalNeuralNetwork()
model.train(x_train, y_train, 50)

result = model.test(x_test)
print(result)

print(f"ACTUAL: {y_test}")
