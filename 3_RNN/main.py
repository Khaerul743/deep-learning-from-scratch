import sys
import time
from typing import Literal

import numpy as np


class RecurrentNeuralNetwork:
    """
    Vanilla Recurrent Neural Network (RNN) built from scratch using NumPy.
    Supports sequence processing, Backpropagation Through Time (BPTT),
    and parameter updates via Adam or SGD optimizers.
    """

    def __init__(self, vocab_size: int, hidden_dim: int = 8):
        self.input_dim = vocab_size
        self.hidden_dim = hidden_dim
        self.output_dim = vocab_size

        # Weight and Bias Initialization
        self.wxh = self._xavier_init(self.input_dim, self.hidden_dim)
        self.whh = self._orthogonal_init(self.hidden_dim)
        self.why = self._xavier_init(self.hidden_dim, self.output_dim)

        self.bh = np.zeros(self.hidden_dim)
        self.by = np.zeros(self.output_dim)
        self.h_init = np.zeros(self.hidden_dim)

        # Gradients
        self.dwxh_total = np.zeros_like(self.wxh)
        self.dwhh_total = np.zeros_like(self.whh)
        self.dbh_total = np.zeros_like(self.bh)
        self.dwhy_total = np.zeros_like(self.why)
        self.dby_total = np.zeros_like(self.by)

        # Optimizer State (Adam)
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.e = 1e-8
        self.lr = 0.001

        self.mwxh = np.zeros_like(self.wxh)
        self.vwxh = np.zeros_like(self.wxh)
        self.mwhh = np.zeros_like(self.whh)
        self.vwhh = np.zeros_like(self.whh)
        self.mbh = np.zeros_like(self.bh)
        self.vbh = np.zeros_like(self.bh)
        self.mwhy = np.zeros_like(self.why)
        self.vwhy = np.zeros_like(self.why)
        self.mby = np.zeros_like(self.by)
        self.vby = np.zeros_like(self.by)

    def _xavier_init(self, fan_in: int, fan_out: int) -> np.ndarray:
        return np.random.randn(fan_out, fan_in) * np.sqrt(2.0 / (fan_in + fan_out))

    def _orthogonal_init(self, dim: int) -> np.ndarray:
        w = np.random.randn(dim, dim)
        q, _ = np.linalg.qr(w)
        return q

    def softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        x = np.asarray(x)
        x_max = np.max(x, axis=axis, keepdims=True)
        e = np.exp(x - x_max)
        return e / np.sum(e, axis=axis, keepdims=True)

    def dtanh(self, x: np.ndarray) -> np.ndarray:
        t = np.tanh(x)
        return 1.0 - t * t

    def categorical_cross_entropy(
        self, y_pred: np.ndarray, y_true: np.ndarray, eps: float = 1e-12
    ) -> float:
        y_pred = np.clip(y_pred, eps, 1.0 - eps)
        return float(-np.sum(y_true * np.log(y_pred)))

    def forward_step(self, x_t: np.ndarray, h_prev: np.ndarray):
        """
        Forward computation for a single time step t.
        """
        z_t = (self.wxh @ x_t) + (self.whh @ h_prev) + self.bh
        h_t = np.tanh(z_t)
        y_t = self.softmax(self.why @ h_t + self.by)
        return z_t, h_t, y_t

    def forward(self, sequence_embeds: np.ndarray):
        """
        Forward pass for a sequence of word embeddings.
        Returns dictionaries containing pre-activations z, hidden states h, and predictions y.
        """
        z = {}
        h = {0: self.h_init.copy()}
        y = {}

        for idx, word_embed in enumerate(sequence_embeds):
            if idx + 1 == len(sequence_embeds):
                continue
            z_t, h_t, y_t = self.forward_step(word_embed, h[idx])
            z[idx + 1] = z_t
            h[idx + 1] = h_t
            y[idx + 1] = y_t

        return z, h, y

    def backward(
        self,
        z: dict,
        h: dict,
        y: dict,
        sequence_embeds: np.ndarray,
    ):
        """
        Backpropagation Through Time (BPTT) over the sequence.
        Accumulates gradients across steps t = T down to 1.
        """
        self.dwxh_total = np.zeros_like(self.wxh)
        self.dwhh_total = np.zeros_like(self.whh)
        self.dbh_total = np.zeros_like(self.bh)
        self.dwhy_total = np.zeros_like(self.why)
        self.dby_total = np.zeros_like(self.by)

        dh_delta = np.zeros_like(self.h_init)

        for i in range(len(y), 0, -1):
            target_t = sequence_embeds[i]
            x_t_prev = sequence_embeds[i - 1]

            dy = y[i] - target_t
            dwhy = np.outer(dy, h[i])
            dby = dy

            dh = dy @ self.why + dh_delta
            dz = dh * self.dtanh(z[i])

            dwxh = np.outer(dz, x_t_prev)
            dwhh = np.outer(dz, h[i - 1])
            dbh = dz

            dh_delta = dz @ self.whh

            self.dwhy_total += dwhy
            self.dby_total += dby
            self.dwhh_total += dwhh
            self.dbh_total += dbh
            self.dwxh_total += dwxh

    def _adam_step(self, m, v, theta, dtheta, t):
        m_new = (self.beta_1 * m) + ((1 - self.beta_1) * dtheta)
        v_new = (self.beta_2 * v) + ((1 - self.beta_2) * np.square(dtheta))

        m_hat = m_new / (1 - (self.beta_1**t))
        v_hat = v_new / (1 - (self.beta_2**t))

        update = (self.lr / (np.sqrt(v_hat) + self.e)) * m_hat
        theta_new = theta - update

        return m_new, v_new, theta_new

    def update_params_adam(self, t: int):
        self.mwxh, self.vwxh, self.wxh = self._adam_step(
            self.mwxh, self.vwxh, self.wxh, self.dwxh_total, t
        )
        self.mwhh, self.vwhh, self.whh = self._adam_step(
            self.mwhh, self.vwhh, self.whh, self.dwhh_total, t
        )
        self.mbh, self.vbh, self.bh = self._adam_step(
            self.mbh, self.vbh, self.bh, self.dbh_total, t
        )
        self.mwhy, self.vwhy, self.why = self._adam_step(
            self.mwhy, self.vwhy, self.why, self.dwhy_total, t
        )
        self.mby, self.vby, self.by = self._adam_step(
            self.mby, self.vby, self.by, self.dby_total, t
        )

    def update_params_sgd(self):
        self.wxh -= self.lr * self.dwxh_total
        self.whh -= self.lr * self.dwhh_total
        self.why -= self.lr * self.dwhy_total
        self.bh -= self.lr * self.dbh_total
        self.by -= self.lr * self.dby_total

    def compute_sequence_loss(self, y: dict, sequence_embeds: np.ndarray) -> float:
        total_loss = []
        for k, v in y.items():
            total_loss.append(self.categorical_cross_entropy(v, sequence_embeds[k]))
        return float(np.mean(total_loss))

    def _train_visualization(self, loss: float, epochs: int, i: int):
        if not hasattr(self, "_train_start"):
            self._train_start = time.time()
        elapsed = time.time() - self._train_start
        avg_per_epoch = elapsed / (i + 1)
        remaining = avg_per_epoch * (epochs - (i + 1))
        eta_min = int(remaining // 60)
        eta_sec = int(remaining % 60)

        bar_len = 40
        filled = int((i + 1) / epochs * bar_len)
        bar = "#" * filled + "-" * (bar_len - filled)

        sys.stdout.write(
            f"\rEpoch {i + 1}/{epochs} [{bar}] loss={loss:.4f} ETA={eta_min:02d}:{eta_sec:02d}"
        )
        sys.stdout.flush()
        if i == epochs - 1:
            print()

    def train(
        self,
        train_set_embeds: list,
        epochs: int = 1000,
        lr: float = 0.001,
        optimizer: Literal["adam", "sgd"] = "adam",
    ):
        self.lr = lr
        t = 0

        for epoch in range(epochs):
            epoch_losses = []
            for sentence_embeds in train_set_embeds:
                t += 1
                z, h, y = self.forward(sentence_embeds)
                loss = self.compute_sequence_loss(y, sentence_embeds)
                epoch_losses.append(loss)

                self.backward(z, h, y, sentence_embeds)

                if optimizer == "adam":
                    self.update_params_adam(t)
                else:
                    self.update_params_sgd()

            mean_epoch_loss = float(np.mean(epoch_losses))
            self._train_visualization(mean_epoch_loss, epochs, epoch)
            time.sleep(0.001)

    def generate_text(
        self,
        start_word: str,
        words: list,
        word_embeds: np.ndarray,
        length: int = 5,
    ) -> str:
        if start_word not in words:
            return f"Kata '{start_word}' tidak ada dalam kosakata."

        current_word_idx = words.index(start_word)
        sentence = [start_word]
        h_step = self.h_init.copy()

        for _ in range(length):
            x = word_embeds[current_word_idx]
            z_t = (self.wxh @ x) + (self.whh @ h_step) + self.bh
            h_step = np.tanh(z_t)

            y_prob = self.softmax(self.why @ h_step + self.by)
            next_idx = int(np.argmax(y_prob))

            sentence.append(words[next_idx])
            current_word_idx = next_idx

        return " ".join(sentence)


if __name__ == "__main__":
    # Vocabulary & Dataset Setup
    words = ["saya", "makan", "minum", "nasi", "kopi", "tadi", "siang", "malam"]
    vocab_size = len(words)
    word_embeds = np.eye(vocab_size)

    train_set = [
        ["saya", "makan", "nasi", "tadi", "siang"],
        ["saya", "minum", "kopi", "tadi", "malam"],
        ["nasi", "tadi", "malam"],
        ["kopi", "tadi", "siang"],
    ]

    def convert_to_embed(sentence: list) -> np.ndarray:
        return np.array([word_embeds[words.index(w)] for w in sentence])

    train_set_embeds = [convert_to_embed(s) for s in train_set]

    # Initialize and train RNN
    print("--- Training Recurrent Neural Network ---")
    rnn = RecurrentNeuralNetwork(vocab_size=vocab_size, hidden_dim=8)
    rnn.train(train_set_embeds, epochs=1000, lr=0.001, optimizer="adam")

    # Text Generation Inference
    print("\n--- Hasil Prediksi Teks ---")
    test_words = ["saya", "nasi", "kopi"]
    for word in test_words:
        generated = rnn.generate_text(
            start_word=word, words=words, word_embeds=word_embeds, length=4
        )
        print(f"Start: '{word}' -> Generated: '{generated}'")
