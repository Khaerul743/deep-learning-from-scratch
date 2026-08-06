import sys
import time
from typing import Literal

import numpy as np


class LongShortTermMemory:
    """
    Long Short-Term Memory (LSTM) built from scratch using NumPy.
    Implements Forget, Input, Output, and Cell Candidate gates,
    Backpropagation Through Time (BPTT), FC projection layers,
    and parameter updates via Adam or SGD optimizers.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 4,
        h_size: int = 4,
        c_size: int = 4,
        fc_hidden_size: int = 10,
    ):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.h_size = h_size
        self.c_size = c_size
        self.fc_hidden_size = fc_hidden_size

        # Embedding Matrix Initialization
        self.E = np.random.randn(vocab_size, embed_dim) * np.sqrt(1.0 / vocab_size)

        # Initial States
        self.h_init = np.zeros(h_size)
        self.c_init = np.zeros(c_size)

        # Gate Weights Initialization (concat size = h_size + embed_dim)
        concat_size = h_size + embed_dim
        scale = np.sqrt(1.0 / concat_size)

        self.wf = np.random.randn(c_size, concat_size) * scale
        self.wi = np.random.randn(c_size, concat_size) * scale
        self.wo = np.random.randn(c_size, concat_size) * scale
        self.wc = np.random.randn(c_size, concat_size) * scale

        self.bf = np.zeros(c_size)
        self.bi = np.zeros(c_size)
        self.bo = np.zeros(c_size)
        self.bc = np.zeros(c_size)

        # Fully Connected (FC) Layers Initialization
        self.w1 = np.random.randn(fc_hidden_size, h_size) * np.sqrt(2.0 / h_size)
        self.b1 = np.zeros(fc_hidden_size)

        self.w2 = np.random.randn(h_size, fc_hidden_size) * np.sqrt(
            2.0 / fc_hidden_size
        )
        self.b2 = np.zeros(h_size)

        self.why = np.random.randn(vocab_size, h_size) * np.sqrt(1.0 / h_size)
        self.by = np.zeros(vocab_size)

        # Optimizer State (Adam)
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.e = 1e-8
        self.lr = 0.001

        self.mwf = np.zeros_like(self.wf)
        self.vwf = np.zeros_like(self.wf)
        self.mwi = np.zeros_like(self.wi)
        self.vwi = np.zeros_like(self.wi)
        self.mwo = np.zeros_like(self.wo)
        self.vwo = np.zeros_like(self.wo)
        self.mwc = np.zeros_like(self.wc)
        self.vwc = np.zeros_like(self.wc)

        self.mbf = np.zeros_like(self.bf)
        self.vbf = np.zeros_like(self.bf)
        self.mbi = np.zeros_like(self.bi)
        self.vbi = np.zeros_like(self.bi)
        self.mbo = np.zeros_like(self.bo)
        self.vbo = np.zeros_like(self.bo)
        self.mbc = np.zeros_like(self.bc)
        self.vbc = np.zeros_like(self.bc)

        self.mw1 = np.zeros_like(self.w1)
        self.vw1 = np.zeros_like(self.w1)
        self.mb1 = np.zeros_like(self.b1)
        self.vb1 = np.zeros_like(self.b1)

        self.mw2 = np.zeros_like(self.w2)
        self.vw2 = np.zeros_like(self.w2)
        self.mb2 = np.zeros_like(self.b2)
        self.vb2 = np.zeros_like(self.b2)

        self.mwhy = np.zeros_like(self.why)
        self.vwhy = np.zeros_like(self.why)
        self.mby = np.zeros_like(self.by)
        self.vby = np.zeros_like(self.by)

    # --- Activation Functions & Derivatives ---
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def sigmoid_deriv(self, x: np.ndarray) -> np.ndarray:
        s = self.sigmoid(x)
        return s * (1.0 - s)

    def tanh_deriv(self, x: np.ndarray) -> np.ndarray:
        t = np.tanh(x)
        return 1.0 - t * t

    def relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def relu_deriv(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)

    def softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        x = np.asarray(x)
        x_max = np.max(x, axis=axis, keepdims=True)
        e = np.exp(x - x_max)
        return e / np.sum(e, axis=axis, keepdims=True)

    def categorical_cross_entropy(
        self, y_pred: np.ndarray, y_true: np.ndarray, eps: float = 1e-12
    ) -> float:
        y_pred = np.clip(y_pred, eps, 1.0 - eps)
        return float(-np.sum(y_true * np.log(y_pred)))

    # --- Forward Pass ---
    def embedding_forward(self, word_idx: int) -> np.ndarray:
        return self.E[word_idx, :]

    def forward_step(self, h_past: np.ndarray, x: np.ndarray, c_past: np.ndarray):
        x_concat = np.concatenate((h_past, x), axis=0)

        # Gates
        ft = self.wf @ x_concat + self.bf
        aft = self.sigmoid(ft)

        it = self.wi @ x_concat + self.bi
        ait = self.sigmoid(it)

        ot = self.wo @ x_concat + self.bo
        aot = self.sigmoid(ot)

        cell_state = self.wc @ x_concat + self.bc
        acell_state = np.tanh(cell_state)

        # Cell State & Hidden State Update
        ct = (aft * c_past) + (ait * acell_state)
        act = np.tanh(ct)
        ht = aot * act

        # Fully Connected Layers
        z1 = self.w1 @ ht + self.b1
        az1 = self.relu(z1)

        z2 = self.w2 @ az1 + self.b2
        az2 = self.relu(z2)

        # Output Prediction
        yt = self.softmax(self.why @ az2 + self.by)

        out_params = {"yt": yt, "ht": ht, "ct": ct}
        support_params = {
            "ft": ft,
            "it": it,
            "ot": ot,
            "cell_state": cell_state,
            "z1": z1,
            "z2": z2,
        }
        activation_params = {
            "aft": aft,
            "ait": ait,
            "aot": aot,
            "acell_state": acell_state,
            "act": act,
            "az1": az1,
            "az2": az2,
        }
        input_param = {"x": x_concat}

        return out_params, support_params, activation_params, input_param

    def forward_predict(self, x: np.ndarray) -> np.ndarray:
        x_concat = np.concatenate((self.h_init, x), axis=0)

        ft = self.wf @ x_concat + self.bf
        aft = self.sigmoid(ft)

        it = self.wi @ x_concat + self.bi
        ait = self.sigmoid(it)

        ot = self.wo @ x_concat + self.bo
        aot = self.sigmoid(ot)

        cell_state = self.wc @ x_concat + self.bc
        acell_state = np.tanh(cell_state)

        ct = (aft * self.c_init) + (ait * acell_state)
        act = np.tanh(ct)
        ht = aot * act

        z1 = self.w1 @ ht + self.b1
        az1 = self.relu(z1)

        z2 = self.w2 @ az1 + self.b2
        az2 = self.relu(z2)

        yt = self.softmax(self.why @ az2 + self.by)
        return yt

    def forward(self, sequence_indices: list):
        out_params_t = {"ht": {0: self.h_init.copy()}, "ct": {0: self.c_init.copy()}, "yt": {}}
        support_params_t = {
            "ft": {},
            "it": {},
            "ot": {},
            "cell_state": {},
            "z1": {},
            "z2": {},
        }
        activation_params_t = {
            "aft": {},
            "ait": {},
            "aot": {},
            "acell_state": {},
            "act": {},
            "az1": {},
            "az2": {},
        }
        xt = {}

        for idx, word_idx in enumerate(sequence_indices):
            if idx + 1 == len(sequence_indices):
                continue
            x_embed = self.embedding_forward(word_idx)
            out_p, supp_p, act_p, inp_p = self.forward_step(
                out_params_t["ht"][idx], x_embed, out_params_t["ct"][idx]
            )

            out_params_t["yt"][idx + 1] = out_p["yt"]
            out_params_t["ht"][idx + 1] = out_p["ht"]
            out_params_t["ct"][idx + 1] = out_p["ct"]

            for k in supp_p:
                support_params_t[k][idx + 1] = supp_p[k]
            for k in act_p:
                activation_params_t[k][idx + 1] = act_p[k]

            xt[idx + 1] = inp_p["x"]

        return out_params_t, support_params_t, activation_params_t, xt

    # --- Backward Pass (BPTT) ---
    def backward(
        self,
        out_params_t: dict,
        support_params_t: dict,
        activation_params_t: dict,
        targets: list,
        xt: dict,
    ):
        total_dwf = np.zeros_like(self.wf)
        total_dwi = np.zeros_like(self.wi)
        total_dwo = np.zeros_like(self.wo)
        total_dwc = np.zeros_like(self.wc)

        total_dbf = np.zeros_like(self.bf)
        total_dbi = np.zeros_like(self.bi)
        total_dbo = np.zeros_like(self.bo)
        total_dbc = np.zeros_like(self.bc)

        total_dw1 = np.zeros_like(self.w1)
        total_dw2 = np.zeros_like(self.w2)
        total_db1 = np.zeros_like(self.b1)
        total_db2 = np.zeros_like(self.b2)

        total_dwhy = np.zeros_like(self.why)
        total_dby = np.zeros_like(self.by)

        total_dht = np.zeros_like(self.h_init)
        total_dct = np.zeros_like(self.c_init)

        seq_len = len(out_params_t["yt"])

        for i in range(seq_len, 0, -1):
            target_onehot = np.eye(self.vocab_size)[targets[i]]
            dy = out_params_t["yt"][i] - target_onehot
            dwhy = np.outer(dy, activation_params_t["az2"][i])
            dby = dy

            daz2 = dy @ self.why
            dz2 = daz2 * self.relu_deriv(support_params_t["z2"][i])

            dw2 = np.outer(dz2, activation_params_t["az1"][i])
            db2 = dz2

            daz1 = dz2 @ self.w2
            dz1 = daz1 * self.relu_deriv(support_params_t["z1"][i])

            dw1 = np.outer(dz1, out_params_t["ht"][i])
            db1 = dz1

            dht = dz1 @ self.w1 + total_dht

            # Output Gate Derivative
            daot = dht * activation_params_t["act"][i]
            dot = daot * self.sigmoid_deriv(support_params_t["ot"][i])
            dwo = np.outer(dot, xt[i])
            dbo = dot
            dxo = dot @ self.wo
            dh_prev_o = dxo[: self.h_size]

            # Cell State Derivative
            dact = dht * activation_params_t["aot"][i]
            dct = dact * self.tanh_deriv(out_params_t["ct"][i]) + total_dct

            # Forget Gate Derivative
            daft = dct * out_params_t["ct"][i - 1]
            dft = daft * self.sigmoid_deriv(support_params_t["ft"][i])
            dwf = np.outer(dft, xt[i])
            dbf = dft
            dxf = dft @ self.wf
            dh_prev_f = dxf[: self.h_size]

            dc_past = dct * activation_params_t["aft"][i]

            # Input Gate Derivative
            dait = dct * activation_params_t["acell_state"][i]
            dit = dait * self.sigmoid_deriv(support_params_t["it"][i])
            dwi = np.outer(dit, xt[i])
            dbi = dit
            dxi = dit @ self.wi
            dh_prev_i = dxi[: self.h_size]

            # Candidate Cell State Derivative
            dacell_state = dct * activation_params_t["ait"][i]
            dcell_state = dacell_state * self.tanh_deriv(
                support_params_t["cell_state"][i]
            )
            dwc = np.outer(dcell_state, xt[i])
            dbc = dcell_state
            dxcell_state = dcell_state @ self.wc
            dh_prev_cell_state = dxcell_state[: self.h_size]

            # Accumulate Gradients
            total_dwf += dwf
            total_dwi += dwi
            total_dwo += dwo
            total_dwc += dwc
            total_dbf += dbf
            total_dbi += dbi
            total_dbo += dbo
            total_dbc += dbc

            total_dw1 += dw1
            total_dw2 += dw2
            total_db1 += db1
            total_db2 += db2
            total_dwhy += dwhy
            total_dby += dby

            dh = dh_prev_o + dh_prev_f + dh_prev_i + dh_prev_cell_state
            total_dht = dh
            total_dct = dc_past

        self.grads = {
            "dwf": total_dwf,
            "dwi": total_dwi,
            "dwo": total_dwo,
            "dwc": total_dwc,
            "dbf": total_dbf,
            "dbi": total_dbi,
            "dbo": total_dbo,
            "dbc": total_dbc,
            "dw1": total_dw1,
            "dw2": total_dw2,
            "db1": total_db1,
            "db2": total_db2,
            "dwhy": total_dwhy,
            "dby": total_dby,
        }

    # --- Optimizer & Parameter Updates ---
    def _adam_step(self, m, v, theta, dtheta, t):
        m_new = (self.beta_1 * m) + ((1 - self.beta_1) * dtheta)
        v_new = (self.beta_2 * v) + ((1 - self.beta_2) * np.square(dtheta))

        m_hat = m_new / (1 - (self.beta_1**t))
        v_hat = v_new / (1 - (self.beta_2**t))

        update = (self.lr / (np.sqrt(v_hat) + self.e)) * m_hat
        theta_new = theta - update

        return m_new, v_new, theta_new

    def update_params_adam(self, t: int):
        self.mwf, self.vwf, self.wf = self._adam_step(
            self.mwf, self.vwf, self.wf, self.grads["dwf"], t
        )
        self.mwi, self.vwi, self.wi = self._adam_step(
            self.mwi, self.vwi, self.wi, self.grads["dwi"], t
        )
        self.mwo, self.vwo, self.wo = self._adam_step(
            self.mwo, self.vwo, self.wo, self.grads["dwo"], t
        )
        self.mwc, self.vwc, self.wc = self._adam_step(
            self.mwc, self.vwc, self.wc, self.grads["dwc"], t
        )

        self.mbf, self.vbf, self.bf = self._adam_step(
            self.mbf, self.vbf, self.bf, self.grads["dbf"], t
        )
        self.mbi, self.vbi, self.bi = self._adam_step(
            self.mbi, self.vbi, self.bi, self.grads["dbi"], t
        )
        self.mbo, self.vbo, self.bo = self._adam_step(
            self.mbo, self.vbo, self.bo, self.grads["dbo"], t
        )
        self.mbc, self.vbc, self.bc = self._adam_step(
            self.mbc, self.vbc, self.bc, self.grads["dbc"], t
        )

        self.mw1, self.vw1, self.w1 = self._adam_step(
            self.mw1, self.vw1, self.w1, self.grads["dw1"], t
        )
        self.mw2, self.vw2, self.w2 = self._adam_step(
            self.mw2, self.vw2, self.w2, self.grads["dw2"], t
        )
        self.mb1, self.vb1, self.b1 = self._adam_step(
            self.mb1, self.vb1, self.b1, self.grads["db1"], t
        )
        self.mb2, self.vb2, self.b2 = self._adam_step(
            self.mb2, self.vb2, self.b2, self.grads["db2"], t
        )

        self.mwhy, self.vwhy, self.why = self._adam_step(
            self.mwhy, self.vwhy, self.why, self.grads["dwhy"], t
        )
        self.mby, self.vby, self.by = self._adam_step(
            self.mby, self.vby, self.by, self.grads["dby"], t
        )

    def update_params_sgd(self):
        for k, param in [
            ("wf", self.wf),
            ("wi", self.wi),
            ("wo", self.wo),
            ("wc", self.wc),
            ("w1", self.w1),
            ("w2", self.w2),
            ("why", self.why),
        ]:
            param -= self.lr * self.grads["d" + k]
        for k, param in [
            ("bf", self.bf),
            ("bi", self.bi),
            ("bo", self.bo),
            ("bc", self.bc),
            ("b1", self.b1),
            ("b2", self.b2),
            ("by", self.by),
        ]:
            param -= self.lr * self.grads["d" + k]

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
        train_set_indices: list,
        epochs: int = 1000,
        lr: float = 0.001,
        optimizer: Literal["adam", "sgd"] = "adam",
    ):
        self.lr = lr
        t = 0

        for epoch in range(epochs):
            epoch_losses = []
            for seq in train_set_indices:
                t += 1
                out_p, supp_p, act_p, xt = self.forward(seq)

                avg_loss = []
                for idx in range(1, len(seq)):
                    target_onehot = np.eye(self.vocab_size)[seq[idx]]
                    loss_t = self.categorical_cross_entropy(
                        out_p["yt"][idx], target_onehot
                    )
                    avg_loss.append(loss_t)

                mean_seq_loss = float(np.mean(avg_loss))
                epoch_losses.append(mean_seq_loss)

                self.backward(out_p, supp_p, act_p, seq, xt)

                if optimizer == "adam":
                    self.update_params_adam(t)
                else:
                    self.update_params_sgd()

            mean_epoch_loss = float(np.mean(epoch_losses))
            self._train_visualization(mean_epoch_loss, epochs, epoch)
            time.sleep(0.001)

    def generate_text(self, start_word: str, words: list, length: int = 5) -> str:
        if start_word not in words:
            return f"Kata '{start_word}' tidak ada dalam kosakata."

        current_word_idx = words.index(start_word)
        sentence = [start_word]

        for _ in range(length):
            x_embed = self.embedding_forward(current_word_idx)
            y_prob = self.forward_predict(x_embed)
            next_idx = int(np.argmax(y_prob))

            sentence.append(words[next_idx])
            current_word_idx = next_idx

        return " ".join(sentence)


if __name__ == "__main__":
    # Vocabulary & Dataset Setup
    words = ["saya", "makan", "nasi", "pake", "ayam"]
    vocab_size = len(words)

    sentence_indices = [words.index(w) for w in words]
    train_set_indices = [sentence_indices]

    print("--- Training Long Short-Term Memory (LSTM) ---")
    lstm = LongShortTermMemory(
        vocab_size=vocab_size,
        embed_dim=4,
        h_size=4,
        c_size=4,
        fc_hidden_size=10,
    )
    lstm.train(train_set_indices, epochs=1000, lr=0.001, optimizer="adam")

    print("\n--- Hasil Prediksi Teks ---")
    test_words = ["saya", "nasi", "pake"]
    for word in test_words:
        generated = lstm.generate_text(start_word=word, words=words, length=3)
        print(f"Start: '{word}' -> Generated: '{generated}'")
