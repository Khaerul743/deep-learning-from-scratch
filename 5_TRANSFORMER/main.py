import sys
import time
from typing import Literal

import numpy as np


class Transformer:
    """
    Decoder-style Transformer model built from scratch using NumPy.
    Implements Sinusoidal Positional Encoding, Masked Multi-Head Attention (MHA),
    Layer Normalization, Residual Connections, Position-wise Feed-Forward Networks (FFN),
    and full Backpropagation for end-to-end training.
    """

    def __init__(self, num_blocks: int = 1, num_heads: int = 2):
        self.total_block = num_blocks
        self.total_head_attention = num_heads

        self.embedings = np.array([])
        self.vocab_size = 0
        self.d_embeds = 0
        self.d_k = 0

        self.gammas = {}
        self.betas = {}

        self.weights = {"WQ": {}, "WK": {}, "WV": {}}
        self.values = {"Q": {}, "K": {}, "V": {}}
        self.weight_outs = {}
        self.attention_cache = {
            "x_input": {},
            "x_concat": {},
            "attention_out": {},
        }

        self.norm_cache = {
            "x": {},
            "x_norm": {},
            "mean": {},
            "var": {},
            "gamma": {},
            "beta": {},
            "eps": {},
        }

        self.ffn_weights = {"w1": {}, "w2": {}, "b1": {}, "b2": {}}
        self.ffn_cache = {"z1": {}, "a1": {}, "z2": {}, "a2": {}}

        self.x_final = np.array([])
        self.why = np.array([])
        self.output = {"logits": np.array([]), "predict": np.array([])}

        self.delta = np.array([])
        self.dwhy = np.array([])

        self.dgammas = {}
        self.dbetas = {}

        self.dffn_weights = {"w2": {}, "w1": {}, "b2": {}, "b1": {}}

        self.dweight_outs = {}
        self.dweights = {"WQ": {}, "WK": {}, "WV": {}}

        self.dx_total = np.array([])
        self.loss_history = []

    # --- Activation Functions & Utilities ---
    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        x = np.asarray(x)
        x_max = np.max(x, axis=axis, keepdims=True)
        e_x = np.exp(x - x_max)
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def _relu_derivative(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        return (x > 0).astype(x.dtype)

    def _create_look_ahead_mask(self, size: int) -> np.ndarray:
        return np.triu(np.ones((size, size)), k=1)

    def build_model(self, vocab_size: int, d_embeds: int):
        """
        Initialize parameter weights, biases, and normalization parameters.
        """
        self.vocab_size = vocab_size
        self.d_embeds = d_embeds
        self.d_k = d_embeds // self.total_head_attention

        # Layer Normalization parameters
        for i in range(self.total_block):
            for j in range(2):
                self.gammas[f"block_{i+1}_{j+1}"] = np.ones((1, d_embeds))
                self.betas[f"block_{i+1}_{j+1}"] = np.zeros((1, d_embeds))

        # Multi-Head Attention & FFN Weights Initialization
        for i in range(self.total_block):
            for j in range(self.total_head_attention):
                std_mha = np.sqrt(1.0 / self.d_embeds)
                self.weights["WQ"][f"block_{i+1}_{j+1}"] = (
                    np.random.randn(self.d_embeds, self.d_k) * std_mha
                )
                self.weights["WK"][f"block_{i+1}_{j+1}"] = (
                    np.random.randn(self.d_embeds, self.d_k) * std_mha
                )
                self.weights["WV"][f"block_{i+1}_{j+1}"] = (
                    np.random.randn(self.d_embeds, self.d_k) * std_mha
                )

            std_out = np.sqrt(1.0 / self.d_embeds)
            self.weight_outs[f"block_{i+1}"] = (
                np.random.randn(self.d_embeds, self.d_embeds) * std_out
            )

            # He Initialization for FFN (ReLU)
            std_ffn1 = np.sqrt(2.0 / self.d_embeds)
            self.ffn_weights["w1"][f"block_{i+1}"] = (
                np.random.randn(self.d_embeds, 4 * self.d_embeds) * std_ffn1
            )

            std_ffn2 = np.sqrt(2.0 / (4 * self.d_embeds))
            self.ffn_weights["w2"][f"block_{i+1}"] = (
                np.random.randn(4 * self.d_embeds, self.d_embeds) * std_ffn2
            )

            self.ffn_weights["b1"][f"block_{i+1}"] = np.zeros(
                (1, 4 * self.d_embeds)
            )
            self.ffn_weights["b2"][f"block_{i+1}"] = np.zeros((1, self.d_embeds))

        # Final Linear Output Projection (Why)
        self.why = np.random.randn(self.d_embeds, self.vocab_size) * np.sqrt(
            1.0 / self.d_embeds
        )

    def positional_encoding(self, seq_len: int, d_model: int) -> np.ndarray:
        """
        Compute Sinusoidal Positional Encoding.
        """
        pe = np.zeros((seq_len, d_model))
        position = np.arange(seq_len)[:, np.newaxis]
        div_term = np.exp(
            np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)
        )

        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return pe

    # --- Forward Pass ---
    def mask_multi_head_forward(self, x: np.ndarray, block: int) -> np.ndarray:
        if x.shape[1] != self.d_embeds:
            raise RuntimeError("The dimension x column does not match d_embeds.")

        mask = self._create_look_ahead_mask(x.shape[0]) * -1e9

        x_out = []
        for i in range(self.total_head_attention):
            head_name = f"block_{block}_{i+1}"
            self.values["Q"][head_name] = x @ self.weights["WQ"][head_name]
            self.values["K"][head_name] = x @ self.weights["WK"][head_name]
            self.values["V"][head_name] = x @ self.weights["WV"][head_name]

            scores = (
                self.values["Q"][head_name] @ self.values["K"][head_name].T
            ) / np.sqrt(self.d_k)
            masked_scores = scores + mask
            S = self._softmax(masked_scores)

            out = S @ self.values["V"][head_name]
            x_out.append(out)

        concat = np.concatenate(x_out, axis=1)
        self.attention_cache["x_concat"][f"block_{block}"] = concat
        self.attention_cache["attention_out"][f"block_{block}"] = (
            concat @ self.weight_outs[f"block_{block}"]
        )
        self.attention_cache["x_input"][f"block_{block}"] = x
        return concat @ self.weight_outs[f"block_{block}"]

    def _layer_norm_forward(
        self, x, gamma, beta, block: int, layer: int, eps: float = 1e-6
    ) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)

        x_norm = (x - mean) / np.sqrt(var + eps)
        out = gamma * x_norm + beta

        key = f"block_{block}_{layer}"
        self.norm_cache["x"][key] = x
        self.norm_cache["x_norm"][key] = x_norm
        self.norm_cache["mean"][key] = mean
        self.norm_cache["var"][key] = var
        self.norm_cache["gamma"][key] = gamma
        self.norm_cache["beta"][key] = beta
        self.norm_cache["eps"][key] = eps

        return out

    def add_and_norm(
        self,
        x_past: np.ndarray,
        x: np.ndarray,
        gamma: float,
        beta: float,
        block: int,
        layer: int,
    ) -> np.ndarray:
        x_rsd = x_past + x
        return self._layer_norm_forward(x_rsd, gamma, beta, block, layer)

    def feed_forward_network(self, x: np.ndarray, block: int) -> np.ndarray:
        key = f"block_{block}"
        z1 = x @ self.ffn_weights["w1"][key] + self.ffn_weights["b1"][key]
        a1 = self._relu(z1)
        z2 = a1 @ self.ffn_weights["w2"][key] + self.ffn_weights["b2"][key]

        self.ffn_cache["z1"][key] = z1
        self.ffn_cache["a1"][key] = a1
        self.ffn_cache["z2"][key] = z2
        return z2

    def forward(self, x: np.ndarray) -> np.ndarray:
        seq_len = x.shape[0]
        x_pos = self.positional_encoding(seq_len, self.d_embeds) + x

        for i in range(self.total_block):
            block_num = i + 1
            attention_out = self.mask_multi_head_forward(x_pos, block=block_num)
            x_hat_1 = self.add_and_norm(
                self.attention_cache["x_input"][f"block_{block_num}"],
                attention_out,
                self.gammas[f"block_{block_num}_1"],
                self.betas[f"block_{block_num}_1"],
                block_num,
                1,
            )
            ffn_out = self.feed_forward_network(x_hat_1, block=block_num)
            x_hat_2 = self.add_and_norm(
                x_hat_1,
                ffn_out,
                self.gammas[f"block_{block_num}_2"],
                self.betas[f"block_{block_num}_2"],
                block_num,
                2,
            )
            x_pos = x_hat_2

        self.x_final = x_hat_2
        self.output["logits"] = x_hat_2 @ self.why
        self.output["predict"] = self._softmax(self.output["logits"])

        return self.output["predict"]

    def _loss(self, pred: np.ndarray, target: np.ndarray) -> float:
        probs_at_target = pred[np.arange(len(target)), target]
        return float(-np.mean(np.log(probs_at_target + 1e-9)))

    # --- Backward Pass ---
    def _layer_norm_backward(
        self, dy: np.ndarray, block: int, layer: int
    ) -> np.ndarray:
        key = f"block_{block}_{layer}"
        x = self.norm_cache["x"][key]
        x_norm = self.norm_cache["x_norm"][key]
        mean = self.norm_cache["mean"][key]
        var = self.norm_cache["var"][key]
        gamma = self.norm_cache["gamma"][key]
        eps = self.norm_cache["eps"][key]

        N, D = x.shape

        dx_norm = dy * gamma
        std_inv = 1.0 / np.sqrt(var + eps)
        dvar = (
            np.sum(dx_norm * (x - mean), axis=-1, keepdims=True)
            * -0.5
            * (std_inv**3)
        )

        dmean = (
            np.sum(dx_norm * -std_inv, axis=-1, keepdims=True)
            + dvar * np.sum(-2.0 * (x - mean), axis=-1, keepdims=True) / N
        )

        dx = (dx_norm * std_inv) + (dvar * 2.0 * (x - mean) / D) + (dmean / D)
        return dx

    def _ffn_backward(self, delta: np.ndarray, block: int) -> np.ndarray:
        key = f"block_{block}"
        z1 = self.ffn_cache["z1"][key]
        a1 = self.ffn_cache["a1"][key]
        x_hat = self.norm_cache["x"][f"block_{block}_2"]

        w1 = self.ffn_weights["w1"][key]
        w2 = self.ffn_weights["w2"][key]

        self.dffn_weights["w2"][key] = a1.T @ delta
        self.dffn_weights["b2"][key] = np.sum(delta, axis=0, keepdims=True)

        da1 = delta @ w2.T
        dz1 = da1.copy()
        dz1[z1 <= 0] = 0

        self.dffn_weights["w1"][key] = x_hat.T @ dz1
        self.dffn_weights["b1"][key] = np.sum(dz1, axis=0, keepdims=True)

        return dz1 @ w1.T

    def _mha_backward(self, dy: np.ndarray, block: int) -> np.ndarray:
        key = f"block_{block}"
        concat = self.attention_cache["x_concat"][key]
        self.dweight_outs[key] = concat.T @ dy

        d_concat = dy @ self.weight_outs[key].T
        d_heads = np.split(d_concat, self.total_head_attention, axis=-1)

        dx_total = np.zeros_like(self.attention_cache["x_input"][key])
        mask = self._create_look_ahead_mask(dy.shape[0])

        for i in range(self.total_head_attention):
            head_name = f"block_{block}_{i+1}"

            Q = self.values["Q"][head_name]
            K = self.values["K"][head_name]
            V = self.values["V"][head_name]

            scores = (Q @ K.T) / np.sqrt(self.d_k)
            masked_scores = scores + (mask * -1e9)
            S = self._softmax(masked_scores)

            dS = d_heads[i] @ V.T
            dV_head = S.T @ d_heads[i]

            d_scores = S * (dS - np.sum(dS * S, axis=-1, keepdims=True))
            d_scores /= np.sqrt(self.d_k)

            dQ_head = d_scores @ K
            dK_head = d_scores.T @ Q

            x_in = self.attention_cache["x_input"][key]
            self.dweights["WQ"][head_name] = x_in.T @ dQ_head
            self.dweights["WK"][head_name] = x_in.T @ dK_head
            self.dweights["WV"][head_name] = x_in.T @ dV_head

            dx_total += dQ_head @ self.weights["WQ"][head_name].T
            dx_total += dK_head @ self.weights["WK"][head_name].T
            dx_total += dV_head @ self.weights["WV"][head_name].T

        return dx_total

    def backward(self) -> np.ndarray:
        ones = np.ones(self.vocab_size)
        for i in range(len(self.output["predict"])):
            if i + 1 == len(self.output["predict"]):
                continue
            one = int(ones[i])
            self.output["predict"][i][i + 1] -= one
        self.output["predict"][-1, :] = 0

        self.delta = self.output["predict"]
        self.dwhy = self.x_final.T @ self.delta
        self.delta = self.delta @ self.why.T

        for i in range(self.total_block, 0, -1):
            # Layer norm 2
            key2 = f"block_{i}_2"
            self.dgammas[key2] = np.sum(
                self.delta * self.norm_cache["x_norm"][key2],
                axis=0,
                keepdims=True,
            )
            self.dbetas[key2] = np.sum(self.delta, axis=0, keepdims=True)

            self.delta = self._layer_norm_backward(self.delta, i, 2)

            # FFN backward
            residual_ffn = self.delta.copy()
            self.delta = self._ffn_backward(self.delta, i)

            # Layer norm 1
            self.delta = self.delta + residual_ffn
            key1 = f"block_{i}_1"
            self.dgammas[key1] = np.sum(
                self.delta * self.norm_cache["x_norm"][key1],
                axis=0,
                keepdims=True,
            )
            self.dbetas[key1] = np.sum(self.delta, axis=0, keepdims=True)
            self.delta = self._layer_norm_backward(self.delta, i, 1)

            residual_mha = self.delta.copy()
            self.delta = self._mha_backward(self.delta, i)
            self.delta = self.delta + residual_mha

        self.dx_total = self.delta
        return self.delta

    def update_weights(self, lr: float = 0.01):
        for block in range(1, self.total_block + 1):
            for head in range(1, self.total_head_attention + 1):
                name = f"block_{block}_{head}"
                self.weights["WQ"][name] -= lr * self.dweights["WQ"][name]
                self.weights["WK"][name] -= lr * self.dweights["WK"][name]
                self.weights["WV"][name] -= lr * self.dweights["WV"][name]

            b_key = f"block_{block}"
            self.weight_outs[b_key] -= lr * self.dweight_outs[b_key]
            self.ffn_weights["w2"][b_key] -= lr * self.dffn_weights["w2"][b_key]
            self.ffn_weights["b2"][b_key] -= lr * self.dffn_weights["b2"][b_key]
            self.ffn_weights["w1"][b_key] -= lr * self.dffn_weights["w1"][b_key]
            self.ffn_weights["b1"][b_key] -= lr * self.dffn_weights["b1"][b_key]

            self.gammas[f"block_{block}_2"] -= (
                lr * self.dgammas[f"block_{block}_2"]
            )
            self.gammas[f"block_{block}_1"] -= (
                lr * self.dgammas[f"block_{block}_1"]
            )
            self.betas[f"block_{block}_2"] -= (
                lr * self.dbetas[f"block_{block}_2"]
            )
            self.betas[f"block_{block}_1"] -= (
                lr * self.dbetas[f"block_{block}_1"]
            )

        self.why -= lr * self.dwhy
        self.embedings -= lr * self.dx_total

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
        embeddings: np.ndarray,
        target: np.ndarray,
        lr: float = 0.001,
        epochs: int = 1000,
    ):
        self.embedings = embeddings.copy()
        for i in range(epochs):
            pred = self.forward(self.embedings)
            loss = self._loss(pred, target)
            self.loss_history.append(loss)

            self._train_visualization(loss, epochs, i)

            self.backward()
            self.update_weights(lr)
            time.sleep(0.001)

    def predicted(self, x: np.ndarray) -> int:
        pred = self.forward(x)
        last_word_probs = pred[-1, :]
        return int(np.argmax(last_word_probs))


if __name__ == "__main__":
    # Vocabulary & Dataset Setup
    vocabulary = ["saya", "makan", "nasi", "pake", "ayam", "goreng"]
    vocab_size = len(vocabulary)
    d_embeds = 4

    np.random.seed(42)
    embeddings = np.random.randn(vocab_size, d_embeds)
    targets = np.array([1, 2, 3, 4, 5])

    print("--- Training Transformer Model ---")
    model = Transformer(num_blocks=1, num_heads=2)
    model.build_model(vocab_size=vocab_size, d_embeds=d_embeds)
    model.train(embeddings, targets, lr=0.001, epochs=1000)

    print("\n--- Hasil Prediksi Teks ---")
    word_map = {vocabulary[idx]: v for idx, v in enumerate(model.embedings)}
    test_word = "ayam"
    word_input = word_map[test_word]
    pred_idx = model.predicted(word_input)

    print(f"Input Word: '{test_word}' -> Predicted Next Token: '{vocabulary[pred_idx]}'")
