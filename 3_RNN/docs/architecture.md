# Architecture Overview — Recurrent Neural Network (RNN) & Backpropagation Through Time (BPTT)

Dokumen ini menjelaskan arsitektur dan prinsip kerja **Vanilla Recurrent Neural Network (RNN)** yang diimplementasikan dari nol (*from scratch*). Berbeda dengan *Artificial Neural Network* (ANN) standar yang memproses input secara independen, RNN dirancang khusus untuk memproses data sekuensial (seperti teks, *time-series*, atau audio) dengan memanfaatkan memori internal berupa **hidden state** ($h_t$).

---

## 1. Gambaran Umum Recurrent Neural Network

Pada arsitektur neural network konvensional (Feedforward / ANN), setiap sampel input diproses tanpa memiliki keterkaitan dengan input sebelum atau sesudahnya. 

Pada RNN, data yang masuk berbentuk urutan (sekuens):

$$
X = (x_1, x_2, \dots, x_T)
$$

Dimana $T$ adalah panjang sekuens (*sequence length*). Untuk memproses data ini, RNN mendaur ulang hidden state dari langkah waktu (*time step*) sebelumnya ($h_{t-1}$) dan menggabungkannya dengan input saat ini ($x_t$).

Secara konseptual, RNN dapat dibayangkan sebagai sebuah cell tunggal yang diputar berulang kali atau dibuka sepanjang waktu (**Unrolling in Time**):

```text
Time Step t=1           Time Step t=2                    Time Step t=T
   x_1                     x_2                              x_T
    │                       │                                │
    ▼                       ▼                                ▼
┌───────┐   h_1         ┌───────┐   h_2       ...        ┌───────┐   h_T
│ Cell  │ ────────────> │ Cell  │ ──────────> ... ─────> │ Cell  │
└───────┘               └───────┘                        └───────┘
    │                       │                                │
    ▼                       ▼                                ▼
   y_1                     y_2                              y_T
```

---

## 2. Feed Forward / Forward Pass pada RNN

Pada setiap time step $t$, RNN melakukan transformasi linear dan non-linear untuk memperbarui *hidden state* dan menghasilkan prediksi output.

### 2.1 Perhitungan Hidden State (Pre-Activation & Activation)

Langkah pertama adalah menghitung nilai **pre-activation** hidden state $z_t$:

$$
z_t = W_{xh} x_t + W_{hh} h_{t-1} + b_h
$$

Dimana:
* $x_t \in \mathbb{R}^{V}$: Vektor input one-hot / embedding pada langkah waktu $t$ (dimensi $V$).
* $h_{t-1} \in \mathbb{R}^{H}$: Hidden state dari langkah waktu sebelumnya (dimensi $H$).
* $W_{xh} \in \mathbb{R}^{H \times V}$: Matriks bobot transformasi input ke hidden.
* $W_{hh} \in \mathbb{R}^{H \times H}$: Matriks bobot transformasi hidden ke hidden (koneksi rekuren).
* $b_h \in \mathbb{R}^{H}$: Vector bias hidden layer.

Setelah memperoleh $z_t$, kita menerapkan fungsi aktivasi **hyperbolic tangent ($\tanh$)** untuk memperoleh hidden state baru $h_t$:

$$
h_t = \tanh(z_t)
$$

Fungsi $\tanh$ membatasi nilai $h_t$ berada pada rentang $[-1, 1]$, menjaga stabilitas nilai state sepanjang sekuens.

---

### 2.2 Output Projection & Softmax Activation

Setelah hidden state $h_t$ diperbarui, model memproyeksikan hidden state ke ruang output (seperti ruang vocabulary):

$$
y_{logits, t} = W_{hy} h_t + b_y
$$

Dimana:
* $W_{hy} \in \mathbb{R}^{V \times H}$: Matriks bobot transformasi hidden ke output.
* $b_y \in \mathbb{R}^{V}$: Bias output.

Untuk memperoleh distribusi probabilitas kata berikutnya, digunakan fungsi **Softmax**:

$$
\hat{y}_t = \text{Softmax}(y_{logits, t}) = \frac{e^{y_{logits, t}}}{\sum_{j=1}^{V} e^{y_{logits, t, j}}}
$$

---

### 2.3 Perhitungan Cross-Entropy Loss Sekuensial

Loss untuk satu langkah waktu $t$ dihitung menggunakan **Categorical Cross-Entropy**:

$$
L_t = -\sum_{i=1}^{V} y_{true, t, i} \log(\hat{y}_{t, i})
$$

Total loss untuk satu sekuens panjang $T$ adalah rata-rata loss di seluruh langkah waktu:

$$
L = \frac{1}{T} \sum_{t=1}^{T} L_t
$$

---

## 3. Backpropagation Through Time (BPTT)

Untuk memperbarui bobot $W_{xh}, W_{hh}, W_{hy}, b_h, b_y$, kita tidak bisa menggunakan backpropagation biasa. Kita harus menggunakan **Backpropagation Through Time (BPTT)**, yaitu propagasi balik error yang dilakukan dengan membuka sekuens secara terbalik dari waktu $t = T$ menuju $t = 1$.

```text
Forward Pass:   x_1 ──> x_2 ──> ... ──> x_T
                 │       │               │
                 ▼       ▼               ▼
Loss:           L_1     L_2     ...     L_T
                 │       │               │
Backward Pass:  dz_1 <─ dz_2 <─ ... <─ dz_T (BPTT Gradient Accumulation)
```

---

### 3.1 Step 1: Gradient pada Layer Output

Pada time step $t$, turunan loss $L_t$ terhadap logits sebelum Softmax disederhanakan menjadi:

$$
\delta_{y, t} = \hat{y}_t - y_{true, t}
$$

Gradient terhadap bobot output $W_{hy}$ dan bias $b_y$ pada step $t$:

$$
\frac{\partial L_t}{\partial W_{hy}} = \delta_{y, t} \otimes h_t = \delta_{y, t} \cdot h_t^T
$$

$$
\frac{\partial L_t}{\partial b_y} = \delta_{y, t}
$$

---

### 3.2 Step 2: Gradient Flow ke Hidden State ($dh_t$)

Gradient yang mengalir ke hidden state $h_t$ berasal dari dua sumber:
1. Error dari output layer pada step $t$ ($\delta_{y, t} W_{hy}$).
2. Error yang dipropagasikan mundur dari time step setelahnya $t+1$ ($\delta_{h, t+1}$).

$$
dh_t = \delta_{y, t} W_{hy} + dh_{\text{delta}}
$$

---

### 3.3 Step 3: Gradient terhadap Pre-Activation ($dz_t$)

Memanfaatkan turunan aktivasi $\tanh$:

$$
\frac{d}{dz} \tanh(z) = 1 - \tanh^2(z) = 1 - h_t^2
$$

Sehingga gradient pre-activation $dz_t$ adalah:

$$
dz_t = dh_t \odot (1 - \tanh^2(z_t))
$$

Dimana $\odot$ adalah perkalian elemen-wise (Hadamard product).

---

### 3.4 Step 4: Akumulasi Gradient Parameter Rekuren & Input

Gradient untuk parameter $W_{xh}, W_{hh},$ dan $b_h$ dihitung pada step $t$ lalu diakumulasikan sepanjang seluruh sekuens $t = T \dots 1$:

$$
\frac{\partial L_t}{\partial W_{xh}} = dz_t \otimes x_{t-1}
$$

$$
\frac{\partial L_t}{\partial W_{hh}} = dz_t \otimes h_{t-1}
$$

$$
\frac{\partial L_t}{\partial b_h} = dz_t
$$

Terakhir, error diteruskan ke time step sebelumnya ($t-1$):

$$
dh_{\text{delta}} = dz_t W_{hh}
$$

Proses ini diulang secara rekursif hingga $t = 1$.

---

## 4. Strategi Inisialisasi Bobot (Weight Initialization)

Inisialisasi bobot sangat krusial pada RNN untuk mencegah keterbatasan propagasi sinyal.

1. **Xavier / Glorot Initialization ($W_{xh}, W_{hy}$)**
   Digunakan untuk proyeksi input-to-hidden dan hidden-to-output:
   $$
   W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{\text{fan\_in} + \text{fan\_out}}}\right)
   $$

2. **Orthogonal Initialization ($W_{hh}$)**
   Untuk bobot rekuren $W_{hh}$, digunakan inisialisasi matriks ortogonal (melalui dekomposisi QR dari matriks acak). Matriks ortogonal memiliki nilai eigen dengan magnitudo $| \lambda | = 1$, yang membantu menjaga magnitudo gradient agar tidak meledak atau hilang secara drastis saat dipropagasikan melintasi banyak langkah waktu.

---

## 5. Masalah Vanishing & Exploding Gradient pada RNN

Salah satu keterbatasan utama Vanilla RNN adalah fenomena **Vanishing & Exploding Gradient**.

Ketika aturan rantai (*chain rule*) dipropagasikan mundur melalui $T$ langkah waktu, gradient hidden state mengandung perkalian berulang matriks bobot $W_{hh}^T$:

$$
\frac{\partial h_T}{\partial h_1} = \prod_{k=2}^{T} \frac{\partial h_k}{\partial h_{k-1}} = \prod_{k=2}^{T} \text{diag}(1 - h_k^2) W_{hh}^T
$$

* Jika nilai singular terbesar dari $W_{hh}$ lebih kecil dari 1 ($\sigma_{\max} < 1$), gradient akan berkurang secara eksponensial menuju 0 (**Vanishing Gradient**), menyebabkan RNN gagal mempelajari ketergantungan jangka panjang (*long-term dependencies*).
* Jika $\sigma_{\max} > 1$, gradient dapat membesar secara eksponensial (**Exploding Gradient**), menyebabkan proses pelatihan menjadi tidak stabil (*NaN* / *Inf*).

Masalah ini menjadi motivasi utama ditemukannya arsitektur seperti **LSTM (Long Short-Term Memory)** dan **GRU (Gated Recurrent Unit)**.

---

## Ringkasan Alur Komputasi

```text
Forward Pass:
x_t ──> [ z_t = Wxh x_t + Whh h_{t-1} + bh ] ──> [ h_t = tanh(z_t) ] ──> [ y_t = Softmax(Why h_t + by) ]
                                                                                  │
                                                                                  ▼
                                                                           Loss Computation

Backward Pass (BPTT):
dwxh <── [ dz_t = dh_t * dtanh(z_t) ] <── [ dh_t = dy_t Why + dh_delta ] <── [ dy_t = y_t - y_true ]
   │
   ├──> dwhh = dz_t * h_{t-1}^T
   └──> dh_delta = dz_t Whh  ──> Propagasi ke step t-1
```
