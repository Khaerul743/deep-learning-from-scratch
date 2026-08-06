# Architecture Overview — Long Short-Term Memory (LSTM)

Dokumen ini menjelaskan arsitektur dan prinsip kerja **Long Short-Term Memory (LSTM)** yang diimplementasikan dari nol (*from scratch*). LSTM dikembangkan oleh Hochreiter & Schmidhuber (1997) sebagai arsitektur *recurrent neural network* khusus yang mampu mengatasi keterbatasan Vanilla RNN dalam mempelajari ketergantungan jangka panjang (*long-term dependencies*).

---

## 1. Motivasi & Arsitektur Utama LSTM

Pada Vanilla RNN, perbaruan hidden state terjadi melalui transformasi non-linear $\tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$. Ketika sekuens semakin panjang, perkalian matriks bobot $W_{hh}$ secara berulang saat Backpropagation Through Time (BPTT) menyebabkan gradient menghilang (*vanishing gradient*) secara eksponensial.

LSTM menyelesaikan masalah ini dengan memperkenalkan dua jalur memori utama pada setiap cell:
1. **Cell State ($C_t$)**: Jalur memori jangka panjang (*long-term memory*) yang mengalir secara teratur dengan sedikit interaksi non-linear (berfungsi sebagai *Constant Error Carousel*).
2. **Hidden State ($h_t$)**: Jalur memori jangka pendek (*short-term memory*) yang digunakan untuk prediksi pada langkah waktu saat ini.

Aliran informasi di dalam cell diatur oleh **tiga gerbang utama (Gates)** berbasis aktivasi Sigmoid:

![Gambaran Arsitektur LSTM](../images/arsitektur.png)

---

## 2. Persamaan Matematika Forward Pass

Pada langkah waktu $t$, cell LSTM menerima input $x_t \in \mathbb{R}^{D}$, hidden state sebelumnya $h_{t-1} \in \mathbb{R}^{H}$, dan cell state sebelumnya $C_{t-1} \in \mathbb{R}^{H}$.

Input digabungkan menjadi vektor konkatenasi:

$$
x_{\text{concat}, t} = [h_{t-1}, x_t] \in \mathbb{R}^{H + D}
$$

### 2.1 Forget Gate ($f_t$)
Menentukan seberapa banyak informasi dari cell state masa lalu ($C_{t-1}$) yang harus **dibuang** atau dilupakan:

$$
f_t = \sigma(W_f \cdot x_{\text{concat}, t} + b_f)
$$

Dimana $\sigma(x) = \frac{1}{1 + e^{-x}}$ menghasilkan nilai antara $0$ (lupakan total) dan $1$ (pertahankan total).

---

### 2.2 Input Gate ($i_t$) & Candidate Cell State ($\tilde{C}_t$)
Menentukan informasi baru mana yang akan **disimpan** ke dalam cell state:

1. **Input Gate**: Menyeleksi dimensi mana dari informasi baru yang perlu diperbarui:
   $$
   i_t = \sigma(W_i \cdot x_{\text{concat}, t} + b_i)
   $$

2. **Candidate Cell State**: Menghitung kandidat nilai baru menggunakan aktivasi $\tanh$:
   $$
   \tilde{C}_t = \tanh(W_c \cdot x_{\text{concat}, t} + b_c)
   $$

---

### 2.3 Pembaruan Cell State ($C_t$)
Cell state diperbarui melalui kombinasi penjumlahan elemen-wise:

$$
C_t = (f_t \odot C_{t-1}) + (i_t \odot \tilde{C}_t)
$$

Dimana $\odot$ adalah perkalian Hadamard (*element-wise product*).

---

### 2.4 Output Gate ($o_t$) & Hidden State ($h_t$)
Menentukan bagian mana dari cell state yang akan **dikeluarkan** sebagai hidden state $h_t$:

1. **Output Gate**:
   $$
   o_t = \sigma(W_o \cdot x_{\text{concat}, t} + b_o)
   $$

2. **Hidden State**:
   $$
   h_t = o_t \odot \tanh(C_t)
   $$

---

### 2.5 Proyeksi Layer Fully Connected (FC) & Loss

Hidden state $h_t$ diproyeksikan melalui layer MLP/Dense:

$$
z_1 = W_1 h_t + b_1 \quad \implies \quad a_{z1} = \text{ReLU}(z_1)
$$

$$
z_2 = W_2 a_{z1} + b_2 \quad \implies \quad a_{z2} = \text{ReLU}(z_2)
$$

$$
\hat{y}_t = \text{Softmax}(W_{hy} a_{z2} + b_y)
$$

Loss dihitung menggunakan Categorical Cross-Entropy:

$$
L_t = -\sum_{k=1}^{V} y_{\text{true}, t, k} \log(\hat{y}_{t, k})
$$

---

## 3. Prinsip Constant Error Carousel (CEC)

Mengapa LSTM mampu menangani ketergantungan jangka panjang?

Pada Vanilla RNN, turunan hidden state bersifat perkalian berulang:

$$
\frac{\partial h_t}{\partial h_{t-1}} = \text{diag}(1 - h_t^2) W_{hh}^T
$$

Pada LSTM, turunan Cell State mengandung komponen penjumlahan:

$$
\frac{\partial C_t}{\partial C_{t-1}} = f_t + C_{t-1} \frac{\partial f_t}{\partial C_{t-1}} + \dots
$$

Jika forget gate $f_t \approx 1$ dan input gate $i_t \approx 0$, maka $\frac{\partial C_t}{\partial C_{t-1}} \approx 1$. Error dapat mengalir mundur melintasi ratusan time step tanpa terdistorsi atau terhapus. Mekanisme jalur linear ini disebut **Constant Error Carousel (CEC)**.

---

## 4. Derivasi Backpropagation Through Time (BPTT)

Selama BPTT, gradient dihitung terbalik dari $t = T$ menuju $t = 1$.

### 4.1 Gradient pada Proyeksi Layer

$$
\delta_{y, t} = \hat{y}_t - y_{\text{true}, t}
$$

$$
\frac{\partial L_t}{\partial W_{hy}} = \delta_{y, t} \otimes a_{z2, t}, \quad \frac{\partial L_t}{\partial b_y} = \delta_{y, t}
$$

Melalui rantai turunan MLP:

$$
\delta_{z2, t} = (\delta_{y, t} W_{hy}) \odot \text{ReLU}'(z_{2, t})
$$

$$
\delta_{z1, t} = (\delta_{z2, t} W_2) \odot \text{ReLU}'(z_{1, t})
$$

$$
dh_t = \delta_{z1, t} W_1 + dh_{\text{total}}
$$

---

### 4.2 Gradient pada Gerbang LSTM

1. **Output Gate**:
   $$
   do_t = (dh_t \odot \tanh(C_t)) \odot \sigma'(o_t)
   $$

2. **Cell State ($dC_t$)**:
   $$
   dC_t = (dh_t \odot o_t \odot \tanh'(C_t)) + dC_{\text{total}}
   $$

3. **Forget Gate**:
   $$
   df_t = (dC_t \odot C_{t-1}) \odot \sigma'(f_t)
   $$

4. **Input Gate**:
   $$
   di_t = (dC_t \odot \tilde{C}_t) \odot \sigma'(i_t)
   $$

5. **Candidate Cell State**:
   $$
   d\tilde{C}_t = (dC_t \odot i_t) \odot \tanh'(\text{cell\_state}_t)
   $$

---

### 4.3 Akumulasi Gradient Bobot Matriks

Gradient untuk bobot gerbang dihitung terhadap vektor input konkatenasi $x_{\text{concat}, t} = [h_{t-1}, x_t]$:

$$
dW_f = df_t \otimes x_{\text{concat}, t}, \quad db_f = df_t
$$

$$
dW_i = di_t \otimes x_{\text{concat}, t}, \quad db_i = di_t
$$

$$
dW_o = do_t \otimes x_{\text{concat}, t}, \quad db_o = do_t
$$

$$
dW_c = d\tilde{C}_t \otimes x_{\text{concat}, t}, \quad db_c = d\tilde{C}_t
$$

---

### 4.4 Propagasi Error ke Step Sebelumnya ($t-1$)

Gradient dipropagasikan ke hidden state sebelumnya ($h_{t-1}$) dan cell state sebelumnya ($C_{t-1}$):

$$
dh_{t-1} = (df_t W_f)_{[:H]} + (di_t W_i)_{[:H]} + (do_t W_o)_{[:H]} + (d\tilde{C}_t W_c)_{[:H]}
$$

$$
dC_{t-1} = dC_t \odot f_t
$$

Nilai $dh_{t-1}$ dan $dC_{t-1}$ diteruskan ke iterasi $t-1$.

---

## 5. Ringkasan Komputasi LSTM

```text
Forward Pass:
[h_{t-1}, x_t] ───> Gates (f_t, i_t, o_t, C~_t)
                         │
C_{t-1} ───────────> [ C_t = f_t*C_{t-1} + i_t*C~_t ] ───> [ h_t = o_t * tanh(C_t) ] ───> MLP ───> Softmax

Backward Pass (BPTT):
dC_{t-1} <───────── [ dC_t = dh_t*o_t*tanh'(C_t) + dC_{t} ] <─── dh_t <─── MLP Gradient <─── (y_t - y_true)
                         │
dh_{t-1} <───────── [ df_t*W_f + di_t*W_i + do_t*W_o + dC~_t*W_c ]
```
