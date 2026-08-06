# Architecture Overview — Feed Forward

Bagian ini menjelaskan bagaimana proses **feed forward** bekerja pada Artificial Neural Network yang dibangun pada repository ini. Feed forward merupakan tahap di mana data bergerak dari input layer menuju output layer untuk menghasilkan prediksi.

Tujuan utama dari proses ini adalah melakukan transformasi data secara bertahap menggunakan operasi linear dan non-linear sehingga model mampu menangkap pola pada data.

---

## 1. Gambaran Umum Feed Forward

Sebelum masuk ke model, data biasanya melalui proses **normalisasi** (jika diperlukan) agar distribusi nilai lebih stabil selama proses training.

Setelah itu, data akan melewati serangkaian layer yang masing-masing terdiri dari dua tahap utama:

1. Transformasi linear (operasi matriks)
2. Fungsi aktivasi (transformasi non-linear)

Secara konseptual:

```text
Input → Linear Transformation → Activation → ... → Prediction
```

---

## 2. Transformasi Linear (Pre-Activation)

Langkah pertama dalam setiap layer adalah melakukan operasi linear antara input dan parameter model.

Secara matematis:

$$
z = XW + b
$$

Dimana:

* $X$ : matriks input (data)
* $W$ : matriks bobot (*weights*)
* $b$ : bias
* $z$ : nilai **pre-activation** (hasil sebelum fungsi aktivasi)

Pada implementasi menggunakan NumPy:

```python
z = X @ W + b
```

### Penjelasan Komponen

* **z**
  Variabel untuk menyimpan hasil transformasi linear sebelum masuk fungsi aktivasi.

* **X**
  Data input yang akan diproses oleh layer.

* **W**
  Parameter bobot model. Biasanya diinisialisasi menggunakan distribusi acak:

```python
W = np.random.randn(input_size, output_size)
```

* **b**
  Bias yang ditambahkan untuk memberikan fleksibilitas translasi fungsi:

```python
b = np.zeros((1, output_size))
```

* **@**
  Operator matrix multiplication di NumPy (setara dengan `np.dot()`).

Transformasi ini disebut **linear transformation** karena hanya melakukan kombinasi linear terhadap input.

---

## 3. Fungsi Aktivasi (Non-Linearity)

Jika neural network hanya terdiri dari operasi linear, maka beberapa layer linear yang ditumpuk tetap ekuivalen dengan satu transformasi linear saja. Artinya, model tidak mampu mempelajari pola non-linear.

Oleh karena itu digunakan **fungsi aktivasi**.

Setelah memperoleh nilai pre-activation $z$, kita menghitung:

$$
a = f(z)
$$

Dimana:

* $f$ = fungsi aktivasi
* $a$ = output layer (*activation*)

Dalam implementasi:

```python
a = ReLU(z)
```

---

### Fungsi Aktivasi yang Umum Digunakan

Beberapa fungsi aktivasi yang umum:

* ReLU (Rectified Linear Unit)
* Sigmoid
* Tanh

Pada project ini digunakan **ReLU** untuk hidden layer.

#### ReLU

$$
ReLU(z) = \max(0, z)
$$

ReLU mengubah semua nilai negatif menjadi nol dan mempertahankan nilai positif.

Keuntungan penggunaan ReLU:

* membantu menjaga stabilitas gradient dibanding sigmoid/tanh
* komputasi sederhana dan efisien
* mempercepat proses training pada banyak kasus

Turunan ReLU:

$$ReLU'(z) = 
\begin{cases} 
1 & \text{jika } z > 0 \\
0 & \text{jika } z \le 0 
\end{cases}$$

Turunan ini memungkinkan gradient tetap mengalir pada neuron yang aktif selama training.

---

## 4. Stacking Layer

Satu layer neural network terdiri dari:

```text
Linear Transformation → Activation
```

Layer-layer ini kemudian ditumpuk (*stacked*) beberapa kali:

$$
a^{(l)} = f\left(W^{(l)} a^{(l-1)} + b^{(l)}\right)
$$

Penumpukan layer memungkinkan model mempelajari hubungan **non-linear kompleks** pada data.

Semakin dalam jaringan, semakin abstrak representasi fitur yang dipelajari.

---

## 5. Output Layer dan Prediction

Tahap terakhir feed forward adalah menghasilkan prediksi.

Pemilihan fungsi aktivasi output bergantung pada jenis masalah.

---

### Binary Classification — Sigmoid

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

Sigmoid mengubah output menjadi rentang:

$$
0 \le \hat{y} \le 1
$$

sehingga dapat diinterpretasikan sebagai probabilitas.

---

### Regression

Untuk regresi biasanya:

* menggunakan aktivasi linear (tanpa fungsi aktivasi tambahan), atau
* dalam beberapa kasus menggunakan ReLU.

Hal ini karena output regresi tidak dibatasi pada rentang tertentu.

---

## 6. Perhitungan Error (Loss Function)

Setelah model menghasilkan prediksi $\hat{y}$, langkah berikutnya adalah menghitung error terhadap nilai sebenarnya $y$.

Loss function digunakan untuk mengukur seberapa jauh prediksi model dari target.

---

### Mean Squared Error (MSE) — Regression

$$
L = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
$$

Digunakan untuk masalah regresi.

---

### Binary Cross Entropy (BCE) — Binary Classification

$$
L = -\left[y\log(\hat{y}) + (1-y)\log(1-\hat{y})\right]
$$

Digunakan untuk klasifikasi biner.

Loss inilah yang nantinya menjadi sinyal utama dalam proses **backpropagation** untuk memperbarui parameter model.

---

## Ringkasan Feed Forward

Secara keseluruhan, proses feed forward dapat dirangkum sebagai berikut:

```text
Input Data
   ↓
Linear Transformation (XW + b)
   ↓
Activation Function
   ↓
Stacked Layers
   ↓
Output Activation
   ↓
Prediction
   ↓
Loss Computation
```

Feed forward menghasilkan prediksi dan nilai error yang akan digunakan pada tahap berikutnya: **backpropagation**.

# Architecture Overview — Backpropagation

Setelah proses **feed forward** menghasilkan prediksi dan nilai loss, langkah berikutnya adalah melakukan **backpropagation**, yaitu proses menghitung gradient untuk memperbarui parameter model agar prediksi menjadi lebih baik.

Jika feed forward menjawab pertanyaan:

> *“Apa hasil prediksi model?”*

maka backpropagation menjawab:

> *“Bagaimana cara memperbaiki parameter agar error menjadi lebih kecil?”*

---

## 1. Intuisi Dasar Backpropagation

Berdasarkan pengalaman saya dalam mengimplementasikan backpropagation, seluruh proses feed forward dapat dibayangkan sebagai **sebuah fungsi matematika besar dan dalam**.

Semua layer yang telah dilewati sebelumnya sebenarnya membentuk satu fungsi komposisi:

$$
L = f_n(f_{n-1}(...f_2(f_1(X))))
$$

Dimana:

* $X$ adalah input data
* setiap $f_i$ merepresentasikan operasi pada satu layer
* $L$ adalah loss function

Tujuan backpropagation adalah mencari bagaimana perubahan setiap parameter ($W$ dan $b$) mempengaruhi nilai loss:

$$
\frac{\partial L}{\partial W}
\quad \text{dan} \quad
\frac{\partial L}{\partial b}
$$

Artinya, kita ingin mengetahui:

> Seberapa besar kontribusi setiap parameter terhadap error model.

---

## 2. Analogi Sumur dan Tangga (Chain Rule)

Untuk memahami backpropagation secara intuitif, kita dapat menggunakan analogi berikut:

Bayangkan **loss function** sebagai sebuah sumur yang sangat dalam. Kita berada di dasar sumur tersebut dan ingin mengetahui bagaimana kita sampai ke sana.

Agar dapat naik kembali ke permukaan, kita membutuhkan tangga.

Dalam kalkulus, tangga tersebut adalah **Chain Rule**.

Alih-alih menghitung turunan setiap parameter dari awal fungsi secara langsung (yang sangat kompleks), kita dapat menurunkannya **secara bertahap**, layer demi layer.

Dengan kata lain:

* setiap layer menerima gradient dari layer berikutnya
* lalu meneruskannya ke layer sebelumnya

Gradient mengalir mundur mengikuti jalur komputasi feed forward.

---

## 3. Chain Rule dalam Neural Network

Karena neural network merupakan komposisi banyak fungsi, maka turunan dihitung menggunakan aturan rantai (*chain rule*).

Misalkan:

$$
L = L(a), \quad a = f(z), \quad z = XW + b
$$

Maka gradient terhadap weight dihitung sebagai:

$$
\frac{\partial L}{\partial W}
=============================

\frac{\partial L}{\partial a}
\cdot
\frac{\partial a}{\partial z}
\cdot
\frac{\partial z}{\partial W}
$$

Setiap komponen memiliki makna:

* $\frac{\partial L}{\partial a}$ → error dari layer berikutnya
* $\frac{\partial a}{\partial z}$ → turunan fungsi aktivasi
* $\frac{\partial z}{\partial W}$ → kontribusi weight terhadap output linear

Inilah alasan mengapa gradient dapat dihitung secara efisien tanpa menghitung ulang seluruh fungsi dari awal.

---

## 4. Aliran Gradient (Gradient Flow)

Pada feed forward, data mengalir dari input ke output:

```text id="forward-flow"
Input → Layer → Layer → Output
```

Sedangkan pada backpropagation, gradient bergerak sebaliknya:

```text id="backward-flow"
Loss ← Layer ← Layer ← Input
```

Setiap layer melakukan dua tugas utama:

1. Menghitung gradient parameter layer tersebut
2. Mengirim gradient ke layer sebelumnya

---

## 5. Gradient pada Output Layer

Backpropagation dimulai dari loss.

Misalkan:

* prediksi: $\hat{y}$
* target: $y$
* loss: $L$

Untuk contoh Mean Squared Error:

$$
L = \frac{1}{n}(y - \hat{y})^2
$$

Gradient pertama yang dihitung adalah:

$$
\frac{\partial L}{\partial \hat{y}}
$$

Nilai ini menjadi sinyal error awal yang akan dipropagasikan ke seluruh jaringan.

---

## 6. Backpropagation pada Satu Layer

Untuk satu layer:

$$
z = XW + b
$$

$$
a = f(z)
$$

Langkah backward:

### 1️⃣ Gradient terhadap pre-activation

$$
dZ = dA \odot f'(z)
$$

dimana $\odot$ adalah perkalian elemen-wise.

---

### 2️⃣ Gradient terhadap weight

$$
dW = X^T dZ
$$

---

### 3️⃣ Gradient terhadap bias

$$
db = \sum dZ
$$

---

### 4️⃣ Gradient untuk layer sebelumnya

$$
dA_{prev} = dZ W^T
$$

Nilai ini kemudian dikirim ke layer sebelumnya dan proses berulang hingga input layer.

---

## 7. Update Parameter (Gradient Descent)

Setelah gradient diperoleh, parameter diperbarui menggunakan gradient descent:

$$
W := W - \alpha dW
$$

$$
b := b - \alpha db
$$

Dimana $\alpha$ adalah learning rate.

Tujuan update ini adalah bergerak menuju nilai loss yang lebih kecil.

---

## 8. Ringkasan Proses Backpropagation

Secara keseluruhan:

```text id="bp-summary"
Compute Loss
      ↓
Compute Output Gradient
      ↓
Apply Chain Rule
      ↓
Compute Layer Gradients
      ↓
Propagate Gradient Backward
      ↓
Update Parameters
```

Backpropagation memungkinkan neural network belajar secara efisien dengan menghitung kontribusi setiap parameter terhadap error tanpa perlu menghitung ulang seluruh fungsi dari awal.

---

## Insight Konseptual

Backpropagation bukanlah proses yang “ajaib”. Ia hanyalah aplikasi sistematis dari aturan turunan kalkulus terhadap fungsi komposisi yang sangat besar.

Feed Forward membangun fungsi.
Backpropagation menurunkan fungsi tersebut.

Keduanya merupakan dua sisi dari proses pembelajaran neural network.
