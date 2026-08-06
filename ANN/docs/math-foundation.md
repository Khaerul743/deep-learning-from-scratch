# Mathematical Foundation

Sebelum mengimplementasikan Artificial Neural Network dari nol, penting untuk memahami terlebih dahulu fondasi matematika yang menjadi dasar dari seluruh proses komputasi di dalam neural network.

Tujuan dari bagian ini bukan untuk membahas matematika secara teoritis mendalam, melainkan untuk memahami **bagaimana persamaan matematika dapat diterjemahkan menjadi baris kode** serta bagaimana konsep tersebut berperan langsung dalam proses pembelajaran model.

Artificial Neural Network pada dasarnya merupakan kombinasi dari dua cabang matematika utama:

1. **Aljabar Linear** — mengatur bagaimana data direpresentasikan dan diproses pada saat *forward pass*.
2. **Kalkulus** — menjelaskan bagaimana model belajar melalui perhitungan gradient pada proses *backpropagation*.

---

## 1. Aljabar Linear

### Intuisi Dasar

Aljabar linear mempelajari representasi data dalam bentuk **vektor** dan **matriks**, serta bagaimana transformasi matematis dilakukan di dalam ruang berdimensi tinggi.

Dalam konteks neural network, setiap data input direpresentasikan sebagai vektor, sedangkan parameter model (weight) direpresentasikan sebagai matriks. Proses pembelajaran model pada dasarnya adalah proses transformasi ruang fitur menggunakan operasi-operasi aljabar linear.

Walaupun secara visual kita sering membayangkan ruang 2D atau 3D, neural network bekerja pada ruang berdimensi jauh lebih tinggi (*high-dimensional space*).

---

### Peran Aljabar Linear dalam Neural Network

Pada proses **feedforward**, operasi utama yang dilakukan adalah:

* Perkalian matriks (matrix multiplication)
* Penjumlahan bias
* Transformasi linear terhadap input

Secara matematis:

[
z = Wx + b
]

Dimana:

* (x) = input vector
* (W) = weight matrix
* (b) = bias vector
* (z) = hasil transformasi linear

Operasi ini memungkinkan model untuk memproyeksikan data ke representasi baru sebelum melewati fungsi aktivasi.

---

### Mapping ke Implementasi Kode

| Konsep Matematika | Implementasi       |
| ----------------- | ------------------ |
| Vektor            | `numpy.array`      |
| Matriks Weight    | Array 2D NumPy     |
| Perkalian Matriks | `np.dot()` / `@`   |
| Penjumlahan Bias  | Broadcasting NumPy |

Contoh sederhana:

```python
z = X @ W + b
```

Baris kode tersebut merupakan representasi langsung dari persamaan linear di atas.

---

## 2. Kalkulus

### Intuisi Dasar

Jika aljabar linear menjadi fondasi dari proses **forward pass**, maka kalkulus menjadi fondasi dari proses **learning** itu sendiri.

Neural network dapat belajar karena mampu mengetahui **seberapa besar kesalahan prediksi** dan bagaimana cara memperbaikinya. Informasi tersebut diperoleh melalui konsep turunan (*derivative*).

Turunan memberikan informasi mengenai:

> Seberapa besar perubahan output jika parameter model sedikit diubah.

Informasi inilah yang disebut sebagai **gradient**.

---

### Peran Kalkulus dalam Backpropagation

Backpropagation menggunakan konsep turunan berantai (*chain rule*) untuk menghitung bagaimana error pada output dipengaruhi oleh setiap parameter di dalam jaringan.

Secara sederhana:

$$
\frac{\partial L}{\partial W}
$$

menunjukkan bagaimana perubahan weight mempengaruhi nilai loss.

Karena neural network terdiri dari banyak fungsi yang saling terhubung, maka gradient dihitung menggunakan chain rule:

$$
\frac{dL}{dW} = \frac{dL}{da} \cdot \frac{da}{dz} \cdot \frac{dz}{dW}
$$

Proses inilah yang memungkinkan error dari layer terakhir mengalir kembali ke layer sebelumnya.

---

### Konsep Kalkulus yang Digunakan

Beberapa konsep kalkulus yang sering muncul dalam implementasi ANN:

* Turunan fungsi (derivative)
* Partial derivative
* Multivariable differentiation
* Chain rule
* Gradient

Konsep-konsep ini digunakan untuk menghitung arah perubahan parameter agar loss semakin kecil.

---

### Mapping ke Implementasi Kode

| Konsep Matematika | Implementasi                   |
| ----------------- | ------------------------------ |
| Turunan Loss      | fungsi backward                |
| Gradient          | nilai return backward pass     |
| Chain Rule        | propagasi gradient antar layer |
| Update Parameter  | gradient descent               |

Contoh konsep dalam kode:

```python
dW = X.T @ dZ
W -= learning_rate * dW
```

Kode tersebut merepresentasikan proses optimasi parameter menggunakan gradient yang dihitung melalui turunan.

---

## Hubungan Aljabar Linear dan Kalkulus

Neural network bekerja karena kombinasi dua konsep ini:

* **Aljabar Linear** → mengubah representasi data (forward computation)
* **Kalkulus** → memperbaiki parameter model (learning process)

Secara sederhana:

```text
Forward Pass  → Linear Algebra
Backward Pass → Calculus
Learning      → Combination of Both
```

Tanpa aljabar linear, model tidak dapat memproses data.
Tanpa kalkulus, model tidak dapat belajar.

---

## Tujuan Bagian Ini

Setelah memahami fondasi matematika ini, pembaca diharapkan dapat:

* Memahami hubungan antara persamaan matematika dan implementasi kode
* Mengerti mengapa operasi tertentu digunakan dalam neural network
* Melihat bahwa proses learning bukanlah “magic”, melainkan hasil dari operasi matematis yang terstruktur

Bagian selanjutnya akan membahas bagaimana konsep-konsep ini diterapkan langsung dalam arsitektur neural network yang dibangun pada repository ini.
