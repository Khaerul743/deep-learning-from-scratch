# Math Foundation

Sebelum mengimplementasikan **Convolutional Neural Network (CNN)** dari nol, penting untuk memahami beberapa konsep matematis dasar yang menjadi fondasi dari arsitektur ini. CNN pada dasarnya adalah kombinasi dari beberapa operasi matematis yang bekerja secara berurutan untuk mengekstraksi pola dari sebuah gambar.

Secara umum CNN terdiri dari beberapa komponen utama:

- Convolution Operation
- Activation Function
- Pooling Operation
- Fully Connected Layer

Semua komponen tersebut bekerja dengan dasar matematika seperti **linear algebra, discrete convolution, dan optimisasi menggunakan gradient descent**.

---

# Representasi Gambar

Dalam konteks machine learning, sebuah gambar direpresentasikan sebagai **tensor numerik**.

Jika sebuah gambar grayscale memiliki ukuran:

$$
H \times W
$$

maka gambar tersebut dapat direpresentasikan sebagai matriks:

$$
X \in \mathbb{R}^{H \times W}
$$

Jika gambar memiliki **channel warna (RGB)** maka bentuknya menjadi:

$$
X \in \mathbb{R}^{H \times W \times C}
$$

dimana:

- $H$ = Height (tinggi gambar)  
- $W$ = Width (lebar gambar)  
- $C$ = Channel (biasanya 3 untuk RGB)

Contoh:

$$
32 \times 32 \times 3
$$

---

# Convolution Operation

Operasi inti dalam CNN adalah **convolution**. Operasi ini bertujuan untuk mengekstraksi fitur dari gambar menggunakan **kernel (filter)**.

Misalkan terdapat:

- Input image $X$
- Kernel $K$

Kernel memiliki ukuran:

$$
k \times k
$$

Operasi convolution dilakukan dengan cara **menggeser kernel di seluruh area gambar** dan menghitung hasil perkalian elemen per elemen.

Secara matematis dapat dituliskan sebagai:

$$
Y(i,j) = \sum_m \sum_n X(i+m, j+n) \cdot K(m,n)
$$

dimana:

- $X$ = input image
- $K$ = kernel
- $Y$ = output feature map

Operasi ini menghasilkan **feature map** yang berisi respon terhadap pola tertentu pada gambar.

Contoh pola yang bisa dideteksi oleh kernel:

- edge
- texture
- shape
- object parts

---

# Stride

**Stride** adalah jumlah pergeseran kernel saat melakukan operasi convolution.

Jika stride dilambangkan dengan $S$, maka output size dapat dihitung dengan:

$$
\frac{N - F}{S} + 1
$$

dimana:

- $N$ = ukuran input
- $F$ = ukuran kernel
- $S$ = stride

Sebagai contoh:

Input = $32 \times 32$  
Kernel = $3 \times 3$  
Stride = $1$

maka output:

$$
30 \times 30
$$

---

# Padding

Padding digunakan untuk **menambahkan border pada gambar** agar ukuran output tidak terlalu kecil setelah convolution.

Jika padding dilambangkan dengan $P$, maka rumus output menjadi:

$$
\frac{N - F + 2P}{S} + 1
$$

Contoh:

- Input = $32$
- Kernel = $3$
- Padding = $1$
- Stride = $1$

maka:

$$
\frac{32 - 3 + 2(1)}{1} + 1 = 32
$$

Sehingga ukuran output tetap **32 × 32**.

---

# Activation Function

Setelah convolution, biasanya diterapkan **activation function** untuk menambahkan non-linearitas pada model.

Salah satu activation function yang paling umum digunakan pada CNN adalah **ReLU (Rectified Linear Unit)**.

Secara matematis:

$$
ReLU(x) = \max(0, x)
$$

Artinya:

- jika $x > 0$ maka output = $x$
- jika $x \le 0$ maka output = $0$

Fungsi ini membantu jaringan untuk mempelajari **representasi non-linear yang lebih kompleks**.

---

# Pooling Operation

Pooling digunakan untuk **mengurangi dimensi feature map** sambil mempertahankan informasi penting.

Pooling juga membantu:

- mengurangi jumlah parameter
- meningkatkan efisiensi komputasi
- membuat model lebih robust terhadap translasi kecil

Jenis pooling yang paling umum adalah **Max Pooling**.

Misalkan ukuran pooling:

$$
2 \times 2
$$

Maka setiap area $2 \times 2$ akan diambil nilai maksimum:

$$
Y = \max(x_1, x_2, x_3, x_4)
$$

Contoh:

$$
\begin{bmatrix}
1 & 3 \\
2 & 4
\end{bmatrix}
\rightarrow 4
$$

---

# Fully Connected Layer

Setelah melalui beberapa layer convolution dan pooling, feature map biasanya akan diubah menjadi **vektor 1 dimensi** menggunakan operasi **flatten**.

Misalkan feature map:

$$
7 \times 7 \times 64
$$

Maka setelah flatten menjadi:

$$
3136
$$

Vektor ini kemudian dimasukkan ke **fully connected layer** yang secara matematis sama seperti layer pada ANN:

$$
z = Wx + b
$$

dimana:

- $W$ = weight matrix
- $x$ = input vector
- $b$ = bias

Output dari layer ini kemudian digunakan untuk **classification atau prediction**.

---

# Loss Function

Untuk melatih CNN, kita menggunakan **loss function** untuk mengukur seberapa besar kesalahan prediksi model.

Untuk kasus klasifikasi biasanya digunakan **Cross Entropy Loss**.

Secara matematis:

$$
L = - \sum y_i \log(\hat{y}_i)
$$

dimana:

- $y_i$ = label sebenarnya
- $\hat{y}_i$ = probabilitas prediksi model

Tujuan training adalah **meminimalkan nilai loss ini**.

---

# Optimization

Parameter CNN dilatih menggunakan **gradient descent** dengan bantuan algoritma **backpropagation**.

Update weight dilakukan dengan rumus:

$$
w = w - \eta \frac{\partial L}{\partial w}
$$

dimana:

- $w$ = weight
- $\eta$ = learning rate
- $\frac{\partial L}{\partial w}$ = gradient dari loss terhadap weight

Proses ini dilakukan berulang hingga model mencapai performa yang optimal.