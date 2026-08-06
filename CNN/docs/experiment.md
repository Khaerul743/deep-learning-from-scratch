# Experiments

Pada bagian ini kita akan membahas beberapa eksperimen yang dilakukan selama pengembangan CNN from scratch. Salah satu hal yang cukup terasa ketika mengimplementasikan convolution secara manual menggunakan **Python loop** adalah masalah **kecepatan komputasi**.

Operasi convolution yang ditulis menggunakan nested loop Python cenderung sangat lambat, terutama ketika ukuran gambar atau jumlah kernel mulai meningkat.

Sebagai contoh implementasi convolution manual:

```python
for i in range(h_out):
    for j in range(w_out):
        region = input_map[i:i+f, j:j+f]
        output[i, j] = np.sum(region * kernel)
```

Pendekatan ini bekerja dengan baik untuk memahami konsep convolution, tetapi secara performa kurang efisien karena:

- Python loop relatif lambat
- operasi tidak memanfaatkan optimisasi **linear algebra pada NumPy**

Untuk mengatasi hal ini, salah satu teknik yang sering digunakan adalah **im2col (image to column)**.

---

# Im2col

## Konsep

Im2col adalah teknik untuk mengubah operasi convolution menjadi **operasi matrix multiplication**.

Ide utamanya adalah:

1. Setiap patch lokal pada gambar diekstrak
2. Patch tersebut diubah menjadi **vektor**
3. Semua patch digabung menjadi **matrix**

Dengan cara ini, convolution dapat dihitung menggunakan satu operasi:

$$
Y = X_{col} \cdot W_{col}
$$

dimana:

- $X_{col}$ = matrix hasil transformasi image
- $W_{col}$ = kernel yang telah di-flatten

Karena operasi matrix multiplication sudah sangat dioptimalkan di NumPy, pendekatan ini jauh lebih cepat dibandingkan nested loop.

---

# Intuisi Im2col

Misalkan kita memiliki gambar:

$$
4 \times 4
$$

dan kernel:

$$
2 \times 2
$$

Convolution akan menghasilkan beberapa patch seperti:

```
patch1
patch2
patch3
...
```

Dengan **im2col**, setiap patch diubah menjadi vector:

```
patch1 -> [x1 x2 x3 x4]
patch2 -> [x5 x6 x7 x8]
```

Kemudian digabung menjadi matrix:

```
X_col =
[ x1 x2 x3 x4 ]
[ x5 x6 x7 x8 ]
[ ...        ]
```

Kernel juga diubah menjadi vector:

```
W_col =
[ k1
  k2
  k3
  k4 ]
```

Sehingga convolution dapat dihitung dengan:

```
Y = X_col @ W_col
```

---

# Implementasi Im2col

Berikut contoh implementasi sederhana **im2col menggunakan NumPy**.

```python
import numpy as np

def im2col(input_map, kernel_size, stride=1):
    H, W = input_map.shape
    F = kernel_size

    H_out = (H - F) // stride + 1
    W_out = (W - F) // stride + 1

    cols = []

    for i in range(H_out):
        for j in range(W_out):
            patch = input_map[
                i*stride : i*stride+F,
                j*stride : j*stride+F
            ]
            cols.append(patch.flatten())

    return np.array(cols)
```

---

# Convolution menggunakan Im2col

Setelah mendapatkan matrix hasil im2col, convolution dapat dihitung dengan **matrix multiplication**.

```python
def conv_im2col(input_map, kernel, stride=1):
    F = kernel.shape[0]

    cols = im2col(input_map, F, stride)

    kernel_col = kernel.flatten()

    output = cols @ kernel_col

    H_out = (input_map.shape[0] - F) // stride + 1
    W_out = (input_map.shape[1] - F) // stride + 1

    return output.reshape(H_out, W_out)
```

Dengan pendekatan ini seluruh operasi convolution dapat dilakukan hanya dengan:

```python
cols @ kernel
```

yang merupakan operasi matrix multiplication yang sangat cepat di NumPy.

---

# Keuntungan Im2col

Beberapa keuntungan dari pendekatan ini adalah:

### 1. Lebih Cepat

Matrix multiplication sudah sangat dioptimalkan pada library linear algebra seperti **BLAS**.

---

### 2. Mudah di-Vectorize

Dengan mengubah convolution menjadi operasi matrix, kita dapat menghindari **Python loop yang lambat**.

---

### 3. Digunakan pada Framework Besar

Teknik im2col juga digunakan pada banyak framework deep learning seperti:

- Caffe
- TensorFlow (versi awal)
- PyTorch (beberapa implementasi backend)

---

# Trade-off Im2col

Walaupun cepat, teknik ini memiliki satu kekurangan utama.

### Memory Overhead

Karena setiap patch gambar disalin menjadi matrix baru, penggunaan memori menjadi lebih besar dibandingkan implementasi convolution langsung.

Namun untuk banyak kasus, trade-off ini sebanding dengan peningkatan performa yang diperoleh.

---

# Kesimpulan

Implementasi convolution menggunakan nested loop sangat berguna untuk memahami bagaimana CNN bekerja secara internal. Namun ketika ukuran data mulai meningkat, pendekatan tersebut menjadi tidak efisien.

Dengan menggunakan teknik **im2col**, operasi convolution dapat diubah menjadi **matrix multiplication**, sehingga memanfaatkan optimisasi yang sudah tersedia pada library linear algebra seperti NumPy.

Pendekatan ini memberikan peningkatan performa yang signifikan dibandingkan implementasi convolution berbasis loop Python.