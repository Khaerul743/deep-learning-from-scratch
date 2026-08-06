# Architecture

Pada bagian ini kita akan membahas **arsitektur model CNN** yang digunakan dalam project ini serta bagaimana setiap komponennya diimplementasikan dalam bentuk kode.

Secara umum, arsitektur **Convolutional Neural Network (CNN)** terdiri dari berbagai komponen seperti:

- Convolution
- Stride
- Padding
- Activation Function
- Pooling
- Fully Connected Layer

Namun pada project ini saya **tidak menggunakan seluruh komponen tersebut**. Hal ini dikarenakan tujuan dari project ini adalah membuat implementasi **CNN from scratch dalam skala kecil** untuk memahami operasi utamanya, bukan untuk membangun model yang kompleks.

Dataset yang digunakan juga relatif sederhana, yaitu **subset dari dataset MNIST** yang hanya berisi **digit 0 dan 1**. Dengan kompleksitas dataset yang rendah, penggunaan seluruh komponen CNN seperti **padding, pooling, atau arsitektur yang dalam (deep architecture)** tidak terlalu diperlukan.

Oleh karena itu, saya memutuskan untuk hanya menggunakan **komponen inti dari CNN** yang menurut saya sudah cukup untuk menyelesaikan permasalahan klasifikasi sederhana ini.

Komponen arsitektur yang digunakan dalam model ini adalah:

1. Convolution  
2. Stride  
3. Fully Connected Layer  

Alasan pemilihan komponen tersebut adalah karena ketiganya sudah cukup untuk membangun pipeline CNN sederhana yang mampu:

- melakukan ekstraksi fitur dasar dari gambar
- mengurangi dimensi secara alami melalui pergeseran kernel (stride)
- melakukan proses klasifikasi pada tahap akhir menggunakan fully connected layer

Dengan pendekatan ini, model tetap mampu melakukan prediksi **digit biner (0 atau 1)** dengan baik, sekaligus menjaga kompleksitas model tetap rendah.

Selain itu, arsitektur yang lebih sederhana juga membantu dalam beberapa hal:

- menghemat resource komputasi
- mempermudah proses debugging
- memudahkan pemahaman terhadap mekanisme forward dan backpropagation pada CNN

Pada bagian berikutnya kita akan membahas **setiap komponen arsitektur tersebut secara lebih detail**, termasuk **implementasi matematis dan implementasi kodenya**.

# Convolution Layer

## Konsep

Convolution merupakan operasi utama dalam **Convolutional Neural Network (CNN)** yang digunakan untuk mengekstraksi pola dari sebuah gambar.

Ide utama dari convolution adalah menggunakan sebuah **kernel (filter)** yang digeser di seluruh area gambar untuk menangkap pola tertentu seperti:

- edge
- garis
- bentuk
- tekstur

Setiap kernel akan menghasilkan sebuah **feature map**, yaitu representasi baru dari gambar yang menyoroti pola tertentu.

Pada implementasi project ini saya menggunakan **dua tahap convolution**:

1. Convolution pertama untuk mengekstraksi feature dari input image.
2. Convolution kedua yang bekerja pada **stack feature map (multi-channel)** yang dihasilkan dari convolution sebelumnya.

Dengan kata lain, output dari convolution pertama tidak langsung digunakan untuk klasifikasi, tetapi terlebih dahulu digabungkan menjadi **tensor multi-channel**, kemudian dilakukan convolution kembali untuk mengekstraksi fitur yang lebih kompleks.

Struktur data yang digunakan pada tahap ini adalah:

- Input image → matrix $(H, W)$
- Feature map stack → tensor $(C_{in}, H, W)$
- Kernel → tensor $(C_{out}, C_{in}, F, F)$

dimana:

- $C_{in}$ = jumlah channel input
- $C_{out}$ = jumlah kernel output
- $F$ = ukuran kernel

---

# Mathematical Formulation

Secara matematis operasi convolution dapat dituliskan sebagai:

$$
Y(i,j) = \sum_m \sum_n X(i+m, j+n) \cdot K(m,n)
$$

dimana:

- $X$ adalah input image
- $K$ adalah kernel
- $Y$ adalah output feature map

Kernel akan melakukan **element-wise multiplication** dengan area lokal pada gambar, kemudian hasilnya dijumlahkan untuk menghasilkan satu nilai pada feature map.

Ukuran output feature map dapat dihitung menggunakan:

$$
H_{out} = \frac{H - F + 2P}{S} + 1
$$

$$
W_{out} = \frac{W - F + 2P}{S} + 1
$$

dimana:

- $H$ = tinggi input
- $W$ = lebar input
- $F$ = ukuran kernel
- $P$ = padding
- $S$ = stride

Pada implementasi project ini **padding tidak digunakan**, sehingga rumusnya menjadi:

$$
H_{out} = \frac{H - F}{S} + 1
$$

---

# Convolution Implementation

Berikut adalah implementasi operasi convolution sederhana menggunakan **NumPy**.

Fungsi ini melakukan convolution antara sebuah **input map** dan **kernel tunggal**.

```python
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
```

### Cara Kerja Implementasi

Proses convolution pada fungsi ini dilakukan dalam beberapa langkah:

1. Menghitung ukuran output feature map.
2. Mengambil setiap **region lokal** dari input image dengan ukuran sama seperti kernel.
3. Melakukan **flatten** pada region tersebut.
4. Mengalikan region dengan kernel menggunakan operasi **dot product**.
5. Hasilnya kemudian disusun kembali menjadi **feature map 2D**.

Pendekatan ini memanfaatkan **operasi linear algebra (matrix multiplication)** untuk mempercepat perhitungan convolution.

---

# Multi Channel Convolution

Setelah feature map pertama dihasilkan, convolution berikutnya dilakukan pada **tensor multi-channel**.

Pada tahap ini setiap kernel akan bekerja pada **seluruh channel input**, kemudian hasilnya dijumlahkan untuk menghasilkan satu feature map output.

Struktur tensor yang digunakan adalah:

Feature Map

$$
(C_{in}, H, W)
$$

Kernel

$$
(C_{out}, C_{in}, F, F)
$$

Implementasinya adalah sebagai berikut:

```python
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
```

### Cara Kerja Multi Channel Convolution

Proses yang terjadi pada fungsi ini adalah:

1. Setiap **output channel** memiliki sekumpulan kernel.
2. Setiap kernel bekerja pada **satu channel input**.
3. Hasil convolution dari setiap channel dijumlahkan.
4. Hasil akhirnya menghasilkan **satu feature map baru**.

Secara konseptual proses ini dapat dituliskan sebagai:

$$
Y_{c_{out}} = \sum_{c_{in}} X_{c_{in}} * K_{c_{out},c_{in}}
$$

dimana:

- $X_{c_{in}}$ = input feature map pada channel tertentu
- $K_{c_{out},c_{in}}$ = kernel untuk channel tersebut
- $*$ = operasi convolution

Dengan pendekatan ini model dapat mempelajari **pola yang lebih kompleks dari kombinasi beberapa feature map**.

---

# Stride

## Konsep

Stride adalah parameter yang menentukan **seberapa jauh kernel bergeser setiap kali melakukan operasi convolution**.

Jika stride bernilai:

- $S = 1$ → kernel bergeser satu pixel setiap langkah
- $S = 2$ → kernel bergeser dua pixel setiap langkah

Semakin besar stride, maka ukuran feature map akan semakin kecil.

Pada project ini saya menggunakan:

$$
stride = 1
$$

Alasan menggunakan stride kecil adalah agar model dapat **menangkap pola pada gambar dengan lebih detail**, terutama karena dataset yang digunakan adalah **MNIST digit 0 dan 1** yang memiliki resolusi kecil.

Dengan stride = 1, setiap bagian kecil dari gambar tetap dianalisis oleh kernel sehingga informasi spasial tidak banyak hilang.

---

# Stride in Implementation

Stride diimplementasikan langsung pada proses pengambilan region pada fungsi convolution.

```python
start_i = i * stride
start_j = j * stride
```

Dengan pendekatan ini kernel akan berpindah sesuai nilai stride yang diberikan.

Jika stride = 1 maka kernel akan melakukan sliding window secara penuh di seluruh gambar.

Pendekatan ini memungkinkan model untuk menangkap pola lokal secara lebih detail dibandingkan menggunakan stride yang lebih besar.

# Flatten Operation

## Konsep

Setelah melewati beberapa operasi convolution, output yang dihasilkan berupa **tensor feature map** dengan dimensi:

$$
(C, H, W)
$$

dimana:

- $C$ = jumlah channel (feature map)
- $H$ = height
- $W$ = width

Namun layer **fully connected** hanya menerima input berupa **vektor satu dimensi**. Oleh karena itu kita perlu melakukan proses **flatten**.

Flatten adalah proses mengubah tensor multi dimensi menjadi **vector 1 dimensi** tanpa mengubah nilai datanya.

Sebagai contoh:

Feature map:

$$
(2, 6, 6)
$$

Setelah flatten menjadi:

$$
72
$$

dimana:

$$
2 \times 6 \times 6 = 72
$$

Flatten tidak melakukan transformasi matematis apapun, hanya **merubah bentuk data** agar dapat digunakan oleh layer berikutnya.

---

# Flatten Implementation

Pada implementasi ini flatten dilakukan menggunakan fungsi `numpy.flatten()`.

```python
flattened = a2.flatten()
```

Output dari operasi ini adalah sebuah vector satu dimensi yang kemudian akan digunakan sebagai **input ke fully connected layer**.

---

# Fully Connected Layer

## Konsep

Setelah feature map diubah menjadi vector satu dimensi, tahap selanjutnya adalah menggunakan **Fully Connected Layer (MLP)** sebagai **classifier head**.

Pada tahap ini model akan belajar memetakan fitur yang telah diekstraksi oleh CNN menjadi **output prediksi**.

Secara matematis fully connected layer dapat dituliskan sebagai:

$$
z = Wx + b
$$

dimana:

- $x$ = input vector
- $W$ = weight matrix
- $b$ = bias
- $z$ = output linear transformation

Setelah operasi linear ini biasanya diterapkan **activation function** untuk menambahkan non-linearity pada model.

Pada project ini digunakan:

- **ReLU** pada hidden layer
- **Sigmoid** pada output layer

Karena model melakukan **binary classification** (digit 0 atau 1).

---

# Mathematical Formulation

Forward propagation pada MLP dapat dituliskan sebagai:

Hidden Layer

$$
z_1 = xW_1 + b_1
$$

$$
a_1 = ReLU(z_1)
$$

Output Layer

$$
z_2 = a_1W_2 + b_2
$$

$$
\hat{y} = sigmoid(z_2)
$$

dimana:

- $\hat{y}$ adalah probabilitas prediksi model.

---

# Fully Connected Implementation

Berikut implementasi forward pass untuk bagian MLP pada kode:

```python
# Forward pass MLP
z1_mlp = flattened @ self.w4 + self.b1
a1_mlp = self.relu(z1_mlp)

z2_mlp = a1_mlp @ self.w5 + self.b2
y_pred = self.sigmoid(z2_mlp)
```

Penjelasan proses:

1. **Linear transformation pertama**

```
z1_mlp = flattened @ self.w4 + self.b1
```

Feature vector hasil flatten dikalikan dengan weight matrix.

2. **Activation function**

```
a1_mlp = self.relu(z1_mlp)
```

ReLU digunakan untuk menambahkan non-linearitas.

3. **Output layer**

```
z2_mlp = a1_mlp @ self.w5 + self.b2
```

4. **Sigmoid activation**

```
y_pred = self.sigmoid(z2_mlp)
```

Sigmoid digunakan karena output yang diinginkan adalah **probabilitas antara 0 dan 1**.

---

# Loss Function

## Konsep

Untuk mengukur seberapa baik model melakukan prediksi digunakan **loss function**.

Karena task yang dilakukan adalah **binary classification**, maka digunakan **Binary Cross Entropy (BCE)**.

Binary Cross Entropy mengukur perbedaan antara:

- label sebenarnya $y$
- probabilitas prediksi model $\hat{y}$

---

# Mathematical Formulation

Secara matematis BCE didefinisikan sebagai:

$$
L = -(y \log(\hat{y}) + (1-y)\log(1-\hat{y}))
$$

dimana:

- $y$ = label sebenarnya
- $\hat{y}$ = probabilitas prediksi model

Tujuan dari proses training adalah **meminimalkan nilai loss ini**.

---

# Loss Implementation

Pada kode, loss dihitung menggunakan fungsi berikut:

```python
loss = self.binary_cross_entropy(y, y_pred)
```

Nilai loss kemudian diakumulasikan untuk setiap sample dalam dataset.

```python
total_loss += loss
```

Loss ini nantinya akan digunakan pada proses **backpropagation** untuk memperbarui parameter model.

---

# Forward Pass Pipeline

Jika kita gabungkan seluruh proses forward pada model ini, maka alur komputasinya dapat diringkas sebagai berikut:

```
Input Image (20x20)
        │
        ▼
Conv Layer 1 (kernel w1)
        │
        ▼
Conv Layer 2 (kernel w2)
        │
        ▼
Stack Feature Maps
        │
        ▼
ReLU Activation
        │
        ▼
Multi Channel Convolution
        │
        ▼
ReLU Activation
        │
        ▼
Flatten
        │
        ▼
Fully Connected Layer
        │
        ▼
ReLU
        │
        ▼
Output Layer
        │
        ▼
Sigmoid
        │
        ▼
Binary Cross Entropy Loss
```

Pipeline ini menunjukkan bagaimana **CNN digunakan sebagai feature extractor**, kemudian hasilnya digunakan oleh **MLP sebagai classifier head** untuk menghasilkan prediksi akhir.

# Backpropagation

## Konsep

Backpropagation adalah proses menghitung **gradient dari loss terhadap setiap parameter model** sehingga parameter tersebut dapat diperbarui untuk meminimalkan error.

Secara umum tujuan dari backpropagation adalah menghitung:

$$
\frac{\partial L}{\partial w}
$$

dimana:

- $L$ = loss function
- $w$ = parameter model

Setelah gradient diperoleh, parameter diperbarui menggunakan algoritma optimisasi seperti **Gradient Descent** atau **Adam Optimizer**.

Pada project ini alur gradient mengalir dari:

```
Loss
 ↓
Output Layer
 ↓
MLP Layer
 ↓
Flatten
 ↓
Convolution Layer
```

Dengan kata lain, gradient dihitung **secara berurutan dari belakang ke depan** mengikuti aturan **chain rule pada kalkulus**.

---

# Backpropagation pada MLP

## Konsep

Bagian MLP pada model ini terdiri dari:

- Hidden layer dengan **ReLU**
- Output layer dengan **Sigmoid**

Output model:

$$
\hat{y} = sigmoid(z_2)
$$

Loss menggunakan **Binary Cross Entropy**.

Untuk kombinasi **sigmoid + BCE**, gradient sederhana menjadi:

$$
\frac{\partial L}{\partial z_2} = \hat{y} - y
$$

Ini adalah salah satu alasan kenapa kombinasi sigmoid dan BCE sering digunakan.

---

# Gradient Output Layer

Gradient pertama yang dihitung adalah terhadap output layer.

Implementasi pada kode:

```python
dy_pred = y_pred - y
```

Gradient weight pada layer output:

```python
dw5 = dy_pred * a1_mlp
db2 = dy_pred * 1
```

Secara matematis:

$$
\frac{\partial L}{\partial W_5} = a_1 \cdot (\hat{y} - y)
$$

---

# Gradient Hidden Layer

Selanjutnya gradient dipropagasikan kembali ke hidden layer.

```python
da1 = dy_pred * self.w5
dz1_mlp = da1 * self.relu_derivative(z1_mlp)
```

Secara matematis:

$$
\frac{\partial L}{\partial z_1} =
\frac{\partial L}{\partial a_1}
\cdot ReLU'(z_1)
$$

Kemudian gradient weight dihitung:

```python
dw4 = np.outer(flattened, dz1_mlp)
db1 = dz1_mlp
```

Secara matematis:

$$
\frac{\partial L}{\partial W_4} = x^T \cdot \delta
$$

dimana:

- $x$ adalah input flatten
- $\delta$ adalah gradient pada layer tersebut

---

# Gradient menuju CNN

Setelah gradient MLP dihitung, gradient harus dikembalikan ke **output convolution layer**.

```python
dx = dz1_mlp @ self.w4.T
```

Karena sebelumnya kita melakukan **flatten**, maka gradient perlu dikembalikan ke bentuk tensor:

```python
grad_from_mlp = dx.reshape(4, 16, 16)
```

Ini berarti gradient sekarang kembali dalam bentuk:

$$
(C, H, W)
$$

yang sesuai dengan bentuk feature map dari CNN.

---

# Backpropagation pada Convolution Layer

Backpropagation pada convolution layer lebih kompleks dibandingkan MLP karena terdapat dua gradient utama yang harus dihitung:

1. Gradient terhadap **kernel**

$$
\frac{\partial L}{\partial W}
$$

2. Gradient terhadap **input feature map**

$$
\frac{\partial L}{\partial X}
$$

---

# Gradient Kernel Convolution

Gradient kernel dihitung dengan melakukan convolution antara **input feature map** dan **delta dari layer berikutnya**.

Implementasi:

```python
dw2d_derivative(self, input_map, delta)
```

Kode:

```python
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
```

Secara matematis:

$$
\frac{\partial L}{\partial W} =
X * \delta
$$

dimana:

- $X$ = input feature map
- $\delta$ = gradient dari layer berikutnya

---

# Gradient Multi Channel Convolution

Untuk convolution dengan banyak channel, gradient dihitung untuk setiap pasangan:

```
(output channel, input channel)
```

Implementasi:

```python
dw_multi_channel_derivative(self, input_map, delta, w)
```

Kode:

```python
for i in range(q):
    for j in range(p):
        result = self.dw2d_derivative(input_map[j], delta[i])
        out[i, j] = result
```

Secara matematis:

$$
\frac{\partial L}{\partial W_{i,j}} =
X_j * \delta_i
$$

dimana:

- $i$ = output channel
- $j$ = input channel

---

# Gradient terhadap Input Feature Map

Selain menghitung gradient kernel, kita juga harus menghitung gradient terhadap **input convolution**.

Ini diperlukan agar gradient dapat dipropagasikan ke layer sebelumnya.

Implementasi:

```python
dx2d_derivative(self, w, delta, x)
```

Kode:

```python
w_flip = np.flip(w)

for i in range(p):
    for j in range(q):
        out[i : i + f, j : j + f] += w_flip * delta[i, j]
```

Secara matematis:

$$
\frac{\partial L}{\partial X} =
\delta * flip(W)
$$

Kernel harus **dibalik (flipped)** sebelum convolution dilakukan.

---

# Backpropagation Multi Channel

Implementasi untuk convolution multi channel:

```python
dx_conv_multi_channel(self, w, delta, x)
```

Kode:

```python
for j in range(C_in):
    for i in range(C_out):
        dx[j] += self.dx2d_derivative(w[i, j], delta[i], x[j])
```

Gradient dari setiap output channel dijumlahkan untuk mendapatkan gradient input channel.

---

# Backpropagation Pipeline

Jika seluruh proses digabungkan, alur gradient pada model ini adalah:

```
Loss
 ↓
Output Layer Gradient
 ↓
Hidden Layer Gradient
 ↓
Flatten Gradient
 ↓
Conv Layer 2 Gradient
 ↓
Conv Layer 1 Gradient
```

---

# Optimizer

Setelah semua gradient dihitung, parameter diperbarui menggunakan **Adam Optimizer**.

Contoh update weight:

```python
self.mw5, self.vw5, self.w5 = self.adam_optimizer(
    self.mw5, self.vw5, self.w5, dw5, self.t
)
```

Adam optimizer menggabungkan dua konsep utama:

- Momentum
- Adaptive learning rate

Secara matematis:

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t
$$

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2
$$

Parameter kemudian diperbarui menggunakan:

$$
w = w - \alpha \frac{m_t}{\sqrt{v_t} + \epsilon}
$$

---

# Tantangan Backpropagation pada CNN

Mengimplementasikan backpropagation pada CNN dari nol memiliki beberapa tantangan utama.

### 1. Dimensi Tensor

Berbeda dengan MLP yang hanya menggunakan matrix, CNN menggunakan **tensor multi dimensi** sehingga pengelolaan shape menjadi lebih kompleks.

---

### 2. Kernel Sharing

Pada convolution layer satu kernel digunakan berulang kali di seluruh area gambar.

Hal ini menyebabkan gradient harus **diakumulasi dari banyak lokasi**.

---

### 3. Multi Channel Convolution

Pada CNN modern setiap convolution melibatkan:

```
input channel
output channel
kernel spatial dimension
```

Sehingga perhitungan gradient harus mempertimbangkan semua kombinasi channel.

---

### 4. Gradient Flow

Jika tidak diimplementasikan dengan benar, gradient bisa:

- hilang (vanishing gradient)
- meledak (exploding gradient)

Oleh karena itu activation seperti **ReLU** sering digunakan untuk menjaga stabilitas training.

---

Dengan memahami dan mengimplementasikan backpropagation secara manual seperti ini, kita dapat melihat secara lebih jelas bagaimana **gradient benar-benar mengalir di dalam arsitektur CNN**, sesuatu yang biasanya disembunyikan oleh framework deep learning seperti TensorFlow atau PyTorch.