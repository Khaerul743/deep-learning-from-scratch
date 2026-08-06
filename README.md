# Deep Learning From Scratch: ANN to Transformer

Implementasi dan penjelasan komprehensif arsitektur Deep Learning — mulai dari **Artificial Neural Network (ANN)** hingga **Transformer** — yang dibangun dari nol (*from scratch*) hanya menggunakan **NumPy** dan **Python**.

---

## 💡 Motivasi & Alasan Pembuatan Proyek

Proyek ini dibangun sebagai **pondasi fundamental untuk mempelajari Artificial Intelligence (AI) secara lebih dalam**.

Dalam perkembangan AI saat ini, penggunaan framework tingkat tinggi seperti PyTorch, TensorFlow, atau Scikit-Learn sangat mempermudah pembuatan model. Namun, abstraksi tingkat tinggi tersebut sering kali membuat mekanisme internal model terasa seperti *black-box*:
* Aliran data dan propagasi turunan (*gradient flow*) tersembunyi di balik otomatisasi autograd.
* Konsep matematika dibalik pembaharuan bobot (*weight updates*) menjadi kurang intuitif.
* Sulit memahami secara presisi mengapa suatu arsitektur gagal atau sukses menangani jenis data tertentu.

Dengan membangun setiap arsitektur dari nol — hanya mengandalkan operasi perkalian matriks dan kalkulus di NumPy — proyek ini memberikan intuisi matematis dan teknis yang kuat mengenai bagaimana model AI sebenarnya "belajar" dan memproses informasi.

---

## 📚 Pondasi Matematika yang Dikuasai

Untuk memahami dan membangun arsitektur deep learning dari nol, terdapat 4 pilar matematika utama yang dipelajari dan diimplementasikan dalam repository ini:

### 1. Aljabar Linear (Linear Algebra)
* **Vektor dan Matriks**: Representasi data input, bobot (*weights*), dan bias ($X, W, b$).
* **Operasi Matriks**: Perkalian matriks ($A \cdot B$), perkalian Hadamard (*element-wise product* $\odot$), dan Transpose ($A^T$).
* **Dekomposisi & Sifat Matriks**: Dekomposisi QR (digunakan pada *Orthogonal Initialization* bobot rekuren $W_{hh}$) dan konsep nilai eigen (*eigenvalues*) untuk analisis *vanishing/exploding gradient*.

### 2. Kalkulus Multivariat (Multivariate Calculus)
* **Turunan Parsial & Gradient**: Menghitung arah dan laju perubahan error terhadap setiap parameter ($\frac{\partial L}{\partial W}$ dan $\frac{\partial L}{\partial b}$).
* **Aturan Rantai (Chain Rule)**: Aturan fundamental yang menggerakkan algoritma **Backpropagation** dan **Backpropagation Through Time (BPTT)** melintasi layer dan waktu.
* **Matriks Jacobian & Gradient Vectors**: Mengalirkan sinyal error mundur melalui fungsi non-linear.

### 3. Probabilitas & Teori Informasi (Probability & Information Theory)
* **Fungsi Aktivasi & Probabilitas**: Distribusi probabilitas kategorikal melalui fungsi **Softmax**.
* **Loss Functions**: Mengukur seberapa jauh estimasi model dari target sebenarnya menggunakan **Categorical Cross-Entropy** dan **Binary Cross-Entropy**.

### 4. Teori Optimasi (Optimization Theory)
* **Gradient Descent & Stochastic Gradient Descent (SGD)**: Algoritma pembaharuan parameter menuju titik minimum fungsi loss.
* **Adam Optimizer**: Menggabungkan konsep *Momentum* (estimasi momen pertama $m_t$) dan *RMSProp* (estimasi momen kedua $v_t$) dilengkapi *Bias Correction* untuk mempercepat konvergensi pelatihan.

---

## 🛠️ Library & Kebutuhan Lingkungan

Proyek ini sengaja dirancang dengan dependensi seminimal mungkin untuk memastikan kemudahan eksekusi dan portabilitas:

* **Python 3.8+**
* **NumPy**: Fondasi utama untuk seluruh komputasi aljabar linear, manipulasi array multidimensi, dan perhitungan matematik.
* **Pandas**: Digunakan untuk pemrosesan dan pembacaan dataset tabular dasar.
* **Matplotlib**: Digunakan untuk plotting kurva loss dan visualisasi hasil eksperimen.

### Instalasi Dependensi

```bash
pip install numpy pandas matplotlib
```

---

## 🗺️ Peta Pembelajaran & Struktur Repository

Repository ini disusun secara hierarkis mengikuti evolusi perkembangan arsitektur Deep Learning:

```text
deep-learning-from-scratch/
├── 1_ANN/          # Artificial Neural Network (Feedforward & Backpropagation)
├── 2_CNN/          # Convolutional Neural Network (Spatial Feature Extraction)
├── 3_RNN/          # Recurrent Neural Network (Sequential Processing)
├── 4_LSTM/         # Long Short-Term Memory (Gated Memory & CEC)
└── 5_TRANSFORMER/  # Transformer (Self-Attention & Positional Encoding)
```

| Modul | Arsitektur | Domain Utama | Fokus Pembelajaran Kunci |
| :--- | :--- | :--- | :--- |
| [1_ANN](file:///home/khaerul/Documents/github/deep-learning-from-scratch/1_ANN) | **Artificial Neural Network** | Tabular / Dasar | Feedforward, Backpropagation Chain Rule, ReLU/Sigmoid, Adam Optimizer. |
| [2_CNN](file:///home/khaerul/Documents/github/deep-learning-from-scratch/2_CNN) | **Convolutional Neural Network** | Gambar / Spasial | Kernel Convolution, Feature Maps, Max Pooling, Spatial Hierarchy. |
| [3_RNN](file:///home/khaerul/Documents/github/deep-learning-from-scratch/3_RNN) | **Recurrent Neural Network** | Sekuensial Pendek | Time Unrolling, Hidden State ($h_t$), Backpropagation Through Time (BPTT). |
| [4_LSTM](file:///home/khaerul/Documents/github/deep-learning-from-scratch/4_LSTM) | **Long Short-Term Memory** | Sekuensial Sedang | Cell State ($C_t$), Gating (*Forget, Input, Output*), Constant Error Carousel (CEC). |
| [5_TRANSFORMER](file:///home/khaerul/Documents/github/deep-learning-from-scratch/5_TRANSFORMER) | **Transformer** | Sekuensial / NLP / SOTA | **Self-Attention**, Multi-Head Attention, Sinusoidal Positional Encoding, LayerNorm. |

---

## 🚀 Cara Menjalankan Proyek

Setiap folder modul bersifat mandiri (*self-contained*) dan memiliki skrip `main.py` berstruktur Berorientasi Objek (OOP) serta dokumentasi detail pada folder `docs/`.

Untuk menjalankan salah satu modul (contoh: Transformer):

```bash
cd 5_TRANSFORMER
python main.py
```

---

## 📐 Filosofi Kode & Desain

1. **Transparansi di atas Abstraksi**: Tidak ada *hidden magic*. Setiap langkah komputasi *forward* dan *backward* ditulis secara eksplisit.
2. **OOP Style yang Clean & Modular**: Seluruh arsitektur dibungkus ke dalam kelas Python yang konsisten, mudah dibaca, dan modular.
3. **Visualisasi Interaktif**: Dilengkapi dengan progress bar pelatihan interaktif yang menampilkan *loss step-by-step* dan estimasi waktu selesai (ETA).

---

## 📝 Catatan Penulis

Proyek ini merupakan bukti perjalanan eksplorasi mandiri dalam menguasai AI dari dasar. Dengan memahami bagaimana matriks, perkalian dot, dan turunan kalkulus bekerja bersama pada level kode paling murni, kita mendapatkan pemahaman yang kokoh untuk merancang, mengoptimalkan, dan mengeksplorasi inovasi arsitektur AI di masa depan.
