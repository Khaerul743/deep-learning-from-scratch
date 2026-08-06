# Recurrent Neural Network From Scratch

Implementasi Recurrent Neural Network (RNN) yang dibangun dari nol (*from scratch*) hanya menggunakan NumPy, dengan tujuan memahami bagaimana pemrosesan data sekuensial dan propagasi gradient melintasi waktu (*Backpropagation Through Time*) bekerja secara fundamental.

---

## Motivasi

Setelah mengimplementasikan **Artificial Neural Network (ANN)** untuk data statis, langkah krusial berikutnya dalam deep learning adalah memahami bagaimana model menangani **data sekuensial** (seperti teks, kalimat, atau data deret waktu).

**Recurrent Neural Network (RNN)** merupakan arsitektur fundamental yang memperkenalkan konsep **memori internal** melalui *hidden state* ($h_t$). Memori ini memungkinkan jaringan untuk mengingat informasi dari langkah waktu sebelumnya dan memanfaatkannya untuk memprediksi elemen berikutnya.

Meskipun framework populer seperti PyTorch atau TensorFlow menyediakan layer `nn.RNN` atau `nn.LSTM` siap pakai, abstraksi tersebut menyembunyikan:
* bagaimana hidden state diperbarui secara step-by-step,
* bagaimana perhitungan *Backpropagation Through Time (BPTT)* mengumpulkan gradient melintasi urutan waktu,
* dan mengapa fenomena *vanishing / exploding gradient* terjadi pada sekuens yang panjang.

Project ini dibuat untuk menghilangkan abstraksi tersebut, menyajikan implementasi RNN yang transparan dan mudah dipahami.

---

## Tujuan Project

Tujuan utama dari project ini adalah:
* Mengimplementasikan Vanilla RNN dari nol tanpa framework deep learning.
* Memahami mekanisme *time unrolling* pada data sekuensial.
* Mengimplementasikan algoritma *Backpropagation Through Time (BPTT)* secara eksplisit.
* Mengimplementasikan inisialisasi bobot khusus (Xavier & Orthogonal Init) untuk stabilitas pelatihan.
* Menjembatani intuisi matematis pengolahan sekuens dengan kode Python & NumPy nyata.

---

## Cakupan Pembahasan

Repository ini membahas komponen inti dari Recurrent Neural Network:

* **Forward Pass Sekuensial** — memproses sekuens kata step-by-step dan memperbarui hidden state $h_t$ menggunakan aktivasi $\tanh$.
* **Backpropagation Through Time (BPTT)** — menghitung dan mengumpulkan gradient dari masa depan ke masa lalu ($t = T \dots 1$).
* **Optimasi Paramater** — pembaruan parameter bobot ($W_{xh}, W_{hh}, W_{hy}, b_h, b_y$) memanfaatkan optimizer **Adam** atau **SGD**.
* **Autoregressive Text Generation** — menghasilkan prediksi kata berikutnya berbasis distribusi probabilitas Softmax (Greedy Search).

---

## Filosofi Desain

Project ini mengutamakan **transparansi komputasi dan struktur OOP yang rapi**.

Kode ditulis dengan pendekatan berorientasi objek (*Object-Oriented Programming*) dalam file [main.py](file:///home/khaerul/Documents/github/deep-learning-from-scratch/RNN/main.py) yang mencakup:
* Kelas `RecurrentNeuralNetwork` yang modular.
* Visualisasi progress pelatihan interaktif lengkap dengan estimasi waktu (ETA) dan nilai loss.
* Pemisahan eksplisit antara peritungan *forward step*, *sequence unrolling*, *BPTT backward*, dan *text generation*.

---

## Gambaran Arsitektur

```text
Input Step t          x_t
                       │
                       ▼
Hidden State     h_{t-1} ──> [ Wxh·x_t + Whh·h_{t-1} + bh ] ──> tanh ──> h_t
                                                                         │
                                                                         ▼
Output Prediction                                                 y_t = Softmax(Why·h_t + by)
```

Proses *unrolling* mengulang sel komputasi di atas sepanjang urutan kata ($t = 1 \dots T$).

---

## Insight Pembelajaran

Melalui pembangunan project ini, beberapa pemahaman utama yang diperoleh antara lain:

* **Internal State sebagai Memori**: Hidden state $h_t$ bertindak sebagai vektor kompresi memori konteks masa lalu.
* **Akumulasi Gradient BPTT**: Gradient terhadap bobot terbagi $W_{hh}$ dan $W_{xh}$ merupakan jumlah akumulasi dari seluruh langkah waktu dalam sekuens.
* **Peran Matriks Ortogonal**: Inisialisasi $W_{hh}$ menggunakan matriks ortogonal membantu menjaga norma gradient tetap stabil saat dipropagasikan mundur melalui banyak langkah waktu.
* **Autoregressive Generation**: Prediksi teks dilakukan secara berantai di mana output pada step $t$ menjadi masukan bagi step $t+1$.

---

## Dokumentasi

Penjelasan matematika lengkap dan teknis detail tersedia pada folder `docs/`:

* [docs/architecture.md](file:///home/khaerul/Documents/github/deep-learning-from-scratch/RNN/docs/architecture.md) — Penjelasan detail rumus forward pass, BPTT, inisialisasi bobot, dan tantangan vanishing gradient.

---

## Cara Menjalankan

Untuk menjalankan proses training dan pengujian prediksi teks:

```bash
python main.py
```

---

## Pengembangan Selanjutnya

* Implementasi **Long Short-Term Memory (LSTM)** dan **Gated Recurrent Unit (GRU)** untuk mengatasi vanishing gradient.
* Implementasi **Gradient Clipping** untuk menangani exploding gradient.
* Penggunaan **Word Embeddings** (Word2Vec / FastText) dibanding one-hot vectors.
* Implementasi **Sampling Techniques** (Temperature Sampling / Top-k / Top-p) saat generasi teks.

---

## Catatan Penulis

Project ini merupakan kelanjutan dari eksplorasi mandiri dalam memahami deep learning secara fundamental — dari ANN hingga RNN dan arsitektur sekuensial yang lebih kompleks.
