# Transformer Architecture From Scratch

Implementasi arsitektur **Transformer** yang dibangun dari nol (*from scratch*) hanya menggunakan NumPy, sebagai puncak akhir dari seri proyek *Deep Learning From Scratch*.

---

## Motivasi & Tujuan Project

Arsitektur **Transformer** yang diperkenalkan oleh Vaswani et al. (2017) dalam makalah monumental *"Attention Is All You Need"* telah merevolusi seluruh lanskap kecerdasan buatan modern (Natural Language Processing, Computer Vision, Multimodal AI, dan Large Language Models seperti GPT, Claude, BERT, serta Gemini).

Sebelum Transformer diciptakan, pemrosesan bahasa bergantung pada arsitektur rekuren seperti RNN dan LSTM yang memiliki keterbatasan mendasar: **pemrosesan sekuensial langkah demi langkah ($O(T)$) yang tidak dapat diparalelkan di GPU**.

Tujuan utama dari proyek ini adalah:
* Mengimplementasikan arsitektur Transformer secara penuh dari nol menggunakan NumPy tanpa bantuan framework deep learning (PyTorch/TensorFlow).
* Memahami mekanisme **Self-Attention** dan **Multi-Head Attention** secara eksplisit.
* Mengimplementasikan **Sinusoidal Positional Encoding** untuk mempertahankan posisi urutan kata tanpa jaringan rekuren.
* Mengimplementasikan **Layer Normalization** dan **Position-wise Feed-Forward Network (FFN)**.
* Menyelesaikan peta evolusi arsitektur deep learning dari fundamental (**ANN → CNN → RNN → LSTM → Transformer**).

---

## Evolusi Arsitektur: Mengapa Transformer Adalah Puncak Pemrosesan Sekuens

Melalui perjalanan seri *Deep Learning From Scratch*, kita telah mempelajari keterbatasan dan evolusi setiap arsitektur:

1. **Artificial Neural Network (ANN)** [1_ANN](file:///home/khaerul/Documents/github/deep-learning-from-scratch/1_ANN):
   - Unggul untuk data berstruktur tabular.
   - Tidak memiliki kesadaran spasial (gambar) maupun temporal (sekuens seiring waktu).

2. **Convolutional Neural Network (CNN)** [2_CNN](file:///home/khaerul/Documents/github/deep-learning-from-scratch/2_CNN):
   - Menggunakan operasi konvolusi (*kernel/filter*) untuk mengekstraksi hierarki fitur spasial pada gambar.
   - Kurang efisien dalam menangani konteks urutan teks jangka panjang.

3. **Recurrent Neural Network (RNN)** [3_RNN](file:///home/khaerul/Documents/github/deep-learning-from-scratch/3_RNN):
   - Memperkenalkan memori internal (*hidden state* $h_t$) untuk data sekuensial.
   - **Kekurangan**: Mengalami *vanishing/exploding gradient* saat sekuens panjang.

4. **Long Short-Term Memory (LSTM)** [4_LSTM](file:///home/khaerul/Documents/github/deep-learning-from-scratch/4_LSTM):
   - Menyelamatkan RNN dengan memperkenalkan *Cell State* ($C_t$) dan 3 gerbang (*Forget, Input, Output*) via *Constant Error Carousel (CEC)*.
   - **Kekurangan**: Pemrosesan tetap wajib dilakukan berurutan step-by-step ($O(T)$ *time bottleneck*), sehingga pelatihan tidak bisa diparalelkan di GPU sepanjang waktu.

5. **Transformer** (Modul Ini):
   - **Menghilangkan Recurrence**: Menggantikan loop rekuren dengan *Self-Attention*.
   - **Paralelisasi Penuh**: Seluruh sekuens kata diproses secara bersamaan di GPU.
   - **Konteks Global Direct ($O(1)$)**: Setiap token dapat berhubungan langsung dengan token lain dalam sekuens tanpa peduli seberapa jauh jaraknya.

---

## Gambaran Arsitektur Transformer

Arsitektur Transformer terdiri dari penumpukan blok perhati (*Attention Blocks*), koneksi residual (*Residual Connections*), *Layer Normalization*, dan *Feed-Forward Networks*:

### 1. Arsitektur Keseluruhan Model Transformer
![Arsitektur Transformer](./images/transformer.png)

---

### 2. Mekanisme Multi-Head Attention & Scaled Dot-Product
![Multi-Head Attention](./images/multihead-attention.png)

---

## Ringkasan Perbandingan Seri Deep Learning

| Arsitektur | Domain Utama | Fitur Kunci | Keterbatasan Utama |
| :--- | :--- | :--- | :--- |
| **ANN** | Tabular / Classification | Dense Linear Transformation & Activations | Tanpa konteks spasial / sekuensial |
| **CNN** | Image / Grid Data | Convolution Kernels & Pooling | Pemrosesan sekuensial tidak alami |
| **RNN** | Short Sequential Data | Hidden State Recurrence ($h_t$) | Vanishing / Exploding Gradient |
| **LSTM** | Medium Sequential Data | Cell State ($C_t$) & Gating Mechanisms | Sequential Bottleneck ($O(T)$ no GPU parallel) |
| **Transformer** | Text / Vision / Multimodal | **Self-Attention & Positional Encoding** | Kompleksitas memori perhatian $O(T^2)$ |

---

## Cakupan Pembahasan & Struktur Kode

File utama [main.py](file:///home/khaerul/Documents/github/deep-learning-from-scratch/5_TRANSFORMER/main.py) mengimplementasikan kelas `Transformer` yang mencakup:

* **Positional Encoding**: Penambahan sinyal frekuensi sinus dan kosinus pada input embedding.
* **Masked Multi-Head Attention**: Proyeksi matriks Query ($Q$), Key ($K$), dan Value ($V$), *Look-Ahead Masking*, serta penggabungan multi-head.
* **Add & Layer Normalization**: Normalisasi fitur per-token berbasis mean & variance serta parameter $\gamma$ dan $\beta$.
* **Position-wise FFN**: Jaringan feed-forward 2 layer berbasis aktivasi ReLU.
* **Backpropagation Engine**: Turunan kalkulus eksplisit untuk seluruh matriks atensi, layer norm, dan FFN.
* **Training & Inference**: Visualisasi progress bar pelatihan dan fungsi prediksi token berikutnya.

---

## Dokumentasi

Penjelasan matematika lengkap dan penurunan rumus turunan tersedia pada folder `docs/`:

* [docs/architecture.md](file:///home/khaerul/Documents/github/deep-learning-from-scratch/5_TRANSFORMER/docs/architecture.md) — Penjelasan detail rumus Self-Attention, Positional Encoding, Layer Normalization, FFN, dan derivasi BPTT / Backpropagation.

---

## Cara Menjalankan

Untuk menjalankan proses training dan pengujian prediksi teks Transformer:

```bash
python main.py
```

---

## Catatan Penulis

Proyek ini menandai penyelesaian eksplorasi penuh **Deep Learning From Scratch** — dari prinsip paling dasar neural network hingga arsitektur state-of-the-art Transformer. Membangun setiap komponen secara manual tanpa bantuan framework memberikan intuisi yang mendalam tentang bagaimana data, matriks, dan gradient bekerja bersama menghasilkan kecerdasan buatan.
