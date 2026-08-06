# Artificial Neural Network From Scratch

Implementasi Artificial Neural Network (ANN) yang dibangun dari nol hanya menggunakan NumPy, dengan tujuan memahami bagaimana proses pembelajaran model terjadi melalui aliran data dan perhitungan gradient secara fundamental.

---

## Motivasi

Pada project ini, saya membangun **Artificial Neural Network (ANN)** dari nol tanpa menggunakan framework deep learning modern, melainkan hanya memanfaatkan NumPy sebagai dasar komputasi numerik.

Artificial Neural Network merupakan salah satu arsitektur paling fundamental dalam deep learning. Arsitektur ini menggambarkan bagaimana data mengalir di dalam model, mulai dari proses **feedforward** hingga **backpropagation** yang memungkinkan model untuk belajar melalui pembaruan parameter berbasis gradient. Bahkan arsitektur yang lebih kompleks seperti CNN, RNN, LSTM, hingga Transformer tetap dibangun di atas prinsip dasar neural network.

Oleh karena itu, memahami ANN secara fundamental menjadi langkah penting sebelum mempelajari arsitektur deep learning yang lebih tinggi.

Berdasarkan pengalaman saya dalam mempelajari machine learning dan deep learning, sering kali sulit untuk benar-benar memahami bagaimana sebuah model bekerja ketika langsung menggunakan framework populer seperti Scikit-learn, TensorFlow, atau PyTorch. Framework tersebut memang meningkatkan produktivitas, namun juga mengabstraksi hampir seluruh proses pelatihan model sehingga:

* aliran data di dalam model sulit diamati secara langsung,
* proses propagasi gradient tidak terlihat secara eksplisit,
* dan mekanisme bagaimana model belajar menjadi kurang intuitif.

Project ini dibuat untuk menghilangkan abstraksi tersebut, sehingga proses pembelajaran model dapat diamati secara jelas. Dengan mengimplementasikan semuanya secara manual, saya — dan juga pembaca yang sedang belajar — dapat memahami bagaimana data dan gradient bekerja bersama hingga sebuah model mampu belajar dari data.

---

## Tujuan Project

Tujuan utama dari project ini adalah:

* Mengimplementasikan neural network tanpa framework deep learning
* Memahami proses forward propagation secara komputasional dan matematis
* Mengimplementasikan backpropagation secara manual
* Memahami bagaimana gradient mengalir antar layer
* Menjembatani konsep matematika dengan implementasi kode nyata

---

## Cakupan Pembahasan

Repository ini membahas komponen inti dari Artificial Neural Network, meliputi:

* **Feedforward** — memahami bagaimana data mengalir dari input, melewati layer dan fungsi aktivasi hingga menghasilkan prediksi.
* **Backpropagation** — memahami bagaimana gradient mengalir dari layer output menuju layer sebelumnya serta bagaimana proses ini memungkinkan model untuk belajar.
* **Fondasi matematika** — setiap implementasi tidak hanya ditampilkan dalam bentuk kode, tetapi juga disertai penjelasan rumus dan intuisi matematisnya.

---

## Filosofi Desain

Project ini mengutamakan **transparansi dibanding abstraksi**.

Alih-alih berfokus pada performa atau penggunaan produksi, sistem dirancang agar menyerupai struktur matematis neural network secara langsung. Setiap komponen secara eksplisit memperlihatkan:

* proses forward computation
* perhitungan gradient
* pembaruan parameter

Tujuan utamanya adalah membuat mekanisme pembelajaran model dapat terlihat dan dipahami, bukan tersembunyi di balik API tingkat tinggi.

---

## Gambaran Arsitektur

![Screenshot App](./images/ann-architecture.png)
_Sumber: [GeeksforGeeks - Artificial Neural Networks](https://www.geeksforgeeks.org/deep-learning/artificial-neural-networks-and-its-applications/)_

Model mengikuti alur komputasi sederhana di mana proses forward menghasilkan prediksi, sedangkan proses backward menyebarkan error untuk memperbarui parameter model.

---

## Insight Pembelajaran

Melalui pembangunan project ini, beberapa pemahaman utama yang dapat diperoleh antara lain:

* Proses learning muncul dari komposisi gradient lokal antar layer
* Backpropagation merupakan implementasi nyata dari chain rule
* Pemilihan activation function mempengaruhi stabilitas training
* Persamaan matematika secara langsung diterjemahkan menjadi perilaku komputasi

---

## Dokumentasi

Penjelasan lebih detail tersedia pada folder `docs/`:

* `architecture.md` — desain sistem dan struktur komponen
* `math-foundation.md` — penjelasan matematis dan intuisi perhitungan
* `experiments.md` — observasi selama proses training

---

## Pengembangan Selanjutnya

Beberapa pengembangan yang dapat dilakukan di masa depan:

* Mini-batch training
* Optimizer lanjutan (Adam, RMSProp)
* Regularisasi model
* Optimasi performa
* Dukungan akselerasi GPU

---

## Catatan Penulis

Project ini merupakan bagian dari eksplorasi pribadi saya untuk memahami deep learning dari level fundamental — bukan hanya menggunakan model, tetapi memahami bagaimana dan mengapa model tersebut dapat belajar.
