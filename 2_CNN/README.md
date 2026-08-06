# Convolutional Neural Network From Scratch

Implementasi **Convolutional Neural Network (CNN)** yang dibangun dari nol menggunakan NumPy untuk memahami bagaimana model deep learning mendeteksi pola pada data gambar melalui operasi konvolusi dan ekstraksi fitur secara bertahap.

---

## Motivasi

Convolutional Neural Network (CNN) merupakan salah satu arsitektur deep learning yang secara khusus dirancang untuk memproses data berbentuk gambar.

Sebenarnya, jika kita memiliki data gambar, kita tetap bisa menggunakan **Artificial Neural Network (ANN)** dengan cara mengubah gambar menjadi kumpulan pixel dan menjadikannya sebagai feature di dalam dataset. Pendekatan ini dikenal sebagai proses **flattening**, di mana matriks gambar diubah menjadi sebuah vektor panjang.

Pendekatan tersebut tidak sepenuhnya salah. Namun, metode ini memiliki keterbatasan penting: ketika gambar diubah menjadi vektor, **struktur spasial antar pixel akan hilang**. Padahal dalam gambar, hubungan antar pixel yang berdekatan sering kali mengandung informasi penting untuk mendeteksi pola.

Di sinilah CNN menjadi lebih efektif.

CNN menggunakan mekanisme **convolution** yang bekerja dengan cara menggeser sebuah kernel (filter) di atas gambar menggunakan pendekatan **sliding window**. Melalui proses ini, CNN mampu mendeteksi pola lokal seperti:

* tepi (edges)
* tekstur
* bentuk sederhana
* hingga pola yang lebih kompleks

Proses ekstraksi fitur ini dilakukan secara bertahap melalui beberapa layer konvolusi sehingga model dapat membangun representasi fitur yang semakin abstrak.

Pada tahap akhir, hasil ekstraksi fitur tersebut biasanya tetap diproses menggunakan **layer fully connected (ANN)** untuk melakukan keputusan akhir atau prediksi.

Melalui project ini, saya ingin memahami bagaimana seluruh mekanisme tersebut bekerja secara fundamental dengan membangun CNN **tanpa menggunakan framework deep learning**.

---

## Tujuan Project

Tujuan utama dari project ini adalah:

* Mengimplementasikan Convolutional Neural Network dari nol menggunakan NumPy
* Memahami bagaimana operasi konvolusi bekerja pada data gambar
* Mengamati bagaimana kernel mengekstraksi fitur dari gambar
* Memahami bagaimana fitur visual berubah sepanjang layer CNN
* Menghubungkan konsep matematis CNN dengan implementasi kode nyata

---

## Cakupan Pembahasan

Repository ini membahas beberapa komponen utama dari Convolutional Neural Network, antara lain:

* **Convolution Layer** — memahami bagaimana kernel melakukan operasi konvolusi pada gambar
* **Feature Map** — melihat bagaimana pola pada gambar diekstraksi menjadi representasi fitur
* **Activation Function** — menambahkan non-linearitas pada feature map
* **Pooling Layer** — melakukan reduksi dimensi fitur sambil mempertahankan informasi penting
* **Fully Connected Layer** — mengubah hasil ekstraksi fitur menjadi prediksi akhir

Setiap bagian tidak hanya ditampilkan dalam bentuk implementasi kode, tetapi juga disertai dengan penjelasan konsep dan intuisi matematisnya.

---

## Filosofi Desain

Project ini dirancang dengan prinsip **transparansi dibanding abstraksi**.

Alih-alih berfokus pada performa atau optimisasi komputasi seperti framework deep learning modern, implementasi ini bertujuan untuk memperlihatkan secara jelas bagaimana setiap komponen CNN bekerja.

Setiap layer secara eksplisit memperlihatkan:

* proses convolution
* pembentukan feature map
* transformasi aktivasi
* aliran data menuju classifier

Dengan pendekatan ini, proses ekstraksi fitur yang biasanya tersembunyi di dalam library dapat dipahami secara lebih mendalam.

---

## Gambaran Arsitektur

Secara umum, arsitektur CNN yang dibangun pada project ini mengikuti alur berikut:

```text
Input Image
     ↓
Convolution Layer
     ↓
Activation Function
     ↓
Pooling Layer
     ↓
Flatten
     ↓
Fully Connected Layer
     ↓
Prediction
```

CNN bekerja dengan mengekstraksi fitur visual secara bertahap sebelum akhirnya membuat keputusan menggunakan layer klasifikasi.

---

## Insight Pembelajaran

Melalui pembangunan CNN dari nol, beberapa pemahaman penting yang dapat diperoleh antara lain:

* CNN mempertahankan struktur spasial gambar melalui operasi konvolusi
* Kernel kecil mampu mendeteksi pola lokal pada gambar
* Feature map merupakan representasi pola yang ditemukan model
* Layer yang lebih dalam menghasilkan fitur yang lebih kompleks
* CNN biasanya tetap menggunakan layer ANN pada tahap klasifikasi

---

## Dokumentasi

Penjelasan lebih lengkap tersedia pada folder `docs/`:

* `math-foundation.md` — fondasi matematika operasi konvolusi
* `architecture.md` — desain layer CNN dan alur komputasi
* `experiments.md` — eksperimen training dan observasi model

---

## Pengembangan Selanjutnya

Beberapa pengembangan yang dapat dilakukan pada project ini:

* Implementasi padding dan stride
* Multi-channel convolution
* Multiple convolution filters
* Optimizer lanjutan
* Training menggunakan dataset gambar yang lebih kompleks

---

## Catatan Penulis

Project ini merupakan bagian dari eksplorasi saya untuk memahami deep learning dari level fundamental — tidak hanya menggunakan model yang sudah tersedia, tetapi juga memahami bagaimana mekanisme ekstraksi fitur pada gambar benar-benar bekerja di dalam arsitektur Convolutional Neural Network.
