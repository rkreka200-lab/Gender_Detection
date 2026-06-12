# Gender Detection (Male/Female) menggunakan YOLO11n

## Deskripsi Proyek

Proyek ini bertujuan untuk mendeteksi gender manusia ke dalam dua
kategori: - Male - Female

Model yang digunakan adalah **YOLO11n (YOLO v11 Nano)** yang dilatih
untuk melakukan klasifikasi gender berdasarkan dataset yang telah diberi
label.

## Tujuan

-   Mengklasifikasikan gender ke dalam kategori Male atau Female.
-   Menerapkan model YOLO11n pada tugas klasifikasi.
-   Mempelajari proses pelatihan, validasi, dan pengujian model AI.

## Dataset

Dataset terdiri dari dua kelas: 1. Male 2. Female

Data diproses dan diberi label sebelum digunakan pada tahap pelatihan
model.

## Arsitektur Model

Model yang digunakan adalah **YOLO11n (Nano)** karena memiliki ukuran
ringan dan cocok untuk perangkat dengan sumber daya terbatas.

Keunggulan YOLO11n: - Ukuran model kecil. - Proses inferensi cepat. -
Cocok untuk implementasi real-time. - Tetap memiliki performa yang baik
untuk berbagai tugas computer vision.

## Tahapan Pengerjaan

1.  Pengumpulan dataset.
2.  Pelabelan data (Male dan Female).
3.  Preprocessing dataset.
4.  Pembagian data training, validation, dan testing.
5.  Pelatihan model YOLO11n.
6.  Evaluasi model.
7.  Pengujian pada data baru.

## Hasil

Model berhasil dilatih untuk membedakan dua kategori gender: - Male -
Female

> Catatan: Nilai akurasi akhir tidak dicantumkan karena data hasil
> evaluasi tidak tersedia.

## Kebutuhan Sistem

-   Python 3.x
-   Ultralytics YOLO
-   OpenCV
-   NumPy

## Kesimpulan

YOLO11n dapat digunakan sebagai model ringan untuk mendeteksi gender
dengan dua kelas (Male dan Female). Performa akhir model bergantung pada
kualitas dataset, jumlah data pelatihan, serta parameter yang digunakan
selama proses training.
