# 📌 Spelling Correction Project

### 🚀 Dikembangkan oleh: Ganang Setyo Hadi (2208107010052)

---

## 📝 Deskripsi
Proyek ini bertujuan untuk mengembangkan sistem **koreksi ejaan otomatis** berbasis NLP (*Natural Language Processing*). Berbagai pendekatan telah dicoba, termasuk **Levenshtein Distance**, **model berbasis konteks**, serta **model hybrid yang menggabungkan metode statistik dan machine learning**. Model ini dievaluasi menggunakan dataset kesalahan ejaan yang umum ditemukan dalam teks berbahasa Inggris.

---

## 📂 Struktur Folder
```
.
├── data                        # Dataset yang digunakan dalam eksperimen
│   ├── aspell.txt
│   ├── big.txt
│   ├── birkbeck.txt
│   ├── spell-testset1.txt
│   ├── spell-testset2.txt
│   └── wikipedia.txt
├── lavenshtein_distance_matrix.py # Implementasi algoritma Levenshtein Distance dan ML
├── result                      # Hasil evaluasi
│   ├── hasil_evaluasi_20250316_171015.txt
│   ├── hasil_evaluasi_20250318_174338.txt
│   ├── hasil_evaluasi_20250318_183946.txt
│   ├── hasil_evaluasi_20250318_202453.txt
│   └── hasil_evaluasi_20250318_212156.txt
└── Tugas01.pdf                  # Laporan akhir proyek
└── README.md                 # Dokumentasi
```

---

## 🛠️ Teknologi yang Digunakan
✅ **Python** 🐍  
✅ **Levenshtein Distance** 📏  
✅ **Machine Learning (Random Forest Classifier)** 🤖  
✅ **Bigram Contextual Correction** 🔍  
✅ **Dataset Spelling Correction dari Kaggle** 📊  

---

## 📊 Hasil Evaluasi
Pendekatan yang telah diuji:

| Model                         | Akurasi | Precision | Recall | F1-Score |
|--------------------------------|---------|----------|--------|----------|
| Levenshtein Distance          | 74.81%  | -        | -      | -        |
| Koreksi Berbasis Konteks      | 74.44%  | -        | -      | -        |
| **Model Hybrid (Final)**      | **77.91%**  | **77.66%**  | **81.47%** | **60.01%** |

Hasil terbaik diperoleh dengan **Model Hybrid**, yang menggabungkan pendekatan statistik dan machine learning untuk meningkatkan akurasi koreksi ejaan.

---

## 📌 Catatan untuk Prof. Taufik
**"Saya sudah mencoba, Prof. 😔"**  
Saya telah mengembangkan model berbasis **RNN dan Transformer**, termasuk mencoba **Neuspell** dan **T5 Pretrained Model**, namun hasilnya jauh dari harapan. Akurasi yang diperoleh bahkan lebih rendah dibandingkan metode yang lebih sederhana. Setelah berkali-kali eksperimen, saya dengan berat hati **memutuskan untuk menyerah** pada pendekatan deep learning. Oleh karena itu, saya memilih metode yang lebih sederhana namun efektif.

Semoga ini bisa menjelaskan keputusan yang saya ambil dalam proyek ini. 🙏

---

## 🎯 Kesimpulan
✅ **Model hybrid terbukti lebih unggul** dibandingkan metode lain dalam proyek ini.  
✅ **Levenshtein Distance tetap menjadi metode dasar** yang cepat dan efektif untuk koreksi ejaan sederhana.  
✅ **Deep Learning bukan solusi terbaik dalam kasus ini** (meskipun sudah dicoba).  

Terima kasih telah membaca! Jika ada pertanyaan, jangan ragu untuk menghubungi saya. 🚀

---

## ❤️ Made with Passion & Coffee ☕
Dibuat dengan penuh perjuangan, eksperimen yang tak terhitung jumlahnya, dan secangkir kopi yang selalu menemani.  
Jika proyek ini membantu atau menginspirasi Anda, jangan lupa ⭐ repository ini!  

✨ *"Perjalanan itu lebih penting dari hasilnya."* ✨
