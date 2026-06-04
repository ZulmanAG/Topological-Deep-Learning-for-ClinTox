# 🧪 Topological Deep Learning for ClinTox: FDA Drug Toxicity Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![PyTorch Lightning](https://img.shields.io/badge/PyTorch_Lightning-2.0%2B-792ee5)

Repositori ini mengimplementasikan **Graph Attention Networks (GATv2)** untuk memprediksi toksisitas kandidat obat berdasarkan dataset klinis (ClinTox) dari Therapeutics Data Commons. 

Proyek ini mengatasi kelemahan representasi molekul 1D kaku (*Morgan Fingerprints*) dengan merepresentasikan molekul sebagai **Grafik Topologi 2D**, memungkinkan model untuk secara adaptif mempelajari interaksi antar-atom yang menentukan sifat toksik suatu senyawa.

---

## ✨ Fitur Utama
- **Pendekatan Topologi Murni:** Mengubah string SMILES menjadi *Rich Graph* (Node = Atom, Edge = Ikatan Kimia).
- **Arsitektur Super Ringan:** Hanya menggunakan **~110 Ribu Parameter** (90% lebih efisien dibanding Baseline MLP).
- **Sistem *Hybrid* & Rem Otomatis:** Mengunci otomatis pada skor `val_auroc` tertinggi untuk mencegah *overfitting*.

---

## 🏆 Performa Model (Hasil Ujian *Scaffold Split*)

| Model | Representasi Data | Jumlah Parameter | Test ROC-AUC | Status Model |
| :--- | :--- | :--- | :--- | :--- |
| Baseline MLP | 1D Morgan Fingerprints | 1.20 M | 0.7480 | Rentan *Overfitting* |
| **GATv2 (Ours)** | **2D Molecular Graph** | **110 K** | **0.8014** | **Stabil / Cerdas Universal** |

---

## 📂 Struktur Repositori Modular
```text
Topological-Deep-Learning-for-ClinTox/
├── configs/
│   └── default_config.yaml
├── src/
│   └── data/
│       └── datamodule.py
├── train.py
└── requirements.txt

```

 🚀 Panduan Penggunaan
1. Memulai Latihan (Training)
Seluruh hyperparameter bisa Anda atur di configs/default_config.yaml. Untuk menjalankan orkestrator training dengan sistem Hybrid Rem Otomatis:
```
python train.py
```
Model akan berlatih dan berhenti secara otomatis ketika performa generalisasi menyentuh titik tertingginya (terkunci otomatis pada skor terbaik).

Developed by ZulmanAG
