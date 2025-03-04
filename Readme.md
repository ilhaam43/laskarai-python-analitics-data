# Dicoding Bike Sharing Analysis Dashboard 🚲

Dashboard analisis data untuk dataset bike sharing dari Capital Bikeshare Washington D.C., menampilkan analisis pengaruh cuaca dan jenis hari terhadap pola penyewaan sepeda.

## Deskripsi Proyek

Dashboard ini menganalisis data penggunaan bike sharing selama periode 2011-2012, dengan fokus pada:
- Pengaruh kondisi cuaca terhadap jumlah penyewaan
- Perbedaan pola penyewaan antara hari kerja dan hari libur
- Analisis segmentasi pengguna casual vs registered

## Setup Environment - Anaconda

```
conda create --name bike-sharing python=3.9
conda activate bike-sharing
pip install -r requirements.txt
```

## Setup Environment - Shell/Terminal

```
mkdir bike_sharing_analysis
cd bike_sharing_analysis
python -m venv venv
source venv/bin/activate  # Untuk Linux/Mac
venv\Scripts\activate     # Untuk Windows
pip install -r requirements.txt
```

## Run Streamlit App

```
streamlit run app.py
```

## Struktur Proyek

```
bike_sharing_analysis/
├── app.py                # Aplikasi Streamlit
├── day.csv               # Dataset harian
├── hour.csv              # Dataset per jam
├── requirements.txt      # Daftar library yang dibutuhkan
└── README.md             # Dokumentasi proyek
```

## Requirements

File `requirements.txt` berisi:

```
streamlit>=1.28.0
pandas>=2.1.0
numpy>=1.26.0
matplotlib>=3.8.0
seaborn>=0.13.0
plotly>=5.18.0
```

## Sumber Data

Dataset berasal dari:
- Capital Bikeshare: http://capitalbikeshare.com/system-data
- Periode: 2011-2012
- Lokasi: Washington D.C., USA

## Creator

[Ilhaam Akmal Abdjul] - [akmalilhaam@gmail.com]