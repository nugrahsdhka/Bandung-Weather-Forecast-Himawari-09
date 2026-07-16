# 🌩️ Bandung Weather Forecast — Himawari-9 CTT Forecasting

Sistem **end-to-end** untuk memprediksi **Cloud Top Temperature (CTT)** — suhu puncak awan, indikator kunci potensi hujan/badai — di wilayah **Kota Bandung**, menggunakan data satelit cuaca geostasioner **Himawari-9** (kanal inframerah, format NetCDF) dan berbagai model *machine learning* (SVR, XGBoost, LightGBM, CatBoost).

Pipeline ini mencakup seluruh siklus: **unduh data satelit → eksplorasi → feature engineering → training model → evaluasi multi-step → inference forecast 3 jam → visualisasi peta 6-panel & animasi GIF.**

---

## 📋 Daftar Isi

- [Konsep Dasar](#-konsep-dasar)
- [Alur Kerja (Pipeline)](#-alur-kerja-pipeline)
- [Struktur Folder](#-struktur-folder)
- [Detail Tiap Tahap](#-detail-tiap-tahap)
- [Instalasi](#-instalasi)
- [Konfigurasi (.env)](#-konfigurasi-env)
- [Cara Menjalankan](#-cara-menjalankan)
- [Output yang Dihasilkan](#-output-yang-dihasilkan)
- [Catatan Teknis](#-catatan-teknis)

---

## 🔑 Konsep Dasar

**Cloud Top Temperature (CTT)** diambil dari kanal inframerah **tbb_13** (*Temperature of Brightness/Blackbody*, kanal ±10.4 µm) satelit Himawari-9. Semakin dingin suhu puncak awan, semakin tinggi awan tersebut menjulang — biasanya berkorelasi dengan awan konvektif (Cumulonimbus) yang berpotensi menimbulkan hujan lebat.

Proyek ini memakai pendekatan **autoregressive per-pixel**: setiap piksel grid di atas Bandung diperlakukan sebagai satu sampel time-series, diprediksi menggunakan nilai historisnya sendiri (lag) ditambah fitur waktu siklikal (jam, hari-dalam-tahun) dan posisi geografis (lat/lon).

Area yang di-subset dari citra full-disk Himawari:

| Batas | Nilai |
|---|---|
| Latitude | -7.0° s/d -6.8° |
| Longitude | 107.5° s/d 107.8° |
| Grid | ±5 baris × 7 kolom piksel |
| Kanal dipakai | `tbb_07` s/d `tbb_16` (10 kanal inframerah) |
| Variabel target | `tbb_13` |

---

## 🔄 Alur Kerja (Pipeline)

```
┌──────────────────────┐
│ 1. Download Data     │  scripts/01_download_data.py
│    (FTP → NetCDF)    │  → subset area Bandung, validasi, resume otomatis
└──────────┬────────────┘
           ▼
┌──────────────────────┐
│ 2. Eksplorasi Data   │  scripts/02_explore_data.py
│    (cek cakupan/gap) │  → laporan cakupan waktu, gap, konsistensi grid
└──────────┬────────────┘
           ▼
┌──────────────────────┐
│ 3a. Feature Eng.     │  scripts/03a_build_features.py
│     (per pixel)      │  → dataset supervised (10/30/60 menit)
└──────────┬────────────┘
           ▼
┌──────────────────────┐
│ 3b. Ekstraksi AR     │  scripts/03b_extract_autoregressive.py
│     (tbb_13 only)    │  → dataset ramping khusus autoregressive
└──────────┬────────────┘
           ▼
┌──────────────────────┐
│ 4. Training Model    │  scripts/04_train_models.py
│    (4 algoritma ML)  │  → SVR, XGBoost, LightGBM, CatBoost per interval
└──────────┬────────────┘
           ▼
┌──────────────────────┐
│ 5. Evaluasi Recursive│  scripts/05_recursive_evaluation.py
│    (horizon 3 jam)   │  → pilih model terbaik berdasar MAE multi-step
└──────────┬────────────┘
           ▼
┌──────────────────────┐
│ 6. Inference         │  scripts/06_run_inference.py
│    (forecast nyata)  │  → forecast recursive 18 langkah x 10 menit,
│                       │    otomatis pakai timestamp terbaru (atau manual)
└──────────┬────────────┘
           ▼
┌──────────────────────┐
│ 7. Visualisasi       │  scripts/07_test_visualization.py
│    (1 frame uji)     │  → render figure 6-panel (cek styling),
│                       │    otomatis ikut run forecast terakhir
└──────────┬────────────┘
           ▼
┌──────────────────────┐
│ 8. Animasi Penuh     │  scripts/08_render_animation.py
│    (GIF 10/30/60 min)│  → animasi GIF untuk tiap interval tampilan,
│                       │    otomatis ikut run forecast terakhir
└──────────────────────┘
```

---

## 📁 Struktur Folder

```
Bandung-Weather-Forecast-Himawari-09/
├── .env.example                    # Template kredensial FTP & Telegram
├── .gitignore
└── scripts/
    ├── 01_download_data.py         # Tahap 1: download + subset FTP
    ├── 02_explore_data.py          # Tahap 2: eksplorasi cakupan data
    ├── 03a_build_features.py       # Tahap 3a: feature engineering
    ├── 03b_extract_autoregressive.py # Tahap 3b: ekstraksi fitur AR
    ├── 04_train_models.py          # Tahap 4: training 4 model ML
    ├── 05_recursive_evaluation.py  # Tahap 5: evaluasi multi-step
    ├── 06_run_inference.py         # Tahap 6: forecast recursive nyata (otomatis/manual)
    ├── 07_test_visualization.py    # Tahap 7: uji render 1 frame (otomatis/manual)
    ├── 08_render_animation.py      # Tahap 8: render animasi GIF penuh (otomatis/manual)
    ├── geojson/
    │   └── KotaBandung.geojson     # Batas administratif kecamatan Bandung
    ├── pipeline/                   # Modul inti (reusable logic)
    │   ├── config.py               # Konfigurasi terpusat (.env, path, area)
    │   ├── ftp_client.py           # Klien FTP (koneksi, retry, download)
    │   ├── netcdf_tools.py         # Subset & validasi NetCDF
    │   ├── file_tracker.py         # Pelacak file yang sudah diproses
    │   ├── feature_engineering.py  # Pembangun dataset supervised per-pixel
    │   ├── model_training.py       # Trainer 4 algoritma ML + evaluasi
    │   ├── recursive_eval.py       # Simulasi forecast recursive multi-step
    │   ├── inference.py            # Pemilihan model terbaik & forecast batch
    │   ├── visualization.py        # Render peta 6-panel (dark theme)
    │   ├── telegram_notifier.py    # Notifikasi progres via Telegram
    │   └── utils.py                # Utilitas umum (path, format ukuran, dll)
    ├── tools/
    │   └── check_csv_features_eng_result.py  # Alat bantu QA dataset CSV
    └── ui/
        └── terminal_display.py     # Output terminal (banner, progress bar, dll)
```

> Folder `data_bandung/`, `dataset/`, `models/`, `forecast_output/`, `visualizations/`, dan `_metadata/` **tidak disertakan di repo** (masuk `.gitignore`) karena berisi data mentah/hasil proses — akan otomatis dibuat saat pipeline dijalankan.

---

## 🧩 Detail Tiap Tahap

### Tahap 1 — Download Data (`01_download_data.py` + `pipeline/ftp_client.py`, `netcdf_tools.py`)

- Terhubung ke server **FTP** (mis. JAXA P-Tree, `ftp.ptree.jaxa.jp`) menggunakan kredensial dari `.env`.
- Menelusuri direktori FTP **secara rekursif**, hanya mengambil file yang mengandung pola nama tertentu (default: `R21_FLDK.02801_02401`, kanal full-disk resolusi 2 km).
- Setiap file NetCDF diunduh ke file sementara, divalidasi ukurannya (anti-korup), lalu **di-subset** hanya untuk area Bandung menggunakan `xarray` (`ds.where(mask_lat & mask_lon, drop=True)`).
- Kalau area Bandung tidak ada datanya di file tersebut → dilewati (tidak disimpan), tapi tetap dicatat sebagai "sudah diperiksa" agar tidak diunduh ulang di run berikutnya (**resume otomatis** via `FileTracker`).
- Retry otomatis hingga `MAX_RETRY=3` kali dengan reconnect FTP jika gagal.
- Notifikasi real-time ke **Telegram** (opsional) tiap direktori selesai diproses atau saat terjadi error file.
- Logging lengkap ke `download_activity.log`.
- **Satu-satunya tahap yang benar-benar incremental** — file yang sudah pernah diperiksa tidak diproses ulang. Tahap 2 s/d 5 selalu memproses **ulang seluruh isi folder data** setiap dijalankan (lihat catatan di bagian [Catatan Teknis](#-catatan-teknis)).

### Tahap 2 — Eksplorasi Data (`02_explore_data.py`)

- Memindai seluruh file `.nc` hasil Tahap 1, mengekstrak timestamp dari nama file (pola regex `NC_H\d+_(\d{8})_(\d{4})_`).
- Melaporkan: rentang waktu data, jumlah hari unik, serta **gap** (slot waktu 10-menit yang hilang) per hari — penting karena Himawari mengirim citra tiap 10 menit dan data yang bolong akan memengaruhi kualitas fitur lag.
- Mengecek konsistensi struktur/dimensi antar file.

### Tahap 3a — Feature Engineering (`03a_build_features.py` + `pipeline/feature_engineering.py`)

Mengubah rangkaian frame CTT (time-series citra) menjadi **dataset supervised per-piksel**:

1. **Preload** seluruh file `.nc` ke memori (10 kanal `tbb_07`–`tbb_16`), validasi grid lat/lon konsisten antar file.
2. Untuk tiap **interval prediksi** (10, 30, 60 menit) dan tiap **titik waktu dasar `t`** yang selaras dengan interval tersebut:
   - Ambil **3 frame lag** (`t`, `t-1·interval`, `t-2·interval`) — jika salah satu tidak lengkap, baris dilewati.
   - Ambil **frame target** (`t + interval`).
   - Hitung **fitur waktu siklikal** (encoding sin/cos) agar model memahami sifat periodik jam dan musim tanpa diskontinuitas:
     - `hour_sin/cos` dari jam-dalam-hari (periode 24 jam)
     - `doy_sin/cos` dari hari-dalam-tahun (periode 365.25 hari)
   - Untuk **setiap piksel** di grid (5×7), buat satu baris berisi: koordinat piksel, lat/lon, fitur waktu, seluruh 10 kanal pada 3 titik lag, dan nilai target `tbb_13`.
3. Simpan sebagai `dataset/features_{interval}min.csv` untuk masing-masing interval.

> ⚠️ Tahap ini melakukan **full rebuild** dari nol setiap dijalankan (bukan menambah baris ke file lama) — dia scan ulang seluruh folder data (`.nc`) yang ada dan menimpa CSV lama. Kalau ada data `.nc` baru, jalankan ulang tahap ini (dan seterusnya) supaya ikut terangkum.

### Tahap 3b — Ekstraksi Fitur Autoregressive (`03b_extract_autoregressive.py`)

Dataset Tahap 3a berisi 10 kanal × 3 lag = fitur sangat lebar. Untuk model final, proyek ini menyederhanakan ke pendekatan **autoregressive murni** — hanya memakai riwayat `tbb_13` itu sendiri (bukan 9 kanal lain), menghasilkan dataset yang jauh lebih ringan:

```
base_time, target_time, pixel_row, pixel_col, lat, lon,
hour_sin, hour_cos, doy_sin, doy_cos,
tbb_13_t, tbb_13_tm1, tbb_13_tm2,
target_tbb_13
```

Disimpan sebagai `dataset/features_{interval}min_ar.csv`.

### Tahap 4 — Training Model (`04_train_models.py` + `pipeline/model_training.py`)

**Fitur input** (`FEATURE_COLUMNS`): `lat, lon, hour_sin, hour_cos, doy_sin, doy_cos, tbb_13_t, tbb_13_tm1, tbb_13_tm2` — 9 fitur.
**Target**: `target_tbb_13`.

- **Split kronologis** (bukan random!): 85% data awal → train, 15% data terakhir → test, dengan cutoff berdasarkan waktu (`base_time`) agar tidak ada kebocoran informasi masa depan. Semua piksel pada satu timestamp yang sama selalu berada di sisi split yang sama.
- Empat algoritma dilatih **secara terpisah untuk tiap interval** (10/30/60 menit):

| Model | Detail Implementasi |
|---|---|
| **SVR** (Support Vector Regression) | Kernel RBF, `C=10.0`, `epsilon=0.1`. Karena kompleksitas SVR kuadratik-kubik terhadap jumlah data, training di-*subsample* maksimal 25.000 baris. Fitur di-*scale* dengan `StandardScaler` (disimpan terpisah sebagai `.joblib` untuk dipakai ulang saat inference). |
| **XGBoost** | `n_estimators=300`, `max_depth=6`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`. |
| **LightGBM** | Parameter setara XGBoost (300 tree, depth 6, lr 0.05, subsample & colsample 0.8). |
| **CatBoost** | `iterations=300`, `depth=6`, `learning_rate=0.05`. |

- Evaluasi pada test set memakai **MAE**, **RMSE**, dan **R²**.
- Model + scaler disimpan ke `models/{interval}min/{model_name}.joblib`.
- Ringkasan seluruh kombinasi model×interval disimpan ke `models/training_summary.csv`.
- Seperti Tahap 3a, ini juga **full rebuild** — semua model dilatih ulang dari nol memakai seluruh dataset yang tersedia saat itu.

### Tahap 5 — Evaluasi Recursive Multi-Step (`05_recursive_evaluation.py` + `pipeline/recursive_eval.py`)

Model di Tahap 4 hanya dievaluasi untuk **satu langkah ke depan**. Tahap ini menguji performa saat model dipakai secara **recursive** (hasil prediksi dijadikan input langkah berikutnya) sepanjang horizon **3 jam (180 menit)**:

1. Membangun **lookup ground-truth** dari dataset resolusi native 10-menit — dipakai untuk membandingkan hasil forecast di semua interval (karena 10/30/60 menit semuanya kelipatan 10 menit).
2. Untuk tiap interval, memilih hingga 200 **titik awal** yang memiliki rantai ground-truth lengkap sampai `n_steps = 180/interval` langkah ke depan.
3. **`recursive_predict`**: pada tiap langkah, model memprediksi `t+interval`, lalu jendela lag digeser — `[tm2, tm1, t] → [tm1, t, prediksi]` — dan prediksi tersebut menjadi input untuk langkah berikutnya (bukan nilai aktual). Ini mensimulasikan kondisi forecast nyata di mana nilai aktual masa depan belum tersedia.
4. Error absolut (MAE) dihitung **per langkah** (misal +10 menit, +20 menit, ..., +180 menit), diakumulasi lintas semua titik awal, lalu disimpan di `models/recursive_evaluation.csv`.
5. Model dengan **rata-rata MAE terendah di seluruh langkah** akan dipilih sebagai model final di Tahap 6.

### Tahap 6 — Inference / Forecast Nyata (`06_run_inference.py` + `pipeline/inference.py`)

- **`select_best_model`**: membaca `recursive_evaluation.csv`, memilih model dengan MAE rata-rata terendah untuk interval dasar (10 menit).
- **`get_initial_windows`**: mengambil kondisi awal (`[tm2, tm1, t]`) untuk **seluruh piksel sekaligus** pada satu `t0` yang ditentukan pengguna.
- **`run_recursive_forecast`**: menjalankan forecast recursive secara **batch** (semua piksel diproses bersamaan tiap langkah, bukan satu per satu) sepanjang 18 langkah × 10 menit = 3 jam, sambil menyertakan nilai aktual (jika tersedia) untuk pembanding.
- **`filter_forecast_by_interval`**: dari hasil resolusi penuh 10 menit, diturunkan versi tampilan 30 dan 60 menit tanpa perlu model terpisah — cukup mengambil kelipatan langkah yang sesuai.

**Mode penentuan titik awal (`T0_STR`):**
- **Otomatis (default, `T0_STR = None`)** — otomatis memakai `base_time` **TERBARU** yang tersedia di `features_10min_ar.csv`. Tidak perlu edit file sama sekali untuk forecast dari data terbaru.
- **Manual** — isi `T0_STR` dengan timestamp tertentu (UTC), harus ada di kolom `base_time` di `features_10min_ar.csv`, kalau mau forecast dari tanggal/jam spesifik (bukan yang terbaru).

**Output & state run:**
- Tiap run disimpan di **subfolder tersendiri** berdasarkan timestamp `t0` (format `YYYYMMDD_HHMM`), supaya hasil dari run yang berbeda-beda tidak saling menumpuk/menimpa:
  ```
  forecast_output/20260705_1000/full10min.csv
  forecast_output/20260705_1000/display30min.csv
  forecast_output/20260705_1000/display60min.csv
  ```
- Info run terakhir (t0, folder, nama file) disimpan ke `forecast_output/last_run_state.json` — dipakai otomatis oleh Tahap 7 & 8 supaya keduanya selalu ikut run forecast yang paling terakhir dijalankan, tanpa perlu isi ulang nama file/timestamp secara manual.

> ⚠️ Timestamp data (`T0_STR`) menggunakan **UTC**, bukan WIB. Tambahkan 7 jam untuk tampilan waktu Indonesia Barat.

### Tahap 7 — Uji Visualisasi (`07_test_visualization.py` + `pipeline/visualization.py`)

Merender **satu frame** dari hasil forecast untuk mengecek styling sebelum membuat animasi penuh. Figure terdiri dari **6 panel** (grid 2×3), bertema *dark modern glassmorphism*:

| Panel | Isi |
|---|---|
| 1. Input | CTT aktual pada `t0` (kondisi awal) |
| 2. Prediksi | CTT hasil prediksi model pada waktu target |
| 3. Aktual | CTT aktual pada waktu target (jika tersedia, untuk validasi) |
| 4. Kelas Awan | Klasifikasi kategori: **Hujan** (<220K), **Mendung** (220–260K), **Tidak Hujan** (>260K) |
| 5. Risk Banjir | Klasifikasi risiko: **Bahaya** (<240K), **Waspada** (240–265K), **Aman** (>265K) |
| 6. Error Map | Peta sebaran spasial error absolut prediksi vs aktual |

Detail teknis render:
- Grid kasar (5×7 piksel) di-**upsampling** 12× memakai interpolasi kubik (`scipy.griddata`) agar kontur tampak halus, dengan fallback ke interpolasi *nearest* untuk titik yang tidak terjangkau kubik.
- Batas kecamatan Kota Bandung digambar dari `geojson/KotaBandung.geojson` (`geopandas`), lengkap dengan label 11 kecamatan.
- Judul figure menampilkan MAE dan skor akurasi (`100 × (1 - MAE/rentang_CTT_aktual)`) bila data aktual tersedia.

**Mode otomatis/manual:** sama seperti Tahap 6, `T0_STR` & `FORECAST_CSV` bisa dikosongkan (`None`) untuk otomatis membaca `forecast_output/last_run_state.json` (run terakhir), atau diisi manual kalau mau cek hasil forecast lama. `STEP_TO_RENDER` (frame mana yang mau dirender) tetap selalu manual.

**Output** disimpan di subfolder sesuai timestamp `t0`, misalnya:
```
visualizations/20260705_1000/test_frame_step18.png
```

### Tahap 8 — Render Animasi Penuh (`08_render_animation.py`)

- Untuk tiap interval tampilan (10/30/60 menit), me-render seluruh frame sepanjang horizon 3 jam menjadi PNG, lalu digabung menjadi **GIF** menggunakan `Pillow`, dengan durasi antar-frame 700ms.
- **Independen dari Tahap 7** — tidak memakai frame yang sudah dibuat `07_test_visualization.py`. Tahap 7 hanya untuk preview cepat 1 frame (cek styling); Tahap 8 me-render ulang semua frame yang dibutuhkan dari nol untuk animasi penuh.

**Mode otomatis/manual:** sama seperti Tahap 6 & 7, `T0_STR` & `FORECAST_CSV` otomatis mengikuti `last_run_state.json` (run terakhir) kalau dikosongkan.

**Output** disimpan di subfolder sesuai timestamp `t0`, misalnya:
```
visualizations/20260705_1000/frames/10min_step01.png
visualizations/20260705_1000/10_menit.gif
visualizations/20260705_1000/30_menit.gif
visualizations/20260705_1000/60_menit.gif
```

---

## ⚙️ Instalasi

Repo ini belum menyertakan `requirements.txt`, berikut daftar dependensi Python yang dibutuhkan berdasarkan analisis kode:

```bash
pip install xarray netCDF4 pandas numpy python-dotenv joblib
pip install scikit-learn xgboost lightgbm catboost
pip install matplotlib scipy geopandas Pillow
```

> `geopandas` memerlukan dependensi sistem tambahan (GDAL/GEOS/PROJ) — di Ubuntu/Debian bisa dibantu dengan `sudo apt install libgdal-dev` sebelum `pip install geopandas`, atau gunakan `conda install geopandas`.

## 🔐 Konfigurasi (.env)

Salin `.env.example` menjadi `.env` di root proyek, lalu isi kredensial FTP data satelit (mis. JAXA P-Tree):

```env
# JMA / JAXA FTP Configuration
FTP_HOST=ftp.ptree.jaxa.jp
FTP_USER=
FTP_PASS=
FTP_REMOTE_DIR=

# Filter nama file yang diunduh
FILENAME_MUST_CONTAIN=R21_FLDK.02801_02401

# Notifikasi Telegram (opsional)
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
```

## ▶️ Cara Menjalankan

Jalankan seluruh skrip **dari folder `scripts/`**, sesuai urutan tahap:

```bash
cd scripts/

# Tahap 1: unduh & subset data Himawari-9 untuk area Bandung
python 01_download_data.py

# Tahap 2 (opsional): cek cakupan waktu & data yang hilang
python 02_explore_data.py

# Tahap 3: feature engineering + ekstraksi fitur autoregressive
python 03a_build_features.py
python 03b_extract_autoregressive.py

# Tahap 4: training 4 model ML untuk tiap interval (10/30/60 menit)
python 04_train_models.py

# Tahap 5: evaluasi recursive multi-step, pilih model terbaik
python 05_recursive_evaluation.py

# Tahap 6: jalankan forecast 3 jam
#   Default: otomatis pakai base_time TERBARU, tidak perlu edit apa pun.
#   Opsional: isi T0_STR di dalam file kalau mau forecast dari timestamp tertentu.
python 06_run_inference.py

# Tahap 7 (opsional): uji render satu frame
#   Otomatis ikut hasil forecast terakhir dari Tahap 6 (last_run_state.json)
python 07_test_visualization.py

# Tahap 8: render animasi GIF penuh untuk semua interval
#   Otomatis ikut hasil forecast terakhir dari Tahap 6 (last_run_state.json)
python 08_render_animation.py
```

> 💡 Kalau nanti menambah data `.nc` baru: cukup ulangi dari Tahap 1 (atau 3a kalau data sudah ada) sampai Tahap 8. Tahap 3a–5 selalu memproses ulang seluruh data dari nol (bukan incremental), jadi tidak perlu langkah tambahan khusus selain menjalankan urutan yang sama.

## 📤 Output yang Dihasilkan

| Folder | Isi |
|---|---|
| `data_bandung/` | File NetCDF hasil subset area Bandung |
| `_metadata/` | Log file yang sudah diperiksa/diproses (untuk resume) |
| `dataset/` | CSV fitur (`features_{interval}min.csv`, `features_{interval}min_ar.csv`) |
| `models/` | Model terlatih (`.joblib`), scaler SVR, `training_summary.csv`, `recursive_evaluation.csv` |
| `forecast_output/` | `last_run_state.json` (info run terakhir) + subfolder per-timestamp `{YYYYMMDD_HHMM}/` berisi `full10min.csv`, `display30min.csv`, `display60min.csv` |
| `visualizations/` | Subfolder per-timestamp `{YYYYMMDD_HHMM}/` berisi frame PNG 6-panel, animasi GIF per interval, dan folder `frames/` (PNG mentah animasi) |
| `download_activity.log` | Log proses download |

## 📝 Catatan Teknis

- **Zona waktu**: semua timestamp di pipeline (nama file, kolom `base_time`/`target_time`) dalam **UTC**. Visualisasi mengonversinya ke **WIB (+7 jam)**.
- **Ukuran grid**: area Bandung yang di-subset menghasilkan grid kecil (~5×7 piksel) sesuai resolusi native Himawari-9 di area tersebut — karena itu peta divisualisasikan dengan interpolasi upsampling agar tampak halus, bukan karena resolusi model yang tinggi.
- **Pendekatan model**: per-piksel + autoregressive, **bukan** model spasial (mis. CNN/ConvLSTM) — setiap piksel diprediksi independen berdasarkan riwayat waktunya sendiri, lat/lon, dan fitur musiman/harian.
- **Pemilihan model final** dilakukan otomatis berdasarkan performa recursive (multi-step), bukan cuma performa satu langkah — ini penting karena error cenderung terakumulasi pada forecast recursive jangka panjang.
- **Incremental vs full-rebuild**: hanya Tahap 1 (download) yang incremental (resume otomatis via `FileTracker`). Tahap 3a, 3b, 4, dan 5 selalu memproses ulang **seluruh** data yang ada di folder setiap dijalankan (bukan menambah ke file lama) — jadi waktu proses akan bertambah seiring makin banyak data yang terkumpul.
- **Run forecast (Tahap 6–8) tidak saling menimpa**: setiap kombinasi timestamp `t0` punya subfolder sendiri di `forecast_output/` dan `visualizations/`, jadi hasil forecast dari tanggal berbeda-beda bisa disimpan berdampingan tanpa perlu diganti nama manual.
- Proyek tampaknya ditujukan sebagai alat bantu **peringatan dini potensi hujan/genangan** di Kota Bandung (lihat kategori "Kelas Awan" dan "Risk Banjir" pada visualisasi). Perlu dicatat: `tbb_13` adalah **proxy suhu awan**, bukan pengukuran curah hujan langsung — untuk validasi/skala yang lebih rigor terhadap kejadian hujan aktual, disarankan menyandingkan dengan data hujan observasi (mis. GSMaP, GPM IMERG, atau data BMKG).