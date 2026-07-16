# ./scripts/032_extract_autoregressive.py
# Mengekstrak fitur autoregressive dari dataset hasil feature engineering dengan hanya mempertahankan variabel tbb_13 beserta target prediksinya.

import os
import pandas as pd

from ui.terminal_display import hr, gap, banner, say_info, say_ok, say_error
from pipeline.config import load_config

INTERVALS_MINUTES = [10, 30, 60]

AR_COLUMNS = [
    "base_time", "target_time", "pixel_row", "pixel_col", "lat", "lon",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
    "tbb_13_t", "tbb_13_tm1", "tbb_13_tm2",
    "target_tbb_13",
]


def main():
    cfg = load_config()
    dataset_dir = os.path.join(cfg.PROJECT_ROOT, "dataset")

    banner("EKSTRAKSI DATASET AUTOREGRESSIVE (tbb_13 saja)")

    for interval in INTERVALS_MINUTES:
        src_path = os.path.join(dataset_dir, f"features_{interval}min.csv")
        if not os.path.exists(src_path):
            say_error(f"File tidak ditemukan: {src_path} (jalankan 03_build_features.py dulu)")
            continue

        say_info(f"Membaca: {src_path}")
        df = pd.read_csv(src_path)

        missing_cols = [c for c in AR_COLUMNS if c not in df.columns]
        if missing_cols:
            say_error(f"Kolom berikut tidak ditemukan di {src_path}: {missing_cols}")
            continue

        df_ar = df[AR_COLUMNS].copy()

        out_path = os.path.join(dataset_dir, f"features_{interval}min_ar.csv")
        df_ar.to_csv(out_path, index=False)
        say_ok(
            f"[interval {interval} menit] Disimpan: {out_path}  "
            f"({len(df_ar)} baris, {df_ar.shape[1]} kolom, {os.path.getsize(out_path)/1024:.1f} KB)"
        )
        gap()

    hr()
    banner("SELESAI")
    say_info("Dataset autoregressive siap. Lanjut ke Tahap 3: training model.")


if __name__ == "__main__":
    main()