# ./scripts/03b_extract_autoregressive.py
# Mengekstrak fitur autoregressive dari dataset hasil feature engineering dengan hanya mempertahankan variabel tbb_13 beserta target prediksinya.

import argparse
import os
import pandas as pd

from ui.terminal_display import hr, gap, banner, say_info, say_ok, say_error, say_skip
from pipeline.config import load_config
from pipeline.incremental_csv import (
    validate_existing_file,
    append_incremental,
    backup_and_clear,
    CorruptDatasetError,
)

INTERVALS_MINUTES = [10, 30, 60]

AR_COLUMNS = [
    "base_time", "target_time", "pixel_row", "pixel_col", "lat", "lon",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
    "tbb_13_t", "tbb_13_tm1", "tbb_13_tm2",
    # Fitur tetangga & anchor (fix #4) -- lihat pipeline/spatial_features.py
    "tbb13_neighbor_mean", "tbb13_neighbor_diff",
    "tbb_13_last_real_obs", "minutes_since_last_real_obs",
    "target_tbb_13",
]

DEDUP_COLS = ["base_time", "pixel_row", "pixel_col"]
SORT_COLS = ["base_time", "pixel_row", "pixel_col"]
SORT_COL = "base_time"
PARSE_DATES = ["base_time", "target_time"]


def parse_args():
    p = argparse.ArgumentParser(description="Ekstraksi fitur autoregressive (incremental by default).")
    p.add_argument(
        "--rebuild", action="store_true",
        help="Backup *_ar.csv lama lalu bangun ulang dari nol (bukan incremental).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config()
    dataset_dir = os.path.join(cfg.PROJECT_ROOT, "dataset")

    mode_label = "REBUILD (full dari nol)" if args.rebuild else "INCREMENTAL"
    banner(f"EKSTRAKSI DATASET AUTOREGRESSIVE (tbb_13 saja) - mode {mode_label}")

    for interval in INTERVALS_MINUTES:
        src_path = os.path.join(dataset_dir, f"features_{interval}min.csv")
        out_path = os.path.join(dataset_dir, f"features_{interval}min_ar.csv")

        if not os.path.exists(src_path):
            say_error(f"File tidak ditemukan: {src_path} (jalankan 03a_build_features.py dulu)")
            continue

        if args.rebuild:
            backup_and_clear(out_path, reason="rebuild 03b")
        else:
            try:
                validate_existing_file(
                    out_path, dedup_cols=DEDUP_COLS, sort_col=SORT_COL, parse_dates=PARSE_DATES,
                )
            except CorruptDatasetError as e:
                say_error(str(e))
                say_info(f"[interval {interval} menit] dilewati supaya tidak menimpa data yang tidak konsisten.")
                gap()
                continue

        say_info(f"Membaca: {src_path}")
        df = pd.read_csv(src_path)

        missing_cols = [c for c in AR_COLUMNS if c not in df.columns]
        if missing_cols:
            say_error(f"Kolom berikut tidak ditemukan di {src_path}: {missing_cols}")
            continue

        df_ar = df[AR_COLUMNS].copy()
        df_ar["base_time"] = pd.to_datetime(df_ar["base_time"])

        if os.path.exists(out_path):
            existing_base_times = set(
                pd.read_csv(out_path, usecols=["base_time"], parse_dates=["base_time"])["base_time"]
            )
            df_new = df_ar[~df_ar["base_time"].isin(existing_base_times)]

            if df_new.empty:
                say_skip(f"[interval {interval} menit] tidak ada base_time baru, dilewati.")
                gap()
                continue

            total_rows = append_incremental(
                df_new, out_path, dedup_cols=DEDUP_COLS, sort_cols=SORT_COLS,
                sort_col=SORT_COL, parse_dates=PARSE_DATES,
            )
            say_ok(
                f"[interval {interval} menit] +{len(df_new)} baris ditambahkan ke: {out_path}  "
                f"({os.path.getsize(out_path)/1024:.1f} KB total, {total_rows} baris, terurut rapi)"
            )
        else:
            total_rows = append_incremental(
                df_ar, out_path, dedup_cols=DEDUP_COLS, sort_cols=SORT_COLS,
                sort_col=SORT_COL, parse_dates=PARSE_DATES,
            )
            say_ok(
                f"[interval {interval} menit] Dibuat baru: {out_path}  "
                f"({total_rows} baris, {df_ar.shape[1]} kolom, {os.path.getsize(out_path)/1024:.1f} KB)"
            )

        gap()

    hr()
    banner("SELESAI")
    say_info("Dataset autoregressive siap. Lanjut ke Tahap 4: training model.")


if __name__ == "__main__":
    main()