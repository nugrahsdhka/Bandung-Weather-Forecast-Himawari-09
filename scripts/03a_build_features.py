# ./scripts/03a_build_features.py
# Membangun dataset supervised learning dari data CTT dengan melakukan feature engineering dan menghasilkan file CSV untuk interval prediksi 10, 30, dan 60 menit.

import argparse
import os

import pandas as pd

from ui.terminal_display import hr, gap, banner, say_info, say_ok, say_skip, say_error
from pipeline.config import load_config
from pipeline.feature_engineering import (
    scan_timestamp_index,
    preload_frames,
    build_interval_dataset,
    get_aligned_candidates,
    required_timestamps_for_candidates,
)
from pipeline.incremental_csv import (
    validate_existing_file,
    append_incremental,
    backup_and_clear,
    CorruptDatasetError,
)

INTERVALS_MINUTES = [10, 30, 60]
LAG_COUNT = 3
DEDUP_COLS = ["base_time", "pixel_row", "pixel_col"]
SORT_COLS = ["base_time", "pixel_row", "pixel_col"]
SORT_COL = "base_time"
PARSE_DATES = ["base_time", "target_time"]


def parse_args():
    p = argparse.ArgumentParser(description="Feature engineering CTT (incremental by default).")
    p.add_argument(
        "--rebuild", action="store_true",
        help="Backup CSV lama lalu bangun ulang dataset dari nol (bukan incremental).",
    )
    return p.parse_args()


def load_existing_base_times(csv_path):
    """Baca base_time yang sudah ada di CSV lama (kalau ada)."""
    if not os.path.exists(csv_path):
        return set()
    return set(pd.read_csv(csv_path, usecols=["base_time"], parse_dates=["base_time"])["base_time"])


def main():
    args = parse_args()
    cfg = load_config()

    mode_label = "REBUILD (full dari nol)" if args.rebuild else "INCREMENTAL"
    banner(f"FEATURE ENGINEERING - CTT FORECASTING (mode {mode_label})")
    say_info(f"Folder data   : {cfg.FINAL_BASE_DIR}")

    output_dir = os.path.join(cfg.PROJECT_ROOT, "dataset")
    os.makedirs(output_dir, exist_ok=True)
    say_info(f"Folder output : {output_dir}")
    hr()

    out_paths = {i: os.path.join(output_dir, f"features_{i}min.csv") for i in INTERVALS_MINUTES}

    # --- Mode rebuild: backup file lama dulu, baru dianggap "belum ada" ---
    if args.rebuild:
        for interval in INTERVALS_MINUTES:
            backup_and_clear(out_paths[interval], reason="rebuild 03a")
        hr()
    else:
        # --- Guard: validasi file lama sebelum dipakai (deteksi korup/setengah-jalan) ---
        try:
            for interval in INTERVALS_MINUTES:
                validate_existing_file(
                    out_paths[interval], dedup_cols=DEDUP_COLS, sort_col=SORT_COL,
                    parse_dates=PARSE_DATES,
                )
        except CorruptDatasetError as e:
            say_error(str(e))
            say_info("Berhenti supaya tidak memproses di atas data yang berpotensi tidak konsisten.")
            return
        say_ok("Validasi file lama: aman (tidak ada duplikat/korupsi/urutan tidak konsisten).")
        hr()

    # Hanya menelusuri nama file (regex timestamp), belum membuka isi file .nc sama sekali.
    index = scan_timestamp_index(cfg.FINAL_BASE_DIR)
    all_ts = sorted(index.keys())
    say_info(f"Total timestamp terindeks: {len(all_ts)}")
    hr()

    # --- Tahap 1: tentukan kandidat baru per interval (tanpa buka file .nc) ---
    plan = {}
    all_required_ts = set()

    for interval in INTERVALS_MINUTES:
        existing_base_times = load_existing_base_times(out_paths[interval])
        aligned_candidates = get_aligned_candidates(all_ts, interval)
        new_candidates = [t for t in aligned_candidates if t not in existing_base_times]

        plan[interval] = {"new_candidates": new_candidates}

        say_info(
            f"[interval {interval} menit] sudah ada: {len(existing_base_times)} base_time  |  "
            f"kandidat baru: {len(new_candidates)}"
        )
        all_required_ts |= required_timestamps_for_candidates(new_candidates, interval, LAG_COUNT)

    hr()

    if not all_required_ts:
        gap()
        banner("SELESAI")
        say_info("Tidak ada base_time baru untuk diproses di semua interval. Dataset sudah up-to-date.")
        return

    # --- Tahap 2: load HANYA file .nc yang dibutuhkan kandidat baru ---
    frames, lat, lon = preload_frames(index, only_timestamps=all_required_ts)
    if not frames:
        say_info("Tidak ada frame yang berhasil dimuat, berhenti.")
        return
    hr()

    # --- Tahap 3: bangun baris baru per interval & tulis (append/rebuild) dgn aman ---
    for interval in INTERVALS_MINUTES:
        gap()
        banner(f"INTERVAL {interval} MENIT")

        new_candidates = plan[interval]["new_candidates"]
        out_path = out_paths[interval]

        if not new_candidates:
            say_skip(f"[interval {interval} menit] tidak ada base_time baru, dilewati.")
            continue

        df_new = build_interval_dataset(
            frames, lat, lon, interval_minutes=interval, lag_count=LAG_COUNT,
            candidates=new_candidates,
        )

        if df_new.empty:
            say_skip(
                f"[interval {interval} menit] kandidat baru ada, tapi frame lag/target-nya "
                f"belum lengkap semua (kemungkinan data terbaru masih parsial) - dilewati."
            )
            continue

        total_rows = append_incremental(
            df_new, out_path, dedup_cols=DEDUP_COLS, sort_cols=SORT_COLS,
            sort_col=SORT_COL, parse_dates=PARSE_DATES,
        )

        say_ok(
            f"[interval {interval} menit] +{len(df_new)} baris ditambahkan ke: {out_path}  "
            f"({os.path.getsize(out_path) / 1024:.1f} KB total, {total_rows} baris, terurut rapi)"
        )

    gap()
    banner("SELESAI")
    say_info("Lanjut ke Tahap 3b: ekstraksi fitur autoregressive.")


if __name__ == "__main__":
    main()