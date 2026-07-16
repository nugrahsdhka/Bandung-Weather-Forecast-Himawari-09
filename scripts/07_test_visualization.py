# 07_test_visualization.py
#
# Tahap 6 (percobaan): render SATU frame dulu dari hasil forecast Tahap 5,
# untuk mengecek styling-nya sebelum kita bikin loop animasi penuh.
#
# GANTI FORECAST_CSV & STEP_TO_RENDER sesuai file & langkah yang mau dicek.
#
# Jalankan dari folder scripts/:
#   pip install matplotlib scipy   (kalau belum ada)
#   python 07_test_visualization.py

import os
import pandas as pd

from ui.terminal_display import hr, banner, say_info, say_ok, say_error
from pipeline.config import load_config
from pipeline.model_training import load_ar_dataset
from pipeline.visualization import render_six_panel, pivot_grid, get_axis_arrays

# ==== GANTI SESUAI KEBUTUHAN ====
T0_STR = "2026-07-05 10:00:00"          # harus sama dengan T0_STR di 06_run_inference.py
FORECAST_CSV = "forecast_20260705_1000_full10min.csv"  # nama file hasil Tahap 5
STEP_TO_RENDER = 18                     # 18 = +180 menit (frame terakhir), 6 = +60 menit, dst.
TZ_OFFSET_HOURS = 7                     # UTC -> WIB
# =================================


def main():
    cfg = load_config()
    dataset_dir = os.path.join(cfg.PROJECT_ROOT, "dataset")
    output_dir = os.path.join(cfg.PROJECT_ROOT, "forecast_output")
    viz_dir = os.path.join(cfg.PROJECT_ROOT, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    banner("TEST VISUALISASI - SATU FRAME")

    forecast_path = os.path.join(output_dir, FORECAST_CSV)
    if not os.path.exists(forecast_path):
        say_error(f"File tidak ditemukan: {forecast_path}")
        return

    say_info(f"Membaca: {forecast_path}")
    forecast_df = pd.read_csv(forecast_path, parse_dates=["forecast_time"])

    frame_df = forecast_df[forecast_df["step"] == STEP_TO_RENDER].reset_index(drop=True)
    if frame_df.empty:
        say_error(f"Tidak ada data untuk step={STEP_TO_RENDER}. Step tersedia: {sorted(forecast_df['step'].unique())}")
        return

    say_info(f"Merender step {STEP_TO_RENDER} ({frame_df['forecast_time'].iloc[0]}) untuk {len(frame_df)} pixel")

    say_info("Memuat kondisi awal (input t0) dari features_10min_ar.csv ...")
    df10 = load_ar_dataset(os.path.join(dataset_dir, "features_10min_ar.csv"))
    t0 = pd.to_datetime(T0_STR)
    initial_df = df10[df10["base_time"] == t0][["pixel_row", "pixel_col", "lat", "lon", "tbb_13_t"]].reset_index(drop=True)
    if initial_df.empty:
        say_error(f"Tidak ada data awal untuk t0={t0}")
        return

    lat_arr, lon_arr = get_axis_arrays(frame_df)
    input_grid = pivot_grid(initial_df, "tbb_13_t")
    pred_grid = pivot_grid(frame_df, "predicted_tbb13")

    has_actual = frame_df["actual_tbb13"].notna().all()
    actual_grid = pivot_grid(frame_df, "actual_tbb13") if has_actual else None

    interval_display = STEP_TO_RENDER * 10  # menit ke depan dari step ini (resolusi native 10 menit)

    t0_wib = t0 + pd.Timedelta(hours=TZ_OFFSET_HOURS)
    forecast_time = frame_df["forecast_time"].iloc[0]
    forecast_wib = forecast_time + pd.Timedelta(hours=TZ_OFFSET_HOURS)

    out_path = os.path.join(viz_dir, f"test_frame_step{STEP_TO_RENDER}.png")
    render_six_panel(
        lat_arr, lon_arr,
        input_grid, pred_grid, actual_grid,
        t0_label=t0_wib.strftime("%H:%M WIB"),
        forecast_label=forecast_wib.strftime("%H:%M WIB"),
        interval_minutes=interval_display,
        out_path=out_path,
    )

    hr()
    say_ok(f"Disimpan: {out_path}")
    say_info("Buka file PNG itu dan cek stylingnya -- kabarin kalau ada yang mau diubah (warna, ukuran font, dll).")


if __name__ == "__main__":
    main()