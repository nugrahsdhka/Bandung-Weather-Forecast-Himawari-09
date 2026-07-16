# 07_test_visualization.py
# Merender satu frame visualisasi enam panel dari hasil recursive forecasting untuk menguji tampilan output.

import os
import json
import pandas as pd

from ui.terminal_display import hr, banner, say_info, say_ok, say_error
from pipeline.config import load_config
from pipeline.model_training import load_ar_dataset
from pipeline.visualization import render_six_panel, pivot_grid, get_axis_arrays

# ==== GANTI SESUAI KEBUTUHAN (opsional) ====
T0_STR = None          # None = otomatis dari last_run_state.json (hasil run 06 terakhir)
FORECAST_CSV = None    # None = otomatis dari last_run_state.json
STEP_TO_RENDER = 18                     # 18 = +180 menit (frame terakhir), 6 = +60 menit, dst.
TZ_OFFSET_HOURS = 7                     # UTC -> WIB
# =================================


def load_last_run_state(root_output_dir):
    state_path = os.path.join(root_output_dir, "last_run_state.json")
    if not os.path.exists(state_path):
        raise FileNotFoundError(
            f"{state_path} tidak ditemukan. Jalankan 06_run_inference.py dulu, "
            "atau isi T0_STR & FORECAST_CSV secara manual di atas."
        )
    with open(state_path) as f:
        return json.load(f)


def main():
    cfg = load_config()
    dataset_dir = os.path.join(cfg.PROJECT_ROOT, "dataset")
    root_output_dir = os.path.join(cfg.PROJECT_ROOT, "forecast_output")
    root_viz_dir = os.path.join(cfg.PROJECT_ROOT, "visualizations")

    banner("TEST VISUALISASI - SATU FRAME")

    t0_str, forecast_csv = T0_STR, FORECAST_CSV
    if t0_str is None or forecast_csv is None:
        state = load_last_run_state(root_output_dir)
        t0_str = t0_str or state["t0"]
        forecast_csv = forecast_csv or state["forecast_csv_full10min"]
        say_info(f"Otomatis pakai run terakhir: t0={t0_str}, file={forecast_csv}")

    t0 = pd.to_datetime(t0_str)
    t0_tag = t0.strftime("%Y%m%d_%H%M")
    viz_dir = os.path.join(root_viz_dir, t0_tag)
    os.makedirs(viz_dir, exist_ok=True)

    forecast_path = os.path.join(root_output_dir, forecast_csv)
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