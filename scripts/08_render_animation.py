# 08_render_animation.py

import os
import pandas as pd
from PIL import Image

from ui.terminal_display import hr, gap, banner, say_info, say_ok, say_error
from pipeline.config import load_config
from pipeline.model_training import load_ar_dataset
from pipeline.visualization import render_six_panel, pivot_grid, get_axis_arrays

# ==== GANTI SESUAI KEBUTUHAN ====
T0_STR = "2026-07-05 10:00:00"                          # harus sama dengan T0_STR di 06_run_inference.py
FORECAST_CSV = "forecast_20260705_1000_full10min.csv"   # file hasil Tahap 5 (resolusi penuh 10 menit)
DISPLAY_INTERVALS = [10, 30, 60]
TZ_OFFSET_HOURS = 7                                      # UTC -> WIB
FRAME_DURATION_MS = 700                                  # kecepatan animasi (ms per frame)
BASE_INTERVAL_MINUTES = 10
HORIZON_MINUTES = 180
# =================================


def get_steps_for_interval(display_interval, base_interval=BASE_INTERVAL_MINUTES, max_step=None):
    max_step = max_step or (HORIZON_MINUTES // base_interval)
    step_multiple = display_interval // base_interval
    return [s for s in range(step_multiple, max_step + 1, step_multiple)]


def main():
    cfg = load_config()
    dataset_dir = os.path.join(cfg.PROJECT_ROOT, "dataset")
    output_dir = os.path.join(cfg.PROJECT_ROOT, "forecast_output")
    viz_dir = os.path.join(cfg.PROJECT_ROOT, "visualizations")
    frames_dir = os.path.join(viz_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    banner("RENDER ANIMASI PENUH")

    forecast_path = os.path.join(output_dir, FORECAST_CSV)
    if not os.path.exists(forecast_path):
        say_error(f"File tidak ditemukan: {forecast_path}")
        return

    say_info(f"Membaca: {forecast_path}")
    forecast_df = pd.read_csv(forecast_path, parse_dates=["forecast_time"])

    say_info("Memuat kondisi awal (input t0) dari features_10min_ar.csv ...")
    df10 = load_ar_dataset(os.path.join(dataset_dir, "features_10min_ar.csv"))
    t0 = pd.to_datetime(T0_STR)
    initial_df = df10[df10["base_time"] == t0][["pixel_row", "pixel_col", "lat", "lon", "tbb_13_t"]].reset_index(drop=True)
    if initial_df.empty:
        say_error(f"Tidak ada data awal untuk t0={t0}")
        return

    lat_arr, lon_arr = get_axis_arrays(initial_df)
    input_grid = pivot_grid(initial_df, "tbb_13_t")
    t0_wib = t0 + pd.Timedelta(hours=TZ_OFFSET_HOURS)
    hr()

    for display_interval in DISPLAY_INTERVALS:
        gap()
        banner(f"INTERVAL TAMPILAN {display_interval} MENIT")

        steps = get_steps_for_interval(display_interval)
        say_info(f"Merender {len(steps)} frame: langkah {steps}")

        frame_paths = []
        for step in steps:
            frame_df = forecast_df[forecast_df["step"] == step].reset_index(drop=True)
            if frame_df.empty:
                say_error(f"Step {step} tidak ada di data, dilewati.")
                continue

            pred_grid = pivot_grid(frame_df, "predicted_tbb13")
            has_actual = frame_df["actual_tbb13"].notna().all()
            actual_grid = pivot_grid(frame_df, "actual_tbb13") if has_actual else None

            forecast_time = frame_df["forecast_time"].iloc[0]
            forecast_wib = forecast_time + pd.Timedelta(hours=TZ_OFFSET_HOURS)

            frame_path = os.path.join(frames_dir, f"{display_interval}min_step{step:02d}.png")
            render_six_panel(
                lat_arr, lon_arr,
                input_grid, pred_grid, actual_grid,
                t0_label=t0_wib.strftime("%H:%M WIB"),
                forecast_label=forecast_wib.strftime("%H:%M WIB"),
                interval_minutes=display_interval,
                out_path=frame_path,
            )
            frame_paths.append(frame_path)
            say_ok(f"Frame step {step} ({forecast_wib.strftime('%H:%M')} WIB) disimpan")

        if not frame_paths:
            say_error(f"Tidak ada frame untuk interval {display_interval} menit, GIF dilewati.")
            continue

        say_info("Menyusun GIF ...")
        images = [Image.open(p).convert("RGB") for p in frame_paths]
        gif_path = os.path.join(viz_dir, f"{display_interval}_menit.gif")
        images[0].save(
            gif_path, save_all=True, append_images=images[1:],
            duration=FRAME_DURATION_MS, loop=0,
        )
        say_ok(f"GIF disimpan: {gif_path}  ({len(images)} frame)")

    gap()
    banner("SELESAI")
    say_info("Semua animasi ada di folder visualizations/ (frame PNG mentah ada di visualizations/frames/).")


if __name__ == "__main__":
    main()