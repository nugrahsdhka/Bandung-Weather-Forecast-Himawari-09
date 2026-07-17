# 08_render_animation.py
# Merender animasi GIF enam panel dari hasil recursive forecasting pada interval tampilan 10, 30, dan 60 menit.

import os
import json
import pandas as pd
from PIL import Image

from ui.terminal_display import hr, gap, banner, say_info, say_ok, say_error
from pipeline.config import load_config
from pipeline.model_training import load_ar_dataset
from pipeline.visualization import render_six_panel, pivot_grid, get_axis_arrays

# ==== GANTI SESUAI KEBUTUHAN (opsional) ====
T0_STR = None          # None = otomatis dari last_run_state.json (hasil run 06 terakhir)
FORECAST_CSV = None    # None = otomatis dari last_run_state.json
MODEL_NAME = None      # None = otomatis dari last_run_state.json (dipakai buat nama folder viz, mis. "_svr")
DISPLAY_INTERVALS = [10, 30, 60]
TZ_OFFSET_HOURS = 7                                      # UTC -> WIB
FRAME_DURATION_MS = 700                                  # kecepatan animasi (ms per frame)
BASE_INTERVAL_MINUTES = 10
HORIZON_MINUTES = 180
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


def get_steps_for_interval(display_interval, base_interval=BASE_INTERVAL_MINUTES, max_step=None):
    max_step = max_step or (HORIZON_MINUTES // base_interval)
    step_multiple = display_interval // base_interval
    return [s for s in range(step_multiple, max_step + 1, step_multiple)]


def main():
    cfg = load_config()
    dataset_dir = os.path.join(cfg.PROJECT_ROOT, "dataset")
    root_output_dir = os.path.join(cfg.PROJECT_ROOT, "forecast_output")
    root_viz_dir = os.path.join(cfg.PROJECT_ROOT, "visualizations")

    banner("RENDER ANIMASI PENUH")

    t0_str, forecast_csv, model_name = T0_STR, FORECAST_CSV, MODEL_NAME
    if t0_str is None or forecast_csv is None:
        state = load_last_run_state(root_output_dir)
        t0_str = t0_str or state["t0"]
        forecast_csv = forecast_csv or state["forecast_csv_full10min"]
        model_name = model_name or state.get("model_name")
        say_info(f"Otomatis pakai run terakhir: t0={t0_str}, model={model_name}, file={forecast_csv}")

    t0 = pd.to_datetime(t0_str)
    tag_suffix = f"_{model_name}" if model_name else ""
    t0_tag = t0.strftime("%Y%m%d_%H%M") + tag_suffix
    viz_dir = os.path.join(root_viz_dir, t0_tag)
    frames_dir = os.path.join(viz_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    forecast_path = os.path.join(root_output_dir, forecast_csv)
    if not os.path.exists(forecast_path):
        say_error(f"File tidak ditemukan: {forecast_path}")
        return

    say_info(f"Membaca: {forecast_path}")
    forecast_df = pd.read_csv(forecast_path, parse_dates=["forecast_time"])

    say_info("Memuat kondisi awal (input t0) dari features_10min_ar.csv ...")
    df10 = load_ar_dataset(os.path.join(dataset_dir, "features_10min_ar.csv"))
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
    say_info(f"Semua animasi ada di folder {viz_dir}/ (frame PNG mentah ada di {frames_dir}/).")


if __name__ == "__main__":
    main()