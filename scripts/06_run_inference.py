# 06_run_inference.py
# Menjalankan proses inference recursive forecasting selama 3 jam menggunakan model terbaik dan menyimpan hasil prediksi untuk visualisasi.

import os
import json
import joblib
import pandas as pd

from ui.terminal_display import hr, gap, banner, say_info, say_ok, say_error
from pipeline.config import load_config
from pipeline.model_training import load_ar_dataset
from pipeline.recursive_eval import build_ground_truth_lookup
from pipeline.inference import (
    select_best_model,
    get_initial_windows,
    run_recursive_forecast,
    filter_forecast_by_interval,
)

# ==== GANTI SESUAI KEBUTUHAN (opsional) ====
T0_STR = "2026-01-03 10:10:00"   # None = otomatis pakai base_time TERBARU di features_10min_ar.csv
BASE_INTERVAL_MINUTES = 10
HORIZON_MINUTES = 180
DISPLAY_INTERVALS = [10, 30, 60]
# =================================


def main():
    cfg = load_config()
    dataset_dir = os.path.join(cfg.PROJECT_ROOT, "dataset")
    models_dir = os.path.join(cfg.PROJECT_ROOT, "models")
    root_output_dir = os.path.join(cfg.PROJECT_ROOT, "forecast_output")
    os.makedirs(root_output_dir, exist_ok=True)

    banner("INFERENCE - FORECAST RECURSIVE 3 JAM")

    say_info("Memilih model terbaik berdasarkan hasil Tahap 4 (recursive_evaluation.csv) ...")
    best_model_name, avg_mae_table = select_best_model(models_dir, BASE_INTERVAL_MINUTES, cfg.MODEL_NAMES)
    say_ok(f"Model terpilih: {best_model_name}")
    print(avg_mae_table.to_string())
    hr()

    interval_dir = os.path.join(models_dir, f"{BASE_INTERVAL_MINUTES}min")
    model = joblib.load(os.path.join(interval_dir, f"{best_model_name}.joblib"))
    scaler_path = os.path.join(interval_dir, f"{best_model_name}_scaler.joblib")
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

    say_info("Memuat dataset 10 menit untuk titik awal & ground-truth lookup ...")
    df10 = load_ar_dataset(os.path.join(dataset_dir, "features_10min_ar.csv"))
    lookup = build_ground_truth_lookup(df10)
    hr()

    if T0_STR is None:
        t0 = df10["base_time"].max()
        say_info(f"T0_STR tidak diisi -> otomatis pakai base_time TERBARU: {t0} (UTC)")
    else:
        t0 = pd.to_datetime(T0_STR)
        say_info(f"Titik awal forecast: {t0} (UTC)")

    t0_tag = t0.strftime("%Y%m%d_%H%M") + f"_{best_model_name}"
    output_dir = os.path.join(root_output_dir, t0_tag)
    os.makedirs(output_dir, exist_ok=True)

    try:
        windows = get_initial_windows(df10, t0)
    except ValueError as e:
        say_error(str(e))
        say_info("Contoh timestamp valid bisa dicek lewat kolom 'base_time' di features_10min_ar.csv")
        return

    say_ok(f"Window awal ditemukan untuk {len(windows)} pixel")

    n_steps = HORIZON_MINUTES // BASE_INTERVAL_MINUTES
    say_info(f"Menjalankan forecast recursive: {n_steps} langkah x {BASE_INTERVAL_MINUTES} menit ...")
    forecast_df = run_recursive_forecast(
        model, scaler, t0, windows,
        n_steps=n_steps, interval_minutes=BASE_INTERVAL_MINUTES, lookup=lookup,
    )
    say_ok(f"Forecast selesai: {len(forecast_df)} baris ({n_steps} langkah x {len(windows)} pixel)")
    hr()

    full_path = os.path.join(output_dir, "full10min.csv")
    forecast_df.to_csv(full_path, index=False)
    say_ok(f"Disimpan (resolusi penuh 10 menit): {full_path}")

    for disp_interval in DISPLAY_INTERVALS:
        if disp_interval == BASE_INTERVAL_MINUTES:
            continue
        sub_df = filter_forecast_by_interval(forecast_df, disp_interval, BASE_INTERVAL_MINUTES)
        sub_path = os.path.join(output_dir, f"display{disp_interval}min.csv")
        sub_df.to_csv(sub_path, index=False)
        say_ok(f"Disimpan (tampilan {disp_interval} menit, {len(sub_df)} baris): {sub_path}")

    state = {
        "t0": t0.strftime("%Y-%m-%d %H:%M:%S"),
        "t0_tag": t0_tag,
        "model_name": best_model_name,
        "base_interval_minutes": BASE_INTERVAL_MINUTES,
        "horizon_minutes": HORIZON_MINUTES,
        "display_intervals": DISPLAY_INTERVALS,
        "forecast_csv_full10min": os.path.join(t0_tag, "full10min.csv"),
    }
    state_path = os.path.join(root_output_dir, "last_run_state.json")
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)
    say_ok(f"State run disimpan: {state_path} (dipakai otomatis oleh 07 & 08)")

    gap()
    banner("SELESAI")
    n_with_actual = forecast_df["actual_tbb13"].notna().sum()
    say_info(f"Baris dengan data aktual tersedia (buat panel pembanding): {n_with_actual}/{len(forecast_df)}")
    say_info("CATATAN: timestamp di data ini UTC. Kalau mau ditampilkan sebagai WIB, tambah 7 jam saat visualisasi.")
    say_info("Lanjut ke Tahap 6: visualisasi 6-panel (kayak GIF referensi) dari file forecast_*.csv ini.")


if __name__ == "__main__":
    main()