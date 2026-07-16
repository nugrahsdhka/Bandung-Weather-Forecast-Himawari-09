# 06_run_inference.py
#
# Tahap 5: Inference / prediksi.
# Pilih model terbaik (otomatis, berdasarkan hasil Tahap 4), jalankan
# forecast recursive 3 jam penuh (18 langkah x 10 menit) dari satu
# timestamp awal, lalu simpan hasilnya -- baik versi lengkap (resolusi
# 10 menit) maupun versi yang sudah disaring untuk tampilan 30/60 menit.
#
# GANTI T0_STR di bawah sesuai timestamp yang mau dijadikan titik awal
# forecast. Harus timestamp yang ada di features_10min_ar.csv (base_time).
# Timestamp di data ini dalam UTC (bukan WIB) -- lihat catatan di akhir output.
#
# Jalankan dari folder scripts/:
#   python 06_run_inference.py

import os
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

# ==== GANTI SESUAI KEBUTUHAN ====
T0_STR = "2026-07-05 10:00:00"   # timestamp awal forecast (UTC), harus ada di features_10min_ar.csv
BASE_INTERVAL_MINUTES = 10
HORIZON_MINUTES = 180
MODEL_NAMES = ["svr", "xgboost", "lightgbm", "catboost"]
DISPLAY_INTERVALS = [10, 30, 60]
# =================================


def main():
    cfg = load_config()
    dataset_dir = os.path.join(cfg.PROJECT_ROOT, "dataset")
    models_dir = os.path.join(cfg.PROJECT_ROOT, "models")
    output_dir = os.path.join(cfg.PROJECT_ROOT, "forecast_output")
    os.makedirs(output_dir, exist_ok=True)

    banner("INFERENCE - FORECAST RECURSIVE 3 JAM")

    say_info("Memilih model terbaik berdasarkan hasil Tahap 4 (recursive_evaluation.csv) ...")
    best_model_name, avg_mae_table = select_best_model(models_dir, BASE_INTERVAL_MINUTES, MODEL_NAMES)
    say_ok(f"Model terpilih: {best_model_name}")
    print(avg_mae_table.to_string())
    hr()

    interval_dir = os.path.join(models_dir, f"{BASE_INTERVAL_MINUTES}min")
    model = joblib.load(os.path.join(interval_dir, f"{best_model_name}.joblib"))
    scaler_path = os.path.join(interval_dir, f"{best_model_name}_scaler.joblib")
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

    say_info(f"Memuat dataset 10 menit untuk titik awal & ground-truth lookup ...")
    df10 = load_ar_dataset(os.path.join(dataset_dir, "features_10min_ar.csv"))
    lookup = build_ground_truth_lookup(df10)
    hr()

    t0 = pd.to_datetime(T0_STR)
    say_info(f"Titik awal forecast: {t0} (UTC)")

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

    full_path = os.path.join(output_dir, f"forecast_{t0.strftime('%Y%m%d_%H%M')}_full10min.csv")
    forecast_df.to_csv(full_path, index=False)
    say_ok(f"Disimpan (resolusi penuh 10 menit): {full_path}")

    for disp_interval in DISPLAY_INTERVALS:
        if disp_interval == BASE_INTERVAL_MINUTES:
            continue
        sub_df = filter_forecast_by_interval(forecast_df, disp_interval, BASE_INTERVAL_MINUTES)
        sub_path = os.path.join(
            output_dir, f"forecast_{t0.strftime('%Y%m%d_%H%M')}_display{disp_interval}min.csv"
        )
        sub_df.to_csv(sub_path, index=False)
        say_ok(f"Disimpan (tampilan {disp_interval} menit, {len(sub_df)} baris): {sub_path}")

    gap()
    banner("SELESAI")
    n_with_actual = forecast_df["actual_tbb13"].notna().sum()
    say_info(f"Baris dengan data aktual tersedia (buat panel pembanding): {n_with_actual}/{len(forecast_df)}")
    say_info("CATATAN: timestamp di data ini UTC. Kalau mau ditampilkan sebagai WIB, tambah 7 jam saat visualisasi.")
    say_info("Lanjut ke Tahap 6: visualisasi 6-panel (kayak GIF referensi) dari file forecast_*.csv ini.")


if __name__ == "__main__":
    main()