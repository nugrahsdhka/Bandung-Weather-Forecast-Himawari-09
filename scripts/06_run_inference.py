# 06_run_inference.py

import os
import json
import joblib
import pandas as pd

from ui.terminal_display import hr, gap, banner, say_info, say_ok, say_error
from pipeline.config import load_config
from pipeline.model_training import load_ar_dataset
from pipeline.ground_truth import build_ground_truth_lookup
from pipeline.inference import (
    select_best_model,
    select_best_direct_model,
    get_initial_windows,
    run_recursive_forecast,
    run_recursive_forecast_ensemble,
    run_direct_then_recursive_forecast,
    annotate_reliability,
    filter_forecast_by_interval,
)

# ==== GANTI SESUAI KEBUTUHAN ====
T0_STR = "2026-01-03 10:10:00"   # None = otomatis pakai base_time TERBARU di features_10min_ar.csv
BASE_INTERVAL_MINUTES = 10
HORIZON_MINUTES = 180
DISPLAY_INTERVALS = [10, 30, 60]

MODE = "hybrid_direct"   # "single" | "ensemble" | "hybrid_direct"
# =================================


def load_single_model(models_dir, interval_minutes, model_name):
    interval_dir = os.path.join(models_dir, f"{interval_minutes}min")
    model = joblib.load(os.path.join(interval_dir, f"{model_name}.joblib"))
    scaler_path = os.path.join(interval_dir, f"{model_name}_scaler.joblib")
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    return model, scaler


def load_direct_model(models_dir, horizon_step, model_name):
    h_dir = os.path.join(models_dir, "direct", f"h{horizon_step}")
    model = joblib.load(os.path.join(h_dir, f"{model_name}.joblib"))
    scaler_path = os.path.join(h_dir, f"{model_name}_scaler.joblib")
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    return model, scaler


def main():
    cfg = load_config()
    dataset_dir = os.path.join(cfg.PROJECT_ROOT, "dataset")
    models_dir = os.path.join(cfg.PROJECT_ROOT, "models")
    root_output_dir = os.path.join(cfg.PROJECT_ROOT, "forecast_output")
    os.makedirs(root_output_dir, exist_ok=True)

    banner(f"INFERENCE - FORECAST RECURSIVE 3 JAM (mode: {MODE})")

    say_info("Memuat dataset 10 menit untuk titik awal & ground-truth lookup ...")
    df10 = load_ar_dataset(os.path.join(dataset_dir, "features_10min_ar.csv"))
    lookup = build_ground_truth_lookup(df10)
    hr()

    if MODE == "ensemble":
        say_info("Mode ENSEMBLE aktif (fix #5): memuat semua model di cfg.MODEL_NAMES ...")
        models_and_scalers = [load_single_model(models_dir, BASE_INTERVAL_MINUTES, m) for m in cfg.MODEL_NAMES]
        model_name_tag = "ensemble"
        say_ok(f"Model diikutkan: {', '.join(cfg.MODEL_NAMES)}")

    elif MODE == "hybrid_direct":
        say_info("Mode HYBRID_DIRECT aktif (fix #2): memilih model direct terbaik per horizon ...")
        direct_models_by_h = {}
        for h in cfg.DIRECT_HORIZON_STEPS:
            best_direct_name = select_best_direct_model(os.path.join(models_dir, "direct"), h, cfg.MODEL_NAMES)
            direct_models_by_h[h] = load_direct_model(models_dir, h, best_direct_name)
            say_ok(f"  horizon +{h * BASE_INTERVAL_MINUTES:>3}min -> {best_direct_name}")

        say_info("Memilih model RECURSIVE terbaik (fix #1) untuk menyambung step setelah fase direct ...")
        best_model_name, model_table = select_best_model(models_dir, BASE_INTERVAL_MINUTES, cfg.MODEL_NAMES)
        say_ok(f"Model recursive penyambung: {best_model_name}")
        print(model_table.to_string())
        recursive_model, recursive_scaler = load_single_model(models_dir, BASE_INTERVAL_MINUTES, best_model_name)
        model_name_tag = f"hybrid_direct+{best_model_name}"

    else:  # "single"
        say_info("Memilih model terbaik (fix #1: MAE + filter collapse_ratio) berdasarkan Tahap 5 ...")
        best_model_name, model_table = select_best_model(models_dir, BASE_INTERVAL_MINUTES, cfg.MODEL_NAMES)
        say_ok(f"Model terpilih: {best_model_name}")
        print(model_table.to_string())
        model, scaler = load_single_model(models_dir, BASE_INTERVAL_MINUTES, best_model_name)
        model_name_tag = best_model_name
    hr()

    if T0_STR is None:
        t0 = df10["base_time"].max()
        say_info(f"T0_STR tidak diisi -> otomatis pakai base_time TERBARU: {t0} (UTC)")
    else:
        t0 = pd.to_datetime(T0_STR)
        say_info(f"Titik awal forecast: {t0} (UTC)")

    t0_tag = t0.strftime("%Y%m%d_%H%M") + f"_{model_name_tag}"
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
    say_info(f"Menjalankan forecast: {n_steps} langkah x {BASE_INTERVAL_MINUTES} menit ...")

    if MODE == "ensemble":
        forecast_df = run_recursive_forecast_ensemble(
            models_and_scalers, t0, windows,
            n_steps=n_steps, interval_minutes=BASE_INTERVAL_MINUTES, lookup=lookup,
        )
    elif MODE == "hybrid_direct":
        forecast_df = run_direct_then_recursive_forecast(
            direct_models_by_h, recursive_model, recursive_scaler, t0, windows,
            direct_horizon_steps=cfg.DIRECT_HORIZON_STEPS, n_steps=n_steps,
            interval_minutes=BASE_INTERVAL_MINUTES, lookup=lookup,
        )
        n_direct = (forecast_df["source"] == "direct").sum()
        n_cont = (forecast_df["source"] == "recursive_continuation").sum()
        say_ok(f"Fase direct: {n_direct} baris  |  Fase recursive lanjutan: {n_cont} baris")
    else:
        forecast_df = run_recursive_forecast(
            model, scaler, t0, windows,
            n_steps=n_steps, interval_minutes=BASE_INTERVAL_MINUTES, lookup=lookup,
        )
    say_ok(f"Forecast selesai: {len(forecast_df)} baris ({n_steps} langkah x {len(windows)} pixel)")

    # Fix #6: tandai step mana yang masih "reliable" berdasarkan collapse_ratio di Tahap 5.
    # Untuk hybrid_direct, dipakai metrik recursive_model (yang dipakai fase sambungan) --
    # fase direct-nya sendiri tidak butuh flag ini karena tidak ada compounding error.
    reliability_model_tag = best_model_name if MODE == "hybrid_direct" else model_name_tag
    forecast_df = annotate_reliability(forecast_df, models_dir, BASE_INTERVAL_MINUTES, reliability_model_tag)
    if MODE == "hybrid_direct":
        # Step-step fase direct selalu ditandai reliable (tidak ada compounding error).
        forecast_df.loc[forecast_df["source"] == "direct", "reliable"] = True

    n_unreliable = (~forecast_df["reliable"]).sum()
    if n_unreliable > 0:
        first_unreliable_step = int(forecast_df.loc[~forecast_df["reliable"], "step"].min())
        say_info(
            f"PERINGATAN: mulai step {first_unreliable_step} (+{first_unreliable_step * BASE_INTERVAL_MINUTES}min), "
            "forecast ditandai 'reliable=False' karena sinyal sudah terbukti kolaps di evaluasi Tahap 5. "
            "Gunakan kolom 'reliable' & 'mae_expected' saat visualisasi/penyajian."
        )
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
        "mode": MODE,
        "model_name": model_name_tag,
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
    say_info("Lanjut ke Tahap 7-8: visualisasi 6-panel & render animasi dari file forecast_*.csv ini.")


if __name__ == "__main__":
    main()
