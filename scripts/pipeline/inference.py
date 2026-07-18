# pipeline/inference.py

from datetime import timedelta

import pandas as pd

from pipeline.model_training import FEATURE_COLUMNS
from pipeline.recursive_eval import cyclical_time_features, get_actual
from pipeline.delta_features import compute_delta_dict


def select_best_model(models_dir, interval_minutes, model_names):
    """
    Pilih model terbaik berdasarkan rata-rata MAE di SELURUH langkah
    recursive (bukan cuma langkah pertama atau langkah terakhir saja),
    dibaca dari models/recursive_evaluation.csv (hasil Tahap 4).
    """
    import os

    eval_path = os.path.join(models_dir, "recursive_evaluation.csv")
    if not os.path.exists(eval_path):
        raise FileNotFoundError(
            f"{eval_path} tidak ditemukan. Jalankan 05_recursive_evaluation.py dulu."
        )

    df = pd.read_csv(eval_path)
    df = df[(df["interval_menit"] == interval_minutes) & (df["model"].isin(model_names))]
    if df.empty:
        raise ValueError(f"Tidak ada data evaluasi untuk interval {interval_minutes} menit.")

    avg_mae = df.groupby("model")["mae"].mean().sort_values()
    best_model = avg_mae.index[0]
    return best_model, avg_mae


def get_initial_windows(df10, t0):
    """
    Ambil window awal [tm2, tm1, t] untuk SEMUA pixel pada waktu dasar t0,
    dari dataset 10 menit -- dipakai sebagai starting point forecasting,
    berapapun interval tampilan yang mau dihasilkan nanti (10/30/60 menit
    semuanya diturunkan dari chain 10 menit yang sama).
    """
    rows = df10[df10["base_time"] == t0]
    if rows.empty:
        raise ValueError(
            f"Tidak ada data pada base_time={t0} di dataset 10 menit. "
            "Pilih timestamp lain yang ada di features_10min_ar.csv."
        )

    windows = {}
    for row in rows.itertuples(index=False):
        windows[(row.pixel_row, row.pixel_col)] = {
            "lat": row.lat,
            "lon": row.lon,
            "window": [row.tbb_13_tm2, row.tbb_13_tm1, row.tbb_13_t],
        }
    return windows


def run_recursive_forecast(model, scaler, t0, windows, n_steps=18, interval_minutes=10, lookup=None):
    """
    Jalankan forecast recursive untuk SEMUA pixel sekaligus (batch),
    sepanjang n_steps. Kalau `lookup` diisi, nilai aktual (kalau ada)
    ikut disertakan per langkah untuk keperluan perbandingan/visualisasi.

    Return DataFrame kolom: step, forecast_time, pixel_row, pixel_col,
    lat, lon, predicted_tbb13, actual_tbb13 (NaN kalau tidak ada data aktual).
    """
    delta = timedelta(minutes=interval_minutes)
    records = []

    state = {k: list(v["window"]) for k, v in windows.items()}
    pixel_keys = list(state.keys())
    cur_ref = t0

    for step in range(1, n_steps + 1):
        tf = cyclical_time_features(cur_ref)
        target_time = cur_ref + delta

        X_rows = []
        for (pr, pc) in pixel_keys:
            lat, lon = windows[(pr, pc)]["lat"], windows[(pr, pc)]["lon"]
            w = state[(pr, pc)]
            X_rows.append({
                "lat": lat, "lon": lon,
                **tf,
                "tbb_13_t": w[2], "tbb_13_tm1": w[1], "tbb_13_tm2": w[0],
                **compute_delta_dict(w[2], w[1], w[0]),
            })
        X = pd.DataFrame(X_rows)[FEATURE_COLUMNS]
        X_eval = scaler.transform(X) if scaler is not None else X
        y_preds = model.predict(X_eval)

        for (pr, pc), y_pred in zip(pixel_keys, y_preds):
            actual_val = get_actual(lookup, pr, pc, target_time) if lookup is not None else None
            records.append({
                "step": step,
                "forecast_time": target_time,
                "pixel_row": pr, "pixel_col": pc,
                "lat": windows[(pr, pc)]["lat"], "lon": windows[(pr, pc)]["lon"],
                "predicted_tbb13": float(y_pred),
                "actual_tbb13": actual_val,
            })
            w = state[(pr, pc)]
            state[(pr, pc)] = [w[1], w[2], float(y_pred)]

        cur_ref = target_time

    return pd.DataFrame(records)


def filter_forecast_by_interval(forecast_df, display_interval_minutes, base_interval_minutes=10):
    """
    Ambil subset frame dari hasil forecast 10-menit untuk ditampilkan
    dengan granularitas yang berbeda (30 atau 60 menit), TANPA perlu
    model terpisah -- cukup pilih step yang preintervalnya kelipatan
    display_interval_minutes.
    """
    if display_interval_minutes % base_interval_minutes != 0:
        raise ValueError("display_interval_minutes harus kelipatan base_interval_minutes (10).")

    step_multiple = display_interval_minutes // base_interval_minutes
    return forecast_df[forecast_df["step"] % step_multiple == 0].copy()