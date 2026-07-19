# pipeline/inference.py

from datetime import timedelta

import numpy as np
import pandas as pd

from pipeline.model_training import FEATURE_COLUMNS
from pipeline.time_features import cyclical_time_features
from pipeline.ground_truth import get_actual
from pipeline.delta_features import compute_delta_dict
from pipeline.spatial_features import neighbor_feature_dict, anchor_feature_dict


COLLAPSE_RATIO_THRESHOLD = 0.5  # sama dengan yang dipakai di 05_recursive_evaluation.py


def select_best_direct_model(models_dir, horizon_step, model_names):
    """
    Fix #2: pilih model direct terbaik (MAE terendah) UNTUK SATU horizon
    tertentu, dibaca dari models/direct/direct_training_summary.csv (hasil
    04b_train_direct_models.py). Model direct tidak perlu difilter
    collapse_ratio seperti model recursive (fix #1) -- karena tiap horizon
    diprediksi langsung dari observasi asli, tidak ada rantai yang bisa
    kolaps sepanjang forecast.
    """
    import os

    summary_path = os.path.join(models_dir, "direct_training_summary.csv")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(
            f"{summary_path} tidak ditemukan. Jalankan 04b_train_direct_models.py dulu."
        )

    df = pd.read_csv(summary_path)
    df = df[(df["horizon_step"] == horizon_step) & (df["model"].isin(model_names))]
    if df.empty:
        raise ValueError(f"Tidak ada data training untuk horizon_step={horizon_step}.")

    return df.sort_values("mae").iloc[0]["model"]


def select_best_model(models_dir, interval_minutes, model_names, collapse_threshold=COLLAPSE_RATIO_THRESHOLD):
    """
    Pilih model terbaik berdasarkan rata-rata MAE di SELURUH langkah
    recursive, dibaca dari models/recursive_evaluation.csv (hasil Tahap 5) --
    TAPI (fix #1) model yang rata-rata collapse_ratio-nya di bawah
    `collapse_threshold` (artinya prediksinya kolaps jadi flat/rata-rata,
    kehilangan variasi spasial) disingkirkan dulu dari kandidat, walau
    MAE-nya kelihatan rendah. MAE rendah + collapse tinggi = model "curang"
    (menang karena flatline mendekati rata-rata, bukan karena akurat).

    Kalau SEMUA model kolaps, tetap kembalikan yang MAE-nya terendah (fallback)
    tapi beri tahu lewat kolom 'status' di tabel yang dikembalikan.
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

    if "collapse_ratio" not in df.columns:
        # Kompatibilitas mundur kalau CSV lama (sebelum fix #1) belum punya kolom ini.
        avg_mae = df.groupby("model")["mae"].mean().sort_values()
        return avg_mae.index[0], avg_mae

    agg = df.groupby("model").agg(mae=("mae", "mean"), collapse_ratio=("collapse_ratio", "mean")).sort_values("mae")
    agg["status"] = np.where(agg["collapse_ratio"] < collapse_threshold, "KOLAPS", "ok")

    non_collapsed = agg[agg["status"] == "ok"]
    if not non_collapsed.empty:
        best_model = non_collapsed["mae"].idxmin()
    else:
        best_model = agg["mae"].idxmin()  # fallback: semua kolaps, tetap pilih MAE terendah

    return best_model, agg


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
            # Anchor: nilai observasi asli terakhir sebelum recursive dimulai
            # -- tidak pernah berubah sepanjang chain (lihat fix #4).
            "last_real_obs": row.tbb_13_t,
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
    last_real_obs = {k: v["last_real_obs"] for k, v in windows.items()}
    pixel_keys = list(state.keys())
    cur_ref = t0

    for step in range(1, n_steps + 1):
        tf = cyclical_time_features(cur_ref)
        target_time = cur_ref + delta

        # Grid nilai "t" saat ini untuk SEMUA piksel (real di step 1, hasil
        # prediksi step sebelumnya di step > 1) -- dipakai fitur tetangga,
        # sama persis dengan cara dihitung saat training (fix #4).
        current_grid = {(pr, pc): state[(pr, pc)][2] for (pr, pc) in pixel_keys}
        minutes_since_real = (step - 1) * interval_minutes

        X_rows = []
        for (pr, pc) in pixel_keys:
            lat, lon = windows[(pr, pc)]["lat"], windows[(pr, pc)]["lon"]
            w = state[(pr, pc)]
            X_rows.append({
                "lat": lat, "lon": lon,
                **tf,
                "tbb_13_t": w[2], "tbb_13_tm1": w[1], "tbb_13_tm2": w[0],
                **compute_delta_dict(w[2], w[1], w[0]),
                **neighbor_feature_dict(current_grid, pr, pc),
                **anchor_feature_dict(last_real_obs[(pr, pc)], minutes_since_real),
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


def run_recursive_forecast_ensemble(models_and_scalers, t0, windows, n_steps=18, interval_minutes=10, lookup=None):
    """
    Fix #5: versi ensemble dari run_recursive_forecast. `models_and_scalers`
    adalah list [(model, scaler_or_None), ...] (biasanya xgboost+lightgbm+
    catboost, urutan sesuai cfg.MODEL_NAMES).

    PENTING: rata-rata dihitung TIAP STEP (bukan menjalankan tiap model
    penuh 18 langkah lalu dirata-rata di akhir) -- supaya error tiap model
    yang polanya beda-beda (lightgbm cepat flatten, catboost drift naik,
    dst.) saling meng-cancel SEBELUM ikut jadi input recursive langkah
    berikutnya, bukan cuma di tampilan akhir.
    """
    delta = timedelta(minutes=interval_minutes)
    records = []

    state = {k: list(v["window"]) for k, v in windows.items()}
    last_real_obs = {k: v["last_real_obs"] for k, v in windows.items()}
    pixel_keys = list(state.keys())
    cur_ref = t0

    for step in range(1, n_steps + 1):
        tf = cyclical_time_features(cur_ref)
        target_time = cur_ref + delta

        current_grid = {(pr, pc): state[(pr, pc)][2] for (pr, pc) in pixel_keys}
        minutes_since_real = (step - 1) * interval_minutes

        X_rows = []
        for (pr, pc) in pixel_keys:
            lat, lon = windows[(pr, pc)]["lat"], windows[(pr, pc)]["lon"]
            w = state[(pr, pc)]
            X_rows.append({
                "lat": lat, "lon": lon,
                **tf,
                "tbb_13_t": w[2], "tbb_13_tm1": w[1], "tbb_13_tm2": w[0],
                **compute_delta_dict(w[2], w[1], w[0]),
                **neighbor_feature_dict(current_grid, pr, pc),
                **anchor_feature_dict(last_real_obs[(pr, pc)], minutes_since_real),
            })
        X = pd.DataFrame(X_rows)[FEATURE_COLUMNS]

        # Prediksi tiap model, lalu rata-ratakan per piksel.
        all_preds = []
        for model, scaler in models_and_scalers:
            X_eval = scaler.transform(X) if scaler is not None else X
            all_preds.append(np.asarray(model.predict(X_eval), dtype=float))
        y_preds = np.mean(np.vstack(all_preds), axis=0)

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


def annotate_reliability(forecast_df, models_dir, interval_minutes, model_name, collapse_threshold=COLLAPSE_RATIO_THRESHOLD):
    """
    Fix #6: tempelkan kolom `mae_expected` (perkiraan error di step itu,
    dari hasil Tahap 5) dan `reliable` (False kalau collapse_ratio di step
    itu sudah di bawah ambang -- artinya sinyal forecast sudah "habis" dan
    sebaiknya tidak dipakai/ditampilkan sebagai angka pasti) ke forecast_df.

    Kalau model_name == "ensemble" (hasil fix #5), dipakai rata-rata metrik
    antar model sebagai perkiraan (ensemble tidak punya baris sendiri di
    recursive_evaluation.csv).
    """
    import os

    eval_path = os.path.join(models_dir, "recursive_evaluation.csv")
    if not os.path.exists(eval_path):
        forecast_df = forecast_df.copy()
        forecast_df["mae_expected"] = np.nan
        forecast_df["reliable"] = True
        return forecast_df

    df = pd.read_csv(eval_path)
    df = df[df["interval_menit"] == interval_minutes]
    if model_name != "ensemble":
        df = df[df["model"] == model_name]

    if df.empty or "collapse_ratio" not in df.columns:
        forecast_df = forecast_df.copy()
        forecast_df["mae_expected"] = np.nan
        forecast_df["reliable"] = True
        return forecast_df

    per_step = df.groupby("langkah_ke").agg(mae_expected=("mae", "mean"), collapse_ratio=("collapse_ratio", "mean")).reset_index()
    per_step["reliable"] = per_step["collapse_ratio"] >= collapse_threshold
    per_step = per_step.rename(columns={"langkah_ke": "step"})

    merged = forecast_df.merge(per_step[["step", "mae_expected", "reliable"]], on="step", how="left")
    merged["reliable"] = merged["reliable"].fillna(True)
    return merged


def run_direct_then_recursive_forecast(
    direct_models_by_h, recursive_model, recursive_scaler, t0, windows,
    direct_horizon_steps, n_steps, interval_minutes=10, lookup=None,
):
    """
    Fix #2 (kompromi): untuk step 1..max(direct_horizon_steps), prediksi
    memakai model DIRECT per-horizon (semuanya dari fitur observasi ASLI di
    t0 -- tidak ada compounding error sama sekali di rentang ini). Untuk
    step setelah itu (biasanya di luar ~90 menit, di mana sinyal spasial
    sudah lemah dan melatih direct-model tambahan kurang sepadan), pipeline
    otomatis menyambung dengan recursive_model biasa, memakai 3 prediksi
    direct TERAKHIR sebagai window awal rantai recursive-nya -- supaya tidak
    ada lompatan/diskontinuitas antara fase direct & fase recursive.

    direct_models_by_h: dict {h: (model, scaler_or_None)}, h = kelipatan
    interval_minutes (mis. h=9 -> +90 menit kalau interval_minutes=10).
    """
    delta = timedelta(minutes=interval_minutes)
    records = []

    last_real_obs = {k: v["last_real_obs"] for k, v in windows.items()}
    pixel_keys = list(windows.keys())

    # ---- Fase direct: SEMUA horizon direct memakai fitur t0 yang SAMA ----
    tf0 = cyclical_time_features(t0)
    current_grid0 = {(pr, pc): windows[(pr, pc)]["window"][2] for (pr, pc) in pixel_keys}
    X0_rows = []
    for (pr, pc) in pixel_keys:
        lat, lon = windows[(pr, pc)]["lat"], windows[(pr, pc)]["lon"]
        w = windows[(pr, pc)]["window"]
        X0_rows.append({
            "lat": lat, "lon": lon,
            **tf0,
            "tbb_13_t": w[2], "tbb_13_tm1": w[1], "tbb_13_tm2": w[0],
            **compute_delta_dict(w[2], w[1], w[0]),
            **neighbor_feature_dict(current_grid0, pr, pc),
            **anchor_feature_dict(last_real_obs[(pr, pc)], 0),
        })
    X0 = pd.DataFrame(X0_rows)[FEATURE_COLUMNS]

    preds_by_step = {pk: {} for pk in pixel_keys}
    sorted_h = sorted(direct_horizon_steps)
    for h in sorted_h:
        if h not in direct_models_by_h:
            continue
        model_h, scaler_h = direct_models_by_h[h]
        X_eval = scaler_h.transform(X0) if scaler_h is not None else X0
        y_preds = model_h.predict(X_eval)
        target_time = t0 + h * delta

        for (pr, pc), y_pred in zip(pixel_keys, y_preds):
            actual_val = get_actual(lookup, pr, pc, target_time) if lookup is not None else None
            records.append({
                "step": h, "forecast_time": target_time,
                "pixel_row": pr, "pixel_col": pc,
                "lat": windows[(pr, pc)]["lat"], "lon": windows[(pr, pc)]["lon"],
                "predicted_tbb13": float(y_pred), "actual_tbb13": actual_val,
                "source": "direct",
            })
            preds_by_step[(pr, pc)][h] = float(y_pred)

    max_direct = max(sorted_h) if sorted_h else 0
    if n_steps <= max_direct or max_direct < 2:
        return pd.DataFrame(records)

    # ---- Fase recursive lanjutan: sambung dari 3 prediksi direct terakhir ----
    state = {}
    for pk in pixel_keys:
        state[pk] = [
            preds_by_step[pk].get(max_direct - 2, last_real_obs[pk]),
            preds_by_step[pk].get(max_direct - 1, last_real_obs[pk]),
            preds_by_step[pk][max_direct],
        ]
    cur_ref = t0 + max_direct * delta

    for step in range(max_direct + 1, n_steps + 1):
        tf = cyclical_time_features(cur_ref)
        target_time = cur_ref + delta
        current_grid = {pk: state[pk][2] for pk in pixel_keys}
        minutes_since_real = (step - 1) * interval_minutes

        X_rows = []
        for (pr, pc) in pixel_keys:
            lat, lon = windows[(pr, pc)]["lat"], windows[(pr, pc)]["lon"]
            w = state[(pr, pc)]
            X_rows.append({
                "lat": lat, "lon": lon,
                **tf,
                "tbb_13_t": w[2], "tbb_13_tm1": w[1], "tbb_13_tm2": w[0],
                **compute_delta_dict(w[2], w[1], w[0]),
                **neighbor_feature_dict(current_grid, pr, pc),
                **anchor_feature_dict(last_real_obs[(pr, pc)], minutes_since_real),
            })
        X = pd.DataFrame(X_rows)[FEATURE_COLUMNS]
        X_eval = recursive_scaler.transform(X) if recursive_scaler is not None else X
        y_preds = recursive_model.predict(X_eval)

        for (pr, pc), y_pred in zip(pixel_keys, y_preds):
            actual_val = get_actual(lookup, pr, pc, target_time) if lookup is not None else None
            records.append({
                "step": step, "forecast_time": target_time,
                "pixel_row": pr, "pixel_col": pc,
                "lat": windows[(pr, pc)]["lat"], "lon": windows[(pr, pc)]["lon"],
                "predicted_tbb13": float(y_pred), "actual_tbb13": actual_val,
                "source": "recursive_continuation",
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