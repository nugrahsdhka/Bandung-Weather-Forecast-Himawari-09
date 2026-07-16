# ./scripts/pipeline/recursive_eval.py

from datetime import timedelta

import numpy as np
import pandas as pd

from pipeline.model_training import FEATURE_COLUMNS


def cyclical_time_features(ts):
    """Encoding siklikal untuk jam-dalam-hari & hari-dalam-tahun (harus identik dengan yang dipakai saat feature engineering)."""
    hour_frac = ts.hour + ts.minute / 60.0
    doy = ts.timetuple().tm_yday
    return {
        "hour_sin": np.sin(2 * np.pi * hour_frac / 24.0),
        "hour_cos": np.cos(2 * np.pi * hour_frac / 24.0),
        "doy_sin": np.sin(2 * np.pi * doy / 365.25),
        "doy_cos": np.cos(2 * np.pi * doy / 365.25),
    }


def build_ground_truth_lookup(df10):
    """
    Bangun lookup {(pixel_row, pixel_col, timestamp): nilai_aktual_tbb_13}
    dari dataset 10 menit (resolusi native), dipakai untuk membandingkan
    hasil recursive forecast terhadap data aktual di SEMUA interval
    (10/30/60 menit sama-sama kelipatan 10 menit).
    """
    lookup = {}
    for row in df10.itertuples(index=False):
        lookup[(row.pixel_row, row.pixel_col, row.base_time)] = row.tbb_13_t
        lookup[(row.pixel_row, row.pixel_col, row.target_time)] = row.target_tbb_13
    return lookup


def get_actual(lookup, pixel_row, pixel_col, ts):
    return lookup.get((pixel_row, pixel_col, ts))


def select_start_points(test_df, interval_minutes, lookup, n_steps, max_points):
    """
    Pilih titik awal (waktu + pixel) dari test set yang punya rantai
    ground-truth LENGKAP sampai n_steps ke depan (dibutuhkan untuk bisa
    menghitung error di setiap langkah recursive, termasuk langkah terakhir).
    """
    delta = timedelta(minutes=interval_minutes)
    candidates = test_df[
        ["base_time", "pixel_row", "pixel_col", "lat", "lon", "tbb_13_t", "tbb_13_tm1", "tbb_13_tm2"]
    ].drop_duplicates(subset=["base_time", "pixel_row", "pixel_col"])

    valid = []
    for row in candidates.itertuples(index=False):
        t0 = row.base_time
        pr, pc = row.pixel_row, row.pixel_col

        ok = True
        for k in range(1, n_steps + 1):
            if get_actual(lookup, pr, pc, t0 + k * delta) is None:
                ok = False
                break

        if ok:
            valid.append({
                "t0": t0, "pixel_row": pr, "pixel_col": pc,
                "lat": row.lat, "lon": row.lon,
                "tbb_13_t": row.tbb_13_t, "tbb_13_tm1": row.tbb_13_tm1, "tbb_13_tm2": row.tbb_13_tm2,
            })
        if len(valid) >= max_points:
            break

    return valid


def recursive_predict(model, scaler, start_point, interval_minutes, n_steps, lookup):
    """
    Simulasikan forecast recursive sepanjang n_steps, mulai dari start_point.
    Tiap langkah: prediksi t+interval, lalu geser jendela [tm2, tm1, t] dan
    pakai hasil prediksi sebagai nilai 't' baru untuk langkah berikutnya.

    Return list of (langkah_ke, error_absolut) untuk langkah yang punya
    ground-truth aktual di lookup.
    """
    delta = timedelta(minutes=interval_minutes)
    lat, lon = start_point["lat"], start_point["lon"]
    pr, pc = start_point["pixel_row"], start_point["pixel_col"]

    window = [start_point["tbb_13_tm2"], start_point["tbb_13_tm1"], start_point["tbb_13_t"]]  # [tm2, tm1, t]
    cur_ref = start_point["t0"]
    results = []

    for k in range(1, n_steps + 1):
        tf = cyclical_time_features(cur_ref)
        X = pd.DataFrame([{
            "lat": lat, "lon": lon,
            **tf,
            "tbb_13_t": window[2], "tbb_13_tm1": window[1], "tbb_13_tm2": window[0],
        }])[FEATURE_COLUMNS]

        X_eval = scaler.transform(X) if scaler is not None else X
        y_pred = float(model.predict(X_eval)[0])

        target_time = cur_ref + delta
        actual_val = get_actual(lookup, pr, pc, target_time)
        if actual_val is not None:
            results.append((k, abs(y_pred - actual_val)))

        window = [window[1], window[2], y_pred]
        cur_ref = target_time

    return results