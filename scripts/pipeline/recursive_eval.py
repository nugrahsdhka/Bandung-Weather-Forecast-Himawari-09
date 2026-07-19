# ./scripts/pipeline/recursive_eval.py
# Evaluasi recursive multi-step forecasting -- direfactor (fix #1 & #4) untuk
# mensimulasikan SEMUA piksel bersamaan per titik awal (t0), persis seperti
# yang terjadi di produksi (pipeline/inference.py run_recursive_forecast).
# Ini penting karena fitur tetangga (fix #4) butuh nilai piksel lain pada
# step yang sama, dan metrik anti-collapse (fix #1) butuh sebaran prediksi
# ANTAR piksel pada step yang sama -- keduanya tidak bisa dihitung benar
# kalau tiap piksel disimulasikan independen dengan t0 yang berbeda-beda.

from datetime import timedelta

import numpy as np

from pipeline.time_features import cyclical_time_features
from pipeline.ground_truth import build_ground_truth_lookup, get_actual
from pipeline.inference import get_initial_windows, run_recursive_forecast

__all__ = [
    "cyclical_time_features", "build_ground_truth_lookup", "get_actual",
    "spatial_collapse_ratio", "spatial_correlation",
    "select_valid_t0", "evaluate_recursive_at_t0",
]


def spatial_collapse_ratio(preds, actuals):
    """
    std(prediksi) / std(aktual) ANTAR piksel pada satu step yang sama.
    Mendekati 0 berarti model kolaps jadi rata dan kehilangan variasi
    spasial -- MAE saja tidak bisa mendeteksi ini (lihat fix #1).
    """
    preds = np.asarray(preds, dtype=float)
    actuals = np.asarray(actuals, dtype=float)
    return float(np.std(preds) / (np.std(actuals) + 1e-6))


def spatial_correlation(preds, actuals):
    """Korelasi spasial antara prediksi & aktual antar piksel pada satu step."""
    preds = np.asarray(preds, dtype=float)
    actuals = np.asarray(actuals, dtype=float)
    if len(preds) < 2 or np.std(preds) < 1e-9 or np.std(actuals) < 1e-9:
        return 0.0
    return float(np.corrcoef(preds, actuals)[0, 1])


def select_valid_t0(df, interval_minutes, lookup, n_steps, max_points):
    """
    Pilih titik awal t0 (bukan per-pixel lagi) dari test set yang: semua
    piksel di base_time itu tersedia, DAN ground-truth aktualnya lengkap
    untuk semua piksel sampai n_steps ke depan (dibutuhkan untuk metrik
    spasial per step, termasuk step terakhir).
    """
    delta = timedelta(minutes=interval_minutes)
    t0_candidates = sorted(df["base_time"].unique())

    valid_t0 = []
    for t0 in t0_candidates:
        pixels = (
            df[df["base_time"] == t0][["pixel_row", "pixel_col"]]
            .drop_duplicates()
            .itertuples(index=False)
        )
        pixels = list(pixels)
        if not pixels:
            continue

        ok = True
        for row in pixels:
            for k in range(1, n_steps + 1):
                if get_actual(lookup, row.pixel_row, row.pixel_col, t0 + k * delta) is None:
                    ok = False
                    break
            if not ok:
                break

        if ok:
            valid_t0.append(t0)
        if len(valid_t0) >= max_points:
            break

    return valid_t0


def evaluate_recursive_at_t0(model, scaler, df, t0, interval_minutes, n_steps, lookup):
    """
    Jalankan recursive forecast joint (semua piksel) dari satu t0, lalu
    kembalikan list dict per step: {langkah_ke, preds, actuals} (list nilai
    antar piksel), siap dipakai menghitung MAE/collapse-ratio/korelasi.
    """
    windows = get_initial_windows(df, t0)
    forecast_df = run_recursive_forecast(
        model, scaler, t0, windows, n_steps=n_steps,
        interval_minutes=interval_minutes, lookup=lookup,
    )

    per_step = []
    for step, group in forecast_df.groupby("step"):
        valid = group.dropna(subset=["actual_tbb13"])
        if valid.empty:
            continue
        per_step.append({
            "langkah_ke": int(step),
            "preds": valid["predicted_tbb13"].to_numpy(),
            "actuals": valid["actual_tbb13"].to_numpy(),
        })
    return per_step
