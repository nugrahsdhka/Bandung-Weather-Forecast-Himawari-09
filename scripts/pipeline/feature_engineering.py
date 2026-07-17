# ./scripts/pipeline/feature_engineering.py
# Mengimplementasikan proses feature engineering dengan mengubah rangkaian frame CTT menjadi dataset supervised learning berbasis piksel untuk berbagai horizon prediksi.

import os
from datetime import timedelta

import numpy as np
import pandas as pd
import xarray as xr

from ui.terminal_display import say_info, say_ok, say_error, say_skip, make_total_progress_bar
from pipeline.netcdf_tools import extract_time_from_filename

CHANNELS = [f"tbb_{i:02d}" for i in range(7, 17)]  # tbb_07 ... tbb_16
TARGET_VAR = "tbb_13"


def scan_timestamp_index(final_base_dir):
    """Telusuri data_bandung/, kembalikan dict {timestamp: path_file}."""
    index = {}
    for root, _dirs, files in os.walk(final_base_dir):
        for f in files:
            if not f.endswith(".nc"):
                continue
            ts = extract_time_from_filename(f)
            if ts is None:
                continue
            index[ts] = os.path.join(root, f)
    return index


def get_aligned_candidates(all_timestamps, interval_minutes):
    """Kandidat titik dasar (t) yang selaras dengan interval, dari daftar timestamp yang tersedia (tanpa membuka file)."""
    return sorted(ts for ts in all_timestamps if (ts.minute % interval_minutes) == 0)


def required_timestamps_for_candidates(candidates, interval_minutes, lag_count=3):
    """Kumpulan timestamp (lag + target) yang dibutuhkan untuk membangun baris pada kandidat yang diberikan."""
    delta = timedelta(minutes=interval_minutes)
    needed = set()
    for t in candidates:
        for k in range(lag_count):
            needed.add(t - k * delta)
        needed.add(t + delta)
    return needed


def preload_frames(timestamp_index, channels=CHANNELS, only_timestamps=None):
    """Memuat file .nc ke memori dan memvalidasi konsistensi grid latitude-longitude.

    Kalau `only_timestamps` diberikan, hanya file dengan timestamp di dalamnya yang
    dibuka (dipakai untuk mode incremental agar tidak perlu load ulang seluruh arsip).
    """
    frames = {}
    lat_ref, lon_ref = None, None
    n_skipped = 0

    items = timestamp_index.items()
    if only_timestamps is not None:
        items = [(ts, path) for ts, path in items if ts in only_timestamps]

    files_sorted = sorted(items)
    say_info(f"Memuat {len(files_sorted)} file .nc ke memori (kanal tbb_07-tbb_16 saja) ...")

    bar = make_total_progress_bar(files_sorted)
    for ts, path in bar:
        try:
            with xr.open_dataset(path) as ds:
                if lat_ref is None:
                    lat_ref = ds.latitude.values.copy()
                    lon_ref = ds.longitude.values.copy()
                elif (
                    ds.latitude.values.shape != lat_ref.shape
                    or ds.longitude.values.shape != lon_ref.shape
                ):
                    say_skip(f"Grid tidak konsisten, dilewati: {os.path.basename(path)}")
                    n_skipped += 1
                    continue

                data = {}
                ok = True
                for ch in channels:
                    if ch not in ds.data_vars:
                        ok = False
                        break
                    data[ch] = ds[ch].values.astype("float32")
                if not ok:
                    n_skipped += 1
                    continue

                frames[ts] = data
        except Exception as e:
            say_error(f"Gagal membuka {os.path.basename(path)}: {e}")
            n_skipped += 1

    say_ok(f"Berhasil dimuat: {len(frames)} frame  |  dilewati: {n_skipped}")
    return frames, lat_ref, lon_ref


def _cyclical_time_features(ts):
    """Encoding siklikal untuk jam-dalam-hari & hari-dalam-tahun."""
    hour_frac = ts.hour + ts.minute / 60.0
    doy = ts.timetuple().tm_yday
    return {
        "hour_sin": np.sin(2 * np.pi * hour_frac / 24.0),
        "hour_cos": np.cos(2 * np.pi * hour_frac / 24.0),
        "doy_sin": np.sin(2 * np.pi * doy / 365.25),
        "doy_cos": np.cos(2 * np.pi * doy / 365.25),
    }


def build_interval_dataset(
    frames, lat, lon, interval_minutes, lag_count=3, channels=CHANNELS, target_var=TARGET_VAR,
    candidates=None,
):
    """Membangun dataset supervised per-pixel untuk forecasting satu langkah dengan hanya menggunakan timestamp yang lengkap.

    `candidates` opsional: daftar titik dasar (t) yang mau diproses. Kalau None, dihitung
    dari seluruh timestamp yang ada di `frames` (perilaku lama / full rebuild). Dipakai
    mode incremental untuk membatasi hanya ke base_time yang belum ada di dataset lama.
    """
    delta = timedelta(minutes=interval_minutes)

    if candidates is None:
        all_ts = sorted(frames.keys())
        # Hanya timestamp yang selaras dengan interval yang jadi kandidat titik dasar (t)
        candidates = [ts for ts in all_ts if (ts.minute % interval_minutes) == 0]

    n_lat, n_lon = lat.shape[0], lon.shape[0]
    rows = []

    say_info(f"[interval {interval_minutes} menit] Kandidat titik dasar: {len(candidates)}")

    for t in candidates:
        lag_times = [t - k * delta for k in range(lag_count)]
        target_time = t + delta

        if target_time not in frames:
            continue
        if any(lt not in frames for lt in lag_times):
            continue

        lag_frames = [frames[lt] for lt in lag_times]
        target_frame = frames[target_time]
        time_feats = _cyclical_time_features(t)

        for i in range(n_lat):
            for j in range(n_lon):
                row = {
                    "base_time": t,
                    "target_time": target_time,
                    "pixel_row": i,
                    "pixel_col": j,
                    "lat": float(lat[i]),
                    "lon": float(lon[j]),
                    **time_feats,
                }
                for ch in channels:
                    for k, lf in enumerate(lag_frames):
                        suffix = "_t" if k == 0 else f"_tm{k}"
                        row[f"{ch}{suffix}"] = float(lf[ch][i, j])
                row[f"target_{target_var}"] = float(target_frame[target_var][i, j])
                rows.append(row)

    df = pd.DataFrame(rows)
    say_ok(f"[interval {interval_minutes} menit] Dataset jadi: {len(df)} baris, {df.shape[1]} kolom")
    return df