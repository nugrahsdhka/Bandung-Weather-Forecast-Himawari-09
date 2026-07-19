# ./scripts/pipeline/model_training.py
# Menyediakan fungsi-fungsi untuk memuat dataset, membagi data secara kronologis, melatih model machine learning, dan mengevaluasi performa hasil prediksi.

import time

import numpy as np
import pandas as pd

from pipeline.delta_features import DELTA_COLUMNS, add_delta_features
from pipeline.spatial_features import EXTRA_COLUMNS

# Fitur delta/tren (tbb_13_delta_t, tbb_13_delta_tm1, tbb_13_accel), fitur
# tetangga (tbb13_neighbor_mean/diff), dan fitur anchor observasi asli
# (tbb_13_last_real_obs, minutes_since_last_real_obs) ditambahkan di sini --
# SATU tempat rujukan untuk seluruh pipeline (03a, 04, 05, 06), supaya
# fiturnya konsisten di semua tahap training & inference. Lihat catatan di
# pipeline/spatial_features.py.
FEATURE_COLUMNS = [
    "lat", "lon",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
    "tbb_13_t", "tbb_13_tm1", "tbb_13_tm2",
] + DELTA_COLUMNS + EXTRA_COLUMNS
TARGET_COLUMN = "target_tbb_13"


def load_ar_dataset(path):
    """Baca dataset autoregressive, pastikan base_time bertipe datetime, tambahkan fitur delta/tren."""
    df = pd.read_csv(path)
    df["base_time"] = pd.to_datetime(df["base_time"])
    df["target_time"] = pd.to_datetime(df["target_time"])
    df = add_delta_features(df)
    return df


def chronological_split(df, test_frac=0.15, time_col="base_time"):
    """Split train/test berdasarkan urutan waktu tanpa random; semua baris dengan base_time yang sama masuk ke sisi yang sama."""
    unique_times = sorted(df[time_col].unique())
    cutoff_idx = int(len(unique_times) * (1 - test_frac))
    cutoff_time = unique_times[cutoff_idx]

    train_df = df[df[time_col] < cutoff_time].reset_index(drop=True)
    test_df = df[df[time_col] >= cutoff_time].reset_index(drop=True)
    return train_df, test_df, cutoff_time


def get_feature_target(df):
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y


LAG_COLUMNS_FOR_NOISE = ["tbb_13_t", "tbb_13_tm1", "tbb_13_tm2"]


def inject_lag_noise(X, noise_std, random_state=42):
    """
    Fix #3: tambahkan noise Gaussian ke kolom lag (tbb_13_t/tm1/tm2) SEBELUM
    training, supaya model terbiasa dengan input yang "berisik" -- meniru
    kondisi saat dipakai recursive (input di step > 1 adalah hasil prediksi
    step sebelumnya, bukan observasi presisi).

    `noise_std` idealnya ~seukuran MAE step-1 aktual model (lihat
    models/training_summary.csv atau recursive_evaluation.csv setelah
    training pertama). Kolom delta_t/delta_tm1/accel & neighbor_* TIDAK
    di-recompute ulang dari lag yang sudah dinoise -- ini konsisten dengan
    fakta bahwa saat recursive sungguhan, fitur turunan itu juga dihitung
    dari window yang sama-sama sudah "kotor" (lihat inference.py), tapi di
    training kita sengaja hanya menoise sinyal utamanya saja supaya
    pengaruh noise tetap terukur & tidak dobel-hitung.
    """
    if noise_std <= 0:
        return X
    rng = np.random.default_rng(random_state)
    X_noisy = X.copy()
    for col in LAG_COLUMNS_FOR_NOISE:
        if col in X_noisy.columns:
            X_noisy[col] = X_noisy[col] + rng.normal(0, noise_std, size=len(X_noisy))
    return X_noisy


def train_svr(X_train, y_train, subsample_n=25000, random_state=42):
    """
    SVR tidak scalable ke data besar (kompleksitas kuadratik-kubik), jadi
    training dilakukan pada subsample acak dari training set (bukan test
    set -- test tetap dievaluasi penuh). Fitur di-scale karena SVR sensitif
    terhadap skala antar fitur (lat/lon vs suhu vs sin/cos).
    """
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler

    if len(X_train) > subsample_n:
        X_sub = X_train.sample(n=subsample_n, random_state=random_state)
        y_sub = y_train.loc[X_sub.index]
    else:
        X_sub, y_sub = X_train, y_train

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sub)

    model = SVR(kernel="rbf", C=10.0, epsilon=0.1)
    model.fit(X_scaled, y_sub)
    return model, scaler


def train_xgboost(X_train, y_train):
    import xgboost as xgb

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_lightgbm(X_train, y_train):
    import lightgbm as lgb

    model = lgb.LGBMRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_catboost(X_train, y_train):
    from catboost import CatBoostRegressor

    model = CatBoostRegressor(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        random_seed=42,
        verbose=False,
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test, scaler=None):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    X_eval = scaler.transform(X_test) if scaler is not None else X_test
    y_pred = model.predict(X_eval)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return {"mae": mae, "rmse": rmse, "r2": r2}


TRAINERS = {
    "svr": train_svr,
    "xgboost": train_xgboost,
    "lightgbm": train_lightgbm,
    "catboost": train_catboost,
}


def train_one_model(model_name, X_train, y_train, noise_std=0.0):
    """
    Dispatch ke fungsi training yang sesuai; return (model, scaler_or_None, detik_training).

    noise_std > 0 (fix #3): X_train di-inject noise Gaussian di kolom lag
    sebelum training (lihat inject_lag_noise). Default 0 = perilaku lama
    (tanpa noise), supaya tidak mengubah hasil training yang sudah ada
    kecuali diaktifkan eksplisit.
    """
    X_train = inject_lag_noise(X_train, noise_std)
    start = time.time()
    if model_name == "svr":
        model, scaler = train_svr(X_train, y_train)
    else:
        model = TRAINERS[model_name](X_train, y_train)
        scaler = None
    elapsed = time.time() - start
    return model, scaler, elapsed