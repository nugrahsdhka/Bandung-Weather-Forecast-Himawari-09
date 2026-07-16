# ./scripts/02_explore_data.py
# Mengeksplorasi dataset NetCDF hasil unduhan dengan menganalisis cakupan temporal, gap data, struktur variabel, dan konsistensi dimensi antar file.

import os
from collections import defaultdict

import xarray as xr

from ui.terminal_display import hr, gap, banner, say_info, say_ok, say_error, say_skip
from pipeline.config import load_config
from pipeline.netcdf_tools import extract_time_from_filename
from pipeline.utils import format_size


def scan_all_files(final_base_dir):
    """Telusuri seluruh data_bandung/, kumpulkan (timestamp, path) tiap file .nc."""
    entries = []
    for root, _dirs, files in os.walk(final_base_dir):
        for f in files:
            if not f.endswith(".nc"):
                continue
            ts = extract_time_from_filename(f)
            if ts is None:
                say_skip(f"Nama file tidak dikenali pola timestamp-nya: {f}")
                continue
            entries.append((ts, os.path.join(root, f)))
    entries.sort(key=lambda x: x[0])
    return entries


def report_coverage(entries):
    """Ringkas cakupan waktu: total file, rentang tanggal, dan gap terhadap kadensi 10 menit."""
    banner("CAKUPAN WAKTU")
    if not entries:
        say_error("Tidak ada file .nc ditemukan di FINAL_BASE_DIR.")
        return

    say_info(f"Total file ditemukan : {len(entries)}")
    say_info(f"Rentang waktu        : {entries[0][0]} s/d {entries[-1][0]}")
    hr()

    per_day = defaultdict(list)
    for ts, _path in entries:
        per_day[ts.date()].append(ts)

    say_info(f"Jumlah hari unik dengan data: {len(per_day)}")
    hr()

    banner("DETAIL PER HARI (gap terhadap kadensi 10 menit)")
    total_missing = 0
    for day in sorted(per_day.keys()):
        timestamps = sorted(per_day[day])
        n = len(timestamps)
        start, end = timestamps[0], timestamps[-1]

        expected_slots = int((end - start).total_seconds() // 600) + 1
        missing = expected_slots - n
        total_missing += missing

        gaps = []
        for i in range(1, len(timestamps)):
            delta_min = (timestamps[i] - timestamps[i - 1]).total_seconds() / 60
            if delta_min > 10:
                gaps.append((timestamps[i - 1], timestamps[i], int(delta_min)))

        status = say_ok if missing == 0 else say_skip
        status(
            f"{day} | {n:3d} file | {start.strftime('%H:%M')}-{end.strftime('%H:%M')} "
            f"| slot hilang (dlm rentang jam yg ada): {missing}"
        )
        for g_start, g_end, delta_min in gaps:
            print(f"      gap {delta_min} menit: {g_start.strftime('%H:%M')} -> {g_end.strftime('%H:%M')}")

    hr()
    say_info(f"Total slot 10-menit hilang (dalam rentang jam yang ada per hari): {total_missing}")
    say_info("Catatan: ini BELUM termasuk jam-jam yang sama sekali tidak terdownload di suatu hari.")
    hr()


def inspect_sample_files(entries, n_samples=3):
    """Buka beberapa file sampel (awal, tengah, akhir) dan cetak struktur NetCDF-nya."""
    banner("STRUKTUR FILE NETCDF (SAMPEL)")
    if not entries:
        return

    idxs = sorted(set([0, len(entries) // 2, len(entries) - 1][:n_samples]))
    shapes_seen = set()

    for idx in idxs:
        ts, path = entries[idx]
        say_info(f"Membuka: {os.path.basename(path)}  ({format_size(os.path.getsize(path))})")
        try:
            with xr.open_dataset(path) as ds:
                hr()
                print(f"  Timestamp (dari nama file) : {ts}")
                print(f"  Dimensi   : {dict(ds.sizes)}")
                print(f"  Koordinat : {list(ds.coords)}")

                if "latitude" in ds.coords:
                    lat = ds.latitude.values
                    print(f"  Lat range aktual (subset)  : {lat.min():.4f} .. {lat.max():.4f}  (n={lat.size})")
                if "longitude" in ds.coords:
                    lon = ds.longitude.values
                    print(f"  Lon range aktual (subset)  : {lon.min():.4f} .. {lon.max():.4f}  (n={lon.size})")

                data_vars = list(ds.data_vars)
                print(f"  Data variables ({len(data_vars)}): {data_vars}")

                for var in data_vars:
                    da = ds[var]
                    try:
                        vmin = float(da.min(skipna=True))
                        vmax = float(da.max(skipna=True))
                        print(f"    - {var:14s} shape={da.shape} dtype={da.dtype} range=[{vmin:.2f}, {vmax:.2f}]")
                    except Exception:
                        print(f"    - {var:14s} shape={da.shape} dtype={da.dtype} (gagal hitung min/max)")

                shapes_seen.add(tuple(ds.sizes.get(d) for d in ds.sizes))

                if ds.attrs:
                    print(f"  Global attrs: {list(ds.attrs.keys())}")
                hr()
                say_ok("OK dibaca")
        except Exception as e:
            say_error(f"Gagal membuka {path}: {e}")
        gap()

    if len(shapes_seen) > 1:
        say_error(f"PERINGATAN: ukuran grid TIDAK konsisten antar file sampel! {shapes_seen}")
    else:
        say_ok(f"Ukuran grid konsisten di seluruh sampel: {shapes_seen}")


def main():
    cfg = load_config()

    banner("EKSPLORASI DATA - CTT FORECASTING")
    say_info(f"Folder data : {cfg.FINAL_BASE_DIR}")
    hr()

    entries = scan_all_files(cfg.FINAL_BASE_DIR)
    report_coverage(entries)
    inspect_sample_files(entries, n_samples=3)

    gap()
    banner("SELESAI")
    say_info("Salin/paste ringkasan di atas balik ke chat supaya bisa lanjut ke Tahap 2 (feature engineering).")


if __name__ == "__main__":
    main()