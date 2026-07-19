# ./scripts/pipeline/ground_truth.py
# Lookup nilai aktual tbb_13 per (pixel, timestamp) -- dipisah dari
# recursive_eval.py supaya tidak circular import dengan inference.py.


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
