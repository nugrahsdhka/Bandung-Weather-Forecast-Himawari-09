# pipeline/visualization.py

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
from scipy.interpolate import griddata
from matplotlib.ticker import FormatStrFormatter
import geopandas as gpd
from pathlib import Path

# ==== Data kecamatan (dipakai sebagai marker + label di peta) ====
KECAMATAN_LIST = [
    {"name": "Bandung Wetan", "lat": -6.9055905, "lon": 107.6109559},
    {"name": "Coblong", "lat": -6.8851992, "lon": 107.6136456},
    {"name": "Sukajadi", "lat": -6.8922842, "lon": 107.5909487},
    {"name": "Batununggal", "lat": -6.9317606, "lon": 107.6431247},
    {"name": "Sumurbandung", "lat": -6.9160833, "lon": 107.6089776},
    {"name": "Cidadap", "lat": -6.8658706, "lon": 107.6060182},
    {"name": "Regol", "lat": -6.9375914, "lon": 107.6091286},
    {"name": "Lengkong", "lat": -6.9309616, "lon": 107.6224602},
    {"name": "Buahbatu", "lat": -6.9557189, "lon": 107.6542139},
    {"name": "Andir", "lat": -6.9136276, "lon": 107.5776047},
    {"name": "Cicendo", "lat": -6.9063897, "lon": 107.5975738},
    # {"name": "Bandung Kulon", "lat": -6.9267005, "lon": 107.5657164},
    # {"name": "Bojongloa Kaler", "lat": -6.9259906, "lon": 107.5912622},
    # {"name": "Kiaracondong", "lat": -6.9221438, "lon": 107.6490717},
    # {"name": "Rancasari", "lat": -6.9466401, "lon": 107.6761534},
    # {"name": "Arcamanik", "lat": -6.9291188, "lon": 107.6764994},
    # {"name": "Cibeunying Kidul", "lat": -6.90069, "lon": 107.6458487},
    # {"name": "Cibeunying Kaler", "lat": -6.8927351, "lon": 107.6354505},
    # {"name": "Babakan Ciparay", "lat": -6.9569775, "lon": 107.5821392},
    # {"name": "Astanaanyar", "lat": -6.9299008, "lon": 107.5993373},
    # {"name": "Sukasari", "lat": -6.8630393, "lon": 107.588082},
    # {"name": "Mandalajati", "lat": -6.9039442, "lon": 107.6776633},
    # {"name": "Antapani", "lat": -6.9135245, "lon": 107.6594029},
    # {"name": "Bandung Kidul", "lat": -6.9526296, "lon": 107.6380345},
    # {"name": "Gedebage", "lat": -6.9436184, "lon": 107.6815776},
    # {"name": "Bojongloa Kidul", "lat": -6.9521836, "lon": 107.5934256},
    # {"name": "Ujung Berung", "lat": -6.9060909, "lon": 107.6911552},
    # {"name": "Cinambo", "lat": -6.9371022, "lon": 107.6925593},
    # {"name": "Panyileukan", "lat": -6.92388, "lon": 107.7004759},
    # {"name": "Cibiru", "lat": -6.9230294, "lon": 107.7199696},
]


BASE_DIR = Path(__file__).resolve().parent.parent
GEOJSON_PATH = BASE_DIR / "geojson" / "KotaBandung.geojson"

BANDUNG_BOUNDARY = gpd.read_file(GEOJSON_PATH)

# ==== Threshold & Color Scheme (Modern Dark) ====
CLOUD_BINS = [220, 260]
CLOUD_LABELS = ["Hujan", "Mendung", "Tidak Hujan"]

CLOUD_COLORS = [
    "#082269",  
    "#5F6B7A",  
    "#8A6A00",  
]

RISK_BINS = [240, 265]
RISK_LABELS = ["Bahaya", "Waspada", "Aman"]
RISK_COLORS = [
    "#991B1B",
    "#A16207",
    "#166534",
]

GRAD_TOP = "#0F172A"
GRAD_BOTTOM = "#1E2937"

CARD_FACE = "#1F2937"
CARD_ALPHA = 0.88
CARD_EDGE = "#334155"

FG_COLOR = "#F1F5F9"
MUTED_COLOR = "#94A3B8"

GLASS_THERMAL = LinearSegmentedColormap.from_list(
        "premium", [
        "#2D6CDF","#62B5E5","#A9DCEE", "#F8F8F8", "#FFF4C2","#FFD96A","#FFB03A","#FF6A2A","#D7191C","#B60808",
    ]
)

GLASS_ERROR = LinearSegmentedColormap.from_list(
    "glass_error", ["#F8FAFC", "#DBB90C", "#CA5E05", "#9A0C0C", "#3D0101"]
)


def classify(values, bins):
    """Klasifikasi array nilai CTT ke indeks kategori (0..n-1) berdasarkan bins."""
    return np.digitize(values, bins)


def pivot_grid(df_frame, value_col, n_lat=5, n_lon=7):
    """Ubah dataframe panjang (pixel_row, pixel_col, value) jadi array 2D (n_lat, n_lon)."""
    grid = np.full((n_lat, n_lon), np.nan)
    for row in df_frame.itertuples(index=False):
        grid[int(row.pixel_row), int(row.pixel_col)] = getattr(row, value_col)
    return grid


def get_axis_arrays(df_frame, n_lat=5, n_lon=7):
    """Ambil array lat (n_lat,) & lon (n_lon,) asli dari data (urutan sesuai pixel_row/pixel_col)."""
    lat_arr = np.zeros(n_lat)
    lon_arr = np.zeros(n_lon)
    for i in range(n_lat):
        lat_arr[i] = df_frame.loc[df_frame["pixel_row"] == i, "lat"].iloc[0]
    for j in range(n_lon):
        lon_arr[j] = df_frame.loc[df_frame["pixel_col"] == j, "lon"].iloc[0]
    return lat_arr, lon_arr


def interpolate_grid(lat_arr, lon_arr, values_2d, factor=12):
    """Upsample grid kasar (5x7) jadi lebih halus pakai interpolasi kubik, biar kontur mulus."""
    lon_grid, lat_grid = np.meshgrid(lon_arr, lat_arr)
    points = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])
    vals = values_2d.ravel()

    valid = ~np.isnan(vals)
    lon_fine = np.linspace(lon_arr.min(), lon_arr.max(), len(lon_arr) * factor)
    lat_fine = np.linspace(lat_arr.min(), lat_arr.max(), len(lat_arr) * factor)
    lon_fine_grid, lat_fine_grid = np.meshgrid(lon_fine, lat_fine)

    vals_fine = griddata(points[valid], vals[valid], (lon_fine_grid, lat_fine_grid), method="cubic")
    if np.isnan(vals_fine).any():
        vals_nearest = griddata(points[valid], vals[valid], (lon_fine_grid, lat_fine_grid), method="nearest")
        vals_fine = np.where(np.isnan(vals_fine), vals_nearest, vals_fine)

    return lat_fine, lon_fine, vals_fine


def _add_gradient_background(fig):
    """Background gradient dark modern."""
    bg_ax = fig.add_axes([0, 0, 1, 1], zorder=-20)
    bg_ax.set_axis_off()
    gradient = np.linspace(0, 1, 256).reshape(-1, 1)
    cmap = LinearSegmentedColormap.from_list("bg_grad", [GRAD_TOP, GRAD_BOTTOM])
    bg_ax.imshow(gradient, aspect="auto", cmap=cmap, extent=[0, 1, 0, 1], origin="upper")
    return bg_ax


def _draw_kecamatan(ax):
    """
    Menggambar batas kecamatan dan nama kecamatan.
    """

    # Garis batas
    BANDUNG_BOUNDARY.boundary.plot(
        ax=ax,
        edgecolor="white",
        linewidth=0.8,
        alpha=0.5,
        zorder=15,
    )

    # Nama kecamatan
    for _, row in BANDUNG_BOUNDARY.iterrows():

        point = row.geometry.representative_point()
        name = row["NAME_3"].replace(" ", "\n", 1)

        ax.text(
            point.x,
            point.y,

            name,

            fontsize=5,
            clip_on=True,

            color="white",

            ha="center",
            va="center",

            zorder=10,
        )


def _style_axis(ax, lat_fine, lon_fine, kecamatan_list):
    ax.set_facecolor("none")
    minx, miny, maxx, maxy = BANDUNG_BOUNDARY.total_bounds

    margin_x = (maxx - minx) * 0.03
    margin_y = (maxy - miny) * 0.03

    ax.set_xlim(minx - margin_x, maxx + margin_x)
    ax.set_ylim(miny - margin_y, maxy + margin_y)

    _draw_kecamatan(ax)

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(colors=MUTED_COLOR, labelsize=6.5, length=0)
    ax.tick_params(axis="y", labelrotation=45)

    for spine in ax.spines.values():
        spine.set_visible(False)


def _plot_continuous(ax, lat_fine, lon_fine, values_fine, title, kecamatan_list, cmap=GLASS_THERMAL):
    ax.set_title(title, color=FG_COLOR, fontsize=11, fontweight="700", pad=8)
    cf = ax.contourf(lon_fine, lat_fine, values_fine, levels=25, cmap=cmap, alpha=0.92)
    cb = plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.03, shrink=0.75)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED_COLOR, labelsize=6)
    cb.set_label("CTT (K)", color=MUTED_COLOR, fontsize=7)
    _style_axis(ax, lat_fine, lon_fine, kecamatan_list)
    cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))


def _add_category_legend(ax, labels, colors):
    patches = [mpatches.Patch(facecolor=c, edgecolor="none", label=l, alpha=0.95) for l, c in zip(labels, colors)]
    leg = ax.legend(
        handles=patches, loc="upper center", bbox_to_anchor=(0.5, -0.18),
        ncol=len(labels), frameon=True, fontsize=7.5, handlelength=1.4,
        columnspacing=1.2, handletextpad=0.6
    )
    leg.get_frame().set_facecolor(CARD_FACE)
    leg.get_frame().set_alpha(0.7)
    leg.get_frame().set_edgecolor(CARD_EDGE)
    for text in leg.get_texts():
        text.set_color(FG_COLOR)


def _plot_categorical(ax, lat_fine, lon_fine, values_fine, bins, labels, colors, title, kecamatan_list):
    ax.set_title(title, color=FG_COLOR, fontsize=11, fontweight="700", pad=8)
    cat = classify(values_fine, bins)
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(list(range(len(labels) + 1)), cmap.N)
    ax.pcolormesh(lon_fine, lat_fine, cat, cmap=cmap, norm=norm, shading="auto", alpha=0.88)
    _style_axis(ax, lat_fine, lon_fine, kecamatan_list)
    _add_category_legend(ax, labels, colors)
    return cat


def render_six_panel(
    lat_arr, lon_arr,
    input_grid, pred_grid, actual_grid,
    t0_label, forecast_label, interval_minutes,
    out_path=None, kecamatan_list=KECAMATAN_LIST, interp_factor=12,
):
    """
    Render figure 6-panel modern minimalist dark theme.
    """
    lat_fine, lon_fine, input_fine = interpolate_grid(lat_arr, lon_arr, input_grid, interp_factor)
    _, _, pred_fine = interpolate_grid(lat_arr, lon_arr, pred_grid, interp_factor)

    has_actual = actual_grid is not None and not np.all(np.isnan(actual_grid))
    if has_actual:
        _, _, actual_fine = interpolate_grid(lat_arr, lon_arr, actual_grid, interp_factor)
        mae = float(np.nanmean(np.abs(pred_grid - actual_grid)))
        ctt_min = float(np.nanmin(actual_grid))
        ctt_max = float(np.nanmax(actual_grid))
        ctt_range = max(ctt_max - ctt_min, 1e-6)
        accuracy = max(0.0, 100.0 * (1 - mae / ctt_range))
    else:
        actual_fine = None
        mae, ctt_min, ctt_max, accuracy = None, None, None, None

    risk_cat_grid = classify(pred_grid, RISK_BINS)
    worst_risk_idx = int(np.nanmax(risk_cat_grid))
    risk_status = RISK_LABELS[worst_risk_idx]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.3333), dpi=120)
    main_axes = list(axes.flat)

    title_line1 = f"Bandung Weather Forecast • {interval_minutes} min"
    if has_actual:
        title_line2 = f"{t0_label} → {forecast_label}  |  {risk_status}  |  MAE: {mae:.2f}K • Acc: {accuracy:.1f}%"
    else:
        title_line2 = f"{t0_label} → {forecast_label}  |  {risk_status}  |  Forecast Mode"

    fig.suptitle(title_line1, color=FG_COLOR, fontsize=15, fontweight="700", y=0.97)
    fig.text(0.5, 0.915, title_line2, color="#38BDF8", fontsize=10.5, ha="center", fontweight="400")

    _plot_continuous(axes[0, 0], lat_fine, lon_fine, input_fine, f"Input • {t0_label}", kecamatan_list)
    _plot_continuous(axes[0, 1], lat_fine, lon_fine, pred_fine, f"Prediksi • {forecast_label}", kecamatan_list)

    if has_actual:
        _plot_continuous(axes[0, 2], lat_fine, lon_fine, actual_fine, f"Aktual • {forecast_label}", kecamatan_list)
        _plot_categorical(axes[1, 0], lat_fine, lon_fine, actual_fine, CLOUD_BINS, CLOUD_LABELS, CLOUD_COLORS, "Kelas Awan", kecamatan_list)
        
        error_fine = np.abs(pred_fine - actual_fine)
        axes[1, 2].set_title(f"Error Map  (MAE = {mae:.3f}K)", color=FG_COLOR, fontsize=11, fontweight="700", pad=8)
        cf = axes[1, 2].contourf(lon_fine, lat_fine, error_fine, levels=20, cmap=GLASS_ERROR, alpha=0.9)
        cb = plt.colorbar(cf, ax=axes[1, 2], fraction=0.046, pad=0.03, shrink=0.75)
        cb.outline.set_visible(False)
        cb.ax.tick_params(colors=MUTED_COLOR, labelsize=6)
        _style_axis(axes[1, 2], lat_fine, lon_fine, kecamatan_list)
    else:
        for ax, msg in [(axes[0, 2], "CTT Aktual"), (axes[1, 0], "Kelas Awan"), (axes[1, 2], "Error")]:
            ax.set_title(f"{msg} (Forecast Mode)", color=FG_COLOR, fontsize=11, fontweight="700", pad=8)
            ax.text(0.5, 0.5, "Data aktual\nbelum tersedia", color=MUTED_COLOR,
                    fontsize=10, ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    _plot_categorical(
        axes[1, 1], lat_fine, lon_fine, pred_fine, RISK_BINS, RISK_LABELS, RISK_COLORS,
        f"Risk Banjir • {risk_status}", kecamatan_list,
    )

    plt.tight_layout(
        rect=[0, 0.03, 1, 0.92],
        h_pad=5.5,
        w_pad=1,
    )

    _add_gradient_background(fig)

    if out_path:
        fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=GRAD_TOP)
        plt.close(fig)
        return out_path
    return fig