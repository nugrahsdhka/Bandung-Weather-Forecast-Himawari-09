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
from scipy.ndimage import gaussian_filter
import geopandas as gpd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
GEOJSON_PATH = BASE_DIR / "geojson" / "KotaBandung.geojson"

BANDUNG_BOUNDARY = gpd.read_file(GEOJSON_PATH)

# ==== Threshold & Color Scheme (Modern Dark) ====
CLOUD_BINS = [220, 260]
CLOUD_LABELS = ["Hujan", "Mendung", "Tidak Hujan"]
CLOUD_COLORS = [
    "#08285B",
    "#6A727D",
    "#B89300",
]

RISK_BINS = [240, 265]
RISK_LABELS = ["Bahaya", "Waspada", "Aman"]
RISK_COLORS = [
    "#9B0000",
    "#B89300",
    "#217C00",
]

GRAD_TOP    = "#06111B"
GRAD_BOTTOM = "#204A73"

CARD_FACE = "#162434"
CARD_ALPHA = 0.78
CARD_EDGE = "#35516B"

FG_COLOR = "#F8FAFC"
MUTED_COLOR = "#C7D5E5"

GLASS_THERMAL = LinearSegmentedColormap.from_list(
    "premium",
    [
        "#08245E",
        "#092762",
        "#0A2B68",
        "#0B2F6E",
        "#0C3375",
        "#0E387D",
        "#104086",
        "#12488F",
        "#15519B",
        "#185AA8",
        "#1D66B8",
        "#2473CA",
        "#2F82DE",
        "#3F93F1",
        "#52A6FF",
        "#68BBFF",
        "#82D1FF",
        "#9EE6FF",
        "#B9F5FF",

        "#A3E68E",
        "#8ED97D",
        "#7FC97A",
        "#92CF71",
        "#A7D46A",
        "#BDD761",
        "#D1D95A",
        "#E4D853",

        "#F1CF47",
        "#F4C33C",
        "#F6B532",
        "#F8A429",
        "#F88F23",
        "#F67A20",
        "#F1661E",

        "#EB521C",
        "#E23F1A",
        "#D62E18",
        "#C91D1D",

        "#B31617",
        "#9B1010",
        "#800A0A",
    ],
    N=1024,
)

GLASS_ERROR = LinearSegmentedColormap.from_list( 
    "glass_error", [ 
        "#F7F3F3", 
        "#DFDFDF", 
        "#FFEB75",
        "#FFD500",
        "#D07B2F",
        "#8A2A17",
        "#490707",
        ]
    )

# ==== Skala CTT (K) & Error (K) yang FIKS, supaya tidak berubah-ubah antar panel/run ====
CTT_VMIN = 220.0
CTT_VMAX = 300.0
CTT_LEVELS = np.linspace(CTT_VMIN, CTT_VMAX, 161)
CTT_TICKS = np.arange(220, 301, 10)

ERROR_VMIN = 0.0
ERROR_VMAX = 10.0
ERROR_LEVELS = np.linspace(ERROR_VMIN, ERROR_VMAX, 21)
ERROR_TICKS = np.arange(220, 301, 10)


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


def interpolate_grid(lat_arr, lon_arr, values_2d, factor=24):
    """Upsample grid kasar (5x7) jadi lebih halus pakai interpolasi kubik, biar kontur mulus."""
    lon_grid, lat_grid = np.meshgrid(lon_arr, lat_arr)
    points = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])
    vals = values_2d.ravel()

    valid = ~np.isnan(vals)
    lon_fine = np.linspace(lon_arr.min(), lon_arr.max(), len(lon_arr) * factor)
    lat_fine = np.linspace(lat_arr.min(), lat_arr.max(), len(lat_arr) * factor)
    lon_fine_grid, lat_fine_grid = np.meshgrid(lon_fine, lat_fine)

    vals_fine = griddata(points[valid], vals[valid], (lon_fine_grid, lat_fine_grid), method="cubic")
    vals_fine = np.clip(
        vals_fine,
        CTT_VMIN,
        CTT_VMAX,
    )
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
        edgecolor="#F5F5F5",
        linewidth=1.1,
        alpha=0.8,
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

            fontsize=6,
            fontweight="bold",
            alpha=0.9,
            clip_on=True,

            path_effects=[
                pe.withStroke(
                    linewidth=1.4,
                    foreground="#000000"
                )
            ],

            color="white",

            ha="center",
            va="center",

            zorder=10,
        )


def _style_axis(ax, lat_fine, lon_fine):
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


def _plot_continuous( ax, lat_fine, lon_fine, values_fine, title, cmap=GLASS_THERMAL, vmin=CTT_VMIN, vmax=CTT_VMAX, levels=CTT_LEVELS, ticks=CTT_TICKS, cb_label="CTT (K)"):
    ax.set_title(
        title,
        color=FG_COLOR,
        fontsize=12,
        fontweight="700",
        pad=8,
    )

    norm = plt.Normalize(vmin, vmax)

    cf = ax.contourf(
        lon_fine,
        lat_fine,
        values_fine,
        levels=levels,
        cmap=cmap,
        norm=norm,
        extend="both",
        antialiased=True,
    )

    cb = plt.colorbar(
        cf,
        ax=ax,
        fraction=0.046,
        pad=0.03,
        shrink=0.83,
        aspect=22,
        ticks=ticks,
    )

    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED_COLOR, labelsize=6)
    cb.set_label(cb_label, color=MUTED_COLOR, fontsize=7)
    cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    _style_axis(ax, lat_fine, lon_fine)

    return cb

def _add_category_legend(ax, labels, colors):
    patches = [mpatches.Patch(facecolor=c, edgecolor="none", label=l, alpha=0.95) for l, c in zip(labels, colors)]
    leg = ax.legend(
        handles=patches, loc="upper center", bbox_to_anchor=(0.5, -0.18),
        ncol=len(labels), frameon=True, fontsize=8, handlelength=1.6, borderpad=0.8, labelspacing=0.8,
        columnspacing=1.2, handletextpad=0.6
    )
    leg.get_frame().set_facecolor(CARD_FACE)
    leg.get_frame().set_alpha(0.7)
    leg.get_frame().set_edgecolor(CARD_EDGE)
    for text in leg.get_texts():
        text.set_color(FG_COLOR)


def _plot_categorical(ax, lat_fine, lon_fine, values_fine, bins, labels, colors, title):
    ax.set_title(title, color=FG_COLOR, fontsize=11, fontweight="700", pad=8)
    cat = classify(values_fine, bins)
    cat_smooth = gaussian_filter(cat.astype(float), sigma=2)
    cmap = LinearSegmentedColormap.from_list(
        "cloud",
        colors,
        N=512,
    )
    norm = BoundaryNorm(list(range(len(labels) + 1)), cmap.N)

    cf = ax.contourf(
        lon_fine,
        lat_fine,
        cat_smooth,
        levels=np.arange(len(labels)+1)-0.5,
        cmap=cmap,
        alpha=0.96,
        antialiased=True,
    )

    ax.contour(
        lon_fine,
        lat_fine,
        cat,
        levels=np.arange(len(labels)+1)-0.5,
        colors="white",
        linewidths=0.18,
        alpha=0.12,
    )

    _style_axis(ax, lat_fine, lon_fine)
    _add_category_legend(ax, labels, colors)
    return cat


def _plot_placeholder(
    ax, lat_fine, lon_fine, title,
    kind="continuous", cmap=GLASS_THERMAL, vmin=CTT_VMIN, vmax=CTT_VMAX,
    ticks=CTT_TICKS, cb_label="CTT (K)", labels=None, colors=None,
):
    """
    Placeholder panel saat data aktual belum tersedia.
    Tetap menggambar colorbar/legend kosong dengan geometri sama
    seperti panel berisi data, supaya ukuran subplot tetap konsisten.
    """
    ax.set_title(title, color=FG_COLOR, fontsize=11, fontweight="700", pad=8)

    if kind == "continuous":
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.03, shrink=0.83, aspect=22, ticks=ticks)
        cb.outline.set_visible(False)
        cb.ax.tick_params(colors=MUTED_COLOR, labelsize=6)
        cb.set_label(cb_label, color=MUTED_COLOR, fontsize=7)
        cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    _style_axis(ax, lat_fine, lon_fine)

    if kind == "categorical":
        _add_category_legend(ax, labels, colors)

    ax.text(
        0.5, 0.5, "Data aktual\nbelum tersedia",
        color=MUTED_COLOR, fontsize=12, fontweight="700", ha="center", va="center",
        transform=ax.transAxes, zorder=20,
    )


def render_six_panel(
    lat_arr, lon_arr,
    input_grid, pred_grid, actual_grid,
    t0_label, forecast_label, interval_minutes,
    out_path=None, interp_factor=24,
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
        ctt_min = CTT_VMIN
        ctt_max = CTT_VMAX
        ctt_range = max(ctt_max - ctt_min, 1e-6)
        accuracy = max(0.0, 100.0 * (1 - mae / ctt_range))
    else:
        actual_fine = None
        mae, ctt_min, ctt_max, accuracy = None, None, None, None

    risk_cat_grid = classify(pred_grid, RISK_BINS)
    worst_risk_idx = int(np.nanmax(risk_cat_grid))
    risk_status = RISK_LABELS[worst_risk_idx]

    fig, axes = plt.subplots(2, 3, figsize=(18,9), dpi=180)
    main_axes = list(axes.flat)

    title_line1 = f"Bandung Weather Forecast • {interval_minutes} min"
    if has_actual:
        title_line2 = f"{t0_label} → {forecast_label}  |  {risk_status}  |  MAE: {mae:.2f}K • Acc: {accuracy:.1f}%"
    else:
        title_line2 = f"{t0_label} → {forecast_label}  |  {risk_status}  |  Forecast Mode"

    fig.suptitle(title_line1, color=FG_COLOR, fontsize=15, fontweight="700", y=0.97)
    fig.text(0.5, 0.915, title_line2, color="#38BDF8", fontsize=10.5, ha="center", fontweight="400")

    # Input & Prediksi selalu memakai skala CTT (K) fiks (CTT_VMIN..CTT_VMAX)
    _plot_continuous(axes[0, 0], lat_fine, lon_fine, input_fine, f"Input • {t0_label}")
    _plot_continuous(axes[0, 1], lat_fine, lon_fine, pred_fine, f"Prediksi • {forecast_label}")

    if has_actual:
        _plot_continuous(axes[0, 2], lat_fine, lon_fine, actual_fine, f"Aktual • {forecast_label}")
        _plot_categorical(axes[1, 0], lat_fine, lon_fine, actual_fine, CLOUD_BINS, CLOUD_LABELS, CLOUD_COLORS, "Kelas Awan")

        error_fine = np.abs(pred_fine - actual_fine)
        axes[1, 2].set_title(f"Error Map  (MAE = {mae:.3f}K)", color=FG_COLOR, fontsize=11, fontweight="700", pad=8)
        cf = axes[1, 2].contourf(
            lon_fine, lat_fine, error_fine,
            levels=ERROR_LEVELS, vmin=ERROR_VMIN, vmax=ERROR_VMAX,
            cmap=GLASS_ERROR, alpha=0.96, extend="max",antialiased=True,
        )
        axes[1,2].contour(
            lon_fine,
            lat_fine,
            error_fine,
            levels=ERROR_LEVELS,
            colors="white",
            linewidths=0.15,
            alpha=0.15,
        )
        cb = plt.colorbar(cf, ax=axes[1, 2], fraction=0.046, pad=0.03, shrink=0.83, aspect=22, ticks=ERROR_TICKS)
        cb.outline.set_visible(False)
        cb.ax.tick_params(colors=MUTED_COLOR, labelsize=6)
        cb.set_label("Error (K)", color=MUTED_COLOR, fontsize=7)
        cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        _style_axis(axes[1, 2], lat_fine, lon_fine)
    else:
        # Panel placeholder tetap menggambar colorbar/legend kosong dengan
        # geometri sama seperti versi berisi data, supaya ukuran panel identik.
        _plot_placeholder(
            axes[0, 2], lat_fine, lon_fine, f"Aktual • {forecast_label}",
            kind="continuous", cmap=GLASS_THERMAL, vmin=CTT_VMIN, vmax=CTT_VMAX,
            ticks=CTT_TICKS, cb_label="CTT (K)",
        )
        _plot_placeholder(
            axes[1, 0], lat_fine, lon_fine, "Kelas Awan",
            kind="categorical", labels=CLOUD_LABELS, colors=CLOUD_COLORS,
        )
        _plot_placeholder(
            axes[1, 2], lat_fine, lon_fine, "Error Map",
            kind="continuous", cmap=GLASS_ERROR, vmin=ERROR_VMIN, vmax=ERROR_VMAX,
            ticks=ERROR_TICKS, cb_label="Error (K)",
        )

    _plot_categorical(
        axes[1, 1], lat_fine, lon_fine, pred_fine, RISK_BINS, RISK_LABELS, RISK_COLORS,
        f"Risk Banjir • {risk_status}",
    )

    plt.tight_layout(
        rect=[0, 0.03, 1, 0.92],
        h_pad=5.5,
        w_pad=1,
    )

    _add_gradient_background(fig)

    if out_path:
        fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=GRAD_TOP)
        plt.close(fig)
        return out_path
    return fig