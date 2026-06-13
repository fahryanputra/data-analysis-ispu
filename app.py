import io
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ============================================================
# Dashboard ISPU Jakarta
# File utama: app.py
# Jalankan: streamlit run app.py
# Letakkan file CSV ispu_jakarta_analysis.csv di folder yang sama,
# atau unggah CSV melalui sidebar dashboard.
# ============================================================

st.set_page_config(
    page_title="Dashboard ISPU Jakarta",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Konfigurasi dasar
# -----------------------------
POLLUTANTS = ["pm10", "pm25", "so2", "co", "o3", "no2"]
POLLUTANT_LABELS = {
    "pm10": "PM10",
    "pm25": "PM2.5",
    "so2": "SO2",
    "co": "CO",
    "o3": "O3",
    "no2": "NO2",
}
LABEL_TO_COL = {v: k for k, v in POLLUTANT_LABELS.items()}

CATEGORY_ORDER = [
    "BAIK",
    "SEDANG",
    "TIDAK SEHAT",
    "SANGAT TIDAK SEHAT",
    "BERBAHAYA",
    "TIDAK TERSEDIA",
]
CATEGORY_COLORS = {
    "BAIK": "#2E7D32",
    "SEDANG": "#F9A825",
    "TIDAK SEHAT": "#EF6C00",
    "SANGAT TIDAK SEHAT": "#C62828",
    "BERBAHAYA": "#6A1B9A",
    "TIDAK TERSEDIA": "#9E9E9E",
}
POLLUTANT_COLORS = {
    "PM10": "#795548",
    "PM2.5": "#455A64",
    "SO2": "#7B1FA2",
    "CO": "#1976D2",
    "O3": "#00897B",
    "NO2": "#D84315",
}

MONTH_ID = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "Mei",
    6: "Jun",
    7: "Jul",
    8: "Agu",
    9: "Sep",
    10: "Okt",
    11: "Nov",
    12: "Des",
}
MONTH_FULL_ID = {
    1: "Januari",
    2: "Februari",
    3: "Maret",
    4: "April",
    5: "Mei",
    6: "Juni",
    7: "Juli",
    8: "Agustus",
    9: "September",
    10: "Oktober",
    11: "November",
    12: "Desember",
}

# Koordinat pendekatan untuk visual ringkasan lokasi stasiun.
# Jika organisasi memiliki koordinat resmi SPKU, ganti nilai berikut dengan data resmi.
STATION_COORDS = {
    "DKI1 Bunderan HI": {"lat": -6.1950, "lon": 106.8230, "wilayah": "Jakarta Pusat"},
    "DKI2 Kelapa Gading": {"lat": -6.1536, "lon": 106.9100, "wilayah": "Jakarta Utara"},
    "DKI3 Jagakarsa": {"lat": -6.3566, "lon": 106.8243, "wilayah": "Jakarta Selatan"},
    "DKI4 Lubang Buaya": {"lat": -6.2900, "lon": 106.9025, "wilayah": "Jakarta Timur"},
    "DKI5 Kebon Jeruk": {"lat": -6.1967, "lon": 106.7586, "wilayah": "Jakarta Barat"},
}

DEFAULT_DATA_CANDIDATES = [
    "ispu_jakarta_analysis.csv",
    "ispu_jakarta_clean.csv",
    "outputs/ispu_jakarta_analysis.csv",
    "data/ispu_jakarta_clean.csv",
]

# -----------------------------
# Style
# -----------------------------
st.html(
    """
    <style>
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2.2rem;
    }
    .main-title {
        font-size: 2.05rem;
        line-height: 1.15;
        font-weight: 800;
        color: #102A43;
        margin-bottom: 0.25rem;
    }
    .subtitle {
        font-size: 1rem;
        color: #486581;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #F7FAFC 100%);
        border: 1px solid #E6EEF6;
        border-radius: 18px;
        padding: 1.05rem 1rem;
        box-shadow: 0 8px 20px rgba(16, 42, 67, 0.06);
        min-height: 120px;
    }
    .metric-label {
        font-size: 0.82rem;
        color: #627D98;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        margin-bottom: 0.45rem;
    }
    .metric-value {
        font-size: 1.95rem;
        line-height: 1.1;
        color: #102A43;
        font-weight: 850;
        margin-bottom: 0.35rem;
    }
    .metric-help {
        font-size: 0.86rem;
        color: #52606D;
    }
    .insight-box {
        border-left: 6px solid #0F766E;
        background: #F0FDFA;
        border-radius: 12px;
        padding: 0.9rem 1rem;
        margin: 0.65rem 0 1rem 0;
        color: #134E4A;
    }
    .warning-box {
        border-left: 6px solid #EA580C;
        background: #FFF7ED;
        border-radius: 12px;
        padding: 0.9rem 1rem;
        margin: 0.65rem 0 1rem 0;
        color: #7C2D12;
    }
    .plain-card {
        background: #FFFFFF;
        border: 1px solid #E6EEF6;
        border-radius: 16px;
        padding: 1rem;
        box-shadow: 0 6px 16px rgba(16, 42, 67, 0.05);
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.7rem;
    }
    </style>
    """,
)


# -----------------------------
# Helper umum
# -----------------------------
def find_default_data_path() -> str | None:
    here = Path(__file__).parent if "__file__" in globals() else Path.cwd()
    candidates = [here / p for p in DEFAULT_DATA_CANDIDATES] + [
        Path.cwd() / p for p in DEFAULT_DATA_CANDIDATES
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return None


@st.cache_data(show_spinner=False)
def load_raw_data_from_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_raw_data_from_bytes(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


def safe_pct(value: float | int | None, decimals: int = 1) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{value:.{decimals}f}%"


def safe_num(value: float | int | None, decimals: int = 1) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{value:,.{decimals}f}".replace(",", ".")


def kpi_card(label: str, value: str, help_text: str = "") -> None:
    st.html(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-help">{help_text}</div>
        </div>
        """,
    )


def insight_box(text: str, kind: str = "insight") -> None:
    css_class = "warning-box" if kind == "warning" else "insight-box"
    st.html(f"<div class='{css_class}'>{text}</div>")


def categorize_ispu(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "TIDAK TERSEDIA"
    if value <= 50:
        return "BAIK"
    if value <= 100:
        return "SEDANG"
    if value <= 200:
        return "TIDAK SEHAT"
    if value <= 300:
        return "SANGAT TIDAK SEHAT"
    return "BERBAHAYA"


def season_from_month(month: int) -> str:
    # Proxy umum Jakarta: Nov-Apr lebih basah/hujan, Mei-Okt lebih kering/kemarau.
    if int(month) in [11, 12, 1, 2, 3, 4]:
        return "Hujan/lebih basah"
    return "Kemarau/lebih kering"


def mode_or_dash(series: pd.Series) -> str:
    series = series.dropna()
    if series.empty:
        return "-"
    mode = series.mode()
    return str(mode.iloc[0]) if not mode.empty else "-"


def add_unhealthy_hline(
    fig: go.Figure, y: float = 100, annotation: str = "Ambang Tidak Sehat = 100"
) -> go.Figure:
    fig.add_hline(
        y=y,
        line_dash="dash",
        line_width=1.5,
        line_color="#D84315",
        annotation_text=annotation,
        annotation_position="top left",
    )
    return fig


def clean_plot_layout(
    fig: go.Figure, height: int = 420, legend_orientation: str = "h"
) -> go.Figure:
    fig.update_layout(
        height=height,
        template="plotly_white",
        margin=dict(l=10, r=10, t=55, b=10),
        legend=dict(
            orientation=legend_orientation,
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        title=dict(font=dict(size=18, color="#102A43")),
        font=dict(color="#243B53"),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#EEF2F7")
    return fig


@st.cache_data(show_spinner=False)
def prepare_ispu_data(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    date_col = "tanggal_validated" if "tanggal_validated" in df.columns else "tanggal"
    if date_col not in df.columns:
        raise ValueError(
            "Kolom tanggal tidak ditemukan. Pastikan CSV memiliki kolom 'tanggal' atau 'tanggal_validated'."
        )

    station_col = "stasiun" if "stasiun" in df.columns else None
    if station_col is None:
        raise ValueError(
            "Kolom stasiun tidak ditemukan. Pastikan CSV memiliki kolom 'stasiun'."
        )

    df["tanggal"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=["tanggal", "stasiun"]).copy()
    df["stasiun"] = df["stasiun"].astype(str).str.strip()

    available_pollutants = [p for p in POLLUTANTS if p in df.columns]
    if not available_pollutants:
        raise ValueError(
            "Tidak ada kolom polutan yang ditemukan. Minimal perlu salah satu: pm10, pm25, so2, co, o3, no2."
        )

    for p in available_pollutants:
        df[p] = pd.to_numeric(df[p], errors="coerce")

    flag_cols = [c for c in df.columns if c.startswith("flag_")]
    for c in flag_cols:
        df[c] = df[c].fillna(False).astype(bool)

    df["_record_marker"] = 1

    named_aggs = {p: (p, "max") for p in available_pollutants}
    for c in flag_cols:
        named_aggs[c] = (c, "max")
    named_aggs["record_source_count"] = ("_record_marker", "sum")

    data = (
        df.groupby(["tanggal", "stasiun"], dropna=False)
        .agg(**named_aggs)
        .reset_index()
        .sort_values(["tanggal", "stasiun"])
    )

    # Pastikan semua kolom polutan ada agar fungsi chart stabil.
    for p in POLLUTANTS:
        if p not in data.columns:
            data[p] = np.nan

    data["max_ispu"] = data[POLLUTANTS].max(axis=1, skipna=True)
    critical_col = data[POLLUTANTS].idxmax(axis=1, skipna=True)
    data["critical"] = critical_col.map(POLLUTANT_LABELS)
    data.loc[data["max_ispu"].isna(), "critical"] = np.nan

    data["kategori"] = data["max_ispu"].apply(categorize_ispu)
    data["is_unhealthy_or_worse"] = data["max_ispu"] > 100
    data["tahun"] = data["tanggal"].dt.year
    data["bulan"] = data["tanggal"].dt.month
    data["bulan_label"] = data["bulan"].map(MONTH_ID)
    data["bulan_nama"] = data["bulan"].map(MONTH_FULL_ID)
    data["hari"] = data["tanggal"].dt.day_name()
    data["musim_proxy"] = data["bulan"].apply(season_from_month)
    data["pm25_tersedia"] = data["pm25"].notna()

    if flag_cols:
        data["jumlah_flag_kualitas_dashboard"] = data[flag_cols].sum(axis=1).astype(int)
    else:
        data["jumlah_flag_kualitas_dashboard"] = 0

    data["punya_duplikat_sumber"] = data["record_source_count"] > 1
    return data


# -----------------------------
# Integrasi data cuaca opsional
# -----------------------------
def _normalize_column_name(col: str) -> str:
    return str(col).strip().lower().replace(" ", "_").replace("-", "_")


@st.cache_data(show_spinner=False)
def load_weather_from_bytes(file_bytes: bytes) -> pd.DataFrame:
    weather = pd.read_csv(io.BytesIO(file_bytes))
    original_cols = list(weather.columns)
    norm_map = {c: _normalize_column_name(c) for c in original_cols}
    reverse_norm = {v: k for k, v in norm_map.items()}

    date_candidates = ["tanggal", "date", "datetime", "waktu", "time", "tgl"]
    date_col = next(
        (reverse_norm[c] for c in date_candidates if c in reverse_norm), None
    )
    if date_col is None:
        raise ValueError("Data cuaca harus memiliki kolom tanggal/date.")

    rename_rules = {
        "curah_hujan_mm": [
            "curah_hujan",
            "curah_hujan_mm",
            "rainfall",
            "precipitation",
            "rr",
            "hujan",
        ],
        "suhu_c": ["suhu", "suhu_c", "temperature", "temp", "tavg", "t2m"],
        "kelembapan_pct": [
            "kelembapan",
            "kelembapan_pct",
            "humidity",
            "rh",
            "relative_humidity",
        ],
        "kecepatan_angin": [
            "kecepatan_angin",
            "wind_speed",
            "windspeed",
            "ff_avg",
            "ws",
        ],
    }

    selected = pd.DataFrame()
    selected["tanggal"] = pd.to_datetime(weather[date_col], errors="coerce")

    for standard_col, candidates in rename_rules.items():
        found_col = next(
            (reverse_norm[c] for c in candidates if c in reverse_norm), None
        )
        if found_col is not None:
            selected[standard_col] = pd.to_numeric(weather[found_col], errors="coerce")

    selected = selected.dropna(subset=["tanggal"])
    if selected.empty:
        raise ValueError("Kolom tanggal pada data cuaca tidak bisa dibaca.")

    metric_cols = [c for c in selected.columns if c != "tanggal"]
    if not metric_cols:
        raise ValueError(
            "Tidak ada kolom cuaca numerik yang terbaca. Gunakan nama seperti curah_hujan, suhu, kelembapan, atau wind_speed."
        )

    return selected.groupby("tanggal", as_index=False)[metric_cols].mean()


# -----------------------------
# Data loading
# -----------------------------
st.sidebar.title("⚙️ Pengaturan")
st.sidebar.caption(
    "Gunakan filter ini untuk menjawab pertanyaan stakeholder secara cepat."
)

default_path = find_default_data_path()
if default_path is None:
    st.error(
        "File data belum ditemukan. Letakkan `ispu_jakarta_analysis.csv` di folder yang sama dengan app.py atau unggah melalui sidebar."
    )
    st.stop()
raw_data = load_raw_data_from_path(default_path)
data_source_name = os.path.basename(default_path)

df_all = prepare_ispu_data(raw_data)

# -----------------------------
# Sidebar filters
# -----------------------------
min_date = df_all["tanggal"].min().date()
max_date = df_all["tanggal"].max().date()

st.sidebar.markdown(
    """
    <style>
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #F8FAFC 0%, #EEF6FF 100%);
    }
    section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p {
        color: #334155;
    }
    .sidebar-card {
        background: rgba(255, 255, 255, 0.88);
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 14px 14px;
        margin: 8px 0 14px 0;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
    }
    .sidebar-title {
        font-size: 1.08rem;
        font-weight: 850;
        color: #0F172A;
        margin-bottom: 2px;
    }
    .sidebar-subtitle {
        font-size: 0.82rem;
        color: #64748B;
        line-height: 1.35;
    }
    .filter-chip {
        display: inline-flex;
        align-items: center;
        background: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-radius: 999px;
        padding: 4px 9px;
        margin: 3px 4px 3px 0;
        font-size: 0.74rem;
        font-weight: 700;
        color: #334155;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# 1. Filter periode
# -----------------------------
st.sidebar.markdown("#### 📅 Periode analisis")

date_preset_options = [
    "Seluruh periode",
    "3 tahun terakhir",
    "5 tahun terakhir",
    "Periode PM2.5 lengkap (2021–2023)",
    "Tahun tertentu",
    "Kustom",
]

date_preset = st.sidebar.selectbox(
    "Pilih periode",
    date_preset_options,
    index=0,
    help="Gunakan preset agar lebih cepat. Pilih Kustom jika ingin menentukan tanggal sendiri.",
)

available_years = sorted(df_all["tahun"].dropna().astype(int).unique().tolist())

if date_preset == "Seluruh periode":
    start_date, end_date = min_date, max_date

elif date_preset == "3 tahun terakhir":
    last_year = max(available_years)
    first_year = max(min(available_years), last_year - 2)
    start_date = pd.Timestamp(year=first_year, month=1, day=1).date()
    end_date = max_date

elif date_preset == "5 tahun terakhir":
    last_year = max(available_years)
    first_year = max(min(available_years), last_year - 4)
    start_date = pd.Timestamp(year=first_year, month=1, day=1).date()
    end_date = max_date

elif date_preset == "Periode PM2.5 lengkap (2021–2023)":
    start_date = max(min_date, pd.Timestamp("2021-01-01").date())
    end_date = min(max_date, pd.Timestamp("2023-12-31").date())

elif date_preset == "Tahun tertentu":
    selected_year = st.sidebar.selectbox(
        "Pilih tahun",
        available_years,
        index=len(available_years) - 1,
    )
    start_date = max(min_date, pd.Timestamp(year=selected_year, month=1, day=1).date())
    end_date = min(max_date, pd.Timestamp(year=selected_year, month=12, day=31).date())

else:
    date_range = st.sidebar.date_input(
        "Pilih rentang tanggal",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        key="custom_date_range",
    )

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = end_date = date_range

st.sidebar.caption(f"Periode aktif: **{start_date}** s.d. **{end_date}**")

# -----------------------------
# 2. Filter stasiun
# -----------------------------
st.sidebar.markdown("#### 📍 Stasiun SPKU")

station_options = sorted(df_all["stasiun"].dropna().unique().tolist())
station_quick_options = ["Semua stasiun"] + station_options

station_quick = st.sidebar.selectbox(
    "Pilihan stasiun",
    station_quick_options,
    index=0,
    help="Pilih semua stasiun atau fokus ke satu stasiun tertentu.",
)

if station_quick == "Semua stasiun":
    selected_stations = station_options
else:
    selected_stations = [station_quick]

with st.sidebar.expander("Pilih beberapa stasiun", expanded=False):
    use_custom_station = st.checkbox(
        "Gunakan pilihan multi-stasiun",
        value=False,
        help="Aktifkan jika ingin membandingkan beberapa stasiun tertentu saja.",
    )

    if use_custom_station:
        selected_stations = st.multiselect(
            "Stasiun yang dianalisis",
            station_options,
            default=station_options,
        )

# -----------------------------
# 3. Filter musim/cuaca
# -----------------------------
st.sidebar.markdown("#### 🌦️ Musim/proxy cuaca")

season_options = ["Semua"] + sorted(df_all["musim_proxy"].dropna().unique().tolist())

selected_season = st.sidebar.selectbox(
    "Pilih musim",
    season_options,
    index=0,
    help="Proxy: Nov–Apr lebih basah/hujan, Mei–Okt lebih kering/kemarau.",
)

# -----------------------------
# 4. Filter kategori ISPU
# -----------------------------
st.sidebar.markdown("#### 🚦 Kategori ISPU")

category_present = [
    c for c in CATEGORY_ORDER if c in df_all["kategori"].dropna().unique()
]

category_filter_mode = st.sidebar.selectbox(
    "Fokus kategori",
    [
        "Semua kategori",
        "Aman relatif (Baik + Sedang)",
        "Tidak Sehat atau lebih buruk",
        "Kategori tertentu",
    ],
    index=0,
)

if category_filter_mode == "Semua kategori":
    selected_categories = category_present

elif category_filter_mode == "Aman relatif (Baik + Sedang)":
    selected_categories = [c for c in ["BAIK", "SEDANG"] if c in category_present]

elif category_filter_mode == "Tidak Sehat atau lebih buruk":
    selected_categories = [
        c
        for c in ["TIDAK SEHAT", "SANGAT TIDAK SEHAT", "BERBAHAYA"]
        if c in category_present
    ]

else:
    selected_categories = st.sidebar.multiselect(
        "Pilih kategori",
        category_present,
        default=category_present,
    )

# Tampilkan chip kategori aktif
category_chip_html = "".join(
    [
        f"""
        <span class="filter-chip">
            <span style="
                width: 9px;
                height: 9px;
                border-radius: 50%;
                background: {CATEGORY_COLORS.get(cat, "#94A3B8")};
                display: inline-block;
                margin-right: 6px;
            "></span>
            {cat}
        </span>
        """
        for cat in selected_categories
    ]
)

st.html(
    """
    <div style="margin-top: 4px; margin-bottom: 10px;">
    </div>
    """
)

# -----------------------------
# 5. Filter pencemar dominan
# -----------------------------
st.sidebar.markdown("#### 🏭 Pencemar dominan")

critical_options = sorted(df_all["critical"].dropna().unique().tolist())
critical_quick_options = ["Semua pencemar"] + critical_options

critical_quick = st.sidebar.selectbox(
    "Pilih pencemar",
    critical_quick_options,
    index=0,
)

if critical_quick == "Semua pencemar":
    selected_critical = critical_options
else:
    selected_critical = [critical_quick]

with st.sidebar.expander("Pilih beberapa pencemar", expanded=False):
    use_custom_critical = st.checkbox(
        "Gunakan pilihan multi-pencemar",
        value=False,
        help="Aktifkan jika ingin membandingkan beberapa pencemar dominan tertentu.",
    )

    if use_custom_critical:
        selected_critical = st.multiselect(
            "Pencemar dominan yang dianalisis",
            critical_options,
            default=critical_options,
        )

# -----------------------------
# 6. Filter kualitas data
# -----------------------------
st.sidebar.markdown("#### ✅ Kualitas data")

with st.sidebar.expander("Pengaturan kualitas data", expanded=False):
    max_flag = int(df_all["jumlah_flag_kualitas_dashboard"].max())

    if max_flag > 0:
        selected_max_flag = st.select_slider(
            "Maksimum jumlah flag kualitas",
            options=list(range(0, max_flag + 1)),
            value=max_flag,
            help="Nilai 0 berarti hanya data tanpa flag kualitas. Default memakai seluruh data terkonsolidasi.",
        )
    else:
        selected_max_flag = 0
        st.info("Seluruh data terfilter tidak memiliki flag kualitas tambahan.")

    only_pm25_period = st.checkbox(
        "Fokus hanya pada data dengan PM2.5 tersedia",
        value=False,
        help="Aktifkan jika ingin membaca insight periode modern ketika PM2.5 sudah tersedia lebih konsisten.",
    )

# -----------------------------
# 7. Terapkan filter
# -----------------------------
filtered = df_all.copy()

filtered = filtered[
    (filtered["tanggal"].dt.date >= start_date)
    & (filtered["tanggal"].dt.date <= end_date)
]

filtered = filtered[filtered["stasiun"].isin(selected_stations)]
filtered = filtered[filtered["kategori"].isin(selected_categories)]
filtered = filtered[filtered["critical"].isin(selected_critical)]
filtered = filtered[filtered["jumlah_flag_kualitas_dashboard"] <= selected_max_flag]

if selected_season != "Semua":
    filtered = filtered[filtered["musim_proxy"] == selected_season]

if only_pm25_period:
    filtered = filtered[filtered["pm25_tersedia"]]

# -----------------------------
# 8. Ringkasan filter aktif
# -----------------------------
st.sidebar.divider()

st.sidebar.markdown("#### 📌 Ringkasan aktif")

st.sidebar.markdown(
    f"""
    <div class="sidebar-card">
        <div class="sidebar-subtitle">
            <b>Observasi:</b> {len(filtered):,.0f}<br>
            <b>Stasiun:</b> {len(selected_stations)} dari {len(station_options)}<br>
            <b>Kategori:</b> {len(selected_categories)} aktif<br>
            <b>Pencemar:</b> {len(selected_critical)} aktif
        </div>
    </div>
    """.replace(",", "."),
    unsafe_allow_html=True,
)

if filtered.empty:
    st.warning(
        "Tidak ada data pada kombinasi filter saat ini. Longgarkan filter di sidebar."
    )
    st.stop()

# -----------------------------
# Header
# -----------------------------
st.html(
    "<div style='margin-top: 1rem' class='main-title'>Dashboard Kualitas Udara Jakarta Berbasis ISPU</div>",
)
st.html(
    f"<div class='subtitle'>Data: <b>{data_source_name}</b> · Periode terfilter: <b>{start_date}</b> s.d. <b>{end_date}</b></div>",
)

with st.expander("📘 Data dictionary singkat & ambang kategori ISPU", expanded=False):
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown(
            """
            **Kolom utama yang digunakan**
            
            - `tanggal`: tanggal pengukuran setelah validasi.
            - `stasiun`: lokasi SPKU DKI1–DKI5.
            - `pm10`, `pm25`, `so2`, `co`, `o3`, `no2`: nilai indeks parameter polutan.
            - `max_ispu`: nilai ISPU harian tertinggi hasil hitung ulang dari seluruh parameter.
            - `critical`: parameter dengan nilai indeks tertinggi pada tanggal-stasiun tersebut.
            - `kategori`: kategori ISPU berdasarkan `max_ispu`.
            - `musim_proxy`: pengelompokan sederhana Nov–Apr sebagai hujan/lebih basah dan Mei–Okt sebagai kemarau/lebih kering.
            """
        )
    with c2:
        threshold_table = pd.DataFrame(
            {
                "Kategori": [
                    "BAIK",
                    "SEDANG",
                    "TIDAK SEHAT",
                    "SANGAT TIDAK SEHAT",
                    "BERBAHAYA",
                ],
                "Rentang ISPU": ["0–50", "51–100", "101–200", "201–300", ">300"],
                "Makna praktis": [
                    "Risiko rendah",
                    "Masih diterima, kelompok sensitif tetap perlu perhatian",
                    "Mulai berdampak pada kesehatan",
                    "Perlu pengurangan aktivitas luar ruang",
                    "Kondisi darurat kesehatan lingkungan",
                ],
            }
        )
        st.dataframe(threshold_table, use_container_width=True, hide_index=True)


# -----------------------------
# Fungsi agregasi metrik
# -----------------------------
def daily_city_metrics(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data.groupby("tanggal", as_index=False)
        .agg(
            rata_ispu=("max_ispu", "mean"),
            ispu_terburuk=("max_ispu", "max"),
            jumlah_stasiun=("stasiun", "nunique"),
            jumlah_stasiun_tidak_sehat=("is_unhealthy_or_worse", "sum"),
        )
        .assign(ada_stasiun_tidak_sehat=lambda d: d["jumlah_stasiun_tidak_sehat"] > 0)
    )


def station_summary(data: pd.DataFrame) -> pd.DataFrame:
    summary = (
        data.groupby("stasiun", as_index=False)
        .agg(
            rata_ispu=("max_ispu", "mean"),
            median_ispu=("max_ispu", "median"),
            p95_ispu=("max_ispu", lambda x: np.nanpercentile(x, 95)),
            hari_stasiun=("tanggal", "nunique"),
            pct_tidak_sehat=("is_unhealthy_or_worse", lambda x: x.mean() * 100),
            pencemar_dominan=("critical", mode_or_dash),
        )
        .sort_values("rata_ispu", ascending=False)
    )
    return summary


def category_distribution(data: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    if by is None:
        dist = (
            data["kategori"]
            .value_counts()
            .reindex(CATEGORY_ORDER)
            .dropna()
            .reset_index()
        )
        dist.columns = ["kategori", "jumlah"]
        dist["persen"] = dist["jumlah"] / dist["jumlah"].sum() * 100
        return dist
    dist = data.groupby([by, "kategori"]).size().reset_index(name="jumlah")
    total = dist.groupby(by)["jumlah"].transform("sum")
    dist["persen"] = dist["jumlah"] / total * 100
    dist["kategori"] = pd.Categorical(
        dist["kategori"], categories=CATEGORY_ORDER, ordered=True
    )
    return dist.sort_values([by, "kategori"])


# -----------------------------
# Tabs dashboard
# -----------------------------
tabs = st.tabs(
    [
        "1 · Overview",
        "2 · Tren Temporal",
        "3 · Antar Stasiun",
        "4 · Pencemar Kritis",
        "5 · Pola Musiman",
    ]
)

# ============================================================
# Dashboard 1 — Overview
# ============================================================
with tabs[0]:
    st.subheader("Overview Kualitas Udara Jakarta")

    city_daily = daily_city_metrics(filtered)
    avg_ispu = filtered["max_ispu"].mean()
    station_day_unhealthy = filtered["is_unhealthy_or_worse"].mean() * 100
    day_any_unhealthy = city_daily["ada_stasiun_tidak_sehat"].mean() * 100
    dominant_pollutant = mode_or_dash(filtered["critical"])

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card(
            "Rata-rata ISPU",
            safe_num(avg_ispu, 1),
            "Nilai rerata pada seluruh stasiun dan tanggal terfilter.",
        )
    with c2:
        kpi_card(
            "Stasiun-hari Tidak Sehat+",
            safe_pct(station_day_unhealthy, 1),
            "Proporsi observasi tanggal × stasiun dengan ISPU > 100.",
        )
    with c3:
        kpi_card(
            "Hari Ada Titik Tidak Sehat+",
            safe_pct(day_any_unhealthy, 1),
            "Hari ketika minimal satu stasiun melewati ISPU 100.",
        )
    with c4:
        kpi_card(
            "Pencemar Dominan",
            dominant_pollutant,
            "Parameter yang paling sering menjadi nilai indeks tertinggi.",
        )

    if avg_ispu > 100 or day_any_unhealthy >= 30:
        insight_box(
            f"<b>Rekomendasi kebijakan:</b> Terapkan peringatan harian kualitas udara berbasis stasiun terburuk agar warga segera tahu kapan harus mengurangi aktivitas luar ruang, memakai masker, atau melindungi kelompok rentan.",
            kind="warning",
        )
    else:
        insight_box(
            f"<b>Interpretasi:</b> Rata-rata ISPU berada pada <b>{safe_num(avg_ispu)}</b>. Tetap perlu pemantauan wilayah karena <b>{safe_pct(day_any_unhealthy)}</b> hari memiliki minimal satu titik Tidak Sehat.",
        )

    left, right = st.columns([1.05, 1])
    with left:
        dist = category_distribution(filtered).copy()

        dist["kategori"] = pd.Categorical(
            dist["kategori"],
            categories=CATEGORY_ORDER,
            ordered=True,
        )

        dist = dist.sort_values("kategori")

        # Label hanya ditampilkan jika proporsi cukup besar agar tidak bertumpuk
        dist["label_persen"] = dist["persen"].apply(
            lambda x: f"{x:.1f}%" if pd.notna(x) and x >= 3 else ""
        )

        st.html(
            """
            <div style="
                background: linear-gradient(135deg, #F8FAFC 0%, #EEF6FF 100%);
                border: 1px solid #E2E8F0;
                border-radius: 16px;
                padding: 14px 16px;
                margin-bottom: 10px;
                box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
            ">
                <div style="font-size: 0.78rem; color: #64748B; font-weight: 700; letter-spacing: .04em; text-transform: uppercase;">
                    Distribusi kategori ISPU
                </div>
                <div style="font-size: 1.15rem; color: #0F172A; font-weight: 850; margin-top: 2px;">
                    Komposisi kualitas udara pada data terfilter
                </div>
                <div style="font-size: 0.86rem; color: #475569; margin-top: 4px;">
                    Warna mengikuti kategori ISPU. Semakin besar porsi kategori Tidak Sehat ke atas, semakin tinggi risiko kebijakan yang perlu ditangani.
                </div>
            </div>
            """,
        )

        fig = px.bar(
            dist,
            x="persen",
            y="kategori",
            orientation="h",
            color="kategori",
            color_discrete_map=CATEGORY_COLORS,
            text="label_persen",
            category_orders={"kategori": CATEGORY_ORDER},
            labels={
                "kategori": "",
                "persen": "Proporsi observasi (%)",
            },
            custom_data=["jumlah", "persen"],
        )

        fig.update_traces(
            textposition="outside",
            cliponaxis=False,
            marker_line=dict(color="rgba(255,255,255,0.9)", width=1),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Proporsi: %{customdata[1]:.1f}%<br>"
                "Jumlah observasi: %{customdata[0]:,}<br>"
                "<extra></extra>"
            ),
        )

        fig.update_layout(
            height=420,
            template="plotly_white",
            showlegend=False,
            margin=dict(l=10, r=52, t=10, b=55),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0)",
            font=dict(color="#334155", size=12),
            bargap=0.34,
            uniformtext_minsize=10,
            uniformtext_mode="hide",
        )

        fig.update_xaxes(
            range=[0, min(100, max(10, dist["persen"].max() * 1.18))],
            ticksuffix="%",
            showgrid=True,
            gridcolor="rgba(226, 232, 240, 0.85)",
            zeroline=False,
            title_standoff=12,
            automargin=True,
        )

        fig.update_yaxes(
            automargin=True,
            tickfont=dict(size=11),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False},
        )

with right:
    latest_date = filtered["tanggal"].max()
    latest = filtered[filtered["tanggal"] == latest_date].copy()

    latest["lat"] = latest["stasiun"].map(
        lambda s: STATION_COORDS.get(s, {}).get("lat")
    )
    latest["lon"] = latest["stasiun"].map(
        lambda s: STATION_COORDS.get(s, {}).get("lon")
    )
    latest["wilayah"] = latest["stasiun"].map(
        lambda s: STATION_COORDS.get(s, {}).get("wilayah", "-")
    )

    latest_map = latest.dropna(subset=["lat", "lon"]).copy()

    if not latest_map.empty:
        latest_map["kode_stasiun"] = (
            latest_map["stasiun"]
            .astype(str)
            .str.extract(r"(DKI\d+)")[0]
            .fillna(latest_map["stasiun"].astype(str))
        )

        st.html(
            f"""
            <div style="
                background: linear-gradient(135deg, #F8FAFC 0%, #EEF6FF 100%);
                border: 1px solid #E2E8F0;
                border-radius: 16px;
                padding: 14px 16px;
                margin-bottom: 10px;
                box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
            ">
                <div style="font-size: 0.78rem; color: #64748B; font-weight: 700; letter-spacing: .04em; text-transform: uppercase;">
                    Kondisi terkini per stasiun
                </div>
                <div style="font-size: 1.15rem; color: #0F172A; font-weight: 850; margin-top: 2px;">
                    Ringkasan lokasi SPKU · {latest_date:%d %b %Y}
                </div>
                <div style="font-size: 0.86rem; color: #475569; margin-top: 4px;">
                    Ukuran titik menunjukkan nilai ISPU. Warna titik menunjukkan kategori kualitas udara.
                </div>
            </div>
            """,
        )

        used_categories = [
            cat for cat in CATEGORY_ORDER if cat in latest_map["kategori"].unique()
        ]
        chip_html = "".join(
            [
                f"""
                <span style="
                    display: inline-flex;
                    align-items: center;
                    gap: 6px;
                    background: #FFFFFF;
                    border: 1px solid #E2E8F0;
                    border-radius: 999px;
                    padding: 5px 10px;
                    margin: 0 6px 8px 0;
                    font-size: 0.78rem;
                    font-weight: 700;
                    color: #334155;
                    box-shadow: 0 3px 10px rgba(15, 23, 42, 0.05);
                ">
                    <span style="
                        width: 10px;
                        height: 10px;
                        border-radius: 50%;
                        background: {CATEGORY_COLORS.get(cat, "#94A3B8")};
                        display: inline-block;
                    "></span>
                    {cat}
                </span>
                """
                for cat in used_categories
            ]
        )

        st.html(
            f"<div style='margin-bottom: 4px;'>{chip_html}</div>",
        )

        fig_map = px.scatter_mapbox(
            latest_map,
            lat="lat",
            lon="lon",
            size="max_ispu",
            size_max=34,
            color="kategori",
            color_discrete_map=CATEGORY_COLORS,
            text="kode_stasiun",
            hover_name="stasiun",
            hover_data={
                "wilayah": True,
                "max_ispu": ":.0f",
                "kategori": True,
                "critical": True,
                "lat": False,
                "lon": False,
                "kode_stasiun": False,
            },
            zoom=9.6,
            height=430,
        )

        fig_map.update_traces(
            textposition="top center",
            textfont=dict(size=12, color="#0F172A"),
            marker=dict(opacity=0.88),
            selector=dict(type="scattermapbox"),
        )

        fig_map.update_layout(
            mapbox_style="carto-positron",
            showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(
            fig_map,
            use_container_width=True,
            config={
                "displayModeBar": False,
                "scrollZoom": True,
            },
        )
    else:
        st.info(
            "Koordinat stasiun belum tersedia untuk peta. Tabel ringkasan tetap ditampilkan di bawah."
        )

    latest_table = (
        latest[["tanggal", "stasiun", "wilayah", "max_ispu", "kategori", "critical"]]
        .sort_values("max_ispu", ascending=False)
        .rename(
            columns={
                "tanggal": "Tanggal",
                "stasiun": "Stasiun",
                "wilayah": "Wilayah",
                "max_ispu": "ISPU",
                "kategori": "Kategori",
                "critical": "Pencemar dominan",
            }
        )
    )
    st.dataframe(latest_table, use_container_width=True, hide_index=True)

# ============================================================
# Dashboard 2 — Tren Temporal
# ============================================================
with tabs[1]:
    st.subheader("Tren Temporal Kualitas Udara")

    st.html(
        """
        <div style="
            background: linear-gradient(135deg, #F8FAFC 0%, #EEF6FF 100%);
            border: 1px solid #E2E8F0;
            border-radius: 16px;
            padding: 14px 16px;
            margin-bottom: 14px;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
        ">
            <div style="font-size: 0.78rem; color: #64748B; font-weight: 700; letter-spacing: .04em; text-transform: uppercase;">
                Tren kualitas udara
            </div>
            <div style="font-size: 1.15rem; color: #0F172A; font-weight: 850; margin-top: 2px;">
                Pergerakan nilai ISPU dari waktu ke waktu
            </div>
            <div style="font-size: 0.86rem; color: #475569; margin-top: 4px;">
                Gunakan granularitas harian, bulanan, atau tahunan untuk melihat pola jangka pendek dan jangka panjang. Garis merah putus-putus menunjukkan ambang Tidak Sehat.
            </div>
        </div>
        """,
    )

    col_a, col_b = st.columns([0.75, 1.25])

    with col_a:
        granularity = st.radio(
            "Granularitas tren",
            ["Harian", "Bulanan", "Tahunan"],
            index=1,
            horizontal=True,
        )

    with col_b:
        trend_mode = st.radio(
            "Mode pembacaan",
            ["Rata-rata kota", "Bandingkan antar stasiun"],
            index=0,
            horizontal=True,
        )

    temp = filtered.copy()

    if granularity == "Harian":
        temp["periode"] = temp["tanggal"]
        period_label = "Tanggal"
        x_tickformat = "%d %b<br>%Y"
        marker_mode = False
        tick_angle = 0
        max_ticks = 10
    elif granularity == "Bulanan":
        temp["periode"] = temp["tanggal"].dt.to_period("M").dt.to_timestamp()
        period_label = "Bulan"
        x_tickformat = "%b<br>%Y"
        marker_mode = True
        tick_angle = 0
        max_ticks = 12
    else:
        temp["periode"] = temp["tanggal"].dt.to_period("Y").dt.to_timestamp()
        period_label = "Tahun"
        x_tickformat = "%Y"
        marker_mode = True
        tick_angle = 0
        max_ticks = 14

    if trend_mode == "Rata-rata kota":
        trend = (
            temp.groupby("periode", as_index=False)
            .agg(
                rata_ispu=("max_ispu", "mean"),
                ispu_terburuk=("max_ispu", "max"),
                pct_tidak_sehat=("is_unhealthy_or_worse", lambda x: x.mean() * 100),
                jumlah_observasi=("max_ispu", "count"),
            )
            .sort_values("periode")
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=trend["periode"],
                y=trend["rata_ispu"],
                mode="lines+markers" if marker_mode else "lines",
                name="Rata-rata ISPU",
                line=dict(width=3, color="#0F766E", shape="spline"),
                marker=dict(
                    size=7,
                    color="#0F766E",
                    line=dict(color="white", width=2),
                ),
                fill="tozeroy",
                fillcolor="rgba(15, 118, 110, 0.08)",
                customdata=np.stack(
                    [
                        trend["pct_tidak_sehat"],
                        trend["ispu_terburuk"],
                        trend["jumlah_observasi"],
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "<b>%{x|%d %b %Y}</b><br>"
                    "Rata-rata ISPU: %{y:.1f}<br>"
                    "ISPU terburuk: %{customdata[1]:.0f}<br>"
                    "Tidak Sehat+: %{customdata[0]:.1f}%<br>"
                    "Observasi: %{customdata[2]:,.0f}<br>"
                    "<extra></extra>"
                ),
            )
        )

        fig.add_hline(
            y=100,
            line_dash="dash",
            line_width=1.5,
            line_color="#DC2626",
            opacity=0.75,
        )

        fig.add_annotation(
            x=1,
            y=100,
            xref="paper",
            yref="y",
            text="Ambang Tidak Sehat = 100",
            showarrow=False,
            xanchor="right",
            yanchor="bottom",
            font=dict(size=11, color="#991B1B"),
            bgcolor="rgba(255,255,255,0.88)",
            bordercolor="rgba(220,38,38,0.25)",
            borderwidth=1,
            borderpad=4,
        )

    else:
        trend = (
            temp.groupby(["periode", "stasiun"], as_index=False)
            .agg(
                rata_ispu=("max_ispu", "mean"),
                pct_tidak_sehat=("is_unhealthy_or_worse", lambda x: x.mean() * 100),
                jumlah_observasi=("max_ispu", "count"),
            )
            .sort_values(["periode", "stasiun"])
        )

        fig = px.line(
            trend,
            x="periode",
            y="rata_ispu",
            color="stasiun",
            markers=marker_mode,
            labels={
                "periode": period_label,
                "rata_ispu": "Rata-rata ISPU",
                "stasiun": "Stasiun",
            },
            custom_data=["pct_tidak_sehat", "jumlah_observasi"],
        )

        fig.update_traces(
            line=dict(width=2.7, shape="spline"),
            marker=dict(size=6, line=dict(color="white", width=1.5)),
            hovertemplate=(
                "<b>%{fullData.name}</b><br>"
                "Periode: %{x|%d %b %Y}<br>"
                "Rata-rata ISPU: %{y:.1f}<br>"
                "Tidak Sehat+: %{customdata[0]:.1f}%<br>"
                "Observasi: %{customdata[1]:,.0f}<br>"
                "<extra></extra>"
            ),
        )

        fig.add_hline(
            y=100,
            line_dash="dash",
            line_width=1.5,
            line_color="#DC2626",
            opacity=0.75,
        )

        fig.add_annotation(
            x=1,
            y=100,
            xref="paper",
            yref="y",
            text="Ambang Tidak Sehat = 100",
            showarrow=False,
            xanchor="right",
            yanchor="bottom",
            font=dict(size=11, color="#991B1B"),
            bgcolor="rgba(255,255,255,0.88)",
            bordercolor="rgba(220,38,38,0.25)",
            borderwidth=1,
            borderpad=4,
        )

    fig.update_layout(
        height=470,
        template="plotly_white",
        margin=dict(l=16, r=22, t=12, b=95),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        font=dict(color="#334155", size=12),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.20,
            xanchor="center",
            x=0.5,
            title_text="",
            font=dict(size=11),
            bgcolor="rgba(255,255,255,0)",
        ),
    )

    fig.update_xaxes(
        title_text="",
        tickformat=x_tickformat,
        nticks=max_ticks,
        tickangle=tick_angle,
        showgrid=False,
        automargin=True,
        rangeslider=dict(visible=True) if granularity == "Harian" else None,
    )

    fig.update_yaxes(
        title_text="Rata-rata ISPU",
        rangemode="tozero",
        showgrid=True,
        gridcolor="rgba(226, 232, 240, 0.85)",
        zeroline=False,
        title_standoff=12,
        automargin=True,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": False},
    )

    annual = filtered.groupby("tahun", as_index=False).agg(
        rata_ispu=("max_ispu", "mean"),
        pct_tidak_sehat=("is_unhealthy_or_worse", lambda x: x.mean() * 100),
        jumlah_stasiun=("stasiun", "nunique"),
        observasi=("tanggal", "count"),
    )
    annual["tahun"] = annual["tahun"].astype(str)

    c1, c2 = st.columns([1.05, 1])
    with c1:
        st.html(
            """
            <div style="
                background: linear-gradient(135deg, #F8FAFC 0%, #EEF6FF 100%);
                border: 1px solid #E2E8F0;
                border-radius: 16px;
                padding: 14px 16px;
                margin-bottom: 10px;
                box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
            ">
                <div style="font-size: 0.78rem; color: #64748B; font-weight: 700; letter-spacing: .04em; text-transform: uppercase;">
                    Tren tahunan
                </div>
                <div style="font-size: 1.15rem; color: #0F172A; font-weight: 850; margin-top: 2px;">
                    Perbandingan nilai ISPU dan risiko Tidak Sehat+
                </div>
                <div style="font-size: 0.86rem; color: #475569; margin-top: 4px;">
                    Bar menunjukkan rata-rata ISPU tahunan. Garis menunjukkan persentase observasi Tidak Sehat atau lebih buruk.
                </div>
            </div>
            """,
        )

        annual_plot = annual.copy()
        annual_plot["tahun"] = annual_plot["tahun"].astype(str)

        max_pct = annual_plot["pct_tidak_sehat"].max()
        max_pct = 0 if pd.isna(max_pct) else max_pct

        fig_year = make_subplots(specs=[[{"secondary_y": True}]])

        fig_year.add_trace(
            go.Bar(
                x=annual_plot["tahun"],
                y=annual_plot["rata_ispu"],
                name="Rata-rata ISPU",
                marker=dict(
                    color=annual_plot["rata_ispu"],
                    colorscale=[
                        [0.00, "#22C55E"],
                        [0.45, "#FACC15"],
                        [0.70, "#FB923C"],
                        [1.00, "#DC2626"],
                    ],
                    line=dict(color="rgba(15, 23, 42, 0.15)", width=1),
                ),
                hovertemplate=(
                    "<b>Tahun %{x}</b><br>Rata-rata ISPU: %{y:.1f}<br><extra></extra>"
                ),
            ),
            secondary_y=False,
        )

        fig_year.add_trace(
            go.Scatter(
                x=annual_plot["tahun"],
                y=annual_plot["pct_tidak_sehat"],
                name="% Tidak Sehat+",
                mode="lines+markers",
                line=dict(width=3, color="#0F766E", shape="spline"),
                marker=dict(
                    size=8,
                    color="#0F766E",
                    line=dict(color="white", width=2),
                ),
                hovertemplate=(
                    "<b>Tahun %{x}</b><br>Tidak Sehat+: %{y:.1f}%<br><extra></extra>"
                ),
            ),
            secondary_y=True,
        )

        fig_year.add_hline(
            y=100,
            line_dash="dash",
            line_width=1.4,
            line_color="#DC2626",
            opacity=0.75,
            secondary_y=False,
        )

        fig_year.add_annotation(
            x=1,
            y=100,
            xref="paper",
            yref="y",
            text="Ambang Tidak Sehat = 100",
            showarrow=False,
            xanchor="right",
            yanchor="bottom",
            font=dict(size=11, color="#991B1B"),
            bgcolor="rgba(255,255,255,0.86)",
            bordercolor="rgba(220,38,38,0.25)",
            borderwidth=1,
            borderpad=4,
        )

        fig_year.update_yaxes(
            title_text="Rata-rata ISPU",
            secondary_y=False,
            rangemode="tozero",
            showgrid=True,
            gridcolor="rgba(226, 232, 240, 0.8)",
            title_standoff=12,
            automargin=True,
        )

        fig_year.update_yaxes(
            title_text="Tidak Sehat+ (%)",
            secondary_y=True,
            range=[0, max(100, max_pct * 1.25)],
            ticksuffix="%",
            showgrid=False,
            title_standoff=12,
            automargin=True,
        )

        fig_year.update_xaxes(
            title_text="",
            tickangle=0 if len(annual_plot) <= 10 else -45,
            tickfont=dict(size=11),
            automargin=True,
        )

        fig_year.update_layout(
            template="plotly_white",
            height=455,
            margin=dict(l=20, r=28, t=18, b=70),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0)",
            font=dict(color="#334155", size=12),
            hovermode="x unified",
            bargap=0.35,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.20,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255,255,255,0)",
                font=dict(size=12),
            ),
        )

        st.plotly_chart(
            fig_year,
            use_container_width=True,
            config={"displayModeBar": False},
        )
    with c2:
        if len(annual) >= 2:
            first_year = annual.iloc[0]
            last_year = annual.iloc[-1]
            delta_ispu = last_year["rata_ispu"] - first_year["rata_ispu"]
            delta_risk = last_year["pct_tidak_sehat"] - first_year["pct_tidak_sehat"]
            arah = "memburuk" if delta_ispu > 0 else "membaik"
            insight_box(
                f"<b>Pembacaan tren:</b> Dari tahun awal ke tahun akhir pada filter aktif, rata-rata ISPU bergerak <b>{arah}</b> sebesar <b>{delta_ispu:+.1f} poin</b>. Proporsi observasi Tidak Sehat+ berubah <b>{delta_risk:+.1f} poin persentase</b>. Gunakan grafik ini untuk melihat periode lonjakan, bukan hanya rata-rata akhir.",
                kind="warning" if delta_ispu > 0 else "insight",
            )
        st.dataframe(
            annual.rename(
                columns={
                    "tahun": "Tahun",
                    "rata_ispu": "Rata-rata ISPU",
                    "pct_tidak_sehat": "% Tidak Sehat+",
                    "jumlah_stasiun": "Jumlah stasiun",
                    "observasi": "Observasi",
                }
            ).round(1),
            use_container_width=True,
            hide_index=True,
        )

# ============================================================
# Dashboard 3 — Perbandingan antar stasiun
# ============================================================
with tabs[2]:
    st.subheader("Perbandingan Kualitas Udara Antar Stasiun")

    summary = station_summary(filtered)
    worst_station = summary.iloc[0]
    best_station = summary.iloc[-1]

    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card(
            "Stasiun prioritas",
            worst_station["stasiun"],
            f"Rata-rata ISPU {worst_station['rata_ispu']:.1f}; Tidak Sehat+ {worst_station['pct_tidak_sehat']:.1f}%.",
        )
    with c2:
        kpi_card(
            "Stasiun relatif terbaik",
            best_station["stasiun"],
            f"Rata-rata ISPU {best_station['rata_ispu']:.1f}; Tidak Sehat+ {best_station['pct_tidak_sehat']:.1f}%.",
        )
    with c3:
        kpi_card(
            "Selisih risiko",
            f"{worst_station['pct_tidak_sehat'] - best_station['pct_tidak_sehat']:.1f} pp",
            "Perbedaan proporsi Tidak Sehat+ antara stasiun prioritas dan terbaik.",
        )

    insight_box(
        f"<b>Rekomendasi kebijakan:</b> Intervensi sebaiknya diprioritaskan ke <b>{worst_station['stasiun']}</b> karena stasiun ini memiliki kombinasi rata-rata ISPU dan frekuensi Tidak Sehat+ paling tinggi pada filter aktif. Namun, pembacaan tetap perlu dikaitkan dengan sumber emisi sekitar stasiun.",
        kind="warning" if worst_station["pct_tidak_sehat"] >= 20 else "insight",
    )

    c1, c2 = st.columns([1, 1.15])
    with c1:
        station_plot = summary.copy().sort_values("rata_ispu", ascending=True)

        station_plot["label_ispu"] = station_plot["rata_ispu"].apply(
            lambda x: f"{x:.1f}" if pd.notna(x) else ""
        )

        max_ispu_station = station_plot["rata_ispu"].max()
        max_ispu_station = 0 if pd.isna(max_ispu_station) else max_ispu_station

        st.html(
            """
            <div style="
                background: linear-gradient(135deg, #F8FAFC 0%, #EEF6FF 100%);
                border: 1px solid #E2E8F0;
                border-radius: 16px;
                padding: 14px 16px;
                margin-bottom: 10px;
                box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
            ">
                <div style="font-size: 0.78rem; color: #64748B; font-weight: 700; letter-spacing: .04em; text-transform: uppercase;">
                    Ranking stasiun
                </div>
                <div style="font-size: 1.15rem; color: #0F172A; font-weight: 850; margin-top: 2px;">
                    Rata-rata ISPU per stasiun SPKU
                </div>
                <div style="font-size: 0.86rem; color: #475569; margin-top: 4px;">
                    Bar yang lebih panjang menunjukkan kualitas udara yang lebih buruk. Garis merah menunjukkan ambang Tidak Sehat.
                </div>
            </div>
            """,
        )

        fig_station = go.Figure()

        fig_station.add_trace(
            go.Bar(
                x=station_plot["rata_ispu"],
                y=station_plot["stasiun"],
                orientation="h",
                text=station_plot["label_ispu"],
                textposition="outside",
                marker=dict(
                    color=station_plot["rata_ispu"],
                    colorscale=[
                        [0.00, "#22C55E"],
                        [0.45, "#FACC15"],
                        [0.70, "#FB923C"],
                        [1.00, "#DC2626"],
                    ],
                    line=dict(color="rgba(255,255,255,0.95)", width=1),
                ),
                customdata=np.stack(
                    [
                        station_plot["pct_tidak_sehat"],
                        station_plot["pencemar_dominan"].astype(str),
                        station_plot["median_ispu"],
                        station_plot["p95_ispu"],
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Rata-rata ISPU: %{x:.1f}<br>"
                    "Median ISPU: %{customdata[2]:.1f}<br>"
                    "P95 ISPU: %{customdata[3]:.1f}<br>"
                    "Tidak Sehat+: %{customdata[0]:.1f}%<br>"
                    "Pencemar dominan: %{customdata[1]}<br>"
                    "<extra></extra>"
                ),
            )
        )

        fig_station.add_vline(
            x=100,
            line_dash="dash",
            line_width=1.5,
            line_color="#DC2626",
            opacity=0.75,
        )

        fig_station.add_annotation(
            x=100,
            y=1.04,
            xref="x",
            yref="paper",
            text="Ambang Tidak Sehat = 100",
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            font=dict(size=11, color="#991B1B"),
            bgcolor="rgba(255,255,255,0.88)",
            bordercolor="rgba(220,38,38,0.25)",
            borderwidth=1,
            borderpad=4,
        )

        fig_station.update_layout(
            height=470,
            template="plotly_white",
            margin=dict(l=10, r=60, t=12, b=55),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0)",
            font=dict(color="#334155", size=12),
            showlegend=False,
            bargap=0.34,
            uniformtext_minsize=10,
            uniformtext_mode="hide",
        )

        fig_station.update_xaxes(
            range=[0, max(110, max_ispu_station * 1.18)],
            title_text="Rata-rata ISPU",
            showgrid=True,
            gridcolor="rgba(226, 232, 240, 0.85)",
            zeroline=False,
            title_standoff=12,
            automargin=True,
        )

        fig_station.update_yaxes(
            title_text="",
            automargin=True,
            tickfont=dict(size=11),
        )

        st.plotly_chart(
            fig_station,
            use_container_width=True,
            config={"displayModeBar": False},
        )
    with c2:
        dist_station = category_distribution(filtered, by="stasiun").copy()

        dist_station["kategori"] = pd.Categorical(
            dist_station["kategori"],
            categories=CATEGORY_ORDER,
            ordered=True,
        )

        # Urutkan stasiun berdasarkan proporsi Tidak Sehat+ agar insight langsung terlihat
        risk_order = (
            dist_station[
                dist_station["kategori"]
                .astype(str)
                .isin(["TIDAK SEHAT", "SANGAT TIDAK SEHAT", "BERBAHAYA"])
            ]
            .groupby("stasiun", observed=False)["persen"]
            .sum()
            .sort_values(ascending=True)
            .index.tolist()
        )

        if not risk_order:
            risk_order = sorted(dist_station["stasiun"].dropna().unique().tolist())

        dist_station["stasiun"] = pd.Categorical(
            dist_station["stasiun"],
            categories=risk_order,
            ordered=True,
        )

        dist_station = dist_station.sort_values(["stasiun", "kategori"])

        # Label hanya ditampilkan untuk segmen yang cukup besar agar tidak bertumpuk
        dist_station["label_persen"] = dist_station["persen"].apply(
            lambda x: f"{x:.1f}%" if pd.notna(x) and x >= 7 else ""
        )

        st.html(
            """
            <div style="
                background: linear-gradient(135deg, #F8FAFC 0%, #EEF6FF 100%);
                border: 1px solid #E2E8F0;
                border-radius: 16px;
                padding: 14px 16px;
                margin-bottom: 10px;
                box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
            ">
                <div style="font-size: 0.78rem; color: #64748B; font-weight: 700; letter-spacing: .04em; text-transform: uppercase;">
                    Komposisi kategori per stasiun
                </div>
                <div style="font-size: 1.15rem; color: #0F172A; font-weight: 850; margin-top: 2px;">
                    Distribusi kualitas udara berdasarkan kategori ISPU
                </div>
                <div style="font-size: 0.86rem; color: #475569; margin-top: 4px;">
                    Semakin besar porsi oranye/merah/ungu, semakin tinggi prioritas pengendalian polusi di stasiun tersebut.
                </div>
            </div>
            """,
        )

        fig_stack = px.bar(
            dist_station,
            x="persen",
            y="stasiun",
            color="kategori",
            orientation="h",
            text="label_persen",
            color_discrete_map=CATEGORY_COLORS,
            category_orders={
                "kategori": CATEGORY_ORDER,
                "stasiun": risk_order,
            },
            labels={
                "stasiun": "",
                "persen": "Proporsi observasi (%)",
                "kategori": "Kategori",
            },
            hover_data={
                "stasiun": True,
                "kategori": True,
                "persen": ":.1f",
                "jumlah": ":,",
                "label_persen": False,
            },
        )

        fig_stack.update_layout(
            barmode="stack",
            height=500,
            template="plotly_white",
            margin=dict(l=10, r=18, t=10, b=90),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0)",
            font=dict(color="#334155", size=12),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.18,
                xanchor="center",
                x=0.5,
                title_text="",
                font=dict(size=11),
                bgcolor="rgba(255,255,255,0)",
            ),
            hovermode="y unified",
            uniformtext_minsize=10,
            uniformtext_mode="hide",
        )

        fig_stack.update_traces(
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(size=11, color="white"),
            marker_line=dict(color="rgba(255,255,255,0.85)", width=1),
            hovertemplate=("<b>%{y}</b><br>Proporsi: %{x:.1f}%<br><extra></extra>"),
        )

        fig_stack.update_xaxes(
            range=[0, 100],
            ticksuffix="%",
            showgrid=True,
            gridcolor="rgba(226, 232, 240, 0.85)",
            zeroline=False,
            title_standoff=12,
            automargin=True,
        )

        fig_stack.update_yaxes(
            automargin=True,
            tickfont=dict(size=11),
        )

        st.plotly_chart(
            fig_stack,
            use_container_width=True,
            config={"displayModeBar": False},
        )

    st.dataframe(
        summary.rename(
            columns={
                "stasiun": "Stasiun",
                "rata_ispu": "Rata-rata ISPU",
                "median_ispu": "Median ISPU",
                "p95_ispu": "P95 ISPU",
                "hari_stasiun": "Jumlah hari",
                "pct_tidak_sehat": "% Tidak Sehat+",
                "pencemar_dominan": "Pencemar dominan",
            }
        ).round(1),
        use_container_width=True,
        hide_index=True,
    )

# ============================================================
# Dashboard 4 — Pencemar kritis
# ============================================================
with tabs[3]:
    st.subheader("Analisis Parameter Pencemar Kritis")

    critical_dist = filtered["critical"].value_counts().reset_index()
    critical_dist.columns = ["critical", "jumlah"]
    critical_dist["persen"] = (
        critical_dist["jumlah"] / critical_dist["jumlah"].sum() * 100
    )
    dominant = critical_dist.iloc[0]

    unhealthy = filtered[filtered["is_unhealthy_or_worse"]]
    dominant_unhealthy = (
        mode_or_dash(unhealthy["critical"]) if not unhealthy.empty else "-"
    )
    dominant_unhealthy_share = 0
    if not unhealthy.empty and dominant_unhealthy != "-":
        dominant_unhealthy_share = (
            unhealthy["critical"].eq(dominant_unhealthy).mean() * 100
        )

    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card(
            "Pencemar paling dominan",
            str(dominant["critical"]),
            f"Muncul pada {dominant['persen']:.1f}% observasi terfilter.",
        )
    with c2:
        kpi_card(
            "Dominan saat Tidak Sehat+",
            dominant_unhealthy,
            f"Mengisi {dominant_unhealthy_share:.1f}% kasus ISPU > 100.",
        )
    with c3:
        pm25_cov = filtered["pm25_tersedia"].mean() * 100
        kpi_card(
            "Cakupan PM2.5",
            safe_pct(pm25_cov, 1),
            "Semakin tinggi, semakin kuat pembacaan risiko partikulat halus.",
        )

    if dominant_unhealthy == "PM2.5":
        insight_box(
            f"<b>Rekomendasi kebijakan:</b> Pada kondisi Tidak Sehat+, pencemar yang paling sering memicu risiko adalah <b>PM2.5</b>. Ini mengarah pada pengendalian sumber partikulat halus: pembakaran, kendaraan, debu jalan/konstruksi, dan sumber emisi industri.",
            kind="warning",
        )
    elif dominant_unhealthy == "O3":
        insight_box(
            f"<b>Rekomendasi kebijakan:</b> Pada kondisi Tidak Sehat+, pencemar dominan adalah <b>O3</b>. Fokus kebijakan perlu diarahkan pada pengurangan prekursor O3 seperti NOx dan VOC dari kendaraan, industri, dan aktivitas pembakaran bahan bakar.",
            kind="warning",
        )
    else:
        insight_box(
            f"<b>Pembacaan:</b> Pencemar dominan pada filter aktif adalah <b>{dominant['critical']}</b>. Gunakan filter periode/stasiun untuk melihat apakah pola ini konsisten atau hanya terjadi di wilayah tertentu.",
        )

    c1, c2 = st.columns([1, 1])
    with c1:
        critical_plot = critical_dist.copy().sort_values("jumlah", ascending=True)

        critical_plot["label_persen"] = critical_plot["persen"].apply(
            lambda x: f"{x:.1f}%" if pd.notna(x) else ""
        )

        max_jumlah = critical_plot["jumlah"].max()
        max_jumlah = 0 if pd.isna(max_jumlah) else max_jumlah

        st.html(
            """
            <div style="
                background: linear-gradient(135deg, #F8FAFC 0%, #EEF6FF 100%);
                border: 1px solid #E2E8F0;
                border-radius: 16px;
                padding: 14px 16px;
                margin-bottom: 10px;
                box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
            ">
                <div style="font-size: 0.78rem; color: #64748B; font-weight: 700; letter-spacing: .04em; text-transform: uppercase;">
                    Distribusi pencemar dominan
                </div>
                <div style="font-size: 1.15rem; color: #0F172A; font-weight: 850; margin-top: 2px;">
                    Parameter yang paling sering menjadi pencemar kritis
                </div>
                <div style="font-size: 0.86rem; color: #475569; margin-top: 4px;">
                    Bar yang lebih panjang menunjukkan parameter yang lebih sering menjadi penentu nilai ISPU harian.
                </div>
            </div>
            """,
        )

        fig_crit = go.Figure()

        fig_crit.add_trace(
            go.Bar(
                x=critical_plot["jumlah"],
                y=critical_plot["critical"],
                orientation="h",
                text=critical_plot["label_persen"],
                textposition="outside",
                marker=dict(
                    color=[
                        POLLUTANT_COLORS.get(p, "#64748B")
                        for p in critical_plot["critical"]
                    ],
                    line=dict(color="rgba(255,255,255,0.95)", width=1),
                ),
                customdata=np.stack(
                    [
                        critical_plot["persen"],
                        critical_plot["jumlah"],
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Jumlah observasi: %{customdata[1]:,.0f}<br>"
                    "Proporsi: %{customdata[0]:.1f}%<br>"
                    "<extra></extra>"
                ),
            )
        )

        fig_crit.update_layout(
            height=440,
            template="plotly_white",
            margin=dict(l=10, r=62, t=10, b=55),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0)",
            font=dict(color="#334155", size=12),
            showlegend=False,
            bargap=0.34,
            uniformtext_minsize=10,
            uniformtext_mode="hide",
        )

        fig_crit.update_xaxes(
            range=[0, max(10, max_jumlah * 1.18)],
            title_text="Jumlah observasi",
            showgrid=True,
            gridcolor="rgba(226, 232, 240, 0.85)",
            zeroline=False,
            title_standoff=12,
            automargin=True,
        )

        fig_crit.update_yaxes(
            title_text="",
            automargin=True,
            tickfont=dict(size=11),
        )

        st.plotly_chart(
            fig_crit,
            use_container_width=True,
            config={"displayModeBar": False},
        )
    with c2:
        crit_station = (
            filtered.groupby(["stasiun", "critical"]).size().reset_index(name="jumlah")
        )

        crit_station["persen"] = (
            crit_station["jumlah"]
            / crit_station.groupby("stasiun")["jumlah"].transform("sum")
            * 100
        )

        # Urutkan stasiun berdasarkan pencemar dominan terbesar agar lebih mudah dibaca
        station_order = (
            crit_station.sort_values("persen", ascending=False)
            .drop_duplicates("stasiun")["stasiun"]
            .tolist()
        )

        crit_station["stasiun"] = pd.Categorical(
            crit_station["stasiun"],
            categories=station_order[::-1],
            ordered=True,
        )

        pollutant_order = [
            p for p in POLLUTANT_COLORS.keys() if p in crit_station["critical"].unique()
        ]

        crit_station["critical"] = pd.Categorical(
            crit_station["critical"],
            categories=pollutant_order,
            ordered=True,
        )

        crit_station = crit_station.sort_values(["stasiun", "critical"])

        # Label hanya muncul untuk segmen besar agar tidak bertumpuk
        crit_station["label_persen"] = crit_station["persen"].apply(
            lambda x: f"{x:.1f}%" if pd.notna(x) and x >= 8 else ""
        )

        st.html(
            """
            <div style="
                background: linear-gradient(135deg, #F8FAFC 0%, #EEF6FF 100%);
                border: 1px solid #E2E8F0;
                border-radius: 16px;
                padding: 14px 16px;
                margin-bottom: 10px;
                box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
            ">
                <div style="font-size: 0.78rem; color: #64748B; font-weight: 700; letter-spacing: .04em; text-transform: uppercase;">
                    Pencemar dominan antar stasiun
                </div>
                <div style="font-size: 1.15rem; color: #0F172A; font-weight: 850; margin-top: 2px;">
                    Karakteristik parameter kritis per lokasi SPKU
                </div>
                <div style="font-size: 0.86rem; color: #475569; margin-top: 4px;">
                    Setiap bar menunjukkan komposisi pencemar dominan pada masing-masing stasiun.
                </div>
            </div>
            """,
        )

        fig_cs = px.bar(
            crit_station,
            x="persen",
            y="stasiun",
            color="critical",
            orientation="h",
            text="label_persen",
            color_discrete_map=POLLUTANT_COLORS,
            category_orders={
                "stasiun": station_order[::-1],
                "critical": pollutant_order,
            },
            labels={
                "stasiun": "",
                "persen": "Proporsi observasi (%)",
                "critical": "Pencemar",
            },
            hover_data={
                "stasiun": True,
                "critical": True,
                "persen": ":.1f",
                "jumlah": ":,",
                "label_persen": False,
            },
        )

        fig_cs.update_layout(
            barmode="stack",
            height=460,
            template="plotly_white",
            margin=dict(l=10, r=18, t=10, b=90),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0)",
            font=dict(color="#334155", size=12),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.18,
                xanchor="center",
                x=0.5,
                title_text="",
                font=dict(size=11),
                bgcolor="rgba(255,255,255,0)",
            ),
            hovermode="y unified",
            uniformtext_minsize=10,
            uniformtext_mode="hide",
        )

        fig_cs.update_traces(
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(size=11, color="white"),
            marker_line=dict(color="rgba(255,255,255,0.85)", width=1),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Pencemar: %{legendgroup}<br>"
                "Proporsi: %{x:.1f}%<br>"
                "<extra></extra>"
            ),
        )

        fig_cs.update_xaxes(
            range=[0, 100],
            ticksuffix="%",
            showgrid=True,
            gridcolor="rgba(226, 232, 240, 0.85)",
            zeroline=False,
            title_standoff=12,
            automargin=True,
        )

        fig_cs.update_yaxes(
            automargin=True,
            tickfont=dict(size=11),
        )

        st.plotly_chart(
            fig_cs,
            use_container_width=True,
            config={"displayModeBar": False},
        )

    st.markdown("#### Parameter yang melewati ambang indeks 100")
    exceed_rows = []
    for p in POLLUTANTS:
        if p in filtered.columns:
            valid = filtered[p].notna()
            if valid.sum() > 0:
                exceed_rows.append(
                    {
                        "Parameter": POLLUTANT_LABELS[p],
                        "Data tersedia": int(valid.sum()),
                        "Rata-rata indeks": filtered.loc[valid, p].mean(),
                        "% indeks >100": (filtered.loc[valid, p] > 100).mean() * 100,
                        "Nilai maksimum": filtered.loc[valid, p].max(),
                    }
                )
    exceed_table = pd.DataFrame(exceed_rows).sort_values(
        "% indeks >100", ascending=False
    )
    st.dataframe(exceed_table.round(1), use_container_width=True, hide_index=True)

    temp = filtered.copy()
    temp["periode_tahun"] = temp["tahun"].astype(str)
    crit_year = (
        temp.groupby(["periode_tahun", "critical"]).size().reset_index(name="jumlah")
    )

    crit_year["persen"] = (
        crit_year["jumlah"]
        / crit_year.groupby("periode_tahun")["jumlah"].transform("sum")
        * 100
    )

    crit_year["tahun_num"] = pd.to_numeric(crit_year["periode_tahun"], errors="coerce")
    crit_year = crit_year.dropna(subset=["tahun_num"]).copy()
    crit_year["tahun_num"] = crit_year["tahun_num"].astype(int)

    pollutant_order = [
        p
        for p in POLLUTANT_COLORS.keys()
        if p in crit_year["critical"].dropna().unique()
    ]

    crit_year["critical"] = pd.Categorical(
        crit_year["critical"],
        categories=pollutant_order,
        ordered=True,
    )

    crit_year = crit_year.sort_values(["tahun_num", "critical"])

    year_values = sorted(crit_year["tahun_num"].unique().tolist())
    tick_step = 1 if len(year_values) <= 10 else 2
    tick_values = year_values[::tick_step]

    st.html(
        """
        <div style="
            background: linear-gradient(135deg, #F8FAFC 0%, #EEF6FF 100%);
            border: 1px solid #E2E8F0;
            border-radius: 16px;
            padding: 14px 16px;
            margin-bottom: 10px;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
        ">
            <div style="font-size: 0.78rem; color: #64748B; font-weight: 700; letter-spacing: .04em; text-transform: uppercase;">
                Tren pencemar dominan
            </div>
            <div style="font-size: 1.15rem; color: #0F172A; font-weight: 850; margin-top: 2px;">
                Perubahan komposisi pencemar dominan antar tahun
            </div>
            <div style="font-size: 0.86rem; color: #475569; margin-top: 4px;">
                Area menunjukkan proporsi tiap pencemar sebagai parameter kritis. Grafik ini membantu melihat pergeseran dominasi polutan dari waktu ke waktu.
            </div>
        </div>
        """,
    )

    fig_cy = px.area(
        crit_year,
        x="tahun_num",
        y="persen",
        color="critical",
        color_discrete_map=POLLUTANT_COLORS,
        category_orders={"critical": pollutant_order},
        labels={
            "tahun_num": "Tahun",
            "persen": "Proporsi (%)",
            "critical": "Pencemar",
        },
        hover_data={
            "periode_tahun": False,
            "tahun_num": True,
            "critical": True,
            "persen": ":.1f",
            "jumlah": ":,",
        },
    )

    fig_cy.update_traces(
        mode="lines",
        line=dict(width=0.8),
        hovertemplate=(
            "<b>Tahun %{x}</b><br>"
            "Pencemar: %{fullData.name}<br>"
            "Proporsi: %{y:.1f}%<br>"
            "<extra></extra>"
        ),
    )

    fig_cy.update_layout(
        height=455,
        template="plotly_white",
        margin=dict(l=14, r=18, t=10, b=92),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        font=dict(color="#334155", size=12),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="center",
            x=0.5,
            title_text="",
            font=dict(size=11),
            bgcolor="rgba(255,255,255,0)",
        ),
    )

    fig_cy.update_xaxes(
        tickmode="array",
        tickvals=tick_values,
        ticktext=[str(y) for y in tick_values],
        showgrid=False,
        title_text="",
        automargin=True,
    )

    fig_cy.update_yaxes(
        range=[0, 100],
        ticksuffix="%",
        showgrid=True,
        gridcolor="rgba(226, 232, 240, 0.85)",
        zeroline=False,
        title_text="Proporsi pencemar dominan",
        title_standoff=12,
        automargin=True,
    )

    st.plotly_chart(
        fig_cy,
        use_container_width=True,
        config={"displayModeBar": False},
    )

# ============================================================
# Dashboard 5 — Pola musiman
# ============================================================
with tabs[4]:
    st.subheader("Pola Musiman Kualitas Udara")

    monthly = (
        filtered.groupby("bulan", as_index=False)
        .agg(
            rata_ispu=("max_ispu", "mean"),
            pct_tidak_sehat=("is_unhealthy_or_worse", lambda x: x.mean() * 100),
            observasi=("tanggal", "count"),
        )
        .sort_values("bulan")
    )
    monthly["bulan_label"] = monthly["bulan"].map(MONTH_ID)
    worst_month = monthly.loc[monthly["rata_ispu"].idxmax()]
    best_month = monthly.loc[monthly["rata_ispu"].idxmin()]

    season = (
        filtered.groupby("musim_proxy", as_index=False)
        .agg(
            rata_ispu=("max_ispu", "mean"),
            pct_tidak_sehat=("is_unhealthy_or_worse", lambda x: x.mean() * 100),
            observasi=("tanggal", "count"),
        )
        .sort_values("rata_ispu", ascending=False)
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card(
            "Bulan terburuk",
            MONTH_FULL_ID[int(worst_month["bulan"])],
            f"Rata-rata ISPU {worst_month['rata_ispu']:.1f}; Tidak Sehat+ {worst_month['pct_tidak_sehat']:.1f}%.",
        )
    with c2:
        kpi_card(
            "Bulan terbaik",
            MONTH_FULL_ID[int(best_month["bulan"])],
            f"Rata-rata ISPU {best_month['rata_ispu']:.1f}; Tidak Sehat+ {best_month['pct_tidak_sehat']:.1f}%.",
        )
    with c3:
        if len(season) >= 2:
            diff_season = season.iloc[0]["rata_ispu"] - season.iloc[-1]["rata_ispu"]
            kpi_card(
                "Gap musim",
                f"{diff_season:.1f} poin",
                f"{season.iloc[0]['musim_proxy']} lebih tinggi pada filter aktif.",
            )
        else:
            kpi_card("Gap musim", "-", "Pilih rentang data yang mencakup dua musim.")

    if int(worst_month["bulan"]) in [6, 7, 8, 9, 10]:
        insight_box(
            f"<b>Rekomendasi kebijakan:</b> Bulan <b>{MONTH_FULL_ID[int(worst_month['bulan'])]}</b> menjadi periode terburuk pada filter aktif. Siapkan paket intervensi sebelum musim kering: inspeksi emisi, pengendalian debu konstruksi/jalan, larangan pembakaran terbuka, dan komunikasi risiko kesehatan.",
            kind="warning",
        )
    else:
        insight_box(
            f"<b>Pembacaan musiman:</b> Bulan terburuk pada filter aktif adalah <b>{MONTH_FULL_ID[int(worst_month['bulan'])]}</b>. Cek kembali stasiun dan pencemar dominan untuk memastikan sumber risiko lokal.",
        )

    c1, c2 = st.columns([1.1, 1])
    with c1:
        fig_month = make_subplots(specs=[[{"secondary_y": True}]])

        monthly_plot = monthly.copy()
        monthly_plot["bulan"] = pd.Categorical(
            monthly_plot["bulan_label"],
            categories=[MONTH_ID[i] for i in range(1, 13)],
            ordered=True,
        )
        monthly_plot = monthly_plot.sort_values("bulan")

        max_pct = monthly_plot["pct_tidak_sehat"].max()
        max_pct = 0 if pd.isna(max_pct) else max_pct

        st.html(
            """
            <div style="
                background: linear-gradient(135deg, #F8FAFC 0%, #EEF6FF 100%);
                border: 1px solid #E2E8F0;
                border-radius: 16px;
                padding: 14px 16px;
                margin-bottom: 10px;
                box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
            ">
                <div style="font-size: 0.78rem; color: #64748B; font-weight: 700; letter-spacing: .04em; text-transform: uppercase;">
                    Pola musiman
                </div>
                <div style="font-size: 1.15rem; color: #0F172A; font-weight: 850; margin-top: 2px;">
                    Rata-rata ISPU dan risiko Tidak Sehat+ per bulan
                </div>
                <div style="font-size: 0.86rem; color: #475569; margin-top: 4px;">
                    Bar menunjukkan rata-rata ISPU bulanan. Garis menunjukkan persentase observasi Tidak Sehat atau lebih buruk.
                </div>
            </div>
            """,
        )

        fig_month.add_trace(
            go.Bar(
                x=monthly_plot["bulan_label"],
                y=monthly_plot["rata_ispu"],
                name="Rata-rata ISPU",
                marker=dict(
                    color=monthly_plot["rata_ispu"],
                    colorscale=[
                        [0.00, "#22C55E"],
                        [0.45, "#FACC15"],
                        [0.70, "#FB923C"],
                        [1.00, "#DC2626"],
                    ],
                    line=dict(color="rgba(15, 23, 42, 0.14)", width=1),
                ),
                hovertemplate=(
                    "<b>%{x}</b><br>Rata-rata ISPU: %{y:.1f}<br><extra></extra>"
                ),
            ),
            secondary_y=False,
        )

        fig_month.add_trace(
            go.Scatter(
                x=monthly_plot["bulan_label"],
                y=monthly_plot["pct_tidak_sehat"],
                name="% Tidak Sehat+",
                mode="lines+markers",
                line=dict(width=3, color="#0F766E", shape="spline"),
                marker=dict(
                    size=8,
                    color="#0F766E",
                    line=dict(color="white", width=2),
                ),
                hovertemplate=(
                    "<b>%{x}</b><br>Tidak Sehat+: %{y:.1f}%<br><extra></extra>"
                ),
            ),
            secondary_y=True,
        )

        fig_month.add_hline(
            y=100,
            line_dash="dash",
            line_width=1.4,
            line_color="#DC2626",
            opacity=0.72,
            secondary_y=False,
        )

        fig_month.add_annotation(
            x=1,
            y=100,
            xref="paper",
            yref="y",
            text="Ambang Tidak Sehat = 100",
            showarrow=False,
            xanchor="right",
            yanchor="bottom",
            font=dict(size=11, color="#991B1B"),
            bgcolor="rgba(255,255,255,0.88)",
            bordercolor="rgba(220,38,38,0.25)",
            borderwidth=1,
            borderpad=4,
        )

        # Highlight periode kemarau/lebih kering: Mei–Okt
        fig_month.add_vrect(
            x0="Mei",
            x1="Okt",
            fillcolor="rgba(251, 146, 60, 0.10)",
            line_width=0,
            layer="below",
        )

        fig_month.add_annotation(
            x="Agu",
            y=1.06,
            xref="x",
            yref="paper",
            text="Periode lebih kering",
            showarrow=False,
            font=dict(size=11, color="#9A3412"),
            bgcolor="rgba(255,247,237,0.9)",
            bordercolor="rgba(251,146,60,0.25)",
            borderwidth=1,
            borderpad=4,
        )

        fig_month.update_yaxes(
            title_text="Rata-rata ISPU",
            secondary_y=False,
            rangemode="tozero",
            showgrid=True,
            gridcolor="rgba(226, 232, 240, 0.85)",
            title_standoff=12,
            automargin=True,
        )

        fig_month.update_yaxes(
            title_text="Tidak Sehat+ (%)",
            secondary_y=True,
            range=[0, max(100, max_pct * 1.25)],
            ticksuffix="%",
            showgrid=False,
            title_standoff=12,
            automargin=True,
        )

        fig_month.update_xaxes(
            title_text="",
            tickangle=0,
            tickfont=dict(size=11),
            showgrid=False,
            automargin=True,
        )

        fig_month.update_layout(
            template="plotly_white",
            height=470,
            margin=dict(l=18, r=28, t=24, b=92),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0)",
            font=dict(color="#334155", size=12),
            hovermode="x unified",
            bargap=0.34,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.18,
                xanchor="center",
                x=0.5,
                title_text="",
                font=dict(size=12),
                bgcolor="rgba(255,255,255,0)",
            ),
        )

        st.plotly_chart(
            fig_month,
            use_container_width=True,
            config={"displayModeBar": False},
        )
    with c2:
        heat = filtered.groupby(["tahun", "bulan"], as_index=False).agg(
            rata_ispu=("max_ispu", "mean"),
            pct_tidak_sehat=("is_unhealthy_or_worse", lambda x: x.mean() * 100),
        )

        heat["bulan_label"] = heat["bulan"].map(MONTH_ID)

        pivot_ispu = heat.pivot(
            index="bulan_label", columns="tahun", values="rata_ispu"
        ).reindex([MONTH_ID[i] for i in range(1, 13)])

        pivot_risk = heat.pivot(
            index="bulan_label", columns="tahun", values="pct_tidak_sehat"
        ).reindex([MONTH_ID[i] for i in range(1, 13)])

        year_cols = [str(c) for c in pivot_ispu.columns]
        year_values = list(pivot_ispu.columns)

        tick_step = 1 if len(year_values) <= 10 else 2
        tick_vals = year_values[::tick_step]
        tick_text = [str(y) for y in tick_vals]

        st.html(
            """
            <div style="
                background: linear-gradient(135deg, #F8FAFC 0%, #EEF6FF 100%);
                border: 1px solid #E2E8F0;
                border-radius: 16px;
                padding: 14px 16px;
                margin-bottom: 10px;
                box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
            ">
                <div style="font-size: 0.78rem; color: #64748B; font-weight: 700; letter-spacing: .04em; text-transform: uppercase;">
                    Heatmap musiman
                </div>
                <div style="font-size: 1.15rem; color: #0F172A; font-weight: 850; margin-top: 2px;">
                    Rata-rata ISPU per bulan dan tahun
                </div>
                <div style="font-size: 0.86rem; color: #475569; margin-top: 4px;">
                    Warna lebih gelap menunjukkan kualitas udara yang lebih buruk. Gunakan pola ini untuk melihat bulan dan tahun yang perlu diantisipasi.
                </div>
            </div>
            """,
        )

        customdata = np.dstack(
            [
                pivot_risk.values,
            ]
        )

        fig_heat = go.Figure(
            data=go.Heatmap(
                z=pivot_ispu.values,
                x=year_values,
                y=list(pivot_ispu.index),
                customdata=customdata,
                zmin=0,
                zmax=max(
                    120,
                    np.nanmax(pivot_ispu.values)
                    if not np.isnan(pivot_ispu.values).all()
                    else 120,
                ),
                colorscale=[
                    [0.00, "#DCFCE7"],
                    [0.35, "#22C55E"],
                    [0.50, "#FEF08A"],
                    [0.65, "#FB923C"],
                    [0.82, "#DC2626"],
                    [1.00, "#581C87"],
                ],
                colorbar=dict(
                    title=dict(text="ISPU", side="right"),
                    thickness=14,
                    len=0.82,
                    y=0.5,
                    tickfont=dict(size=10),
                ),
                hovertemplate=(
                    "<b>%{y} %{x}</b><br>"
                    "Rata-rata ISPU: %{z:.1f}<br>"
                    "Tidak Sehat+: %{customdata[0]:.1f}%<br>"
                    "<extra></extra>"
                ),
            )
        )

        fig_heat.update_layout(
            height=470,
            template="plotly_white",
            margin=dict(l=12, r=42, t=10, b=58),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0)",
            font=dict(color="#334155", size=12),
        )

        fig_heat.update_xaxes(
            title_text="",
            tickmode="array",
            tickvals=tick_vals,
            ticktext=tick_text,
            showgrid=False,
            side="bottom",
            automargin=True,
            tickfont=dict(size=11),
        )

        fig_heat.update_yaxes(
            title_text="",
            autorange="reversed",
            showgrid=False,
            automargin=True,
            tickfont=dict(size=11),
        )

        fig_heat.add_shape(
            type="rect",
            xref="paper",
            yref="paper",
            x0=0,
            x1=1,
            y0=0,
            y1=1,
            line=dict(color="rgba(15, 23, 42, 0.08)", width=1),
        )

        st.plotly_chart(
            fig_heat,
            use_container_width=True,
            config={"displayModeBar": False},
        )

    st.markdown("#### Hubungan cuaca dan kualitas udara")
    weather_cols = [
        c
        for c in ["curah_hujan_mm", "suhu_c", "kelembapan_pct", "kecepatan_angin"]
        if c in filtered.columns and filtered[c].notna().any()
    ]
    if weather_cols:
        daily_weather = filtered.groupby("tanggal", as_index=False).agg(
            rata_ispu=("max_ispu", "mean"), **{c: (c, "mean") for c in weather_cols}
        )
        selected_weather_col = st.selectbox(
            "Pilih variabel cuaca untuk dibandingkan dengan ISPU",
            weather_cols,
            format_func=lambda x: {
                "curah_hujan_mm": "Curah hujan",
                "suhu_c": "Suhu",
                "kelembapan_pct": "Kelembapan",
                "kecepatan_angin": "Kecepatan angin",
            }.get(x, x),
        )
        fig_weather = px.scatter(
            daily_weather,
            x=selected_weather_col,
            y="rata_ispu",
            title="Relasi harian antara variabel cuaca dan rata-rata ISPU",
            labels={
                selected_weather_col: selected_weather_col.replace("_", " ").title(),
                "rata_ispu": "Rata-rata ISPU",
            },
        )
        fig_weather = add_unhealthy_hline(fig_weather)
        fig_weather = clean_plot_layout(fig_weather, height=430)
        st.plotly_chart(fig_weather, use_container_width=True)
    else:
        c1, c2 = st.columns([1, 1])
        with c1:
            season_display = season.rename(
                columns={
                    "musim_proxy": "Musim/proxy cuaca",
                    "rata_ispu": "Rata-rata ISPU",
                    "pct_tidak_sehat": "% Tidak Sehat+",
                    "observasi": "Observasi",
                }
            ).round(1)
            st.dataframe(season_display, use_container_width=True, hide_index=True)
        with c2:
            st.info(
                "Belum ada data cuaca harian eksternal. Untuk analisis lebih kuat, unggah CSV cuaca dari BMKG/open data dengan kolom tanggal dan salah satu variabel: curah hujan, suhu, kelembapan, atau kecepatan angin. Sementara ini dashboard memakai proxy musim."
            )

# -----------------------------
# Footer: data siap unduh
# -----------------------------
st.divider()
footer_col1, footer_col2 = st.columns([1.4, 0.8])
with footer_col1:
    st.caption(
        "Catatan: dashboard menghitung ulang `max_ispu` dan `critical` dari nilai parameter polutan, lalu mengonsolidasikan duplikasi pada level tanggal × stasiun dengan pendekatan konservatif: nilai parameter maksimum dipertahankan."
    )
with footer_col2:
    export_cols = [
        "tanggal",
        "stasiun",
        "pm10",
        "pm25",
        "so2",
        "co",
        "o3",
        "no2",
        "max_ispu",
        "critical",
        "kategori",
        "musim_proxy",
        "record_source_count",
        "jumlah_flag_kualitas_dashboard",
    ]
    export_cols = [c for c in export_cols if c in filtered.columns]
    csv_export = filtered[export_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Unduh data terfilter",
        data=csv_export,
        file_name="ispu_jakarta_dashboard_filtered.csv",
        mime="text/csv",
        use_container_width=True,
    )
