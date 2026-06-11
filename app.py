import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# =============================================================
# DASHBOARD BI KUALITAS UDARA ISPU DKI JAKARTA
# Data input: ispu_jakarta_analysis.csv
# =============================================================

st.set_page_config(
    page_title="Dashboard BI ISPU DKI Jakarta",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = Path("./data/ispu_jakarta_clean.csv")


# Dashboard dapat dijalankan dari folder project maupun dari folder yang sama
# dengan file app.py. Urutan pertama tetap mengikuti struktur project pengguna.
ALTERNATIVE_DATA_PATHS = [
    DATA_PATH,
    Path(__file__).resolve().parent / "ispu_jakarta_analysis.csv",
    Path("ispu_jakarta_analysis.csv"),
]


def resolve_data_path() -> Path:
    for candidate in ALTERNATIVE_DATA_PATHS:
        if candidate.exists():
            return candidate
    # Kembalikan path utama agar pesan error tetap informatif.
    return DATA_PATH


# -----------------------------
# Styling sederhana dan profesional
# -----------------------------
st.markdown(
    """
    <style>
    .main .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
    .metric-card {
        background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 18px 20px;
        min-height: 118px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
    }
    .metric-label {font-size: 0.86rem; color: #64748b; margin-bottom: 0.35rem;}
    .metric-value {font-size: 1.7rem; font-weight: 800; color: #0f172a; line-height: 1.1;}
    .metric-help {font-size: 0.78rem; color: #64748b; margin-top: 0.5rem;}
    .insight-box {
        background: #f8fafc;
        border-left: 5px solid #2563eb;
        border-radius: 10px;
        padding: 14px 16px;
        margin-top: 10px;
        margin-bottom: 18px;
    }
    .action-box {
        background: #ecfdf5;
        border-left: 5px solid #16a34a;
        border-radius: 10px;
        padding: 14px 16px;
        margin-top: 10px;
        margin-bottom: 18px;
    }
    .warning-box {
        background: #fffbeb;
        border-left: 5px solid #f59e0b;
        border-radius: 10px;
        padding: 14px 16px;
        margin-top: 10px;
        margin-bottom: 18px;
    }
    h1, h2, h3 {letter-spacing: -0.02em;}
    </style>
    """,
    unsafe_allow_html=True,
)

STANDARD_CATEGORY_ORDER = [
    "BAIK",
    "SEDANG",
    "TIDAK SEHAT",
    "SANGAT TIDAK SEHAT",
    "BERBAHAYA",
    "TIDAK DIKETAHUI",
]
CATEGORY_ORDER = STANDARD_CATEGORY_ORDER.copy()
PARAMETER_ORDER = ["PM10", "PM2.5", "SO2", "CO", "O3", "NO2"]
PARAMETER_COLS = ["pm10", "pm25", "so2", "co", "o3", "no2"]
CATEGORY_COLOR = {
    "BAIK": "#22c55e",
    "SEDANG": "#eab308",
    "TIDAK SEHAT": "#f97316",
    "SANGAT TIDAK SEHAT": "#ef4444",
    "BERBAHAYA": "#7e22ce",
    "TIDAK DIKETAHUI": "#94a3b8",
}
PARAMETER_COLOR = {
    "PM10": "#64748b",
    "PM2.5": "#ef4444",
    "SO2": "#eab308",
    "CO": "#22c55e",
    "O3": "#3b82f6",
    "NO2": "#a855f7",
}


def normalize_text(value):
    if pd.isna(value):
        return np.nan
    return str(value).strip().upper().replace("_", " ")


def classify_ispu(value):
    """Klasifikasi kategori ISPU berdasarkan rentang nilai umum ISPU Indonesia."""
    if pd.isna(value):
        return "TIDAK DIKETAHUI"
    try:
        x = float(value)
    except Exception:
        return "TIDAK DIKETAHUI"
    if x <= 50:
        return "BAIK"
    if x <= 100:
        return "SEDANG"
    if x <= 200:
        return "TIDAK SEHAT"
    if x <= 300:
        return "SANGAT TIDAK SEHAT"
    return "BERBAHAYA"


def choose_column(df, candidates, required=True):
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise KeyError(
            f"Kolom wajib tidak ditemukan. Kandidat yang dicari: {', '.join(candidates)}. "
            f"Kolom tersedia: {', '.join(df.columns)}"
        )
    return None


@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    date_col = choose_column(
        df, ["tanggal_validated", "tanggal", "date", "tanggal_data"]
    )
    station_col = choose_column(df, ["stasiun", "station", "nama_stasiun"])
    ispu_col = choose_column(
        df, ["max_recomputed", "max", "max_original", "nilai_ispu", "ispu"]
    )
    critical_col = choose_column(
        df,
        [
            "critical_recomputed",
            "critical_standardized",
            "critical",
            "parameter_dominan",
            "pencemar_dominan",
        ],
        required=False,
    )
    category_col = choose_column(
        df, ["categori", "kategori", "category", "kategori_ispu"], required=False
    )
    quality_flag_col = choose_column(
        df,
        ["jumlah_flag_kualitas", "quality_flag_count", "jumlah_flag", "flag_count"],
        required=False,
    )
    validation_status_col = choose_column(
        df,
        ["status_validasi_ringkas", "status_validasi", "validation_status"],
        required=False,
    )
    duplicate_flag_col = choose_column(
        df,
        ["flag_duplicate_key_tanggal_stasiun", "flag_duplicate", "is_duplicate"],
        required=False,
    )
    row_id_col = choose_column(
        df, ["row_id_original", "row_id", "index_original"], required=False
    )

    # Standarisasi nama kolom kerja agar dashboard stabil walau nama kolom sumber berbeda.
    df["tanggal_dashboard"] = pd.to_datetime(df[date_col], errors="coerce")
    df["stasiun_dashboard"] = df[station_col].astype(str).str.strip()
    df["ispu_dashboard"] = pd.to_numeric(df[ispu_col], errors="coerce")
    if row_id_col:
        df["row_id_dashboard"] = pd.to_numeric(df[row_id_col], errors="coerce")
    else:
        df["row_id_dashboard"] = np.arange(len(df))

    if critical_col:
        df["critical_dashboard"] = df[critical_col].map(normalize_text)
    else:
        # Jika kolom pencemar dominan tidak ada, hitung dari nilai parameter terbesar.
        available_params = [c for c in PARAMETER_COLS if c in df.columns]
        if available_params:
            param_map = {
                "pm10": "PM10",
                "pm25": "PM2.5",
                "so2": "SO2",
                "co": "CO",
                "o3": "O3",
                "no2": "NO2",
            }
            param_numeric = df[available_params].apply(pd.to_numeric, errors="coerce")
            df["critical_dashboard"] = param_numeric.idxmax(axis=1).map(param_map)
        else:
            df["critical_dashboard"] = "TIDAK DIKETAHUI"

    critical_fix = {
        "PM25": "PM2.5",
        "PM2 5": "PM2.5",
        "PM2.5": "PM2.5",
        "PM10": "PM10",
        "SO2": "SO2",
        "CO": "CO",
        "O3": "O3",
        "NO2": "NO2",
    }
    df["critical_dashboard"] = df["critical_dashboard"].map(
        lambda x: critical_fix.get(x, x)
    )
    df["critical_dashboard"] = df["critical_dashboard"].fillna("TIDAK DIKETAHUI")

    if category_col:
        # Sesuai arahan: kategori dashboard mengikuti kolom kategori/categori dari dataset.
        # Nilai tidak direklasifikasi berdasarkan rentang ISPU agar visualisasi konsisten
        # dengan hasil cleaning/validasi pada dataset sumber.
        df["kategori_dashboard"] = (
            df[category_col].map(normalize_text).fillna("TIDAK DIKETAHUI")
        )

        # Hanya rapikan variasi penulisan umum tanpa mengubah kategori yang memang ada
        # di dataset, misalnya kategori tambahan seperti LUARBIASA tetap dipertahankan.
        category_fix = {
            "TIDAKSEHAT": "TIDAK SEHAT",
            "SANGAT TIDAKSEHAT": "SANGAT TIDAK SEHAT",
            "SANGAT TIDAK_SEHAT": "SANGAT TIDAK SEHAT",
        }
        df["kategori_dashboard"] = df["kategori_dashboard"].map(
            lambda x: category_fix.get(x, x)
        )
    else:
        # Fallback hanya dipakai jika dataset tidak menyediakan kolom kategori.
        df["kategori_dashboard"] = df["ispu_dashboard"].apply(classify_ispu)

    df = df.dropna(
        subset=["tanggal_dashboard", "stasiun_dashboard", "ispu_dashboard"]
    ).copy()
    df["tahun_dashboard"] = df["tanggal_dashboard"].dt.year
    df["bulan_dashboard"] = df["tanggal_dashboard"].dt.month
    df["nama_bulan_dashboard"] = df["tanggal_dashboard"].dt.month_name()
    df["tahun_bulan_dashboard"] = (
        df["tanggal_dashboard"].dt.to_period("M").dt.to_timestamp()
    )
    df["tahun_str_dashboard"] = df["tahun_dashboard"].astype(str)
    df["is_unhealthy_dashboard"] = df["kategori_dashboard"].isin(
        ["TIDAK SEHAT", "SANGAT TIDAK SEHAT", "BERBAHAYA"]
    )

    # Metadata kualitas data. Kolom ini dipakai untuk menentukan apakah sebuah
    # observasi boleh masuk KPI/rata-rata atau hanya ditampilkan sebagai audit.
    if quality_flag_col:
        df["jumlah_flag_kualitas_dashboard"] = (
            pd.to_numeric(df[quality_flag_col], errors="coerce").fillna(0).astype(int)
        )
    else:
        flag_cols = [c for c in df.columns if c.lower().startswith("flag_")]
        if flag_cols:
            df["jumlah_flag_kualitas_dashboard"] = (
                df[flag_cols].astype(bool).sum(axis=1).astype(int)
            )
        else:
            df["jumlah_flag_kualitas_dashboard"] = 0

    if validation_status_col:
        df["status_validasi_dashboard"] = (
            df[validation_status_col].fillna("Tidak diketahui").astype(str).str.strip()
        )
    else:
        df["status_validasi_dashboard"] = np.where(
            df["jumlah_flag_kualitas_dashboard"].eq(0),
            "Tidak ada flag",
            "Ada flag kualitas",
        )

    key_cols = ["tanggal_dashboard", "stasiun_dashboard"]
    duplicate_from_key = df.duplicated(subset=key_cols, keep=False)
    if duplicate_flag_col:
        duplicate_from_flag = df[duplicate_flag_col].fillna(False).astype(bool)
        df["is_duplicate_key_dashboard"] = duplicate_from_flag | duplicate_from_key
    else:
        df["is_duplicate_key_dashboard"] = duplicate_from_key

    duplicate_count = df.groupby(key_cols)["ispu_dashboard"].transform("size")
    df["duplicate_count_dashboard"] = duplicate_count.fillna(1).astype(int)

    conflict_parts = []
    for conflict_col in ["ispu_dashboard", "kategori_dashboard", "critical_dashboard"]:
        nunique = df.groupby(key_cols)[conflict_col].transform(
            lambda x: x.astype(str).nunique(dropna=False)
        )
        conflict_parts.append(nunique.gt(1))
    df["has_duplicate_conflict_dashboard"] = (
        np.logical_or.reduce(conflict_parts) & df["is_duplicate_key_dashboard"]
    )

    df["is_data_valid_dashboard"] = (
        df["jumlah_flag_kualitas_dashboard"].eq(0)
        & df["status_validasi_dashboard"]
        .str.upper()
        .str.contains("TIDAK ADA FLAG", na=False)
        & ~df["is_duplicate_key_dashboard"]
        & ~df["has_duplicate_conflict_dashboard"]
    )
    df["status_data_dashboard"] = np.where(
        df["is_data_valid_dashboard"],
        "Data valid untuk KPI",
        "Data bermasalah/perlu audit",
    )

    # Pastikan nilai parameter polutan numerik jika kolom tersedia.
    for col in PARAMETER_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Flag khusus untuk PM2.5. Missing PM2.5 tidak diimputasi; analisis PM2.5
    # dibatasi pada baris/periode ketika nilai pm25 tersedia di dataset.
    if "pm25" in df.columns:
        df["pm25_available_dashboard"] = df["pm25"].notna()
    else:
        df["pm25_available_dashboard"] = False

    return df


def metric_card(label, value, help_text=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def insight_box(title, text):
    st.markdown(
        f"""
        <div class="insight-box">
        <b>{title}</b><br>{text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def action_box(title, text):
    st.markdown(
        f"""
        <div class="action-box">
        <b>{title}</b><br>{text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def warning_box(title, text):
    st.markdown(
        f"""
        <div class="warning-box">
        <b>{title}</b><br>{text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_pct(x):
    if pd.isna(x):
        return "0,0%"
    return f"{x:.1f}%".replace(".", ",")


def month_name_id(month_number):
    names = {
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
    return names.get(int(month_number), str(month_number))


def safe_mode(series):
    mode = series.dropna()
    if mode.empty:
        return "Tidak tersedia"
    return str(mode.mode().iloc[0])


def get_category_order(dataframe: pd.DataFrame) -> list[str]:
    """Urutan kategori mengikuti dataset, dengan kategori standar tetap lebih rapi."""
    available = [
        c
        for c in dataframe["kategori_dashboard"].dropna().astype(str).unique().tolist()
        if c and c.upper() != "NAN"
    ]
    ordered_standard = [c for c in STANDARD_CATEGORY_ORDER if c in available]
    extras = sorted([c for c in available if c not in ordered_standard])
    return ordered_standard + extras


def select_best_duplicate_records(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Ambil satu observasi terbaik untuk setiap pasangan tanggal-stasiun."""
    if dataframe.empty:
        return dataframe.copy()
    sort_cols = [
        "tanggal_dashboard",
        "stasiun_dashboard",
        "jumlah_flag_kualitas_dashboard",
        "has_duplicate_conflict_dashboard",
        "row_id_dashboard",
    ]
    work = dataframe.sort_values(
        sort_cols,
        ascending=[True, True, True, True, False],
        na_position="last",
    )
    return work.drop_duplicates(
        subset=["tanggal_dashboard", "stasiun_dashboard"], keep="first"
    ).copy()


def apply_quality_filters(
    dataframe: pd.DataFrame,
    use_valid_only: bool,
    max_quality_flags: int,
    selected_validation_statuses: list[str],
    duplicate_policy: str,
) -> pd.DataFrame:
    """Terapkan filter kualitas sebelum KPI/visualisasi dihitung."""
    out = dataframe.copy()

    if use_valid_only:
        out = out[out["is_data_valid_dashboard"]].copy()
    else:
        out = out[out["jumlah_flag_kualitas_dashboard"].le(max_quality_flags)].copy()
        if selected_validation_statuses:
            out = out[
                out["status_validasi_dashboard"].isin(selected_validation_statuses)
            ].copy()

    if duplicate_policy == "Keluarkan semua baris duplikat tanggal-stasiun":
        out = out[~out["is_duplicate_key_dashboard"]].copy()
    elif duplicate_policy == "Ambil 1 baris terbaik per tanggal-stasiun":
        out = select_best_duplicate_records(out)

    return out


def quality_summary_table(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe.empty:
        return pd.DataFrame(columns=["Status Data", "Jumlah Baris", "Persentase"])
    out = dataframe["status_data_dashboard"].value_counts().reset_index()
    out.columns = ["Status Data", "Jumlah Baris"]
    out["Persentase"] = out["Jumlah Baris"] / out["Jumlah Baris"].sum() * 100
    return out


def prepare_category_distribution(df, category_order):
    out = (
        df.groupby(["stasiun_dashboard", "kategori_dashboard"], as_index=False)
        .size()
        .rename(columns={"size": "jumlah_hari"})
    )
    total = out.groupby("stasiun_dashboard")["jumlah_hari"].transform("sum")
    out["persentase"] = np.where(total > 0, out["jumlah_hari"] / total * 100, 0)
    out["kategori_dashboard"] = pd.Categorical(
        out["kategori_dashboard"], categories=category_order, ordered=True
    )
    return out.sort_values(["stasiun_dashboard", "kategori_dashboard"])


# -----------------------------
# Load data
# -----------------------------
st.title("🌫️ Dashboard Business Intelligence Kualitas Udara ISPU DKI Jakarta")
st.caption(
    "Dashboard interaktif berdasarkan data ISPU DKI Jakarta yang telah melalui proses cleaning dan validasi."
)

try:
    active_data_path = resolve_data_path()
    df = load_data(active_data_path)
except Exception as error:
    st.error(f"Gagal membaca data: {error}")
    st.stop()

if df.empty:
    st.warning("Data kosong setelah proses pembacaan dan standardisasi kolom.")
    st.stop()

category_order = get_category_order(df)
category_color_map = {
    category: CATEGORY_COLOR.get(category, "#94a3b8") for category in category_order
}

# -----------------------------
# Sidebar filters
# -----------------------------
with st.sidebar:
    st.header("Filter Dashboard")
    st.caption("Gunakan filter berikut untuk menyesuaikan seluruh dashboard.")

    min_date = df["tanggal_dashboard"].min().date()
    max_date = df["tanggal_dashboard"].max().date()
    selected_dates = st.date_input(
        "Rentang tanggal",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
        start_date, end_date = selected_dates
    else:
        start_date, end_date = min_date, max_date

    stations = sorted(df["stasiun_dashboard"].dropna().unique().tolist())
    selected_stations = st.multiselect(
        "Stasiun SPKU",
        options=stations,
        default=stations,
    )

    categories_available = category_order
    selected_categories = st.multiselect(
        "Kategori kualitas udara",
        options=categories_available,
        default=categories_available,
    )

    params_available = [
        p for p in PARAMETER_ORDER if p in df["critical_dashboard"].unique()
    ]
    extra_params = sorted(
        [
            p
            for p in df["critical_dashboard"].dropna().unique()
            if p not in params_available
        ]
    )
    params_options = params_available + extra_params
    selected_params = st.multiselect(
        "Parameter pencemar dominan",
        options=params_options,
        default=params_options,
    )

    st.divider()
    st.subheader("Filter Kualitas Data")
    use_valid_data_only = st.checkbox(
        "Hitung KPI/visualisasi hanya dari data valid",
        value=True,
        help=(
            "Jika aktif, KPI dan rata-rata hanya memakai baris tanpa flag kualitas, "
            "bukan duplikat tanggal-stasiun, dan status validasinya 'Tidak ada flag'."
        ),
    )

    max_flag_available = int(df["jumlah_flag_kualitas_dashboard"].max())
    max_flag_available = int(df["jumlah_flag_kualitas_dashboard"].max())

    if max_flag_available > 0:
        max_quality_flags = st.slider(
            "Maksimum jumlah_flag_kualitas yang boleh masuk analisis",
            min_value=0,
            max_value=max_flag_available,
            value=max_flag_available,
            step=1,
            help=(
                "Semakin kecil nilainya, semakin ketat data yang masuk ke KPI dan grafik utama. "
                "Nilai 0 berarti hanya data tanpa flag kualitas yang digunakan."
            ),
        )
    else:
        max_quality_flags = 0
        st.info(
            "Seluruh data pada dataset memiliki jumlah_flag_kualitas = 0. "
            "Filter maksimum flag kualitas tidak ditampilkan karena tidak ada variasi nilai flag."
        )

    validation_status_options = sorted(
        df["status_validasi_dashboard"].dropna().unique().tolist()
    )
    selected_validation_statuses = st.multiselect(
        "Status validasi ringkas",
        options=validation_status_options,
        default=validation_status_options
        if not use_valid_data_only
        else [s for s in validation_status_options if "Tidak ada flag" in s],
        help="Filter ini memakai kolom status_validasi_ringkas dari dataset.",
    )

    duplicate_policy = st.selectbox(
        "Kebijakan duplikasi tanggal-stasiun",
        options=[
            "Ambil 1 baris terbaik per tanggal-stasiun",
            "Keluarkan semua baris duplikat tanggal-stasiun",
            "Gunakan semua baris",
        ],
        index=0,
        help=(
            "Baris terbaik dipilih berdasarkan jumlah flag paling sedikit; jika seri, "
            "dipilih row_id_original terakhir sebagai hasil cleaning terbaru."
        ),
    )

    st.divider()
    st.caption(f"Sumber data: `{active_data_path.name}` hasil cleaning.")

base_filtered = df[
    (df["tanggal_dashboard"].dt.date >= start_date)
    & (df["tanggal_dashboard"].dt.date <= end_date)
    & (df["stasiun_dashboard"].isin(selected_stations))
    & (df["kategori_dashboard"].isin(selected_categories))
    & (df["critical_dashboard"].isin(selected_params))
].copy()

filtered_before_quality = base_filtered.copy()
filtered = apply_quality_filters(
    base_filtered,
    use_valid_only=use_valid_data_only,
    max_quality_flags=max_quality_flags,
    selected_validation_statuses=selected_validation_statuses,
    duplicate_policy=duplicate_policy,
)

if filtered.empty:
    st.warning(
        "Tidak ada data yang sesuai dengan kombinasi filter saat ini. Ubah filter untuk menampilkan visualisasi."
    )
    st.stop()

period_text = f"{filtered['tanggal_dashboard'].min().date()} s.d. {filtered['tanggal_dashboard'].max().date()}"
st.markdown(
    f"**Periode terpilih:** {period_text} &nbsp;&nbsp;|&nbsp;&nbsp; "
    f"**Jumlah observasi analitik:** {len(filtered):,} baris &nbsp;&nbsp;|&nbsp;&nbsp; "
    f"**Stasiun tercakup:** {filtered['stasiun_dashboard'].nunique()} stasiun".replace(
        ",", "."
    )
)

with st.expander("Ringkasan kualitas data pada filter aktif", expanded=True):
    total_base_rows = len(filtered_before_quality)
    valid_base_rows = int(filtered_before_quality["is_data_valid_dashboard"].sum())
    flagged_base_rows = total_base_rows - valid_base_rows
    duplicate_rows = int(filtered_before_quality["is_duplicate_key_dashboard"].sum())
    conflict_rows = int(
        filtered_before_quality["has_duplicate_conflict_dashboard"].sum()
    )

    q1, q2, q3, q4 = st.columns(4)
    with q1:
        st.metric(
            "Baris sebelum filter kualitas", f"{total_base_rows:,}".replace(",", ".")
        )
    with q2:
        st.metric("Baris masuk KPI/visualisasi", f"{len(filtered):,}".replace(",", "."))
    with q3:
        st.metric("Data valid", f"{valid_base_rows:,}".replace(",", "."))
    with q4:
        st.metric(
            "Baris bermasalah/perlu audit", f"{flagged_base_rows:,}".replace(",", ".")
        )

    st.caption(
        f"Duplikasi tanggal-stasiun pada filter awal: {duplicate_rows:,} baris; "
        f"duplikasi dengan konflik nilai/kategori/critical: {conflict_rows:,} baris. "
        f"Kebijakan duplikasi aktif: {duplicate_policy}.".replace(",", ".")
    )

    quality_table = quality_summary_table(filtered_before_quality)
    if not quality_table.empty:
        quality_table["Persentase"] = quality_table["Persentase"].round(1)
        st.dataframe(quality_table, use_container_width=True, hide_index=True)

pm25_available_filtered = filtered[filtered["pm25_available_dashboard"]].copy()
if not pm25_available_filtered.empty:
    pm25_period_text = (
        f"{pm25_available_filtered['tanggal_dashboard'].min().date()} s.d. "
        f"{pm25_available_filtered['tanggal_dashboard'].max().date()}"
    )
else:
    pm25_period_text = "Tidak tersedia pada filter aktif"

# -----------------------------
# Dashboard tabs
# -----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "1. Overview Kualitas Udara",
        "2. Tren Temporal",
        "3. Perbandingan Antar Stasiun",
        "4. Parameter Pencemar Kritis",
        "5. Pola Musiman",
    ]
)

# =============================================================
# Dashboard 1 — Overview Kualitas Udara
# =============================================================
with tab1:
    st.header("Dashboard 1 — Overview Kualitas Udara")

    avg_ispu = filtered["ispu_dashboard"].mean()
    unhealthy_pct = filtered["is_unhealthy_dashboard"].mean() * 100
    dominant_param = safe_mode(filtered["critical_dashboard"])
    dominant_category = safe_mode(filtered["kategori_dashboard"])

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card(
            "Rata-rata ISPU",
            f"{avg_ispu:.1f}",
            "Rata-rata nilai ISPU pada filter aktif",
        )
    with c2:
        metric_card(
            "% Hari Tidak Sehat",
            format_pct(unhealthy_pct),
            "Kategori TIDAK SEHAT atau lebih buruk",
        )
    with c3:
        metric_card(
            "Pencemar Dominan",
            dominant_param,
            "Parameter yang paling sering menjadi pencemar kritis",
        )
    with c4:
        metric_card(
            "Kategori Terbanyak",
            dominant_category,
            "Kategori kualitas udara yang paling sering muncul",
        )

    st.subheader("Kontrol Kualitas Data yang Masuk KPI")
    col_qchart, col_qnote = st.columns([1.15, 1])
    with col_qchart:
        quality_counts = quality_summary_table(filtered_before_quality)
        if not quality_counts.empty:
            fig_quality = px.bar(
                quality_counts,
                x="Status Data",
                y="Jumlah Baris",
                text="Jumlah Baris",
                title="Data Valid vs Bermasalah Sebelum Filter Kualitas",
                labels={"Jumlah Baris": "Jumlah Baris"},
            )
            fig_quality.update_traces(textposition="outside")
            fig_quality.update_layout(height=360, showlegend=False)
            st.plotly_chart(fig_quality, use_container_width=True)
    with col_qnote:
        warning_box(
            "Prinsip perhitungan KPI",
            "KPI dan rata-rata pada dashboard dihitung dari <b>data analitik terfilter</b>. "
            "Secara default, hanya data valid yang masuk perhitungan. Baris dengan flag kualitas, "
            "status validasi bermasalah, atau duplikasi tanggal-stasiun tidak ikut rata-rata kecuali filter kualitas diubah oleh user.",
        )

    st.subheader("Ringkasan Kondisi Terkini per Stasiun")
    latest_idx = (
        filtered.sort_values("tanggal_dashboard")
        .groupby("stasiun_dashboard")["tanggal_dashboard"]
        .idxmax()
    )
    latest_station = filtered.loc[
        latest_idx,
        [
            "tanggal_dashboard",
            "stasiun_dashboard",
            "ispu_dashboard",
            "kategori_dashboard",
            "critical_dashboard",
        ],
    ].copy()
    latest_station = latest_station.sort_values("ispu_dashboard", ascending=False)
    latest_station_display = latest_station.rename(
        columns={
            "tanggal_dashboard": "Tanggal Terbaru",
            "stasiun_dashboard": "Stasiun",
            "ispu_dashboard": "ISPU",
            "kategori_dashboard": "Kategori",
            "critical_dashboard": "Pencemar Dominan",
        }
    )
    latest_station_display["Tanggal Terbaru"] = latest_station_display[
        "Tanggal Terbaru"
    ].dt.strftime("%Y-%m-%d")
    latest_station_display["ISPU"] = latest_station_display["ISPU"].round(1)
    st.dataframe(latest_station_display, use_container_width=True, hide_index=True)

    col_chart1, col_chart2 = st.columns([1.2, 1])
    with col_chart1:
        fig_latest = px.bar(
            latest_station,
            x="stasiun_dashboard",
            y="ispu_dashboard",
            color="kategori_dashboard",
            color_discrete_map=category_color_map,
            category_orders={"kategori_dashboard": category_order},
            text="ispu_dashboard",
            title="Kondisi ISPU Terkini per Stasiun",
            labels={
                "stasiun_dashboard": "Stasiun",
                "ispu_dashboard": "Nilai ISPU",
                "kategori_dashboard": "Kategori",
            },
        )
        fig_latest.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig_latest.update_layout(
            height=430, xaxis_tickangle=-20, legend_title_text="Kategori"
        )
        st.plotly_chart(fig_latest, use_container_width=True)

    with col_chart2:
        category_count = filtered["kategori_dashboard"].value_counts().reset_index()
        category_count.columns = ["Kategori", "Jumlah Hari"]
        fig_category = px.pie(
            category_count,
            names="Kategori",
            values="Jumlah Hari",
            hole=0.45,
            color="Kategori",
            color_discrete_map=category_color_map,
            title="Komposisi Kategori Kualitas Udara",
        )
        fig_category.update_layout(height=430)
        st.plotly_chart(fig_category, use_container_width=True)

    worst_latest = latest_station.iloc[0]
    insight_box(
        "Analisis utama",
        f"Pada periode terpilih, rata-rata ISPU berada di angka <b>{avg_ispu:.1f}</b> dengan "
        f"proporsi hari tidak sehat sebesar <b>{format_pct(unhealthy_pct)}</b>. Parameter yang paling sering menjadi "
        f"pencemar dominan adalah <b>{dominant_param}</b>. Kondisi terkini tertinggi tercatat di "
        f"<b>{worst_latest['stasiun_dashboard']}</b> dengan ISPU <b>{worst_latest['ispu_dashboard']:.1f}</b> "
        f"dan kategori <b>{worst_latest['kategori_dashboard']}</b>.",
    )
    action_box(
        "Insight tindak lanjut untuk DLH DKI Jakarta",
        f"Prioritaskan pemantauan lapangan dan komunikasi risiko pada stasiun <b>{worst_latest['stasiun_dashboard']}</b> "
        f"ketika nilai ISPU terkini menjadi yang tertinggi, terutama bila pencemar dominannya <b>{worst_latest['critical_dashboard']}</b>. "
        "Informasi ini dapat digunakan untuk menentukan lokasi inspeksi sumber emisi, penyebaran imbauan masyarakat, "
        "dan penguatan koordinasi lintas wilayah.",
    )

# =============================================================
# Dashboard 2 — Tren Temporal
# =============================================================
with tab2:
    st.header("Dashboard 2 — Tren Temporal")

    aggregation = st.radio(
        "Pilih granularitas tren",
        options=["Harian", "Bulanan", "Tahunan"],
        horizontal=True,
    )

    if aggregation == "Harian":
        temp = filtered.groupby(
            ["tanggal_dashboard", "stasiun_dashboard"], as_index=False
        )["ispu_dashboard"].mean()
        x_col = "tanggal_dashboard"
        x_label = "Tanggal"
    elif aggregation == "Bulanan":
        temp = filtered.groupby(
            ["tahun_bulan_dashboard", "stasiun_dashboard"], as_index=False
        )["ispu_dashboard"].mean()
        x_col = "tahun_bulan_dashboard"
        x_label = "Bulan"
    else:
        temp = filtered.groupby(
            ["tahun_dashboard", "stasiun_dashboard"], as_index=False
        )["ispu_dashboard"].mean()
        x_col = "tahun_dashboard"
        x_label = "Tahun"

    fig_trend = px.line(
        temp,
        x=x_col,
        y="ispu_dashboard",
        color="stasiun_dashboard",
        markers=(aggregation == "Tahunan"),
        title=f"Tren Rata-rata ISPU {aggregation} per Stasiun",
        labels={
            x_col: x_label,
            "ispu_dashboard": "Rata-rata ISPU",
            "stasiun_dashboard": "Stasiun",
        },
    )
    fig_trend.update_layout(height=500, legend_title_text="Stasiun")
    st.plotly_chart(fig_trend, use_container_width=True)

    yearly = (
        filtered.groupby("tahun_dashboard", as_index=False)
        .agg(
            rata_rata_ispu=("ispu_dashboard", "mean"),
            persentase_tidak_sehat=("is_unhealthy_dashboard", lambda x: x.mean() * 100),
            jumlah_observasi=("ispu_dashboard", "size"),
        )
        .sort_values("tahun_dashboard")
    )

    col_y1, col_y2 = st.columns([1.3, 1])
    with col_y1:
        fig_year = px.bar(
            yearly,
            x="tahun_dashboard",
            y="rata_rata_ispu",
            text="rata_rata_ispu",
            title="Rata-rata ISPU Tahunan",
            labels={"tahun_dashboard": "Tahun", "rata_rata_ispu": "Rata-rata ISPU"},
        )
        fig_year.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig_year.update_layout(height=420)
        st.plotly_chart(fig_year, use_container_width=True)

    with col_y2:
        best_year = yearly.loc[yearly["rata_rata_ispu"].idxmin()]
        worst_year = yearly.loc[yearly["rata_rata_ispu"].idxmax()]
        st.subheader("Tahun Terbaik dan Terburuk")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Indikator": "Tahun kualitas udara terbaik",
                        "Tahun": int(best_year["tahun_dashboard"]),
                        "Rata-rata ISPU": round(best_year["rata_rata_ispu"], 1),
                        "% Hari Tidak Sehat": round(
                            best_year["persentase_tidak_sehat"], 1
                        ),
                    },
                    {
                        "Indikator": "Tahun kualitas udara terburuk",
                        "Tahun": int(worst_year["tahun_dashboard"]),
                        "Rata-rata ISPU": round(worst_year["rata_rata_ispu"], 1),
                        "% Hari Tidak Sehat": round(
                            worst_year["persentase_tidak_sehat"], 1
                        ),
                    },
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

    insight_box(
        "Analisis utama",
        f"Berdasarkan rata-rata ISPU tahunan pada filter aktif, tahun dengan kualitas udara terbaik adalah "
        f"<b>{int(best_year['tahun_dashboard'])}</b> dengan rata-rata ISPU <b>{best_year['rata_rata_ispu']:.1f}</b>. "
        f"Tahun dengan kualitas udara terburuk adalah <b>{int(worst_year['tahun_dashboard'])}</b> dengan rata-rata ISPU "
        f"<b>{worst_year['rata_rata_ispu']:.1f}</b>.",
    )
    action_box(
        "Insight tindak lanjut untuk DLH DKI Jakarta",
        "Gunakan pola tahun terbaik sebagai pembanding kebijakan, misalnya periode ketika pengendalian emisi, pembatasan aktivitas, "
        "atau kondisi meteorologis lebih mendukung penurunan ISPU. Tahun terburuk perlu dijadikan dasar evaluasi untuk memperkuat "
        "program pengendalian sumber pencemar dan sistem peringatan dini.",
    )

# =============================================================
# Dashboard 3 — Perbandingan Antar Stasiun
# =============================================================
with tab3:
    st.header("Dashboard 3 — Perbandingan Antar Stasiun")

    station_summary = (
        filtered.groupby("stasiun_dashboard", as_index=False)
        .agg(
            rata_rata_ispu=("ispu_dashboard", "mean"),
            median_ispu=("ispu_dashboard", "median"),
            maksimum_ispu=("ispu_dashboard", "max"),
            persentase_tidak_sehat=("is_unhealthy_dashboard", lambda x: x.mean() * 100),
            jumlah_hari=("ispu_dashboard", "size"),
        )
        .sort_values("rata_rata_ispu", ascending=False)
    )

    col_s1, col_s2 = st.columns([1.15, 1])
    with col_s1:
        fig_station = px.bar(
            station_summary,
            x="stasiun_dashboard",
            y="rata_rata_ispu",
            text="rata_rata_ispu",
            title="Perbandingan Rata-rata ISPU Antar Stasiun",
            labels={"stasiun_dashboard": "Stasiun", "rata_rata_ispu": "Rata-rata ISPU"},
        )
        fig_station.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig_station.update_layout(height=470, xaxis_tickangle=-20)
        st.plotly_chart(fig_station, use_container_width=True)

    with col_s2:
        fig_box = px.box(
            filtered,
            x="stasiun_dashboard",
            y="ispu_dashboard",
            points="outliers",
            title="Distribusi Nilai ISPU per Stasiun",
            labels={"stasiun_dashboard": "Stasiun", "ispu_dashboard": "Nilai ISPU"},
        )
        fig_box.update_layout(height=470, xaxis_tickangle=-20)
        st.plotly_chart(fig_box, use_container_width=True)

    category_dist = prepare_category_distribution(filtered, category_order)
    fig_cat_station = px.bar(
        category_dist,
        x="stasiun_dashboard",
        y="persentase",
        color="kategori_dashboard",
        color_discrete_map=category_color_map,
        category_orders={"kategori_dashboard": category_order},
        title="Distribusi Kategori Kualitas Udara Antar Stasiun",
        labels={
            "stasiun_dashboard": "Stasiun",
            "persentase": "Persentase Hari (%)",
            "kategori_dashboard": "Kategori",
        },
    )
    fig_cat_station.update_layout(
        height=500, barmode="stack", xaxis_tickangle=-20, legend_title_text="Kategori"
    )
    st.plotly_chart(fig_cat_station, use_container_width=True)

    station_summary_display = station_summary.copy()
    station_summary_display.columns = [
        "Stasiun",
        "Rata-rata ISPU",
        "Median ISPU",
        "ISPU Maksimum",
        "% Hari Tidak Sehat",
        "Jumlah Observasi",
    ]
    station_summary_display[
        ["Rata-rata ISPU", "Median ISPU", "ISPU Maksimum", "% Hari Tidak Sehat"]
    ] = station_summary_display[
        ["Rata-rata ISPU", "Median ISPU", "ISPU Maksimum", "% Hari Tidak Sehat"]
    ].round(1)
    st.dataframe(station_summary_display, use_container_width=True, hide_index=True)

    worst_station = station_summary.iloc[0]
    best_station = station_summary.iloc[-1]
    insight_box(
        "Analisis utama",
        f"Stasiun dengan rata-rata ISPU tertinggi adalah <b>{worst_station['stasiun_dashboard']}</b> "
        f"dengan rata-rata <b>{worst_station['rata_rata_ispu']:.1f}</b> dan proporsi hari tidak sehat "
        f"<b>{format_pct(worst_station['persentase_tidak_sehat'])}</b>. Stasiun dengan rata-rata ISPU terendah adalah "
        f"<b>{best_station['stasiun_dashboard']}</b> dengan rata-rata <b>{best_station['rata_rata_ispu']:.1f}</b>.",
    )
    action_box(
        "Insight tindak lanjut untuk DLH DKI Jakarta",
        f"Fokuskan evaluasi sumber emisi di area yang direpresentasikan oleh <b>{worst_station['stasiun_dashboard']}</b>, "
        "terutama pada periode ketika kategori TIDAK SEHAT meningkat. Stasiun dengan kondisi relatif lebih baik dapat dijadikan pembanding "
        "untuk melihat perbedaan karakteristik wilayah, lalu hasilnya diterjemahkan menjadi prioritas pengendalian yang lebih spesifik lokasi.",
    )

# =============================================================
# Dashboard 4 — Parameter Pencemar Kritis
# =============================================================
with tab4:
    st.header("Dashboard 4 — Analisis Parameter Pencemar Kritis")

    pm25_total_rows = len(filtered)
    pm25_available_rows = int(filtered["pm25_available_dashboard"].sum())
    pm25_missing_rows = pm25_total_rows - pm25_available_rows
    pm25_available_pct = (
        (pm25_available_rows / pm25_total_rows * 100) if pm25_total_rows else 0
    )

    warning_box(
        "Catatan metodologis PM2.5",
        f"Analisis PM2.5 dibatasi pada baris/periode ketika kolom <b>pm25</b> memiliki nilai. "
        f"Pada filter aktif, PM2.5 tersedia pada <b>{pm25_available_rows:,}</b> dari <b>{pm25_total_rows:,}</b> observasi "
        f"(<b>{pm25_available_pct:.1f}%</b>). Periode PM2.5 tersedia: <b>{pm25_period_text}</b>. "
        "Nilai PM2.5 yang missing tidak diimputasi agar dashboard tidak menciptakan data pencemar buatan.".replace(
            ",", "."
        ),
    )

    # Khusus komposisi dan tren pencemar dominan, baris dengan critical = PM2.5
    # hanya dipakai jika nilai pm25 tersedia. Baris pencemar lain tetap mengikuti
    # filter dashboard umum.
    param_analysis = filtered.copy()
    invalid_pm25_critical = (param_analysis["critical_dashboard"] == "PM2.5") & (
        ~param_analysis["pm25_available_dashboard"]
    )
    excluded_pm25_critical = int(invalid_pm25_critical.sum())
    param_analysis = param_analysis.loc[~invalid_pm25_critical].copy()

    if excluded_pm25_critical > 0:
        st.caption(
            f"Catatan: {excluded_pm25_critical:,} baris dengan pencemar dominan PM2.5 dikeluarkan dari analisis parameter karena nilai pm25 tidak tersedia.".replace(
                ",", "."
            )
        )

    if param_analysis.empty:
        warning_box(
            "Data parameter tidak tersedia",
            "Tidak ada data pencemar dominan yang valid pada filter aktif setelah pembatasan PM2.5. "
            "Perluas rentang tanggal atau pilih parameter selain PM2.5.",
        )
    else:
        param_count = (
            param_analysis.groupby("critical_dashboard", as_index=False)
            .size()
            .rename(columns={"size": "jumlah_hari"})
            .sort_values("jumlah_hari", ascending=False)
        )
        param_count["persentase"] = (
            param_count["jumlah_hari"] / param_count["jumlah_hari"].sum() * 100
        )

        col_p1, col_p2 = st.columns([1, 1.1])
        with col_p1:
            fig_param_pie = px.pie(
                param_count,
                names="critical_dashboard",
                values="jumlah_hari",
                hole=0.45,
                color="critical_dashboard",
                color_discrete_map=PARAMETER_COLOR,
                title="Komposisi Parameter Pencemar Dominan",
            )
            fig_param_pie.update_layout(height=450)
            st.plotly_chart(fig_param_pie, use_container_width=True)

        with col_p2:
            fig_param_bar = px.bar(
                param_count,
                x="critical_dashboard",
                y="jumlah_hari",
                text="persentase",
                color="critical_dashboard",
                color_discrete_map=PARAMETER_COLOR,
                title="Frekuensi Kemunculan Pencemar Dominan",
                labels={
                    "critical_dashboard": "Parameter",
                    "jumlah_hari": "Jumlah Hari",
                    "persentase": "Persentase",
                },
            )
            fig_param_bar.update_traces(
                texttemplate="%{text:.1f}%", textposition="outside"
            )
            fig_param_bar.update_layout(height=450, showlegend=False)
            st.plotly_chart(fig_param_bar, use_container_width=True)

        monthly_param = (
            param_analysis.groupby(
                ["tahun_bulan_dashboard", "critical_dashboard"], as_index=False
            )
            .size()
            .rename(columns={"size": "jumlah_hari"})
        )
        fig_param_trend = px.line(
            monthly_param,
            x="tahun_bulan_dashboard",
            y="jumlah_hari",
            color="critical_dashboard",
            color_discrete_map=PARAMETER_COLOR,
            title="Tren Bulanan Kemunculan Parameter Pencemar Dominan",
            labels={
                "tahun_bulan_dashboard": "Bulan",
                "jumlah_hari": "Jumlah Hari",
                "critical_dashboard": "Parameter",
            },
        )
        fig_param_trend.update_layout(height=500, legend_title_text="Parameter")
        st.plotly_chart(fig_param_trend, use_container_width=True)

        available_parameter_cols = [c for c in PARAMETER_COLS if c in filtered.columns]
        if available_parameter_cols:
            st.subheader("Rata-rata Nilai Parameter dengan Konteks Ketersediaan Data")

            total_filtered_rows = len(filtered)
            min_coverage_for_direct_comparison = 80.0
            rows = []
            label_map = {
                "pm10": "PM10",
                "pm25": "PM2.5",
                "so2": "SO2",
                "co": "CO",
                "o3": "O3",
                "no2": "NO2",
            }
            for col in available_parameter_cols:
                if col == "pm25":
                    series = filtered.loc[filtered["pm25_available_dashboard"], col]
                    basis = "Hanya data PM2.5 tersedia; tidak diimputasi"
                    catatan = (
                        "Terbatas"
                        if filtered["pm25_available_dashboard"].mean() * 100
                        < min_coverage_for_direct_comparison
                        else "Memadai"
                    )
                else:
                    series = filtered[col]
                    basis = "Data tersedia pada filter aktif"
                    catatan = "Memadai"
                n_available = int(series.notna().sum())
                coverage_pct = (
                    (n_available / total_filtered_rows * 100)
                    if total_filtered_rows
                    else 0
                )
                rows.append(
                    {
                        "parameter_kolom": col,
                        "parameter": label_map.get(col, col.upper()),
                        "rata_rata_nilai": series.mean(),
                        "jumlah_data_tersedia": n_available,
                        "coverage_pct": coverage_pct,
                        "basis_analisis": basis,
                        "status_kelengkapan": (
                            "Cakupan memadai"
                            if coverage_pct >= min_coverage_for_direct_comparison
                            else "Cakupan rendah - jangan dibandingkan langsung"
                        ),
                        "catatan": catatan,
                    }
                )

            param_mean = pd.DataFrame(rows).dropna(subset=["rata_rata_nilai"])
            if not param_mean.empty:
                pm25_row = param_mean[param_mean["parameter"] == "PM2.5"]
                if (
                    not pm25_row.empty
                    and float(pm25_row.iloc[0]["coverage_pct"])
                    < min_coverage_for_direct_comparison
                ):
                    warning_box(
                        "Catatan penting interpretasi PM2.5",
                        f"PM2.5 memiliki rata-rata <b>{pm25_row.iloc[0]['rata_rata_nilai']:.1f}</b>, tetapi hanya tersedia pada "
                        f"<b>{int(pm25_row.iloc[0]['jumlah_data_tersedia']):,}</b> dari <b>{total_filtered_rows:,}</b> observasi "
                        f"(<b>{pm25_row.iloc[0]['coverage_pct']:.1f}%</b>). Karena cakupannya rendah, PM2.5 tidak boleh dibaca sebagai "
                        "peringkat polutan tertinggi yang sepenuhnya sebanding dengan parameter lain.",
                    )

                param_mean["label_bar"] = param_mean.apply(
                    lambda r: (
                        f"{r['rata_rata_nilai']:.1f}<br>n={int(r['jumlah_data_tersedia']):,}<br>{r['coverage_pct']:.1f}%".replace(
                            ",", "."
                        )
                    ),
                    axis=1,
                )
                param_mean["parameter_label"] = param_mean.apply(
                    lambda r: (
                        f"{r['parameter']} ({r['catatan']})"
                        if r["status_kelengkapan"].startswith("Cakupan rendah")
                        else r["parameter"]
                    ),
                    axis=1,
                )

                show_low_coverage = st.checkbox(
                    "Tampilkan parameter dengan cakupan data rendah pada grafik rata-rata",
                    value=True,
                    help="Jika dimatikan, parameter dengan coverage di bawah 80% dikeluarkan dari grafik rata-rata agar tidak dibaca sebagai perbandingan langsung.",
                )
                chart_data = param_mean.copy()
                if not show_low_coverage:
                    chart_data = chart_data[
                        chart_data["coverage_pct"] >= min_coverage_for_direct_comparison
                    ]

                fig_param_value = px.bar(
                    chart_data.sort_values("rata_rata_nilai", ascending=False),
                    x="parameter_label",
                    y="rata_rata_nilai",
                    text="label_bar",
                    color="status_kelengkapan",
                    title="Rata-rata Nilai Parameter Pencemar disertai Jumlah Data dan Coverage",
                    labels={
                        "parameter_label": "Parameter",
                        "rata_rata_nilai": "Rata-rata Nilai",
                        "status_kelengkapan": "Status kelengkapan",
                    },
                    hover_data={
                        "parameter": True,
                        "jumlah_data_tersedia": True,
                        "coverage_pct": ":.1f",
                        "basis_analisis": True,
                        "rata_rata_nilai": ":.1f",
                    },
                )
                fig_param_value.update_traces(textposition="outside")
                fig_param_value.update_layout(
                    height=500, legend_title_text="Status kelengkapan"
                )
                st.plotly_chart(fig_param_value, use_container_width=True)

                fig_coverage = px.bar(
                    param_mean.sort_values("coverage_pct", ascending=False),
                    x="parameter",
                    y="coverage_pct",
                    text="coverage_pct",
                    color="status_kelengkapan",
                    title="Coverage Data per Parameter Pencemar",
                    labels={
                        "parameter": "Parameter",
                        "coverage_pct": "Coverage Data (%)",
                        "status_kelengkapan": "Status kelengkapan",
                    },
                    hover_data={"jumlah_data_tersedia": True},
                )
                fig_coverage.add_hline(
                    y=min_coverage_for_direct_comparison,
                    line_dash="dash",
                    annotation_text="Ambang pembanding langsung 80%",
                    annotation_position="top left",
                )
                fig_coverage.update_traces(
                    texttemplate="%{text:.1f}%", textposition="outside"
                )
                fig_coverage.update_layout(
                    height=420, legend_title_text="Status kelengkapan"
                )
                st.plotly_chart(fig_coverage, use_container_width=True)

                param_table = param_mean[
                    [
                        "parameter",
                        "rata_rata_nilai",
                        "jumlah_data_tersedia",
                        "coverage_pct",
                        "status_kelengkapan",
                        "basis_analisis",
                    ]
                ].copy()
                param_table = param_table.rename(
                    columns={
                        "parameter": "Parameter",
                        "rata_rata_nilai": "Rata-rata Nilai",
                        "jumlah_data_tersedia": "Jumlah Data Tersedia",
                        "coverage_pct": "Coverage (%)",
                        "status_kelengkapan": "Status Kelengkapan",
                        "basis_analisis": "Basis Analisis",
                    }
                )
                param_table[["Rata-rata Nilai", "Coverage (%)"]] = param_table[
                    ["Rata-rata Nilai", "Coverage (%)"]
                ].round(1)
                st.dataframe(param_table, use_container_width=True, hide_index=True)

                st.caption(
                    "Grafik rata-rata parameter tidak berdiri sendiri. Gunakan bersama coverage data. "
                    "PM2.5 tetap ditampilkan sebagai informasi penting, tetapi diberi label cakupan rendah bila data tersedia sedikit."
                )

        top_param = param_count.iloc[0]
        insight_box(
            "Analisis utama",
            f"Parameter yang paling sering menjadi pencemar dominan adalah <b>{top_param['critical_dashboard']}</b> "
            f"dengan <b>{int(top_param['jumlah_hari']):,}</b> observasi atau sekitar <b>{top_param['persentase']:.1f}%</b> "
            "dari total data parameter valid pada filter aktif.".replace(",", "."),
        )
        action_box(
            "Insight tindak lanjut untuk DLH DKI Jakarta",
            f"Susun intervensi spesifik berdasarkan pencemar dominan <b>{top_param['critical_dashboard']}</b>. "
            "Untuk PM2.5, gunakan hasil hanya pada periode ketika data tersedia sehingga kebijakan tidak didasarkan pada estimasi buatan. "
            "Jika partikulat halus dominan, pengawasan dapat diarahkan pada sumber debu jalan, transportasi, konstruksi, dan pembakaran terbuka; "
            "sedangkan ketika ozon dominan, evaluasi perlu memperhatikan prekursor emisi dan kondisi pembentukan ozon.",
        )

# =============================================================
# Dashboard 5 — Pola Musiman
# =============================================================
with tab5:
    st.header("Dashboard 5 — Pola Musiman")

    seasonal = filtered.groupby(
        ["bulan_dashboard", "stasiun_dashboard"], as_index=False
    ).agg(
        rata_rata_ispu=("ispu_dashboard", "mean"),
        persentase_tidak_sehat=("is_unhealthy_dashboard", lambda x: x.mean() * 100),
        jumlah_hari=("ispu_dashboard", "size"),
    )
    seasonal["bulan_nama"] = seasonal["bulan_dashboard"].map(month_name_id)

    heatmap_matrix = seasonal.pivot(
        index="stasiun_dashboard", columns="bulan_dashboard", values="rata_rata_ispu"
    )
    month_labels = [month_name_id(m) for m in heatmap_matrix.columns]

    fig_heatmap = go.Figure(
        data=go.Heatmap(
            z=heatmap_matrix.values,
            x=month_labels,
            y=heatmap_matrix.index.tolist(),
            colorscale="YlOrRd",
            colorbar=dict(title="Rata-rata ISPU"),
            hovertemplate="Bulan=%{x}<br>Stasiun=%{y}<br>Rata-rata ISPU=%{z:.1f}<extra></extra>",
        )
    )
    fig_heatmap.update_layout(
        title="Heatmap Rata-rata ISPU berdasarkan Bulan dan Stasiun",
        height=480,
        xaxis_title="Bulan",
        yaxis_title="Stasiun",
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

    monthly_overall = (
        filtered.groupby("bulan_dashboard", as_index=False)
        .agg(
            rata_rata_ispu=("ispu_dashboard", "mean"),
            persentase_tidak_sehat=("is_unhealthy_dashboard", lambda x: x.mean() * 100),
            jumlah_hari=("ispu_dashboard", "size"),
        )
        .sort_values("bulan_dashboard")
    )
    monthly_overall["bulan_nama"] = monthly_overall["bulan_dashboard"].map(
        month_name_id
    )

    col_m1, col_m2 = st.columns([1, 1])
    with col_m1:
        fig_month_avg = px.line(
            monthly_overall,
            x="bulan_nama",
            y="rata_rata_ispu",
            markers=True,
            title="Rata-rata ISPU per Bulan",
            labels={"bulan_nama": "Bulan", "rata_rata_ispu": "Rata-rata ISPU"},
        )
        fig_month_avg.update_layout(
            height=430,
            xaxis_categoryorder="array",
            xaxis_categoryarray=[month_name_id(i) for i in range(1, 13)],
        )
        st.plotly_chart(fig_month_avg, use_container_width=True)

    with col_m2:
        fig_month_unhealthy = px.bar(
            monthly_overall,
            x="bulan_nama",
            y="persentase_tidak_sehat",
            text="persentase_tidak_sehat",
            title="Persentase Hari Tidak Sehat per Bulan",
            labels={
                "bulan_nama": "Bulan",
                "persentase_tidak_sehat": "% Hari Tidak Sehat",
            },
        )
        fig_month_unhealthy.update_traces(
            texttemplate="%{text:.1f}%", textposition="outside"
        )
        fig_month_unhealthy.update_layout(
            height=430,
            xaxis_categoryorder="array",
            xaxis_categoryarray=[month_name_id(i) for i in range(1, 13)],
        )
        st.plotly_chart(fig_month_unhealthy, use_container_width=True)

    worst_month = monthly_overall.loc[monthly_overall["rata_rata_ispu"].idxmax()]
    best_month = monthly_overall.loc[monthly_overall["rata_rata_ispu"].idxmin()]
    worst_unhealthy_month = monthly_overall.loc[
        monthly_overall["persentase_tidak_sehat"].idxmax()
    ]

    insight_box(
        "Analisis utama",
        f"Bulan dengan rata-rata ISPU tertinggi adalah <b>{worst_month['bulan_nama']}</b> "
        f"dengan rata-rata ISPU <b>{worst_month['rata_rata_ispu']:.1f}</b>. Bulan dengan rata-rata ISPU terendah adalah "
        f"<b>{best_month['bulan_nama']}</b> dengan rata-rata ISPU <b>{best_month['rata_rata_ispu']:.1f}</b>. "
        f"Persentase hari tidak sehat tertinggi muncul pada bulan <b>{worst_unhealthy_month['bulan_nama']}</b> "
        f"sebesar <b>{format_pct(worst_unhealthy_month['persentase_tidak_sehat'])}</b>.",
    )
    action_box(
        "Insight tindak lanjut untuk DLH DKI Jakarta",
        f"Perkuat program pencegahan sebelum memasuki bulan <b>{worst_month['bulan_nama']}</b>, "
        "misalnya melalui inspeksi sumber emisi, pengendalian debu/konstruksi, koordinasi transportasi, dan komunikasi risiko kesehatan. "
        "Pola musiman ini membantu DLH menyusun kalender intervensi, bukan hanya merespons setelah kualitas udara memburuk.",
    )

st.divider()
st.caption(
    "© 2026 • Dashboard BI ISPU DKI Jakarta | Dibuat untuk demonstrasi interaktif analisis kualitas udara."
)
