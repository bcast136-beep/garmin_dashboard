# garmin_dashboard.py
import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# =========================
# Page / Sidebar
# =========================
st.set_page_config(page_title="Garmin HRV Status Forecast (Next 1–3 days)", layout="wide")
st.title("🩺 Garmin HRV Status Forecast — Overnight HRV (Next 1–3 days)")

st.sidebar.header("⚙️ Settings")
local_default_path = "HRV Status Garmin.csv"  # your exact file name
uploaded = st.sidebar.file_uploader("📂 Upload your **HRV Status Garmin.csv**", type=["csv"])
forecast_horizon = st.sidebar.selectbox("Forecast horizon (days ahead)", [1, 2, 3], index=2)
roll_days = st.sidebar.slider("Rolling window for trend features (days)", 3, 14, 7, 1)

st.sidebar.markdown("---")
st.sidebar.info("This app expects columns like: **Date**, **Overnight HRV**, **Baseline** (e.g., `88ms - 107ms`), **7d Avg`**.\n"
                "It trains a small regression model on ≤28 days and predicts the next 1–3 days of **Overnight HRV**.")

# =========================
# File loading
# =========================
def load_dataframe():
    # Priority: uploaded -> local file
    if uploaded is not None:
        st.success("✅ File uploaded.")
        return pd.read_csv(uploaded)
    if os.path.exists(local_default_path):
        st.success("✅ Using local file: HRV Status Garmin.csv")
        return pd.read_csv(local_default_path)
    st.error("❌ No data file found. Upload `HRV Status Garmin.csv` or place it next to this script.")
    st.stop()

raw = load_dataframe()

# =========================
# Parsing helpers
# =========================
def parse_date_any(s):
    s = str(s).strip()
    # If no 4-digit year, append current year
    if not re.search(r"\b\d{4}\b", s):
        s = f"{s} {pd.Timestamp.today().year}"
    dt = pd.to_datetime(s, errors="coerce", utc=True, infer_datetime_format=True)
    return dt.tz_convert(None) if dt is not pd.NaT else dt

def to_ms(x):
    if pd.isna(x): return np.nan
    s = str(x).lower().replace("ms", "").strip()
    # Sometimes Garmin writes '—' or blanks
    s = s.replace("—", "").replace("--", "").strip()
    return pd.to_numeric(s, errors="coerce")

def parse_baseline_range(x):
    """Return (low, high, mid) from '88ms - 107ms'."""
    if pd.isna(x): return (np.nan, np.nan, np.nan)
    s = str(x).lower().replace(" ", "")
    m = re.match(r"(\d+)\s*ms?-+(\d+)\s*ms?", s)
    if not m:
        # Sometimes baseline appears as a single number
        val = to_ms(x)
        return (val, val, val)
    low, high = float(m.group(1)), float(m.group(2))
    return (low, high, (low + high) / 2.0)

# =========================
# Normalize columns and clean
# =========================
df = raw.copy()

# Normalize headers (lowercase, underscores)
df.columns = df.columns.str.strip()

# Expected names (tolerant)
# We’ll map common variants to a canonical set
col_map = {}
for c in df.columns:
    c_norm = c.strip().lower().replace(" ", "_")
    if c_norm in ("date",):
        col_map[c] = "date"
    elif c_norm in ("overnight_hrv", "overnight_hrv_(ms)", "overnight", "overnight_average_hrv_(ms)"):
        col_map[c] = "overnight_hrv"
    elif c_norm in ("baseline", "hrv_baseline_(ms)"):
        col_map[c] = "baseline"
    elif c_norm in ("7d_avg", "7d_average", "7-day_avg", "7-day_average", "7_day_average", "7d_avg_(ms)"):
        col_map[c] = "seven_day_avg"
# Apply the mapping (keep only known)
df = df.rename(columns=col_map)
required = ["date", "overnight_hrv", "baseline", "seven_day_avg"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"❌ Missing required column(s): {missing}\n\nFound columns: {list(raw.columns)}")
    st.stop()

# Parse/clean each field
df = df[required].copy()
df["date"] = df["date"].apply(parse_date_any)
df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

# Overnight HRV and 7d Avg are like '109ms'
df["overnight_hrv"] = df["overnight_hrv"].apply(to_ms)
df["seven_day_avg"] = df["seven_day_avg"].apply(to_ms)

# Baseline as a range '88ms - 107ms'
baseline_parsed = df["baseline"].apply(parse_baseline_range)
df["baseline_low"]  = [t[0] for t in baseline_parsed]
df["baseline_high"] = [t[1] for t in baseline_parsed]
df["baseline_mid"]  = [t[2] for t in baseline_parsed]

# Basic NA handling
df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill()

# Guardrails for small Garmin export
n_rows = len(df)
if n_rows > 28:
    st.warning(f"ℹ️ Detected {n_rows} rows (more than typical 28-day export). Proceeding with all rows.")
elif n_rows < 8:
    st.warning(f"⚠️ Only {n_rows} rows detected — forecasts may be unstable. More days improve reliability.")

# =========================
# Feature engineering
# =========================
# Deviations and rates
df["dev_vs_baseline"] = df["overnight_hrv"] - df["baseline_mid"]
df["dev_vs_7davg"]    = df["overnight_hrv"] - df["seven_day_avg"]
df["hrv_change_rate"] = df["overnight_hrv"].pct_change()

# Rolling features (short windows to fit 28 rows)
df["hrv_roll_mean"] = df["overnight_hrv"].rolling(roll_days).mean()
df["hrv_roll_std"]  = df["overnight_hrv"].rolling(roll_days).std()
df["dev_roll_mean"] = df["dev_vs_baseline"].rolling(roll_days).mean()
df["dev_roll_std"]  = df["dev_vs_baseline"].rolling(roll_days).std()

df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill()

# =========================
# Data preview
# =========================
st.markdown("### 📋 Data Preview (cleaned)")
st.dataframe(
    df[["date","overnight_hrv","baseline_low","baseline_high","baseline_mid","seven_day_avg"]],
    use_container_width=True, hide_index=True
)

# =========================
# Model: Predict Overnight HRV (regression)
# =========================
feature_cols = [
    "baseline_low","baseline_high","baseline_mid",
    "seven_day_avg",
    "dev_vs_baseline","dev_vs_7davg","hrv_change_rate",
    "hrv_roll_mean","hrv_roll_std","dev_roll_mean","dev_roll_std"
]
target_col = "overnight_hrv"

X_all = df[feature_cols].copy()
y_all = df[target_col].copy()

# Handle any residual NaNs
X_all = X_all.replace([np.inf, -np.inf], np.nan).ffill().bfill()
y_all = y_all.replace([np.inf, -np.inf], np.nan).ffill().bfill()

# Small, interpretable model suited for ≤28 rows
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=6,
    min_samples_split=3,
    random_state=42
)

# Cross-validated metrics (don’t waste rows)
if len(df) >= 8:
    n_splits = min(5, max(2, len(df)//5))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mae_scores = -cross_val_score(model, X_all, y_all, cv=kf, scoring="neg_mean_absolute_error")
    r2_scores  =  cross_val_score(model, X_all, y_all, cv=kf, scoring="r2")
    cv_mae = float(mae_scores.mean())
    cv_r2  = float(r2_scores.mean())
else:
    cv_mae, cv_r2 = np.nan, np.nan

# Fit once for explanation and forecasting
try:
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y_all, test_size=0.2 if len(df) > 12 else 0.1, random_state=42
    )
except ValueError:
    X_tr, X_te, y_tr, y_te = X_all, X_all, y_all, y_all

model.fit(X_tr, y_tr)
y_pred_holdout = model.predict(X_te)
holdout_mae = mean_absolute_error(y_te, y_pred_holdout)
holdout_r2  = r2_score(y_te, y_pred_holdout)

# Metrics panel
st.markdown("### 🧪 Model Performance")
c1, c2, c3 = st.columns(3)
c1.metric("CV MAE (ms)", f"{cv_mae:.2f}" if not np.isnan(cv_mae) else "—")
c2.metric("CV R²", f"{cv_r2:.2f}" if not np.isnan(cv_r2) else "—")
c3.metric("Holdout MAE (ms)", f"{holdout_mae:.2f}")

# =========================
# Forecast next 1–3 days (recursive)
# =========================
st.markdown("### 🔮 Forecast (Overnight HRV)")
hist = df.copy()

def build_feature_row(frame: pd.DataFrame, idx: int) -> pd.Series:
    """Build features at row idx (expects engineered columns present)."""
    row = frame.iloc[idx].copy()
    return row[feature_cols]

def append_forecast_row(frame: pd.DataFrame, next_date, next_hrv):
    """Append a new day and recompute engineered columns needed for next step."""
    # Keep baseline trend flat with tiny drift from last known
    base_low  = frame["baseline_low"].iloc[-1]  + np.random.normal(0, 0.05)
    base_high = frame["baseline_high"].iloc[-1] + np.random.normal(0, 0.05)
    base_mid  = (base_low + base_high) / 2.0

    # Update 7d average with rolling mean of overnight_hrv including forecast
    temp = pd.concat([frame[["date","overnight_hrv"]],
                      pd.DataFrame({"date":[next_date], "overnight_hrv":[next_hrv]})],
                     ignore_index=True).sort_values("date")
    sev_next = temp["overnight_hrv"].rolling(7).mean().iloc[-1]
    if np.isnan(sev_next):
        sev_next = temp["overnight_hrv"].tail(3).mean()

    new = {
        "date": next_date,
        "overnight_hrv": float(next_hrv),
        "baseline_low": float(base_low),
        "baseline_high": float(base_high),
        "baseline_mid": float(base_mid),
        "seven_day_avg": float(sev_next)
    }
    new["dev_vs_baseline"] = new["overnight_hrv"] - new["baseline_mid"]
    new["dev_vs_7davg"]    = new["overnight_hrv"] - new["seven_day_avg"]
    prev_hrv = frame["overnight_hrv"].iloc[-1]
    new["hrv_change_rate"] = (new["overnight_hrv"] - prev_hrv) / max(prev_hrv, 1e-6)

    # For rolling features, compute on concatenated frame
    tmp = pd.concat([frame, pd.DataFrame([new])], ignore_index=True)
    tmp["hrv_roll_mean"] = tmp["overnight_hrv"].rolling(roll_days).mean()
    tmp["hrv_roll_std"]  = tmp["overnight_hrv"].rolling(roll_days).std()
    tmp["dev_roll_mean"] = tmp["dev_vs_baseline"].rolling(roll_days).mean()
    tmp["dev_roll_std"]  = tmp["dev_vs_baseline"].rolling(roll_days).std()

    # Pull the last computed rolling values
    new["hrv_roll_mean"] = float(tmp["hrv_roll_mean"].iloc[-1])
    new["hrv_roll_std"]  = float(tmp["hrv_roll_std"].iloc[-1]) if not np.isnan(tmp["hrv_roll_std"].iloc[-1]) else 0.0
    new["dev_roll_mean"] = float(tmp["dev_roll_mean"].iloc[-1])
    new["dev_roll_std"]  = float(tmp["dev_roll_std"].iloc[-1]) if not np.isnan(tmp["dev_roll_std"].iloc[-1]) else 0.0

    return pd.concat([frame, pd.DataFrame([new])], ignore_index=True)

future_points = []
working = hist.copy()
for step in range(1, forecast_horizon + 1):
    last_idx = len(working) - 1
    feats = build_feature_row(working, last_idx).to_frame().T.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    next_pred = float(model.predict(feats)[0])
    next_date = working["date"].iloc[-1] + pd.Timedelta(days=1)

    future_points.append({"date": next_date, "overnight_hrv": next_pred, "type": "Forecast"})
    working = append_forecast_row(working, next_date, next_pred)

# =========================
# Plot: history + forecast
# =========================
plot_df_hist = hist[["date","overnight_hrv"]].copy()
plot_df_hist["type"] = "History"
plot_df = pd.concat([plot_df_hist, pd.DataFrame(future_points)], ignore_index=True)

st.markdown("#### Overnight HRV — History and Forecast")
fig, ax = plt.subplots(figsize=(10,4))
hist_mask = plot_df["type"] == "History"
fc_mask   = plot_df["type"] == "Forecast"

ax.plot(plot_df.loc[hist_mask, "date"], plot_df.loc[hist_mask, "overnight_hrv"],
        label="History", linewidth=2)
ax.plot(plot_df.loc[fc_mask, "date"], plot_df.loc[fc_mask, "overnight_hrv"],
        label=f"Forecast (+{forecast_horizon}d)", linewidth=2, linestyle="--")

ax.set_ylabel("Overnight HRV (ms)")
ax.set_xlabel("Date")
ax.legend(loc="upper left")
ax.grid(True, alpha=0.25)
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
fig.autofmt_xdate(rotation=30)
st.pyplot(fig)

# =========================
# Download your ORIGINAL file (unchanged)
# =========================
st.markdown("---")
st.caption("Download the **exact original** Garmin export (no modifications).")

original_path = local_default_path
if uploaded is not None:
    # If uploaded, offer to download exactly the uploaded bytes
    uploaded.seek(0)
    file_bytes = uploaded.read()
    st.download_button(
        "⬇️ Download Original (uploaded) HRV Status Garmin.csv",
        data=file_bytes,
        file_name="HRV Status Garmin.csv",
        mime="text/csv"
    )
elif os.path.exists(original_path):
    with open(original_path, "rb") as f:
        file_bytes = f.read()
    st.download_button(
        "⬇️ Download Original HRV Status Garmin.csv",
        data=file_bytes,
        file_name="HRV Status Garmin.csv",
        mime="text/csv"
    )
else:
    st.info("Upload your Garmin file to enable original-file download.")
    
# Make forecast visually smoother and connect better to history
ax.plot(
    plot_df.loc[fc_mask, "date"], 
    plot_df.loc[fc_mask, "overnight_hrv"],
    label=f"Forecast (+{forecast_horizon}d)",
    linewidth=2,
    linestyle="--",
    color="darkorange",
    marker="o"
)

# Keep y-axis consistent with historical trend
ymin = max(0, plot_df["overnight_hrv"].min() - 5)
ymax = plot_df["overnight_hrv"].max() + 5
ax.set_ylim(ymin, ymax)

# Add annotation to clarify forecast region
last_hist_date = plot_df.loc[hist_mask, "date"].max()
ax.axvline(x=last_hist_date, color="gray", linestyle=":", alpha=0.7)
ax.text(last_hist_date, ymax - 5, "Forecast starts →", rotation=0, color="gray", fontsize=9)

