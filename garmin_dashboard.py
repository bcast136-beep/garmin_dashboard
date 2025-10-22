# garmin_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, recall_score

# -------------------------
# Page & Header
# -------------------------
st.set_page_config(page_title="Garmin HRV Status Forecast", layout="wide")
st.image("https://upload.wikimedia.org/wikipedia/commons/3/3d/Garmin_logo.svg", width=140)
st.title("🩺 Garmin HRV Status Forecast (Next 1–3 Days)")

st.markdown("""
This dashboard ingests **Garmin HRV Status** exports (columns like `Date`, `Overnight HRV`, `Baseline`, `7-day average`)
and predicts **tomorrow’s HRV status** (Low / Stable / High) — with optional recursive forecasts out to 3 days.
""")

# -------------------------
# Sidebar (Config)
# -------------------------
st.sidebar.header("⚙️ Configuration")

uploaded = st.sidebar.file_uploader("📂 Upload Garmin HRV Status CSV", type=["csv"])

# Deviation threshold (ms) to label Low/High vs Stable
dev_threshold = st.sidebar.slider("Deviation threshold (ms) for Low/High vs Stable",
                                  min_value=3, max_value=15, value=5, step=1)
forecast_horizon = st.sidebar.selectbox("Forecast horizon (days ahead)",
                                        options=[1, 2, 3], index=0)

# Rolling window (days) for engineered trends
roll_days = st.sidebar.slider("Rolling window for trend features (days)",
                              min_value=3, max_value=14, value=7, step=1)

st.sidebar.markdown("---")
st.sidebar.info("Tip: If your export is daily (overnight HRV), this app forecasts **days**. "
                "For **hourly** forecasts, import higher-frequency HRV/Stress logs.")

# -------------------------
# Load Data
# -------------------------
def load_sample():
    # A tiny inline sample to keep the app demo-able if no file uploaded.
    # Replace with your real export or keep as is for demos.
    dates = pd.date_range("2025-09-15", periods=40, freq="D")
    rng = np.random.default_rng(42)
    baseline = 55 + rng.normal(0, 1.2, size=len(dates)).cumsum()/20 + 45  # smooth-ish baseline ~ 45-65
    overnight = baseline + rng.normal(0, 6, size=len(dates))              # overnight fluctuates around baseline
    avg7 = pd.Series(overnight).rolling(7).mean().fillna(method="bfill")
    df = pd.DataFrame({
        "Date": dates,
        "Overnight HRV": np.round(overnight, 1),
        "Baseline": np.round(baseline, 1),
        "7-day average": np.round(avg7, 1)
    })
    return df

if uploaded is not None:
    raw = pd.read_csv(uploaded)
    st.sidebar.success("✅ File uploaded")
else:
    st.sidebar.info("Using built-in sample dataset")
    raw = load_sample()

# -------------------------
# Normalize column names & parse
# -------------------------
df = raw.copy()
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
# Attempt to detect/rename common variants
col_map = {
    "date": "date",
    "overnight_hrv": "overnight_hrv",
    "overnight": "overnight_hrv",
    "baseline": "baseline",
    "7_day_average": "seven_day_average",
    "7-day_average": "seven_day_average",
    "7-day_avg": "seven_day_average",
    "seven_day_average": "seven_day_average"
}
# Create a normalized frame with expected cols if present
norm = {}
for k, v in col_map.items():
    if k in df.columns:
        norm[v] = df[k]
df = pd.DataFrame(norm)

# Basic validation
required_cols = ["date", "overnight_hrv", "baseline", "seven_day_average"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Your file is missing required column(s): {missing}. "
             f"Found columns: {list(raw.columns)}")
    st.stop()

# Parse date
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# -------------------------
# Feature Engineering
# -------------------------
# Core deviations and change rates
df["hrv_deviation"]     = df["overnight_hrv"] - df["baseline"]
df["weekly_deviation"]  = df["overnight_hrv"] - df["seven_day_average"]
df["hrv_change_rate"]   = df["overnight_hrv"].pct_change()

# Rolling stats (over 'roll_days')
df["overnight_roll_mean"] = df["overnight_hrv"].rolling(roll_days).mean()
df["overnight_roll_std"]  = df["overnight_hrv"].rolling(roll_days).std()
df["dev_roll_mean"]       = df["hrv_deviation"].rolling(roll_days).mean()
df["dev_roll_std"]        = df["hrv_deviation"].rolling(roll_days).std()

# Fill initial NaNs from rolling with back/forward fill to keep rows
df = df.ffill().bfill()

# Create CURRENT status (for plotting only)
def status_from_dev(d):
    if d > dev_threshold:
        return 1   # High
    if d < -dev_threshold:
        return -1  # Low
    return 0       # Stable

df["status_today"] = df["hrv_deviation"].apply(status_from_dev)

# Create FUTURE label (target) = tomorrow's status (shift -1 day)
df["future_status"] = df["status_today"].shift(-1)

# Drop the final row without a future label
df_model = df.dropna(subset=["future_status"]).copy()
df_model["future_status"] = df_model["future_status"].astype(int)

# Feature set for the model
feature_cols = [
    "overnight_hrv", "baseline", "seven_day_average",
    "hrv_deviation", "weekly_deviation", "hrv_change_rate",
    "overnight_roll_mean", "overnight_roll_std",
    "dev_roll_mean", "dev_roll_std"
]
X_all = df_model[feature_cols].copy()
y_all = df_model["future_status"].copy()

# Handle any potential inf/nan in engineered features
X_all = X_all.replace([np.inf, -np.inf], np.nan).fillna(method="ffill").fillna(method="bfill")

# -------------------------
# Train / Test Split & Model
# -------------------------
# Class imbalance: use class_weight='balanced' to avoid extra dependencies
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.3, random_state=42, stratify=y_all
)

model = RandomForestClassifier(
    n_estimators=400,
    max_depth=10,
    min_samples_split=4,
    class_weight="balanced",
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
macro_rec = recall_score(y_test, y_pred, average="macro")
report = classification_report(
    y_test, y_pred,
    target_names=["Low (-1)", "Stable (0)", "High (1)"],
    digits=3
)

# -------------------------
# Top Layout: Data & Metrics
# -------------------------
st.markdown("### 📋 Data Preview")
st.dataframe(df.tail(10), use_container_width=True, hide_index=True)

m1, m2, m3 = st.columns(3)
m1.metric("Accuracy", f"{acc:.2f}")
m2.metric("Macro Recall", f"{macro_rec:.2f}")
m3.metric("Training Samples", f"{len(X_train)}")

with st.expander("Show detailed classification report"):
    st.text(report)

# -------------------------
# Feature Importance
# -------------------------
st.markdown("### 📊 Feature Importance")
importances = model.feature_importances_
order = np.argsort(importances)[::-1]
labels = [feature_cols[i] for i in order]
fig_imp, ax_imp = plt.subplots(figsize=(8,4))
ax_imp.bar(range(len(labels)), importances[order], color="royalblue")
ax_imp.set_xticks(range(len(labels)))
ax_imp.set_xticklabels(labels, rotation=45, ha="right")
ax_imp.set_ylabel("Importance")
ax_imp.set_title("Which inputs most influence tomorrow's HRV status?")
st.pyplot(fig_imp)

# -------------------------
# Trend Plot (Overnight HRV vs Baseline vs 7-day avg)
# -------------------------
st.markdown("### 📈 HRV Trends vs Baseline")
fig_trend, ax_t = plt.subplots(figsize=(10,4))
ax_t.plot(df["date"], df["overnight_hrv"], label="Overnight HRV", linewidth=2)
ax_t.plot(df["date"], df["baseline"], label="Baseline", linewidth=1.8)
ax_t.plot(df["date"], df["seven_day_average"], label="7-day average", linewidth=1.8)
ax_t.set_ylabel("HRV (ms)")
ax_t.set_xlabel("Date")
ax_t.legend(loc="upper left")
ax_t.set_title("Overnight HRV compared to Baseline & Weekly Trend")
ax_t.xaxis.set_major_locator(mdates.AutoDateLocator())
ax_t.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
fig_trend.autofmt_xdate(rotation=30)
st.pyplot(fig_trend)

# -------------------------
# Next-Day & Multi-Day Forecast (Recursive)
# -------------------------
st.markdown("### 🔮 Forecast: Predicted HRV Status")

def predict_next_day_status(latest_row: pd.Series) -> tuple[int, float]:
    """Return (status_class, confidence) for next day using trained model."""
    X_latest = latest_row[feature_cols].to_frame().T.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    probs = model.predict_proba(X_latest)[0]
    classes = model.classes_  # should be [-1, 0, 1]
    idx = np.argmax(probs)
    return int(classes[idx]), float(probs[idx])

def format_status_label(x: int) -> str:
    return { -1: "Low", 0: "Stable", 1: "High" }.get(int(x), "Unknown")

def color_for_status(x: int) -> str:
    return { -1: "red", 0: "gold", 1: "green" }.get(int(x), "gray")

# Build a working copy to simulate rolling forward
df_fore = df.copy()

# We’ll forecast n days ahead recursively by:
# 1) Predicting tomorrow’s status from today’s features
# 2) Updating engineered features with a simple scenario assumption:
#    if predicted High → overnight_hrv slightly above baseline
#    if predicted Low  → slightly below baseline
#    if Stable         → close to 7-day avg
# (This is a lightweight, presentation-friendly approach)
future_rows = []
cur = df_fore.iloc[-1:].copy()  # start from the latest day

for step in range(1, forecast_horizon + 1):
    # Predict next-day status
    status_pred, conf = predict_next_day_status(cur.iloc[-1])

    # Synthesize a plausible next-day "overnight_hrv" from status (for iterative features)
    base = cur.iloc[-1]["baseline"]
    avg7 = cur.iloc[-1]["seven_day_average"]
    if status_pred == 1:   # High
        overnight_next = base + dev_threshold + np.random.normal(1.0, 0.5)
    elif status_pred == -1:  # Low
        overnight_next = base - dev_threshold + np.random.normal(-1.0, 0.5)
    else:  # Stable
        overnight_next = avg7 + np.random.normal(0.0, 0.5)

    next_date = cur.iloc[-1]["date"] + pd.Timedelta(days=1)
    next_baseline = cur.iloc[-1]["baseline"] + np.random.normal(0.0, 0.1)   # tiny drift
    # Update 7-day average via simple rolling update
    hist = pd.concat([df_fore[["date","overnight_hrv"]], pd.DataFrame({"date":[next_date], "overnight_hrv":[overnight_next]})]).sort_values("date")
    seven_day_avg_next = hist["overnight_hrv"].rolling(7).mean().iloc[-1]
    if np.isnan(seven_day_avg_next):
        seven_day_avg_next = hist["overnight_hrv"].tail(3).mean()

    row = {
        "date": next_date,
        "overnight_hrv": float(overnight_next),
        "baseline": float(next_baseline),
        "seven_day_average": float(seven_day_avg_next)
    }
    # Compute engineered features for the new row
    row["hrv_deviation"]     = row["overnight_hrv"] - row["baseline"]
    row["weekly_deviation"]  = row["overnight_hrv"] - row["seven_day_average"]
    prev_overnight = cur.iloc[-1]["overnight_hrv"]
    row["hrv_change_rate"]   = (row["overnight_hrv"] - prev_overnight) / max(prev_overnight, 1e-6)

    # Rolling proxies: reuse same window length; approximate by using last window on concatenated series
    tmp = pd.concat([df_fore, pd.DataFrame([row])], ignore_index=True)
    tmp["overnight_roll_mean"] = tmp["overnight_hrv"].rolling(roll_days).mean()
    tmp["overnight_roll_std"]  = tmp["overnight_hrv"].rolling(roll_days).std()
    tmp["dev_roll_mean"]       = tmp["hrv_deviation"].rolling(roll_days).mean()
    tmp["dev_roll_std"]        = tmp["hrv_deviation"].rolling(roll_days).std()
    tail = tmp.iloc[-1][["overnight_roll_mean","overnight_roll_std","dev_roll_mean","dev_roll_std"]].to_dict()
    row.update(tail)

    row["pred_status"] = status_pred
    row["confidence"]  = conf

    future_rows.append(row)
    # Append to cur/df_fore to feed the next iteration
    cur = pd.concat([cur, pd.DataFrame([row])], ignore_index=True)
    df_fore = pd.concat([df_fore, pd.DataFrame([row])], ignore_index=True)

# Present forecast table
if future_rows:
    fut = pd.DataFrame(future_rows)
    fut["status_label"] = fut["pred_status"].apply(format_status_label)
    fut["conf_pct"] = (fut["confidence"] * 100).round(1)
    st.dataframe(
        fut[["date","overnight_hrv","baseline","seven_day_average","status_label","conf_pct"]],
        use_container_width=True, hide_index=True
    )

    # Plot confidence and status
    st.markdown("#### Forecast Confidence & Status")
    fig_fc, ax1 = plt.subplots(figsize=(10,4))
    ax1.plot(fut["date"], fut["confidence"], linewidth=2, label="Confidence (0–1)")
    ax1.set_ylabel("Confidence")
    ax1.set_xlabel("Date")
    ax1.set_ylim(0, 1)
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax1.grid(True, alpha=0.2)

    ax2 = ax1.twinx()
    ax2.scatter(fut["date"], fut["pred_status"], c=[color_for_status(x) for x in fut["pred_status"]],
                label="Predicted Status", s=60)
    ax2.set_ylabel("Status (-1=Low, 0=Stable, 1=High)")
    ax1.set_title(f"Predicted HRV Status for Next {forecast_horizon} Day(s)")
    fig_fc.autofmt_xdate(rotation=30)

    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    st.pyplot(fig_fc)

# -------------------------
# Download sample (optional)
# -------------------------
st.markdown("---")
st.caption("Need a sample file with the expected columns? Download one below.")
sample_csv = df[["date","overnight_hrv","baseline","seven_day_average"]].copy()
sample_csv = sample_csv.rename(columns={
    "date":"Date",
    "overnight_hrv":"Overnight HRV",
    "baseline":"Baseline",
    "seven_day_average":"7-day average"
})
st.download_button(
    "⬇️ Download Sample HRV Status CSV",
    data=sample_csv.to_csv(index=False).encode("utf-8"),
    file_name="garmin_hrv_status_sample.csv",
    mime="text/csv"
)
