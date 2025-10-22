# garmin_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report, accuracy_score, recall_score, confusion_matrix
from sklearn.utils.validation import check_is_fitted

# =========================
# Page setup
# =========================
st.set_page_config(page_title="Garmin HRV Status Forecast (28-day)", layout="wide")
st.title("🩺 Garmin HRV Status Forecast — Small-Data Edition")
st.caption("Designed for Garmin's 28-day HRV Status export (Date, Overnight HRV, Baseline, 7d Avg).")

# =========================
# Sidebar controls
# =========================
st.sidebar.header("⚙️ Settings")
uploaded = st.sidebar.file_uploader("📂 Upload `HRV Status Garmin.csv`", type=["csv"])

dev_threshold = st.sidebar.slider("Deviation threshold (ms): Low/High vs Stable",
                                  min_value=3, max_value=20, value=5, step=1)
roll_days = st.sidebar.slider("Rolling window (days) for trend features",
                              min_value=3, max_value=14, value=7, step=1)
forecast_horizon = st.sidebar.selectbox("Forecast horizon (days ahead)", [1, 2, 3], index=0)

st.sidebar.markdown("---")
st.sidebar.info("Garmin export contains **daily (overnight)** HRV values (max ~28 rows). "
                "This app forecasts **next-day status** (and recursively up to 3 days).")

# =========================
# Helpers
# =========================
def load_sample():
    dates = pd.date_range("2025-09-01", periods=28, freq="D")
    rng = np.random.default_rng(7)
    baseline = 55 + rng.normal(0, 0.6, size=len(dates)).cumsum() / 12 + 45
    overnight = baseline + rng.normal(0, 6, size=len(dates))
    avg7 = pd.Series(overnight).rolling(7).mean().fillna(method="bfill")
    df = pd.DataFrame({
        "Date": dates,
        "Overnight HRV": np.round(overnight, 1),
        "Baseline": np.round(baseline, 1),
        "7d Avg": np.round(avg7, 1)
    })
    return df

def normalize_columns(df_in: pd.DataFrame) -> pd.DataFrame:
    # lower, strip, underscores
    df = df_in.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    # Map common variants to internal names
    rename_map = {
        "date": "date",
        "overnight_hrv": "overnight_hrv",
        "overnight": "overnight_hrv",
        "baseline": "baseline",
        "7d_avg": "seven_day_average",
        "7_day_average": "seven_day_average",
        "7-day_average": "seven_day_average",
        "7-day_avg": "seven_day_average",
        "seven_day_average": "seven_day_average",
    }
    norm = {}
    for c in df.columns:
        key = rename_map.get(c, None)
        if key:
            norm[key] = df[c]
    return pd.DataFrame(norm)

def status_from_dev(d: float, th: float) -> int:
    if d > th:
        return 1   # High
    if d < -th:
        return -1  # Low
    return 0       # Stable

def status_label(x: int) -> str:
    return { -1: "Low", 0: "Stable", 1: "High" }.get(int(x), "Unknown")

def status_color(x: int) -> str:
    return { -1: "red", 0: "gold", 1: "green" }.get(int(x), "gray")

# =========================
# Load data
# =========================
if uploaded is not None:
    raw = pd.read_csv(uploaded)
    st.success("✅ File uploaded.")
else:
    st.info("Using built-in 28-day sample (upload your real `HRV Status Garmin.csv` for best results).")
    raw = load_sample()

norm = normalize_columns(raw)

required = ["date", "overnight_hrv", "baseline", "seven_day_average"]
missing = [c for c in required if c not in norm.columns]
if missing:
    st.error(f"Missing required column(s): {missing}. Found columns: {list(raw.columns)}")
    st.stop()

df = norm.copy()
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# Guardrail: Garmin export size
if len(df) > 28:
    st.warning(f"File has {len(df)} rows; Garmin HRV export is typically <= 28 days. Proceeding with all rows.")
elif len(df) < 10:
    st.warning(f"Only {len(df)} rows detected — forecasts may be unstable. More days improve reliability.")

# =========================
# Feature engineering
# =========================
df["hrv_deviation"]    = df["overnight_hrv"] - df["baseline"]
df["weekly_deviation"] = df["overnight_hrv"] - df["seven_day_average"]
df["hrv_change_rate"]  = df["overnight_hrv"].pct_change()

# Rolling features (keep short to fit 28 rows)
df["overnight_roll_mean"] = df["overnight_hrv"].rolling(roll_days).mean()
df["overnight_roll_std"]  = df["overnight_hrv"].rolling(roll_days).std()
df["dev_roll_mean"]       = df["hrv_deviation"].rolling(roll_days).mean()
df["dev_roll_std"]        = df["hrv_deviation"].rolling(roll_days).std()

# Fill minimal NaNs to preserve rows
df = df.ffill().bfill()

# Current (today) status for display
df["status_today"] = df["hrv_deviation"].apply(lambda d: status_from_dev(d, dev_threshold))

# Future label = tomorrow's status
df["future_status"] = df["status_today"].shift(-1)
df_model = df.dropna(subset=["future_status"]).copy()
df_model["future_status"] = df_model["future_status"].astype(int)

feature_cols = [
    "overnight_hrv", "baseline", "seven_day_average",
    "hrv_deviation", "weekly_deviation", "hrv_change_rate",
    "overnight_roll_mean", "overnight_roll_std",
    "dev_roll_mean", "dev_roll_std"
]
X_all = df_model[feature_cols].replace([np.inf, -np.inf], np.nan).ffill().bfill()
y_all = df_model["future_status"]

# =========================
# Data preview & class balance
# =========================
st.markdown("### 📋 Data Preview")
st.dataframe(df.tail(10), use_container_width=True, hide_index=True)

st.markdown("### ⚖️ Class Balance at Current Threshold")
cls_counts = y_all.value_counts().sort_index()
cls_view = pd.DataFrame({
    "HRV Status": [status_label(i) for i in cls_counts.index.tolist()],
    "Count": cls_counts.values
})
st.dataframe(cls_view, use_container_width=True, hide_index=True)

# =========================
# Modeling for tiny datasets
# =========================
if y_all.nunique() < 2:
    st.error("❌ Only one class present at this threshold — cannot train a classifier. "
             "Lower the deviation threshold or use a longer date range.")
    st.stop()

model = DecisionTreeClassifier(max_depth=3, random_state=42)

# Prefer K-Fold CV for small N
k = min(5, max(2, len(X_all)//5))  # aim for at least 2 folds
cv = KFold(n_splits=k, shuffle=True, random_state=42)
cv_acc = cross_val_score(model, X_all, y_all, cv=cv, scoring="accuracy")
st.markdown("### 🧪 Cross-Validated Performance")
c1, c2 = st.columns(2)
c1.metric("CV Accuracy (mean)", f"{cv_acc.mean():.2f}")
c2.metric("CV Accuracy (std)", f"{cv_acc.std():.2f}")

# Fit once for explanations/importance & to enable forecast
# (Use a light split or full fit if splitting fails)
try:
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y_all, test_size=0.2 if len(X_all) > 12 else 0.1,
        random_state=42, stratify=y_all if y_all.nunique() > 1 else None
    )
except ValueError:
    X_tr, X_te, y_tr, y_te = X_all, X_all, y_all, y_all

model.fit(X_tr, y_tr)
y_pred = model.predict(X_te)
acc = accuracy_score(y_te, y_pred)
macro_rec = recall_score(y_te, y_pred, average="macro")

with st.expander("Detailed classification report"):
    st.text(classification_report(y_te, y_pred, target_names=["Low (-1)","Stable (0)","High (1)"], digits=3))

m1, m2 = st.columns(2)
m1.metric("Holdout Accuracy", f"{acc:.2f}")
m2.metric("Holdout Macro Recall", f"{macro_rec:.2f}")

# =========================
# Feature importance
# =========================
st.markdown("### 📊 Feature Importance (Decision Tree)")
try:
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    labels = [feature_cols[i] for i in order]
    fig_imp, ax_imp = plt.subplots(figsize=(8,4))
    ax_imp.bar(range(len(labels)), importances[order], color="royalblue")
    ax_imp.set_xticks(range(len(labels)))
    ax_imp.set_xticklabels(labels, rotation=45, ha="right")
    ax_imp.set_ylabel("Importance")
    ax_imp.set_title("Which inputs most influence tomorrow's status?")
    st.pyplot(fig_imp)
except Exception:
    st.info("Feature importances unavailable.")

# =========================
# Trend plot
# =========================
st.markdown("### 📈 Overnight HRV vs Baseline & 7d Avg")
fig_trend, ax_t = plt.subplots(figsize=(10,4))
ax_t.plot(df["date"], df["overnight_hrv"], label="Overnight HRV", linewidth=2)
ax_t.plot(df["date"], df["baseline"], label="Baseline", linewidth=1.8)
ax_t.plot(df["date"], df["seven_day_average"], label="7d Avg", linewidth=1.8)
ax_t.set_ylabel("HRV (ms)")
ax_t.set_xlabel("Date")
ax_t.legend(loc="upper left")
ax_t.set_title("Overnight HRV vs Baseline vs 7d Avg")
ax_t.xaxis.set_major_locator(mdates.AutoDateLocator())
ax_t.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
fig_trend.autofmt_xdate(rotation=30)
st.pyplot(fig_trend)

# =========================
# Next 1–3 day forecast (recursive)
# =========================
st.markdown("### 🔮 Forecast (next 1–3 day status)")

def predict_status_row(row_series: pd.Series) -> tuple[int, float]:
    X_row = row_series[feature_cols].to_frame().T.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    probs = model.predict_proba(X_row)[0] if hasattr(model, "predict_proba") else None
    pred = model.predict(X_row)[0]
    conf = float(np.max(probs)) if probs is not None else 1.0
    return int(pred), conf

df_fore = df.copy()
future_rows = []
cur = df_fore.iloc[-1:].copy()

for step in range(1, forecast_horizon + 1):
    # Compute features for the last row (ensure all engineered cols exist)
    last = cur.iloc[-1].copy()

    # Predict next day status
    pred_status, conf = predict_status_row(last)

    # Synthesize a plausible next-day overnight HRV to roll features forward
    base = float(last["baseline"])
    avg7 = float(last["seven_day_average"])
    if pred_status == 1:
        overnight_next = base + dev_threshold + np.random.normal(1.0, 0.4)
    elif pred_status == -1:
        overnight_next = base - dev_threshold + np.random.normal(-1.0, 0.4)
    else:
        overnight_next = avg7 + np.random.normal(0.0, 0.4)

    next_date = last["date"] + pd.Timedelta(days=1)
    next_baseline = base + np.random.normal(0.0, 0.1)

    # Update rolling 7d avg from concatenated history
    hist = pd.concat([df_fore[["date","overnight_hrv"]],
                      pd.DataFrame({"date":[next_date], "overnight_hrv":[overnight_next]})],
                      ignore_index=True).sort_values("date")
    seven_day_next = hist["overnight_hrv"].rolling(7).mean().iloc[-1]
    if np.isnan(seven_day_next):
        seven_day_next = hist["overnight_hrv"].tail(3).mean()

    row = {
        "date": next_date,
        "overnight_hrv": float(overnight_next),
        "baseline": float(next_baseline),
        "seven_day_average": float(seven_day_next),
    }
    row["hrv_deviation"]    = row["overnight_hrv"] - row["baseline"]
    row["weekly_deviation"] = row["overnight_hrv"] - row["seven_day_average"]
    prev = float(last["overnight_hrv"])
    row["hrv_change_rate"]  = (row["overnight_hrv"] - prev) / max(prev, 1e-6)

    tmp = pd.concat([df_fore, pd.DataFrame([row])], ignore_index=True)
    tmp["overnight_roll_mean"] = tmp["overnight_hrv"].rolling(roll_days).mean()
    tmp["overnight_roll_std"]  = tmp["overnight_hrv"].rolling(roll_days).std()
    tmp["dev_roll_mean"]       = tmp["hrv_deviation"].rolling(roll_days).mean()
    tmp["dev_roll_std"]        = tmp["hrv_deviation"].rolling(roll_days).std()
    tail = tmp.iloc[-1][["overnight_roll_mean","overnight_roll_std","dev_roll_mean","dev_roll_std"]].to_dict()
    row.update(tail)

    row["pred_status"] = pred_status
    row["confidence"]  = conf

    future_rows.append(row)
    cur = pd.concat([cur, pd.DataFrame([row])], ignore_index=True)
    df_fore = pd.concat([df_fore, pd.DataFrame([row])], ignore_index=True)

if future_rows:
    fut = pd.DataFrame(future_rows)
    fut["status_label"] = fut["pred_status"].apply(status_label)
    fut["conf_pct"] = (fut["confidence"] * 100).round(1)
    st.dataframe(
        fut[["date","overnight_hrv","baseline","seven_day_average","status_label","conf_pct"]],
        use_container_width=True, hide_index=True
    )

    fig_fc, ax1 = plt.subplots(figsize=(10,4))
    ax1.plot(fut["date"], fut["confidence"], label="Confidence (0–1)", linewidth=2)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Confidence")
    ax1.set_xlabel("Date")
    ax1.grid(True, alpha=0.2)
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    ax2 = ax1.twinx()
    ax2.scatter(fut["date"], fut["pred_status"],
                c=[status_color(x) for x in fut["pred_status"]], s=60, label="Predicted Status")
    ax2.set_ylabel("Status  (-1=Low, 0=Stable, 1=High)")
    ax1.set_title(f"Predicted HRV Status — Next {forecast_horizon} day(s)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    st.pyplot(fig_fc)

# =========================
# Download your actual Garmin-style file
# =========================
st.markdown("---")
st.caption("Download your current (processed) Garmin HRV file with Garmin-style headers.")

actual = df[["date","overnight_hrv","baseline","seven_day_average"]].copy()
actual = actual.rename(columns={
    "date": "Date",
    "overnight_hrv": "Overnight HRV",
    "baseline": "Baseline",
    "seven_day_average": "7d Avg"   # match real Garmin naming
})
st.download_button(
    "⬇️ Download: HRV Status Garmin.csv",
    data=actual.to_csv(index=False).encode("utf-8"),
    file_name="HRV Status Garmin.csv",
    mime="text/csv"
)
