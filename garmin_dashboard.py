import streamlit as st
import pandas as pd
import numpy as np
import base64
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, accuracy_score

# Set Up Page
st.set_page_config(page_title="Stress Forecast Dashboard", layout="wide")
st.title("Stress Forecast Dashboard")

st.markdown("""
This dashboard analyzes **heart rate variability (HRV)** data to predict future stress levels 
using Random Forest machine learning. You can upload your own Garmin HRV CSV file or explore 
the built-in simulated dataset.
""")
with open("simulated_rr_stress_30s_large.csv", "rb") as file:
    csv_data = file.read()

# Add download button
st.download_button(
    label="⬇️ Download Sample HRV CSV Data",
    data=csv_data,
    file_name="simulated_rr_stress_30s_large.csv",
    mime="text/csv",
)

# --- SIDEBAR SETTINGS ---
st.sidebar.header(" Configuration")
uploaded_file = st.sidebar.file_uploader("📂 Upload Garmin HRV CSV", type=["csv"])
window_size = st.sidebar.slider("Rolling Window Size", 30, 200)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.sidebar.success("File uploaded successfully!")
else:
    st.sidebar.info("Using built-in simulated dataset")
    data = pd.read_csv("simulated_rr_stress_5min.csv")

# --- DATA PREVIEW SECTION ---
st.markdown("### Dataset Preview")
st.write("This dataset represents HRV readings recorded in 5-minute intervals with corresponding stress level classifications.")
st.dataframe(
    data.head(10),
    use_container_width=True,
    hide_index=True
)

with st.expander("Show Summary Statistics"):
    st.write(data.describe(include='all'))

# --- MINI BAR CHART ---
st.markdown("#### Stress Level Distribution")
st.bar_chart(data['stress_level'].value_counts(), use_container_width=True)

# --- FEATURE EXTRACTION ---
features = []
for level, group in data.groupby("stress_level"):
    rr = group['rr_interval'].to_numpy()
    avnn = np.mean(rr)
    sdnn = np.std(rr, ddof=1)
    rmssd = np.sqrt(np.mean(np.diff(rr)**2))
    for val in rr:
        features.append([avnn, sdnn, rmssd, level])

features_df = pd.DataFrame(features, columns=["AVNN", "SDNN", "RMSSD", "stress_level"])
features_df['time'] = data['time'].values[:len(features_df)]
mapping = {"low": 0, "medium": 1, "high": 2}
features_df["stress_level"] = features_df["stress_level"].map(mapping)

# --- MODEL TRAINING ---
X = features_df[["AVNN", "SDNN", "RMSSD"]]
y = features_df["stress_level"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

recall = recall_score(y_test, predictions, average='macro')
accuracy = accuracy_score(y_test, predictions)

# --- PERFORMANCE SUMMARY ---
st.markdown("### Model Performance Summary")
col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", f"{accuracy:.2f}")
col2.metric("Macro Recall", f"{recall:.2f}")
col3.metric("Training Samples", len(X_train))

# --- FEATURE IMPORTANCE PLOT ---
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
labels = [X.columns[i] for i in indices]

with st.container():
    st.subheader(" Feature Importance Visualization")
    st.write("Shows which HRV metrics most influence stress prediction.")
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(range(len(labels)), importances[indices], align='center', color="royalblue")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Importance")
    ax.set_title("Feature Importances for HRV Stress Prediction")
    st.pyplot(fig)

# --- FORECASTING SECTION ---
st.markdown("### 2-Hour Stress Forecast")

data = features_df.copy()
data['rmssd_roll3'] = data['RMSSD'].rolling(window=window_size).mean()
data['sdnn_roll3'] = data['SDNN'].rolling(window=window_size).mean()
data['avnn_roll3'] = data['AVNN'].rolling(window=window_size).mean()
data = data.dropna().reset_index(drop=True)
data['future_stress'] = data['stress_level'].shift(-24)
data = data.dropna().reset_index(drop=True)

forecast_features = ['rmssd_roll3', 'sdnn_roll3', 'avnn_roll3']
X_future = data[forecast_features]
y_future = data['future_stress']

X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_future, y_future, test_size=0.3, random_state=42)
forecast_model = RandomForestClassifier(n_estimators=200, random_state=42)
forecast_model.fit(X_train_f, y_train_f)

preds_future = forecast_model.predict(X_test_f)
probs_future = forecast_model.predict_proba(X_test_f)
confidence = probs_future.max(axis=1)

pred_df = pd.DataFrame({
    'timestamp': data['time'].iloc[X_test_f.index],
    'predicted_future_stress': preds_future,
    'confidence': confidence
}).sort_values('timestamp')

# --- SMOOTHING + CONFIDENCE GATING ---
pred_df['smoothed_stress'] = (
    pd.Series(pred_df['predicted_future_stress'])
    .rolling(window=5, center=True)
    .apply(lambda x: round(x.mean()), raw=False)
    .fillna(method='bfill')
    .fillna(method='ffill')
)

# Clean up NaN edges and drop incomplete points
pred_df = pred_df.dropna(subset=['smoothed_stress', 'confidence']).reset_index(drop=True)

pred_df.loc[pred_df['confidence'] < 0.6, 'smoothed_stress'] = np.nan
pred_df['smoothed_stress'] = pred_df['smoothed_stress'].ffill()

# --- MAIN FORECAST CHART ---
data['time'] = pd.to_datetime(data['time'])
pred_df['timestamp'] = pd.to_datetime(pred_df['timestamp'])

fig, ax1 = plt.subplots(figsize=(10,5))
ax1.plot(pred_df['timestamp'], pred_df['confidence'], color='orange', linewidth=2, label='Model Confidence (0–1)')
ax1.set_ylabel('Confidence', color='orange')
ax1.tick_params(axis='y', labelcolor='orange')

ax2 = ax1.twinx()
ax2.plot(pred_df['timestamp'], pred_df['smoothed_stress'], color='royalblue', linewidth=2.5, label='Smoothed Predicted Stress')
ax2.set_ylabel('Stress Level (0=Low, 1=Med, 2=High)', color='royalblue')
ax2.tick_params(axis='y', labelcolor='royalblue')

ax1.set_xlabel('Time')
ax1.set_title('Smoothed 2-Hour Stress Forecast vs Confidence')

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper right')

# Format X-axis to show readable 5-minute intervals
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax1.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))  # show every 30 minutes
fig.autofmt_xdate(rotation=45)

st.pyplot(fig)

# --- COLOR-CODED STRESS SCATTER ---
st.markdown("### 2-Hour Predicted Stress Levels ")
colors = pred_df['smoothed_stress'].map({0: 'green', 1: 'gold', 2: 'red'})
fig2, ax = plt.subplots(figsize=(10,5))
ax.scatter(pred_df['timestamp'], pred_df['smoothed_stress'], c=colors, label='Stress Level', alpha=0.8)
ax.set_ylabel("Stress Level (0=Low, 1=Med, 2=High)")
ax.set_xlabel("Time")
ax.set_title("Predicted Stress Levels by Color")
st.pyplot(fig2)

st.subheader("HRV Trend Over Time")
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(features_df["time"], features_df["AVNN"], label="AVNN", color="royalblue")
ax.plot(features_df["time"], features_df["RMSSD"], label="RMSSD", color="orange")
ax.set_xlabel("Time")
ax.set_ylabel("HRV (ms)")
ax.legend()
ax.set_title("HRV Metrics Over Time")
st.pyplot(fig)


st.success("✅ Dashboard loaded successfully. Adjust the smoothing or upload new data to explore!")
