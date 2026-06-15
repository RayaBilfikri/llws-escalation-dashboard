from pathlib import Path
import pandas as pd
import shutil

SOURCE_DIR = Path(r"C:\LLWS_XGBOOST_POC")
PUBLIC_DIR = Path(r"C:\LLWS_DASHBOARD_PUBLIC")

PUBLIC_DIR.mkdir(parents=True, exist_ok=True)

# Load final evaluation checkpoint from original project
df_eval = pd.read_parquet(SOURCE_DIR / "df_eval_d16_final.parquet")

# Prepare time column
possible_time_cols = ["date and time", "time", "datetime", "timestamp"]
time_col = None

for col in possible_time_cols:
    if col in df_eval.columns:
        time_col = col
        break

if time_col is not None:
    df_eval[time_col] = pd.to_datetime(df_eval[time_col])
    df_eval["monitor_time"] = df_eval[time_col]
elif isinstance(df_eval.index, pd.DatetimeIndex):
    df_eval = df_eval.reset_index()
    df_eval["monitor_time"] = df_eval.iloc[:, 0]
else:
    raise ValueError("Kolom waktu tidak ditemukan. Cek struktur df_eval.")

# Demo window for public dashboard
start_time = pd.Timestamp("2025-10-19 02:20:00")
end_time = pd.Timestamp("2025-10-19 03:10:00")

df_demo = df_eval[
    (df_eval["monitor_time"] >= start_time) &
    (df_eval["monitor_time"] <= end_time)
].copy()

# Keep only columns needed by the public dashboard
keep_cols = ["monitor_time", "prob", "prob_smooth", "actual", "alert_hold"]
df_demo = df_demo[keep_cols].copy()

# Save demo data
df_demo.to_parquet(PUBLIC_DIR / "df_eval_demo.parquet", index=False)

# Copy safe config files
for filename in [
    "alert_config_final.pkl",
    "feature_columns_final.pkl",
    "model_metadata_final.pkl",
]:
    src = SOURCE_DIR / filename
    dst = PUBLIC_DIR / filename

    if src.exists():
        shutil.copy2(src, dst)
        print(f"Copied: {filename}")
    else:
        print(f"Skipped, not found: {filename}")

print("\nDemo data created successfully.")
print("Saved to:", PUBLIC_DIR / "df_eval_demo.parquet")
print("Shape:", df_demo.shape)
print(df_demo.head())