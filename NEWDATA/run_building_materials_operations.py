"""
Building Materials Transactions - Time Series Analysis Pipeline
Applies all 16 operations from time_series_operations_complete.ipynb
"""

import os
import json
import base64
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'plots')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("1 & 2. CONFIGURATION & IMPORTS")
print("=" * 70)

FILE_PATH = 'building_materials_transactions.csv'
DATE_COLUMN = 'date'
VALUE_COLUMN = 'revenue'
FREQUENCY = 'W-MON'
SEASONAL_PERIOD = 52
FILL_METHOD = 'interpolate'
ROLLING_WINDOW = 7

print(f"FILE_PATH: {FILE_PATH}")
print(f"DATE_COLUMN: {DATE_COLUMN}")
print(f"VALUE_COLUMN: {VALUE_COLUMN}")
print(f"FREQUENCY: {FREQUENCY}")
print(f"SEASONAL_PERIOD: {SEASONAL_PERIOD}")
print(f"FILL_METHOD: {FILL_METHOD}")
print(f"ROLLING_WINDOW: {ROLLING_WINDOW}")

print("\n" + "=" * 70)
print("3. LOAD DATA + BASIC CLEANING")
print("=" * 70)

df_raw = pd.read_csv(FILE_PATH)
print("Original shape:", df_raw.shape)
print("Original columns:", df_raw.columns.tolist())

# Drop completely empty rows and columns
df_clean = df_raw.dropna(axis=0, how='all').dropna(axis=1, how='all')

# Parse date column
df_clean[DATE_COLUMN] = pd.to_datetime(df_clean[DATE_COLUMN], errors='coerce')
df_clean = df_clean.dropna(subset=[DATE_COLUMN])

# Convert value column to numeric
df_clean[VALUE_COLUMN] = pd.to_numeric(df_clean[VALUE_COLUMN], errors='coerce')

# For transaction-level records, aggregate by date (sum total revenue across transactions)
if df_clean[DATE_COLUMN].duplicated().any():
    print(f"Dataset contains transaction logs ({len(df_clean)} records across {df_clean[DATE_COLUMN].nunique()} unique dates).")
    print(f"Aggregating '{VALUE_COLUMN}' by '{DATE_COLUMN}' to establish regular weekly time series...")
    df_agg = df_clean.groupby(DATE_COLUMN, as_index=False)[VALUE_COLUMN].sum()
else:
    df_agg = df_clean[[DATE_COLUMN, VALUE_COLUMN]].sort_values(DATE_COLUMN).drop_duplicates(subset=DATE_COLUMN, keep='last')

df = df_agg.sort_values(DATE_COLUMN).reset_index(drop=True)
print("Aggregated time series shape:", df.shape)
print("First 5 records:")
print(df.head())

print("\n" + "=" * 70)
print("4. SET DATETIME INDEX + CHECK MISSING VALUES")
print("=" * 70)

ts = df.set_index(DATE_COLUMN)[VALUE_COLUMN].sort_index().astype(float)
print("Start date:", ts.index.min())
print("End date:", ts.index.max())
print("Number of observations:", len(ts))
print("Missing values in raw index:", ts.isna().sum())
print("Missing percentage:", round(ts.isna().mean() * 100, 2), "%")

# Plot 1: Original Time Series
plt.figure(figsize=(14, 5))
plt.plot(ts.index, ts.values / 1e6, color='#1f77b4', linewidth=1.5)
plt.title('Original Time Series: Weekly Building Materials Revenue (Millions $)', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Revenue ($ Millions)')
plt.tight_layout()
p1_path = os.path.join(OUTPUT_DIR, '01_original_time_series.png')
plt.savefig(p1_path, dpi=200)
plt.close()
print(f"Saved plot: {p1_path}")

print("\n" + "=" * 70)
print("5. RESAMPLE TO A REGULAR TIME INTERVAL")
print("=" * 70)

inferred_freq = pd.infer_freq(ts.dropna().index) if len(ts.dropna()) >= 3 else None
freq = FREQUENCY or inferred_freq
print("Inferred frequency:", inferred_freq)
print("Applied frequency:", freq)

if freq is not None:
    ts = ts.resample(freq).mean()

print("Observations after regular resampling:", len(ts))
print("Missing values after resampling:", ts.isna().sum())

print("\n" + "=" * 70)
print("6. FILL MISSING VALUES")
print("=" * 70)

before_missing = ts.isna().sum()
if FILL_METHOD == 'interpolate':
    ts_clean = ts.interpolate(method='time').ffill().bfill()
elif FILL_METHOD == 'ffill':
    ts_clean = ts.ffill().bfill()
elif FILL_METHOD == 'bfill':
    ts_clean = ts.bfill().ffill()
elif FILL_METHOD == 'mean':
    ts_clean = ts.fillna(ts.mean())
else:
    raise ValueError("Invalid FILL_METHOD")

after_missing = ts_clean.isna().sum()
print("Missing before filling:", before_missing)
print("Missing after filling:", after_missing)

# Plot 2: Missing Value Treatment
plt.figure(figsize=(14, 5))
plt.plot(ts.index, ts / 1e6, label='Before filling', alpha=0.7, color='#7f7f7f', linestyle='--')
plt.plot(ts_clean.index, ts_clean / 1e6, label='After filling (clean series)', linewidth=1.5, color='#2ca02c')
plt.title('Missing-Value Treatment: Weekly Building Materials Revenue', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Revenue ($ Millions)')
plt.legend()
plt.tight_layout()
p2_path = os.path.join(OUTPUT_DIR, '02_missing_value_treatment.png')
plt.savefig(p2_path, dpi=200)
plt.close()
print(f"Saved plot: {p2_path}")

print("\n" + "=" * 70)
print("7. DESCRIPTIVE TIME-SERIES PLOTS & ROLLING STATISTICS")
print("=" * 70)

rolling_mean = ts_clean.rolling(ROLLING_WINDOW).mean()
rolling_std = ts_clean.rolling(ROLLING_WINDOW).std()

plt.figure(figsize=(15, 6))
plt.plot(ts_clean / 1e6, label='Original Cleaned', color='#1f77b4', alpha=0.6)
plt.plot(rolling_mean / 1e6, label=f'{ROLLING_WINDOW}-period Rolling Mean', color='#d62728', linewidth=2)
plt.plot(rolling_std / 1e6, label=f'{ROLLING_WINDOW}-period Rolling Std Dev', color='#ff7f0e', linestyle='--', linewidth=1.8)
plt.title(f'Descriptive Statistics: Weekly Revenue with {ROLLING_WINDOW}-Week Rolling Mean & Std Dev', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Revenue ($ Millions)')
plt.legend(loc='upper left')
plt.tight_layout()
p3_path = os.path.join(OUTPUT_DIR, '03_rolling_statistics.png')
plt.savefig(p3_path, dpi=200)
plt.close()
print(f"Saved plot: {p3_path}")

print("\n" + "=" * 70)
print("8. TIME-SERIES DECOMPOSITION: TREND, SEASONALITY, RESIDUAL")
print("=" * 70)

period = SEASONAL_PERIOD or 52
print("Seasonal period used:", period)

decomposition = seasonal_decompose(ts_clean, model='additive', period=period, extrapolate_trend='freq')

fig = decomposition.plot()
fig.set_size_inches(15, 10)
fig.axes[0].set_ylabel('Observed ($)')
fig.axes[1].set_ylabel('Trend ($)')
fig.axes[2].set_ylabel('Seasonal ($)')
fig.axes[3].set_ylabel('Residual ($)')
plt.suptitle('Classical Additive Decomposition (Period=52 Weeks)', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
p4_path = os.path.join(OUTPUT_DIR, '04_seasonal_decomposition.png')
plt.savefig(p4_path, dpi=200)
plt.close()
print(f"Saved plot: {p4_path}")

components_df = pd.DataFrame({
    'Observed': decomposition.observed,
    'Trend': decomposition.trend,
    'Seasonality': decomposition.seasonal,
    'Irregular_Residual': decomposition.resid
})
print("Sample of Decomposed Components:")
print(components_df.head())

print("\n" + "=" * 70)
print("9. INDIVIDUAL COMPONENT PLOTS & AVERAGE SEASONAL CYCLE")
print("=" * 70)

# 9A: Trend
plt.figure(figsize=(15, 4))
plt.plot(decomposition.trend / 1e6, color='#d62728', linewidth=2)
plt.title('Extracted Trend Component ($ Millions)', fontsize=13, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Trend ($ Millions)')
plt.tight_layout()
p5_path = os.path.join(OUTPUT_DIR, '05_trend_component.png')
plt.savefig(p5_path, dpi=200)
plt.close()

# 9B: Seasonality
plt.figure(figsize=(15, 4))
plt.plot(decomposition.seasonal / 1e6, color='#2ca02c', linewidth=1.5)
plt.title('Extracted 52-Week Seasonal Component ($ Millions)', fontsize=13, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Seasonal Effect ($ Millions)')
plt.tight_layout()
p6_path = os.path.join(OUTPUT_DIR, '06_seasonality_component.png')
plt.savefig(p6_path, dpi=200)
plt.close()

# 9C: Residual
plt.figure(figsize=(15, 4))
plt.plot(decomposition.resid / 1e6, color='#7f7f7f', linewidth=1.2)
plt.axhline(0, color='black', linestyle='--', alpha=0.5)
plt.title('Extracted Irregular / Residual Component ($ Millions)', fontsize=13, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Residual ($ Millions)')
plt.tight_layout()
p7_path = os.path.join(OUTPUT_DIR, '07_residual_component.png')
plt.savefig(p7_path, dpi=200)
plt.close()

# 9D: Average Seasonal Pattern across 52 weeks
seasonal_pattern = decomposition.seasonal.groupby(np.arange(len(decomposition.seasonal)) % period).mean()
plt.figure(figsize=(12, 4))
plt.plot(range(1, period + 1), seasonal_pattern.values / 1e6, marker='o', color='#9467bd', linewidth=1.8, markersize=4)
plt.title('Average Seasonal Profile Across 52-Week Annual Cycle', fontsize=13, fontweight='bold')
plt.xlabel('Week of Year (1 to 52)')
plt.ylabel('Average Seasonal Effect ($ Millions)')
plt.tight_layout()
p8_path = os.path.join(OUTPUT_DIR, '08_average_seasonal_pattern.png')
plt.savefig(p8_path, dpi=200)
plt.close()
print("Saved component plots 05, 06, 07, 08.")

print("\n" + "=" * 70)
print("10. CYCLIC / LONG-WAVE MOVEMENT")
print("=" * 70)

cycle_window = max(period * 2, ROLLING_WINDOW * 3) # 104 weeks (2 years)
cycle = ts_clean.rolling(cycle_window, center=True, min_periods=max(2, cycle_window // 2)).mean()

plt.figure(figsize=(15, 6))
plt.plot(ts_clean / 1e6, alpha=0.4, color='#1f77b4', label='Weekly Observed Revenue')
plt.plot(cycle / 1e6, linewidth=2.5, color='#e377c2', label=f'Long-Wave Rolling Mean ({cycle_window}-Week Multi-Year Cycle)')
plt.title(f'Cyclic / Long-Wave Movement Diagnostic ({cycle_window}-Week Window)', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Revenue ($ Millions)')
plt.legend()
plt.tight_layout()
p9_path = os.path.join(OUTPUT_DIR, '09_cyclic_long_wave.png')
plt.savefig(p9_path, dpi=200)
plt.close()
print(f"Saved plot: {p9_path}")

print("\n" + "=" * 70)
print("11. SMOOTHING METHODS")
print("=" * 70)

# A) Simple Moving Average
sma = ts_clean.rolling(window=ROLLING_WINDOW, center=True).mean()

# B) Exponentially Weighted Moving Average
ewma = ts_clean.ewm(span=ROLLING_WINDOW, adjust=False).mean()

# C) Simple Exponential Smoothing
ses_model = SimpleExpSmoothing(ts_clean, initialization_method='estimated').fit()
ses = ses_model.fittedvalues

# D) Holt's Linear Trend Smoothing
holt_model = Holt(ts_clean, initialization_method='estimated').fit(optimized=True)
holt = holt_model.fittedvalues

plt.figure(figsize=(15, 7))
plt.plot(ts_clean / 1e6, label='Original Observed', alpha=0.35, color='#7f7f7f')
plt.plot(sma / 1e6, label=f'Simple Moving Average (Window={ROLLING_WINDOW})', color='#1f77b4', linewidth=1.6)
plt.plot(ewma / 1e6, label=f'EWMA (Span={ROLLING_WINDOW})', color='#ff7f0e', linewidth=1.6)
plt.plot(ses / 1e6, label=f'Simple Exp Smoothing (alpha={ses_model.params.get("smoothing_level", 0):.3f})', color='#2ca02c', linewidth=1.6)
plt.plot(holt / 1e6, label=f"Holt's Linear Smoothing (alpha={holt_model.params.get('smoothing_level', 0):.3f}, beta={holt_model.params.get('smoothing_trend', 0):.3f})", color='#d62728', linewidth=1.6)
plt.title('Comparison of Time Series Smoothing Methods', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Revenue ($ Millions)')
plt.legend()
plt.tight_layout()
p10_path = os.path.join(OUTPUT_DIR, '10_smoothing_methods_comparison.png')
plt.savefig(p10_path, dpi=200)
plt.close()
print(f"Saved plot: {p10_path}")

# Calculate in-sample fit metrics
mae_ses = mean_absolute_error(ts_clean, ses)
rmse_ses = np.sqrt(mean_squared_error(ts_clean, ses))
mae_holt = mean_absolute_error(ts_clean, holt)
rmse_holt = np.sqrt(mean_squared_error(ts_clean, holt))
print(f"SES Model Fit  -> MAE: ${mae_ses:,.2f}, RMSE: ${rmse_ses:,.2f}")
print(f"Holt Model Fit -> MAE: ${mae_holt:,.2f}, RMSE: ${rmse_holt:,.2f}")

print("\n" + "=" * 70)
print("12. STL DECOMPOSITION - ROBUST OPTION")
print("=" * 70)

stl = STL(ts_clean, period=period, robust=True).fit()
fig = stl.plot()
fig.set_size_inches(15, 10)
fig.axes[0].set_ylabel('Observed ($)')
fig.axes[1].set_ylabel('Trend ($)')
fig.axes[2].set_ylabel('Seasonal ($)')
fig.axes[3].set_ylabel('Resid ($)')
plt.suptitle('Robust STL Decomposition (Period=52 Weeks)', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
p11_path = os.path.join(OUTPUT_DIR, '11_stl_decomposition.png')
plt.savefig(p11_path, dpi=200)
plt.close()
print(f"Saved plot: {p11_path}")

print("\n" + "=" * 70)
print("13. STATIONARITY TEST (ADF)")
print("=" * 70)

def run_adf_test(series, name='Series'):
    series = series.dropna()
    res = adfuller(series, autolag='AIC')
    print(f"--- ADF Test: {name} ---")
    print(f"ADF Statistic: {res[0]:.6f}")
    print(f"p-value: {res[1]:.4e}")
    print(f"Used lags: {res[2]}")
    print(f"Observations: {res[3]}")
    print("Critical Values:")
    for k, v in res[4].items():
        print(f"   {k}: {v:.4f}")
    verdict = "Likely stationary (p < 0.05)" if res[1] < 0.05 else "Likely non-stationary (p >= 0.05)"
    print(f"Conclusion: {verdict}\n")
    return res

adf_orig = run_adf_test(ts_clean, "Original Cleaned Series")

ts_diff = ts_clean.diff().dropna()
adf_diff = run_adf_test(ts_diff, "First-Differenced Series")

plt.figure(figsize=(15, 5))
plt.plot(ts_diff / 1e6, color='#8c564b', linewidth=1.3)
plt.axhline(0, color='black', linestyle='--', alpha=0.5)
plt.title('First Difference of Weekly Revenue: Stationarity Transformation', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Differenced Revenue ($ Millions)')
plt.tight_layout()
p12_path = os.path.join(OUTPUT_DIR, '12_first_difference.png')
plt.savefig(p12_path, dpi=200)
plt.close()
print(f"Saved plot: {p12_path}")

print("\n" + "=" * 70)
print("14. ACF AND PACF ANALYSIS")
print("=" * 70)

lags = min(40, max(1, len(ts_clean) // 3))
print(f"Computing ACF and PACF with {lags} lags...")

fig, axes = plt.subplots(2, 1, figsize=(14, 8))
plot_acf(ts_clean.dropna(), lags=lags, ax=axes[0], color='#1f77b4')
axes[0].set_title('Autocorrelation Function (ACF) - Weekly Revenue', fontsize=13, fontweight='bold')
plot_pacf(ts_clean.dropna(), lags=lags, ax=axes[1], method='ywm', color='#2ca02c')
axes[1].set_title('Partial Autocorrelation Function (PACF) - Weekly Revenue', fontsize=13, fontweight='bold')
plt.tight_layout()
p13_path = os.path.join(OUTPUT_DIR, '13_acf_pacf.png')
plt.savefig(p13_path, dpi=200)
plt.close()
print(f"Saved plot: {p13_path}")

print("\n" + "=" * 70)
print("15. LOG TRANSFORMATION")
print("=" * 70)

if (ts_clean > 0).all():
    log_ts = np.log(ts_clean)
    plt.figure(figsize=(15, 5))
    plt.plot(log_ts, color='#17becf', linewidth=1.5)
    plt.title('Log-Transformed Series: ln(Revenue)', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('ln(Revenue)')
    plt.tight_layout()
    p14_path = os.path.join(OUTPUT_DIR, '14_log_transformed.png')
    plt.savefig(p14_path, dpi=200)
    plt.close()
    print(f"Saved plot: {p14_path}")
else:
    print("Log transformation skipped because the series contains zero or negative values.")

print("\n" + "=" * 70)
print("16. SAVE THE PROCESSED DATA")
print("=" * 70)

output_df = pd.DataFrame({
    DATE_COLUMN: ts_clean.index,
    VALUE_COLUMN: ts_clean.values,
    'Decomp_Trend': decomposition.trend.values,
    'Decomp_Seasonality': decomposition.seasonal.values,
    'Decomp_Residual': decomposition.resid.values,
    'STL_Trend': stl.trend.values,
    'STL_Seasonality': stl.seasonal.values,
    'STL_Residual': stl.resid.values,
    'SMA_7': sma.values,
    'EWMA_7': ewma.values,
    'SES_Fitted': ses.values,
    'Holt_Fitted': holt.values
})

out_csv1 = 'processed_time_series.csv'
out_csv2 = 'building_materials_processed_time_series.csv'
output_df.to_csv(out_csv1, index=False)
output_df.to_csv(out_csv2, index=False)

print(f"Exported processed data to: {out_csv1} and {out_csv2}")
print(f"Columns exported: {output_df.columns.tolist()}")
print(f"Total rows: {len(output_df)}")
print("\nSample Preview:")
print(output_df.head())

print("\n" + "=" * 70)
print("ALL 16 OPERATIONS COMPLETED SUCCESSFULLY!")
print("=" * 70)
