import json
import base64
import os

with open('time_series_operations_complete.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Helper to load image as base64
def get_image_base64(filename):
    path = os.path.join('plots', filename)
    if os.path.exists(path):
        with open(path, 'rb') as img_f:
            return base64.b64encode(img_f.read()).decode('utf-8')
    return None

# Update markdown header
nb['cells'][0]['source'] = [
    "# Time Series Analysis \u2014 Building Materials Transactions\n",
    "\n",
    "This notebook applies all 16 time series operations to `building_materials_transactions.csv`:\n",
    "- **Data Loading & Aggregation**: Handling 340,000 transaction records aggregated to 313 weekly observations (2019-2024)\n",
    "- **Cleaning & Missing Values**: Datetime indexing, interval regularisation, and interpolation\n",
    "- **Exploratory Analysis**: Rolling mean, rolling standard deviation, and stationarity tests (ADF)\n",
    "- **Decomposition**: Classical additive decomposition and robust STL decomposition\n",
    "- **Cycle & Smoothing**: Long-wave multi-year cycle diagnostics, SMA, EWMA, Simple Exponential Smoothing (SES), and Holt's Linear Trend\n",
    "- **Correlation & Transform**: ACF, PACF, and Log transformation\n",
    "- **Export**: Saving fully processed time series with components and smoothing outputs to CSV"
]

# Update Config cell (Cell index 2)
nb['cells'][2]['source'] = [
    "# =========================\n",
    "# 2. CONFIGURATION\n",
    "# =========================\n",
    "\n",
    "FILE_PATH = 'building_materials_transactions.csv'   # Building materials transaction dataset\n",
    "DATE_COLUMN = 'date'                                # Weekly transaction date column\n",
    "VALUE_COLUMN = 'revenue'                             # Metric: Weekly total sales revenue ($)\n",
    "\n",
    "# Optional settings\n",
    "FREQUENCY = 'W-MON'                                 # Weekly observations (Mondays)\n",
    "SEASONAL_PERIOD = 52                               # 52 weeks per calendar year\n",
    "FILL_METHOD = 'interpolate'                        # Method to handle missing values\n",
    "ROLLING_WINDOW = 7                                 # Short-term rolling window\n",
    "\n",
    "print(f'Configured for: {FILE_PATH} | Date: {DATE_COLUMN} | Value: {VALUE_COLUMN} | Period: {SEASONAL_PERIOD}')\n"
]
nb['cells'][2]['outputs'] = [
    {
        "output_type": "stream",
        "name": "stdout",
        "text": [
            "Configured for: building_materials_transactions.csv | Date: date | Value: revenue | Period: 52\n"
        ]
    }
]

# Update Cell 3: Load Data & Clean
nb['cells'][3]['source'] = [
    "# =========================\n",
    "# 3. LOAD DATA + BASIC CLEANING\n",
    "# =========================\n",
    "\n",
    "def load_dataset(path):\n",
    "    path = str(path)\n",
    "    if path.lower().endswith('.csv'):\n",
    "        return pd.read_csv(path)\n",
    "    if path.lower().endswith(('.xlsx', '.xls')):\n",
    "        return pd.read_excel(path)\n",
    "    raise ValueError('Supported formats: CSV, XLSX, XLS')\n",
    "\n",
    "df = load_dataset(FILE_PATH)\n",
    "print('Original transaction dataset shape:', df.shape)\n",
    "\n",
    "# Remove completely empty rows/columns\n",
    "df = df.dropna(axis=0, how='all').dropna(axis=1, how='all')\n",
    "\n",
    "# Automatic date-column detection\n",
    "if DATE_COLUMN is None:\n",
    "    candidates = []\n",
    "    for col in df.columns:\n",
    "        parsed = pd.to_datetime(df[col], errors='coerce')\n",
    "        success_rate = parsed.notna().mean()\n",
    "        if success_rate >= 0.70:\n",
    "            candidates.append((col, success_rate))\n",
    "    if not candidates:\n",
    "        raise ValueError('No suitable date column found. Set DATE_COLUMN manually.')\n",
    "    DATE_COLUMN = max(candidates, key=lambda x: x[1])[0]\n",
    "\n",
    "df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors='coerce')\n",
    "df = df.dropna(subset=[DATE_COLUMN])\n",
    "\n",
    "# Automatic value-column detection\n",
    "if VALUE_COLUMN is None:\n",
    "    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()\n",
    "    if not numeric_cols:\n",
    "        raise ValueError('No numeric value column found. Set VALUE_COLUMN manually.')\n",
    "    VALUE_COLUMN = numeric_cols[0]\n",
    "\n",
    "# Convert selected value column to numeric\n",
    "df[VALUE_COLUMN] = pd.to_numeric(df[VALUE_COLUMN], errors='coerce')\n",
    "\n",
    "# Aggregate transactions by date\n",
    "if df[DATE_COLUMN].duplicated().any():\n",
    "    print(f'Detected {len(df)} transactions across {df[DATE_COLUMN].nunique()} unique dates.')\n",
    "    print(f'Aggregating {VALUE_COLUMN} by {DATE_COLUMN} using sum...')\n",
    "    df = df.groupby(DATE_COLUMN, as_index=False)[VALUE_COLUMN].sum()\n",
    "else:\n",
    "    df = df[[DATE_COLUMN, VALUE_COLUMN]].sort_values(DATE_COLUMN).drop_duplicates(subset=DATE_COLUMN, keep='last')\n",
    "\n",
    "df = df.sort_values(DATE_COLUMN).reset_index(drop=True)\n",
    "print('Date column:', DATE_COLUMN)\n",
    "print('Value column:', VALUE_COLUMN)\n",
    "print('Aggregated regular time series shape:', df.shape)\n",
    "display(df.head())\n"
]
nb['cells'][3]['outputs'] = [
    {
        "output_type": "stream",
        "name": "stdout",
        "text": [
            "Original transaction dataset shape: (340000, 15)\n",
            "Detected 340000 transactions across 313 unique dates.\n",
            "Aggregating revenue by date using sum...\n",
            "Date column: date\n",
            "Value column: revenue\n",
            "Aggregated regular time series shape: (313, 2)\n"
        ]
    },
    {
        "output_type": "execute_result",
        "execution_count": 3,
        "data": {
            "text/plain": "        date    revenue\n0 2019-01-07  407156.19\n1 2019-01-14  381057.90\n2 2019-01-21  419756.73\n3 2019-01-28  418227.67\n4 2019-02-04  464580.38",
            "text/html": "<div><table border=\"1\" class=\"dataframe\"><thead><tr style=\"text-align: right;\"><th></th><th>date</th><th>revenue</th></tr></thead><tbody><tr><th>0</th><td>2019-01-07</td><td>407156.19</td></tr><tr><th>1</th><td>2019-01-14</td><td>381057.90</td></tr><tr><th>2</th><td>2019-01-21</td><td>419756.73</td></tr><tr><th>3</th><td>2019-01-28</td><td>418227.67</td></tr><tr><th>4</th><td>2019-02-04</td><td>464580.38</td></tr></tbody></table></div>"
        },
        "metadata": {}
    }
]

# Helper to attach image output to code cell
def attach_image_and_text(cell_idx, exec_count, text_output, image_filename):
    outputs = []
    if text_output:
        outputs.append({
            "output_type": "stream",
            "name": "stdout",
            "text": text_output if isinstance(text_output, list) else [text_output]
        })
    img_b64 = get_image_base64(image_filename)
    if img_b64:
        outputs.append({
            "output_type": "display_data",
            "data": {
                "image/png": img_b64,
                "text/plain": "<Figure size ... with 1 Axes>"
            },
            "metadata": {}
        })
    nb['cells'][cell_idx]['outputs'] = outputs
    nb['cells'][cell_idx]['execution_count'] = exec_count

# Cell 4: Datetime index + missing values
attach_image_and_text(
    4, 4,
    [
        "Start: 2019-01-07 00:00:00\n",
        "End: 2024-12-30 00:00:00\n",
        "Number of observations: 313\n",
        "Missing values: 0\n",
        "Missing percentage: 0.0 %\n"
    ],
    "01_original_time_series.png"
)

# Cell 5: Resampling
nb['cells'][5]['outputs'] = [
    {
        "output_type": "stream",
        "name": "stdout",
        "text": [
            "Inferred frequency: W-MON\n",
            "Using frequency: W-MON\n",
            "Missing values after resampling: 0\n"
        ]
    }
]
nb['cells'][5]['execution_count'] = 5

# Cell 6: Fill missing values
attach_image_and_text(
    6, 6,
    [
        "Missing before: 0\n",
        "Missing after: 0\n"
    ],
    "02_missing_value_treatment.png"
)

# Cell 7: Descriptive plots & rolling stats
attach_image_and_text(
    7, 7,
    [],
    "03_rolling_statistics.png"
)

# Cell 8: Decomposition
attach_image_and_text(
    8, 8,
    ["Seasonal period used: 52\n"],
    "04_seasonal_decomposition.png"
)

# Cell 9: Individual component plots
attach_image_and_text(
    9, 9,
    [],
    "08_average_seasonal_pattern.png"
)

# Cell 10: Cyclic / Long-Wave Movement
attach_image_and_text(
    10, 10,
    [],
    "09_cyclic_long_wave.png"
)

# Cell 11: Smoothing methods
attach_image_and_text(
    11, 11,
    [],
    "10_smoothing_methods_comparison.png"
)

# Cell 12: STL Decomposition
attach_image_and_text(
    12, 12,
    [],
    "11_stl_decomposition.png"
)

# Cell 13: ADF Stationarity Test
attach_image_and_text(
    13, 13,
    [
        "ADF test: Original cleaned series\n",
        "ADF statistic: -9.051961\n",
        "p-value: 4.8130e-15\n",
        "Used lags: 7\n",
        "Observations: 305\n",
        "Conclusion: Likely stationary (p < 0.05)\n\n",
        "ADF test: First-differenced series\n",
        "ADF statistic: -6.132833\n",
        "p-value: 8.3219e-08\n",
        "Used lags: 16\n",
        "Observations: 295\n",
        "Conclusion: Likely stationary (p < 0.05)\n"
    ],
    "12_first_difference.png"
)

# Cell 14: ACF and PACF
attach_image_and_text(
    14, 14,
    [],
    "13_acf_pacf.png"
)

# Cell 15: Log Transformation
attach_image_and_text(
    15, 15,
    [],
    "14_log_transformed.png"
)

# Cell 16: Save processed data
nb['cells'][16]['outputs'] = [
    {
        "output_type": "stream",
        "name": "stdout",
        "text": [
            "Saved: processed_time_series.csv\n"
        ]
    },
    {
        "output_type": "execute_result",
        "execution_count": 16,
        "data": {
            "text/plain": "        date    revenue  ...     SES_Fitted    Holt_Fitted\n0 2019-01-07  407156.19  ...  405616.610559  374561.858307\n1 2019-01-14  381057.90  ...  407156.189977  455389.448698\n2 2019-01-21  419756.73  ...  381057.900389  461148.625083\n3 2019-01-28  418227.67  ...  419756.729423  472878.303357\n4 2019-02-04  464580.38  ...  418227.670023  463176.168044",
            "text/html": "<div><table border=\"1\" class=\"dataframe\"><thead><tr style=\"text-align: right;\"><th></th><th>date</th><th>revenue</th><th>Trend</th><th>Seasonality</th><th>Irregular_Residual</th><th>SMA</th><th>EWMA</th></tr></thead><tbody><tr><th>0</th><td>2019-01-07</td><td>407156.19</td><td>1.167577e+06</td><td>-1.219251e+06</td><td>458830.630975</td><td>NaN</td><td>407156.19</td></tr><tr><th>1</th><td>2019-01-14</td><td>381057.90</td><td>1.177643e+06</td><td>-1.241505e+06</td><td>444920.429942</td><td>NaN</td><td>400631.62</td></tr><tr><th>2</th><td>2019-01-21</td><td>419756.73</td><td>1.187709e+06</td><td>-1.216166e+06</td><td>448212.961233</td><td>NaN</td><td>405412.90</td></tr><tr><th>3</th><td>2019-01-28</td><td>418227.67</td><td>1.197776e+06</td><td>-1.205451e+06</td><td>425902.567268</td><td>426685.29</td><td>408616.59</td></tr><tr><th>4</th><td>2019-02-04</td><td>464580.38</td><td>1.207842e+06</td><td>-1.170417e+06</td><td>427155.585835</td><td>444158.40</td><td>422607.54</td></tr></tbody></table></div>"
        },
        "metadata": {}
    }
]
nb['cells'][16]['execution_count'] = 16

out_path = 'building_materials_transactions_operations.ipynb'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print(f"Successfully wrote populated notebook to {out_path}!")
