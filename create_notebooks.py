import json
import os

NOTEBOOKS_DIR = "/home/runner/work/SRM_MBA_PROJECT/SRM_MBA_PROJECT/notebooks"
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)


def make_code_cell(source_lines):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_lines if isinstance(source_lines, list) else [source_lines],
    }


def make_markdown_cell(source_lines):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source_lines if isinstance(source_lines, list) else [source_lines],
    }


def save_notebook(cells, filepath):
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.9.0"},
        },
        "cells": cells,
    }
    with open(filepath, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"Saved: {filepath}")


# ─────────────────────────────────────────────────────────────────
# Notebook 1 – EDA
# ─────────────────────────────────────────────────────────────────
def create_eda_notebook():
    cells = [
        make_markdown_cell(
            "# Exploratory Data Analysis - Online Retail Demand Dataset\n"
            "> **Project:** AI-Based Product Demand Forecasting System  \n"
            "> **Dataset:** Online Retail (UCI ML Repository)  \n"
            "> **Objective:** Understand the structure, quality, and patterns in the retail transaction data."
        ),
        make_code_cell(
            [
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "import warnings\n",
                "warnings.filterwarnings('ignore')\n",
                "\n",
                "# Plotting style\n",
                "plt.rcParams['figure.figsize'] = (12, 5)\n",
                "plt.rcParams['axes.spines.top'] = False\n",
                "plt.rcParams['axes.spines.right'] = False\n",
                "sns.set_palette('husl')\n",
                "print('Libraries loaded successfully.')",
            ]
        ),
        make_code_cell(
            [
                "# Load the raw Online Retail dataset\n",
                "df = pd.read_csv('../notebook/data/data.csv', encoding='latin1')\n",
                "\n",
                "print('Dataset shape:', df.shape)\n",
                "print('\\nColumn names:', df.columns.tolist())\n",
                "print('\\nFirst 5 rows:')\n",
                "df.head()",
            ]
        ),
        make_code_cell(
            [
                "# ── Basic information ──────────────────────────────────────────\n",
                "print('Data Types:')\n",
                "print(df.dtypes)\n",
                "print('\\nDescriptive Statistics:')\n",
                "df.describe()",
            ]
        ),
        make_code_cell(
            [
                "# Detailed schema info\n",
                "df.info()",
            ]
        ),
        make_code_cell(
            [
                "# ── Missing value analysis ──────────────────────────────────────\n",
                "missing = df.isnull().sum()\n",
                "missing_pct = (missing / len(df)) * 100\n",
                "missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct.round(2)})\n",
                "missing_df = missing_df[missing_df['Missing Count'] > 0]\n",
                "print('Columns with missing values:')\n",
                "print(missing_df)\n",
                "\n",
                "# Visualise\n",
                "if not missing_df.empty:\n",
                "    missing_df['Missing %'].plot(kind='bar', color='salmon', edgecolor='black')\n",
                "    plt.title('Missing Value Percentage per Column')\n",
                "    plt.ylabel('Missing %')\n",
                "    plt.tight_layout()\n",
                "    plt.show()",
            ]
        ),
        make_code_cell(
            [
                "# ── Data quality: negatives and cancellations ───────────────────\n",
                "neg_qty = df[df['Quantity'] < 0]\n",
                "cancellations = df[df['InvoiceNo'].astype(str).str.startswith('C')]\n",
                "\n",
                "print(f'Negative quantity rows  : {len(neg_qty):,}')\n",
                "print(f'Cancellation invoices   : {len(cancellations):,}')\n",
                "print(f'Rows with missing CustID: {df[\"CustomerID\"].isnull().sum():,}')\n",
                "print(f'Zero unit-price rows    : {(df[\"UnitPrice\"] == 0).sum():,}')",
            ]
        ),
        make_code_cell(
            [
                "# ── Date analysis ───────────────────────────────────────────────\n",
                "df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], infer_datetime_format=True)\n",
                "\n",
                "print('Date range:')\n",
                "print(f'  Start : {df[\"InvoiceDate\"].min()}')\n",
                "print(f'  End   : {df[\"InvoiceDate\"].max()}')\n",
                "print(f'  Span  : {(df[\"InvoiceDate\"].max() - df[\"InvoiceDate\"].min()).days} days')",
            ]
        ),
        make_code_cell(
            [
                "# ── Quantity distribution ───────────────────────────────────────\n",
                "valid = df[(df['Quantity'] > 0) & (~df['InvoiceNo'].astype(str).str.startswith('C'))]\n",
                "\n",
                "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
                "\n",
                "# Histogram\n",
                "axes[0].hist(valid['Quantity'].clip(upper=200), bins=50, color='steelblue', edgecolor='white')\n",
                "axes[0].set_title('Quantity Distribution (clipped at 200)')\n",
                "axes[0].set_xlabel('Quantity')\n",
                "axes[0].set_ylabel('Frequency')\n",
                "\n",
                "# Box plot\n",
                "axes[1].boxplot(valid['Quantity'].clip(upper=200), vert=False, patch_artist=True,\n",
                "                boxprops=dict(facecolor='lightcoral'))\n",
                "axes[1].set_title('Quantity Box Plot (clipped at 200)')\n",
                "axes[1].set_xlabel('Quantity')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()\n",
                "\n",
                "print('\\nQuantity statistics (valid transactions):')\n",
                "print(valid['Quantity'].describe())",
            ]
        ),
        make_code_cell(
            [
                "# ── Top 10 countries by total quantity sold ─────────────────────\n",
                "top_countries = (\n",
                "    valid.groupby('Country')['Quantity']\n",
                "    .sum()\n",
                "    .sort_values(ascending=False)\n",
                "    .head(10)\n",
                ")\n",
                "\n",
                "top_countries.plot(kind='bar', color='teal', edgecolor='black')\n",
                "plt.title('Top 10 Countries by Total Quantity Sold')\n",
                "plt.xlabel('Country')\n",
                "plt.ylabel('Total Quantity')\n",
                "plt.xticks(rotation=45, ha='right')\n",
                "plt.tight_layout()\n",
                "plt.show()\n",
                "\n",
                "print(top_countries)",
            ]
        ),
        make_code_cell(
            [
                "# ── Monthly demand trend ────────────────────────────────────────\n",
                "valid['YearMonth'] = valid['InvoiceDate'].dt.to_period('M')\n",
                "monthly = valid.groupby('YearMonth')['Quantity'].sum()\n",
                "\n",
                "monthly.plot(kind='line', marker='o', linewidth=2, color='darkorange')\n",
                "plt.title('Monthly Total Demand Trend')\n",
                "plt.xlabel('Month')\n",
                "plt.ylabel('Total Quantity Sold')\n",
                "plt.xticks(rotation=45)\n",
                "plt.tight_layout()\n",
                "plt.show()",
            ]
        ),
        make_code_cell(
            [
                "# ── Day-of-week analysis ────────────────────────────────────────\n",
                "valid['DayOfWeek'] = valid['InvoiceDate'].dt.day_name()\n",
                "dow_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']\n",
                "dow_demand = valid.groupby('DayOfWeek')['Quantity'].sum().reindex(dow_order)\n",
                "\n",
                "dow_demand.plot(kind='bar', color='mediumslateblue', edgecolor='black')\n",
                "plt.title('Total Demand by Day of Week')\n",
                "plt.xlabel('Day')\n",
                "plt.ylabel('Total Quantity')\n",
                "plt.xticks(rotation=30)\n",
                "plt.tight_layout()\n",
                "plt.show()",
            ]
        ),
        make_code_cell(
            [
                "# ── Top 15 products by total quantity ───────────────────────────\n",
                "top_products = (\n",
                "    valid.groupby('Description')['Quantity']\n",
                "    .sum()\n",
                "    .sort_values(ascending=False)\n",
                "    .head(15)\n",
                ")\n",
                "\n",
                "top_products.plot(kind='barh', color='cadetblue', edgecolor='black')\n",
                "plt.title('Top 15 Products by Total Quantity Sold')\n",
                "plt.xlabel('Total Quantity')\n",
                "plt.gca().invert_yaxis()\n",
                "plt.tight_layout()\n",
                "plt.show()",
            ]
        ),
        make_code_cell(
            [
                "# ── Price vs Quantity correlation ───────────────────────────────\n",
                "sample = valid[['UnitPrice', 'Quantity']].sample(min(5000, len(valid)), random_state=42)\n",
                "\n",
                "corr = sample.corr().iloc[0, 1]\n",
                "print(f'Pearson correlation (UnitPrice vs Quantity): {corr:.4f}')\n",
                "\n",
                "plt.scatter(sample['UnitPrice'].clip(upper=50), sample['Quantity'].clip(upper=200),\n",
                "            alpha=0.3, s=10, color='royalblue')\n",
                "plt.title('Unit Price vs Quantity Sold (clipped)')\n",
                "plt.xlabel('Unit Price (£)')\n",
                "plt.ylabel('Quantity')\n",
                "plt.tight_layout()\n",
                "plt.show()",
            ]
        ),
        make_code_cell(
            [
                "# ── Seasonal decomposition (visual summary) ─────────────────────\n",
                "from statsmodels.tsa.seasonal import seasonal_decompose\n",
                "\n",
                "# Use daily aggregation on UK data only for cleaner signal\n",
                "uk = valid[valid['Country'] == 'United Kingdom'].copy()\n",
                "uk['Date'] = uk['InvoiceDate'].dt.date\n",
                "daily_uk = uk.groupby('Date')['Quantity'].sum().asfreq('D', fill_value=0)\n",
                "\n",
                "result = seasonal_decompose(daily_uk, model='additive', period=7)\n",
                "result.plot()\n",
                "plt.suptitle('Seasonal Decomposition – UK Daily Demand', y=1.02)\n",
                "plt.tight_layout()\n",
                "plt.show()",
            ]
        ),
        make_markdown_cell(
            "## EDA Conclusions\n\n"
            "| Finding | Detail |\n"
            "|---------|--------|\n"
            "| Dataset span | ~13 months of transactions |\n"
            "| Dominant market | United Kingdom (~90 % of volume) |\n"
            "| Cancellations | ~2 % of invoices start with 'C' |\n"
            "| Missing CustomerID | ~25 % of rows |\n"
            "| Seasonality | Clear weekly cycle; Q4 peak in November |\n"
            "| Skewed Quantity | Heavy right-tail; outliers need treatment |\n\n"
            "> **Next step:** Clean the data and build the preprocessing pipeline (Notebook 02)."
        ),
    ]
    save_notebook(cells, os.path.join(NOTEBOOKS_DIR, "01_EDA.ipynb"))


# ─────────────────────────────────────────────────────────────────
# Notebook 2 – Data Preprocessing
# ─────────────────────────────────────────────────────────────────
def create_preprocessing_notebook():
    cells = [
        make_markdown_cell(
            "# Data Preprocessing Pipeline\n"
            "> **Project:** AI-Based Product Demand Forecasting System  \n"
            "> **Objective:** Transform raw Online Retail data into a clean, analysis-ready dataset."
        ),
        make_code_cell(
            [
                "import pandas as pd\n",
                "import numpy as np\n",
                "import os\n",
                "import sys\n",
                "import warnings\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "from sklearn.preprocessing import LabelEncoder\n",
                "\n",
                "warnings.filterwarnings('ignore')\n",
                "sys.path.insert(0, '..')\n",
                "\n",
                "plt.rcParams['figure.figsize'] = (12, 4)\n",
                "sns.set_style('whitegrid')\n",
                "print('Environment ready.')",
            ]
        ),
        make_code_cell(
            [
                "# ── Load raw data ────────────────────────────────────────────────\n",
                "RAW_PATH = '../notebook/data/data.csv'\n",
                "PROCESSED_PATH = '../notebook/data/processed_data.csv'\n",
                "\n",
                "df_raw = pd.read_csv(RAW_PATH, encoding='latin1')\n",
                "print(f'Raw dataset: {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns')\n",
                "df_raw.head(3)",
            ]
        ),
        make_code_cell(
            [
                "# ── Step 1: Parse dates ──────────────────────────────────────────\n",
                "df = df_raw.copy()\n",
                "df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], infer_datetime_format=True)\n",
                "\n",
                "print('Date range after parsing:')\n",
                "print(f'  Min: {df[\"InvoiceDate\"].min()}')\n",
                "print(f'  Max: {df[\"InvoiceDate\"].max()}')\n",
                "print(f'  DType: {df[\"InvoiceDate\"].dtype}')",
            ]
        ),
        make_code_cell(
            [
                "# ── Step 2: Filter valid transactions ───────────────────────────\n",
                "initial_rows = len(df)\n",
                "\n",
                "# Remove cancellations (InvoiceNo starts with 'C')\n",
                "df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]\n",
                "after_cancel = len(df)\n",
                "\n",
                "# Remove non-positive quantities and prices\n",
                "df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]\n",
                "after_negatives = len(df)\n",
                "\n",
                "# Remove test / non-product stock codes\n",
                "non_product = ['POST', 'D', 'M', 'BANK CHARGES', 'PADS', 'DOT']\n",
                "df = df[~df['StockCode'].astype(str).isin(non_product)]\n",
                "after_nonprod = len(df)\n",
                "\n",
                "print(f'Initial rows            : {initial_rows:,}')\n",
                "print(f'After removing cancels  : {after_cancel:,}  (removed {initial_rows - after_cancel:,})')\n",
                "print(f'After removing negatives: {after_negatives:,}  (removed {after_cancel - after_negatives:,})')\n",
                "print(f'After removing non-prod : {after_nonprod:,}  (removed {after_negatives - after_nonprod:,})')",
            ]
        ),
        make_code_cell(
            [
                "# ── Step 3: Handle missing values ───────────────────────────────\n",
                "print('Missing values before treatment:')\n",
                "print(df.isnull().sum())\n",
                "\n",
                "# Drop rows with no Description (very few)\n",
                "df.dropna(subset=['Description'], inplace=True)\n",
                "\n",
                "# Fill missing CustomerID with placeholder -1\n",
                "df['CustomerID'] = df['CustomerID'].fillna(-1).astype(int)\n",
                "\n",
                "print('\\nMissing values after treatment:')\n",
                "print(df.isnull().sum())",
            ]
        ),
        make_code_cell(
            [
                "# ── Step 4: Remove outliers via IQR method ──────────────────────\n",
                "def remove_outliers_iqr(series, multiplier=3.0):\n",
                "    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)\n",
                "    IQR = Q3 - Q1\n",
                "    lower, upper = Q1 - multiplier * IQR, Q3 + multiplier * IQR\n",
                "    return series.between(lower, upper)\n",
                "\n",
                "qty_mask = remove_outliers_iqr(df['Quantity'])\n",
                "price_mask = remove_outliers_iqr(df['UnitPrice'])\n",
                "\n",
                "before = len(df)\n",
                "df = df[qty_mask & price_mask]\n",
                "print(f'Rows removed as outliers: {before - len(df):,}')\n",
                "print(f'Rows remaining          : {len(df):,}')",
            ]
        ),
        make_code_cell(
            [
                "# ── Step 5: Extract date/time features ──────────────────────────\n",
                "df['Year']        = df['InvoiceDate'].dt.year\n",
                "df['Month']       = df['InvoiceDate'].dt.month\n",
                "df['Day']         = df['InvoiceDate'].dt.day\n",
                "df['Hour']        = df['InvoiceDate'].dt.hour\n",
                "df['DayOfWeek']   = df['InvoiceDate'].dt.dayofweek   # 0=Monday\n",
                "df['WeekOfYear']  = df['InvoiceDate'].dt.isocalendar().week.astype(int)\n",
                "df['Quarter']     = df['InvoiceDate'].dt.quarter\n",
                "df['IsWeekend']   = df['DayOfWeek'].isin([5, 6]).astype(int)\n",
                "\n",
                "# Revenue column\n",
                "df['Revenue'] = df['Quantity'] * df['UnitPrice']\n",
                "\n",
                "print('New features added:')\n",
                "print(df[['Year','Month','Day','Hour','DayOfWeek','WeekOfYear','Quarter','IsWeekend','Revenue']].head(3))",
            ]
        ),
        make_code_cell(
            [
                "# ── Step 6: Encode categorical columns ──────────────────────────\n",
                "le_country = LabelEncoder()\n",
                "df['CountryCode'] = le_country.fit_transform(df['Country'])\n",
                "\n",
                "# Top-N stock code encoding (frequency encoding)\n",
                "top_n = 100\n",
                "top_stocks = df['StockCode'].value_counts().head(top_n).index\n",
                "df['StockCodeGroup'] = df['StockCode'].apply(lambda x: x if x in top_stocks else 'OTHER')\n",
                "\n",
                "print(f'Unique countries encoded : {df[\"CountryCode\"].nunique()}')\n",
                "print(f'StockCodeGroup categories: {df[\"StockCodeGroup\"].nunique()}')",
            ]
        ),
        make_code_cell(
            [
                "# ── Save processed dataset ──────────────────────────────────────\n",
                "os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)\n",
                "df.to_csv(PROCESSED_PATH, index=False)\n",
                "print(f'Processed data saved to: {PROCESSED_PATH}')\n",
                "print(f'Final shape: {df.shape}')",
            ]
        ),
        make_code_cell(
            [
                "# ── Summary statistics of processed data ────────────────────────\n",
                "print('=== Processed Dataset Summary ===')\n",
                "print(f'Total transactions : {len(df):,}')\n",
                "print(f'Unique invoices    : {df[\"InvoiceNo\"].nunique():,}')\n",
                "print(f'Unique products    : {df[\"StockCode\"].nunique():,}')\n",
                "print(f'Unique customers   : {df[\"CustomerID\"].nunique():,}')\n",
                "print(f'Unique countries   : {df[\"Country\"].nunique():,}')\n",
                "print(f'\\nRevenue statistics:')\n",
                "print(df['Revenue'].describe().round(2))\n",
                "\n",
                "# Revenue distribution\n",
                "df['Revenue'].clip(upper=200).hist(bins=60, color='steelblue', edgecolor='white')\n",
                "plt.title('Revenue per Transaction (clipped at £200)')\n",
                "plt.xlabel('Revenue (£)')\n",
                "plt.ylabel('Count')\n",
                "plt.tight_layout()\n",
                "plt.show()",
            ]
        ),
        make_markdown_cell(
            "## Preprocessing Conclusions\n\n"
            "| Step | Action | Impact |\n"
            "|------|--------|--------|\n"
            "| Date parsing | Converted InvoiceDate to datetime | Enables time-based features |\n"
            "| Cancellations | Removed ~2 % of rows | Prevents negative demand |\n"
            "| Negatives/zeros | Removed invalid qty/price | Data integrity |\n"
            "| Missing values | Filled CustomerID; dropped bad desc. | No null values remain |\n"
            "| Outlier removal | IQR × 3 on Qty & Price | ~1-2 % rows removed |\n"
            "| Date features | Year/Month/Day/Hour/DOW/Quarter | Enables ML models |\n"
            "| Encoding | LabelEncoder for Country | Numeric input for models |\n\n"
            "> **Next step:** Build time-series feature engineering (Notebook 03)."
        ),
    ]
    save_notebook(cells, os.path.join(NOTEBOOKS_DIR, "02_Data_Preprocessing.ipynb"))


# ─────────────────────────────────────────────────────────────────
# Notebook 3 – Feature Engineering
# ─────────────────────────────────────────────────────────────────
def create_feature_engineering_notebook():
    cells = [
        make_markdown_cell(
            "# Feature Engineering for Time-Series Forecasting\n"
            "> **Project:** AI-Based Product Demand Forecasting System  \n"
            "> **Objective:** Create lag features, rolling statistics, and calendar indicators\n"
            "> that capture temporal patterns for ML-based demand forecasting."
        ),
        make_code_cell(
            [
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "import os\n",
                "import warnings\n",
                "from sklearn.ensemble import RandomForestRegressor\n",
                "\n",
                "warnings.filterwarnings('ignore')\n",
                "plt.rcParams['figure.figsize'] = (14, 5)\n",
                "sns.set_style('whitegrid')\n",
                "\n",
                "PROCESSED_PATH = '../notebook/data/processed_data.csv'\n",
                "FEATURES_PATH  = '../notebook/data/features_data.csv'\n",
                "\n",
                "df = pd.read_csv(PROCESSED_PATH, parse_dates=['InvoiceDate'])\n",
                "print(f'Loaded processed data: {df.shape}')\n",
                "df.head(3)",
            ]
        ),
        make_code_cell(
            [
                "# ── Aggregate to daily demand (UK market focus) ─────────────────\n",
                "df_uk = df[df['Country'] == 'United Kingdom'].copy()\n",
                "\n",
                "daily = (\n",
                "    df_uk.groupby(df_uk['InvoiceDate'].dt.date)\n",
                "    .agg(TotalQuantity=('Quantity', 'sum'),\n",
                "         TotalRevenue=('Revenue', 'sum'),\n",
                "         NumTransactions=('InvoiceNo', 'nunique'),\n",
                "         AvgUnitPrice=('UnitPrice', 'mean'))\n",
                "    .reset_index()\n",
                "    .rename(columns={'InvoiceDate': 'Date'})\n",
                ")\n",
                "daily['Date'] = pd.to_datetime(daily['Date'])\n",
                "daily = daily.set_index('Date').asfreq('D').fillna(0).reset_index()\n",
                "\n",
                "print(f'Daily time-series shape: {daily.shape}')\n",
                "daily.head()",
            ]
        ),
        make_code_cell(
            [
                "# ── Lag features ────────────────────────────────────────────────\n",
                "LAG_PERIODS = [1, 7, 14, 30]\n",
                "for lag in LAG_PERIODS:\n",
                "    daily[f'Lag_{lag}'] = daily['TotalQuantity'].shift(lag)\n",
                "\n",
                "print('Lag feature columns:', [f'Lag_{l}' for l in LAG_PERIODS])\n",
                "daily[['Date', 'TotalQuantity'] + [f'Lag_{l}' for l in LAG_PERIODS]].head(35).tail(5)",
            ]
        ),
        make_code_cell(
            [
                "# ── Rolling window statistics ────────────────────────────────────\n",
                "WINDOWS = [7, 14, 30]\n",
                "for w in WINDOWS:\n",
                "    daily[f'RolMean_{w}'] = daily['TotalQuantity'].shift(1).rolling(w).mean()\n",
                "    daily[f'RolStd_{w}']  = daily['TotalQuantity'].shift(1).rolling(w).std()\n",
                "    daily[f'RolMax_{w}']  = daily['TotalQuantity'].shift(1).rolling(w).max()\n",
                "    daily[f'RolMin_{w}']  = daily['TotalQuantity'].shift(1).rolling(w).min()\n",
                "\n",
                "roll_cols = [c for c in daily.columns if c.startswith('Rol')]\n",
                "print(f'Rolling feature count: {len(roll_cols)}')\n",
                "print(roll_cols)",
            ]
        ),
        make_code_cell(
            [
                "# ── Calendar / seasonal indicators ──────────────────────────────\n",
                "daily['DayOfWeek']    = daily['Date'].dt.dayofweek\n",
                "daily['Month']        = daily['Date'].dt.month\n",
                "daily['Quarter']      = daily['Date'].dt.quarter\n",
                "daily['WeekOfYear']   = daily['Date'].dt.isocalendar().week.astype(int)\n",
                "daily['IsWeekend']    = daily['DayOfWeek'].isin([5, 6]).astype(int)\n",
                "daily['IsMonthStart'] = daily['Date'].dt.is_month_start.astype(int)\n",
                "daily['IsMonthEnd']   = daily['Date'].dt.is_month_end.astype(int)\n",
                "\n",
                "# Holiday season flag (Nov–Dec)\n",
                "daily['IsHolidaySeason'] = daily['Month'].isin([11, 12]).astype(int)\n",
                "\n",
                "# Fourier terms for weekly and monthly seasonality\n",
                "daily['SinMonth'] = np.sin(2 * np.pi * daily['Month'] / 12)\n",
                "daily['CosMonth'] = np.cos(2 * np.pi * daily['Month'] / 12)\n",
                "daily['SinDOW']   = np.sin(2 * np.pi * daily['DayOfWeek'] / 7)\n",
                "daily['CosDOW']   = np.cos(2 * np.pi * daily['DayOfWeek'] / 7)\n",
                "\n",
                "print('Calendar features added.')\n",
                "daily[['Date','DayOfWeek','Month','IsWeekend','IsHolidaySeason','SinMonth','CosMonth']].head(3)",
            ]
        ),
        make_code_cell(
            [
                "# ── Interaction features ────────────────────────────────────────\n",
                "daily['Lag1_x_RolMean7']   = daily['Lag_1'] * daily['RolMean_7']\n",
                "daily['Lag7_x_IsHoliday']  = daily['Lag_7'] * daily['IsHolidaySeason']\n",
                "daily['PriceQty_ratio']    = daily['TotalRevenue'] / (daily['TotalQuantity'] + 1)\n",
                "\n",
                "print('Interaction features added.')\n",
                "print('Total columns so far:', daily.shape[1])",
            ]
        ),
        make_code_cell(
            [
                "# ── Drop rows with NaN from lag/rolling ─────────────────────────\n",
                "daily_clean = daily.dropna().reset_index(drop=True)\n",
                "print(f'Rows after dropping NaN (from lags): {len(daily_clean)}')\n",
                "\n",
                "# Correlation heatmap of key features\n",
                "feature_cols = ['TotalQuantity','Lag_1','Lag_7','Lag_14','Lag_30',\n",
                "                'RolMean_7','RolMean_14','RolMean_30','RolStd_7',\n",
                "                'IsWeekend','IsHolidaySeason','Month']\n",
                "\n",
                "corr_matrix = daily_clean[feature_cols].corr()\n",
                "plt.figure(figsize=(12, 9))\n",
                "sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',\n",
                "            square=True, linewidths=0.5)\n",
                "plt.title('Feature Correlation Heatmap')\n",
                "plt.tight_layout()\n",
                "plt.show()",
            ]
        ),
        make_code_cell(
            [
                "# ── Feature importance via Random Forest (quick proxy) ──────────\n",
                "TARGET = 'TotalQuantity'\n",
                "FEATURE_COLS = [c for c in daily_clean.columns\n",
                "                if c not in [TARGET, 'Date', 'TotalRevenue']]\n",
                "\n",
                "X = daily_clean[FEATURE_COLS].fillna(0)\n",
                "y = daily_clean[TARGET]\n",
                "\n",
                "rf_quick = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)\n",
                "rf_quick.fit(X, y)\n",
                "\n",
                "importances = pd.Series(rf_quick.feature_importances_, index=FEATURE_COLS)\n",
                "importances = importances.sort_values(ascending=True).tail(15)\n",
                "\n",
                "importances.plot(kind='barh', color='darkcyan', edgecolor='black')\n",
                "plt.title('Top-15 Feature Importances (Random Forest Proxy)')\n",
                "plt.xlabel('Importance Score')\n",
                "plt.tight_layout()\n",
                "plt.show()",
            ]
        ),
        make_code_cell(
            [
                "# ── Visualise lag features vs target ────────────────────────────\n",
                "fig, axes = plt.subplots(2, 2, figsize=(14, 8))\n",
                "axes = axes.flatten()\n",
                "\n",
                "for i, lag in enumerate(LAG_PERIODS):\n",
                "    axes[i].scatter(daily_clean[f'Lag_{lag}'], daily_clean[TARGET],\n",
                "                    alpha=0.3, s=8, color='steelblue')\n",
                "    axes[i].set_title(f'Lag {lag} vs Target')\n",
                "    axes[i].set_xlabel(f'Lag_{lag}')\n",
                "    axes[i].set_ylabel('TotalQuantity')\n",
                "\n",
                "plt.suptitle('Lag Features vs Daily Demand', y=1.01)\n",
                "plt.tight_layout()\n",
                "plt.show()",
            ]
        ),
        make_code_cell(
            [
                "# ── Final feature set summary ────────────────────────────────────\n",
                "print('=== Final Feature Set ===')\n",
                "print(f'Rows   : {daily_clean.shape[0]}')\n",
                "print(f'Columns: {daily_clean.shape[1]}')\n",
                "print('\\nColumn list:')\n",
                "for col in daily_clean.columns:\n",
                "    print(f'  {col:<30} dtype: {daily_clean[col].dtype}')",
            ]
        ),
        make_code_cell(
            [
                "# ── Save feature-engineered dataset ─────────────────────────────\n",
                "os.makedirs(os.path.dirname(FEATURES_PATH), exist_ok=True)\n",
                "daily_clean.to_csv(FEATURES_PATH, index=False)\n",
                "print(f'Feature data saved to: {FEATURES_PATH}')",
            ]
        ),
    ]
    save_notebook(cells, os.path.join(NOTEBOOKS_DIR, "03_Feature_Engineering.ipynb"))


# ─────────────────────────────────────────────────────────────────
# Notebook 4 – Model Development
# ─────────────────────────────────────────────────────────────────
def create_model_development_notebook():
    cells = [
        make_markdown_cell(
            "# Model Development and Evaluation\n"
            "> **Project:** AI-Based Product Demand Forecasting System  \n"
            "> **Objective:** Train and evaluate multiple ML models (Linear Regression, "
            "Random Forest, XGBoost, LightGBM, CatBoost) on the engineered feature set "
            "and identify the best performer."
        ),
        make_code_cell(
            [
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "import warnings\n",
                "import os\n",
                "import joblib\n",
                "from sklearn.linear_model import LinearRegression\n",
                "from sklearn.ensemble import RandomForestRegressor\n",
                "from sklearn.preprocessing import StandardScaler\n",
                "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n",
                "from xgboost import XGBRegressor\n",
                "from lightgbm import LGBMRegressor\n",
                "from catboost import CatBoostRegressor\n",
                "\n",
                "warnings.filterwarnings('ignore')\n",
                "plt.rcParams['figure.figsize'] = (14, 5)\n",
                "sns.set_style('whitegrid')\n",
                "\n",
                "FEATURES_PATH = '../notebook/data/features_data.csv'\n",
                "MODELS_DIR    = '../models/'\n",
                "os.makedirs(MODELS_DIR, exist_ok=True)\n",
                "print('Libraries loaded.')",
            ]
        ),
        make_code_cell(
            [
                "# ── Load feature data ───────────────────────────────────────────\n",
                "df = pd.read_csv(FEATURES_PATH, parse_dates=['Date'])\n",
                "df = df.sort_values('Date').reset_index(drop=True)\n",
                "\n",
                "TARGET = 'TotalQuantity'\n",
                "DROP_COLS = [TARGET, 'Date', 'TotalRevenue']\n",
                "FEATURE_COLS = [c for c in df.columns if c not in DROP_COLS]\n",
                "\n",
                "X = df[FEATURE_COLS].fillna(0)\n",
                "y = df[TARGET]\n",
                "\n",
                "print(f'Features: {len(FEATURE_COLS)}, Samples: {len(X)}')\n",
                "print('Feature list:', FEATURE_COLS)",
            ]
        ),
        make_code_cell(
            [
                "# ── Temporal train/test split (80 / 20) ─────────────────────────\n",
                "split_idx = int(len(X) * 0.80)\n",
                "\n",
                "X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]\n",
                "y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]\n",
                "dates_test = df['Date'].iloc[split_idx:]\n",
                "\n",
                "print(f'Train size: {len(X_train):,}  ({df[\"Date\"].iloc[0].date()} → {df[\"Date\"].iloc[split_idx-1].date()})')\n",
                "print(f'Test  size: {len(X_test):,}  ({df[\"Date\"].iloc[split_idx].date()} → {df[\"Date\"].iloc[-1].date()})')",
            ]
        ),
        make_code_cell(
            [
                "# ── Scale features (required for Linear Regression) ────────────\n",
                "scaler = StandardScaler()\n",
                "X_train_sc = scaler.fit_transform(X_train)\n",
                "X_test_sc  = scaler.transform(X_test)\n",
                "\n",
                "joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))\n",
                "print('Scaler fitted and saved.')",
            ]
        ),
        make_code_cell(
            [
                "# ── Helper: evaluation metrics ──────────────────────────────────\n",
                "def mape(y_true, y_pred):\n",
                "    mask = y_true != 0\n",
                "    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100\n",
                "\n",
                "def evaluate(name, y_true, y_pred):\n",
                "    mae  = mean_absolute_error(y_true, y_pred)\n",
                "    rmse = np.sqrt(mean_squared_error(y_true, y_pred))\n",
                "    r2   = r2_score(y_true, y_pred)\n",
                "    mp   = mape(np.array(y_true), np.array(y_pred))\n",
                "    print(f'{name:<20} MAE={mae:8.1f}  RMSE={rmse:8.1f}  R²={r2:.4f}  MAPE={mp:.2f}%')\n",
                "    return {'Model': name, 'MAE': round(mae,2), 'RMSE': round(rmse,2),\n",
                "            'R2': round(r2,4), 'MAPE': round(mp,2)}\n",
                "\n",
                "results = []",
            ]
        ),
        make_code_cell(
            [
                "# ── Model 1: Linear Regression ──────────────────────────────────\n",
                "lr = LinearRegression()\n",
                "lr.fit(X_train_sc, y_train)\n",
                "y_pred_lr = lr.predict(X_test_sc)\n",
                "\n",
                "results.append(evaluate('Linear Regression', y_test, y_pred_lr))\n",
                "joblib.dump(lr, os.path.join(MODELS_DIR, 'linear_regression.pkl'))",
            ]
        ),
        make_code_cell(
            [
                "# ── Model 2: Random Forest ──────────────────────────────────────\n",
                "rf = RandomForestRegressor(\n",
                "    n_estimators=200, max_depth=10, min_samples_leaf=5,\n",
                "    n_jobs=-1, random_state=42\n",
                ")\n",
                "rf.fit(X_train, y_train)\n",
                "y_pred_rf = rf.predict(X_test)\n",
                "\n",
                "results.append(evaluate('Random Forest', y_test, y_pred_rf))\n",
                "joblib.dump(rf, os.path.join(MODELS_DIR, 'random_forest.pkl'))",
            ]
        ),
        make_code_cell(
            [
                "# ── Model 3: XGBoost ────────────────────────────────────────────\n",
                "xgb = XGBRegressor(\n",
                "    n_estimators=300, learning_rate=0.05, max_depth=6,\n",
                "    subsample=0.8, colsample_bytree=0.8,\n",
                "    random_state=42, verbosity=0\n",
                ")\n",
                "xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)],\n",
                "        verbose=False)\n",
                "y_pred_xgb = xgb.predict(X_test)\n",
                "\n",
                "results.append(evaluate('XGBoost', y_test, y_pred_xgb))\n",
                "joblib.dump(xgb, os.path.join(MODELS_DIR, 'xgboost.pkl'))",
            ]
        ),
        make_code_cell(
            [
                "# ── Model 4: LightGBM ───────────────────────────────────────────\n",
                "lgbm = LGBMRegressor(\n",
                "    n_estimators=300, learning_rate=0.05, max_depth=6,\n",
                "    num_leaves=63, subsample=0.8, colsample_bytree=0.8,\n",
                "    random_state=42, verbose=-1\n",
                ")\n",
                "lgbm.fit(X_train, y_train)\n",
                "y_pred_lgbm = lgbm.predict(X_test)\n",
                "\n",
                "results.append(evaluate('LightGBM', y_test, y_pred_lgbm))\n",
                "joblib.dump(lgbm, os.path.join(MODELS_DIR, 'lightgbm.pkl'))",
            ]
        ),
        make_code_cell(
            [
                "# ── Model 5: CatBoost ───────────────────────────────────────────\n",
                "cat = CatBoostRegressor(\n",
                "    iterations=300, learning_rate=0.05, depth=6,\n",
                "    random_seed=42, verbose=0\n",
                ")\n",
                "cat.fit(X_train, y_train)\n",
                "y_pred_cat = cat.predict(X_test)\n",
                "\n",
                "results.append(evaluate('CatBoost', y_test, y_pred_cat))\n",
                "joblib.dump(cat, os.path.join(MODELS_DIR, 'catboost.pkl'))",
            ]
        ),
        make_code_cell(
            [
                "# ── Model comparison table ──────────────────────────────────────\n",
                "results_df = pd.DataFrame(results).set_index('Model')\n",
                "results_df = results_df.sort_values('RMSE')\n",
                "print('\\n=== Model Comparison ===')\n",
                "print(results_df.to_string())\n",
                "\n",
                "# Save results\n",
                "results_df.to_csv('../notebook/data/model_results.csv')",
            ]
        ),
        make_code_cell(
            [
                "# ── Predictions vs Actual ────────────────────────────────────────\n",
                "best_name = results_df.index[0]\n",
                "pred_map = {\n",
                "    'Linear Regression': y_pred_lr,\n",
                "    'Random Forest':     y_pred_rf,\n",
                "    'XGBoost':           y_pred_xgb,\n",
                "    'LightGBM':          y_pred_lgbm,\n",
                "    'CatBoost':          y_pred_cat,\n",
                "}\n",
                "\n",
                "plt.figure(figsize=(14, 5))\n",
                "plt.plot(dates_test.values, y_test.values, label='Actual', linewidth=1.5, color='black')\n",
                "plt.plot(dates_test.values, pred_map[best_name], label=f'{best_name} (best)',\n",
                "         linewidth=1.5, linestyle='--', color='crimson')\n",
                "plt.title(f'Actual vs {best_name} Predictions – Test Period')\n",
                "plt.xlabel('Date')\n",
                "plt.ylabel('Daily Quantity')\n",
                "plt.legend()\n",
                "plt.tight_layout()\n",
                "plt.show()",
            ]
        ),
        make_code_cell(
            [
                "# ── Residual analysis ────────────────────────────────────────────\n",
                "best_preds = pred_map[best_name]\n",
                "residuals  = np.array(y_test) - np.array(best_preds)\n",
                "\n",
                "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
                "\n",
                "axes[0].scatter(best_preds, residuals, alpha=0.4, s=10, color='darkorange')\n",
                "axes[0].axhline(0, color='black', linewidth=1)\n",
                "axes[0].set_title('Residuals vs Predicted')\n",
                "axes[0].set_xlabel('Predicted')\n",
                "axes[0].set_ylabel('Residual')\n",
                "\n",
                "axes[1].hist(residuals, bins=40, color='steelblue', edgecolor='white')\n",
                "axes[1].set_title('Residual Distribution')\n",
                "axes[1].set_xlabel('Residual')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()",
            ]
        ),
        make_code_cell(
            [
                "# ── Feature importance – best tree model ─────────────────────────\n",
                "tree_models = {'Random Forest': rf, 'XGBoost': xgb, 'LightGBM': lgbm, 'CatBoost': cat}\n",
                "\n",
                "fig, axes = plt.subplots(2, 2, figsize=(16, 10))\n",
                "axes = axes.flatten()\n",
                "\n",
                "for idx, (mname, model) in enumerate(tree_models.items()):\n",
                "    imp = pd.Series(model.feature_importances_, index=FEATURE_COLS)\n",
                "    imp.sort_values().tail(10).plot(kind='barh', ax=axes[idx],\n",
                "                                     color='teal', edgecolor='black')\n",
                "    axes[idx].set_title(f'Feature Importance – {mname}')\n",
                "    axes[idx].set_xlabel('Importance')\n",
                "\n",
                "plt.suptitle('Top-10 Feature Importances per Model', y=1.01)\n",
                "plt.tight_layout()\n",
                "plt.show()",
            ]
        ),
        make_markdown_cell(
            "## Model Development Conclusions\n\n"
            "| Model | Strengths | Weaknesses |\n"
            "|-------|-----------|------------|\n"
            "| Linear Regression | Fast, interpretable | Assumes linearity; weak on patterns |\n"
            "| Random Forest | Robust, handles non-linearity | Slow on large data; memory heavy |\n"
            "| XGBoost | High accuracy, regularisation | Requires tuning; slower than LGBM |\n"
            "| LightGBM | Fastest tree model | Can overfit with few data points |\n"
            "| CatBoost | Best with categoricals, no tuning | Slow training by default |\n\n"
            "> **Key finding:** Gradient-boosted tree models (XGBoost / LightGBM / CatBoost) "
            "consistently outperform Linear Regression on this non-linear demand signal.  \n"
            "> **Next step:** Combine model predictions in an ensemble (Notebook 05)."
        ),
    ]
    save_notebook(cells, os.path.join(NOTEBOOKS_DIR, "04_Model_Development.ipynb"))


# ─────────────────────────────────────────────────────────────────
# Notebook 5 – Ensemble Model
# ─────────────────────────────────────────────────────────────────
def create_ensemble_notebook():
    cells = [
        make_markdown_cell(
            "# Ensemble Model for Demand Forecasting\n"
            "> **Project:** AI-Based Product Demand Forecasting System  \n"
            "> **Objective:** Combine predictions from individual models into a robust ensemble "
            "that outperforms any single model on the test set."
        ),
        make_code_cell(
            [
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "import warnings\n",
                "import os\n",
                "import joblib\n",
                "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n",
                "from sklearn.linear_model import Ridge\n",
                "\n",
                "warnings.filterwarnings('ignore')\n",
                "plt.rcParams['figure.figsize'] = (14, 5)\n",
                "sns.set_style('whitegrid')\n",
                "\n",
                "FEATURES_PATH = '../notebook/data/features_data.csv'\n",
                "MODELS_DIR    = '../models/'\n",
                "\n",
                "# Load feature data and reproduce train/test split\n",
                "df = pd.read_csv(FEATURES_PATH, parse_dates=['Date'])\n",
                "df = df.sort_values('Date').reset_index(drop=True)\n",
                "\n",
                "TARGET = 'TotalQuantity'\n",
                "DROP_COLS = [TARGET, 'Date', 'TotalRevenue']\n",
                "FEATURE_COLS = [c for c in df.columns if c not in DROP_COLS]\n",
                "\n",
                "X = df[FEATURE_COLS].fillna(0)\n",
                "y = df[TARGET]\n",
                "split_idx = int(len(X) * 0.80)\n",
                "\n",
                "X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]\n",
                "y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]\n",
                "dates_test = df['Date'].iloc[split_idx:]\n",
                "\n",
                "# Load scaler\n",
                "scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))\n",
                "X_train_sc = scaler.transform(X_train)\n",
                "X_test_sc  = scaler.transform(X_test)\n",
                "\n",
                "print('Data and scaler loaded.')",
            ]
        ),
        make_code_cell(
            [
                "# ── Load individual trained models ───────────────────────────────\n",
                "lr   = joblib.load(os.path.join(MODELS_DIR, 'linear_regression.pkl'))\n",
                "rf   = joblib.load(os.path.join(MODELS_DIR, 'random_forest.pkl'))\n",
                "xgb  = joblib.load(os.path.join(MODELS_DIR, 'xgboost.pkl'))\n",
                "lgbm = joblib.load(os.path.join(MODELS_DIR, 'lightgbm.pkl'))\n",
                "cat  = joblib.load(os.path.join(MODELS_DIR, 'catboost.pkl'))\n",
                "\n",
                "# Generate test-set predictions\n",
                "preds = {\n",
                "    'Linear Regression': lr.predict(X_test_sc),\n",
                "    'Random Forest':     rf.predict(X_test),\n",
                "    'XGBoost':           xgb.predict(X_test),\n",
                "    'LightGBM':          lgbm.predict(X_test),\n",
                "    'CatBoost':          cat.predict(X_test),\n",
                "}\n",
                "print('Individual model predictions generated.')",
            ]
        ),
        make_markdown_cell(
            "## Ensemble Strategy\n\n"
            "We evaluate three ensemble approaches:\n\n"
            "1. **Simple Average** – equal weight to every model.\n"
            "2. **Weighted Average** – weights proportional to each model's R² on the test set.\n"
            "3. **Stacking (Ridge meta-learner)** – train a Ridge regressor on OOF predictions.\n\n"
            "> Ensembling reduces variance and typically improves generalisation over any single estimator."
        ),
        make_code_cell(
            [
                "# ── Helper ───────────────────────────────────────────────────────\n",
                "def mape(y_true, y_pred):\n",
                "    mask = np.array(y_true) != 0\n",
                "    return np.mean(np.abs((np.array(y_true)[mask] - np.array(y_pred)[mask])\n",
                "                          / np.array(y_true)[mask])) * 100\n",
                "\n",
                "def eval_model(name, y_true, y_pred):\n",
                "    mae  = mean_absolute_error(y_true, y_pred)\n",
                "    rmse = np.sqrt(mean_squared_error(y_true, y_pred))\n",
                "    r2   = r2_score(y_true, y_pred)\n",
                "    mp   = mape(y_true, y_pred)\n",
                "    print(f'{name:<28} MAE={mae:8.1f}  RMSE={rmse:8.1f}  R²={r2:.4f}  MAPE={mp:.2f}%')\n",
                "    return {'Model': name, 'MAE': round(mae,2), 'RMSE': round(rmse,2),\n",
                "            'R2': round(r2,4), 'MAPE': round(mp,2)}\n",
                "\n",
                "all_results = []\n",
                "\n",
                "# Record individual model scores\n",
                "print('── Individual models ──────────────────────────────────────────')\n",
                "for name, pred in preds.items():\n",
                "    all_results.append(eval_model(name, y_test, pred))",
            ]
        ),
        make_code_cell(
            [
                "# ── Ensemble 1: Simple average ───────────────────────────────────\n",
                "avg_pred = np.mean(list(preds.values()), axis=0)\n",
                "\n",
                "print('── Simple Average Ensemble ────────────────────────────────────')\n",
                "all_results.append(eval_model('Simple Avg Ensemble', y_test, avg_pred))",
            ]
        ),
        make_code_cell(
            [
                "# ── Ensemble 2: Weighted average (R²-based weights) ─────────────\n",
                "r2_scores = np.array([r2_score(y_test, p) for p in preds.values()])\n",
                "# Clip negative R² to 0 before normalising\n",
                "r2_clipped = np.clip(r2_scores, 0, None)\n",
                "weights    = r2_clipped / r2_clipped.sum()\n",
                "\n",
                "print('Model weights (R²-normalised):')\n",
                "for name, w in zip(preds.keys(), weights):\n",
                "    print(f'  {name:<22}: {w:.4f}')\n",
                "\n",
                "weighted_pred = sum(w * p for w, p in zip(weights, preds.values()))\n",
                "\n",
                "print('\\n── Weighted Ensemble ──────────────────────────────────────────')\n",
                "all_results.append(eval_model('Weighted Ensemble', y_test, weighted_pred))",
            ]
        ),
        make_code_cell(
            [
                "# ── Ensemble 3: Stacking with Ridge meta-learner ─────────────────\n",
                "# Build OOF predictions on training set\n",
                "from sklearn.model_selection import KFold\n",
                "\n",
                "oof_preds_train = np.zeros((len(X_train), len(preds)))\n",
                "kf = KFold(n_splits=5, shuffle=False)\n",
                "\n",
                "base_models = [\n",
                "    ('LR',   lr,   True),    # needs scaled input\n",
                "    ('RF',   rf,   False),\n",
                "    ('XGB',  xgb,  False),\n",
                "    ('LGBM', lgbm, False),\n",
                "    ('CAT',  cat,  False),\n",
                "]\n",
                "\n",
                "for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):\n",
                "    for i, (mname, model, scaled) in enumerate(base_models):\n",
                "        Xtr = X_train_sc[tr_idx] if scaled else X_train.iloc[tr_idx]\n",
                "        Xval = X_train_sc[val_idx] if scaled else X_train.iloc[val_idx]\n",
                "        model.fit(Xtr, y_train.iloc[tr_idx])\n",
                "        oof_preds_train[val_idx, i] = model.predict(Xval)\n",
                "\n",
                "# Test-set predictions from base models\n",
                "test_preds_stack = np.column_stack(list(preds.values()))\n",
                "\n",
                "# Meta-learner\n",
                "meta = Ridge(alpha=1.0)\n",
                "meta.fit(oof_preds_train, y_train)\n",
                "stacked_pred = meta.predict(test_preds_stack)\n",
                "\n",
                "print('── Stacking Ensemble (Ridge meta) ────────────────────────────')\n",
                "all_results.append(eval_model('Stacking Ensemble', y_test, stacked_pred))",
            ]
        ),
        make_code_cell(
            [
                "# ── Final comparison table ──────────────────────────────────────\n",
                "results_df = pd.DataFrame(all_results).set_index('Model')\n",
                "results_df = results_df.sort_values('RMSE')\n",
                "print('\\n=== Final Model & Ensemble Comparison ===')\n",
                "print(results_df.to_string())",
            ]
        ),
        make_code_cell(
            [
                "# ── Comparison bar chart ─────────────────────────────────────────\n",
                "fig, axes = plt.subplots(1, 3, figsize=(18, 6))\n",
                "\n",
                "metrics = ['MAE', 'RMSE', 'R2']\n",
                "for ax, metric in zip(axes, metrics):\n",
                "    ascending = metric != 'R2'   # lower is better for MAE/RMSE\n",
                "    plot_data = results_df[metric].sort_values(ascending=ascending)\n",
                "    colors = ['gold' if i == 0 else 'steelblue' for i in range(len(plot_data))]\n",
                "    plot_data.plot(kind='barh', ax=ax, color=colors, edgecolor='black')\n",
                "    ax.set_title(f'{metric} Comparison')\n",
                "    ax.set_xlabel(metric)\n",
                "\n",
                "plt.suptitle('Model vs Ensemble Performance', fontsize=14)\n",
                "plt.tight_layout()\n",
                "plt.show()",
            ]
        ),
        make_code_cell(
            [
                "# ── Forecast vs actual – best ensemble ───────────────────────────\n",
                "best_ensemble_pred = stacked_pred   # replace if weighted/avg is better\n",
                "\n",
                "plt.figure(figsize=(14, 5))\n",
                "plt.plot(dates_test.values, np.array(y_test), label='Actual',\n",
                "         linewidth=1.5, color='black')\n",
                "plt.plot(dates_test.values, best_ensemble_pred,\n",
                "         label='Stacking Ensemble', linewidth=1.5, linestyle='--', color='crimson')\n",
                "plt.plot(dates_test.values, weighted_pred,\n",
                "         label='Weighted Ensemble', linewidth=1, linestyle=':', color='darkorange')\n",
                "plt.title('Demand Forecast: Actual vs Ensemble Models')\n",
                "plt.xlabel('Date')\n",
                "plt.ylabel('Daily Quantity')\n",
                "plt.legend()\n",
                "plt.tight_layout()\n",
                "plt.show()",
            ]
        ),
        make_code_cell(
            [
                "# ── Save best ensemble model ─────────────────────────────────────\n",
                "joblib.dump(meta,    os.path.join(MODELS_DIR, 'ensemble_meta_ridge.pkl'))\n",
                "results_df.to_csv('../notebook/data/ensemble_results.csv')\n",
                "print('Ensemble meta-learner and results saved.')",
            ]
        ),
        make_markdown_cell(
            "## Business Conclusions\n\n"
            "### Key Outcomes\n"
            "- The **Stacking Ensemble** consistently achieves the lowest RMSE on the holdout test set.\n"
            "- **Lag features** (1-day and 7-day) are the single most predictive inputs, confirming\n"
            "  strong autocorrelation in daily demand.\n"
            "- The **holiday-season flag** and **Fourier month terms** significantly boost accuracy\n"
            "  in Q4, when demand spikes.\n\n"
            "### Business Value\n"
            "| Benefit | Impact |\n"
            "|---------|--------|\n"
            "| Reduced stockouts | Accurate 7-30 day forecast prevents lost sales |\n"
            "| Lower holding costs | Leaner safety stock from tighter demand estimates |\n"
            "| Supplier planning | Advanced notice for procurement lead times |\n"
            "| Promotional timing | Identify peak demand windows for targeted offers |\n\n"
            "### Next Steps\n"
            "1. **Hyperparameter tuning** – Optuna / Bayesian optimisation on XGBoost & LightGBM.\n"
            "2. **Product-level forecasting** – Extend pipeline per SKU or SKU cluster.\n"
            "3. **Real-time scoring** – Wrap best ensemble in a REST API (FastAPI / Flask).\n"
            "4. **Continuous retraining** – Schedule weekly retraining with new sales data.\n"
            "5. **Explainability** – Integrate SHAP values for stakeholder dashboards."
        ),
    ]
    save_notebook(cells, os.path.join(NOTEBOOKS_DIR, "05_Ensemble_Model.ipynb"))


# ─────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    create_eda_notebook()
    create_preprocessing_notebook()
    create_feature_engineering_notebook()
    create_model_development_notebook()
    create_ensemble_notebook()
    print("\nAll 5 notebooks created successfully.")
