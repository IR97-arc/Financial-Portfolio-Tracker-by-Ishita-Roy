import pandas as pd

# Step 1: Load the data
df = pd.read_csv('financial_portfolio_tracker_complex.csv')
print("🔹 First few rows of raw data:")
print(df.head())

# Step 2: Clean column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Step 3: Check missing values before cleaning
print("\n🔍 Missing values before cleaning:")
print(df.isnull().sum())

# Step 4: Fill missing values
text_columns = ['account', 'asset_symbol', 'asset_type', 'sector']
numeric_columns = ['units_held', 'unit_price', 'market_value', 'roi', 'dividend']

for col in text_columns:
    if col in df.columns:
        df[col] = df[col].fillna('Unknown').str.strip()

for col in numeric_columns:
    if col in df.columns:
        df[col] = df[col].fillna(0)

# Step 5: Convert 'date' column to datetime
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Step 6: Remove duplicate rows
df = df.drop_duplicates()

# Step 7: Logical consistency – recalculate market value
df['recalculated_value'] = df['units_held'] * df['unit_price']
df['value_diff'] = df['market_value'] - df['recalculated_value']

# Step 8: Outlier handling
# Cap market_value at 99th percentile
cap_value = df['market_value'].quantile(0.99)
df['market_value'] = df['market_value'].apply(lambda x: cap_value if x > cap_value else x)

# Cap ROI to a realistic range (-1 to 1)
df['roi'] = df['roi'].clip(lower=-1, upper=1)

# Step 9: Date filtering (last 5 years only)
df = df[df['date'] >= '2020-01-01']

# Step 10: Standardize asset_type categories
df['asset_type'] = df['asset_type'].str.lower().replace({
    'mutualfund': 'mutual fund',
    'stock ': 'stock',
    'etf ': 'etf'
})

# Step 11: Feature engineering
df['annualized_roi'] = df['roi'] * 100
df['holding_days'] = (pd.Timestamp.today() - df['date']).dt.days

# Step 12: Drop helper columns if not needed
df.drop(columns=['recalculated_value', 'value_diff'], inplace=True)

# Step 13: Final check
print("\n✅ Missing values after cleaning:")
print(df.isnull().sum())
print("\n✅ Data types:")
print(df.dtypes)

# Step 14: Save cleaned data
df.to_csv("cleaned_financial_portfolio_advanced.csv", index=False)
print("\n📁 Cleaned dataset saved as 'cleaned_financial_portfolio_advanced.csv'")
