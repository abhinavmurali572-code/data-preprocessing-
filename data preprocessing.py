# ML Assignment 1: Data Preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler

# Load dataset
df = pd.read_csv('data.csv')
print(f"Original Shape: {df.shape}")
print(df.head())

# ================= 1. DATA EXPLORATION (2 Marks) =================
# Explore data, list unique values in each feature and find its length
print("\n--- Unique values in each column ---")
for col in df.columns:
    unique_vals = df[col].unique()
    print(f"{col}: {len(unique_vals)} unique values -> {unique_vals[:10]}")  # first 10 only

print("\n--- Data Types ---")
print(df.dtypes)

print("\n--- Statistical Analysis ---")
print(df.describe(include='all'))

print("\n--- Missing Values ---")
print(df.isnull().sum())

# Renaming columns - clean names (lowercase, no spaces)
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
print("\n--- Renamed Columns ---")
print(df.columns.tolist())

# If you want custom renaming:
# df.rename(columns={'old_name': 'new_name', 'salary': 'income'}, inplace=True)

# ================= 2. DATA CLEANING (2 Marks) =================
# a) Find missing and inappropriate values, treat appropriately
# b) Replace value 0 in age as NaN
if 'age' in df.columns:
    df['age'] = df['age'].replace(0, np.nan)
    print(f"\nReplaced 0 in age with NaN. Nulls in age now: {df['age'].isnull().sum()}")

# Treat null values - using mean/median/mode
for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype in ['int64', 'float64']:
            # For numeric - use median (better for outliers)
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"Filled nulls in numeric column '{col}' with median: {median_val}")
        else:
            # For categorical - use mode
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            print(f"Filled nulls in categorical column '{col}' with mode: {mode_val}")

# c) Remove duplicate rows
before_dedup = len(df)
df.drop_duplicates(inplace=True)
after_dedup = len(df)
print(f"\nRemoved {before_dedup - after_dedup} duplicate rows")

# d) Find outliers (using IQR method)
print("\n--- Outlier Detection ---")
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5*IQR
    upper = Q3 + 1.5*IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f"{col}: {len(outliers)} outliers (bounds: {lower:.2f} to {upper:.2f})")

    # Visualize outliers
    plt.figure()
    sns.boxplot(y=df[col])
    plt.title(f'Boxplot - Outliers in {col}')
    plt.show()

# Optional: Remove/Cap outliers - capping example for age and salary
if 'salary' in df.columns:
    Q1 = df['salary'].quantile(0.25)
    Q3 = df['salary'].quantile(0.75)
    IQR = Q3 - Q1
    df['salary'] = np.where(df['salary'] > Q3 + 1.5*IQR, Q3 + 1.5*IQR, df['salary'])
    df['salary'] = np.where(df['salary'] < Q1 - 1.5*IQR, Q1 - 1.5*IQR, df['salary'])

# ================= 3. DATA ANALYSIS (2 Marks) =================
# Filter data with age >40 and salary <5000, Plot chart with age and salary
if 'age' in df.columns and 'salary' in df.columns:
    filtered = df[(df['age'] > 40) & (df['salary'] < 5000)]
    print(f"\nFiltered data (age>40 & salary<5000): {len(filtered)} rows")
    print(filtered.head())

    plt.figure()
    plt.scatter(filtered['age'], filtered['salary'], alpha=0.6, color='red')
    plt.title('Age vs Salary (Age>40 and Salary<5000)')
    plt.xlabel('Age')
    plt.ylabel('Salary')
    plt.grid(True)
    plt.show()

# Count number of people from each place and represent visually
# Assuming column name is 'place', 'city', 'location' - adjust as per your dataset
place_col = None
for possible in ['place', 'city', 'location', 'address', 'region']:
    if possible in df.columns:
        place_col = possible
        break

if place_col:
    place_counts = df[place_col].value_counts()
    print(f"\nPeople from each {place_col}:\n", place_counts)

    plt.figure()
    place_counts.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title(f'Number of People from Each {place_col}')
    plt.xlabel(place_col)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()

    plt.figure()
    plt.pie(place_counts, labels=place_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title(f'Distribution by {place_col}')
    plt.show()
else:
    print("\nNo place/location column found - check your column names")

# ================= 4. DATA ENCODING (2 Marks) =================
# Convert categorical variables to numerical
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f"\nCategorical columns to encode: {categorical_cols}")

# a) Label Encoding
le = LabelEncoder()
df_label_encoded = df.copy()
for col in categorical_cols:
    df_label_encoded[col] = le.fit_transform(df_label_encoded[col].astype(str))
print("\n--- After Label Encoding ---")
print(df_label_encoded.head())

# b) One-Hot Encoding
df_onehot = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
print("\n--- After One-Hot Encoding ---")
print(df_onehot.head())
print(f"Shape after one-hot: {df_onehot.shape}")

# For final modeling, use df_onehot
# Using sklearn OneHotEncoder alternative:
# ohe = OneHotEncoder(sparse_output=False, drop='first')
# encoded = ohe.fit_transform(df[categorical_cols])

# ================= 5. FEATURE SCALING (2 Marks) =================
# After encoding, perform scaling using StandardScaler and MinMaxScaler
# Use the label encoded or one-hot numeric dataframe
X = df_label_encoded.select_dtypes(include=[np.number])

# StandardScaler (mean=0, std=1)
scaler_std = StandardScaler()
X_standard = scaler_std.fit_transform(X)
X_standard_df = pd.DataFrame(X_standard, columns=X.columns)
print("\n--- After StandardScaler ---")
print(X_standard_df.head())
print(X_standard_df.describe())

# MinMaxScaler (range 0-1)
scaler_minmax = MinMaxScaler()
X_minmax = scaler_minmax.fit_transform(X)
X_minmax_df = pd.DataFrame(X_minmax, columns=X.columns)
print("\n--- After MinMaxScaler ---")
print(X_minmax_df.head())
print(X_minmax_df.describe())

# Visualization of scaling effect
if 'age' in X.columns and 'salary' in X.columns:
    plt.figure(figsize=(15,4))
    
    plt.subplot(1,3,1)
    plt.scatter(X['age'], X['salary'])
    plt.title('Original')
    plt.xlabel('Age'); plt.ylabel('Salary')
    
    plt.subplot(1,3,2)
    plt.scatter(X_standard_df['age'], X_standard_df['salary'])
    plt.title('StandardScaler')
    plt.xlabel('Age'); plt.ylabel('Salary')
    
    plt.subplot(1,3,3)
    plt.scatter(X_minmax_df['age'], X_minmax_df['salary'])
    plt.title('MinMaxScaler')
    plt.xlabel('Age'); plt.ylabel('Salary')
    
    plt.tight_layout()
    plt.show()

print("\n--- FINAL PREPROCESSED DATASETS READY ---")
print(f"Label Encoded + Standard Scaled Shape: {X_standard_df.shape}")
print(f"One-Hot Encoded Shape: {df_onehot.shape}")

# Save cleaned files
# X_standard_df.to_csv('cleaned_standard_scaled.csv', index=False)
# X_minmax_df.to_csv('cleaned_minmax_scaled.csv', index=False)
