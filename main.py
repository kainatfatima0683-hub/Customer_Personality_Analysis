import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("03_Data_Quality", exist_ok=True)
# ==========================================
# Load Dataset
# ==========================================

file_path = "01_Raw_Data/marketing_campaign.csv"

df = pd.read_csv(file_path, sep="\t")

# ==========================================
# Task 3: Basic Dataset Inspection
# ==========================================

print("=" * 60)
print("CUSTOMER PERSONALITY ANALYSIS")
print("=" * 60)

# Shape
rows, columns = df.shape

print(f"\nNumber of Rows      : {rows}")
print(f"Number of Columns   : {columns}")

# File Size
size = os.path.getsize(file_path)

print(f"Dataset Size (KB)   : {round(size/1024,2)} KB")

print("File Type           : CSV")

# Encoding
try:
    import chardet

    with open(file_path, "rb") as f:
        result = chardet.detect(f.read())

    print("Encoding            :", result["encoding"])

except:
    print("Encoding            : UTF-8")

print("\n" + "=" * 60)
print("COLUMN NAMES")
print("=" * 60)

print(df.columns.tolist())

print("\n" + "=" * 60)
print("DATA TYPES")
print("=" * 60)

print(df.dtypes)

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print(df.info())

print("\n" + "=" * 60)
print("FIRST FIVE ROWS")
print("=" * 60)

print(df.head())

print("\n")
print("="*60)
print("VARIABLE INSPECTION")
print("="*60)

variable_table = pd.DataFrame({
    "Variable": df.columns,
    "Data Type": df.dtypes.astype(str).values,
    "Example": df.iloc[0].values
})

print(variable_table)

variable_table.to_csv(
    "04_Data_Dictionary/Variable_Inspection.csv",
    index=False
)

print("\nVariable Inspection Saved Successfully")

# ==========================================
# Task 5: Missing Values Analysis
# ==========================================

print("\n" + "=" * 60)
print("MISSING VALUES ANALYSIS")
print("=" * 60)

missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100

missing_report = pd.DataFrame({
    "Variable": df.columns,
    "Missing Values": missing_values.values,
    "Percentage": missing_percentage.values.round(2)
})

print(missing_report)

# Columns without missing values
no_missing = missing_report[missing_report["Missing Values"] == 0]["Variable"]

print("\nColumns Without Missing Values:")
print(list(no_missing))

# Save Report
missing_report.to_csv(
    "03_Data_Quality/Missing_Values_Report.csv",
    index=False
)

print("\nMissing Values Report Saved Successfully!")

# ==========================================
# Task 6: Duplicate Records Analysis
# ==========================================

print("\n" + "=" * 60)
print("TASK 6: DUPLICATE RECORDS")
print("=" * 60)

# Check duplicate rows
duplicate_rows = df.duplicated().sum()

# Check duplicate customer IDs
duplicate_ids = df["ID"].duplicated().sum()

print(f"Duplicate Rows: {duplicate_rows}")
print(f"Duplicate Customer IDs: {duplicate_ids}")

# Create report
duplicate_report = pd.DataFrame({
    "Check": ["Duplicate Rows", "Duplicate Customer IDs"],
    "Count": [duplicate_rows, duplicate_ids]
})

# Save report
duplicate_report.to_csv(
    "03_Data_Quality/Duplicate_Report.csv",
    index=False
)

print("\nDuplicate report saved successfully!")

print("=" * 60)
print("TASK 7: DATA TYPE VALIDATION")
print("=" * 60)

data_types = pd.DataFrame({
    "Column": df.columns,
    "Data Type": df.dtypes.astype(str).values
})

print(data_types)

data_types.to_csv(
    "03_Data_Quality/Data_Type_Report.csv",
    index=False
)

print("\nData Type Report Saved Successfully!")

print("=" * 60)
print("TASK 8: OUTLIER DETECTION")
print("=" * 60)

numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

outlier_report = []

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower) | (df[col] > upper)].shape[0]

    outlier_report.append({
        "Variable": col,
        "Outliers": outliers
    })

outlier_report = pd.DataFrame(outlier_report)

print(outlier_report)

outlier_report.to_csv(
    "03_Data_Quality/Outlier_Report.csv",
    index=False
)

print("\nOutlier Report Saved Successfully!")

print("=" * 60)
print("TASK 9: DATA CLEANING")
print("=" * 60)

# Missing values ko median se fill karna
df["Income"] = df["Income"].fillna(df["Income"].median())

# Date column ko datetime me convert karna
df["Dt_Customer"] = pd.to_datetime(
    df["Dt_Customer"],
    format="%d-%m-%Y"
)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# Cleaned dataset save karna
df.to_csv(
    "03_Data_Quality/Cleaned_Customer_Personality.csv",
    index=False
)

print("\nCleaned Dataset Saved Successfully!")

# ============================================================
# TASK 10: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

print("=" * 60)
print("TASK 10: EXPLORATORY DATA ANALYSIS")
print("=" * 60)

eda_df = df.copy()

plt.figure(figsize=(8,5))

sns.histplot(
    eda_df["Income"],
    bins=30,
    kde=True,
    color="skyblue"
)

plt.title("Income Distribution")
plt.xlabel("Income")
plt.ylabel("Frequency")

plt.savefig("05_Screenshots/Education_Distribution.png")

plt.show()

print("Income Histogram Saved Successfully!")

# Education Distribution
plt.figure(figsize=(8,5))
sns.countplot(x='Education', data=df)
plt.xticks(rotation=45)
plt.title("Education Distribution")
plt.tight_layout()
plt.savefig("05_Screenshots/Education_Distribution.png")
plt.close()

print("Education Distribution Saved Successfully!")

plt.figure(figsize=(8,5))
sns.countplot(x="Marital_Status", data=df)
plt.xticks(rotation=45)
plt.title("Marital Status Distribution")
plt.tight_layout()
plt.savefig("05_Screenshots/Marital_Status_Distribution.png")
plt.close()

print("Marital Status Distribution Saved Successfully!")

plt.figure(figsize=(6,4))
sns.countplot(x="Response", data=df)
plt.title("Campaign Response")
plt.savefig("05_Screenshots/Response_Count.png")
plt.close()

print("Response Count Saved Successfully!")

plt.figure(figsize=(6,4))
sns.boxplot(y=df["MntWines"])
plt.title("Wine Spending Boxplot")
plt.savefig("05_Screenshots/Wine_Boxplot.png")
plt.close()

print("Wine Boxplot Saved Successfully!")