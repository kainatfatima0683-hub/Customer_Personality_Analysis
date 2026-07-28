import pandas as pd
import numpy as np

df = pd.read_csv("01_Raw_Data/marketing_campaign.csv", sep="\t")

df["Income"] = df["Income"].fillna(df["Income"].median())

df.drop_duplicates(inplace=True)
df.drop_duplicates(subset="ID", inplace=True)

df["Dt_Customer"] = pd.to_datetime(
    df["Dt_Customer"],
    format="%d-%m-%Y"
)

df["Enrollment_Year"] = df["Dt_Customer"].dt.year
df["Enrollment_Month"] = df["Dt_Customer"].dt.month
df["Enrollment_Day"] = df["Dt_Customer"].dt.day

df["Education"] = df["Education"].str.strip().str.title()

df["Marital_Status"] = df["Marital_Status"].str.strip().str.title()

df["Marital_Status"] = df["Marital_Status"].replace({
    "Alone": "Single",
    "Absurd": "Other",
    "Yolo": "Other"
})

current_year = 2026

df["Age"] = current_year - df["Year_Birth"]

# ==========================================
# Handle Outliers using IQR Capping
# ==========================================

outlier_columns = [
    "Income",
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds"
]

for col in outlier_columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    # Cap values
    df[col] = df[col].clip(lower=lower, upper=upper)

print("\nOutliers handled using IQR Capping.")

# Save Outlier Handling Summary
outlier_method = pd.DataFrame({
    "Variable": outlier_columns,
    "Method": ["IQR Capping"] * len(outlier_columns),
    "Reason": ["Reduce impact of extreme values while preserving records"] * len(outlier_columns)
})

outlier_method.to_csv(
    "03_Data_Quality/Outlier_Handling_Report.csv",
    index=False
)

print("Outlier Handling Report Saved Successfully!")

df.to_csv(
    "03_Cleaned_Data/customer_personality_cleaned.csv",
    index=False
)

print("Preprocessing Completed Successfully!")