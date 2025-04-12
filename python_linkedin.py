import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

df = pd.read_csv("C:\\Users\\mohit\\Desktop\\Childcare_Need_Supply_Cleaned.csv")
sns.set(style="whitegrid")


# === Basic Data Exploration ===
print("\n--- BASIC DATA INSPECTION ---")
print("Head:\n", df.head())
print("\nTail:\n", df.tail())
print("\nInfo:")
df.info()
print("\nDescribe:\n", df.describe())

# Drop NA rows if needed
df_cleaned = df.dropna()
print("\nData shape after dropping NA:", df_cleaned.shape)


# === Statistical Measures ===
print("\n--- STATISTICAL MEASURES ---")
print("Mean of 'unserved_estimate':", df['unserved_estimate'].mean())
print("Standard Deviation of 'unserved_estimate':", df['unserved_estimate'].std())
print("Variance of 'unserved_estimate':", df['unserved_estimate'].var())
print("Covariance Matrix:\n", df[['unserved_estimate', 'private_pay_estimate', 'subsidy']].cov())

# Quartiles and Outliers
Q1 = df['unserved_estimate'].quantile(0.25)
Q3 = df['unserved_estimate'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print("\n--- OUTLIER DETECTION ---")
print(f"Q1: {Q1}, Q3: {Q3}")
print(f"Lower Bound: {lower_bound}, Upper Bound: {upper_bound}")
outliers = df[(df['unserved_estimate'] < lower_bound) | (df['unserved_estimate'] > upper_bound)]
print("Number of outliers in 'unserved_estimate':", outliers.shape[0])
# Z-score
df['zscore_unserved'] = stats.zscore(df['unserved_estimate'].fillna(0))

#Charts & Plots

# 1. Box Plot for unserved estimates
plt.figure(figsize=(7, 4))
sns.boxplot(data=df, y="unserved_estimate", color='skyblue')
plt.title("Box Plot - Unserved Estimate")
plt.tight_layout()
plt.show()

# 2. Heatmap of correlation
plt.figure(figsize=(10, 6))
corr = df[['unserved_estimate', 'private_pay_estimate', 'subsidy', 'percent']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Heatmap - Correlation")
plt.tight_layout()
plt.show()

# 3. Line chart: Unserved Estimate by Age Group (Mean)
df_line = df.groupby("age_group")["unserved_estimate"].mean().reset_index()
plt.figure(figsize=(8, 4))
sns.lineplot(data=df_line, x='age_group', y='unserved_estimate', marker='o')
plt.title("Line Chart - Avg Unserved Estimate by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Average Unserved Estimate")
plt.tight_layout()
plt.show()

# 4. Pair Plot
sns.pairplot(df[['unserved_estimate', 'private_pay_estimate', 'subsidy']].dropna())
plt.suptitle("Pairplot - Numeric Distributions", y=1.02)
plt.tight_layout()
plt.show()

# charts


amount_cols = [
    'Contract Amount FY17', 'Contract Amount FY18', 'Contract Amount FY19',
    'Contract Amount FY20', 'Contract Amount FY21', 'Contract Amount FY22',
    'Contract Amount FY23', 'Contract Amount FY24', 'Total Contract Amount'
]
df_amounts = df[amount_cols].copy()

# Drop rows with all NaNs across these columns
df_amounts = df_amounts.dropna(how='all')

# 1. Box Plot for Total Contract Amount
plt.figure(figsize=(7, 4))
sns.boxplot(data=df_amounts, y="Total Contract Amount", color='skyblue')
plt.title("Box Plot - Total Contract Amount")
plt.tight_layout()
plt.show()



# 2. Heatmap of correlation among fiscal years and total contract amount
plt.figure(figsize=(10, 6))
corr = df_amounts.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Heatmap - Correlation of Contract Amounts")
plt.tight_layout()
plt.show()

# 3. Line chart: Average Contract Amount by Fiscal Year
df_line = df_amounts.drop(columns='Total Contract Amount').mean().reset_index()
df_line.columns = ['Fiscal Year', 'Average Contract Amount']
plt.figure(figsize=(8, 4))
sns.lineplot(data=df_line, x='Fiscal Year', y='Average Contract Amount', marker='o')
plt.title("Line Chart - Avg Contract Amount by Fiscal Year")
plt.xlabel("Fiscal Year")
plt.ylabel("Average Amount")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 4. Pair Plot of a subset of the fiscal years + total amount
sns.pairplot(df_amounts[['Contract Amount FY19', 'Contract Amount FY20', 'Contract Amount FY21', 'Total Contract Amount']].dropna())
plt.suptitle("Pairplot - Contract Amounts", y=1.02)
plt.tight_layout()
plt.show()



# 1. Total number of unserved children by age group
print("1. Total unserved children by age group:")
total_unserved_by_age = df.groupby("age_group")["unserved_estimate"].sum()
print(total_unserved_by_age)

# Convert to DataFrame
df_age = total_unserved_by_age.reset_index(name='unserved')
plt.figure(figsize=(8,5))
sns.barplot(data=df_age, x='age_group', y='unserved', hue='age_group', palette="Blues_d", legend=False)
plt.title("Unserved Children by Age Group")
plt.ylabel("Unserved Estimate")
plt.xlabel("Age Group")
plt.tight_layout()
plt.show()

# 2. Average subsidy provided by SMI bracket
print("\n2. Average subsidy by SMI bracket:")
avg_subsidy = df.groupby("smi_bracket")["subsidy"].mean()
print(avg_subsidy)

df_smi = avg_subsidy.reset_index(name='average_subsidy')
plt.figure(figsize=(8,5))
sns.barplot(data=df_smi, x='smi_bracket', y='average_subsidy', hue='smi_bracket', palette="Oranges", legend=False)
plt.title("Average Subsidy by SMI Bracket")
plt.ylabel("Average Subsidy")
plt.xlabel("SMI Bracket")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3. Top 5 counties with highest unserved children
print("\n3. Top 5 counties with highest unserved children:")
top_unserved = df.groupby("geo_name")["unserved_estimate"].sum().sort_values(ascending=False).head(5)
print(top_unserved)

df_top_unserved = top_unserved.reset_index(name='unserved')
plt.figure(figsize=(8, 5))
sns.barplot(data=df_top_unserved, x='geo_name', y='unserved', hue='geo_name', palette="Greens_d", legend=False)
plt.title("Top 5 Counties by Unserved Children")
plt.ylabel("Unserved Estimate")
plt.xlabel("County")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 4. Compare private pay and subsidy-based served children
print("\n4. Comparison of total Private Pay and Subsidy:")
total_private_pay = df["private_pay_estimate"].sum()
total_subsidy = df["subsidy"].sum()
print("Total Private Pay:", total_private_pay)
print("Total Subsidy:", total_subsidy)

df_funds = pd.DataFrame({
    "Type": ["Private Pay", "Subsidy"],
    "Amount": [total_private_pay, total_subsidy]
})

plt.figure(figsize=(6, 5))
sns.barplot(data=df_funds, x='Type', y='Amount', hue='Type', palette="Set2", legend=False)
plt.title("Total Private Pay vs Subsidy")
plt.ylabel("Total Amount")
plt.tight_layout()
plt.show()

# 5. Age group with highest average percent served
print("\n5. Average percent served by age group:")
avg_percent_by_age = df.groupby("age_group")["percent"].mean().sort_values(ascending=False)
print(avg_percent_by_age)

df_percent = avg_percent_by_age.reset_index(name='average_percent')
plt.figure(figsize=(8,5))
sns.lineplot(data=df_percent, x='age_group', y='average_percent', marker='o', color='purple')
plt.title("Average Percent Served by Age Group")
plt.ylabel("Average Percent")
plt.xlabel("Age Group")
plt.tight_layout()
plt.show()



