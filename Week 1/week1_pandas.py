import pandas as pd

# Load the Iris dataset
df = pd.read_csv("iris.csv")

# Display first 5 rows
print("========== First 5 Rows ==========")
print(df.head())

# Dataset information
print("\n========== Dataset Info ==========")
df.info()

# Check missing values
print("\n========== Missing Values ==========")
print(df.isnull().sum())

# Statistical summary
print("\n========== Statistics ==========")
print(df.describe())

# Shape (Rows, Columns)
print("\n========== Shape ==========")
print(df.shape)

# Column names
print("\n========== Column Names ==========")
print(df.columns)

# Features (Input)
X = df.drop("Species", axis=1)

# Label (Output)
y = df["Species"]

print("\n========== Features (X) ==========")
print(X.head())

print("\n========== Labels (y) ==========")
print(y.head())

# Save cleaned dataset
df.to_csv("cleaned_iris.csv", index=False)

print("\n✅ Cleaned dataset saved as 'cleaned_iris.csv'")