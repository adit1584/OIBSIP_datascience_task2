import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df1 = pd.read_csv("Unemployment in India.csv")
df2 = pd.read_csv("Unemployment_Rate_upto_11_2020.csv")

df1.columns = df1.columns.str.strip()
df2.columns = df2.columns.str.strip()
df1["Date"] = pd.to_datetime(df1["Date"], errors="coerce", dayfirst=True)
df2["Date"] = pd.to_datetime(df2["Date"], errors="coerce", dayfirst=True)

merged_df = pd.merge(df1, df2, on=["Region", "Date"], how="outer", suffixes=("_df1", "_df2"))
merged_df["Unemployment Rate (%)"] = merged_df[
    ["Estimated Unemployment Rate (%)_df1", "Estimated Unemployment Rate (%)_df2"]
].mean(axis=1)

merged_df = merged_df[
    ["Region", "Date", "Unemployment Rate (%)", "Estimated Employed_df1",
     "Estimated Labour Participation Rate (%)_df1", "Area", "longitude", "latitude"]
].dropna(subset=["Unemployment Rate (%)", "Region", "Date"])

merged_df = merged_df.sort_values(by="Date")

merged_df["Area"] = merged_df["Area"].astype("category").cat.codes
merged_df["Region"] = merged_df["Region"].astype("category").cat.codes
merged_df["Year"] = merged_df["Date"].dt.year
merged_df["Month"] = merged_df["Date"].dt.month

X = merged_df[["Region", "Area", "longitude", "latitude",
               "Estimated Employed_df1", "Estimated Labour Participation Rate (%)_df1", "Year", "Month"]]
y = merged_df["Unemployment Rate (%)"]
X = X.fillna(X.mean())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Actual Unemployment Rate (%)")
plt.ylabel("Predicted Unemployment Rate (%)")
plt.title("Actual vs Predicted Unemployment Rate")
plt.show()

top_regions = merged_df["Region"].value_counts().head(5).index
plt.figure(figsize=(10, 6))
for region in top_regions:
    region_data = merged_df[merged_df["Region"] == region]
    plt.plot(region_data["Date"], region_data["Unemployment Rate (%)"], label=f"Region {region}")
plt.title("Unemployment Rate Trend (Top 5 Regions)")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.legend()
plt.show()
