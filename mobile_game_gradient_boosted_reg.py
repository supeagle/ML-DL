
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("classified_train_data.csv")
# csupeagle

# Basics
df.head()

df.describe()

df["installs"] = pd.to_numeric(df["installs"].str.replace(",", "", regex=False))

df["LTV_D28"] = df["revenue_d28"]

selected_features = [f"revenue_d{i}" for i in range(8)] + [f"retained_d{i}" for i in range(1, 8)]
selected_features.insert(0, "installs")

X = df[selected_features]
y = df["LTV_D28"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

polynomial_features = PolynomialFeatures(degree=2, include_bias=False)
scaler = StandardScaler()

pipeline = Pipeline([
    ("polynomial_features", polynomial_features),
    ("scaler", scaler),
    ("model", GradientBoostingRegressor(random_state=42, n_estimators=180, learning_rate=0.095, max_depth=8))
])

# Ton of adjustment for lr,ts,md,ne

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_val)

mae = mean_absolute_error(y_val, y_pred)
mse = mean_squared_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R^2 Score: {r2}")

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_val, y=y_pred, marker="D", color="orange", alpha=0.6, edgecolor="black", s=90, label="Predicted LTV_D28")
sns.lineplot(x=[min(y_val), max(y_val)], y=[min(y_val), max(y_val)], color="green", linestyle='-.', linewidth=2, label="Perfect Prediction Line")
plt.title("Actual vs Predicted LTV_D28")
plt.xlabel("Actual LTV_D28")
plt.ylabel("Predicted LTV_D28")
plt.legend()
plt.grid(True, linestyle='--', color='lightgrey', linewidth=0.8)
plt.show()

