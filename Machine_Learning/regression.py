from data import data_utils as utils
import pandas as pd
import ast
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
import os
import numpy as np

import statsmodels.api as sm

relations_path = "data/data/relations.json"
sentiment_file = "Machine_Learning/results.txt"
roots = {
    "conservative", "politics", "republican",
    "liberal", "democrats", "progressive",
    "joerogan", "trump" }

# Load
with open(sentiment_file, "r") as f:
    raw = f.read().strip()
if raw.startswith("dict_items(") and raw.endswith(")"):
    raw = raw[len("dict_items("):-1]
sentiments = dict(ast.literal_eval(raw))

# Distances
root_to_subs = utils.map_roots_to_subreddits(
    file_path=relations_path,
    roots=roots
)

# DataFrame
rows = []
for subs in root_to_subs.values():
    for sub, dist in subs:
        if sub in sentiments:
            rows.append((dist, sentiments[sub]))

df = pd.DataFrame(rows, columns=["distance", "sentiment"])
print("Data sample:\n", df.head(), "\n")

# Pooled Linear Regression
X = df[["distance"]]
y = df["sentiment"]

print(df)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
cv_model = LinearRegression()
cv_scores = cross_val_score(cv_model, X_train, y_train, cv=5, scoring='r2') # cross-validation\
print(cv_scores)

model = LinearRegression().fit(X_train, y_train)
print(f"Intercept: {model.intercept_:.4f}")
print(f"Slope: {model.coef_[0]:.4f}")
print(f"R-squared: {model.score(X, y):.4f}\n")


l2_norm = np.linalg.norm(y - model.predict(X)) 
print("L2 NORM")
print(l2_norm)

# Visualization
xs = np.linspace(df.distance.min(), df.distance.max(), 200).reshape(-1, 1)
ys = model.predict(xs)
yhat = model.predict(X_test)

# Plotting
plt.figure(figsize=(8, 5))
plt.scatter(df.distance, df.sentiment, alpha=0.3, label="subreddits")
plt.plot(xs, ys, linewidth=2, label="linear fit")
plt.xlabel("Distance from root")
plt.ylabel("Average sentiment")
plt.title("Pooled Sentiment vs. Distance Regression")
plt.legend()
plt.tight_layout()
plt.show()

X_sm = sm.add_constant(X)
ols_model = sm.OLS(y, X_sm).fit()

print(ols_model.summary())