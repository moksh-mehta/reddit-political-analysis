from data import data_utils as utils
import pandas as pd
import ast
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os

relations_path = "data/relations.json"
sentiment_file = "results.txt"
roots = {
    "conservative", "politics", "republican",
    "liberal", "democrats", "progressive",
    "joerogan", "trump" }

with open(sentiment_file, "r") as f:
    raw = f.read().strip()

# remove the leading "dict_items(" and trailing ")"
if raw.startswith("dict_items(") and raw.endswith(")"):
    raw = raw[len("dict_items("):-1]

# evaluate into a list of (subreddit, sentiment) tuples
items = ast.literal_eval(raw)
sentiments = dict(items)

# loading distances
root_to_subs = utils.map_roots_to_subreddits(
    file_path=relations_path,
    roots=roots
)

# dataframe

rows = []
for root, subs in root_to_subs.items():
    for sub, dist in subs:
        if sub in sentiments:
            rows.append({
                "root":      root,
                "subreddit": sub,
                "distance":  dist,
                "sentiment": sentiments[sub]
            })

df = pd.DataFrame(rows)
print("Merged data sample:")
print(df.head(), "\n")

# linear regression - pooled

X = df[["distance"]]
y = df["sentiment"]

model = LinearRegression().fit(X, y)
print(f"Pooled model → intercept = {model.intercept_:.4f}, "
      f"slope = {model.coef_[0]:.4f}, R² = {model.score(X, y):.4f}\n")

# linear regression -- per root

print("Per-root regressions:")
for root, group in df.groupby("root"):
    Xg, yg = group[["distance"]], group["sentiment"]
    m = LinearRegression().fit(Xg, yg)
    print(f"  {root:<12} → intercept={m.intercept_:.3f}, "
          f"slope={m.coef_[0]:.3f}, R²={m.score(Xg, yg):.3f}")
print()

# visualize the fit of pooled

plt.scatter(df["distance"], df["sentiment"], alpha=0.3)
plt.plot(df["distance"], model.predict(X), linewidth=2)
plt.xlabel("Distance from root")
plt.ylabel("Average sentiment")
plt.title("Sentiment vs. Distance Regression")
plt.show()