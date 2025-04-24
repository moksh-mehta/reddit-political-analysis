import pandas as pd
import ast
import ast
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os
import sys
parent_dir = os.path.abspath('..')
data_utils_path = os.path.join(parent_dir, 'data')
sys.path.append(data_utils_path)
from data import data_utils as utils
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_regression
import numpy as np

import statsmodels.api as sm

centers_to_community_to_text = pd.read_csv("center_to_text.csv")
rows_as_dicts = {}
centers_to_community_to_text['1'] = centers_to_community_to_text['1'].apply(ast.literal_eval)

rows = []
for _, row in centers_to_community_to_text.iterrows():
    dictionary = row['1']
    for dist, biases in dictionary.items():
            for b in biases:
                rows.append({
                    "root":      row['0'],
                    "distance":  dist,
                    "bias":  b
                })



df = pd.DataFrame(rows)


X = df[["distance"]]
y = df["bias"]

Xtrainval, Xtest, ytrainval, ytest = train_test_split(X, y, test_size = 0.2, random_state = 0)

model = LinearRegression().fit(Xtrainval, ytrainval)
overall_scores = cross_val_score(model, Xtrainval, ytrainval, cv=5)
print(f"Pooled model → intercept = {model.intercept_:.4f}, "
      f"slope = {model.coef_[0]:.4f}, R² = {model.score(X, y):.4f}\n")
print(overall_scores)


yhat = model.predict(Xtest)  #TODO: Fill in the right answer for ???
print(yhat[:5])

# Print testing R-squared
testing_r = model.score(Xtest, ytest) #TODO: Fill in the right answer for ???
print('Testing R-squared:', testing_r)

plt.scatter(df["distance"], df["bias"], alpha=0.3)
plt.plot(df["distance"], model.predict(X), linewidth=2)
plt.xlabel("Distance from root")
plt.ylabel("Average political bias")
plt.title("Political Bias vs. Distance Regression")
plt.show()

X_sm = sm.add_constant(X)
ols_model = sm.OLS(y, X_sm).fit()

print(ols_model.summary())