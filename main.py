"""
Steps to follow: (❌, ✅)
❌ - Explore and research which algorithm would work best for this use case (regression or classification) 
❌ - Document the findings in a markdown cell (3-5 lines) on why you chose this algorithm. 
❌ - Train the algorithm using Python.
❌ - Keep the solution as simple as possible. We are not looking for the best machine-learning algorithm.  
(PS: We are interested in seeing that you know how to work with machine learning.) 
❌ - Turn in a JUPYTER NOTEBOOK on canvas. 

Task:
❌ - Make a prediction algorithm which predicts the price of this stock on a specific date (GME). 
❌ - Input will be date.
❌ - Output will be price (close value in the data file).
❌ - Also show the confusion matrix of your model.
❌ - Also show the prediction accuracy of your model.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import confusion_matrix, accuracy_score

df = pd.read_csv("data/GME_stock.csv")

# Input date
try: 
    dmin_date = pd.to_datetime(df["date"].min())
    dmax_date = pd.to_datetime(df["date"].max())

    input = input("Enter a date (YYYY-MM-DD): ")
    if input.strip() == "":
        raise ValueError("Date must be specified!")
    elif pd.to_datetime(input) < dmin_date or pd.to_datetime(input) > dmax_date:
        raise ValueError(f"Date must be between {dmin_date.date()} and {dmax_date.date()}")

except ValueError as e:
    print(f"Error: {e}")

# Data Preprocessing for rows before and after the given date
# Makes sure that the pricing of the stock is indicative of the target date based on how far forwards the training set predicts in relation to the test set
df_train = df[df["date"] < input].copy()
df_test = df[df["date"] >= input].copy()

# Take high_price and low_price average as target
df_train["target"] = (df_train["high_price"] + df_train["low_price"]) / 2
df_test["target"] = (df_test["high_price"] + df_test["low_price"]) / 2
df["target"] = (df["high_price"] + df["low_price"]) / 2
average_price = df["target"].mean()
print(f"Average Price: {average_price}")
std_price = df["target"].std()
print(f"Standard Deviation of Price: {std_price}")

# High spread so use based on SD
df_train["target_class"] = pd.cut(df_train["target"], 
                                  bins=[-float("inf"), average_price - std_price, average_price + std_price, float("inf")], 
                                  labels=[0,1,2]) 
print(df_train["target_class"].value_counts())

# Prepareing Data for Training
X = df_train.drop(columns=["date", "target", "target_class", "adjclose_price"])
y = df_train["target_class"]
print(X.head())

# Initilize model
model = LogisticRegression(max_iter=1000)

# Perform RFE
rfe = RFE(estimator=model, n_features_to_select=5)
rfe.fit(X, y)

# List features
selected_features = X.columns[rfe.support_]
print(f"Selected Features: {selected_features}")


# Split Data

# Train model




# print(df.head())

# print(output)

