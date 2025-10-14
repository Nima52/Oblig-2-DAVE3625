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

df = pd.read_csv('data/GME_stock.csv')

# Regression analysis
# ❌ explanation needed

#

#

#

def main():
    print(df.head())


if __name__ == "__main__":
    main()
