
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.nonparametric.kde import KDEUnivariate
from statsmodels.nonparametric import smoothers_lowess
from pandas import Series, DataFrame
from patsy import dmatrices
from sklearn import datasets, svm
# import xgboost as xgb

if __name__ == '__main__':
    inputfile = "rawData/train.csv"
    traindata = pd.read_csv(inputfile)

    inputfile2 = "rawData/test.csv"
    testdata = pd.read_csv(inputfile2)

    from sklearn import tree

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(train_x, train_y)

    test_x = testdata.loc[:, ['Fare']]
    test_x = test_x.replace(np.NAN, 0)

    test_y = clf.predict(test_x)

    outputdf2 = testdata.loc[:, ['PassengerId']].merge(outputdf, left_index=True, right_index=True)
    outputdf2.to_csv("submission.csv", index=False)




