
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

from sklearn import tree

if __name__ == '__main__':
    inputfile2 = "rawData/test.csv"
    testdata = pd.read_csv(inputfile2)
    inputfile = "rawData/train.csv"
    traindata = pd.read_csv(inputfile)
    traindata = traindata.replace(np.NAN, 0)
    traindata['Sex'] = traindata['Sex'].apply(lambda x:
                                              1 if x=='male' else 0)
    traindata['Embarked'] = traindata['Embarked'].apply(lambda x:0 if x=='S' else (1 if x=='C' else 2))

    train_x = traindata.loc[:, ['Embarked','Fare','Age','Pclass','Sex','SibSp','Parch']]
    train_y = traindata.loc[:,['Survived']]

    clf = tree.DecisionTreeClassifier()
    # clf = LogisticRegression()

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(train_x, train_y)

    test_x = testdata.loc[:, ['Embarked','Fare','Age','Pclass','Sex','SibSp','Parch']]
    test_x['Sex'] = test_x['Sex'].apply(lambda x:
                                              1 if x=='male' else 0)
    test_x['Embarked'] = test_x['Embarked'].apply(lambda x:0 if x=='S' else (1 if x=='C' else 2))

    test_x = test_x.replace(np.NAN, 0)

    test_y = clf.predict(test_x)
    outputdf = pd.DataFrame(test_y, columns=['Survived'])

    outputdf2 = testdata.loc[:, ['PassengerId']].merge(outputdf, left_index=True, right_index=True)
    outputdf2.to_csv("submission1.csv", index=False)




