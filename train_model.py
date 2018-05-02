
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from test import testfunc
# import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

if __name__ == '__main__':
    inputfile = "rawData/train.csv"
    traindata = pd.read_csv(inputfile)
    traindata = traindata.replace(np.NAN, 0)
    traindata['Sex'] = traindata['Sex'].apply(lambda x:
                                              1 if x=='male' else 0)
    traindata['Embarked'] = traindata['Embarked'].apply(lambda x:0 if x=='S' else (1 if x=='C' else 2))

    # inputfile2 = "rawData/test.csv"
    # testdata = pd.read_csv(inputfile2)

    from sklearn import tree
    train_x, test_x, train_y, test_y = train_test_split(
        traindata.loc[:, ['Embarked','Fare','Age','Pclass','Sex','SibSp','Parch']], traindata.loc[:,['Survived']], test_size=0.33)
    print (train_x.index.size)
    print(test_x.index.size)
    clf = tree.DecisionTreeClassifier()
    # clf = LogisticRegression()

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(train_x, train_y)

    predict_y = clf.predict(test_x)
    predict_y = pd.DataFrame(predict_y, columns=['Survived'])
    predict_y.head(5)

    print (testfunc(predict_y, test_y, 'Survived'))

    # outputdf2 = testdata.loc[:, ['PassengerId']].merge(outputdf, left_index=True, right_index=True)
    # outputdf2.to_csv("submission1.csv", index=False)




