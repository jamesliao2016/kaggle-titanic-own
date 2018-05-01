
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from test import testfunc
# import xgboost as xgb

if __name__ == '__main__':
    inputfile = "rawData/train.csv"
    traindata = pd.read_csv(inputfile)
    traindata = traindata.replace(np.NAN, 0)

    # inputfile2 = "rawData/test.csv"
    # testdata = pd.read_csv(inputfile2)

    from sklearn import tree
    train_x, test_x, train_y, test_y = train_test_split(
        traindata.loc[:, ['Fare']], traindata.loc[:,['Survived']], test_size=0.33)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(train_x, train_y)

    predict_y = clf.predict(test_x)
    predict_y = pd.DataFrame(predict_y, columns=['Survived'])

    print (testfunc(predict_y, test_y, 'Survived'))

    # outputdf2 = testdata.loc[:, ['PassengerId']].merge(outputdf, left_index=True, right_index=True)
    # outputdf2.to_csv("submission.csv", index=False)




