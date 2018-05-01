import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.nonparametric.kde import KDEUnivariate
from statsmodels.nonparametric import smoothers_lowess
from pandas import Series, DataFrame
from sklearn import datasets, svm
# from test import testfunc


def testfunc(df_predict,df_true,com_var):
    df_merge = df_true.merge(df_predict, left_index=True, right_index=True)
    df_merge['correct_fit']=df_merge.apply(lambda x: 1 if x[com_var+'_x']==x[com_var+'_y'] else 0,axis=1)
    num_total = df_merge.index.size
    num_correct = df_merge['correct_fit'].sum()
    pct_correct = num_correct/num_total
    return pct_correct

if __name__ == '__main__':
    inputfile = "rawData/train.csv"
    traindata = pd.read_csv(inputfile)
    train_y = traindata.loc[:, ['Survived']]
    train_y2 = traindata.loc[:, ['Survived']]
    cc = testfunc(train_y,train_y2,'Survived')
    print(cc)
