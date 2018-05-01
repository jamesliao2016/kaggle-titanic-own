import pandas as pd

def testfunc(df_predict,df_true,com_var):
    df_merge = df_true.merge(df_predict, left_index=True, right_index=True)
    df_merge['correct_fit']=df_merge.apply(lambda x: 1 if x[com_var+'_x']==x[com_var+'_y'] else 0,axis=1)
    num_total = df_merge.index.size
    num_correct = df_merge['correct_fit'].sum()
    pct_correct = num_correct/num_total
    return pct_correct

