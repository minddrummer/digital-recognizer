'''
###test on the random forest parameter setting
### feature engineering

### neural network
### boosting model
### svm model on multiple labels
### final ensembling again
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ggplot as glt
import seaborn as sbn
from sklearn.ensemble import RandomForestClassifier
pd.set_option('display.max_rows',2000)




data = pd.read_csv('train.csv') #reading data is so easy
y  = data.label
data.drop('label',axis=1, inplace =True)
test = pd.read_csv('test.csv')
test_result = pd.read_csv('sample_submission.csv')




rfc = RandomForestClassifier(n_estimators=50,max_depth=10,oob_score=True)
rfc.fit(data, y)
print rfc.oob_score_

test_y = rfc.predict(test)

test_result.loc[:,'Label'] = test_y
test_result.to_csv('test_result.csv', index=False)














