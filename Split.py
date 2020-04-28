import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as lm
from sklearn import svm, metrics
from sklearn.externals import joblib

dataset = pd.read_csv('E:\Study\Python\IDS\kddcup99.csv',low_memory=False)
##print ("Whole dataset count : \n",dataset.shape)
##print ("\n\nColumns in whole dataset : \n",dataset.columns)

##print (dataset.count)
##dataset.plot
##plt.show()

y = dataset.label
x = np.array(dataset.drop(['flag'],axis=1))

##x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
##print( "\n\nTraining dataset x : \n",x_train.shape)
##print( "\n\nTraining dataset y : \n",y_train.shape)
##print( "\n\nTesting dataset x : \n",x_test.shape)
##print( "\n\nTesting dataset y : \n",y_test.shape)


gb = dataset.groupby(['protocol_type','service','flag','label'])
##print("\n\nDisplaying types of Protocols , Services , Flags and Labels Which is used : \n",gb.first())

##Training Linear Regression Model
model=lm().fit(x,y)

#Creating Model which can be imported
joblib.dump(model,'model.pkl')
