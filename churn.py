# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 21:34:08 2019

@author: pc
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn import metrics
import numpy as npf
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import chi2_contingency
#check current working dicstnory
os.getcwd()
os.chdir("C:\\Users\\pc\\Desktop\\data analyst classes")
dataset=pd.read_csv("churn-data.csv")
dataset.describe()
# dimension of data 
dataset.shape
# Number of rows
dataset.shape[0]
# number of columns
dataset.shape[1]
# name of columns
list(dataset)
# data detailat
dataset.info()

# Missing value analysis
dataset.isnull().sum()    # there is no missing value in data

# Creating Numerical and categorical list
dataset_int=["account length","area code","number vmail messages","total day minutes","total day calls","total day charge","total eve minutes","total eve calls","total eve charge","total night minutes","total night calls","total night charge","total intl minutes","total intl calls","total intl charge","customer service calls"]
dataset_obj1=["state","international plan","voice mail plan","phone number"]
       
#apply chi2 test
for i in dataset_obj1:
    print(i)
    chi2,p,dof,ex=chi2_contingency(pd.crosstab(dataset["churn"],dataset[i]))
    print(p)
            
 #change categrical into numeric
lis = []
for i in range(0, dataset.shape[1]):
    #print(i)
    if(dataset.iloc[:,i].dtypes == 'object'):
        dataset.iloc[:,i] = pd.Categorical(dataset.iloc[:,i])
        #print(marketing_train[[i]])
        dataset.iloc[:,i] = dataset.iloc[:,i].cat.codes 
        dataset.iloc[:,i] = dataset.iloc[:,i].astype('int')
        
        lis.append(dataset.columns[i])

#drop phone number cause p value excide by o.05        
dataset=dataset.drop("phone number",axis=1 )      
dataset_obj=["state","international plan","voice mail plan","phone number"] 

#detrmine corr
df_corr = dataset.loc[:,dataset_int]  
corr = df_corr.corr()
f,ax=plt.subplots(figsize=(7,5))
sns.heatmap(corr,mask=np.zeros_like(corr,dtype=np.bool),cmap=sns.diverging_palette(220,10,as_cmap=True),square=True,ax=ax)    
plt.show()

#drop varible by corr

#check correlation  and drop varibale which is high corrilated 
columns=np.full((corr.shape[0],),True,dtype=bool)
for i in range(corr.shape[0]):
    for j in range(i+1,corr.shape[0]):
        if corr.iloc[i,j]>=0.9:
            if columns [j]:
                columns[j]=False
                
                
selected_columns=dataset.loc[:,dataset_int].columns[columns]
final_df=dataset[selected_columns]

final_df.info()

# to check outliers from the dataset  
plt.boxplot(dataset["account length"])
#replce outliers from the lower boundry layer to upper boundry layer
    
def outlier_defect(df):
    for i in df.describe().columns:
        q1=df.describe().at["25%",i]
        q3=df.describe().at["75%",i]
        IQR=(q3-q1)
        ltv=(q1-1.5*IQR)
        utv=(q3+1.5*IQR)
        x=np.array(df[i])
        p=[]
        for j in x:
             if j<ltv:
                p.append(ltv)
             elif j>utv:
                p.append(utv)
             else:
                p.append(j)
        df[i]=p
    return(df)

   # check agin outlier with the help of box plot 
dataset22=outlier_defect(final_df)

# to convertd into int value apply encoder
from sklearn.preprocessing import LabelEncoder        

le=LabelEncoder()
dataset.state=le.fit_transform(dataset.state)
#split x and y 
x=pd.concat([dataset22,dataset.loc[:,dataset_obj]],axis=1)

y=dataset["churn"]
x.info()

# Model develpoment - decision tree
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size = 0.2)

# applying decsion tree
C50_model = tree.DecisionTreeClassifier(criterion='entropy').fit(x_train, y_train)
C50_Predictions = C50_model.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_test,C50_Predictions))
X=x.columns

dot_data = tree.export_graphviz(C50_model, out_file=None, max_depth=3, feature_names=X, class_names=['1','0'],filled=True, rounded=True,special_characters=True)
import graphviz
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

graph2 = graphviz.Source(dot_data)
graph2.render("final")

params = {'max_features': ['auto', 'sqrt', 'log2'],'min_samples_split': [2,3,4,5,6,7,8,9,10], 'min_samples_leaf':[1,2,3,4,5,6,7,8,9,10,11],'max_depth':[2,3,4,5,6,7,8,9]}

params

# Initializing Decision Tree
#gridsearch cv
DTC = tree.DecisionTreeClassifier()
# Building and Fitting Model

DTC1 = GridSearchCV(DTC, param_grid=params)
DTC_RS=DTC1.fit(x_train,y_train)

modelF = DTC_RS.best_estimator_
modelF

pred_modelF = modelF.predict(x_test)
metrics.accuracy_score(y_test,pred_modelF)          

#randomized search cv
from scipy.stats import randint as sp_randint
param_grid2 = {'max_features': ['auto', 'sqrt', 'log2'],
          'min_samples_split': sp_randint(2,10), 
          'min_samples_leaf': sp_randint(1,11),
         'max_depth':sp_randint(2,8)}

# fitting the model
DTC_RS = RandomizedSearchCV(DTC, param_distributions=param_grid2,n_iter=100)
DTC_RS1 = DTC_RS.fit(x_train,y_train)

#Best parameters
model2=DTC_RS1.best_estimator_
pred_model2=model2.predict(x_test)
# Accuracy of model
metrics.accuracy_score(y_test,pred_model2)

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test,pred_modelF )
auc(false_positive_rate, true_positive_rate)
# confusion metrics 
confusion_metrics=metrics.confusion_matrix(y_test,pred_modelF)
confusion_metrics
# Visualization of confusion metrics
%matplotlib inline
class_name=[0,1]
fig,ax=plt.subplots()
ticks_marks=np.arange(len(class_name))
plt.xticks(ticks_marks, class_name)
plt.yticks(ticks_marks, class_name)
sns.heatmap(pd.DataFrame(confusion_metrics),annot=True,cmap="YlGnBu",fmt="g")
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title("Confusion matrix",y=1.1)
plt.ylabel("actual label")
plt.xlabel("predicted label")

#.................................................................................

#applying random forest
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=30)
rfc_fit=rfc.fit(x_train,y_train)
#prediction on train
rfc_pred_train =rfc_fit.predict(x_train)
# Accuracy on train
train_accuracy =metrics.accuracy_score(y_train,rfc_pred_train)
train_accuracy
# prediction on test
rfc_pred = rfc_fit.predict(x_test)
# test accuracy
test_accuracy = metrics.accuracy_score(y_test,rfc_pred)
test_accuracy
#using hyperparamter
#gridsearch and random searc
params_RF = {"max_depth": [3,5,6,7,8], "max_features":['auto', 'sqrt', 'log2'],
"min_samples_split": [2, 3, 10], "min_samples_leaf": [1, 3, 10],
"criterion": ["gini", "entropy"]}
params_RF
model_RF = GridSearchCV(RandomForestClassifier(), param_grid=params_RF)
model_RF.fit(x_train,y_train)
# Best Parameters
	
model_RF.best_params_
# Predict and Check Accuracy for train
rfc_pred_train1 =model_RF.predict(x_train)
train_accuracy1 =metrics.accuracy_score(y_train,rfc_pred_train1)
train_accuracy
# prediction on test
rfc_pred1 = model_RF.predict(x_test)
# test accuracy
test_accuracy = metrics.accuracy_score(y_test,rfc_pred1)
test_accuracy
# Random Search
# Parameters
from scipy.stats import randint 
params_RF_RS = {"max_depth": randint(3,8),
"max_features":['auto', 'sqrt', 'log2'], "min_samples_split":randint (2,10),
"min_samples_leaf":randint (1,10),
"criterion": ["gini", "entropy"]}
# Building and Fitting Model
RF_RS = RandomizedSearchCV(RandomForestClassifier(), param_distributions=params_RF_RS,n_iter=100)
RF_RS.fit(x_train,y_train)
# Best Parameters
RF_RS.best_params_
# Predict and Check Accuracy
# Predict and Check Accuracy for train
rfc_pred_train2 =RF_RS.predict(x_train)
train_accuracy2 =metrics.accuracy_score(y_train,rfc_pred_train2)
train_accuracy
# prediction on test
rfc_pred2 = RF_RS.predict(x_test)

# test accuracy
test_accuracy = metrics.accuracy_score(y_test,rfc_pred2)
test_accuracy
#rfc_pred_prob =RF_RS.predict_proba(x_test)
SS_Residual = sum((y_test-rfc_pred2)**2)
rmse = np.sqrt(np.mean(SS_Residual))
rmse

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test,rfc_pred2 )
auc(false_positive_rate, true_positive_rate)
#rfc_pred_prob =RF_RS.predict_proba(x_test)


#ks
rfc_pred_proba= RF_RS.predict_proba(x_test)
rfc_pred_proba=pd.DataFrame(rfc_pred_proba)
rfc_pred_proba1=rfc_pred_proba[1]

y_test
y_test1=y_test.to_csv("y_test_churn",index=False)
y_test1=pd.read_csv("y_test_churn",header=None ,names=["a"])
Test_Data1 = pd.concat([y_test1,rfc_pred_proba1],axis =1)
Test_Data1.columns =["Dep_flag","Prob"]
Test_Data1.columns
Test_Data1['decile'] = pd.qcut(Test_Data1['Prob'],10,labels=['1','2','3','4','5','6','7','8','9','10'])
Test_Data1.head()

Test_Data1.columns = ['Event','Probability','Decile']
Test_Data1.head()

Test_Data1['NonEvent'] = 1-Test_Data1['Event']
Test_Data1.head()


df1 =pd.pivot_table(data=Test_Data1,index=['Decile'],values=['Event','NonEvent','Probability'],aggfunc={'Event':[np.sum],'NonEvent': [np.sum],'Probability' : [np.min,np.max]})
df1.head()
df1.reset_index()

df1.columns = ['Event_Count','NonEvent_Count','max_score','min_score']
df1['Total_Cust'] = df1['Event_Count']+df1['NonEvent_Count']
df1
#  Sort the min_score in descending order.

df2 = df1.sort_values(by='min_score',ascending=False)
df2

df2['Event_Rate'] = (df2['Event_Count'] / df2['Total_Cust']).apply('{0:.2%}'.format)
default_sum = df2['Event_Count'].sum()
nonEvent_sum = df2['NonEvent_Count'].sum()
df2['Event %'] = (df2['Event_Count']/default_sum).apply('{0:.2%}'.format)
df2['Non_Event %'] = (df2['NonEvent_Count']/nonEvent_sum).apply('{0:.2%}'.format)
df2


df2['ks_stats'] = np.round(((df2['Event_Count'] / df2['Event_Count'].sum()).cumsum() -(df2['NonEvent_Count'] / df2['NonEvent_Count'].sum()).cumsum()), 4) * 100
df2

flag = lambda x: '*****' if x == df2['ks_stats'].max() else ''
df2['max_ks'] = df2['ks_stats'].apply(flag)
df2
df2.to_csv("ks_test_churnnew.csv")