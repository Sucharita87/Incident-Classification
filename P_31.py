# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 00:40:59 2020

@author: SUCHARITA
"""
# XGBoost, Random Forest and KNN gave the maximum accuracy and model was deployed using XGBoost 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler  
from sklearn.preprocessing import StandardScaler
from statsmodels.graphics.gofplots import qqplot
df= pd.read_excel("F:\\ExcelR\\P_31\\Incidents_service.xlsx")
df.info() # bool(3), datetime64[ns](3), int64(3), object(16)
df.describe()
df.columns
df.shape #(141712, 25)
df.skew() 
# accepting range of skewness is +-0.5
# apart from"confimation check" (0.9, close to desired range) none values fit the desired range
# "active" negatively skewed, while "count_opening" "count_reassign" "count_updated" are highly positively skewd
df.kurtosis() 
# accepting range of kurtosis is +-3
#  "count_opening" "count_reassign" "count_updated" are highly positive and depicts absence of normal distribution
# "confirmation_check" "doc_knowledge and active" are in range

# dataset is not normal

# understanding data distribution for various attributes
df['ID'].value_counts() # 24918
df['ID_status'].value_counts() 
df["impact"].value_counts() # high: 3491, medium: 134335, low: 3886
df['active'].value_counts() # true: 116726, false: 24986
df["location"].value_counts() # 224 different locations
df['ID_caller'].value_counts() # 5244 different ID s
df['opened_by'].value_counts() # 207 different values
df['Created_by'].value_counts() # 185 different values
df['updated_by'].value_counts() # 846 different values
df['notify'].value_counts() # email: 119, do not notify: 141593
df['type_contact'].value_counts() # phn: 140462, self service: 995, email: 220, ivr: 18, direct opening: 17
df['Doc_knowledge'].value_counts() # false: 116349, true: 25363
df['category_ID'].value_counts()  # multiple classes
df['user_symptom'].value_counts()  # 525 types
df['Support_group'].value_counts() # 78 types
df['support_incharge'].value_counts()# 234
df['support_incharge'].value_counts().plot(kind="pie")  
df['confirmation_check'].value_counts()  # false: 100740, true:40972
df['problem_id'].value_counts() # 253 types
df['change request'].value_counts() # 182 types
df['updated_at'].value_counts().plot(kind="pie") 

# replacing junk values
df["problem_id"].replace({"?": "NA"}, inplace=True)
df['problem_id'].value_counts() # NA: 139417
df["change request"].replace({"?": "NA"}, inplace=True)
df['change request'].value_counts() # NA: 140721

# understanding categorical and numerical values
categorical = [var for var in df.columns if df[var].dtype=='O']
print('There are {} categorical variables\n'.format(len(categorical)))
print('The categorical variables are :\n\n', categorical)
df[categorical].isnull().sum() # shows no null values

for var in categorical:  # view frequency of each categorical variable type, junk values are there
    print(df[var].value_counts()) # notify(139417), problem_id(140721) have almost 98% missing values
    
for var in categorical: 
    print(df[var].value_counts()/np.float(len(df)))    
# certain variables are having a major share in prediction
# gr 70 in "support_grp" account for 40.7%
# sym 491 in "user_symptom" account for 59.9%
# category 26,42,53,46,23 and 9  in "category_id" accounts for almost 50% 
# location 204,161,143 in "location" account for almost 57%
# phone under "type_contact" account for 99%
#updated_by 908, 44, 60 acount for 40%
# created by 10 account for 55%

for var in categorical:  # check different unique lebels for each
    print(var, 'contains ', len(df[var].unique()), ' labels')
    
numerical = [var for var in df.columns if df[var].dtype!='O']
print('There are {} numerical variables\n'.format(len(numerical)))
print('The numerical variables are :', numerical)
df[numerical].isnull().sum()

for var in numerical:  # view frequency of each numerical variable type, junk values are there
    print(df[var].value_counts())
    
for var in numerical: 
    print(df[var].value_counts()/np.float(len(df)))    # all proportions add upto 100%
# true in "confirmation check" account for 71 %
# false in "doc_knowledge" account for 82%
# 4/7/16 and 17/3/16 have almost 37% values under "created_at"
# 0,1,2,3,4 in count_updated account for 64%
# 0 in count_opening account for 98%
# 0,1,2 in count_reassign account for almost 85%
# true in "active" account for 82%


for var in numerical:  # check different unique lebels for each
    print(var, ' contains ', len(df[var].unique()), ' labels')

# encoding all categorical variables
# "ID_status" have -100, which needs to be converted into string value
df["ID_status"].replace({-100: "minus hundred"}, inplace=True)
df.ID_status.value_counts()

# creating new dataframe by dropping attributes with junk values
df= df.drop(['problem_id', 'change request', 'count_opening'],axis=1)
df.shape #(141712, 22)

# removing duplicate values
duplicate= df[df.duplicated()] 
df1= df.drop_duplicates() # there are no duplicate values
df1.shape
# check for outliers

df.boxplot() # certain outliers are present, "count_reassign, count_updated, count_opening"


# label encoding categorical variables

string_new= ['ID','ID_status','ID_caller','active','type_contact','impact','notify', 'opened_time', 'created_at', 'updated_at', 'Doc_knowledge', 'confirmation_check','opened_by', 'Created_by', 'updated_by', 'location', 'category_ID', 'user_symptom',  'Support_group', 'support_incharge']
number= preprocessing.LabelEncoder()
for i in string_new:
   df[i] = number.fit_transform(df[i])

# total dataset visualization in one plane
sns.boxplot(data=df["count_updated"],orient="h", palette="vlag") # outliers present
sns.boxplot(data=df["ID"],orient="h", palette="vlag") # no outlier
sns.boxplot(data=df["ID_status"],orient="h", palette="vlag") # no outlier
sns.boxplot(data=df["count_reassign"],orient="h", palette="vlag") # outliers present
df["count_updated"].describe()

# using zscore to identify outliers
from scipy import stats
z = np.abs(stats.zscore(df))
threshold = 3
print(np.where(z > 3))
print(z)
print(z[20][16]) # outlier at position (20,17) have a z -score value of 4.39
#The first array contains the list of row numbers 
#and second array respective column numbers, which mean z[55][1] have a Z-score higher than 3.
# impute outliers
Q1 = df.quantile(0.25)
Q3 =df.quantile(0.75)
IQR = Q3 - Q1
low= Q1 - 1.5*IQR
high = Q3 + 1.5*IQR
print(IQR) # gives IQR for all attributes

df["count_updated"].median() # 3
df["count_updated"].mode()
df["count_updated"].mean() # 5
df["count_reassign"].median()# 1
df["count_reassign"].mode()
df["count_reassign"].mean() # 1.1

# imputing each attribute with outlier one by one
df2= df
print("Change the outliers with median ",df2['count_updated'].median())
print("Change the outliers with median ",df2['count_reassign'].median())
df2.loc[(df2['count_updated'] < low) | (df2['count_updated'] > high)] = df2['count_updated'].median()
df2.loc[(df2['count_reassign'] < low) | (df2['count_reassign'] > high)] = df2['count_reassign'].median()
sns.boxplot(data= df2["count_updated"], orient = "h", palette="vlag")

# impute all attributes with outliers togther by creating anew dataframe and loop

df3= df.loc[:,["count_updated","count_reassign"]]
df3.describe

for col_name in df3.select_dtypes(include=np.number).columns:
    print(col_name)
    q1 = df3[col_name].quantile(0.25)
    q3 = df3[col_name].quantile(0.75)
    iqr = q3 - q1
    low = q1-1.5*iqr
    high = q3+1.5*iqr 
    print("Change the outliers with median ",df3[col_name].median())
    df3.loc[(df3[col_name] < low) | (df3[col_name] > high)] = df3[col_name].median()

sns.boxplot(data= df3["count_updated"], orient = "h", palette="vlag")
sns.boxplot(data= df3["count_reassign"], orient = "h", palette="vlag")

df4= df.drop(["count_updated","count_reassign"], axis= 1)
df4.shape
df_new= pd.concat([df4, df3],axis= 1)
df_new.shape
sns.boxplot(data=df_new["count_updated"],orient="h", palette="vlag")


df_new.corr()   
corr= df_new.corr()
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns,cmap="YlGnBu",annot=True, fmt='.1g')
# multiple corelations arer established

plt.rcParams["figure.figsize"] = (40,30) # for jupyter
g = sns.PairGrid(df_new)
g.map(plt.scatter)

df_new.describe() # range of values and mean is varying to a huge extent

# we have to check whether the data is (normal)fitting the  gaussian bell curve or not
# visual histogram plot 
df_new.hist()
plt.show()# data is not normal

# again study the data symmetry using skewness and kurtosis of the encoded data
df_new.skew() # ideal range is +-0.5
df_new.kurtosis() # ideal range +-3


# qq plot for normality test

qqplot(df_new["updated_at"], line='s')
plt.show()
qqplot(df_new["active"], line='s')
plt.show()
# data is not normal

x= df_new.drop(["impact"],axis= 1)
x.info()
x.shape

y= df_new["impact"] # series
y= pd.DataFrame(y)
y["impact"].value_counts() # 1(medium): 134335, 2(low): 3886, 0(high): 3491
# define the min max scaler
scaler= MinMaxScaler()
d_scale= scaler.fit_transform(x) # an array is getting created 
d_scale.shape #(141712, 21)
#convert array to dataframe
d_scale= pd.DataFrame(d_scale, columns= x.columns)
d_scale.describe()
d_scale.skew()
d_scale.kurtosis()
d_scale.hist()
plt.show()

z_scale = np.abs(stats.zscore(d_scale))
threshold = 3
print(np.where(z_scale > 3))
print(z_scale)
qqplot(d_scale["updated_at"], line='s')
plt.show()


std=StandardScaler()
d_std= std.fit_transform(x)# an array is getting created
stats.zscore(d_std)
#convert array to dataframe
d_std= pd.DataFrame(d_std, columns= x.columns)
d_std.skew()
d_std.kurtosis()
d_std.hist()
plt.show()

# y = impact (target), x= d_scale (predictors)

from imblearn.under_sampling import NearMiss

# define the undersampling method
undersample = NearMiss(version=1, n_neighbors=3)
# transform the dataset
x_re, y_re = undersample.fit_resample(d_scale, y)
x_re.shape # (10473, 21)
y_re.shape # (10473,1)
y_re["impact"].value_counts() # 1(medium): 3491, 2(low): 3491, 0(high): 3491
x_re["notify"].value_counts() # all values are "0", lets drop the column
x_re = x_re.drop(['notify'], axis= 1)
y_re["impact"].value_counts().plot(kind="pie")


# use x_re and y_re for test train split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_re, y_re, test_size=.2, random_state=42)
x_col = x_train.columns
print(x_col)
x_train.corr()   
corr= x_train.corr()
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns,cmap="YlGnBu",annot=True, fmt='.1g')
# multiple corelations are established

# Selecting best features===================================================================

# use SelectKBest and chi2, preferd when feature names are not imp
from sklearn.feature_selection import SelectKBest, chi2
bestfeatures= SelectKBest(score_func=chi2, k=5,)
fit= bestfeatures.fit(x_train, y_train)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(x_train.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(20,'Score'))  #print 20 best features

# use Lasso, get the feature names=============================================================
from sklearn.linear_model import LassoCV
clf = LassoCV().fit(x_train, y_train.values.ravel())
importance = np.abs(clf.coef_)
print(importance)
idx_five = importance.argsort()[-5] # we want to set the threshold slightly above the third highest coef_ calculated by LassoCV() from our data.
threshold = importance[idx_five] + 0.01
idx_features = (-importance).argsort()[:15]
name_features = np.array(x_col)[idx_features]
print('Selected features: {}'.format(name_features))


# decisison tree classifier feature importance=====================================================
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(x_train, y_train.values.ravel())
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=x_train.columns)
feat_importances.nlargest(15).plot(kind='barh')
plt.show()

# Random forest feature selection===============================================================
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(x_train, y_train.values.ravel())
importance = rf.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.show()

# XGBoost feature selection====================================================================
from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(x_train, y_train.values.ravel())
importance = xgb.feature_importances_
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance, same as RF
plt.bar([x for x in range(len(importance))], importance)
plt.show()
from xgboost import plot_importance
plot_importance(xgb)

#=========================================================================================================
#MODEL BUILDING
#============================================================================================================

predictor = x_train.loc[:,['category_ID','opened_time','ID_status' ,'updated_at',
 'Support_group', 'support_incharge', 'location' , 'count_updated',
 'user_symptom','ID_caller']]

pred_test= x_test.loc[:,['category_ID','opened_time','ID_status' ,'updated_at',
 'Support_group', 'support_incharge', 'location' , 'count_updated',
 'user_symptom','ID_caller']]


# Building logistic regression model-----------------------------------

from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
logcla = LogisticRegression()
logcla.fit(predictor, y_train.values.ravel())
logcla.coef_ # coefficients of features 
logcla.predict_proba (predictor) # Probability values 
y_pred = logcla.predict(predictor)
y_pred= pd.DataFrame(y_pred, columns= y_train.columns)
y_prob = pd.DataFrame(logcla.predict_proba(predictor.iloc[: ,:]))
new_df = pd.concat([predictor,y_prob],axis=1)

confusion_matrix = confusion_matrix(y_train,y_pred)
print (confusion_matrix)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logcla.score(pred_test, y_test)))
# 65%

print(classification_report(y_train, y_pred))
# 65%

# Building Naive Bayes Model---------------------------------------------
from sklearn.naive_bayes import GaussianNB 
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
gnb = GaussianNB()
mnb = MultinomialNB()

pred_gnb = gnb.fit(predictor,y_train.values.ravel()).predict(pred_test)
confusion_matrix(y_test,pred_gnb)
print ("Accuracy",(341+553+400)/(341+162+186+85+553+67+145+156+400)) # 61.7%
pd.crosstab(y_test.values.flatten(),pred_gnb) # confusion matrix 
np.mean(pred_gnb==y_test.values.flatten()) # 61.76%


pred_mnb = mnb.fit(predictor,y_train.values.ravel()).predict(pred_test)
confusion_matrix(y_test,pred_mnb)
print("Accuracy",(339+536+347)/(339+182+168+111+536+58+174+180+347))  # 58.3%
pd.crosstab(y_test.values.flatten(),pred_mnb) # confusion matrix  
np.mean(pred_mnb==y_test.values.flatten()) # 58.3%


# Building Decision Tree Model-------------------------------------
from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
dtc = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=0)
plt.figure(figsize=(12,8))
tree.plot_tree(dtc.fit(predictor, y_train),filled=True)
plt.show()
dtc.fit(predictor, y_train)
train_pred =dtc.predict(predictor)
train_accuracy = accuracy_score(y_train,train_pred)
train_accuracy # 73.4%
train_confusion = confusion_matrix(y_train,train_pred)
train_confusion
np.mean(train_pred ==y_train.values.flatten()) # 73.4%

##Prediction on test data
test_pred = dtc.predict(pred_test)
test_accuracy = accuracy_score(y_test,test_pred)
test_accuracy # 73.84%
test_confusion = confusion_matrix(y_test,test_pred)
test_confusion
np.mean(test_pred ==y_test.values.flatten()) # 74.13%

print(classification_report(y_test, test_pred)) # accuracy = 74% , almost similar train and test result

# gini index
# model building 
dtc2 = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=0)
plt.figure(figsize=(12,8))
tree.plot_tree(dtc2.fit(predictor, y_train),filled=True)
plt.show()
dtc2.fit(predictor, y_train)
train_pred2 =dtc2.predict(predictor)
train_pred2_accuracy = accuracy_score(y_train,train_pred2)
train_pred2_accuracy # 73.02%
train_pred2_confusion = confusion_matrix(y_train,train_pred2)
train_pred2_confusion
np.mean(train_pred2 ==y_train.values.flatten()) # 73.02%


##Prediction on test data
test_pred2 = dtc2.predict(pred_test)
test_pred2_accuracy = accuracy_score(y_test,test_pred2)
test_pred2_accuracy # 73.93%
test_pred2_confusion = confusion_matrix(y_test,test_pred2)
test_pred2_confusion
np.mean(test_pred2 ==y_test.values.flatten()) # 73.6%

print(classification_report(y_test, test_pred2)) # accuracy = 74% , almost similar train and test result

# Building Random Forest model---------------------------------------

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs=5,oob_score=True,n_estimators=100,criterion="entropy")
rf.fit(predictor, y_train.values.ravel()) # Fitting RandomForestClassifier model from sklearn.ensemble 
rf.estimators_ 
rf.classes_ # array[0,1,2]
rf.n_classes_ # 3 levels
rf.n_features_  # Number of input features in model is 18
rf.n_outputs_ # Number of outputs when fit performed is 1
rf.oob_score_  # 91.89%
rf.predict(predictor)
len(rf.predict(predictor))
type(rf.predict(predictor))

from sklearn.model_selection import cross_validate
model = RandomForestClassifier(random_state=1)
cv = cross_validate(model, predictor, y_train, cv=5)
print(cv['test_score'])
print(cv['test_score'].mean())

pred =rf.predict(predictor)
y_pred = pd.DataFrame(pred) # convert to dataframe so that it can be added in 
type(y_pred)
confusion_matrix(y_train,y_pred) # Confusion matrix
print(classification_report(y_train, y_pred)) # accuracy 100%

y_test_pred = rf.predict(pred_test)
confusion_matrix(y_test,y_test_pred)
print(classification_report(y_test,y_test_pred)) # accuracy 92%


#BUilding model with Artificial Neural Network-------------------------

# Importing necessary models for implementation of ANN
from keras.models import Sequential
from keras.layers import Dense #, Activation,Layer,Lambda
model = Sequential()
model.add(Dense(10, input_dim=18, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# compile keras model, classification model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit keras model
model.fit(np.array(predictor),np.array(y_train), epochs=10, batch_size=10)
model_loss = pd.DataFrame(model.history.history)
model_loss.plot()

# evaluate keras mdoel accuracy
_,accuracy = model.evaluate(pred_dt, y_train)
print('Accuracy: %.2f' % (accuracy*100)) 
train_pred = model.predict_classes(predictor)
print(classification_report(y_train,train_pred)) 
print(confusion_matrix(y_train,train_pred))
#[[286   1]

#Building KNN model -----------------------------------------------
from sklearn.neighbors import KNeighborsClassifier as KNC
knn = KNC(n_neighbors=7)
knn.fit(predictor,y_train.values.ravel())
y_train_pred = knn.predict(predictor)
knn.score(predictor,y_train) # 81.86%

# check prediction accuracy of train data and classification error
print(confusion_matrix(y_train, y_train_pred))
print(classification_report(y_train, y_train_pred)) # accuracy = 82%

# fit test data
knn.score(pred_test,y_test) # 76.75%
y_test_pred = knn.predict(pred_test)
print(confusion_matrix(y_test, y_test_pred)) 
print(classification_report(y_test, y_test_pred))  # accuracy = 77%

# for 3 neighbors
knn = KNC(n_neighbors= 3)
knn.fit(predictor, y_train.values.ravel())
y_train_pred= knn.predict(predictor)
knn.score(predictor, y_train) # 89.4%
print(confusion_matrix(y_train, y_train_pred))
print(classification_report(y_train, y_train_pred)) # accuracy = 91%

# fit test data
y_test_pred = knn.predict(pred_test)
print(confusion_matrix(y_test, y_test_pred)) 
print(classification_report(y_test, y_test_pred))  # accuracy = 80%


# getting optimal "k" value
a= []
for i in range (2,50,3):
    knn = KNC(n_neighbors=i)
    knn.fit(predictor, y_train.values.ravel())
    train_acc = knn.score(predictor, y_train)
    test_acc = knn.score(pred_test, y_test)
    a.append([train_acc,test_acc])

plt.plot(np.arange(2,50,3),[i[0] for i in a],"bo-")
plt.plot(np.arange(2,50,3), [i[1] for i in a], "rs-")
plt.legend(["train","test"])

# Building model with XGBoost---------------------------------

xgb1 = XGBClassifier(n_estimators=2000,learning_rate=0.3)
xgb1.fit(predictor,y_train.values.ravel())
train_pred_xgb = xgb1.predict(predictor)
print(confusion_matrix(y_train, train_pred_xgb))
print(classification_report(y_train, train_pred_xgb)) # 100%
y_train= pd.DataFrame(y_train)

test_pred_xgb = xgb1.predict(pred_test)
print(confusion_matrix(y_test, test_pred_xgb ))
print(classification_report(y_test, test_pred_xgb )) # 95%
# Variable importance plot 
from xgboost import plot_importance
plot_importance(xgb1)


#Building model with  SVM-----------------------------
from sklearn.svm import SVC
#kernel = linear
# create model
model_linear= SVC(kernel = "linear")
model_linear.fit(predictor, y_train.values.ravel())

#test model 
pred_test_linear = model_linear.predict(pred_test)
y_test= y_test.to_numpy()
np.mean(pred_test_linear == y_test)  # 33.36%

# kernel = poly
# create model
model_poly= SVC(kernel = "poly")
model_poly.fit(predictor, y_train.values.ravel())

#test model 
pred_test_poly = model_poly.predict(pred_test)
np.mean(pred_test_poly == y_test)  # 33.35%

# kernel = rbf
# create model
model_rbf= SVC(kernel = "rbf")
model_rbf.fit(predictor, y_train.values.ravel())

#test model 
pred_test_rbf = model_rbf.predict(pred_test)
np.mean(pred_test_rbf == y_test)  # 33.35%

# kernel = sigmoid
# create model
model_sigmoid= SVC(kernel = "sigmoid")
model_sigmoid.fit(predictor, y_train.values.ravel())

#test model 
pred_test_sigmoid = model_sigmoid.predict(pred_test)
np.mean(pred_test_sigmoid == y_test)  # 33.28%


























