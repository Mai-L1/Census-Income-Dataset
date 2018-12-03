
# coding: utf-8

# # CMKE 136 Capstone Project Fall 2018

# ## Import Packages

# In[151]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from datetime import datetime
import scipy as sp
from scipy import stats
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler


# ## Initial Analysis

# In[152]:


#Import data and create dataframe
df_all=pd.read_csv('all.csv')


# In[153]:


#Display dataframe
df_all.head(n=10)


# In[154]:


#Get information about each column including data types
df_all.info()


# In[155]:


#Replace the class attribute with numerice values: <=50K with 0 and >50K with 1
replacements = {
  ' <=50K':0,
  ' >50K':1,
  ' <=50K.':0,
  ' >50K.':1,  
}
df_all['Income'].replace(replacements, inplace=True)


# In[156]:


#Replace ? (missing values) with NaN
df_all2=df_all.replace(" ?",np.NaN)


# In[157]:


#Display the new dataframe
df_all2.head(n=10)


# In[158]:


#Count the number of records for each attribute
df_all2.count()


# ### Univariate Analysis

# In[159]:


#Create a dataframe with only the numeric variables
numeric_all = pd.DataFrame(df_all2,columns=['Age','Fnlwgt','EducationNum','CapitalGain','CapitalLoss','HoursPerWeek'])


# In[160]:


#Create a five number summary of the numeric attributes
numeric_all.describe()


# In[161]:


#Boxplot of numeric values- Age
plt.boxplot(df_all2['Age'])
print('Kurtosis',df_all2['Age'].kurtosis())
sp.stats.normaltest(df_all2.Age)


# In[162]:


#Histogram of Age
df_all2['Age'].plot.hist(bins=20)


# In[163]:


#Boxplot of numeric values- Fnlwgt
plt.boxplot(df_all2['Fnlwgt'])
print('Kurtosis',df_all2['Fnlwgt'].kurtosis())
sp.stats.normaltest(df_all2.Fnlwgt)


# In[164]:


#Histogram of Fnlwgt
df_all2['Fnlwgt'].plot.hist(bins=20)


# In[165]:


#Boxplot of numeric values- EducationNum
plt.boxplot(df_all2['EducationNum'])
print('Kurtosis',df_all2['EducationNum'].kurtosis())
sp.stats.normaltest(df_all2.EducationNum)


# In[166]:


#Histogram of EducationNum
df_all2['EducationNum'].plot.hist(bins=20)


# In[167]:


#Boxplot of numeric values- CapitalGain
plt.boxplot(df_all2['CapitalGain'])
print('Kurtosis',df_all2['CapitalGain'].kurtosis())
sp.stats.normaltest(df_all2.CapitalGain)


# In[168]:


#Histogram of CapitalGain
df_all2['CapitalGain'].plot.hist(bins=20)


# In[169]:


#Boxplot of numeric values- CapitalLoss
plt.boxplot(df_all2['CapitalLoss'])
print('Kurtosis',df_all2['CapitalLoss'].kurtosis())
sp.stats.normaltest(df_all2.CapitalLoss)


# In[170]:


#Histogram of CapitalLoss
df_all2['CapitalLoss'].plot.hist(bins=20)


# In[171]:


#Boxplot of numeric values- HoursPerWeek
plt.boxplot(df_all2['HoursPerWeek'])
print('Kurtosis',df_all2['HoursPerWeek'].kurtosis())
sp.stats.normaltest(df_all2.HoursPerWeek)


# In[172]:


#Histogram of HoursPerWeek
df_all2['HoursPerWeek'].plot.hist(bins=20)


# In[173]:


#Histogram of class attribute Income
df_all2['Income'].value_counts().plot(kind='bar')
print('Ratio of income >50K: ', df_all2['Income'].sum()/df_all2['Income'].count())
print('Ratio of income <=50K: ', 1-df_all2['Income'].sum()/df_all2['Income'].count())


# In[174]:


#As seen from the Income histogram, the data is unbalanced in favour of income <50K


# In[175]:


#Histogram of categorical variables - Workclass
df_all2['Workclass'].value_counts().plot(kind='bar')


# In[176]:


#Histogram of categorical variables - Education
df_all2['Education'].value_counts().plot(kind='bar')


# In[177]:


#Histogram of categorical variables - Marital-Status
df_all2['MaritalStatus'].value_counts().plot(kind='bar')


# In[178]:


#Histogram of categorical variables - Occupation
df_all2['Occupation'].value_counts().plot(kind='bar')


# In[179]:


#Histogram of categorical variables - Relationship
df_all2['Relationship'].value_counts().plot(kind='bar')


# In[180]:


#Histogram of categorical variables - Race
df_all2['Race'].value_counts().plot(kind='bar')


# In[181]:


#Histogram of categorical variables - Sex
df_all2['Sex'].value_counts().plot(kind='bar')


# In[182]:


#Histogram of categorical variables - Native Country
df_all2['Country'].value_counts().plot(kind='bar')


# ### Bivariate Analysis

# In[183]:


#Graph the distribution of each of the variables and plot each of the variables against the class attribute
g = sns.pairplot(df_all2, kind="reg")


# In[184]:


#Create correlation matrix of the numeric variables
corr = df_all2.corr(method='pearson')
corr.style.background_gradient().set_precision(2)


# In[185]:


#There is a weak correlation between Income & Age, Income & Education-Num, Income & Capital-Gain and Income & Hours-Per-Week


# In[186]:


#Plotting each of the independent variables against the class attribute: Workclass vs Income
df_all2.groupby(['Workclass', 'Income']).size().unstack().plot(kind='bar', stacked=True)


# In[187]:


#Occupation vs Income
df_all2.groupby(['Occupation', 'Income']).size().unstack().plot(kind='bar', stacked=True)


# In[188]:


#Relationship vs Income
df_all2.groupby(['Relationship', 'Income']).size().unstack().plot(kind='bar', stacked=True)


# In[189]:


#Sex vs Income
df_all2.groupby(['Sex', 'Income']).size().unstack().plot(kind='bar', stacked=True)


# In[190]:


#Race vs Income
df_all2.groupby(['Race', 'Income']).size().unstack().plot(kind='bar', stacked=True)


# In[191]:


#Country vs Income
df_all2.groupby(['Country', 'Income']).size().unstack().plot(kind='bar', stacked=True)


# In[192]:


#Marital-Status vs Income
df_all2.groupby(['MaritalStatus', 'Income']).size().unstack().plot(kind='bar', stacked=True)


# ### Further Preprocessing

# In[193]:


#Count the number of missing values for each attribute
df_all2.isnull().sum()


# In[194]:


#Identify duplicate rows
df_all2[df_all2.duplicated(keep='first')]


# In[195]:


#Count number of duplicate rows/values
df_all2[df_all2.duplicated(keep='first')].count()


# In[196]:


#There are 52 duplicate rows in the dataset


# In[197]:


#Drop duplicated rows
df_all3=df_all2.drop_duplicates(keep='first', inplace=False)


# In[198]:


#Check that all duplicate rows have been removed
df_all3[df_all3.duplicated(keep='first')].count()


# In[199]:


#Checking how many missing values are left once duplicated values are removed
df_all3.isnull().sum()


# In[200]:


#Drop rows where 'Occupation' is missing
df_all4=df_all3.dropna(axis=0, how='any', thresh=None, subset=['Occupation'], inplace=False)


# In[201]:


#Checking how many missing values are left once rows missing 'Occupation' are dropped
df_all4.isnull().sum()


# In[202]:


#Removing rows where 'Occupation' is missing, removes all rows where 'Workclass' is missing


# In[203]:


#Replace missing 'Country' values with the mode 'United-States
df_all5=df_all4.fillna('United States')


# In[204]:


#Check that all missing values have been removed
df_all5.isnull().sum()


# In[205]:


#Standardize the numeric columns as the are all non-normal
df_all6=df_all5.copy()
column_names_to_standardize = ['Age','Fnlwgt','EducationNum','CapitalGain','CapitalLoss','HoursPerWeek']
x = df_all6[column_names_to_standardize].values
scaler= StandardScaler()
x_scaled = scaler.fit_transform(x)
df_temp = pd.DataFrame(x_scaled, columns=column_names_to_standardize, index = df_all6.index)
df_all6[column_names_to_standardize] = df_temp
df_all6


# In[206]:


df_all6['Age'].plot.hist(bins=20)


# In[207]:


df_all6['Fnlwgt'].plot.hist(bins=20)


# In[208]:


df_all6['EducationNum'].plot.hist(bins=20)


# In[209]:


df_all6['CapitalGain'].plot.hist(bins=20)


# In[210]:


df_all6['CapitalLoss'].plot.hist(bins=20)


# In[211]:


df_all6['HoursPerWeek'].plot.hist(bins=20)


# ## Exploratory Analysis

# In[212]:


#Drop the Education column as Education-Num is the exact numeric version of this attribute
df_all7=df_all6.drop('Education',axis=1)


# In[213]:


#Drop the Fnlwgt column as it does not add much predictive value. Fnlwgt = of people the census takers believe that observation represents
df_all8=df_all7.drop('Fnlwgt',axis=1)


# In[214]:


#Applying one-hot encoding the get dummy variables for each of the categorical variables
df_copy=df_all8.copy()

df_one_hot=pd.get_dummies(df_copy, columns=["Workclass","MaritalStatus","Occupation","Relationship","Race","Sex","Country"], prefix=["Workclass","MaritalStatus","Occupation","Relationship","Race","Sex","Country"])


# In[215]:


#Examening the new dataframe and the total number of columns
df_one_hot.info()


# ### Feature Selection

# In[216]:


#With the help of the decision tree classifier using Entropy index to rank each features based on importance
array = df_one_hot.values
X=array[:,1:]
Y=array[:,0]
model=DecisionTreeClassifier(criterion='entropy')
model.fit(X,Y)


# In[217]:


#Plotting all the features and their importance
pd.Series(model.feature_importances_, index=df_one_hot.columns[1:]).plot.bar(color='steelblue', figsize=(12, 6))


# In[218]:


#To increase readability, plotting the top 22 most important features
feat_importances =pd.Series(model.feature_importances_, index=df_one_hot.columns[1:])
feat_importances.nlargest(22).plot(kind='barh')


# In[219]:


#Listing all the features and their importance in descending order
pd.Series(model.feature_importances_, index=df_one_hot.columns[1:]).sort_values(ascending=False).head(n=30)


# In[220]:


#Calculate the mean of the feature importances
pd.Series(model.feature_importances_, index=df_one_hot.columns[1:]).mean()


# In[221]:


#Calculate the median of the feature importances
pd.Series(model.feature_importances_, index=df_one_hot.columns[1:]).median()


# In[222]:


#Calculate the 75th percentile of the feature importances
np.percentile(pd.Series(model.feature_importances_, index=df_one_hot.columns[1:]),75)


# In[223]:


#Create a new dataframe with features where the feature importance are all in the 75th percentile
df_clean=df_one_hot[['Income','MaritalStatus_ Married-civ-spouse','Age','EducationNum','CapitalGain','HoursPerWeek','CapitalLoss','Workclass_ Private','Occupation_ Exec-managerial','Workclass_ Self-emp-not-inc','Occupation_ Prof-specialty','Occupation_ Craft-repair','Country_ United-States','Occupation_ Sales','Race_ White','Workclass_ Local-gov','Workclass_ Self-emp-inc','Race_ Black','Occupation_ Adm-clerical','Occupation_ Transport-moving','Occupation_ Other-service','Workclass_ State-gov','Sex_ Female']]


# In[224]:


#Checking that the desired features were dropped from the dataframe
df_clean.info()


# ## Data Modelling

# In[225]:


#Split the data into two; one dataframe for the class attribute and one dataframe for the rest of the attributes
dataset=df_clean.values
X=dataset[:,1:]
y=dataset[:,0]


# ### Classification Approach A: Splitting between train and test using an 80:20 split

# In[226]:


#Split to dataset into train and test set to prepare for modelling
#80% of the data will be used for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


# In[227]:


#Classifier 1a) Decision Tree
start_time = datetime.now()

classifier_dt = DecisionTreeClassifier(criterion='entropy')  

#Train algorithm on training data
classifier_dt.fit(X_train, y_train)

#Make predictions on train data
y_predict_train = classifier_dt.predict(X_train)

#Make predictions on test data
y_pred = classifier_dt.predict(X_test)

print('Train Accuracy Score: ', accuracy_score(y_train, y_predict_train))
print('Test Accuracy Score: ', accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))


# In[255]:


#Classifier 2a) KNN
start_time = datetime.now()

classifier_knn = KNeighborsClassifier(n_neighbors=9)
classifier_knn.fit(X_train, y_train)
y_predict_train = classifier_knn.predict(X_train)
y_pred = classifier_knn.predict(X_test)
print('Train Accuracy Score: ', accuracy_score(y_train, y_predict_train))
print('Test Accuracy Score: ', accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))


# In[148]:


#Classifier 3a) Naive Bayes
start_time = datetime.now()

classifier_nb = GaussianNB()
classifier_nb.fit(X_train, y_train)
y_predict_train = classifier_nb.predict(X_train)
y_pred = classifier_nb.predict(X_test)
print('Train Accuracy Score: ', accuracy_score(y_train, y_predict_train))
print('Test Accuracy Score: ', accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))


# In[149]:


#Classifier 4a) SVM
start_time = datetime.now()

classifier_svm = SVC(kernel='sigmoid') 
classifier_svm.fit(X_train, y_train)
y_predict_train = classifier_svm.predict(X_train)
y_pred = classifier_svm.predict(X_test)
print('Train Accuracy Score: ', accuracy_score(y_train, y_predict_train))
print('Test Accuracy Score: ', accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))


# In[258]:


#Ensemble Method a - Random Forrest Classifier
start_time = datetime.now()

num_trees = 60
max_features = 22
classifier_rf = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
classifier_rf.fit(X_train, y_train)
y_predict_train = classifier_rf.predict(X_train)
y_pred = classifier_rf.predict(X_test)
print('Train Accuracy Score: ', accuracy_score(y_train, y_predict_train))
print('Test Accuracy Score: ', accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))


# ### Classification Approach B: Splitting between train and test using three fold cross-validation

# In[228]:


#Split the data using three fold cross-validation
seed=7
threefold= model_selection.KFold(n_splits=3, random_state=seed)


# In[229]:


#Classifier 1b) Decision Tree
start_time = datetime.now()

model_dt=tree.DecisionTreeClassifier(criterion='entropy')
results_dt = model_selection.cross_val_score(model_dt, X, y, cv=threefold)
print('Accuracy: ',results_dt.mean())


end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))


# In[281]:


#Classifier 2b) KNN
start_time = datetime.now()

model_knn = KNeighborsClassifier(n_neighbors=9)
results_knn = model_selection.cross_val_score(model_knn, X, y, cv=threefold)
print('Accuracy: ',results_knn.mean())

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))


# In[83]:


#Classifier 3b) Naive Bayes
start_time = datetime.now()

model_nb = GaussianNB()
results_nb= model_selection.cross_val_score(model_nb, X, y, cv=threefold)
print('Accuracy: {}',results_nb.mean())

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))


# In[84]:


#Classifier 4b) SVM
start_time = datetime.now()

model_svm=SVC(kernel='sigmoid')
results_svm = model_selection.cross_val_score(model_svm, X, y, cv=threefold)
print('Accuracy: ',results_svm.mean())

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))


# In[85]:


#Ensemble Method b - Random Forest Classifier
start_time = datetime.now()

model_rf = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
results_rf = model_selection.cross_val_score(model_rf, X, y, cv=threefold)
print('Accuracy: ',results_rf.mean())

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))


# ### Classification Approach C: Splitting between train and test using five fold cross-validation

# In[86]:


#Split the data using three fold cross-validation
seed=7
fivefold= model_selection.KFold(n_splits=5, random_state=seed)


# In[230]:


#Classifier 1c) Decision Tree
start_time = datetime.now()

model_dt=tree.DecisionTreeClassifier(criterion='entropy')
results_dt = model_selection.cross_val_score(model_dt, X, y, cv=fivefold)
print('Accuracy: ',results_dt.mean())


end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))


# In[282]:


#Classifier 2c) KNN
start_time = datetime.now()

model_knn = KNeighborsClassifier(n_neighbors=9)
results_knn = model_selection.cross_val_score(model_knn, X, y, cv=fivefold)
print('Accuracy: ',results_knn.mean())

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))


# In[89]:


#Classifier 3c) Naive Bayes
start_time = datetime.now()

model_nb = GaussianNB()
results_nb= model_selection.cross_val_score(model_nb, X, y, cv=fivefold)
print('Accuracy: ',results_nb.mean())

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))


# In[90]:


#Classifier 4c) SVM
start_time = datetime.now()

model_svm=SVC(kernel='sigmoid')
results_svm = model_selection.cross_val_score(model_svm, X, y, cv=fivefold)
print('Accuracy: ',results_svm.mean())

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))


# In[91]:


#Ensemble Method c - Random Forest Classifier
start_time = datetime.now()

model_rf = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
results_rf = model_selection.cross_val_score(model_rf, X, y, cv=fivefold)
print('Accuracy: ',results_rf.mean())

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))


# ### Classification Approach D: Splitting between train and test using seven fold cross-validation

# In[231]:


#Split the data using three fold cross-validation
seed=7
sevenfold= model_selection.KFold(n_splits=7, random_state=seed)


# In[232]:


#Classifier 1d) Decision Tree
start_time = datetime.now()

model_dt=tree.DecisionTreeClassifier(criterion='entropy')
results_dt = model_selection.cross_val_score(model_dt, X, y, cv=sevenfold)
print('Accuracy: ', results_dt.mean())


end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))


# In[283]:


#Classifier 2d) KNN
start_time = datetime.now()

model_knn = KNeighborsClassifier(n_neighbors=9)
results_knn = model_selection.cross_val_score(model_knn, X, y, cv=sevenfold)
print('Accuracy: ', results_knn.mean())

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))


# In[95]:


#Classifier 3d) Naive Bayes
start_time = datetime.now()

model_nb = GaussianNB()
results_nb= model_selection.cross_val_score(model_nb, X, y, cv=sevenfold)
print('Accuracy: ', results_nb.mean())

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))


# In[96]:


#Classifier 4d) SVM
start_time = datetime.now()

model_svm=SVC(kernel='sigmoid')
results_svm = model_selection.cross_val_score(model_svm, X, y, cv=sevenfold)
print('Accuracy: ',results_svm.mean())

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))


# In[97]:


#Ensemble Method d - Random Forest Classifier
start_time = datetime.now()

model_rf = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
results_rf = model_selection.cross_val_score(model_rf, X, y, cv=sevenfold)
print('Accuracy: ', results_rf.mean())

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))


# ### Classification Approach E: Splitting between train and test using ten fold cross-validation

# In[233]:


#Split the data using three fold cross-validation
seed=7
tenfold= model_selection.KFold(n_splits=10, random_state=seed)


# In[234]:


#Classifier 1e) Decision Tree
start_time = datetime.now()

model_dt=tree.DecisionTreeClassifier(criterion='entropy')
results_dt = model_selection.cross_val_score(model_dt, X, y, cv=tenfold)
print('Accuracy: ', results_dt.mean())


end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))


# In[284]:


#Classifier 2e) KNN
start_time = datetime.now()

model_knn = KNeighborsClassifier(n_neighbors=9)
results_knn = model_selection.cross_val_score(model_knn, X, y, cv=tenfold)
print('Accuracy: ', results_knn.mean())

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))


# In[101]:


#Classifier 3e) Naive Bayes
start_time = datetime.now()

model_nb = GaussianNB()
results_nb= model_selection.cross_val_score(model_nb, X, y, cv=tenfold)
print('Accurcy: ', results_nb.mean())

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))


# In[102]:


#Classifier 4e) SVM
start_time = datetime.now()

model_svm=SVC(kernel='sigmoid')
results_svm = model_selection.cross_val_score(model_svm, X, y, cv=tenfold)
print('Accuracy: ', results_svm.mean())

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))


# In[103]:


#Ensemble Method e - Random Forest Classifier
start_time = datetime.now()

model_rf = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
results_rf = model_selection.cross_val_score(model_rf, X, y, cv=tenfold)
print('Accuracy: ', results_rf.mean())

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))


# ### Classification Approach F: Balancing the dataset by undersampling the majority class and runninng the classifiers using ten fold cross-validation. Sampling without replacement

# #### Creating a balanced dataframe by undersampling the majority class (income >50K) without replacement

# In[238]:


df_clean['Income'].value_counts().plot(kind='bar')


# In[239]:


print('Ratio of income >50K: ', df_clean['Income'].sum()/df_clean['Income'].count())
print('Ratio of income <=50K: ', 1-df_clean['Income'].sum()/df_clean['Income'].count())


# In[240]:


#Find number of records where income >50K
low_income = len(df_clean[df_clean['Income'] == 1])
low_income


# In[241]:


#Get indices of income <=50K
low_income_indices = df_clean[df_clean.Income == 0].index


# In[242]:


#Random sample low income indices
random_indices = np.random.choice(low_income_indices,low_income, replace=False)


# In[243]:


#Get indices of income <=50K
high_income_indices = df_clean[df_clean.Income == 1].index


# In[244]:


#Concatenate high income indices with low income ones
under_sample_indices = np.concatenate([high_income_indices,random_indices])


# In[245]:


#Get balanced dataframe
under_sample = df_clean.loc[under_sample_indices]


# In[246]:


under_sample['Income'].value_counts().plot(kind='bar')


# ### Modelling

# In[259]:


#Split the data into two; one dataframe for the class attribute and one dataframe for the rest of the attributes
dataset=under_sample.values
A=dataset[:,1:]
b=dataset[:,0]


# In[260]:


#Split to dataset into train and test set to prepare for modelling
#80% of the data will be used for training
A_train, A_test, b_train, b_test = train_test_split(A, b, test_size=0.20)


# In[262]:


#Classifier 1f) Decision Tree

start_time = datetime.now()
classifier_dt = DecisionTreeClassifier(criterion='entropy')  
classifier_dt.fit(A_train, b_train)
b_predict_train = classifier_dt.predict(A_train)
b_pred = classifier_dt.predict(A_test)

print('Train Accuracy Score: ', accuracy_score(b_train, b_predict_train))
print('Test Accuracy Score: ', accuracy_score(b_test, b_pred))
print(confusion_matrix(b_test, b_pred))  
print(classification_report(b_test, b_pred))

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))


# In[264]:


#Classifier 2f) KNN
start_time = datetime.now()

classifier_knn = KNeighborsClassifier(n_neighbors=9)
classifier_knn.fit(A_train, b_train)
b_predict_train = classifier_knn.predict(A_train)
b_pred = classifier_knn.predict(A_test)
print('Train Accuracy Score: ', accuracy_score(b_train, b_predict_train))
print('Test Accuracy Score: ', accuracy_score(b_test, b_pred))
print(confusion_matrix(b_test, b_pred))  
print(classification_report(b_test, b_pred))

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))


# In[265]:


#Classifier 3f) Naive Bayes
start_time = datetime.now()

classifier_nb = GaussianNB()
classifier_nb.fit(A_train, b_train)
b_predict_train = classifier_nb.predict(A_train)
b_pred = classifier_nb.predict(A_test)
print('Train Accuracy Score: ', accuracy_score(b_train, b_predict_train))
print('Test Accuracy Score: ', accuracy_score(b_test, b_pred))
print(confusion_matrix(b_test, b_pred))  
print(classification_report(b_test, b_pred))

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))


# In[266]:


#Classifier 4f) SVM
start_time = datetime.now()

classifier_svm = SVC(kernel='sigmoid') 
classifier_svm.fit(A_train, b_train)
b_predict_train = classifier_svm.predict(A_train)
b_pred = classifier_svm.predict(A_test)
print('Train Accuracy Score: ', accuracy_score(b_train, b_predict_train))
print('Test Accuracy Score: ', accuracy_score(b_test, b_pred))
print(confusion_matrix(b_test, b_pred))  
print(classification_report(b_test, b_pred))

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))


# In[267]:


#Ensemble Method f - Random Forrest Classifier
start_time = datetime.now()

num_trees = 60
max_features = 22
classifier_rf = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
classifier_rf.fit(A_train, b_train)
b_predict_train = classifier_rf.predict(A_train)
b_pred = classifier_rf.predict(A_test)
print('Train Accuracy Score: ', accuracy_score(b_train, b_predict_train))
print('Test Accuracy Score: ', accuracy_score(b_test, b_pred))
print(confusion_matrix(b_test, b_pred))  
print(classification_report(b_test, b_pred))

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))


# ### Classification Approach G: Balancing the dataset by undersampling the majority class and runninng the classifiers using ten fold cross-validation. Sampling with replacement

# #### Creating a balanced dataframe by undersampling the majority class (income >50K) with replacement

# In[235]:


#Find number of records where income >50K
low_income2 = len(df_clean[df_clean['Income'] == 1])

#Get indices of income <=50K
low_income_indices2 = df_clean[df_clean.Income == 0].index

#Random sample low income indices
random_indices2 = np.random.choice(low_income_indices2,low_income2, replace=True)

#Get indices of income <=50K
high_income_indices2 = df_clean[df_clean.Income == 1].index

#Concatenate high income indices with low income ones
under_sample_indices2 = np.concatenate([high_income_indices2,random_indices2])

#Get balanced dataframe
under_sample2 = df_clean.loc[under_sample_indices2]


# #### Modelling

# In[268]:


#Split the data into two; one dataframe for the class attribute and one dataframe for the rest of the attributes
dataset=under_sample2.values
C=dataset[:,1:]
d=dataset[:,0]


# In[269]:


#Split to dataset into train and test set to prepare for modelling
#80% of the data will be used for training
C_train, C_test, d_train, d_test = train_test_split(C, d, test_size=0.20)


# In[270]:


#Classifier 1g) Decision Tree

start_time = datetime.now()
classifier_dt = DecisionTreeClassifier(criterion='entropy')  
classifier_dt.fit(C_train, d_train)
d_predict_train = classifier_dt.predict(C_train)
d_pred = classifier_dt.predict(C_test)

print('Train Accuracy Score: ', accuracy_score(d_train, d_predict_train))
print('Test Accuracy Score: ', accuracy_score(d_test, d_pred))
print(confusion_matrix(d_test, d_pred))  
print(classification_report(d_test, d_pred))

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))


# In[271]:


#Classifier 2g) KNN
start_time = datetime.now()

classifier_knn = KNeighborsClassifier(n_neighbors=9)
classifier_knn.fit(C_train, d_train)
d_predict_train = classifier_knn.predict(C_train)
d_pred = classifier_knn.predict(C_test)
print('Train Accuracy Score: ', accuracy_score(d_train, d_predict_train))
print('Test Accuracy Score: ', accuracy_score(d_test, d_pred))
print(confusion_matrix(d_test, d_pred))  
print(classification_report(d_test, d_pred))

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))


# In[272]:


#Classifier 3g) Naive Bayes
start_time = datetime.now()

classifier_nb = GaussianNB()
classifier_nb.fit(C_train, d_train)
d_predict_train = classifier_nb.predict(C_train)
d_pred = classifier_nb.predict(C_test)
print('Train Accuracy Score: ', accuracy_score(d_train, d_predict_train))
print('Test Accuracy Score: ', accuracy_score(d_test, d_pred))
print(confusion_matrix(d_test, d_pred))  
print(classification_report(d_test, d_pred))

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))


# In[273]:


#Classifier 4g) SVM
start_time = datetime.now()

classifier_svm = SVC(kernel='sigmoid') 
classifier_svm.fit(C_train, d_train)
d_predict_train = classifier_svm.predict(C_train)
d_pred = classifier_svm.predict(C_test)
print('Train Accuracy Score: ', accuracy_score(d_train, d_predict_train))
print('Test Accuracy Score: ', accuracy_score(d_test, d_pred))
print(confusion_matrix(d_test, d_pred))  
print(classification_report(d_test, d_pred))

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))


# In[277]:


#Ensemble Method g - Random Forrest Classifier
start_time = datetime.now()

num_trees = 60
max_features = 22
classifier_rf = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
classifier_rf.fit(C_train, d_train)
d_predict_train = classifier_rf.predict(C_train)
d_pred = classifier_rf.predict(C_test)
print('Train Accuracy Score: ', accuracy_score(d_train, d_predict_train))
print('Test Accuracy Score: ', accuracy_score(d_test, d_pred))
print(confusion_matrix(d_test, d_pred))  
print(classification_report(d_test, d_pred))

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))


# In[278]:


#Using Random Forest w/balanced dataset w/replacement to determine most useful features for predicting income
num_trees = 60
max_features = 22
classifier_rf = RandomForestClassifier(criterion='entropy',n_estimators=num_trees, max_features=max_features)
classifier_rf.fit(C, d)


# In[280]:


#To increase readability, plotting the top 10 most important features
feat_importances =pd.Series(model.feature_importances_, index=df_one_hot.columns[1:])
feat_importances.nlargest(10).plot(kind='barh')

