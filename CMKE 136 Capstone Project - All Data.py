
# coding: utf-8

# # CMKE 136 Capstone Project Fall 2018

# ## Import Packages

# In[28]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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


# ## Initial Analysis

# In[29]:


#Import data and create dataframe
df_all=pd.read_csv('all.csv')


# In[30]:


#Display dataframe
df_all.head(n=10)


# In[31]:


#Get information about each column including data types
df_all.info()


# In[32]:


#Replace the class attribute with numerice values: <=50K with 0 and >50K with 1
replacements = {
  ' <=50K':0,
  ' >50K':1,
  ' <=50K.':0,
  ' >50K.':1,  
}
df_all['Income'].replace(replacements, inplace=True)


# In[33]:


#Replace ? (missing values) with NaN
df_all2=df_all.replace(" ?",np.NaN)


# In[34]:


#Display the new dataframe
df_all2.head(n=10)


# In[35]:


#Count the number of records for each attribute
df_all2.count()


# ### Univariate Analysis

# In[36]:


#Create a dataframe with only the numeric variables
numeric_all = pd.DataFrame(df_all2,columns=['Age','Fnlwgt','Education-Num','Capital-Gain','Capital-Loss','Hours-Per-Week','Income'])


# In[37]:


#Create a five number summary of the numeric attributes
numeric_all.describe()


# In[38]:


#Boxplot of numeric values- Age
plt.boxplot(df_all2['Age'])
print('Kurtosis',df_all2['Age'].kurtosis())


# In[39]:


#Boxplot of numeric values- Fnlwgt
plt.boxplot(df_all2['Fnlwgt'])
print('Kurtosis',df_all2['Fnlwgt'].kurtosis())


# In[40]:


#Boxplot of numeric values- Education-Num
plt.boxplot(df_all2['Education-Num'])
print('Kurtosis',df_all2['Education-Num'].kurtosis())


# In[41]:


#Boxplot of numeric values- Capital-Gain
plt.boxplot(df_all2['Capital-Gain'])
print('Kurtosis',df_all2['Capital-Gain'].kurtosis())


# In[45]:


#Boxplot of numeric values- Capital-Loss
plt.boxplot(df_all2['Capital-Loss'])
print('Kurtosis',df_all2['Capital-Loss'].kurtosis())


# In[43]:


#Boxplot of numeric values- Hours-Per-Week
plt.boxplot(df_all2['Hours-Per-Week'])
print('Kurtosis',df_all2['Hours-Per-Week'].kurtosis())


# In[44]:


#Boxplot of class attribute- Income
plt.boxplot(df_all2['Income'])
print('Kurtosis',df_all2['Income'].kurtosis())


# In[21]:


#Histogram of class attribute Income
df_all2['Income'].value_counts().plot(kind='bar')


# In[22]:


#As seen from the Income histogram, the data is unbalanced in favour of income <50K


# In[23]:


#Histogram of categorical variables - Workclass
df_all2['Workclass'].value_counts().plot(kind='bar')


# In[24]:


#Histogram of categorical variables - Education
df_all2['Education'].value_counts().plot(kind='bar')


# In[25]:


#Histogram of categorical variables - Marital-Status
df_all2['Marital-Status'].value_counts().plot(kind='bar')


# In[26]:


#Histogram of categorical variables - Occupation
df_all2['Occupation'].value_counts().plot(kind='bar')


# In[27]:


#Histogram of categorical variables - Relationship
df_all2['Relationship'].value_counts().plot(kind='bar')


# In[28]:


#Histogram of categorical variables - Race
df_all2['Race'].value_counts().plot(kind='bar')


# In[29]:


#Histogram of categorical variables - Sex
df_all2['Sex'].value_counts().plot(kind='bar')


# In[30]:


#Histogram of categorical variables - Native Country
df_all2['Country'].value_counts().plot(kind='bar')


# ### Bivariate Analysis

# In[31]:


#Graph the distribution of each of the variables and plot each of the variables against the class attribute
g = sns.pairplot(df_all2)


# In[32]:


#Plotting each of the independent variables against the class attribute: Workclass vs Income
df_all2.groupby(['Workclass', 'Income']).size().unstack().plot(kind='bar', stacked=True)


# In[33]:


#Occupation vs Income
df_all2.groupby(['Occupation', 'Income']).size().unstack().plot(kind='bar', stacked=True)


# In[34]:


#Relationship vs Income
df_all2.groupby(['Relationship', 'Income']).size().unstack().plot(kind='bar', stacked=True)


# In[35]:


#Sex vs Income
df_all2.groupby(['Sex', 'Income']).size().unstack().plot(kind='bar', stacked=True)


# In[36]:


#Race vs Income
df_all2.groupby(['Race', 'Income']).size().unstack().plot(kind='bar', stacked=True)


# In[37]:


#Country vs Income
df_all2.groupby(['Country', 'Income']).size().unstack().plot(kind='bar', stacked=True)


# In[38]:


#Marital-Status vs Income
df_all2.groupby(['Marital-Status', 'Income']).size().unstack().plot(kind='bar', stacked=True)


# ### Multivariate Analysis

# In[39]:


#Create correlation matrix of the numeric variables
corr = df_all2.corr(method='pearson')
corr.style.background_gradient().set_precision(2)


# In[46]:


#There is a weak correlation between Income & Age, Income & Education-Num, Income & Capital-Gain and Income & Hours-Per-Week


# ### Further Preprocessing

# In[41]:


#Count the number of missing values for each attribute
df_all2.isnull().sum()


# In[42]:


#Identify duplicate rows
df_all2[df_all2.duplicated(keep='first')]


# In[43]:


#Count number of duplicate rows/values
df_all2[df_all2.duplicated(keep='first')].count()


# In[44]:


#There are 52 duplicate rows in the dataset


# In[45]:


#Drop duplicated rows
df_all3=df_all2.drop_duplicates(keep='first', inplace=False)


# In[46]:


#Check that all duplicate rows have been removed
df_all3[df_all3.duplicated(keep='first')].count()


# In[47]:


#Checking how many missing values are left once duplicated values are removed
df_all3.isnull().sum()


# In[48]:


#Drop rows where 'Occupation' is missing
df_all4=df_all3.dropna(axis=0, how='any', thresh=None, subset=['Occupation'], inplace=False)


# In[49]:


#Checking how many missing values are left once rows missing 'Occupation' are dropped
df_all4.isnull().sum()


# In[50]:


#Removing rows where 'Occupation' is missing, removes all rows where 'Workclass' is missing


# In[51]:


#Replace missing 'Country' values with the mode 'United-States
df_all5=df_all4.fillna('United States')


# In[52]:


#Check that all missing values have been removed
df_all5.isnull().sum()


# ## Exploratory Analysis

# In[53]:


#Drop the Education column as Education-Num is the exact numeric version of this attribute
df_all6=df_all5.drop('Education',axis=1)


# In[56]:


#Applying one-hot encoding the get dummy variables for each of the categorical variables
df_copy=df_all6.copy()

df_one_hot=pd.get_dummies(df_copy, columns=["Workclass","Marital-Status","Occupation","Relationship","Race","Sex","Country"], prefix=["Workclass","Marital-Status","Occupation","Relationship","Race","Sex","Country"])


# In[57]:


#Examening the new dataframe and the total number of columns
df_one_hot.info()


# ### Feature Selection

# In[58]:


#With the help of the decision tree classifier using the Gini index to rank each features based on importance
array = df_one_hot.values
X=array[:,1:]
Y=array[:,0]
model=DecisionTreeClassifier()
model.fit(X,Y)


# In[59]:


#Plotting all the features and their importance
pd.Series(model.feature_importances_, index=df_one_hot.columns[1:]).plot.bar(color='steelblue', figsize=(12, 6))


# In[60]:


#Listing all the features and their importance in descending order
pd.Series(model.feature_importances_, index=df_one_hot.columns[1:]).sort_values(ascending=False).head(n=20)


# In[61]:


#Listing all the features and their importance in ascending order
pd.Series(model.feature_importances_, index=df_one_hot.columns[1:]).sort_values(ascending=True).head(n=20)


# In[62]:


#To increase readability, plotting the top 20 most important features
feat_importances =pd.Series(model.feature_importances_, index=df_one_hot.columns[1:])
feat_importances.nlargest(20).plot(kind='barh')


# In[63]:


#Making a copy of the dataset for the purposes of trying PCA
df_copy2=df_one_hot.copy()


# In[64]:


#Applying PCA for the purposes of feature selection
A = df_copy2.drop('Income', 1)  
b = df_copy2['Income']  
A_train, A_test, b_train, b_test = train_test_split(A, b, test_size=0.2, random_state=0) 
sc = StandardScaler()  
A_train = sc.fit_transform(A_train)  
A_test = sc.transform(A_test)
pca = PCA()
A_train = pca.fit_transform(A_train)  
A_test = pca.transform(A_test)
pca.explained_variance_ratio_


# In[65]:


#Examine how many components are needed in order to explain variance
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');


# In[69]:


#From the PCA graph, it appears as if 60 of the components explain 80% of the variance, 80 of the components explain 100% of the variance.


# In[70]:


#Create a new dataframe where the top 15 features with importance ~ 0.000000 are removed
df_clean=df_one_hot.drop(['Country_ Honduras','Country_ Hong','Country_ Scotland','Country_ Holand-Netherlands','Country_ Ecuador','Country_ Guatemala','Country_ El-Salvador','Country_ Haiti','Occupation_ Armed-Forces','Country_ Thailand','Country_ Outlying-US(Guam-USVI-etc)','Country_ Columbia','Country_ Peru','Country_ Nicaragua','Country_ Laos'],axis=1)


# In[71]:


#Checking that the desired features were dropped from the dataframe
df_clean.info()


# ## Data Modelling

# In[72]:


#Split the data into two; one dataframe for the class attribute and one dataframe for the rest of the attributes
dataset=df_clean.values
X=dataset[:,1:]
y=dataset[:,0]


# ### Classification Approach A: Splitting between train and test using an 80:20 split

# In[73]:


#Split to dataset into train and test set to prepare for modelling
#80% of the data will be used for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


# In[97]:


#Classifier 1a) Decision Tree
classifier_dt = DecisionTreeClassifier()  

#Train algorithm on training data
classifier_dt.fit(X_train, y_train)

#Make predictions on test data
y_pred = classifier_dt.predict(X_test)

print('Test Accuracy Score: ', accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))                                 


# In[92]:


#Classifier 2a) KNN
classifier_knn = KNeighborsClassifier(n_neighbors=25)
classifier_knn.fit(X_train, y_train)
y_pred = classifier_knn.predict(X_test)
print('Test Accuracy Score: ', accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))


# In[93]:


#Classifier 3a) Naive Bayes
classifier_nb = GaussianNB()
classifier_nb.fit(X_train, y_train)
y_pred = classifier_nb.predict(X_test)
print('Test Accuracy Score: ', accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))


# In[95]:


#Classifier 4a) SVM
classifier_svm = SVC(kernel='sigmoid') 
classifier_svm.fit(X_train, y_train)
y_pred = classifier_svm.predict(X_test)
print('Test Accuracy Score: ', accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))


# In[94]:


#Ensemble Method a - Random Forrest Classifier
num_trees = 100
max_features = 74
classifier_rf = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
classifier_rf.fit(X_train, y_train)
y_pred = classifier_rf.predict(X_test)
print('Test Accuracy Score: ', accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))


# ### Classification Approach B: Splitting between train and test using 10 fold cross-validation

# In[80]:


#Split the data using 10 fold cross-validation
seed=7
kfold= model_selection.KFold(n_splits=10, random_state=seed)


# In[81]:


#Classifier 1b) Decision Tree
model_dt=tree.DecisionTreeClassifier()
results_dt = model_selection.cross_val_score(model_dt, X, y, cv=kfold)
print(results_dt.mean())


# In[82]:


#Classifier 2b) KNN
model_knn = KNeighborsClassifier(n_neighbors=25)
results_knn = model_selection.cross_val_score(model_knn, X, y, cv=kfold)
print(results_knn.mean())


# In[121]:


#Classifier 3b) Naive Bayes
model_nb = GaussianNB()
results_nb= model_selection.cross_val_score(model_nb, X, y, cv=kfold)
print(results_nb.mean())


# In[84]:


#Classifier 4b) SVM
model_svm=SVC(kernel='sigmoid')
results_svm = model_selection.cross_val_score(model_svm, X, y, cv=kfold)
print(results_svm.mean())


# In[83]:


#Ensemble Method b - Random Forrest Classifier
model_rf = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
results_rf = model_selection.cross_val_score(model_rf, X, y, cv=kfold)
print(results_rf.mean())

