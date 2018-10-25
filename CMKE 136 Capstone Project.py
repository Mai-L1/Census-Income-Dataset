
# coding: utf-8

# In[1]:


#Import pandas
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# In[2]:


#Import data
df_train=pd.read_csv('train.csv')
df_test=pd.read_csv('test.csv')


# In[3]:


#Display training set
display(df_train)


# In[4]:


#Get the datatype of each column
df_train.dtypes


# In[5]:


#Replace <=50K with 0 and >50K with 1
replacements = {
  ' <=50K':0,
  ' >50K':1,
}
df_train['Income'].replace(replacements, inplace=True)


# In[6]:


#Replace ? (missing values) with NaN
df_train2=df_train.replace(" ?",np.NaN)


# In[7]:


#Display new dataframe
display(df_train2)


# In[8]:


#Count the number of records of each attribute
df_train2.count()


# In[9]:


#Find number of missing values in the dataset
df_train2.isnull().sum()


# In[10]:


#Create a dataframe with only the numeric columns
numeric_train = pd.DataFrame(df_train2,columns=['Age','Fnlwgt','Education-Num','Capital-Gain','Capital-Loss','Hours-Per-Week','Income'])


# In[11]:


#Create five number summary of the numeric attributes
numeric_train.describe()


# In[12]:


#UNIVARITE ANALYSIS


# In[13]:


g = sns.pairplot(df_train2)


# In[14]:


#Boxplot of numeric values- Age
plt.boxplot(df_train2['Age'])


# In[15]:


#Boxplot of numeric values- Fnlwgt
plt.boxplot(df_train2['Fnlwgt'])


# In[16]:


#Boxplot of numeric values- Education-Num
plt.boxplot(df_train2['Education-Num'])


# In[17]:


#Boxplot of numeric values- Capital-Gain
plt.boxplot(df_train2['Capital-Gain'])


# In[18]:


#Boxplot of numeric values- Capital-Loss
plt.boxplot(df_train2['Capital-Loss'])


# In[19]:


#Boxplot of numeric values- Hours-Per-Week
plt.boxplot(df_train2['Hours-Per-Week'])


# In[20]:


#Boxplot of numeric values- Income
plt.boxplot(df_train2['Income'])


# In[21]:


#Histogram of Income
df_train2['Income'].value_counts().plot(kind='bar')


# In[22]:


#Histogram of categorical variables - Workclass
df_train2['Workclass'].value_counts().plot(kind='bar')


# In[23]:


#Histogram of categorical variables - Education
df_train2['Education'].value_counts().plot(kind='bar')


# In[24]:


#Histogram of categorical variables - Marital-Status
df_train2['Marital-Status'].value_counts().plot(kind='bar')


# In[25]:


#Histogram of categorical variables - Occupation
df_train2['Occupation'].value_counts().plot(kind='bar')


# In[26]:


#Histogram of categorical variables - Relationship
df_train2['Relationship'].value_counts().plot(kind='bar')


# In[27]:


#Histogram of categorical variables - Race
df_train2['Race'].value_counts().plot(kind='bar')


# In[28]:


#Histogram of categorical variables - Sex
df_train2['Sex'].value_counts().plot(kind='bar')


# In[29]:


#Histogram of categorical variables - Native Country
df_train2['Country'].value_counts().plot(kind='bar')


# In[30]:


#BIVARIATE ANALYSIS


# In[31]:


#Workclass vs Income
df_train2.groupby(['Workclass', 'Income']).size().unstack().plot(kind='bar', stacked=True)


# In[32]:


#Occupation vs Income
df_train2.groupby(['Occupation', 'Income']).size().unstack().plot(kind='bar', stacked=True)


# In[33]:


#Relationship vs Income
df_train2.groupby(['Relationship', 'Income']).size().unstack().plot(kind='bar', stacked=True)


# In[34]:


#Sex vs Income
df_train2.groupby(['Sex', 'Income']).size().unstack().plot(kind='bar', stacked=True)


# In[35]:


#Race vs Income
df_train2.groupby(['Race', 'Income']).size().unstack().plot(kind='bar', stacked=True)


# In[36]:


#Country vs Income
df_train2.groupby(['Country', 'Income']).size().unstack().plot(kind='bar', stacked=True)


# In[37]:


#Marital-Status vs Income
df_train2.groupby(['Marital-Status', 'Income']).size().unstack().plot(kind='bar', stacked=True)


# In[38]:


#MULTIVARIATE ANALYSIS


# In[39]:


#Create correlation matrix for the numeric variables
corr = df_train2.corr()
corr.style.background_gradient().set_precision(2)


# In[40]:


#Identify duplicate rows
df_train2[df_train2.duplicated(keep='first')]


# In[41]:


#Count number of duplicate rows/values
df_train2[df_train2.duplicated(keep='first')].count()


# In[42]:


#Drop duplicated rows
df_train3=df_train2.drop_duplicates(subset=None, keep='first', inplace=False)


# In[43]:


#Check that we dont have anymore duplicated rows
df_train3[df_train3.duplicated(keep='first')].count()


# In[49]:


#Checking how many missing values are left once duplicated values are removed
df_train3.isnull().sum()


# In[44]:


#Drop rows where 'Occupation' is missing
df_train4=df_train3.dropna(axis=0, how='any', thresh=None, subset=['Occupation'], inplace=False)


# In[50]:


#Checking how many missing values are left once rows missing 'Occupation' are dropped
df_train4.isnull().sum()


# In[45]:


#Note, removing rows where 'Occupation' is missing, also removes all rows where 'Workclass' is missing


# In[51]:


#Replace missing 'Country' values with 'United-States'
df_train5=df_train4['Country'].fillna('United-States')


# In[52]:


#Check that there are no missing values left
df_train5.isnull().sum()


# In[ ]:


#Ask Pradeep about feature description and how to address outliers
#Need to remove duplicate rows
#Need to remove rows where occupation is missing
#Need to replace missing workclass with private
#Need to replace missing country with US

