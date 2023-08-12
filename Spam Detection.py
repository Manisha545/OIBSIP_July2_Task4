#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libraries
import numpy as np
import pandas as pd
import re


# In[2]:


# sms spam data loading and encoding dataset
df= pd.read_csv(r"C:\Users\manis\Downloads\spam\spam.csv", encoding= 'ISO-8859-1', encoding_errors = 'strict')


# In[3]:


df.head()


# In[4]:


# cleaning this dataset

df=df.drop(df.columns[[2,3,4]], axis=1)


# In[5]:


df.head()


# In[6]:


# Renaming multiple columns
df.rename(columns={'v1': 'Label', 'v2': 'Text'}, inplace=True)


# In[7]:


df.head()


# In[8]:


df.Text[0]


# In[9]:


# Acutally this data are contained huge ammount of space and dot. thats why we can use regex for given only words and numbers.
# Apply the regular expression to the "Text" column and join the results with a space
df['Text'] = df['Text'].apply(lambda x: ' '.join(re.findall(r'[A-Za-z0-9]+', str(x))))


# In[10]:


df['Text'][0]


# In[11]:


# Label data are convert into a decimal number just label encoding using by lambda
df.Label=df.Label.apply(lambda x: 1 if x=="spam" else 0)


# In[12]:


df.head()


# In[13]:


from sklearn.model_selection import train_test_split


# In[14]:


# data are divied into training and testing dataset
X_train, X_test, y_train, y_test=train_test_split(df.Text, df.Label, test_size=0.2)


# In[15]:


X_train.shape


# In[16]:


# import counvectorization for text transfrom 
from sklearn.feature_extraction.text import CountVectorizer


# In[17]:


cv=CountVectorizer()


# In[18]:


X_train_cv= cv.fit_transform(X_train.values)


# In[19]:


X_train


# In[20]:


X_train_cv.toarray()


# In[21]:


X_train_cv.shape # there are 7676 unique words are locate in our BOW


# In[22]:


cv.get_feature_names_out()[2000:2100]


# In[23]:


#cv.vocabulary_
X_train_np=X_train_cv.toarray()
X_train_np[0]


# In[24]:


np.where(X_train_np[0]!=0)


# In[25]:


X_train[:5]


# In[26]:


np.where(X_train_np[2961]!=0)


# In[27]:


X_train_np[2961][1644]


# In[28]:


#model /classify this model


# In[29]:


# impor Naive Bayes algorithm for prediction 
from sklearn.naive_bayes import MultinomialNB


# In[30]:


NB=MultinomialNB()


# In[31]:


NB.fit(X_train_cv, y_train)


# In[32]:


X_test_cv= cv.transform(X_test)


# In[33]:


y_pred=NB.predict(X_test_cv)


# In[34]:


from sklearn.metrics import accuracy_score, classification_report


# In[35]:


print(classification_report(y_test, y_pred))


# In[36]:


print(accuracy_score(y_test, y_pred))


# In[ ]:





# In[ ]:




