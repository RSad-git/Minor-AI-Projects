#!/usr/bin/env python
# coding: utf-8

# ####SVM

# In[1]:


import os, codecs
import numpy as np
import sklearn
from sklearn.metrics import confusion_matrix
import pandas as pd


# In[2]:


datapath='C://Users//dfury//OneDrive//Documents//assignments//Spring22//AI//Project 3//Handwritten Digits Dataset//train-labels.idx1-ubyte'
path='C://Users//dfury//OneDrive//Documents//assignments//Spring22//AI//Project 3//Handwritten Digits Dataset//train-images.idx3-ubyte'
Acc=[]
Accl=[]
train={}
test={}
def con(b):
    return int(codecs.encode(b, 'hex'),16)

with open(path,'rb') as f:
    data= f.read()
    type= con(data[:4])
    print(type)
    l=con(data[4:8])
    r=con(data[8:12])
    c=con(data[12:16])
    print(l,r,c)
m=np.frombuffer(data,dtype=np.uint8,offset=16)
m=m.reshape(l,r*c)


train['images']=m
    

with open(datapath,'rb') as f:
    datal= f.read()
    type= con(datal[:4])
    print(type)
    lb=con(data[4:8])
    print(lb)
m=np.frombuffer(datal,dtype=np.uint8,offset=8)
m=m.reshape(lb)
train['labels']=m


# In[ ]:





# In[3]:


print(train['images'].shape)


# In[4]:


##training set
x=train['images'][1000:1999]
y=train['labels'][1000:1999]


# In[5]:


#test set
tx=train['images'][3000:3099]
ty=train['labels'][3000:3099]


# In[6]:


from sklearn import svm
clf_svm=svm.SVC(kernel='linear')
clf_svm.fit(x,y)


# In[7]:


#storing predicted values
p=clf_svm.predict(tx)


# In[8]:


#Accuracy
Acc.append(clf_svm.score(tx,ty))
Accl.append('SVM 1000 training,100 testing')


# In[9]:


print(f'{Accl[0]} Accuracy:{Acc[0]}')


# Part2

# In[10]:


x2=train['images'][20000:29999]
y2=train['labels'][20000:29999]
tx2=train['images'][30000:30099]
ty2=train['labels'][30000:30099]


# In[11]:


clf_svm2=svm.SVC(kernel='linear')
clf_svm2.fit(x2,y2)


# In[12]:


p2=clf_svm2.predict(tx2)
Acc.append(clf_svm2.score(tx2,ty2))
Accl.append('SVM 10000 training,100 testing')


# In[13]:


print(f'{Accl[1]} Accuracy:{Acc[1]}')


# Part3

# In[14]:


clf_svm3=svm.SVC(kernel='linear')
clf_svm3.fit(x2,y2)


# In[15]:


tx3=train['images'][30000:30999]
ty3=train['labels'][30000:30999]


# In[16]:


p3=clf_svm3.predict(tx3)
Acc.append(clf_svm3.score(tx3,ty3))
Accl.append('SVM 10000 training,1000 testing')
print(f'{Accl[2]} Accuracy:{Acc[2]}')


# In[ ]:





# Part4 Confusion Matrix

# In[17]:


cm=confusion_matrix(ty,p)
df_cm=pd.DataFrame(cm)
print("Confusion Matrix for Part 1\n")
print(df_cm)

cm=confusion_matrix(ty2,p2)
df_cm=pd.DataFrame(cm)
print("\n Confusion Matrix for Part 2\n")
print(df_cm)

cm=confusion_matrix(ty3,p3)
df_cm=pd.DataFrame(cm)
print("\n Confusion Matrix for Part 3\n")
print(df_cm)


# Part 5

# In[18]:


clf_svm3rbf=svm.SVC(kernel='rbf')
clf_svm3rbf.fit(x2,y2)
pr=clf_svm3rbf.predict(tx3)
Acc.append(clf_svm3rbf.score(tx3,ty3))
Accl.append('SVM using rbf kernel')
print(f'{Accl[3]} Accuracy:{Acc[3]}')


# In[19]:


clf_svm3p=svm.SVC(kernel='poly')
clf_svm3p.fit(x2,y2)
pp=clf_svm3p.predict(tx3)
Acc.append(clf_svm3p.score(tx3,ty3))
Accl.append('SVM using polynomial kernel')
print(f'{Accl[4]} Accuracy:{Acc[4]}')


# In[20]:


clf_svm3s=svm.SVC(kernel='sigmoid')
clf_svm3s.fit(x2,y2)
ps=clf_svm3s.predict(tx3)
Acc.append(clf_svm3s.score(tx3,ty3))
Accl.append('SVM using sigmoid kernel')
print(f'{Accl[5]} Accuracy:{Acc[5]}')


# Part 6

# In[21]:


th=5
tax=(x2>th)*255
print(tax[1])


# In[22]:


tax3=(tx3>th)*255


# In[23]:


clf_svmb=svm.SVC(kernel='linear')
clf_svmb.fit(tax,y2)
p3b=clf_svmb.predict(tax3)
Acc.append(clf_svmb.score(tax3,ty3))
Accl.append('SVM on thresholded binary image')
print(f'{Accl[-1]} Accuracy:{Acc[-1]}')


# In[24]:


dfc=pd.DataFrame(
    {
        "Method": Accl,
        "Accuracy":Acc
},index=['Part1','Part2','Part3','Part5_rbf','Part5_poly','Part5_sigmoid','Part6'])
dfc


# In[ ]:




