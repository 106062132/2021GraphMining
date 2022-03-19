#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd 
df_shopee = pd.read_csv('clothes_shopee/shopee長袖女.csv',index_col=0)  


# In[17]:


num_sold_array = df_shopee['num_sold']
num_sold_array2 = [0] * len(num_sold_array)
index = 0
for item in num_sold_array:
    num_sold_array2[index] = str.replace(item,",","")
    index += 1


# In[18]:


x = list(map(int,num_sold_array2))


# In[19]:


cnt_array = [0]*max(x)
for i in range(len(x)):
    cnt_array[x[i]-1] += 1


# In[20]:


x_array = [0] * max(x)
for i in range(max(x)):
    x_array[i] = i+1


# In[21]:


import matplotlib.pyplot as plt
import numpy as np
plt.plot(x_array[0:80],cnt_array[0:80])
plt.show()


# In[22]:


plt.plot(np.log(x_array[0:80]),np.log(np.array(cnt_array[0:80])))
plt.show()


# In[23]:


Gamma = -(np.log(cnt_array[0]) - np.log(cnt_array[50] )) / (np.log(x_array[0]) - np.log(x_array[50]))


# In[24]:


print(Gamma)


# In[25]:


A = np.log(cnt_array[0]) + Gamma * np.log(x_array[0])


# In[26]:


print(A)


# In[28]:





# In[ ]:




