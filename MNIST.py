#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import linear, relu, sigmoid
get_ipython().run_line_magic('matplotlib', 'widget')
import matplotlib.pyplot as plt


# In[2]:





# In[5]:


pip install python-mnist


# In[31]:


from mnist import MNIST

data = MNIST(r'E:\AB\MACHINE LEARNING THINGS\MultiClass\files')

X, y = data.load_training()


# In[32]:


import random
a = random.randrange(0, len(X))
print(data.display(X[a]))


# In[16]:


print (X[0])


# In[33]:


import numpy as np
X = np.array(X)
y = np.array(y)


# In[37]:


print ('The shape of X is: ' + str(X.shape))
print ('The shape of y is: ' + str(y.shape))


# In[36]:


import random
a = random.randrange(0, len(X))
print(data.display(X[a]))


# In[ ]:





# In[80]:


model = Sequential(
    [               
        tf.keras.Input(shape=(784,)),
        Dense(units=25,input_shape=[784], activation='relu',name="L1"),
        Dense(units=15, activation='relu',name="L2"),
        Dense(units=10, activation='linear',name="L3"),
    ] 
)


# In[81]:


model.summary()


# In[82]:


model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0015),
)

history = model.fit(X,y,epochs=40)


# In[110]:


imgt = X[90]
pred = model.predict(np.array([imgt]))
preds = tf.nn.softmax(pred)
print(preds)


# In[111]:


yhat = np.argmax(pred)
print(f"np.argmax(prediction_p): {yhat}")
print("Actual: " + str(y[90]))


# In[112]:


print(data.display(X[90]))


# In[ ]:




