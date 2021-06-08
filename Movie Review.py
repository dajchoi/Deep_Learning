#!/usr/bin/env python
# coding: utf-8

# In[3]:


from keras.datasets import imdb


# In[6]:


(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)  #frequently used num_words words -> adequate data size


# In[7]:


train_data[0]


# In[9]:


train_labels[0]


# In[13]:


max([max(sequence) for sequence in train_data]) #check index


# In[14]:


word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_review = ' '.join([reverse_word_index.get(i-3, '?') for i in train_data[0]])


# In[16]:


decoded_review #all keys with 0, 1, 2 is padding, start of doc, not in dict, so is removed by 3 and turned into ?


# In[19]:


import numpy as np
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension)) #change to an integer(0) tensor
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)


# In[20]:


x_train[0]


# In[21]:


y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')


# In[34]:


train_data


# In[35]:


from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


# In[36]:


model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])


# In[37]:


from keras import optimizers

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
             loss='binary_crossentropy',
             metrics=['accuracy'])


# In[38]:


"""
from keras import losses
from keras import metrics

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
             loss=losses.binary_crossentropy,
             metrics=[metrics.binary_accuracy])
"""


# In[39]:


x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]


# In[40]:


history = model.fit(partial_x_train,
                   partial_y_train,
                   epochs=20,
                   batch_size=512,
                   validation_data=(x_val, y_val))


# In[44]:


import matplotlib.pyplot as plt

history_dict = history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title("Training and validation loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# In[49]:


plt.clf()
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')


# In[50]:


history_dict.keys()


# In[52]:


model.fit(x_train, y_train, epochs=4, batch_size=512)


# In[53]:


results = model.evaluate(x_test, y_test)


# In[ ]:




