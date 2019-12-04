import pandas as pd
import numpy as np

np.random.seed(1212)

import keras
from keras.models import Model
from keras.layers import *
from keras import optimizers

train_url = r'C:\Users\helle\Desktop\train.csv'
#train_url = r'C:\Users\helle\Desktop\train_0.csv'

df_train = pd.read_csv(train_url)

df_features = df_train.iloc[:, 1:785]
df_label = df_train.iloc[:, 0]

train_x = df_train.iloc[:,1:].values.astype('float32')
train_y = df_train.iloc[:,0].values.astype('int32')

train_x = train_x.reshape(train_x.shape[:1] + (28, 28, 1))
train_y = keras.utils.to_categorical(train_y)
num_classes = train_y.shape[1]

train_x = train_x / 255

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#load the libs
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
print(tf.__version__)

#my_callbacks = [tf.keras.callbacks.EarlyStopping(patience=3, min_delta=0, monitor='val_loss')]


#plt.figure()
#plt.imshow(train_x[0].reshape(28, 28))
#plt.colorbar()
#plt.grid(False)
#plt.show()

#class_names = [0,1,2,3,4,5,6]

#plot a group of features and labels to check data
#plt.figure(figsize=(10,10))
#for i in range(25):
#    plt.subplot(5,5,i+1)
#    plt.xticks([])
#    plt.yticks([])
#    plt.grid(False)
#    plt.imshow(train_x[i].reshape(28, 28), cmap=plt.cm.binary)
#    plt.xlabel(class_names[np.argmax(train_y[i])])
#plt.show()

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#load the libs
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os

my_callbacks = [tf.keras.callbacks.EarlyStopping(patience=10, min_delta=0, monitor='val_loss')]

#define the model and layers
model = keras.Sequential([ 
    #start: 1@28x28 image matrices

    #convolution with 32 filters that use a 3x3 kernel (convolution window) and stride of 1
    keras.layers.Conv2D(32, (3,3), input_shape=(28, 28, 1), strides=(1,1), activation='relu'),
    #now: 32@26x26

    #subsampling using max pooling and a 2x2 filter (largest element from the rectified feature map)
    keras.layers.MaxPool2D(pool_size=(2,2)),
    #now: 32@13x13 matricies

    #convolution with 64 filters that use a 3x3 kernel (convolution window) and stride of 1
    keras.layers.Conv2D(64, (3,3), strides=(1,1), activation='relu'),
    #now: 64@11x11

    #subsampling using max pooling
    keras.layers.MaxPool2D(pool_size=(2,2)),
    #now: 64@5x5

    #flatten to a single vector
    keras.layers.Flatten(),
    #now: flattened to 1600

    #first fully connected layer with 128 units
    keras.layers.Dense(128, activation=tf.nn.relu),

    #drop 20% of units to help prevent overfitting
    #keras.layers.Dropout(0.5),

    #softmax layer for classification
    keras.layers.Dense(num_classes, activation=tf.nn.softmax)
])

#compile the model
model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.001), 
#sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.5, nesterov=True)

#model.compile(optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1), 
#model.compile(optimizer=sgd, 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#print a summary of the model
model.summary()

#model achitecture
#from IPython.display import SVG
#from keras.utils.vis_utils import model_to_dot
#SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))

#train the model
hist = model.fit(x=train_x, 
            y=train_y,
            #batch_size=32,
            batch_size = 256,
            #epochs=30,
            epochs=100,
            verbose=1,
            #callbacks = my_callbacks,
            validation_split=0.15,
            shuffle=True
            )

#evaluate
print("정확도 : %.4f" % (model.evaluate(train_x, train_y)[1]))

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

mpl.rcParams['axes.unicode_minus'] = False

#path = 'C:\\Users\\helle\\AppData\\Local\\Microsoft\\Windows\\Fonts\\Maplestory Light.ttf'
#font_name = fm.FontProperties(fname=path, size=50).get_name()
#print(font_name)
#plt.rc('font', family=font_name)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.plot(hist.history['loss'])
plt.title("Cost function during training")
plt.ylabel("Cost function Value")
plt.subplot(1, 2, 2)
plt.title("Performance during training")
plt.ylabel("performance value")
plt.plot(hist.history['acc'], 'b-', label="Training Performance")
plt.plot(hist.history['val_acc'], 'r:', label="Verification Performance")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title("Performance during training")
plt.ylabel("performance value")
plt.plot(hist.history['loss'], 'b-', label="Training loss")
plt.plot(hist.history['val_loss'], 'r:', label="Verification loss")
plt.legend()
plt.tight_layout()
plt.show()

