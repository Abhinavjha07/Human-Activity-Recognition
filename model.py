import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,Conv3D, MaxPooling3D
import numpy as np
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.utils import np_utils

train_X = np.load('train_X.npy')
train_Y = np.load('train_Y.npy')
test_X = np.load('test_X.npy')
test_Y = np.load('test_Y.npy')

num_classes = 14

X = []

for i in range(train_X.shape[0]):
    z = train_X[i].shape[0]
    
    if z < 498:
        temp = np.zeros(shape = (498,64,64))
        temp[0:z,:,:] = train_X[i][0]
        train_X[i] = temp
    X.append(train_X[i])

X = np.array(X)
train_X = X


X = []
for i in range(test_X.shape[0]):
    z = test_X[i].shape[0]
    if z < 498:
        temp = np.zeros(shape = (498,64,64))
        temp[0:z,:,:] = test_X[i]
        test_X[i] = temp
    X.append(test_X[i])

test_X = np.array(X)
train_Y = np.reshape(train_Y,(-1,14))
test_Y = np.reshape(test_Y,(-1,14))
train_X = train_X[:,50:400,:,:]
test_X = test_X[:,50:400,:,:]
print(test_Y.shape,train_Y.shape)
train_X = np.reshape(train_X,(-1,350,64*64))
test_X = np.reshape(test_X,(-1,350,64*64))
input_shape=(350,64*64)
train_X = np.concatenate((train_X,test_X),axis = 0)
train_Y = np.concatenate((train_Y,test_Y),axis = 0)

model = Sequential()
#model.add(Conv3D(32,kernel_size = (3,3,3),activation='relu',input_shape = shape))
#model.add(MaxPooling3D(pool_size = (3,3,3)))

#model.add(Conv3D(64,kernel_size = (1,1,1),activation='relu'))



#model.add(Flatten())
#model.add(Dense(256,activation='relu'))
#model.add(Dense(128,activation='relu'))
model.add(LSTM(128,return_sequences=True,dropout=0.2,recurrent_dropout = 0.2,input_shape = input_shape))
model.add(LSTM(64))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(128,activation='relu'))


model.add(Dense(num_classes,activation='softmax'))
model.compile(loss = 'categorical_crossentropy',optimizer = 'adam',metrics = ['accuracy'])

BATCH_SIZE = 10
EPOCHS = 50
model.fit(train_X,train_Y,batch_size = BATCH_SIZE,epochs = EPOCHS , shuffle=True ,validation_data=(test_X, test_Y),verbose = 1)

model.save('my_model1')



