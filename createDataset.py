import os
import numpy as np
import cv2
from random import shuffle
from tqdm import tqdm

IMG_SIZE = 64
TRAIN_DIR = 'train/'
TEST_DIR = 'test/'

def label_img(img):
    cl = img.split('.')[1]
    x = np.zeros((1,14),dtype = int)
    x[0,int(cl)-1] = 1
    return x

def create_train_data():
    train_X = []
    train_Y = []
    for d in tqdm(os.listdir(TRAIN_DIR)):
        X = []
        label = label_img(d)
        path = os.path.join(TRAIN_DIR,d)
       
        for img in tqdm(os.listdir(path)):
           
            
            img = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
            X.append(np.array(img))
        
        train_X.append(np.array(X))
        train_Y.append(np.array(label))
        
    
    #shuffle(train_data)

    
    
    train_X = np.array(train_X)
    train_Y =np.array(train_Y)
    
    #train_X = np.reshape(train_X,(42,1))
    print(train_X[0].shape)
    np.save('train_X.npy',train_X)
    np.save('train_Y.npy',train_Y)


def create_test_data():
    test_X = []
    test_Y = []
    for d in tqdm(os.listdir(TEST_DIR)):
        X = []
        label = label_img(d)
        path = os.path.join(TEST_DIR,d)
        for img in tqdm(os.listdir(path)):
            img = cv2.resize(cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE),(IMG_SIZE,IMG_SIZE))
            X.append(np.array(img))
        
        test_X.append(np.array(X))
        test_Y.append(np.array(label)) 
    test_X = np.array(test_X)
    test_Y =np.array(test_Y) 
            
    np.save('test_X.npy',test_X)
    np.save('test_Y.npy',test_Y)



create_train_data()

create_test_data()


            


    
        
        
        
