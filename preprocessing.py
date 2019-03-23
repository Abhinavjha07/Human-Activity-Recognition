import cv2 as cv
import numpy as np
import os
from tqdm import tqdm

for file in tqdm(os.listdir('./Keck Dataset/trainingfiles/')):
    count =0
    dir_name = file
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    if not os.path.exists('train'):
        os.makedirs('train')
    dir_name = file+'/'
    cap = cv.VideoCapture('./Keck Dataset/trainingfiles/'+file)
    print(cap.get(cv.CAP_PROP_FPS))
    cap.set(cv.CAP_PROP_POS_MSEC,0.5*1000)
    fgbg = cv.bgsegm.createBackgroundSubtractorMOG()
    static_black = None
    while(cap.isOpened()):
        frameId = cap.get(1)
        ret,frame = cap.read()

        if(ret!=True):
            break
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) 
        frame= cv.resize(frame, (64,64), interpolation = cv.INTER_LINEAR)

  
        mask = fgbg.apply(frame)
        f_name = "frame%d.jpg" %count
        count+=1
        cv.imwrite(dir_name+f_name,mask)
        

    cap.release()
    print('Done!')






    

    
