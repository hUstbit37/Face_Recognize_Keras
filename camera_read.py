import bz2
import os
import cv2
import tensorflow as tf
from align import AlignDlib
from urllib.request import urlopen
from model import create_model
from keras.models import model_from_json
import numpy as np
cap = cv2.VideoCapture("rtsp://192.168.7.137:10085")
loaded_model = create_model()
loaded_model.load_weights("./weights/model.h5")
while(True):
    # Capture frame-by-frame 
    alignment = AlignDlib('models/landmarks.dat')   
    ret, frame = cap.read()    
    #img = frame
    # Our operations on the frame come here    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    bb = alignment.getLargestFaceBoundingBox(frame)
    jc_aligned = alignment.align(96, frame, bb, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
    #Display the resulting frame    
    jc_aligned = np.expand_dims(jc_aligned, axis=0)
    print(jc_aligned.shape)
    a = np.argmax(loaded_model.predict(jc_aligned))+1
    cv2.rectangle(frame,(bb.left(),bb.top()),(bb.right(),bb.right()+(bb.top()-bb.left())),(0,0,255),10)
    cv2.imshow('anh thuoc nhom:'+ str(a),frame) 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

